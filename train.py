"""
Production Training Pipeline for PCOM-01 Vulnerability Forecasting.

Trains the Class-Specialized Dual-Partition Poisson Gradient Boosting Framework:
- Partition A (Core Framework & Ecosystem Anchor): 156 packages
- Partition B (Longitudinal npm Distribution): 484 packages
- Severity-Tuned Hyperparameters:
    * Low:      max_depth=2, min_samples_leaf=6, l2_reg=2.5, Poisson loss
    * Moderate: max_depth=3, min_samples_leaf=5, l2_reg=1.5, Poisson loss
    * High:     max_depth=2, min_samples_leaf=6, l2_reg=2.0, Poisson loss
    * Critical: max_depth=1 (decision stump), min_samples_leaf=15, l2_reg=10.0
    * Unknown:  Strictly constant 0.000000
- Per-Target Blending:
    * Low:      80% Partition A / 20% Partition B
    * Moderate: 20% Partition A / 80% Partition B
    * High:     50% Partition A / 50% Partition B
    * Critical: 50% Partition A / 50% Partition B
    * Unknown:  Constant 0.0
"""
from __future__ import annotations

import os
import re
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from packaging.version import Version, InvalidVersion
from sklearn.ensemble import HistGradientBoostingRegressor

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

TARGETS = [
    "target_ghsa_low_count_90d",
    "target_ghsa_moderate_count_90d",
    "target_ghsa_high_count_90d",
    "target_ghsa_critical_count_90d",
    "target_ghsa_unknown_count_90d",
]

AS_OF_DATE = pd.Timestamp("2026-07-01", tz="UTC")
PURL_ALIASES = ["package_url", "package_purl", "pkg_purl", "purl_id", "package", "id"]

TARGET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "target_ghsa_low_count_90d": {
        "d1": dict(loss="poisson", max_iter=160, learning_rate=0.04, max_depth=2, min_samples_leaf=6, l2_regularization=2.5, random_state=42),
        "d2": dict(loss="poisson", max_iter=160, learning_rate=0.04, max_depth=2, min_samples_leaf=8, l2_regularization=3.0, random_state=42),
        "blend": {"d1": 0.80, "d2": 0.20},
        "soft_cap": 3.0,
    },
    "target_ghsa_moderate_count_90d": {
        "d1": dict(loss="poisson", max_iter=180, learning_rate=0.04, max_depth=3, min_samples_leaf=5, l2_regularization=1.5, random_state=42),
        "d2": dict(loss="poisson", max_iter=180, learning_rate=0.04, max_depth=3, min_samples_leaf=4, l2_regularization=1.0, random_state=42),
        "blend": {"d1": 0.20, "d2": 0.80},
        "soft_cap": 4.5,
    },
    "target_ghsa_high_count_90d": {
        "d1": dict(loss="poisson", max_iter=150, learning_rate=0.04, max_depth=2, min_samples_leaf=6, l2_regularization=2.0, random_state=42),
        "d2": dict(loss="poisson", max_iter=150, learning_rate=0.04, max_depth=2, min_samples_leaf=6, l2_regularization=2.0, random_state=42),
        "blend": {"d1": 0.50, "d2": 0.50},
        "soft_cap": 3.5,
    },
    "target_ghsa_critical_count_90d": {
        "d1": dict(loss="poisson", max_iter=100, learning_rate=0.02, max_depth=1, min_samples_leaf=15, l2_regularization=10.0, random_state=42),
        "d2": dict(loss="poisson", max_iter=100, learning_rate=0.02, max_depth=1, min_samples_leaf=15, l2_regularization=10.0, random_state=42),
        "blend": {"d1": 0.50, "d2": 0.50},
        "soft_cap": 1.5,
    },
    "target_ghsa_unknown_count_90d": {
        "type": "constant",
        "value": 0.0,
        "blend": {"d1": 0.50, "d2": 0.50},
        "soft_cap": 0.0,
    },
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes column names, strips BOM, and maps standard aliases."""
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").replace("﻿", "").strip().lower() for c in df.columns]
    renames = {}
    for c in df.columns:
        new = c
        new = re.sub(r"^(issues_(?:opened|closed|reopened))_count_m(\d+)$", r"\1_m\2", new)
        new = re.sub(r"^package_downloads_days_observed_", "package_download_days_observed_", new)
        if new == "created_at":
            new = "package_created_at"
        new = re.sub(r"_m(\d+)$", lambda m: "_M" + m.group(1), new)
        if new != c:
            renames[c] = new
    df = df.rename(columns=renames)
    if "purl" not in df.columns:
        for alt in PURL_ALIASES:
            if alt in df.columns and df[alt].astype(str).str.startswith("pkg:").any():
                df = df.rename(columns={alt: "purl"})
                break
    return df


def package_name_from_purl(purl: str) -> str:
    """Extracts package name from PURL string."""
    if not isinstance(purl, str):
        return "unknown"
    purl = purl.strip().replace("%40", "@")
    if not purl.startswith("pkg:npm/"):
        return "unknown"
    val = purl[len("pkg:npm/"):]
    if "@" not in val.lstrip("@"):
        return val
    return val.rsplit("@", 1)[0]


def stable_version_from_purl(purl: str) -> Tuple[str, Optional[Version]]:
    """Parses SemVer; returns (name, Version) or (name, None) if prerelease / invalid."""
    try:
        name = package_name_from_purl(purl)
        raw_v = purl.strip().rsplit("@", 1)[1]
        v = Version(raw_v)
        if v.is_prerelease or v.is_devrelease:
            return name, None
        return name, v
    except Exception:
        return package_name_from_purl(purl), None


def add_date_features(df: pd.DataFrame, cutoff: pd.Timestamp = AS_OF_DATE) -> pd.DataFrame:
    """Computes package age and version age in days at the cutoff date."""
    df = df.copy()
    for col, out in [
        ("package_created_at", "package_age_days_at_cutoff"),
        ("published_at", "version_age_days_at_cutoff"),
    ]:
        if col in df.columns:
            dt = pd.to_datetime(df[col], errors="coerce", utc=True)
            df[out] = (cutoff - dt).dt.total_seconds() / 86400.0
        else:
            df[out] = 0.0
    return df


def make_package_dataset(csv_path: str | Path) -> Tuple[pd.DataFrame, list[str]]:
    """Reduces multi-version rows into single package rows using the latest stable release."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = normalize_columns(df)
    
    parsed = df["purl"].apply(stable_version_from_purl)
    df["_package"] = parsed.apply(lambda x: x[0])
    df["_version"] = parsed.apply(lambda x: x[1])
    valid = df[df["_version"].notna()].copy()
    if valid.empty:
        raise ValueError("No stable SemVer rows found in dataset.")
        
    rows = []
    for pkg, grp in valid.groupby("_package", sort=True):
        latest_idx = max(grp.index, key=lambda idx: grp.loc[idx, "_version"])
        row = grp.loc[latest_idx].drop(labels=["_package", "_version"]).to_dict()
        row["package_name"] = pkg
        rows.append(row)
        
    result = pd.DataFrame(rows).reset_index(drop=True)
    result = add_date_features(result)
    return result, TARGETS


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Identifies viable numeric features with non-zero variation."""
    excluded = set(TARGETS) | {"purl", "package_name", "package_created_at", "published_at", "snapshot_cutoff"}
    candidates = [c for c in df.columns if c not in excluded]
    numeric = df[candidates].apply(pd.to_numeric, errors="coerce")
    return [c for c in numeric.columns if numeric[c].notna().any() and numeric[c].nunique(dropna=True) > 1]


def load_partitions(anchor_csv: Path, dist_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Loads and formats the dual dataset partitions."""
    print(f"Loading Partition 1 (Anchor): {anchor_csv}")
    data1, _ = make_package_dataset(anchor_csv)
    
    print(f"Loading Partition 2 (Distribution): {dist_csv}")
    df2_raw = pd.read_csv(dist_csv)
    if "snapshot_cutoff" in df2_raw.columns:
        df2 = df2_raw[df2_raw["snapshot_cutoff"] == "2026-07-01"].copy()
    else:
        df2 = df2_raw.copy()
    if "purl" not in df2.columns and "package_name" in df2.columns and "version" in df2.columns:
        df2["purl"] = "pkg:npm/" + df2["package_name"].astype(str) + "@" + df2["version"].astype(str)
        
    tmp_path = Path("/tmp/pcom_partition2_proc.csv")
    df2.to_csv(tmp_path, index=False)
    data2, _ = make_package_dataset(tmp_path)
    
    feature_cols = get_feature_columns(data1)
    return data1, data2, feature_cols


def train_production_bundle(data1: pd.DataFrame, data2: pd.DataFrame, feature_cols: list[str]) -> Dict[str, Any]:
    """Fits target-specialized estimators across both partitions and returns bundle."""
    X1 = data1[feature_cols].apply(pd.to_numeric, errors="coerce")
    y1 = data1[TARGETS].apply(pd.to_numeric, errors="coerce").fillna(0)
    
    X2 = data2[feature_cols].apply(pd.to_numeric, errors="coerce")
    y2 = data2[TARGETS].apply(pd.to_numeric, errors="coerce").fillna(0)
    
    models_d1 = {}
    models_d2 = {}
    soft_caps = {}
    blend_weights = {}
    
    for t in TARGETS:
        cfg = TARGET_CONFIGS[t]
        soft_caps[t] = cfg["soft_cap"]
        blend_weights[t] = cfg["blend"]
        
        if cfg.get("type") == "constant":
            models_d1[t] = {"type": "constant", "value": cfg["value"]}
            models_d2[t] = {"type": "constant", "value": cfg["value"]}
        else:
            m1 = HistGradientBoostingRegressor(**cfg["d1"])
            m1.fit(X1, y1[t])
            models_d1[t] = {"type": "model", "estimator": m1}
            
            m2 = HistGradientBoostingRegressor(**cfg["d2"])
            m2.fit(X2, y2[t])
            models_d2[t] = {"type": "model", "estimator": m2}
            
    return {
        "models_d1": models_d1,
        "models_d2": models_d2,
        "blend_weights": blend_weights,
        "feature_columns": feature_cols,
        "target_columns": TARGETS,
        "soft_caps": soft_caps,
    }


def main():
    parser = argparse.ArgumentParser(description="Train PCOM-01 Champion Ensemble")
    parser.add_argument("--anchor", type=Path, default=Path("/Users/predator/Desktop/PCOM-main/dataset_initial.csv"), help="Partition 1 CSV path")
    parser.add_argument("--distribution", type=Path, default=BASE_DIR / "training_dataset.csv", help="Partition 2 CSV path")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "models" / "pcom_champion_ensemble.joblib", help="Output model bundle path")
    args = parser.parse_args()
    
    t0 = time.time()
    data1, data2, feature_cols = load_partitions(args.anchor, args.distribution)
    print(f"Features: {len(feature_cols)} | P1: {len(data1)} pkgs | P2: {len(data2)} pkgs")
    
    print("\nFitting Class-Specialized Dual-Partition Estimators...")
    bundle = train_production_bundle(data1, data2, feature_cols)
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output)
    print(f"\n[SUCCESS] Exported production ensemble bundle to {args.output} ({args.output.stat().st_size / 1024:.1f} KB)")
    print(f"Total training pipeline completed in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
