"""
Hierarchical Dual-Partition Poisson Gradient Boosting Framework
Engineered for npm Security Advisory (GHSA) 90-Day Vulnerability Forecasting.

Key Architectural Components:
1. Dual-Partition Learning Architecture:
   - Partition A (Core Framework & High-Impact Ecosystem Distribution, 156 packages)
   - Partition B (Broad-Spectrum Longitudinal npm Package Distribution, 484 packages)
   - Both sub-ensembles trained with Poisson loss, shallow depth (max_depth=2),
     and leaf sample constraints (min_samples_leaf=5) to regularize expected arrival rates.
2. Robust Semantic Version Resolver:
   - Full SemVer 2.0 parser extracting the latest stable non-prerelease release.
   - Automatic fallback to chronological timestamp resolution for anonymized packages.
3. Schema Normalizer:
   - Automatic BOM stripping, column lowercasing, alias mapping, and temporal window alignment.
4. Asymmetric Risk Guardrails:
   - Explicit soft capping to bound predictions within empirical limits and prevent
     squared-error amplification under Relative Squared Error (RSE) loss.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd
from packaging.version import Version, InvalidVersion

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "pcom_champion_ensemble.joblib"

TARGET_COLS = [
    "target_ghsa_low_count_90d",
    "target_ghsa_moderate_count_90d",
    "target_ghsa_high_count_90d",
    "target_ghsa_critical_count_90d",
    "target_ghsa_unknown_count_90d",
]

AS_OF_DATE = pd.Timestamp("2026-07-01", tz="UTC")
PURL_ALIASES = ["package_url", "package_purl", "pkg_purl", "purl_id", "package", "id"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans column names so differences in BOM, casing, or prefixes do not break feature alignment."""
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip().lower() for c in df.columns]

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
    """Extracts package name from npm PURL (e.g. pkg:npm/%40scope/pkg@1.0.0 -> @scope/pkg)."""
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
    """Parses SemVer; returns (name, Version) or (name, None) if prerelease / unparseable."""
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


def newest_row_without_purl(df: pd.DataFrame, cutoff: pd.Timestamp = AS_OF_DATE) -> pd.DataFrame:
    """When purl is absent, select newest version row by published_at or fallback to last row."""
    if df.empty:
        return pd.DataFrame([{}])
    idx = df.index[-1]
    if "published_at" in df.columns:
        dates = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
        if dates.notna().any():
            idx = dates.idxmax()
    row = df.loc[idx].to_dict()
    row["package_name"] = "unknown"
    return add_date_features(pd.DataFrame([row]), cutoff=cutoff)


def prepare_single_package_row(df: pd.DataFrame, cutoff_date: str = "2026-07-01") -> pd.DataFrame:
    """
    Reduces multi-row input CSV (multiple versions for one package)
    to a single feature row using the highest stable semantic version.
    """
    df = normalize_columns(df)
    cutoff = pd.Timestamp(cutoff_date, tz="UTC")

    if "purl" not in df.columns or not df["purl"].notna().any():
        return newest_row_without_purl(df, cutoff=cutoff)

    parsed = df["purl"].apply(stable_version_from_purl)
    df["_pkg"] = parsed.apply(lambda x: x[0])
    df["_ver"] = parsed.apply(lambda x: x[1])

    valid = df[df["_ver"].notna()].copy()
    if valid.empty:
        # If no valid stable SemVer, use timestamp fallback
        return newest_row_without_purl(df, cutoff=cutoff)

    latest_idx = max(valid.index, key=lambda idx: valid.loc[idx, "_ver"])
    row = valid.loc[latest_idx].drop(labels=["_pkg", "_ver"]).to_dict()
    row["package_name"] = valid.loc[latest_idx, "_pkg"]
    return add_date_features(pd.DataFrame([row]), cutoff=cutoff)


class ChampionForecaster:
    """
    Production Forecasting Engine for PCOM-01 Competition.
    Loads and serves the Dual-Partition Regularized Poisson Ensemble.
    """
    def __init__(self, model_path: Path | str = MODEL_PATH):
        self.model_path = Path(model_path)
        self.bundle: Optional[Dict[str, Any]] = None
        self.load()

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model bundle not found at {self.model_path}")
        self.bundle = joblib.load(self.model_path)
        print(f"[ChampionForecaster] Model pipeline active from {self.model_path} ({len(self.bundle['feature_columns'])} features)")

    def predict(self, df: pd.DataFrame, cutoff_date: str = "2026-07-01") -> Tuple[pd.DataFrame, float]:
        t0 = time.perf_counter()
        if self.bundle is None:
            self.load()

        pkg_row = prepare_single_package_row(df, cutoff_date=cutoff_date)
        feature_cols = self.bundle["feature_columns"]
        X = pkg_row.reindex(columns=feature_cols).apply(pd.to_numeric, errors="coerce")

        output = {}
        for target in TARGET_COLS:
            # Sub-ensemble A: Core Framework Partition
            key_a = "models_anchor" if "models_anchor" in self.bundle else "models_d1"
            s1 = self.bundle[key_a][target]
            v1 = s1["value"] if s1["type"] == "constant" else float(s1["estimator"].predict(X)[0])

            # Sub-ensemble B: Broad Ecosystem Partition
            key_b = "models_distribution" if "models_distribution" in self.bundle else "models_d2"
            s2 = self.bundle[key_b][target]
            v2 = s2["value"] if s2["type"] == "constant" else float(s2["estimator"].predict(X)[0])

            # Weighted hierarchical ensemble (supports target-specific or global weights)
            weights = self.bundle.get("blend_weights", {})
            if target in weights and isinstance(weights[target], dict):
                tw = weights[target]
                w1 = tw.get("anchor", tw.get("d1", 0.5))
                w2 = tw.get("distribution", tw.get("d2", 0.5))
            else:
                w1 = weights.get("anchor", weights.get("d1", 0.5))
                w2 = weights.get("distribution", weights.get("d2", 0.5))
            v_blend = w1 * v1 + w2 * v2

            # Soft cap protection against squared error blowouts
            cap = self.bundle.get("soft_caps", {}).get(target, 5.0)
            val = float(np.clip(v_blend, 0.0, cap))
            if np.isnan(val) or np.isinf(val):
                val = 0.0
            output[target] = round(val, 6)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return pd.DataFrame([output], columns=TARGET_COLS), elapsed_ms
