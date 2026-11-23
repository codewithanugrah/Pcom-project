import re
import numpy as np
import pandas as pd
from packaging.version import Version, InvalidVersion

SEVERITIES = ["low", "moderate", "high", "critical", "unknown"]
TARGET_COLS = [f"target_ghsa_{s}_count_90d" for s in SEVERITIES]
DEFAULT_TEST_CUTOFF = pd.Timestamp("2026-07-01T00:00:00Z")

_DECAY_WEIGHTS = np.exp(-0.25 * np.arange(12))  # M0=1.0, M1=0.78, ..., M11=0.064
_FAST_DECAY = np.exp(-0.60 * np.arange(12))  # Active burst (M0..M2)
_SLOW_DECAY = np.exp(-0.10 * np.arange(12))  # Chronic risk (M0..M11)
_TIME_X = np.arange(12, dtype=float)  # 0 (M11) -> 11 (M0)

PURL_ALIASES = ["package_url", "package_purl", "pkg_purl", "purl_id", "package", "id"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans column names so small naming differences, BOM marks, or case variations
    do not break feature engineering.
    """
    df = df.copy()
    # Strip BOM and whitespace, convert to lowercase
    df.columns = [str(c).replace("\ufeff", "").replace("﻿", "").strip().lower() for c in df.columns]

    renames = {}
    for c in df.columns:
        new = c
        # Fix legacy issue count naming: issues_opened_count_m0 -> issues_opened_M0
        new = re.sub(r"^(issues_(?:opened|closed|reopened))_count_m(\d+)$", r"\1_m\2", new)
        # Fix plural vs singular typo: package_downloads_days_observed -> package_download_days_observed
        new = re.sub(r"^package_downloads_days_observed_", "package_download_days_observed_", new)
        if new == "created_at":
            new = "package_created_at"
        # The schema convention writes the monthly windows as uppercase _M0 ... _M11
        new = re.sub(r"_m(\d+)$", lambda m: "_M" + m.group(1), new)
        if new != c:
            renames[c] = new
    df = df.rename(columns=renames)

    # Check for PURL under aliases if "purl" not found
    if "purl" not in df.columns:
        for alt in PURL_ALIASES:
            if alt in df.columns and df[alt].astype(str).str.startswith("pkg:").any():
                df = df.rename(columns={alt: "purl"})
                break
    return df


def parse_stable_semver(purl_or_version: str):
    """
    Parses a version string or purl into (is_stable, parsed_version).
    Extracts version from purl if needed (e.g. pkg:npm/@scope/name@1.2.3 -> 1.2.3).
    Returns (is_stable: bool, version_obj: Version | None).
    """
    if not isinstance(purl_or_version, str):
        return False, None
    raw = purl_or_version.strip()
    if raw.startswith("pkg:"):
        raw = raw.replace("%40", "@")
        if "@" in raw.lstrip("@"):
            raw = raw.rsplit("@", 1)[1]
    raw = raw.lstrip("v").strip()
    try:
        v = Version(raw)
        is_stable = not (v.is_prerelease or v.is_devrelease)
        return is_stable, v
    except (InvalidVersion, ValueError, TypeError):
        return False, None


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    if pd.isna(num) or pd.isna(den) or den == 0:
        return default
    return float(num) / float(den)


def _linear_slope(values_m11_to_m0: np.ndarray) -> float:
    """Linear slope over M11..M0 (chronological order)."""
    mask = ~np.isnan(values_m11_to_m0)
    if mask.sum() < 2:
        return 0.0
    x = _TIME_X[mask]
    y = values_m11_to_m0[mask]
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def _get_m_array(row: pd.Series, prefix: str) -> np.ndarray:
    """Returns float array [M0, M1, ..., M11]."""
    return np.array([row.get(f"{prefix}_M{i}", np.nan) for i in range(12)], dtype=float)


def transform_package_df(
    pkg_df: pd.DataFrame, cutoff_date: pd.Timestamp | str | None = None
) -> dict:
    """
    Collapses an N-row package snapshot DataFrame (matching Input.csv schema)
    into a single engineered feature dictionary.
    Works identically on training snapshots and unseen test Input.csv files.
    """
    df = normalize_columns(pkg_df)

    if cutoff_date is None:
        if "snapshot_cutoff" in df.columns and df["snapshot_cutoff"].notna().any():
            cutoff = pd.Timestamp(str(df["snapshot_cutoff"].dropna().iloc[0]))
        else:
            cutoff = DEFAULT_TEST_CUTOFF
    else:
        cutoff = pd.Timestamp(cutoff_date)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")

    # Safe datetime parsing
    if "published_at" in df.columns:
        df["_pub_dt"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    else:
        df["_pub_dt"] = pd.NaT

    # Semantic version ranking if purl or version column is available
    ver_col = "purl" if "purl" in df.columns else ("version" if "version" in df.columns else None)
    if ver_col:
        parsed_info = df[ver_col].apply(parse_stable_semver)
        df["_is_stable"] = parsed_info.apply(lambda x: x[0])
        df["_semver"] = parsed_info.apply(lambda x: x[1])

        stable_df = df[df["_is_stable"] & df["_semver"].notna()]
        if len(stable_df) > 0:
            df_for_anchor = stable_df
        else:
            df_for_anchor = df[df["_semver"].notna()] if df["_semver"].notna().any() else df

        if len(df_for_anchor) > 0 and df_for_anchor["_semver"].notna().any():
            latest_idx = df_for_anchor["_semver"].idxmax()
        else:
            latest_idx = df["_pub_dt"].idxmax() if df["_pub_dt"].notna().any() else df.index[-1]
    else:
        latest_idx = df["_pub_dt"].idxmax() if df["_pub_dt"].notna().any() else df.index[-1]

    first_idx = df["_pub_dt"].idxmin() if df["_pub_dt"].notna().any() else df.index[0]
    first_row = df.loc[first_idx]
    last_row = df.loc[latest_idx]
    n_versions = len(df)

    feats = {}

    # =========================================================================
    # BLOCK 1: VERSION TRAJECTORY & CODEBASE BLOAT
    # =========================================================================
    if "package_created_at" in df.columns and df["package_created_at"].notna().any():
        created_dt = pd.to_datetime(df["package_created_at"], utc=True, errors="coerce").dropna().min()
    else:
        created_dt = pd.NaT

    pkg_age_days = max(0.0, (cutoff - created_dt).total_seconds() / 86400.0) if pd.notna(created_dt) else np.nan
    days_since_last = max(0.0, (cutoff - last_row["_pub_dt"]).total_seconds() / 86400.0) if pd.notna(last_row["_pub_dt"]) else np.nan
    earliest_pub = df["_pub_dt"].dropna().min() if df["_pub_dt"].notna().any() else pd.NaT
    days_since_first = max(0.0, (cutoff - earliest_pub).total_seconds() / 86400.0) if pd.notna(earliest_pub) else np.nan
    release_span = max(0.0, days_since_first - days_since_last) if pd.notna(days_since_first) and pd.notna(days_since_last) else 0.0

    feats["num_versions_12m"] = float(n_versions)
    feats["log_num_versions_12m"] = float(np.log1p(n_versions))
    feats["package_age_days"] = pkg_age_days
    feats["log_package_age_days"] = float(np.log1p(pkg_age_days)) if pd.notna(pkg_age_days) else np.nan
    feats["days_since_last_release"] = days_since_last
    feats["release_span_days"] = release_span

    # Inter-release cadence
    if n_versions >= 2:
        gaps = df["_pub_dt"].diff().dt.total_seconds().dropna().values / 86400.0
        feats["release_gap_mean"] = float(np.mean(gaps))
        feats["release_gap_std"] = float(np.std(gaps))
        feats["release_gap_min"] = float(np.min(gaps))
        feats["release_gap_max"] = float(np.max(gaps))
    else:
        feats["release_gap_mean"] = 365.0
        feats["release_gap_std"] = 0.0
        feats["release_gap_min"] = 365.0
        feats["release_gap_max"] = 365.0

    rel_m = _get_m_array(last_row, "package_stable_release_count")
    rel_12m = np.nansum(rel_m)
    rel_3m = np.nansum(rel_m[:3])
    feats["releases_3m"] = float(rel_3m)
    feats["release_burst_ratio"] = _safe_div(rel_3m, rel_12m + 1.0)

    # Latest version metadata
    sizes = pd.to_numeric(df["unpacked_size"], errors="coerce").dropna()
    files = pd.to_numeric(df["file_count"], errors="coerce").dropna()

    latest_size = float(sizes.iloc[-1]) if len(sizes) > 0 else np.nan
    first_size = float(sizes.iloc[0]) if len(sizes) > 0 else np.nan
    latest_files = float(files.iloc[-1]) if len(files) > 0 else np.nan
    first_files = float(files.iloc[0]) if len(files) > 0 else np.nan

    feats["latest_unpacked_size"] = latest_size
    feats["log_latest_unpacked_size"] = float(np.log1p(max(0.0, latest_size))) if pd.notna(latest_size) else np.nan
    feats["latest_file_count"] = latest_files
    feats["log_latest_file_count"] = float(np.log1p(max(0.0, latest_files))) if pd.notna(latest_files) else np.nan
    feats["latest_code_density"] = _safe_div(latest_size, latest_files + 1.0, default=np.nan) if pd.notna(latest_size) and pd.notna(latest_files) else np.nan

    # Size & file evolution across the 12m window
    if len(sizes) >= 2:
        size_diffs = np.diff(sizes.values)
        feats["size_delta"] = latest_size - first_size
        feats["size_pct_change"] = _safe_div(latest_size - first_size, first_size + 1.0)
        feats["size_std"] = float(np.std(sizes.values))
        feats["size_max_jump"] = float(np.max(size_diffs))
    else:
        feats["size_delta"] = 0.0
        feats["size_pct_change"] = 0.0
        feats["size_std"] = 0.0
        feats["size_max_jump"] = 0.0

    if len(files) >= 2:
        feats["file_count_delta"] = latest_files - first_files
        feats["file_count_pct_change"] = _safe_div(latest_files - first_files, first_files + 1.0)
    else:
        feats["file_count_delta"] = 0.0
        feats["file_count_pct_change"] = 0.0

    # Dependencies
    dep_cols = [
        "dependency_count", "optional_dependency_count",
        "peer_dependency_count", "dev_dependency_count", "bundled_dependency_count"
    ]
    total_deps = 0.0
    for c in dep_cols:
        val = float(last_row.get(c, 0) or 0)
        feats[f"latest_{c}"] = val
        total_deps += val
    feats["latest_total_dep_count"] = total_deps
    feats["dep_count_delta"] = float(last_row.get("dependency_count", 0) or 0) - float(first_row.get("dependency_count", 0) or 0)
    feats["dev_dep_count_delta"] = float(last_row.get("dev_dependency_count", 0) or 0) - float(first_row.get("dev_dependency_count", 0) or 0)

    # =========================================================================
    # BLOCK 2: HISTORICAL VULNERABILITY GRAVITY (STRONGEST SIGNAL)
    # =========================================================================
    total_life = 0.0
    total_12m_vec = np.zeros(12, dtype=float)
    sev_weights = {"low": 1.0, "moderate": 2.0, "high": 4.0, "critical": 8.0, "unknown": 1.0}
    weighted_life = 0.0
    weighted_12m = 0.0

    for s in SEVERITIES:
        life = float(last_row.get(f"ghsa_{s}_count", 0) or 0)
        m_arr = np.nan_to_num(_get_m_array(last_row, f"ghsa_{s}_count"), nan=0.0)

        feats[f"ghsa_{s}_lifetime"] = life
        feats[f"ghsa_{s}_12m"] = float(np.sum(m_arr))
        feats[f"ghsa_{s}_6m"] = float(np.sum(m_arr[:6]))
        feats[f"ghsa_{s}_3m"] = float(np.sum(m_arr[:3]))
        feats[f"ghsa_{s}_m0"] = float(m_arr[0])
        feats[f"ghsa_{s}_exp_decay"] = float(np.dot(m_arr, _DECAY_WEIGHTS))

        total_life += life
        total_12m_vec += m_arr
        weighted_life += life * sev_weights[s]
        weighted_12m += float(np.sum(m_arr)) * sev_weights[s]

    total_12m = float(np.sum(total_12m_vec))
    total_6m = float(np.sum(total_12m_vec[:6]))
    total_3m = float(np.sum(total_12m_vec[:3]))

    feats["ghsa_total_lifetime"] = total_life
    feats["ghsa_total_12m"] = total_12m
    feats["ghsa_total_6m"] = total_6m
    feats["ghsa_total_3m"] = total_3m
    feats["ghsa_total_m0"] = float(total_12m_vec[0])
    feats["ghsa_total_exp_decay"] = float(np.dot(total_12m_vec, _DECAY_WEIGHTS))
    feats["ghsa_weighted_severity_lifetime"] = weighted_life
    feats["ghsa_weighted_severity_12m"] = weighted_12m

    feats["ghsa_12m_to_lifetime_ratio"] = _safe_div(total_12m, total_life + 1.0)
    feats["ghsa_3m_to_12m_ratio"] = _safe_div(total_3m, total_12m + 1.0)
    feats["active_ghsa_months_12m"] = float(np.sum(total_12m_vec > 0))

    nonzero_months = np.where(total_12m_vec > 0)[0]
    feats["months_since_last_ghsa"] = float(nonzero_months[0]) if len(nonzero_months) > 0 else 12.0

    age_years = max(pkg_age_days / 365.25, 0.25) if pd.notna(pkg_age_days) else 1.0
    feats["ghsa_lifetime_rate_per_year"] = total_life / age_years

    for s in SEVERITIES:
        feats[f"ghsa_{s}_share_lifetime"] = _safe_div(feats[f"ghsa_{s}_lifetime"], total_life + 1.0)

    # =========================================================================
    # BLOCK 3: NORMALIZED DOWNLOAD MOMENTUM
    # =========================================================================
    dl_arr = np.nan_to_num(_get_m_array(last_row, "package_downloads"), nan=0.0)
    days_arr = np.nan_to_num(_get_m_array(last_row, "package_download_days_observed"), nan=0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        daily_dl = np.where(days_arr > 0, dl_arr / days_arr, 0.0)

    dl_total_12m = float(np.sum(dl_arr))
    dl_daily_12m_mean = float(np.mean(daily_dl))
    dl_daily_3m_mean = float(np.mean(daily_dl[:3]))
    dl_daily_old3m_mean = float(np.mean(daily_dl[9:12]))

    feats["log_dl_total_12m"] = float(np.log1p(max(0.0, dl_total_12m)))
    feats["log_dl_daily_m0"] = float(np.log1p(max(0.0, daily_dl[0])))
    feats["log_dl_daily_3m_mean"] = float(np.log1p(max(0.0, dl_daily_3m_mean)))
    feats["log_dl_daily_12m_mean"] = float(np.log1p(max(0.0, dl_daily_12m_mean)))
    feats["dl_momentum_3m_vs_12m"] = _safe_div(dl_daily_3m_mean, dl_daily_12m_mean + 1.0)
    feats["dl_momentum_recent_vs_old"] = _safe_div(dl_daily_3m_mean, dl_daily_old3m_mean + 1.0)
    feats["dl_trend_slope"] = _linear_slope(np.log1p(np.maximum(0.0, daily_dl[::-1])))
    feats["dl_cv"] = _safe_div(float(np.std(daily_dl)), dl_daily_12m_mean + 1.0)

    # =========================================================================
    # BLOCK 4: MAINTENANCE & REPO HEALTH (GITHUB M0..M11)
    # =========================================================================
    iss_open = _get_m_array(last_row, "issues_opened")
    has_gh = 0.0 if np.all(np.isnan(iss_open)) else 1.0
    feats["has_github_data"] = has_gh

    if has_gh == 1.0:
        iss_open = np.nan_to_num(iss_open, nan=0.0)
        iss_closed = np.nan_to_num(_get_m_array(last_row, "issues_closed"), nan=0.0)
        iss_reopen = np.nan_to_num(_get_m_array(last_row, "issues_reopened"), nan=0.0)
        iss_end = np.nan_to_num(_get_m_array(last_row, "issues_open_at_month_end"), nan=0.0)

        pr_open = np.nan_to_num(_get_m_array(last_row, "pull_requests_opened"), nan=0.0)
        pr_merged = np.nan_to_num(_get_m_array(last_row, "pull_requests_merged"), nan=0.0)
        pr_unmerged = np.nan_to_num(_get_m_array(last_row, "pull_requests_closed_unmerged"), nan=0.0)
        pr_reopen = np.nan_to_num(_get_m_array(last_row, "pull_requests_reopened"), nan=0.0)
        pr_end = np.nan_to_num(_get_m_array(last_row, "pull_requests_open_at_month_end"), nan=0.0)

        iss_open_12m, iss_open_3m = float(np.sum(iss_open)), float(np.sum(iss_open[:3]))
        iss_closed_12m, iss_closed_3m = float(np.sum(iss_closed)), float(np.sum(iss_closed[:3]))
        iss_reopen_12m = float(np.sum(iss_reopen))

        pr_open_12m, pr_open_3m = float(np.sum(pr_open)), float(np.sum(pr_open[:3]))
        pr_merged_12m, pr_merged_3m = float(np.sum(pr_merged)), float(np.sum(pr_merged[:3]))
        pr_unmerged_12m = float(np.sum(pr_unmerged))
        pr_reopen_12m = float(np.sum(pr_reopen))

        feats["issues_opened_12m"] = iss_open_12m
        feats["issues_opened_3m"] = iss_open_3m
        feats["issues_closed_12m"] = iss_closed_12m
        feats["issues_reopened_12m"] = iss_reopen_12m
        feats["issue_close_ratio_12m"] = _safe_div(iss_closed_12m, iss_open_12m + 1.0)
        feats["issue_close_ratio_3m"] = _safe_div(iss_closed_3m, iss_open_3m + 1.0)
        feats["issue_reopen_ratio_12m"] = _safe_div(iss_reopen_12m, iss_closed_12m + 1.0)
        feats["issues_surge_3m_vs_9m"] = _safe_div(iss_open_3m / 3.0, (float(np.sum(iss_open[3:])) / 9.0) + 1.0)

        feats["prs_opened_12m"] = pr_open_12m
        feats["prs_opened_3m"] = pr_open_3m
        feats["prs_merged_12m"] = pr_merged_12m
        feats["prs_closed_unmerged_12m"] = pr_unmerged_12m
        feats["prs_reopened_12m"] = pr_reopen_12m
        feats["pr_merge_ratio_12m"] = _safe_div(pr_merged_12m, pr_open_12m + 1.0)
        feats["pr_merge_ratio_3m"] = _safe_div(pr_merged_3m, pr_open_3m + 1.0)
        feats["pr_reject_ratio_12m"] = _safe_div(pr_unmerged_12m, pr_open_12m + 1.0)
        feats["prs_surge_3m_vs_9m"] = _safe_div(pr_open_3m / 3.0, (float(np.sum(pr_open[3:])) / 9.0) + 1.0)

        feats["issues_open_m0"] = float(iss_end[0])
        feats["issues_backlog_delta"] = float(iss_end[0] - iss_end[11])
        feats["issues_backlog_ratio"] = _safe_div(float(iss_end[0]), iss_open_12m + 1.0)
        feats["prs_open_m0"] = float(pr_end[0])
        feats["prs_backlog_delta"] = float(pr_end[0] - pr_end[11])
        feats["prs_backlog_ratio"] = _safe_div(float(pr_end[0]), pr_open_12m + 1.0)
    else:
        gh_keys = [
            "issues_opened_12m", "issues_opened_3m", "issues_closed_12m", "issues_reopened_12m",
            "issue_close_ratio_12m", "issue_close_ratio_3m", "issue_reopen_ratio_12m", "issues_surge_3m_vs_9m",
            "prs_opened_12m", "prs_opened_3m", "prs_merged_12m", "prs_closed_unmerged_12m", "prs_reopened_12m",
            "pr_merge_ratio_12m", "pr_merge_ratio_3m", "pr_reject_ratio_12m", "prs_surge_3m_vs_9m",
            "issues_open_m0", "issues_backlog_delta", "issues_backlog_ratio",
            "prs_open_m0", "prs_backlog_delta", "prs_backlog_ratio",
        ]
        for k in gh_keys:
            feats[k] = np.nan

    # =========================================================================
    # BLOCK 5: CROSS-DOMAIN INTERACTIONS
    # =========================================================================
    feats["ghsa_12m_x_log_dl"] = total_12m * feats["log_dl_total_12m"]
    feats["ghsa_exp_decay_x_log_dl"] = feats["ghsa_total_exp_decay"] * feats["log_dl_total_12m"]
    feats["prs_per_release_12m"] = _safe_div(feats["prs_opened_12m"], n_versions + 1.0, default=np.nan) if has_gh == 1.0 else np.nan
    feats["issues_per_log_dl"] = _safe_div(feats["issues_opened_12m"], feats["log_dl_total_12m"] + 1.0, default=np.nan) if has_gh == 1.0 else np.nan
    feats["size_x_deps"] = (
        feats["log_latest_unpacked_size"] * float(np.log1p(total_deps))
        if pd.notna(feats["log_latest_unpacked_size"]) else np.nan
    )

    # =========================================================================
    # BLOCK 6: TRIAGE BOTTLENECKS, MULTI-SCALE DECAY & VOLATILITY
    # =========================================================================
    # Multi-scale vulnerability dynamics
    feats["ghsa_total_fast_decay"] = float(np.dot(total_12m_vec, _FAST_DECAY))
    feats["ghsa_total_slow_decay"] = float(np.dot(total_12m_vec, _SLOW_DECAY))
    feats["ghsa_acceleration_3m"] = float(np.sum(total_12m_vec[:3]) - np.sum(total_12m_vec[3:6]))
    feats["ghsa_per_release_12m"] = _safe_div(total_12m, n_versions + 1.0)

    # Attack surface & hotfix volatility
    feats["runtime_to_dev_dep_ratio"] = _safe_div(
        feats["latest_dependency_count"], feats["latest_dev_dependency_count"] + 1.0
    )
    feats["release_gap_cv"] = _safe_div(feats["release_gap_std"], feats["release_gap_mean"] + 1.0)
    feats["size_max_jump_ratio"] = (
        _safe_div(feats["size_max_jump"], latest_size + 1.0) if pd.notna(latest_size) else np.nan
    )

    # Maintenance & triage bottlenecks
    if has_gh == 1.0:
        feats["pr_to_issue_ratio_12m"] = _safe_div(feats["prs_opened_12m"], feats["issues_opened_12m"] + 1.0)
        feats["merged_pr_to_issue_ratio_12m"] = _safe_div(feats["prs_merged_12m"], feats["issues_opened_12m"] + 1.0)
        feats["issue_clearance_velocity"] = _safe_div(feats["issues_closed_12m"], feats["issues_open_m0"] + 1.0)
        feats["pr_clearance_velocity"] = _safe_div(feats["prs_merged_12m"], feats["prs_open_m0"] + 1.0)
        feats["pr_merge_to_reject_ratio"] = _safe_div(feats["prs_merged_12m"], feats["prs_closed_unmerged_12m"] + 1.0)
        feats["exposure_to_maintenance"] = _safe_div(
            feats["log_dl_total_12m"], float(np.log1p(feats["prs_merged_12m"])) + 1.0
        )
    else:
        for k in [
            "pr_to_issue_ratio_12m", "merged_pr_to_issue_ratio_12m",
            "issue_clearance_velocity", "pr_clearance_velocity",
            "pr_merge_to_reject_ratio", "exposure_to_maintenance",
        ]:
            feats[k] = np.nan

    return feats


def build_feature_matrix(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Converts raw multi-row training_dataset.csv into package-snapshot level:
      X: engineered feature DataFrame
      y: 5-column target DataFrame
      meta: DataFrame with ['package_name', 'snapshot_cutoff', 'num_rows']
    """
    X_rows, y_rows, meta_rows = [], [], []

    for (pkg, cutoff_str), grp in train_df.groupby(["package_name", "snapshot_cutoff"], sort=True):
        feats = transform_package_df(grp, cutoff_date=cutoff_str)
        X_rows.append(feats)
        y_rows.append(grp[TARGET_COLS].iloc[0].to_dict())
        meta_rows.append({
            "package_name": pkg,
            "snapshot_cutoff": cutoff_str,
            "num_rows": len(grp),
        })

    return pd.DataFrame(X_rows), pd.DataFrame(y_rows), pd.DataFrame(meta_rows)


if __name__ == "__main__":
    train_raw = pd.read_csv("training_dataset.csv")
    X_train, y_train, meta = build_feature_matrix(train_raw)

    test_raw = pd.read_csv("Prediction Rounds - Round 1 - Input.csv")
    X_test = pd.DataFrame([transform_package_df(test_raw, cutoff_date="2026-07-01")])

    assert list(X_train.columns) == list(X_test.columns), "Column mismatch!"
    assert not np.isinf(X_train.to_numpy(dtype=float)).any(), "Inf found in X_train!"
    assert not np.isinf(X_test.to_numpy(dtype=float)).any(), "Inf found in X_test!"

    print(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
    print(f"X_test  shape: {X_test.shape}")
    print(f"Columns match 100%: {list(X_train.columns) == list(X_test.columns)}")
    print(f"Total NaNs in X_test: {int(X_test.isna().sum().sum())}")
    nan_cols = X_train.isna().sum()
    nan_cols = nan_cols[nan_cols > 0]
    print(f"Columns with NaNs in X_train ({len(nan_cols)}): {nan_cols.to_dict()}")
    print("\nSample engineered features for Round 1 Test Input (dompurify):")
    sample_keys = [
        "num_versions_12m", "package_age_days", "days_since_last_release",
        "size_pct_change", "ghsa_total_lifetime", "ghsa_total_12m", "ghsa_total_exp_decay",
        "months_since_last_ghsa", "dl_momentum_recent_vs_old", "prs_surge_3m_vs_9m",
    ]
    for k in sample_keys:
        print(f"  {k}: {X_test.loc[0, k]:.4f}")