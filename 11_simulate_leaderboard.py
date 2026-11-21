import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from features import TARGET_COLS, transform_package_df

# Round 1 leaderboard reference scores (sorted ascending)
R1_BOARD = [
    0.1738, 1.1176, 1.1409, 1.2156, 1.3094, 1.3753,
    1.4142, 1.4142, 1.5212, 1.5831, 1.7184, 2.1482,
    2.2724, 3.0193, 3.7075, 4.1231, 5.3447,
]


def simulated_rank(euclid_dist: float) -> str:
    rank = 1 + sum(1 for s in R1_BOARD if s < euclid_dist)
    return f"#{rank}/{len(R1_BOARD) + 1}"


def predict_bundle_oof(bundle_path: str, eng_df: pd.DataFrame, is_2026: np.ndarray, filter_2025: bool = False):
    bundle = joblib.load(bundle_path)
    df = eng_df[eng_df["snapshot_cutoff"] >= "2025-07-01"].reset_index(drop=True) if filter_2025 else eng_df.copy()
    meta = df[["package_name", "snapshot_cutoff"]]
    y = df[TARGET_COLS].values.astype(float)
    X = df.drop(columns=["package_name", "snapshot_cutoff", "num_rows"] + TARGET_COLS)
    mask_26 = (meta["snapshot_cutoff"] == "2026-07-01").values

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X, y, meta["package_name"].values))
    oof = np.zeros_like(y, dtype=float)

    for fold, (_, va_idx) in enumerate(splits):
        X_va_full = X.iloc[va_idx]
        for t_i in range(len(TARGET_COLS)):
            w = bundle["weights"][t_i]
            if np.all(w == 0):
                continue
            cols = bundle["selected_features"][t_i] if "selected_features" in bundle else bundle["feature_cols"]
            X_va = X_va_full[cols]
            preds_m = np.zeros((len(va_idx), len(bundle["model_names"])), dtype=float)
            for m_i, m_name in enumerate(bundle["model_names"]):
                if w[m_i] < 1e-6:
                    continue
                models = bundle["fitted_models"][t_i][m_name]
                # Handle both 5-fold (1 seed) and 15-model (5 folds x 3 seeds)
                n_seeds = max(1, len(models) // 5)
                fold_models = models[fold * n_seeds : (fold + 1) * n_seeds]
                p = np.zeros(len(va_idx), dtype=float)
                for fm in fold_models:
                    if fm is None:
                        continue
                    raw_p = fm.predict(X_va)
                    if m_name.endswith("_log1p"):
                        raw_p = np.expm1(np.clip(raw_p, -5.0, 5.0))
                    p += np.clip(raw_p, 0.0, None) / len(fold_models)
                preds_m[:, m_i] = p
            oof[va_idx, t_i] = np.clip(preds_m @ w, 0.0, None)

    return meta[mask_26].reset_index(drop=True), y[mask_26], oof[mask_26]


if __name__ == "__main__":
    eng = pd.read_csv("engineered_dataset.csv")
    is_2026 = (eng["snapshot_cutoff"] == "2026-07-01").values

    configs = [
        ("06_untuned_3yr", "models/pcom_ensemble.pkl", False),
        ("07_tuned_3yr",   "models/pcom_ensemble_tuned.pkl", False),
        ("10_final_25_26", "models/pcom_final_ensemble.pkl", True),
    ]

    print("=== MODEL COMPARISON: ALL 484 PACKAGES vs VULNERABLE PACKAGES ONLY (EUCLIDEAN DISTANCE) ===")
    print(f"{'Model':<18} | {'All 484 Mean':>12} | {'Vuln-Only Mean':>14} | {'Vuln Median':>11} | {'Beats Zero %':>12}")
    print("-" * 80)

    best_meta, best_y, best_oof = None, None, None
    for name, path, f25 in configs:
        try:
            m26, y26, p26 = predict_bundle_oof(path, eng, is_2026, filter_2025=f25)
        except Exception as e:
            print(f"{name:<18} | Skipped ({e})")
            continue

        dists = np.linalg.norm(y26 - p26, axis=1)
        zero_dists = np.linalg.norm(y26, axis=1)
        vuln_mask = y26.sum(axis=1) > 0

        beats_zero = 100.0 * np.mean(dists[vuln_mask] < zero_dists[vuln_mask])
        print(
            f"{name:<18} | {dists.mean():>12.4f} | {dists[vuln_mask].mean():>14.4f} | "
            f"{np.median(dists[vuln_mask]):>11.4f} | {beats_zero:>11.1f}%"
        )
        if name == "10_final_25_26":
            best_meta, best_y, best_oof = m26, y26, p26

    # Print per-package simulated leaderboard on the top 20 vulnerable 2026 packages
    if best_meta is not None:
        vuln_idx = np.where(best_y.sum(axis=1) > 0)[0]
        # Sort by total ground-truth vulnerabilities descending
        vuln_idx = vuln_idx[np.argsort(best_y[vuln_idx].sum(axis=1))[::-1]]

        print(f"\n=== SIMULATED COMPETITION ROUNDS: TOP 20 UNSEEN VULNERABLE 2026 PACKAGES (10_final_25_26) ===")
        print(f"{'Package':<24} | {'GT [L,M,H,C]':<15} | {'Pred [L,M,H,C]':<23} | {'Euclid':>6} | {'Zero':>6} | {'R1 Rank'}")
        print("-" * 96)
        for idx in vuln_idx[:20]:
            pkg = best_meta.loc[idx, "package_name"][:24]
            gt = best_y[idx, :4]
            pr = best_oof[idx, :4]
            d = float(np.linalg.norm(best_y[idx] - best_oof[idx]))
            z = float(np.linalg.norm(best_y[idx]))
            gt_s = f"[{int(gt[0])},{int(gt[1])},{int(gt[2])},{int(gt[3])}]"
            pr_s = f"[{pr[0]:.2f},{pr[1]:.2f},{pr[2]:.2f},{pr[3]:.2f}]"
            print(f"{pkg:<24} | {gt_s:<15} | {pr_s:<23} | {d:>6.3f} | {z:>6.3f} | {simulated_rank(d)}")