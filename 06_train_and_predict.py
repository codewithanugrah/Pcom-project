import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

from features import transform_package_df, TARGET_COLS, SEVERITIES

MODEL_NAMES = ["cat_rmse", "cat_log1p", "xgb_l2", "xgb_log1p", "lgb_l2", "lgb_log1p"]
ARTIFACT_PATH = os.path.join("models", "pcom_ensemble.pkl")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _build_model(model_name: str, seed: int = 42):
    if model_name in ("cat_rmse", "cat_log1p"):
        return CatBoostRegressor(
            iterations=350, depth=4, learning_rate=0.03, l2_leaf_reg=5.0,
            loss_function="RMSE", random_seed=seed, verbose=0,
        )
    if model_name in ("xgb_l2", "xgb_log1p"):
        return xgb.XGBRegressor(
            n_estimators=250, max_depth=3, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0,
            objective="reg:squarederror", random_state=seed, n_jobs=-1,
        )
    if model_name in ("lgb_l2", "lgb_log1p"):
        return lgb.LGBMRegressor(
            n_estimators=250, max_depth=3, num_leaves=7, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0,
            objective="regression", random_state=seed, n_jobs=-1, verbose=-1,
        )
    raise ValueError(f"Unknown model: {model_name}")


def _fit_one(model_name: str, X_tr: pd.DataFrame, y_tr: np.ndarray, seed: int = 42):
    if np.all(y_tr == 0):
        return None
    m = _build_model(model_name, seed=seed)
    y_fit = np.log1p(y_tr) if model_name.endswith("_log1p") else y_tr
    m.fit(X_tr, y_fit)
    return m


def _predict_one(model, model_name: str, X: pd.DataFrame) -> np.ndarray:
    if model is None:
        return np.zeros(len(X), dtype=float)
    pred = model.predict(X)
    if model_name.endswith("_log1p"):
        pred = np.expm1(pred)
    return np.clip(pred, 0.0, None)


def optimize_weights(oof_2026: np.ndarray, y_2026: np.ndarray) -> np.ndarray:
    """Finds non-negative weights summing to 1 that minimize 2026 OOF RMSE."""
    n_models = oof_2026.shape[1]
    if np.all(y_2026 == 0):
        return np.zeros(n_models, dtype=float)

    def loss(w):
        pred = oof_2026 @ w
        return np.sqrt(np.mean((y_2026 - pred) ** 2))

    init_w = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0)] * n_models
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    res = minimize(loss, init_w, method="SLSQP", bounds=bounds, constraints=constraints)
    w = np.clip(res.x, 0.0, 1.0)
    return w / w.sum() if w.sum() > 0 else init_w


def train_pipeline(data_path: str = "engineered_dataset.csv"):
    eng = pd.read_csv(data_path)
    meta = eng[["package_name", "snapshot_cutoff", "num_rows"]]
    y = eng[TARGET_COLS]
    X = eng.drop(columns=["package_name", "snapshot_cutoff", "num_rows"] + TARGET_COLS)
    feature_cols = list(X.columns)

    gkf = GroupKFold(n_splits=5)
    groups = meta["package_name"].values
    is_2026 = (meta["snapshot_cutoff"] == "2026-07-01").values

    # Store OOF predictions: shape (N_samples, N_targets, N_models)
    oof = np.zeros((len(X), len(TARGET_COLS), len(MODEL_NAMES)), dtype=float)
    # Store fitted fold models: models[target_idx][model_name] = list of 5 fold models
    fitted_models = {t_i: {m: [] for m in MODEL_NAMES} for t_i in range(len(TARGET_COLS))}

    print(f"Training 5-fold Package-Grouped CV on {len(X)} snapshots ({meta['package_name'].nunique()} packages)...")
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        for t_i, t_col in enumerate(TARGET_COLS):
            y_tr = y[t_col].iloc[tr_idx].values
            for m_i, m_name in enumerate(MODEL_NAMES):
                model = _fit_one(m_name, X_tr, y_tr, seed=42 + fold)
                fitted_models[t_i][m_name].append(model)
                oof[va_idx, t_i, m_i] = _predict_one(model, m_name, X_va)

    # Evaluate individual models on 2026 OOF
    y_2026 = y.iloc[is_2026].values
    oof_2026 = oof[is_2026]

    print("\n--- Individual Model 2026 OOF RMSE ---")
    print(f"{'Model':<12} | {'2026 RMSE':>9} | {'Low':>6} {'Mod':>6} {'High':>6} {'Crit':>6}")
    print("-" * 56)
    for m_i, m_name in enumerate(MODEL_NAMES):
        pred_m = oof_2026[:, :, m_i]
        overall = rmse(y_2026, pred_m)
        per_s = [rmse(y_2026[:, t_i], pred_m[:, t_i]) for t_i in range(4)]
        print(f"{m_name:<12} | {overall:>9.4f} | {per_s[0]:>6.3f} {per_s[1]:>6.3f} {per_s[2]:>6.3f} {per_s[3]:>6.3f}")

    # Optimize per-severity ensemble weights on 2026 OOF
    weights = np.zeros((len(TARGET_COLS), len(MODEL_NAMES)), dtype=float)
    ens_2026 = np.zeros_like(y_2026, dtype=float)

    print("\n--- Learned Per-Severity Ensemble Weights (2026 OOF) ---")
    for t_i, sev in enumerate(SEVERITIES):
        w = optimize_weights(oof_2026[:, t_i, :], y_2026[:, t_i])
        weights[t_i] = w
        ens_2026[:, t_i] = oof_2026[:, t_i, :] @ w
        w_str = ", ".join(f"{m}={wt:.2f}" for m, wt in zip(MODEL_NAMES, w) if wt >= 0.01)
        print(f"  {sev:<9} (RMSE={rmse(y_2026[:, t_i], ens_2026[:, t_i]):.4f}): {w_str or 'all-zero'}")

    ens_overall = rmse(y_2026, ens_2026)
    zero_overall = rmse(y_2026, np.zeros_like(y_2026))
    print(f"\n>>> ENSEMBLE 2026 CV RMSE: {ens_overall:.4f} (Zero Baseline: {zero_overall:.4f}, -{100*(zero_overall-ens_overall)/zero_overall:.1f}%)")

    # Save artifact
    os.makedirs("models", exist_ok=True)
    bundle = {
        "feature_cols": feature_cols,
        "model_names": MODEL_NAMES,
        "fitted_models": fitted_models,
        "weights": weights,
    }
    joblib.dump(bundle, ARTIFACT_PATH)
    print(f"Saved ensemble bundle to {ARTIFACT_PATH}")

    # Sanity check on Round 1 Input if present
    r1_path = "Prediction Rounds - Round 1 - Input.csv"
    if os.path.exists(r1_path):
        pred_r1 = predict_file(r1_path, cutoff_date="2026-07-01", quiet=True)
        gt_r1 = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
        euclid = float(np.linalg.norm(gt_r1 - pred_r1))
        print(f"\nRound 1 Sanity Check (dompurify):")
        print(f"  Pred: {[round(v, 4) for v in pred_r1]} | Euclidean Dist: {euclid:.4f}")


def predict_file(input_csv: str, cutoff_date: str = "2026-07-01", output_csv: str | None = None, quiet: bool = False) -> np.ndarray:
    bundle = joblib.load(ARTIFACT_PATH)
    raw_df = pd.read_csv(input_csv)

    feats = transform_package_df(raw_df, cutoff_date=cutoff_date)
    X_test = pd.DataFrame([feats])[bundle["feature_cols"]]

    final_pred = np.zeros(len(TARGET_COLS), dtype=float)
    for t_i in range(len(TARGET_COLS)):
        w = bundle["weights"][t_i]
        if np.all(w == 0):
            final_pred[t_i] = 0.0
            continue
        model_preds = np.zeros(len(bundle["model_names"]), dtype=float)
        for m_i, m_name in enumerate(bundle["model_names"]):
            if w[m_i] < 1e-6:
                continue
            fold_models = bundle["fitted_models"][t_i][m_name]
            fold_vals = [_predict_one(fm, m_name, X_test)[0] for fm in fold_models]
            model_preds[m_i] = float(np.mean(fold_vals))
        final_pred[t_i] = float(np.clip(np.dot(w, model_preds), 0.0, None))

    if not quiet:
        print(f"\nPrediction for {input_csv} ({len(raw_df)} version rows collapsed -> 1 row):")
        for col, val in zip(TARGET_COLS, final_pred):
            print(f"  {col:<32}: {val:.6f}")

    if output_csv:
        out_df = pd.DataFrame([dict(zip(TARGET_COLS, final_pred))])
        out_df.to_csv(output_csv, index=False)
        if not quiet:
            print(f"Saved 1-row prediction to {output_csv}")

    return final_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train 5-fold ensemble and save to models/")
    parser.add_argument("--predict", type=str, default=None, help="Path to competition Input.csv")
    parser.add_argument("--cutoff", type=str, default="2026-07-01", help="Cutoff date for --predict")
    parser.add_argument("--out", type=str, default=None, help="Optional output CSV path for 1-row prediction")
    args = parser.parse_args()

    if args.train:
        train_pipeline()
    if args.predict:
        predict_file(args.predict, cutoff_date=args.cutoff, output_csv=args.out)
    if not args.train and not args.predict:
        parser.print_help()