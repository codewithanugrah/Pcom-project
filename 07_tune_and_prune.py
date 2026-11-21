import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import optuna
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

from features import transform_package_df, TARGET_COLS, SEVERITIES

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_NAMES = ["cat_rmse", "cat_log1p", "xgb_l2", "xgb_log1p", "lgb_l2", "lgb_log1p"]
ARTIFACT_PATH = os.path.join("models", "pcom_ensemble_tuned.pkl")
BASELINE_RMSE = 0.3678


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _build_model(model_name: str, params: dict, seed: int = 42):
    if model_name in ("cat_rmse", "cat_log1p"):
        return CatBoostRegressor(
            iterations=params.get("iterations", 350),
            depth=params.get("depth", 4),
            learning_rate=params.get("learning_rate", 0.03),
            l2_leaf_reg=params.get("l2_leaf_reg", 5.0),
            subsample=params.get("subsample", 0.8),
            loss_function="RMSE",
            random_seed=seed,
            verbose=0,
        )
    if model_name in ("xgb_l2", "xgb_log1p"):
        return xgb.XGBRegressor(
            n_estimators=params.get("n_estimators", 250),
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.03),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.7),
            min_child_weight=params.get("min_child_weight", 3),
            reg_alpha=params.get("reg_alpha", 0.1),
            reg_lambda=params.get("reg_lambda", 5.0),
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
        )
    if model_name in ("lgb_l2", "lgb_log1p"):
        return lgb.LGBMRegressor(
            n_estimators=params.get("n_estimators", 250),
            max_depth=params.get("max_depth", 3),
            num_leaves=params.get("num_leaves", 7),
            learning_rate=params.get("learning_rate", 0.03),
            subsample=params.get("subsample", 0.8),
            subsample_freq=1,
            colsample_bytree=params.get("colsample_bytree", 0.7),
            min_child_samples=params.get("min_child_samples", 15),
            reg_alpha=params.get("reg_alpha", 0.1),
            reg_lambda=params.get("reg_lambda", 5.0),
            objective="regression",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    raise ValueError(f"Unknown model: {model_name}")


def _fit_one(model_name: str, X_tr: pd.DataFrame, y_tr: np.ndarray, params: dict, seed: int = 42):
    if np.all(y_tr == 0):
        return None
    m = _build_model(model_name, params, seed=seed)
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


def _cv_oof_2026(model_name: str, X: pd.DataFrame, y_vec: np.ndarray, splits, is_2026: np.ndarray, params: dict):
    oof = np.zeros(len(X), dtype=float)
    models = []
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = _fit_one(model_name, X.iloc[tr_idx], y_vec[tr_idx], params, seed=42 + fold)
        models.append(m)
        oof[va_idx] = _predict_one(m, model_name, X.iloc[va_idx])
    return oof[is_2026], rmse(y_vec[is_2026], oof[is_2026]), models


def rank_and_prune_features(X: pd.DataFrame, y_vec: np.ndarray, splits, is_2026: np.ndarray, sev_name: str) -> list[str]:
    """Ranks features via 5-fold XGB+LGB gain importance and picks the best top-K on 2026 OOF RMSE."""
    if np.all(y_vec == 0):
        return list(X.columns)

    imp = np.zeros(X.shape[1], dtype=float)
    for fold, (tr_idx, _) in enumerate(splits):
        m_xgb = _fit_one("xgb_l2", X.iloc[tr_idx], y_vec[tr_idx], {}, seed=42 + fold)
        m_lgb = _fit_one("lgb_l2", X.iloc[tr_idx], y_vec[tr_idx], {}, seed=42 + fold)
        imp += m_xgb.feature_importances_ / (m_xgb.feature_importances_.sum() + 1e-9)
        imp += m_lgb.feature_importances_ / (m_lgb.feature_importances_.sum() + 1e-9)

    ranked_cols = [X.columns[i] for i in np.argsort(imp)[::-1]]
    top5 = ", ".join(ranked_cols[:5])

    best_k, best_score = len(ranked_cols), float("inf")
    probe_model = "xgb_log1p" if sev_name in ("low", "critical") else "xgb_l2"

    for k in [15, 25, 40, 60, 85, len(ranked_cols)]:
        cols_k = ranked_cols[:k]
        _, score, _ = _cv_oof_2026(probe_model, X[cols_k], y_vec, splits, is_2026, {})
        if score < best_score:
            best_score = score
            best_k = k

    print(f"  [{sev_name:<8}] Best K={best_k:>3}/116 ({probe_model} 2026 RMSE={best_score:.4f}) | Top 5: {top5}")
    return ranked_cols[:best_k]


def tune_model(model_name: str, X: pd.DataFrame, y_vec: np.ndarray, splits, is_2026: np.ndarray, n_trials: int = 25) -> dict:
    """Optuna hyperparameter search minimizing 2026 OOF RMSE."""
    if np.all(y_vec == 0):
        return {}

    def objective(trial: optuna.Trial):
        if model_name.startswith("cat_"):
            params = {
                "iterations": trial.suggest_int("iterations", 150, 600, step=50),
                "depth": trial.suggest_int("depth", 3, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 25.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            }
        elif model_name.startswith("xgb_"):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 2, 5),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 30.0, log=True),
            }
        else:  # lgb_
            depth = trial.suggest_int("max_depth", 2, 5)
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": depth,
                "num_leaves": trial.suggest_int("num_leaves", 4, min(31, 2 ** depth)),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 30.0, log=True),
            }
        _, score, _ = _cv_oof_2026(model_name, X, y_vec, splits, is_2026, params)
        return score

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_weights(oof_2026: np.ndarray, y_2026: np.ndarray) -> np.ndarray:
    n_models = oof_2026.shape[1]
    if np.all(y_2026 == 0):
        return np.zeros(n_models, dtype=float)

    def loss(w):
        return np.sqrt(np.mean((y_2026 - oof_2026 @ w) ** 2))

    init_w = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0)] * n_models
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    res = minimize(loss, init_w, method="SLSQP", bounds=bounds, constraints=constraints)
    w = np.clip(res.x, 0.0, 1.0)
    return w / w.sum() if w.sum() > 0 else init_w


def run_tune_and_prune(data_path: str = "engineered_dataset.csv", n_trials: int = 25):
    eng = pd.read_csv(data_path)
    meta = eng[["package_name", "snapshot_cutoff", "num_rows"]]
    y = eng[TARGET_COLS]
    X = eng.drop(columns=["package_name", "snapshot_cutoff", "num_rows"] + TARGET_COLS)

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X, y, meta["package_name"].values))
    is_2026 = (meta["snapshot_cutoff"] == "2026-07-01").values
    y_2026 = y.iloc[is_2026].values

    print(f"=== STEP 1: PER-SEVERITY FEATURE PRUNING ({len(X)} snapshots, {is_2026.sum()} in 2026) ===")
    selected_features = {}
    for t_i, sev in enumerate(SEVERITIES):
        selected_features[t_i] = rank_and_prune_features(X, y[TARGET_COLS[t_i]].values, splits, is_2026, sev)

    print(f"\n=== STEP 2: OPTUNA TUNING ({n_trials} trials/model) ===")
    oof_2026 = np.zeros((is_2026.sum(), len(TARGET_COLS), len(MODEL_NAMES)), dtype=float)
    fitted_models = {t_i: {} for t_i in range(len(TARGET_COLS))}
    best_params_all = {t_i: {} for t_i in range(len(TARGET_COLS))}

    for t_i, sev in enumerate(SEVERITIES):
        y_vec = y[TARGET_COLS[t_i]].values
        X_sub = X[selected_features[t_i]]
        if np.all(y_vec == 0):
            for m_name in MODEL_NAMES:
                fitted_models[t_i][m_name] = [None] * 5
                best_params_all[t_i][m_name] = {}
            continue

        row_scores = []
        for m_i, m_name in enumerate(MODEL_NAMES):
            best_p = tune_model(m_name, X_sub, y_vec, splits, is_2026, n_trials=n_trials)
            preds_2026, score, models = _cv_oof_2026(m_name, X_sub, y_vec, splits, is_2026, best_p)
            oof_2026[:, t_i, m_i] = preds_2026
            fitted_models[t_i][m_name] = models
            best_params_all[t_i][m_name] = best_p
            row_scores.append(f"{m_name}={score:.4f}")
        print(f"  [{sev:<8}] " + " | ".join(row_scores))

    print("\n=== STEP 3: ENSEMBLE WEIGHT OPTIMIZATION (2026 OOF) ===")
    weights = np.zeros((len(TARGET_COLS), len(MODEL_NAMES)), dtype=float)
    ens_2026 = np.zeros_like(y_2026, dtype=float)

    for t_i, sev in enumerate(SEVERITIES):
        w = optimize_weights(oof_2026[:, t_i, :], y_2026[:, t_i])
        weights[t_i] = w
        ens_2026[:, t_i] = oof_2026[:, t_i, :] @ w
        w_str = ", ".join(f"{m}={wt:.2f}" for m, wt in zip(MODEL_NAMES, w) if wt >= 0.01)
        print(f"  {sev:<9} (RMSE={rmse(y_2026[:, t_i], ens_2026[:, t_i]):.4f}): {w_str or 'all-zero'}")

    ens_overall = rmse(y_2026, ens_2026)
    zero_overall = rmse(y_2026, np.zeros_like(y_2026))
    delta = ens_overall - BASELINE_RMSE
    print(f"\n>>> TUNED ENSEMBLE 2026 CV RMSE: {ens_overall:.4f} (vs Untuned {BASELINE_RMSE:.4f}: {delta:+.4f} | vs Zero {zero_overall:.4f}: -{100*(zero_overall-ens_overall)/zero_overall:.1f}%)")

    os.makedirs("models", exist_ok=True)
    bundle = {
        "selected_features": selected_features,
        "model_names": MODEL_NAMES,
        "fitted_models": fitted_models,
        "best_params": best_params_all,
        "weights": weights,
    }
    joblib.dump(bundle, ARTIFACT_PATH)
    print(f"Saved tuned bundle to {ARTIFACT_PATH}")

    r1_path = "Prediction Rounds - Round 1 - Input.csv"
    if os.path.exists(r1_path):
        pred_r1 = predict_file(r1_path, cutoff_date="2026-07-01", quiet=True)
        gt_r1 = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
        euclid = float(np.linalg.norm(gt_r1 - pred_r1))
        print(f"\nRound 1 Sanity Check (dompurify):")
        print(f"  Pred: {[round(float(v), 4) for v in pred_r1]} | Euclidean Dist: {euclid:.4f}")


def predict_file(input_csv: str, cutoff_date: str = "2026-07-01", output_csv: str | None = None, quiet: bool = False) -> np.ndarray:
    bundle = joblib.load(ARTIFACT_PATH)
    raw_df = pd.read_csv(input_csv)
    feats = transform_package_df(raw_df, cutoff_date=cutoff_date)
    X_full = pd.DataFrame([feats])

    final_pred = np.zeros(len(TARGET_COLS), dtype=float)
    for t_i in range(len(TARGET_COLS)):
        w = bundle["weights"][t_i]
        if np.all(w == 0):
            final_pred[t_i] = 0.0
            continue
        X_sub = X_full[bundle["selected_features"][t_i]]
        model_preds = np.zeros(len(bundle["model_names"]), dtype=float)
        for m_i, m_name in enumerate(bundle["model_names"]):
            if w[m_i] < 1e-6:
                continue
            fold_models = bundle["fitted_models"][t_i][m_name]
            model_preds[m_i] = float(np.mean([_predict_one(fm, m_name, X_sub)[0] for fm in fold_models]))
        final_pred[t_i] = float(np.clip(np.dot(w, model_preds), 0.0, None))

    if not quiet:
        print(f"\nPrediction for {input_csv} ({len(raw_df)} version rows collapsed -> 1 row):")
        for col, val in zip(TARGET_COLS, final_pred):
            print(f"  {col:<32}: {val:.6f}")

    if output_csv:
        pd.DataFrame([dict(zip(TARGET_COLS, final_pred))]).to_csv(output_csv, index=False)
        if not quiet:
            print(f"Saved 1-row prediction to {output_csv}")

    return final_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run feature pruning + Optuna tuning")
    parser.add_argument("--trials", type=int, default=25, help="Optuna trials per model per severity")
    parser.add_argument("--predict", type=str, default=None, help="Path to competition Input.csv")
    parser.add_argument("--cutoff", type=str, default="2026-07-01", help="Cutoff date for --predict")
    parser.add_argument("--out", type=str, default=None, help="Optional output CSV path")
    args = parser.parse_args()

    if args.tune:
        run_tune_and_prune(n_trials=args.trials)
    if args.predict:
        predict_file(args.predict, cutoff_date=args.cutoff, output_csv=args.out)
    if not args.tune and not args.predict:
        parser.print_help()