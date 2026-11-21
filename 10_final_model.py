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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import HuberRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

from features import transform_package_df, TARGET_COLS, SEVERITIES

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_NAMES = [
    "cat_rmse", "cat_log1p",
    "xgb_l2", "xgb_log1p",
    "lgb_l2", "lgb_log1p",
    "huber_log1p",
]
SEEDS = [42, 1024, 2026]
ARTIFACT_PATH = os.path.join("models", "pcom_final_ensemble.pkl")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _build_model(model_name: str, params: dict, n_train: int, seed: int = 42):
    if model_name.startswith("cat_"):
        return CatBoostRegressor(
            iterations=params.get("iterations", 350),
            depth=params.get("depth", 4),
            learning_rate=params.get("learning_rate", 0.03),
            l2_leaf_reg=params.get("l2_leaf_reg", 6.0),
            subsample=params.get("subsample", 0.8),
            loss_function="RMSE",
            random_seed=seed,
            verbose=0,
        )
    if model_name.startswith("xgb_"):
        return xgb.XGBRegressor(
            n_estimators=params.get("n_estimators", 250),
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.03),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.7),
            min_child_weight=params.get("min_child_weight", 3),
            reg_alpha=params.get("reg_alpha", 0.1),
            reg_lambda=params.get("reg_lambda", 6.0),
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
        )
    if model_name.startswith("lgb_"):
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
            reg_lambda=params.get("reg_lambda", 6.0),
            objective="regression",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    if model_name == "huber_log1p":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("qt", QuantileTransformer(n_quantiles=min(200, n_train), output_distribution="normal", random_state=seed)),
            ("huber", HuberRegressor(alpha=params.get("alpha", 50.0), max_iter=500)),
        ])
    raise ValueError(f"Unknown model: {model_name}")


def _fit_one(model_name: str, X_tr: pd.DataFrame, y_tr: np.ndarray, params: dict, seed: int = 42):
    if np.all(y_tr == 0):
        return None
    m = _build_model(model_name, params, n_train=len(X_tr), seed=seed)
    y_fit = np.log1p(y_tr) if model_name.endswith("_log1p") else y_tr
    m.fit(X_tr, y_fit)
    return m


def _predict_one(model, model_name: str, X: pd.DataFrame) -> np.ndarray:
    if model is None:
        return np.zeros(len(X), dtype=float)
    pred = model.predict(X)
    if model_name.endswith("_log1p"):
        pred = np.expm1(np.clip(pred, -5.0, 5.0))
    return np.clip(pred, 0.0, None)


def _cv_oof_2026(model_name: str, X: pd.DataFrame, y_vec: np.ndarray, splits, is_2026: np.ndarray, params: dict, seeds: list[int]):
    oof = np.zeros(len(X), dtype=float)
    models = []
    for fold, (tr_idx, va_idx) in enumerate(splits):
        fold_pred = np.zeros(len(va_idx), dtype=float)
        for s in seeds:
            m = _fit_one(model_name, X.iloc[tr_idx], y_vec[tr_idx], params, seed=s + fold)
            models.append(m)
            fold_pred += _predict_one(m, model_name, X.iloc[va_idx]) / len(seeds)
        oof[va_idx] = fold_pred
    return oof[is_2026], rmse(y_vec[is_2026], oof[is_2026]), models


def prune_features(X: pd.DataFrame, y_vec: np.ndarray, splits, is_2026: np.ndarray, sev: str) -> list[str]:
    if np.all(y_vec == 0):
        return list(X.columns)

    imp = np.zeros(X.shape[1], dtype=float)
    use_log = sev in ("low", "critical")
    for fold, (tr_idx, _) in enumerate(splits):
        m = xgb.XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, reg_lambda=6.0,
            random_state=42 + fold, n_jobs=-1,
        )
        m.fit(X.iloc[tr_idx], np.log1p(y_vec[tr_idx]) if use_log else y_vec[tr_idx])
        imp += m.feature_importances_

    ranked = [X.columns[i] for i in np.argsort(imp)[::-1]]
    best_k, best_err = 25, float("inf")
    probe = "xgb_log1p" if use_log else "xgb_l2"

    for k in [15, 25, 40, 60]:
        _, err, _ = _cv_oof_2026(probe, X[ranked[:k]], y_vec, splits, is_2026, {}, seeds=[42])
        if err < best_err:
            best_err, best_k = err, k

    print(f"  [{sev:<8}] K={best_k:>2}/129 (probe 2026 RMSE={best_err:.4f}) | Top 5: {', '.join(ranked[:5])}")
    return ranked[:best_k]


def tune_model(model_name: str, X: pd.DataFrame, y_vec: np.ndarray, splits, is_2026: np.ndarray, n_trials: int = 15) -> dict:
    if np.all(y_vec == 0) or model_name == "huber_log1p":
        return {"alpha": 50.0} if model_name == "huber_log1p" else {}

    def objective(trial: optuna.Trial):
        if model_name.startswith("cat_"):
            params = {
                "iterations": trial.suggest_int("iterations", 200, 550, step=50),
                "depth": trial.suggest_int("depth", 3, 5),
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.06, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2.0, 20.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.65, 0.95),
            }
        elif model_name.startswith("xgb_"):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 150, 450, step=50),
                "max_depth": trial.suggest_int("max_depth", 2, 4),
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.06, log=True),
                "subsample": trial.suggest_float("subsample", 0.65, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.45, 0.9),
                "min_child_weight": trial.suggest_int("min_child_weight", 2, 12),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 2.0, 25.0, log=True),
            }
        else:
            depth = trial.suggest_int("max_depth", 2, 4)
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 150, 450, step=50),
                "max_depth": depth,
                "num_leaves": trial.suggest_int("num_leaves", 4, min(16, 2 ** depth)),
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.06, log=True),
                "subsample": trial.suggest_float("subsample", 0.65, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.45, 0.9),
                "min_child_samples": trial.suggest_int("min_child_samples", 8, 35),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 2.0, 25.0, log=True),
            }
        _, score, _ = _cv_oof_2026(model_name, X, y_vec, splits, is_2026, params, seeds=[42])
        return score

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_weights(oof_2026: np.ndarray, y_2026: np.ndarray, max_weight: float = 0.60) -> np.ndarray:
    n_models = oof_2026.shape[1]
    if np.all(y_2026 == 0):
        return np.zeros(n_models, dtype=float)

    def loss(w):
        return np.sqrt(np.mean((y_2026 - oof_2026 @ w) ** 2))

    init_w = np.ones(n_models) / n_models
    bounds = [(0.0, max_weight)] * n_models
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    res = minimize(loss, init_w, method="SLSQP", bounds=bounds, constraints=constraints)
    w = np.clip(res.x, 0.0, max_weight)
    return w / w.sum() if w.sum() > 0 else init_w


def train_final(data_path: str = "engineered_dataset.csv", n_trials: int = 15, max_weight: float = 0.60):
    eng = pd.read_csv(data_path)
    eng = eng[eng["snapshot_cutoff"] >= "2025-07-01"].reset_index(drop=True)

    meta = eng[["package_name", "snapshot_cutoff"]]
    y = eng[TARGET_COLS]
    X = eng.drop(columns=["package_name", "snapshot_cutoff", "num_rows"] + TARGET_COLS)

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X, y, meta["package_name"].values))
    is_2026 = (meta["snapshot_cutoff"] == "2026-07-01").values
    y_2026 = y.iloc[is_2026].values.astype(float)

    print(f"=== FINAL MODEL TRAINING (2025+2026: {len(X)} snapshots, {meta['package_name'].nunique()} packages, {is_2026.sum()} in 2026) ===")
    selected_features = {
        t_i: prune_features(X, y[TARGET_COLS[t_i]].values, splits, is_2026, sev)
        for t_i, sev in enumerate(SEVERITIES)
    }

    print(f"\n=== LIGHT OPTUNA TUNING ({n_trials} trials) + {len(SEEDS)}-SEED AVERAGING ===")
    oof_2026 = np.zeros((is_2026.sum(), len(TARGET_COLS), len(MODEL_NAMES)), dtype=float)
    fitted_models = {t_i: {} for t_i in range(len(TARGET_COLS))}
    best_params_all = {t_i: {} for t_i in range(len(TARGET_COLS))}

    for t_i, sev in enumerate(SEVERITIES):
        y_vec = y[TARGET_COLS[t_i]].values
        X_sub = X[selected_features[t_i]]
        if np.all(y_vec == 0):
            for m_name in MODEL_NAMES:
                fitted_models[t_i][m_name] = [None]
                best_params_all[t_i][m_name] = {}
            continue

        row_scores = []
        for m_i, m_name in enumerate(MODEL_NAMES):
            best_p = tune_model(m_name, X_sub, y_vec, splits, is_2026, n_trials=n_trials)
            preds_2026, score, models = _cv_oof_2026(m_name, X_sub, y_vec, splits, is_2026, best_p, seeds=SEEDS)
            oof_2026[:, t_i, m_i] = preds_2026
            fitted_models[t_i][m_name] = models
            best_params_all[t_i][m_name] = best_p
            row_scores.append(f"{m_name}={score:.4f}")
        print(f"  [{sev:<8}] " + " | ".join(row_scores))

    print(f"\n=== FINAL ENSEMBLE WEIGHTS (max_weight={max_weight:.2f}) ===")
    weights = np.zeros((len(TARGET_COLS), len(MODEL_NAMES)), dtype=float)
    ens_2026 = np.zeros_like(y_2026, dtype=float)

    for t_i, sev in enumerate(SEVERITIES):
        w = optimize_weights(oof_2026[:, t_i, :], y_2026[:, t_i], max_weight=max_weight)
        weights[t_i] = w
        ens_2026[:, t_i] = oof_2026[:, t_i, :] @ w
        w_str = ", ".join(f"{m}={wt:.2f}" for m, wt in zip(MODEL_NAMES, w) if wt >= 0.01)
        print(f"  {sev:<9} (RMSE={rmse(y_2026[:, t_i], ens_2026[:, t_i]):.4f}): {w_str or 'all-zero'}")

    ens_overall = rmse(y_2026, ens_2026)
    zero_overall = rmse(y_2026, np.zeros_like(y_2026))
    print(f"\n>>> FINAL 2026 CV RMSE: {ens_overall:.4f} (Zero Baseline: {zero_overall:.4f}, -{100*(zero_overall-ens_overall)/zero_overall:.1f}%)")

    os.makedirs("models", exist_ok=True)
    bundle = {
        "selected_features": selected_features,
        "model_names": MODEL_NAMES,
        "fitted_models": fitted_models,
        "best_params": best_params_all,
        "weights": weights,
    }
    joblib.dump(bundle, ARTIFACT_PATH)
    print(f"Saved final ensemble bundle to {ARTIFACT_PATH}")

    r1_path = "Prediction Rounds - Round 1 - Input.csv"
    if os.path.exists(r1_path):
        pred_r1 = predict_file(r1_path, cutoff_date="2026-07-01", quiet=True)
        gt_r1 = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
        print(f"\nRound 1 Sanity Check (dompurify):")
        print(f"  Pred: {[round(float(v), 4) for v in pred_r1]} | Euclidean Dist: {float(np.linalg.norm(gt_r1 - pred_r1)):.4f}")


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
            models = bundle["fitted_models"][t_i][m_name]
            model_preds[m_i] = float(np.mean([_predict_one(m, m_name, X_sub)[0] for m in models]))
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
    parser.add_argument("--train", action="store_true", help="Train final 3-seed 5-fold ensemble")
    parser.add_argument("--trials", type=int, default=15, help="Optuna trials per model per severity")
    parser.add_argument("--predict", type=str, default=None, help="Path to competition Input.csv")
    parser.add_argument("--cutoff", type=str, default="2026-07-01", help="Cutoff date for --predict")
    parser.add_argument("--out", type=str, default=None, help="Optional output CSV path")
    args = parser.parse_args()

    if args.train:
        train_final(n_trials=args.trials)
    if args.predict:
        predict_file(args.predict, cutoff_date=args.cutoff, output_csv=args.out)
    if not args.train and not args.predict:
        parser.print_help()