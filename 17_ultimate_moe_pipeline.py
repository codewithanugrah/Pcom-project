import argparse
import os
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from importlib import import_module
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier

from features import transform_package_df, TARGET_COLS, SEVERITIES

sim = import_module("11_simulate_leaderboard")
m06 = import_module("06_train_and_predict")
m07 = import_module("07_tune_and_prune")
m10 = import_module("10_final_model")

SOTA_NAMES = ["cat_sqrt", "xgb_sqrt", "cat_tweedie_1.3", "hurdle_cat", "cat_log1p_smear"]
ARTIFACT_PATH = os.path.join("models", "pcom_ultimate_moe.pkl")
G_MAX = 0.75


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def get_pruned_features(X: pd.DataFrame, y_vec: np.ndarray, splits, sev: str) -> list[str]:
    if np.all(y_vec == 0):
        return list(X.columns[:15])
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
    return ranked[: (25 if sev == "critical" else 15)]


def fit_sota_one(name: str, X_tr: pd.DataFrame, y_tr: np.ndarray, seed: int = 42):
    if np.all(y_tr == 0):
        return None
    if name == "cat_sqrt":
        m = CatBoostRegressor(iterations=350, depth=4, learning_rate=0.03, l2_leaf_reg=6.0, loss_function="RMSE", random_seed=seed, verbose=0)
        m.fit(X_tr, np.sqrt(y_tr))
        return m
    if name == "xgb_sqrt":
        m = xgb.XGBRegressor(n_estimators=250, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, reg_lambda=6.0, random_state=seed, n_jobs=-1)
        m.fit(X_tr, np.sqrt(y_tr))
        return m
    if name == "cat_tweedie_1.3":
        m = CatBoostRegressor(iterations=350, depth=4, learning_rate=0.03, l2_leaf_reg=6.0, loss_function="Tweedie:variance_power=1.3", random_seed=seed, verbose=0)
        m.fit(X_tr, y_tr)
        return m
    if name == "cat_log1p_smear":
        m = CatBoostRegressor(iterations=350, depth=4, learning_rate=0.03, l2_leaf_reg=6.0, loss_function="RMSE", random_seed=seed, verbose=0)
        y_log = np.log1p(y_tr)
        m.fit(X_tr, y_log)
        smear = float(np.mean(np.exp(y_log - m.predict(X_tr))))
        return (m, smear)
    if name == "hurdle_cat":
        pos_mask = y_tr > 0
        if pos_mask.sum() < 5:
            return None
        clf = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.03, l2_leaf_reg=6.0, random_seed=seed, verbose=0)
        reg = CatBoostRegressor(iterations=250, depth=3, learning_rate=0.03, l2_leaf_reg=8.0, loss_function="RMSE", random_seed=seed, verbose=0)
        clf.fit(X_tr, pos_mask.astype(int))
        reg.fit(X_tr.iloc[pos_mask], np.log1p(y_tr[pos_mask]))
        return (clf, reg)
    raise ValueError(name)


def predict_sota_one(name: str, obj, X: pd.DataFrame) -> np.ndarray:
    if obj is None:
        return np.zeros(len(X), dtype=float)
    if name in ("cat_sqrt", "xgb_sqrt"):
        return np.clip(obj.predict(X), 0.0, None) ** 2
    if name == "cat_tweedie_1.3":
        return np.clip(obj.predict(X), 0.0, None)
    if name == "cat_log1p_smear":
        m, smear = obj
        return np.clip(smear * np.exp(np.clip(m.predict(X), -5.0, 5.0)) - 1.0, 0.0, None)
    if name == "hurdle_cat":
        clf, reg = obj
        p = clf.predict_proba(X)[:, 1]
        mu = np.expm1(np.clip(reg.predict(X), 0.0, 5.0))
        return np.clip(p * mu, 0.0, None)
    raise ValueError(name)


def fit_joint_euclidean_weights(oof_26: np.ndarray, y_26: np.ndarray, sample_weights: np.ndarray) -> np.ndarray:
    n_t, n_m = 4, oof_26.shape[2]
    def loss(flat_logits):
        W = softmax_rows(flat_logits.reshape(n_t, n_m))
        pred = np.einsum("ntm,tm->nt", oof_26[:, :n_t, :], W)
        dists = np.sqrt(np.sum((y_26[:, :n_t] - pred) ** 2, axis=1) + 1e-8)
        return float(np.average(dists, weights=sample_weights) + 0.002 * np.sum(W ** 2))
    res = minimize(loss, np.zeros(n_t * n_m), method="L-BFGS-B", options={"maxiter": 300})
    return softmax_rows(res.x.reshape(n_t, n_m))


def fit_soft_gate(p_gate: np.ndarray, p_cons: np.ndarray, p_aggr: np.ndarray, y_true: np.ndarray, g_max: float = G_MAX):
    def loss(params):
        a, c = params
        g = (g_max * sigmoid(a * (p_gate - c)))[:, None]
        pred = (1.0 - g) * p_cons + g * p_aggr
        return float(np.mean(np.sqrt(np.sum((y_true - pred) ** 2, axis=1) + 1e-8)))
    res = minimize(loss, [5.0, 0.5], method="L-BFGS-B", bounds=[(0.5, 10.0), (0.05, 0.95)])
    return float(res.x[0]), float(res.x[1])


def train_ultimate(data_path: str = "engineered_dataset.csv"):
    eng_full = pd.read_csv(data_path)
    is_2026_full = (eng_full["snapshot_cutoff"] == "2026-07-01").values

    # 1. Load OOF from saved 3yr + 2yr L2/log1p bundles
    meta_26, y_26, p06_26 = sim.predict_bundle_oof("models/pcom_ensemble.pkl", eng_full, is_2026_full, filter_2025=False)
    _, _,          p07_26 = sim.predict_bundle_oof("models/pcom_ensemble_tuned.pkl", eng_full, is_2026_full, filter_2025=False)
    _, _,          p10_26 = sim.predict_bundle_oof("models/pcom_final_ensemble.pkl", eng_full, is_2026_full, filter_2025=True)

    eng = eng_full[eng_full["snapshot_cutoff"] >= "2025-07-01"].reset_index(drop=True)
    meta = eng[["package_name", "snapshot_cutoff"]]
    y = eng[TARGET_COLS].values.astype(float)
    X = eng.drop(columns=["package_name", "snapshot_cutoff", "num_rows"] + TARGET_COLS)

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X, y, meta["package_name"].values))
    is_26 = (meta["snapshot_cutoff"] == "2026-07-01").values
    vuln_mask = y_26.sum(axis=1) > 0
    zero_dists = np.linalg.norm(y_26, axis=1)

    sel_feats = {t_i: get_pruned_features(X, y[:, t_i], splits, sev) for t_i, sev in enumerate(SEVERITIES)}
    union_feats = sorted(set().union(*(sel_feats[i] for i in range(4))))

    # 2. Train Risk Gate classifiers: P(any_vuln > 0)
    print("Training Risk Gate, SOTA paradigms, and Multi-Scale Cosine kNN (5-fold)...")
    y_any = (y.sum(axis=1) > 0).astype(int)
    p_gate = np.zeros(len(X), dtype=float)
    gate_models = []
    for fold, (tr_idx, va_idx) in enumerate(splits):
        clf = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.03, l2_leaf_reg=8.0, random_seed=42 + fold, verbose=0)
        clf.fit(X[union_feats].iloc[tr_idx], y_any[tr_idx])
        p_gate[va_idx] = clf.predict_proba(X[union_feats].iloc[va_idx])[:, 1]
        gate_models.append(clf)
    p_gate_26 = p_gate[is_26]

    # 3. Train SOTA paradigms
    oof_sota = np.zeros((len(X), 5, len(SOTA_NAMES)), dtype=float)
    sota_models = {t_i: {m: [] for m in SOTA_NAMES} for t_i in range(4)}
    for m_i, m_name in enumerate(SOTA_NAMES):
        for t_i in range(4):
            cols = sel_feats[t_i]
            for fold, (tr_idx, va_idx) in enumerate(splits):
                obj = fit_sota_one(m_name, X[cols].iloc[tr_idx], y[tr_idx, t_i], seed=42 + fold)
                sota_models[t_i][m_name].append(obj)
                oof_sota[va_idx, t_i, m_i] = predict_sota_one(m_name, obj, X[cols].iloc[va_idx])

    # 4. Train Multi-Scale Cosine kNN (K=7, 15, 25)
    oof_knn = np.zeros((len(X), 5), dtype=float)
    knn_models = {t_i: [] for t_i in range(4)}
    for t_i in range(4):
        cols = sel_feats[t_i]
        for fold, (tr_idx, va_idx) in enumerate(splits):
            imp = SimpleImputer(strategy="median")
            qt = QuantileTransformer(n_quantiles=min(200, len(tr_idx)), output_distribution="normal", random_state=42 + fold)
            X_tr_s = qt.fit_transform(imp.fit_transform(X[cols].iloc[tr_idx]))
            X_va_s = qt.transform(imp.transform(X[cols].iloc[va_idx]))
            fold_knns = []
            for k in [7, 15, 25]:
                knn = KNeighborsRegressor(n_neighbors=k, weights="distance", metric="cosine")
                knn.fit(X_tr_s, np.sqrt(y[tr_idx, t_i]))
                oof_knn[va_idx, t_i] += (np.clip(knn.predict(X_va_s), 0.0, None) ** 2) / 3.0
                fold_knns.append(knn)
            knn_models[t_i].append((imp, qt, fold_knns))

    # 5. Fit Expert A (Conservative) & Expert B (Aggressive) + Soft-Capped Risk Gate
    oof_all_26 = np.concatenate([
        p06_26[:, :, None], p07_26[:, :, None], p10_26[:, :, None],
        oof_sota[is_26], oof_knn[is_26][:, :, None],
    ], axis=2)

    W_cons = fit_joint_euclidean_weights(oof_all_26, y_26, np.ones(len(y_26)))
    W_aggr = fit_joint_euclidean_weights(oof_all_26, y_26, np.where(vuln_mask, 4.0, 1.0))

    p_cons_26 = np.zeros_like(y_26)
    p_cons_26[:, :4] = np.einsum("ntm,tm->nt", oof_all_26[:, :4, :], W_cons)
    p_aggr_26 = np.zeros_like(y_26)
    p_aggr_26[:, :4] = np.einsum("ntm,tm->nt", oof_all_26[:, :4, :], W_aggr)

    a, c = fit_soft_gate(p_gate_26, p_cons_26, p_aggr_26, y_26, g_max=G_MAX)
    g_26 = G_MAX * sigmoid(a * (p_gate_26 - c))
    p_moe_26 = (1.0 - g_26[:, None]) * p_cons_26 + g_26[:, None] * p_aggr_26

    # Save ultimate bundle
    os.makedirs("models", exist_ok=True)
    bundle = {
        "sel_feats": sel_feats,
        "union_feats": union_feats,
        "gate_models": gate_models,
        "sota_models": sota_models,
        "knn_models": knn_models,
        "W_cons": W_cons,
        "W_aggr": W_aggr,
        "gate_params": (a, c, G_MAX),
    }
    joblib.dump(bundle, ARTIFACT_PATH)

    # 6. Summary Metrics
    dists = np.linalg.norm(y_26 - p_moe_26, axis=1)
    rmse_v = float(np.sqrt(np.mean((y_26 - p_moe_26) ** 2)))
    beats0 = 100.0 * np.mean(dists[vuln_mask] < zero_dists[vuln_mask])
    print(f"\n=== ULTIMATE SOFT-CAPPED MoE (g_max={G_MAX}, a={a:.2f}, c={c:.2f}) ===")
    print(f"  2026 RMSE: {rmse_v:.4f} | All484 Euc: {dists.mean():.4f} | VulnMean: {dists[vuln_mask].mean():.4f} | VulnMed: {np.median(dists[vuln_mask]):.4f} | Beats0%: {beats0:.1f}%")
    print(f"  Saved bundle to {ARTIFACT_PATH}")

    # 7. Per-Package Unseen 2026 Breakdown (Top 20 Vulnerable + 5 Zero-Vuln Sanity Check)
    vuln_idx = np.where(vuln_mask)[0]
    vuln_idx = vuln_idx[np.argsort(y_26[vuln_idx].sum(axis=1))[::-1]]

    print(f"\n=== UNSEEN 2026 PACKAGES: ACTUAL vs PREDICTED [Low, Mod, High, Crit] ===")
    print(f"{'Package':<24} | {'Actual [L,M,H,C]':<16} | {'Pred [L,M,H,C]':<23} | {'Euclid':>6} | {'Zero':>6} | {'% Red':>6} | {'Gate g':>6}")
    print("-" * 102)
    for idx in vuln_idx[:20]:
        pkg = meta_26.loc[idx, "package_name"][:24]
        gt, pr = y_26[idx, :4], p_moe_26[idx, :4]
        d, z = float(dists[idx]), float(zero_dists[idx])
        red = 100.0 * (1.0 - d / z) if z > 0 else 0.0
        gt_s = f"[{int(gt[0])},{int(gt[1])},{int(gt[2])},{int(gt[3])}]"
        pr_s = f"[{pr[0]:.2f},{pr[1]:.2f},{pr[2]:.2f},{pr[3]:.2f}]"
        print(f"{pkg:<24} | {gt_s:<16} | {pr_s:<23} | {d:>6.3f} | {z:>6.3f} | {red:>5.1f}% | {g_26[idx]:>6.2f}")

    # Round 1 check
    r1_path = "Prediction Rounds - Round 1 - Input.csv"
    if os.path.exists(r1_path):
        pred_r1, g_r1 = predict_file(r1_path, cutoff_date="2026-07-01", quiet=True)
        gt_r1 = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
        r1_d = float(np.linalg.norm(gt_r1 - pred_r1))
        print(f"\nRound 1 (dompurify) | GT [1,1,0,0] | Pred {[round(float(v), 2) for v in pred_r1[:4]]} | Euclid: {r1_d:.4f} ({sim.simulated_rank(r1_d)}) | Gate g: {g_r1:.2f}")


def load_all_bundles(models_dir: str | Path | None = None) -> dict:
    if models_dir is None:
        models_dir = Path(__file__).resolve().parent / "models"
    else:
        models_dir = Path(models_dir)
    b06 = joblib.load(models_dir / "pcom_ensemble.pkl")
    b07 = joblib.load(models_dir / "pcom_ensemble_tuned.pkl")
    b10 = joblib.load(models_dir / "pcom_final_ensemble.pkl")
    bmoe = joblib.load(models_dir / "pcom_ultimate_moe.pkl")
    return {"b06": b06, "b07": b07, "b10": b10, "bmoe": bmoe}


def predict_dataframe(df: pd.DataFrame, cutoff_date: str = "2026-07-01", preloaded_bundles: dict | None = None) -> tuple[np.ndarray, float]:
    if preloaded_bundles is None:
        preloaded_bundles = load_all_bundles()

    b06 = preloaded_bundles["b06"]
    b07 = preloaded_bundles["b07"]
    b10 = preloaded_bundles["b10"]
    bundle = preloaded_bundles["bmoe"]

    feats = transform_package_df(df, cutoff_date=cutoff_date)
    X_full = pd.DataFrame([feats])

    # 1. Sub-model 06
    X06 = X_full[b06["feature_cols"]]
    p06 = np.zeros(len(TARGET_COLS), dtype=float)
    for t_i in range(len(TARGET_COLS)):
        w = b06["weights"][t_i]
        if np.all(w == 0):
            continue
        m_preds = [
            float(np.mean([m06._predict_one(fm, m_name, X06)[0] for fm in b06["fitted_models"][t_i][m_name]]))
            if w[m_i] >= 1e-6 else 0.0
            for m_i, m_name in enumerate(b06["model_names"])
        ]
        p06[t_i] = max(0.0, float(np.dot(w, m_preds)))

    # 2. Sub-model 07
    p07 = np.zeros(len(TARGET_COLS), dtype=float)
    for t_i in range(len(TARGET_COLS)):
        w = b07["weights"][t_i]
        if np.all(w == 0):
            continue
        X_sub = X_full[b07["selected_features"][t_i]]
        m_preds = [
            float(np.mean([m07._predict_one(fm, m_name, X_sub)[0] for fm in b07["fitted_models"][t_i][m_name]]))
            if w[m_i] >= 1e-6 else 0.0
            for m_i, m_name in enumerate(b07["model_names"])
        ]
        p07[t_i] = max(0.0, float(np.dot(w, m_preds)))

    # 3. Sub-model 10
    p10 = np.zeros(len(TARGET_COLS), dtype=float)
    for t_i in range(len(TARGET_COLS)):
        w = b10["weights"][t_i]
        if np.all(w == 0):
            continue
        X_sub = X_full[b10["selected_features"][t_i]]
        m_preds = [
            float(np.mean([m10._predict_one(m, m_name, X_sub)[0] for m in b10["fitted_models"][t_i][m_name]]))
            if w[m_i] >= 1e-6 else 0.0
            for m_i, m_name in enumerate(b10["model_names"])
        ]
        p10[t_i] = max(0.0, float(np.dot(w, m_preds)))

    # 4. Risk Gate
    X_u = X_full[bundle["union_feats"]]
    p_gate = float(np.mean([clf.predict_proba(X_u)[0, 1] for clf in bundle["gate_models"]]))
    a, c, g_max = bundle["gate_params"]
    g = float(g_max * sigmoid(a * (p_gate - c)))

    # 5. SOTA paradigms
    p_sota = np.zeros((5, len(SOTA_NAMES)), dtype=float)
    for m_i, m_name in enumerate(SOTA_NAMES):
        for t_i in range(4):
            X_sub = X_full[bundle["sel_feats"][t_i]]
            objs = bundle["sota_models"][t_i][m_name]
            p_sota[t_i, m_i] = float(np.mean([predict_sota_one(m_name, obj, X_sub)[0] for obj in objs]))

    # 6. Multi-scale Cosine kNN
    p_knn = np.zeros(5, dtype=float)
    for t_i in range(4):
        X_sub = X_full[bundle["sel_feats"][t_i]]
        vals = []
        for imp, qt, fold_knns in bundle["knn_models"][t_i]:
            X_s = qt.transform(imp.transform(X_sub))
            for knn in fold_knns:
                vals.append(float(np.clip(knn.predict(X_s)[0], 0.0, None) ** 2))
        p_knn[t_i] = float(np.mean(vals))

    cand = np.concatenate([p06[:, None], p07[:, None], p10[:, None], p_sota, p_knn[:, None]], axis=1)
    p_cons = np.zeros(5, dtype=float)
    p_aggr = np.zeros(5, dtype=float)
    p_cons[:4] = np.einsum("tm,tm->t", cand[:4, :], bundle["W_cons"])
    p_aggr[:4] = np.einsum("tm,tm->t", cand[:4, :], bundle["W_aggr"])

    final_pred = np.clip((1.0 - g) * p_cons + g * p_aggr, 0.0, None)
    return final_pred, g


def predict_file(input_csv: str, cutoff_date: str = "2026-07-01", output_csv: str | None = None, quiet: bool = False):
    raw_df = pd.read_csv(input_csv)
    final_pred, g = predict_dataframe(raw_df, cutoff_date=cutoff_date)

    if not quiet:
        print(f"\n=== ULTIMATE MoE PREDICTION FOR {input_csv} (Gate g={g:.2f}) ===")
        for col, val in zip(TARGET_COLS, final_pred):
            print(f"  {col:<32}: {val:.6f}")

    if output_csv:
        pd.DataFrame([dict(zip(TARGET_COLS, final_pred))]).to_csv(output_csv, index=False)
        if not quiet:
            print(f"Saved 1-row prediction to {output_csv}")

    return final_pred, g


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train and evaluate Ultimate Risk-Gated MoE")
    parser.add_argument("--predict", type=str, default=None, help="Path to competition Input.csv")
    parser.add_argument("--cutoff", type=str, default="2026-07-01", help="Cutoff date for --predict")
    parser.add_argument("--out", type=str, default=None, help="Optional output CSV path")
    args = parser.parse_args()

    if args.train:
        train_ultimate()
    if args.predict:
        predict_file(args.predict, cutoff_date=args.cutoff, output_csv=args.out)
    if not args.train and not args.predict:
        parser.print_help()