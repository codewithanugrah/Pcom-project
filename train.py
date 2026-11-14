import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 1. Load your comprehensive engineered dataset
df = pd.read_csv("training_dataset_engineered.csv", keep_default_na=False)
df = df.dropna(subset=['package_name'])

exclude_cols = ['package_name', 'version', 'purl', 'published_at', 'package_created_at']
target_cols = [c for c in df.columns if c.startswith('target_ghsa_')]

X = df.drop(columns=[c for c in exclude_cols + target_cols if c in df.columns], errors='ignore')
X = X.fillna(0)

groups = df['package_name']
print(f"Loaded {df.shape[0]} rows across {groups.nunique()} packages.")
print("Running in Optuna-Only Mode on GPU (CUDA)...")

# Dictionary to store the winning parameters for the final output
final_params_dict = {}

# 2. Tune for each severity
for target in target_cols:
    y = df[target]

    print(f"\n{'=' * 50}")
    print(f"Tuning Hyperparameters for: {target}")
    print(f"{'=' * 50}")

    if y.sum() == 0:
        print("No vulnerabilities observed in the training set. Skipping.")
        continue

    # Bypass Optuna entirely for the synthetic unknown target
    if target == 'target_ghsa_unknown_count_90d':
        print("Synthetic target detected. Skipping optimization.")
        continue

    # --- OPTUNA HYPERPARAMETER TUNING ---
    print("Running Optuna Search (100 Trials) using 3-Fold CV on GPU...")


    def objective(trial):
        param = {
            'objective': 'reg:tweedie',
            'tree_method': 'hist',
            'device': 'cuda',  # GPU Acceleration
            'random_state': 42,
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 400, step=50),
            'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.1, 1.9),
            'max_delta_step': trial.suggest_int('max_delta_step', 0, 3)
        }

        gkf = GroupKFold(n_splits=3)
        cv_mae = []

        for t_idx, v_idx in gkf.split(X, y, groups):
            X_t, y_t = X.iloc[t_idx], y.iloc[t_idx]
            X_v, y_v = X.iloc[v_idx], y.iloc[v_idx]

            weights_t = np.where(y_t > 0, 50.0, 1.0)

            model = xgb.XGBRegressor(**param)
            model.fit(X_t, y_t, sample_weight=weights_t)
            preds = model.predict(X_v)

            # Safety net: Prune trial if gradient explosion causes NaNs
            if np.isnan(preds).any():
                raise optuna.TrialPruned("Gradient explosion detected. Pruning unstable trial.")

            cv_mae.append(mean_absolute_error(y_v, preds))

        return np.mean(cv_mae)


    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    best_params = study.best_params
    print(f"✅ Optuna Found Best Params: {best_params}")

    # Store the winning dictionary
    final_params_dict[target] = best_params

# --- FINAL OUTPUT GENERATION ---
print("\n" + "=" * 50)
print("🎉 ALL OPTIMIZATIONS COMPLETE!")
print("Copy and paste this exact dictionary into your train_ensemble.py script:\n")
print("best_params_dict = {")
for key, value in final_params_dict.items():
    print(f"    '{key}': {value},")
print("}")
print("=" * 50 + "\n")