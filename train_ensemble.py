import pandas as pd
import numpy as np
import xgboost as xgb
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# 1. Load your Comprehensive & Engineered Dataset
df = pd.read_csv("training_dataset_engineered.csv", keep_default_na=False)
df = df.dropna(subset=['package_name'])

exclude_cols = ['package_name', 'version', 'purl', 'published_at', 'package_created_at']
target_cols = [c for c in df.columns if c.startswith('target_ghsa_')]

X = df.drop(columns=[c for c in exclude_cols + target_cols if c in df.columns], errors='ignore')
X = X.fillna(0) 

print(f"Loaded {df.shape[0]} rows.")
print("Preparing to train the Hybrid Ensemble (Tweedie + Poisson) on GPU...")

# =====================================================================
# 2. PASTE YOUR BEST OPTUNA PARAMETERS HERE
# (Update these with the results from your 100-trial GPU run!)
# =====================================================================
best_params_dict = {
    'target_ghsa_low_count_90d': {'max_depth': 8, 'learning_rate': 0.08577247053237218, 'n_estimators': 350, 'tweedie_variance_power': 1.8381587000774957, 'max_delta_step': 0},
    'target_ghsa_moderate_count_90d': {'max_depth': 8, 'learning_rate': 0.09880059818767375, 'n_estimators': 400, 'tweedie_variance_power': 1.73790488377007, 'max_delta_step': 3},
    'target_ghsa_high_count_90d': {'max_depth': 8, 'learning_rate': 0.09830551329731078, 'n_estimators': 400, 'tweedie_variance_power': 1.8044558292645192, 'max_delta_step': 0},
    'target_ghsa_critical_count_90d': {'max_depth': 8, 'learning_rate': 0.07822901488749721, 'n_estimators': 400, 'tweedie_variance_power': 1.801403898573774, 'max_delta_step': 0},
}

# 3. Train the Ensemble for each severity
for target in target_cols:
    y = df[target]
    
    print(f"\n{'='*50}")
    print(f"Training Hybrid Ensemble for: {target}")
    print(f"{'='*50}")
    
    # Bypass for the synthetic unknown target (or if a target has pure zeros)
    if target == 'target_ghsa_unknown_count_90d' or y.sum() == 0:
        print("Synthetic or zero-variance target detected. Generating safe baseline ensemble...")
        
        # Model A Baseline
        model_a = xgb.XGBRegressor(objective='reg:tweedie', tree_method='hist', device='cuda', max_depth=3, learning_rate=0.01, n_estimators=100)
        model_a.fit(X, y)
        model_a.save_model(f"xgb_model_{target}_tweedie.json")
        
        # Model B Baseline
        model_b = xgb.XGBRegressor(objective='count:poisson', tree_method='hist', device='cuda', max_depth=3, learning_rate=0.01, n_estimators=100)
        model_b.fit(X, y)
        model_b.save_model(f"xgb_model_{target}_poisson.json")
        continue

    # Load the specific parameters for this severity
    base_params = best_params_dict.get(target, {})
    weights = np.where(y > 0, 50.0, 1.0)

    # --- MODEL A: THE TWEEDIE SHIELD ---
    print("  -> Training Model A: XGBoost Tweedie (Conservative & Safe)...")
    tweedie_params = base_params.copy()
    tweedie_params.update({
        'objective': 'reg:tweedie',
        'tree_method': 'hist',
        'device': 'cuda',
        'random_state': 42
    })
    
    model_a = xgb.XGBRegressor(**tweedie_params)
    model_a.fit(X, y, sample_weight=weights)
    model_a.save_model(f"xgb_model_{target}_tweedie.json")

    # --- MODEL B: THE POISSON SWORD ---
    print("  -> Training Model B: XGBoost Poisson (Aggressive Risk-Detection)...")
    poisson_params = base_params.copy()
    # Poisson objective crashes if we pass tweedie_variance_power, so we pop it out
    poisson_params.pop('tweedie_variance_power', None) 
    poisson_params.update({
        'objective': 'count:poisson',
        'tree_method': 'hist',
        'device': 'cuda',
        'random_state': 42
    })
    
    model_b = xgb.XGBRegressor(**poisson_params)
    model_b.fit(X, y, sample_weight=weights)
    model_b.save_model(f"xgb_model_{target}_poisson.json")
    
    print(f"  ✅ Ensemble pair successfully saved for {target}.")