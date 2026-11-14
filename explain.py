import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

print("Loading engineered dataset...")
df = pd.read_csv("training_dataset_engineered.csv", keep_default_na=False)

exclude_cols = ['package_name', 'version', 'purl', 'published_at', 'package_created_at']
target_cols = [c for c in df.columns if c.startswith('target_ghsa_')]

X = df.drop(columns=[c for c in exclude_cols + target_cols if c in df.columns], errors='ignore')
X = X.fillna(0)

model_filename = "xgb_model_target_ghsa_critical_count_90d_tweedie.json"

if not os.path.exists(model_filename):
    print(f"❌ Error: Could not find {model_filename}. Did you run train_ensemble.py yet?")
    exit(1)

print(f"Loading Critical severity Tweedie model...")
model = xgb.XGBRegressor()
model.load_model(model_filename)

# Align columns just in case
expected_features = model.get_booster().feature_names
if expected_features:
    missing_cols = [col for col in expected_features if col not in X.columns]
    if missing_cols:
        missing_df = pd.DataFrame(0.0, index=X.index, columns=missing_cols)
        X = pd.concat([X, missing_df], axis=1)
    X = X[expected_features]

print("Calculating SHAP values (analyzing a 5,000 row sample for speed)...")
# For speed on large datasets, we analyze a random sample of 5,000 rows
X_sample = X.sample(n=min(5000, len(X)), random_state=42)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

print("Generating SHAP Summary Plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, show=False)
plt.title("SHAP Feature Importance: Critical Vulnerabilities (Tweedie Ensemble)")
plt.tight_layout()
plt.savefig("shap_critical_summary.png", dpi=300, bbox_inches='tight')
print("✅ Saved high-res plot to shap_critical_summary.png")