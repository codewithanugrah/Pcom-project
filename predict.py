import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import warnings

# Suppress pandas fragmentation warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def engineer_inference_features(df):
    """Applies the exact same feature engineering logic used in training to the test data."""
    # 1. Dependency Ratios
    df['dev_to_prod_ratio'] = df.get('dev_dependency_count', 0) / (df.get('dependency_count', 0) + 1)
    df['total_dependencies'] = df.get('dependency_count', 0) + df.get('dev_dependency_count', 0) + df.get(
        'optional_dependency_count', 0)

    for i in range(12):
        # 2. PR Merge Ratios
        df[f'pr_merge_ratio_M{i}'] = df.get(f'pull_requests_merged_M{i}', 0) / (
                df.get(f'pull_requests_opened_M{i}', 0) + 1)

        # 3. Issue Resolution Velocity
        df[f'issue_close_ratio_M{i}'] = df.get(f'issues_closed_M{i}', 0) / (df.get(f'issues_opened_M{i}', 0) + 1)

        # 4. Download Momentum
        if i < 11:
            curr_dl = df.get(f'package_downloads_M{i}', 0)
            next_dl = df.get(f'package_downloads_M{i + 1}', 0)
            df[f'download_momentum_M{i}_to_M{i + 1}'] = (curr_dl + 1) / (next_dl + 1)

    # 5. Temporal Features (Age in Days) - SAFELY HANDLED
    now = pd.Timestamp.now('UTC')

    if 'package_created_at' in df.columns:
        created_dt = pd.to_datetime(df['package_created_at'], errors='coerce', utc=True)
        df['package_age_days'] = (now - created_dt).dt.days.fillna(0)
    else:
        df['package_age_days'] = 0.0

    if 'published_at' in df.columns:
        published_dt = pd.to_datetime(df['published_at'], errors='coerce', utc=True)
        df['version_age_days'] = (now - published_dt).dt.days.fillna(0)
    else:
        df['version_age_days'] = 0.0

    return df


def predict(input_csv):
    print(f"Loading input data from {input_csv}...")
    df = pd.read_csv(input_csv, keep_default_na=False)

    # 1. Engineer the features dynamically on the test set
    df = engineer_inference_features(df)

    # 2. Isolate features
    exclude_cols = ['package_name', 'version', 'purl', 'published_at', 'package_created_at']
    target_cols = [c for c in df.columns if c.startswith('target_ghsa_')]

    X = df.drop(columns=[c for c in exclude_cols + target_cols if c in df.columns], errors='ignore')
    X = X.fillna(0)

    predictions_output = {}
    severities = ['low', 'moderate', 'high', 'critical', 'unknown']

    for sev in severities:
        target_name = f"target_ghsa_{sev}_count_90d"
        tweedie_file = f"xgb_model_{target_name}_tweedie.json"
        poisson_file = f"xgb_model_{target_name}_poisson.json"

        # Safe fallback: If either model is missing, default to 0.0
        if not os.path.exists(tweedie_file) or not os.path.exists(poisson_file):
            print(f"⚠️ Missing ensemble models for {sev}. Defaulting to 0.0.")
            predictions_output[target_name] = [0.0] * len(X)
            continue

        # Load Model A (Tweedie Shield)
        model_a = xgb.XGBRegressor()
        model_a.load_model(tweedie_file)

        # Schema Alignment: Guarantee columns perfectly match the training signature
        expected_features = model_a.get_booster().feature_names
        if expected_features:
            missing_cols = [col for col in expected_features if col not in X.columns]
            if missing_cols:
                missing_df = pd.DataFrame(0.0, index=X.index, columns=missing_cols)
                X = pd.concat([X, missing_df], axis=1)
            X_aligned = X[expected_features]
        else:
            X_aligned = X

        preds_a = model_a.predict(X_aligned)

        # Load Model B (Poisson Sword)
        model_b = xgb.XGBRegressor()
        model_b.load_model(poisson_file)
        preds_b = model_b.predict(X_aligned)

        # The Hybrid Blend: Average the two predictions, bounded to 0
        final_preds = np.maximum(0, (preds_a + preds_b) / 2.0)
        predictions_output[target_name] = final_preds.tolist()

    # Format the final output as requested by the professor's schema
    final_results = []
    for i in range(len(df)):
        result_row = {
            "purl": df.iloc[i]['purl'],
            "predictions": {
                "target_ghsa_low_count_90d": float(predictions_output["target_ghsa_low_count_90d"][i]),
                "target_ghsa_moderate_count_90d": float(predictions_output["target_ghsa_moderate_count_90d"][i]),
                "target_ghsa_high_count_90d": float(predictions_output["target_ghsa_high_count_90d"][i]),
                "target_ghsa_critical_count_90d": float(predictions_output["target_ghsa_critical_count_90d"][i]),
                "target_ghsa_unknown_count_90d": float(predictions_output["target_ghsa_unknown_count_90d"][i])
            }
        }
        final_results.append(result_row)

    output_filename = "predictions.json"
    with open(output_filename, "w") as f:
        json.dump(final_results, f, indent=4)

    print(f"✅ Ensemble predictions successfully saved to {output_filename}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <input_dataset.csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    predict(input_file)