import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_historical():
    print("Loading engineered dataset...")
    df = pd.read_csv("training_dataset_engineered.csv", keep_default_na=False)

    # Isolate targets and features
    exclude_cols = ['package_name', 'version', 'purl', 'published_at', 'package_created_at']
    target_cols = [c for c in df.columns if c.startswith('target_ghsa_')]

    # We will use a random 20% holdout for evaluation
    # (For a true paper, you would do a time-based split, but this works to verify the math)
    test_df = df.sample(frac=0.2, random_state=42)
    X_test = test_df.drop(columns=[c for c in exclude_cols + target_cols if c in test_df.columns], errors='ignore')
    X_test = X_test.fillna(0)

    severities = ['low', 'moderate', 'high', 'critical']

    print("\n" + "=" * 60)
    print("HISTORICAL EVALUATION: HYBRID ENSEMBLE (20% HOLDOUT)")
    print("=" * 60)

    for sev in severities:
        target_name = f"target_ghsa_{sev}_count_90d"
        tweedie_file = f"xgb_model_{target_name}_tweedie.json"
        poisson_file = f"xgb_model_{target_name}_poisson.json"

        y_true = test_df[target_name]

        try:
            # Align features
            model_a = xgb.XGBRegressor()
            model_a.load_model(tweedie_file)
            expected_features = model_a.get_booster().feature_names
            X_aligned = X_test[expected_features] if expected_features else X_test

            # Predict
            preds_a = model_a.predict(X_aligned)
            model_b = xgb.XGBRegressor()
            model_b.load_model(poisson_file)
            preds_b = model_b.predict(X_aligned)

            final_preds = np.maximum(0, (preds_a + preds_b) / 2.0)

            mae = mean_absolute_error(y_true, final_preds)
            rmse = np.sqrt(mean_squared_error(y_true, final_preds))

            print(f"\nSeverity: {sev.upper()}")
            print(f"  -> Mean Absolute Error (MAE): {mae:.4f}")
            print(f"  -> Root Mean Squared Error (RMSE): {rmse:.4f}")

            # Show actual vs predicted for non-zero cases
            non_zero_mask = y_true > 0
            if non_zero_mask.sum() > 0:
                print("  -> Sample (Where Actual > 0):")
                sample_idx = test_df[non_zero_mask].head(4).index
                for idx in sample_idx:
                    package_name = test_df.loc[idx, 'package_name']
                    actual_val = y_true.loc[idx]
                    # Find the position of this index in the prediction array
                    pred_pos = np.where(X_test.index == idx)[0][0]
                    pred_val = final_preds[pred_pos]
                    print(f"     {package_name}: Actual = {actual_val}  |  Predicted = {pred_val:.4f}")
            else:
                print("  -> (No actual vulnerabilities in this historical split either - data is highly zero-inflated)")

        except Exception as e:
            print(f"\nSeverity: {sev.upper()} -> Missing models or alignment error ({e})")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    evaluate_historical()