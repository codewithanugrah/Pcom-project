import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Load the scaled dataset
df = pd.read_csv("training_dataset.csv")

# 2. Separate features, targets, and group constraints
exclude_cols = ['package_name', 'version', 'purl', 'published_at']
target_cols = [c for c in df.columns if c.startswith('target_ghsa_')]

X = df.drop(columns=exclude_cols + target_cols)
groups = df['package_name']

print(f"Loaded {df.shape[0]} rows across {df['package_name'].nunique()} packages.")

# 3. Train and evaluate for each severity constraint
for target in target_cols:
    y = df[target]
    print(f"\n--- Predicting {target} ---")
    
    if y.sum() == 0:
        print("No vulnerabilities observed for this severity in the dataset. Skipping.")
        continue

    # Standard 5-fold cross validation for the larger dataset
    gkf = GroupKFold(n_splits=5) 
    
    rmse_scores = []
    mae_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        # 4. CPU-Optimized XGBoost
        model = xgb.XGBRegressor(
            objective='count:poisson',
            tree_method='hist', # Highly efficient on CPU
            n_jobs=-1,          # Utilize all CPU cores
            n_estimators=300,   # Increased capacity to learn from more packages
            learning_rate=0.03,
            max_depth=5,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        # 5. Evaluate strictly on RMSE and MAE
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        
        print(f"Fold {fold+1} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
        
    print(f"Average RMSE: {np.mean(rmse_scores):.4f}")
    print(f"Average MAE: {np.mean(mae_scores):.4f}")