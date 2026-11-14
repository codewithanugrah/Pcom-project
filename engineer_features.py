import pandas as pd
import numpy as np


def engineer_dataset(input_csv="training_dataset_comprehensive.csv", output_csv="training_dataset_engineered.csv"):
    print("Loading raw dataset...")
    df = pd.read_csv(input_csv, keep_default_na=False)

    print("Engineering advanced ratios, velocities, and temporal features...")
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
            df[f'download_momentum_M{i}_to_M{i + 1}'] = (df.get(f'package_downloads_M{i}', 0) + 1) / (
                        df.get(f'package_downloads_M{i + 1}', 0) + 1)

    # 5. Temporal Features (Age in Days) - SAFELY HANDLED (Moved outside the loop!)
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

    print(f"Adding new features including package and version age...")
    df.to_csv(output_csv, index=False)
    print(f"✅ Engineered dataset saved to {output_csv}. New shape: {df.shape}")


if __name__ == "__main__":
    engineer_dataset()