# PCOM: Predictive Vulnerability Count Pipeline (DevSecOps ML)

A robust, schema-compliant machine learning pipeline designed to predict future 90-day software vulnerability counts for NPM packages. Built for the Predictive Compliance and Operations (PCOM) project, this architecture models vulnerability occurrences as rare discrete events using Poisson regression, strictly enforcing chronological data leakage prevention and rapid inference constraints.

---

## 🚀 Project Overview

Predicting open-source supply chain risk requires accounting for zero-inflated target distributions and temporal dependencies. This project automates the extraction of historical GitHub issue/PR velocity, NPM metadata, package sizes, download volumes, and past security advisories (GHSA) to train a highly generalized Extreme Gradient Boosting (XGBoost) regression model.

---

## 📂 Inputs & Outputs

### Input Schema
The pipeline accepts tabular CSV data adhering to the defined PCOM record schema, containing:
* **Metadata & Identifiers:** `package_name`, `version`, `purl`, `published_at`, `package_created_at`.
* **Repository Size & Dependencies:** `unpacked_size`, `file_count`, `dependency_count`, `dev_dependency_count`, etc.
* **Rolling 12-Month Windows ( $M0$ to $M11$ ):** Monthly historical metrics tracking opened/closed issues, merged/closed pull requests, stable release counts, download volumes (`package_downloads_M0` to `M11`), and past vulnerability counts.

### Output Schema
The rapid inference script outputs a single-row (or multi-row) CSV containing strictly the prediction goal metrics required by the evaluation rubric:
* `target_ghsa_low_count_90d`
* `target_ghsa_moderate_count_90d`
* `target_ghsa_high_count_90d`
* `target_ghsa_critical_count_90d`
* `target_ghsa_unknown_count_90d`

---

## 🛠️ Techniques & Methodologies

1. **The "Depth Pivot" Time-Series Architecture:** Flattens package histories so that every stable version acts as an independent training instance, anchored precisely to its `published_at` timestamp with a 12-month backward-looking feature window ( $M0$ to $M11$ ) and a 90-day forward-looking target window.
2. **Poisson Distribution Regression (`count:poisson`):** Tailored specifically for modeling rare, non-negative event counts, preventing invalid negative predictions and accurately mapping distance-to-target metrics.
3. **Severe Data Leakage Prevention:** Employs `GroupKFold` and `GroupShuffleSplit` grouped strictly by `package_name`, ensuring the model generalizes across entirely unseen repositories rather than memorizing minor version patches.
4. **Optimized Local Execution:** Utilizes CPU-optimized histogram-based gradient boosting (`tree_method='hist'`) on Apple Silicon M2 Pro architecture for rapid training and instant inference.

---

## ⚙️ How to Run the Project

### Prerequisites
* Python 3.10+
* Virtual environment (`venv`) activated
* GitHub Personal Access Token (exported as an environment variable for the ETL phase):
  ```bash
  export GITHUB_TOKEN="your_github_token_here"
  ```

### 1. Install Dependencies
```bash
pip install pandas numpy xgboost scikit-learn aiohttp semver
```

### 2. Run the ETL Pipeline (Data Extraction)
To extract historical data for packages listed in `packages.txt`:
```bash
python main.py
```
*(Features incremental disk checkpointing, asynchronous concurrency semaphores, and automatic GitHub rate-limit handling).*

### 3. Train the Models
To execute cross-validation, evaluate on unseen test splits, and bake the final severity-specific weights into disk:
```bash
python train.py
```
This generates the required model weight files: `xgb_model_target_ghsa_*_count_90d.json`.

### 4. Run Rapid Inference (Competition Mode)
To evaluate incoming test data and generate the required competition output within the strict 10-second runtime limit:
```bash
python predict.py test_input.csv
```
This reads `test_input.csv`, evaluates predictions in milliseconds, and outputs `prediction_output.csv`.