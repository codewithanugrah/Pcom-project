# PCOM-01 npm Vulnerability Forecasting Engine

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Benchmark](https://img.shields.io/badge/RSE-0.0459-brightgreen.svg)](#benchmarks)

High-performance, production-ready machine learning service for forecasting 90-day GitHub Security Advisory (GHSA) arrival counts across the npm package ecosystem. Built for the PCOM-01 evaluation platform with sub-50ms inference latency, zero hard-cap distortion, and wire-protocol compliance.

---

## 🌟 Key Architectural Innovations

```
                            Incoming Package Time-Series
                                         │
        ┌────────────────────────────────┴────────────────────────────────┐
        ▼                                                                 ▼
 ┌───────────────┐                                                 ┌───────────────┐
 │ PARTITION D1  │                                                 │ PARTITION D2  │
 │ (Anchor Core) │                                                 │(Distribution) │
 └───────┬───────┘                                                 └───────┬───────┘
         │                                                                 │
         ├─ Low:      max_depth=2, leaf=6, l2=2.5, iter=160                ├─ Low:      max_depth=2, leaf=8, l2=3.0, iter=160
         ├─ Moderate: max_depth=3, leaf=5, l2=1.5, iter=180                ├─ Moderate: max_depth=3, leaf=4, l2=1.0, iter=180
         ├─ High:     max_depth=2, leaf=6, l2=2.0, iter=150                ├─ High:     max_depth=2, leaf=6, l2=2.0, iter=150
         ├─ Critical: max_depth=1, leaf=15, l2=10.0                        ├─ Critical: max_depth=1, leaf=15, l2=10.0
         └─ Unknown:  0.000000                                             └─ Unknown:  0.000000
         │                                                                 │
         └───────────────────────────────┬─────────────────────────────────┘
                                         │
                                         ▼
                       Per-Target Weighted Partition Blend
                       • Low:      80% D1  +  20% D2  (Anchor priority)
                       • Moderate: 20% D1  +  80% D2  (Distribution priority)
                       • High:     50% D1  +  50% D2  (Variance cancel)
                       • Critical: 50% D1  +  50% D2  (Heavy L2 shrinkage)
                       • Unknown:  Strictly 0.000000
                                         │
                                         ▼
                       Smooth Soft-Saturation Guardrails
                       • Low ≤ 3.0 | Mod ≤ 4.5 | High ≤ 3.5 | Crit ≤ 1.5
```

### 1. Class-Specialized Dual-Partition Ensembling
Software package vulnerabilities suffer from extreme zero-inflation (>97% zeros) and severe overdispersion ($\text{Var} \gg \text{Mean}$). Rather than using a single multi-output regressor where Moderate dominates 90% of split decisions, we train **dedicated estimators per severity class** across two de-correlated data partitions ($\rho \approx 0.74$):
- **Partition D1 (Anchor Core):** Fitted on high-impact framework distributions, capturing historical advisory velocity and vulnerability persistence.
- **Partition D2 (Longitudinal Distribution):** Fitted on longitudinal ecosystem metrics across 484 packages, anchoring false-positive suppression on clean packages.

### 2. Natural Canonical Exponential Link
All estimators use regularized Poisson deviance with an exponential link:
$$\hat{y}(x) = \exp(w^T x + b_0)$$
For clean packages without recent CVE activity, predictions automatically shrink to $0.001 - 0.005$. For active packages with recent CVE clusters, predictions scale naturally without being artificially clamped.

### 3. Smooth Soft-Saturation
Unlike crude hard caps (`np.clip(pred, 0, C)`) which destroy derivative information and rank ordering, we implement monotonic soft-saturation buffers to bound maximum leaf extrapolation on extreme outliers without cutting off genuine signals.

---

## 📊 Benchmarks & Performance

Evaluated against the official competition ground truth test packages:

| Evaluation Criterion | Ground Truth Target | Model Prediction | Metric Score | Inference Latency |
| :--- | :---: | :---: | :---: | :---: |
| **Round 1 (`dompurify`)** | `[1, 1, 0, 0, 0]` | `[0.6355, 1.0735, 0.3840, 0.0122, 0.0]` | **`0.5347` (Euclidean)** | 50.1 ms |
| **Round 2 (`axios`)** | `[0, 1, 0, 0, 0]` | `[0.0131, 0.2255, 0.1297, 0.0343, 0.0]` | **`0.0459` (RSE)** | 37.9 ms |
| **Clean Packages (316 pkgs)**| `[0, 0, 0, 0, 0]` | Average across all zeros: `0.0154` | **`1.399` (Total SSE)**| < 40 ms |

---

## 📁 Repository Structure

```
.
├── champion_model.py      # Production forecasting engine & SemVer resolution
├── main.py                # FastAPI microservice for Compy prediction wire-protocol
├── features.py            # Feature definitions and time-series aggregations
├── train.py               # Self-contained training pipeline for dual-partition ensemble
├── test_schema_complete.py# End-to-end schema resilience & latency test suite
├── deploy_vm.sh           # Automated 1-click VM deployment script
├── pyproject.toml         # Package configuration & metadata
├── requirements.txt       # Production dependencies
├── Dockerfile             # Container configuration for production deployment
└── models/
    └── pcom_champion_ensemble.joblib # Production model bundle (1.3 MB)
```

---

## 🚀 Quickstart

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/codewithanugrah/Pcom-project.git
cd Pcom-project

# Create virtual environment & install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify Schema Compliance & Latency
Run the offline verification suite:
```bash
python test_schema_complete.py
```
Expected output:
```
--- TEST 1: Zero-History Package (4 versions, 0 GHSAs) ---
✅ TEST 1 PASSED: Zero-history predictions effectively throttled to ~0.0 under RSE!
--- TEST 2: High-Activity Vulnerable Package (Historical GHSAs) ---
✅ TEST 2 PASSED: High-activity vulnerable package evaluated with bounded predictions!
--- TEST 3: Schema Resilience (BOM + Lowercase _m0..m11 + No purl) ---
✅ TEST 3 PASSED: Normalized column aliases, BOM, and missing purl perfectly!
--- TEST 4: Benchmark Latency Across 10 Iterations ---
✅ TEST 4 PASSED: Well within 2-second competition latency budget (< 100ms average)!
```

### 3. Run the API Service Locally
```bash
python main.py
```
Check health endpoint:
```bash
curl http://localhost:8000/health
# {"health":"up","models_ready":true}
```

### 4. Retrain Model
To retrain the dual-partition model:
```bash
python train.py
```

### 5. Deploy to Production VM
To push the latest code, tags, and restart the VM service:
```bash
./deploy_vm.sh
```

---

## 🔒 Security & Data Confidentiality

- **Confidentiality:** All training datasets, captured evaluation rounds, and raw time-series CSVs are strictly ignored via `.gitignore` and never committed.
- **Fail-Safe Fallback:** The `/predict` background task catches network exceptions and submits an emergency zero-vector fallback payload to prevent turn dropouts.
