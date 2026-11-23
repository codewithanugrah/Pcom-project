import io
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import uvicorn
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

from champion_model import ChampionForecaster, TARGET_COLS

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

app = FastAPI(title="PCOM-01 Vulnerability Forecasting API")

# Pre-load champion ensemble into memory at startup
print("[STARTUP] Pre-loading Champion Poisson Ensemble into memory...", flush=True)
_t_load = time.time()
try:
    forecaster = ChampionForecaster(model_path=BASE_DIR / "models" / "pcom_champion_ensemble.joblib")
    print(f"[STARTUP] Successfully loaded champion ensemble in {time.time() - _t_load:.2f}s. Service ready for Round 3!", flush=True)
except Exception as e:
    print(f"[STARTUP ERROR] Failed to pre-load champion ensemble: {e}", flush=True)
    traceback.print_exc()
    forecaster = None


class EvaluationBody(BaseModel):
    testdata_download_endpoint: str
    prediction_upload_endpoint: str


def _infer_cutoff_date(df: pd.DataFrame) -> str:
    """
    Infers the snapshot cutoff date.
    Per competition contract, the outcome window is 2026-07-01 to 2026-09-30 (Q3 2026),
    so the default cutoff is strictly '2026-07-01'.
    If 'snapshot_cutoff' is explicitly provided in the input CSV, it is respected.
    """
    if "snapshot_cutoff" in df.columns:
        vals = df["snapshot_cutoff"].dropna()
        if len(vals) > 0:
            return str(vals.iloc[0]).strip()
    return "2026-07-01"


def run_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes in-memory inference using the Champion Poisson Ensemble
    on package time-series data and produces a single-row prediction DataFrame
    with exact columns: target_ghsa_low_count_90d, ..., target_ghsa_unknown_count_90d.
    Values are guaranteed to be finite, non-negative float64 (>= 0.0).
    """
    if df is None or len(df) == 0:
        return pd.DataFrame([{col: 0.0 for col in TARGET_COLS}], columns=TARGET_COLS)

    cutoff_str = _infer_cutoff_date(df)
    try:
        if forecaster is not None:
            pred_df, _ = forecaster.predict(df, cutoff_date=cutoff_str)
            preds = pred_df[TARGET_COLS].iloc[0].values
        else:
            preds = np.zeros(len(TARGET_COLS), dtype=float)
    except Exception as exc:
        print(f"[ERROR] Inference exception in run_model: {exc}", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        preds = np.zeros(len(TARGET_COLS), dtype=float)

    # Validate output length and replace any non-finite / negative values
    if len(preds) < len(TARGET_COLS):
        padded = np.zeros(len(TARGET_COLS), dtype=float)
        padded[: len(preds)] = preds
        preds = padded
    else:
        preds = preds[: len(TARGET_COLS)]

    preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
    preds = np.clip(preds, 0.0, None)

    row = {col: round(float(val), 6) for col, val in zip(TARGET_COLS, preds)}
    return pd.DataFrame([row], columns=TARGET_COLS)


def start_prediction(download_url: str, upload_url: str):
    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    print(f"\n[{ts_str}] [ROUND 3] New prediction request received from Compy", flush=True)
    print(f"[{ts_str}] Download URL: {download_url}", flush=True)
    print(f"[{ts_str}] Upload URL:   {upload_url}", flush=True)

    input_data = None
    prediction_data = None
    raw_saved_path = None

    try:
        # 1. Download test CSV from Compy
        res = requests.get(download_url, timeout=30)
        res.raise_for_status()
        raw_csv_text = res.text

        # 2. Capture raw input CSV to disk (safe try/except block)
        try:
            capture_dir = BASE_DIR / "captured_inputs" / "round_3"
            capture_dir.mkdir(parents=True, exist_ok=True)
            raw_saved_path = capture_dir / f"round3_input_{ts_str}.csv"
            with open(raw_saved_path, "w", encoding="utf-8") as f:
                f.write(raw_csv_text)
            print(f"[{ts_str}] Captured raw input CSV to: {raw_saved_path}", flush=True)
        except Exception as save_err:
            print(f"[{ts_str}] Warning: Could not save input capture to disk: {save_err}", flush=True)

        # 3. Parse input CSV
        input_data = pd.read_csv(io.StringIO(raw_csv_text), sep=",", decimal=".")
        cutoff_str = _infer_cutoff_date(input_data)
        print(f"[{ts_str}] Input data shape: {input_data.shape} | Inferred cutoff: {cutoff_str}", flush=True)

        # 4. Run model inference
        t0 = time.time()
        prediction_data = run_model(input_data)
        elapsed_ms = (time.time() - t0) * 1000.0

        assert len(prediction_data) == 1, "Prediction DataFrame must contain exactly 1 row"
        pred_dict = prediction_data.iloc[0].to_dict()
        print(f"[{ts_str}] Predicted targets ({elapsed_ms:.1f}ms): {pred_dict}", flush=True)

        # 5. Capture prediction output to disk
        try:
            pred_saved_path = capture_dir / f"round3_pred_{ts_str}.csv"
            prediction_data.to_csv(pred_saved_path, index=False)
        except Exception:
            pass

        # 6. Upload 1-row CSV to Compy
        output_buffer = io.StringIO()
        prediction_data.to_csv(output_buffer, sep=",", decimal=".", header=True, index=False)
        output_buffer.seek(0)

        files = {"predictions": ("prediction.csv", output_buffer.getvalue(), "text/csv")}
        upload_res = requests.post(upload_url, files=files, timeout=30)
        upload_res.raise_for_status()
        print(f"[{ts_str}] [SUCCESS] Compy Upload Result: {upload_res.status_code} - {upload_res.text}", flush=True)

    except Exception as e:
        print(f"[{ts_str}] [ERROR] Prediction workflow failed: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()

        # Fail-safe upload fallback: Attempt to upload 1-row 0.0s so Compy does not drop our turn
        try:
            fallback_df = pd.DataFrame([{col: 0.0 for col in TARGET_COLS}], columns=TARGET_COLS)
            buf = io.StringIO()
            fallback_df.to_csv(buf, sep=",", decimal=".", header=True, index=False)
            buf.seek(0)
            files = {"predictions": ("prediction.csv", buf.getvalue(), "text/csv")}
            fb_res = requests.post(upload_url, files=files, timeout=15)
            print(f"[{ts_str}] Fallback zero-prediction uploaded: {fb_res.status_code}", flush=True)
        except Exception as fb_err:
            print(f"[{ts_str}] Fallback upload also failed: {fb_err}", flush=True)


@app.post("/predict", status_code=200)
async def prediction(payload: EvaluationBody, background_tasks: BackgroundTasks):
    background_tasks.add_task(
        start_prediction,
        payload.testdata_download_endpoint,
        payload.prediction_upload_endpoint,
    )
    return {"status": "prediction started"}


@app.get("/health", status_code=200)
async def health():
    models_ready = forecaster is not None and forecaster.bundle is not None
    return {"health": "up", "models_ready": models_ready}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
