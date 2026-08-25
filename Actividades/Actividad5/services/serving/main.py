"""Serving microservice: exposes the gold-layer model over HTTP.

Only reads from the gold volume — decoupled from ingestion/cleaning/training,
which may run on a schedule while this stays up as a long-lived API.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

GOLD_DIR = Path(os.environ.get("GOLD_DIR", "/data/gold"))
MODEL_PATH = GOLD_DIR / "model.joblib"
METRICS_PATH = GOLD_DIR / "metrics.json"

app = FastAPI(title="Churn Prediction Service")
_model = None


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(extra="allow")


@app.on_event("startup")
def load_model() -> None:
    global _model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}; run the training service first.")
    _model = joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/metrics")
def metrics() -> dict:
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Metrics not available yet.")
    return json.loads(METRICS_PATH.read_text())


@app.post("/predict")
def predict(features: CustomerFeatures) -> dict:
    row = pd.DataFrame([features.model_dump()])
    probability = _model.predict_proba(row)[0, 1]
    return {"churn_probability": float(probability), "churn_predicted": bool(probability >= 0.5)}
