"""REST API for serving the persisted house-price model."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/house_price_model.joblib"))
app = FastAPI(title="House Price Prediction API", version="1.0.0")


class PredictionRequest(BaseModel):
    features: dict[str, Any]


@lru_cache
def load_artifact() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}. Train a model first.")
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok" if MODEL_PATH.exists() else "model_not_ready"}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    try:
        artifact = load_artifact()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"target": artifact["target"], "features": artifact["features"], "metrics": artifact["metrics"]}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, float]:
    try:
        artifact = load_artifact()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    missing = sorted(set(artifact["features"]) - set(request.features))
    if missing:
        raise HTTPException(status_code=422, detail={"missing_features": missing})
    record = {name: request.features[name] for name in artifact["features"]}
    value = float(artifact["pipeline"].predict(pd.DataFrame([record]))[0])
    return {"predicted_price": round(value, 2)}
