"""Train and persist a reproducible house-price regression pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DEFAULT_TARGETS = ("SalePrice", "price", "Price", "median_house_value")


def infer_target(frame: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in frame.columns:
            raise ValueError(f"Target column '{requested}' was not found.")
        return requested
    for name in DEFAULT_TARGETS:
        if name in frame.columns:
            return name
    raise ValueError("Could not infer target column. Pass --target explicitly.")


def make_pipeline(features: pd.DataFrame) -> Pipeline:
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [name for name in features.columns if name not in numeric]
    if not numeric and not categorical:
        raise ValueError("The dataset has no feature columns.")
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([( "imputer", SimpleImputer(strategy="median"))]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    return Pipeline([
        ("preprocess", ColumnTransformer(transformers=transformers)),
        ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    ])


def train(data_path: str | Path, target: str | None = None, output: str | Path = "artifacts/house_price_model.joblib", mlflow_enabled: bool = False) -> dict:
    frame = pd.read_csv(data_path)
    target_name = infer_target(frame, target)
    frame = frame.dropna(subset=[target_name]).copy()
    y = pd.to_numeric(frame.pop(target_name), errors="raise")
    X_train, X_test, y_train, y_test = train_test_split(frame, y, test_size=0.2, random_state=42)
    pipeline = make_pipeline(X_train)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics = {
        "mae": round(float(mean_absolute_error(y_test, predictions)), 2),
        "mse": round(float(mean_squared_error(y_test, predictions)), 2),
        "rmse": round(float(mean_squared_error(y_test, predictions) ** 0.5), 2),
        "r2": round(float(r2_score(y_test, predictions)), 4),
    }
    artifact = {
        "pipeline": pipeline,
        "features": frame.columns.tolist(),
        "target": target_name,
        "metrics": metrics,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    if mlflow_enabled:
        import mlflow
        mlflow.set_experiment("house-price-prediction")
        with mlflow.start_run():
            mlflow.log_params({"model": "RandomForestRegressor", "rows": len(frame), "target": target_name})
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(output_path))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to a CSV dataset")
    parser.add_argument("--target", help="Target column name")
    parser.add_argument("--output", default="artifacts/house_price_model.joblib")
    parser.add_argument("--mlflow", action="store_true", help="Log metrics and artifact to local MLflow")
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.target, args.output, args.mlflow), indent=2))


if __name__ == "__main__":
    main()
