# House Price Prediction MLOps

An end-to-end regression project for estimating a home's sale price. It follows the reference project design with data validation, preprocessing, model training and evaluation, REST inference, containerization, and CI.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.train --data data/raw/sample_housing.csv
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` to try the API. The model artifact is written to `artifacts/house_price_model.joblib`.

## Train on a real dataset

Put a CSV in `data/raw/` and point the trainer to it. The target column can be `SalePrice`, `price`, `Price`, or `median_house_value`; use `--target` for another name.

```powershell
python -m src.train --data data/raw/train.csv --target SalePrice
```

The pipeline automatically imputes missing values, one-hot encodes categorical fields, and trains a Random Forest regressor. It reports MAE, MSE, RMSE, and R2 on a held-out test set. Add `--mlflow` to log the run locally with MLflow.

## API

`POST /predict` accepts a JSON object whose keys match the training feature names.

```json
{
  "area_sqft": 1800,
  "bedrooms": 3,
  "bathrooms": 2,
  "age_years": 8,
  "location": "Suburb",
  "parking": "Yes"
}
```

`GET /health` confirms the service and model status. `GET /model-info` exposes the artifact metadata.

## Quality and deployment

```powershell
pytest
docker build -t house-price-api .
docker run -p 8000:8000 house-price-api
```

GitHub Actions runs tests on pushes and pull requests. For a production dataset, version the data outside Git, retain MLflow run metadata, and redeploy only a validated model artifact.

## Deploy on Render

This repository includes `render.yaml`, which instructs Render to install dependencies, train the included sample model during the build, start the FastAPI service on Render's assigned port, and check `/health` before serving traffic.

1. Push this folder to a GitHub repository.
2. In the [Render Dashboard](https://dashboard.render.com/), select **New +** then **Blueprint** and connect that repository.
3. Confirm the `house-price-prediction-api` service and click **Apply**.
4. When the deployment succeeds, open `<your-service>.onrender.com/docs` for the interactive API documentation and `<your-service>.onrender.com/health` to confirm readiness.

To deploy a trained production model, replace `data/raw/sample_housing.csv` with your dataset (or update the build command and target column), then push the change. Do not store personal or confidential training data in the public repository.
