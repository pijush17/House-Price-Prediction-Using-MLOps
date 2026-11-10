from pathlib import Path

from src.train import train


def test_train_writes_usable_artifact(tmp_path: Path):
    metrics = train("data/raw/sample_housing.csv", output=tmp_path / "model.joblib")
    assert (tmp_path / "model.joblib").exists()
    assert set(metrics) == {"mae", "mse", "rmse", "r2"}
