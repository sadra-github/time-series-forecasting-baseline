"""Time-series forecasting baselines with a chronological holdout split."""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "sample_timeseries.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def load_series(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate a date/value CSV, sorted in chronological order."""
    frame = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "value"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = frame.loc[:, ["date", "value"]].copy()
    frame["value"] = pd.to_numeric(frame["value"], errors="raise")
    if frame["date"].isna().any() or frame["value"].isna().any():
        raise ValueError("The series contains missing dates or values.")
    if frame["date"].duplicated().any():
        raise ValueError("Dates must be unique.")
    frame = frame.sort_values("date").reset_index(drop=True)
    if len(frame) < 30:
        raise ValueError("At least 30 observations are required.")
    return frame


def chronological_split(frame: pd.DataFrame, test_fraction: float = 0.2):
    """Return train/test frames without shuffling; the test follows the train."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    split_idx = int(len(frame) * (1 - test_fraction))
    if split_idx < 14 or len(frame) - split_idx < 1:
        raise ValueError("The selected split leaves too few training observations.")
    return frame.iloc[:split_idx].copy(), frame.iloc[split_idx:].copy()


def make_lag_features(values: np.ndarray, lags=(1, 7, 14)) -> tuple[np.ndarray, np.ndarray]:
    """Build one-step features from historical observations only."""
    values = np.asarray(values, dtype=float)
    max_lag = max(lags)
    if len(values) <= max_lag:
        raise ValueError("Not enough observations to construct lag features.")
    x, y = [], []
    for i in range(max_lag, len(values)):
        x.append([values[i - lag] for lag in lags])
        y.append(values[i])
    return np.asarray(x), np.asarray(y)


def forecast_lag_model(train_values: np.ndarray, test_values: np.ndarray) -> np.ndarray:
    """Fit a lag model on train data, then predict each test step using available history."""
    train_values = np.asarray(train_values, dtype=float)
    test_values = np.asarray(test_values, dtype=float)
    lags = (1, 7, 14)
    x_train, y_train = make_lag_features(train_values, lags=lags)
    model = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.06, max_leaf_nodes=15, random_state=42)
    model.fit(x_train, y_train)

    history = list(train_values)
    predictions = []
    for observed in test_values:
        features = np.asarray([[history[-lag] for lag in lags]], dtype=float)
        prediction = float(model.predict(features)[0])
        predictions.append(prediction)
        # Rolling-origin one-step evaluation: actual observation becomes available
        # before forecasting the next timestamp.
        history.append(float(observed))
    return np.asarray(predictions)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate MAE and RMSE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("Actual and predicted arrays must have the same non-empty shape.")
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def run(data_path: str | Path = DATA_PATH, output_dir: str | Path = OUTPUT_DIR) -> pd.DataFrame:
    """Run baselines, save metrics and a forecast comparison plot."""
    frame = load_series(data_path)
    train, test = chronological_split(frame, test_fraction=0.2)
    train_values = train["value"].to_numpy(dtype=float)
    test_values = test["value"].to_numpy(dtype=float)

    predictions = {
        "Naive (last observed)": np.r_[train_values[-1], test_values[:-1]],
        "Seasonal naive (lag 7)": np.asarray([
            (train_values.tolist() + test_values[:i].tolist())[-7]
            for i in range(len(test_values))
        ], dtype=float),
        "HistGradientBoosting (lags 1, 7, 14)": forecast_lag_model(train_values, test_values),
    }

    rows = []
    for name, pred in predictions.items():
        scores = evaluate(test_values, pred)
        rows.append({"model": name, **scores})
    metrics = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path / "metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(test["date"], test_values, label="Actual", linewidth=2)
    for name, pred in predictions.items():
        ax.plot(test["date"], pred, label=name, alpha=0.85)
    ax.set_title("Time Series Forecasting Baselines (Chronological Test Set)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path / "forecast_comparison.png", dpi=160)
    # Intentionally leave the figure available for interactive inspection.

    print(f"Observations: {len(frame)}")
    print(f"Training observations: {len(train)} ({train['date'].min().date()} to {train['date'].max().date()})")
    print(f"Test observations: {len(test)} ({test['date'].min().date()} to {test['date'].max().date()})")
    print("\nTest metrics (lower is better):")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSaved: {output_path / 'metrics.csv'}")
    print(f"Saved: {output_path / 'forecast_comparison.png'}")
    return metrics


if __name__ == "__main__":
    run()
