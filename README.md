# Time Series Forecasting Baseline

A small, reproducible Python project that compares simple forecasting baselines on a chronological time-series split.

## What it demonstrates

- Time-aware train/test separation (no random shuffling).
- Naive and seasonal-naive forecasts.
- A lag-feature machine-learning baseline using `HistGradientBoostingRegressor`.
- MAE and RMSE evaluation.
- An actual-versus-predicted plot and a metrics CSV.
- Unit tests for split integrity and forecast behavior.

## Dataset

`data/sample_timeseries.csv` is a synthetic daily series created for demonstration. It is not real-world or NASA battery data. The target contains trend, weekly seasonality, and noise.

Columns:
- `date`: daily timestamp
- `value`: observed time-series value

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\\Scripts\\activate`.

## Run

From the repository root:

```bash
python src/forecasting.py
```

The script prints the chronological split and model metrics, then saves:
- `outputs/forecast_comparison.png`
- `outputs/metrics.csv`

The final test segment is held out chronologically. The machine-learning model uses only lag values available before the prediction time; this is a simple one-step rolling-origin evaluation where observed test values become available for later predictions.

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Limitations

This is a teaching baseline, not a production forecasting system. Results depend on the synthetic data-generating process. Real applications require domain-aware validation, robust missing-data handling, and comparison against appropriate seasonal and operational baselines.

## License

MIT. See `LICENSE`.
