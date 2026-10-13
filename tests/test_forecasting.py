import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forecasting import chronological_split, evaluate, make_lag_features, load_series


class ForecastingTests(unittest.TestCase):
    def test_chronological_split_keeps_future_in_test(self):
        frame = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=50, freq="D"),
            "value": np.arange(50, dtype=float),
        })
        train, test = chronological_split(frame, test_fraction=0.2)
        self.assertLess(train["date"].max(), test["date"].min())
        self.assertEqual(len(train), 40)
        self.assertEqual(len(test), 10)

    def test_metrics_are_computed_correctly(self):
        scores = evaluate(np.array([1.0, 3.0]), np.array([2.0, 1.0]))
        self.assertAlmostEqual(scores["MAE"], 1.5)
        self.assertAlmostEqual(scores["RMSE"], np.sqrt(2.5))

    def test_lag_features_use_prior_values(self):
        x, y = make_lag_features(np.arange(20, dtype=float), lags=(1, 3))
        self.assertEqual(x.shape, (17, 2))
        self.assertEqual(y[0], 3.0)
        self.assertEqual(x[0].tolist(), [2.0, 0.0])

    def test_sample_data_loads_in_time_order(self):
        frame = load_series(PROJECT_ROOT / "data" / "sample_timeseries.csv")
        self.assertGreaterEqual(len(frame), 30)
        self.assertTrue(frame["date"].is_monotonic_increasing)


if __name__ == "__main__":
    unittest.main()
