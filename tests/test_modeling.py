"""Regression checks for fold-local preprocessing and explicit input identity."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modeling.evaluation import clip_predictions, estimator, train_predict
from modeling.features import load_features


class ModelingTests(unittest.TestCase):
    def test_snapshot_uses_formulation_ids(self):
        data = Path(__file__).resolve().parents[1] / "data"
        features = load_features(data)
        rows = pd.read_csv(data / "corrected_dataset.csv").row_index
        self.assertEqual(features.index.tolist(), rows.tolist())
        self.assertNotIn("ratio_imputed", features)
        for outcome in [
            "drug_loading_percent",
            "encapsulation_efficiency_percent",
            "particle_size_nm",
            "zeta_potential_mV",
        ]:
            self.assertNotIn(outcome, features)

    def test_column_selection_does_not_use_test_rows(self):
        X = pd.DataFrame(
            {
                "observed": [1.0, 2.0, 3.0, 4.0, 5.0],
                "test_only_variable": [np.nan, np.nan, np.nan, 10.0, 20.0],
            }
        )
        _, columns = train_predict(
            "ridge",
            X,
            np.array([1.0, 2.0, 3.0, 40.0, 50.0]),
            np.array([0, 1, 2]),
            np.array([3, 4]),
            "zeta_potential_mV",
            42,
        )
        self.assertEqual(columns, ["observed"])

    def test_missing_median_and_scale_fit_training_only(self):
        model = estimator("ridge", 42)
        model.fit(pd.DataFrame({"x": [1.0, 3.0, np.nan]}), [1.0, 3.0, 2.0])
        self.assertEqual(model.named_steps["impute"].statistics_[0], 2.0)
        self.assertEqual(model.named_steps["scale"].mean_[0], 2.0)

    def test_units_apply_per_endpoint(self):
        np.testing.assert_array_equal(
            clip_predictions([-1.0, 150.0], "drug_loading_percent"), [0.0, 100.0]
        )
        np.testing.assert_array_equal(
            clip_predictions([-200.0, 150.0], "zeta_potential_mV"), [-200.0, 150.0]
        )
        np.testing.assert_array_equal(
            clip_predictions([-1.0, 150.0], "particle_size_nm"), [0.0, 150.0]
        )


if __name__ == "__main__":
    unittest.main()
