"""Check row-level provenance and complete grouped inner-OOF scoring."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


class ResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = Path(__file__).resolve().parents[1] / "results"
        cls.oof = pd.read_csv(directory / "oof_predictions.csv")
        cls.inner = pd.read_csv(directory / "inner_oof_predictions.csv")
        cls.scores = pd.read_csv(directory / "inner_scores.csv")
        cls.alloc = pd.read_csv(directory / "fold_allocations.csv")
        cls.metrics = pd.read_csv(directory / "metrics.csv")

    def test_inner_score_uses_every_training_row_once(self):
        for row in self.scores.itertuples(index=False):
            values = self.inner[
                (self.inner.target == row.target)
                & (self.inner.outer_fold == row.outer_fold)
                & (self.inner.config == row.config)
            ]
            self.assertEqual(len(values), row.n)
            self.assertFalse(values.row_index.duplicated().any())
            self.assertAlmostEqual(
                mean_squared_error(values.observed, values.prediction) ** 0.5,
                row.inner_oof_rmse,
                places=9,
            )

    def test_all_grouped_splits_are_disjoint(self):
        for _, records in self.alloc.groupby(["target", "outer_fold", "inner_fold"]):
            byrole = {role: set(part.group) for role, part in records.groupby("role")}
            roles = list(byrole)
            self.assertFalse(byrole[roles[0]] & byrole[roles[1]])

    def test_pooled_metrics_use_explicit_oof_rows(self):
        for row in self.metrics.itertuples(index=False):
            values = self.oof[
                (self.oof.target == row.target) & (self.oof.model == row.model)
            ]
            self.assertEqual(len(values), row.n)
            self.assertFalse(values.row_index.duplicated().any())
            self.assertAlmostEqual(
                mean_squared_error(values.observed, values.prediction) ** 0.5,
                row.rmse,
                places=9,
            )
            self.assertAlmostEqual(
                r2_score(values.observed, values.prediction), row.r2, places=9
            )

    def test_no_blend_model_remains(self):
        for target, records in self.oof.groupby("target"):
            self.assertNotIn("EqualBlend", set(records.model))


if __name__ == "__main__":
    unittest.main()
