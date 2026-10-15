"""Regenerate results/ceiling_classifier.csv and
results/calibration_diagnostics.csv from the released out-of-fold
predictions (results/oof_predictions.csv).

These two tables are derived views of the released OOF predictions: the
ceiling classifier table stores the pooled classification metrics for the
EE >= 95% question; the calibration table stores the empirical error and
calibration diagnostics per target and model.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statistics
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"

oof = pd.read_csv(RESULTS / "oof_predictions.csv")

# ---- ceiling classifier table (EE >= 95%) ----
rows = []
ee = oof[oof.target == "encapsulation_efficiency_percent"]
for model, sub in ee.groupby("model"):
    y = (sub.observed.to_numpy(float) >= 95).astype(int)
    proba = sub.prediction.to_numpy(float)
    rows.append({
        "target": "encapsulation_efficiency_percent",
        "model": model,
        "n": len(sub),
        "base_rate": round(float(y.mean()), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "average_precision": round(float(average_precision_score(y, proba)), 4),
    })
pd.DataFrame(rows).to_csv(RESULTS / "ceiling_classifier.csv", index=False)

# ---- calibration diagnostics ----
rows = []
for (target, model), sub in oof.groupby(["target", "model"]):
    y = sub.observed.to_numpy(float)
    p = sub.prediction.to_numpy(float)
    mean_y = y.mean()
    mean_p = p.mean()
    sse = float(((y - p) ** 2).sum())
    sst = float(((y - mean_y) ** 2).sum())
    errors = np.sort(np.abs(y - p))
    pos = (len(errors) - 1) * 0.9
    low = int(pos)
    p90 = errors[low] + (pos - low) * (errors[min(low + 1, len(errors) - 1)] - errors[low])
    cov = float(((y - mean_y) * (p - mean_p)).sum())
    pv = float(((p - mean_p) ** 2).sum())
    rows.append({
        "target": target,
        "model": model,
        "n": len(y),
        "p90_absolute_error": p90,
        "observed_mean": mean_y,
        "observed_sample_sd": statistics.stdev(y),
        "predicted_on_observed_slope": cov / sst,
        "predicted_on_observed_intercept": mean_p - cov / sst * mean_y,
        "observed_on_predicted_slope": cov / pv,
        "observed_on_predicted_intercept": mean_y - cov / pv * mean_p,
        "mae_over_observed_sd": np.abs(y - p).mean() / y.std(ddof=1),
        "rmse_over_mae": np.sqrt(((y - p) ** 2).mean()) / np.abs(y - p).mean(),
    })
pd.DataFrame(rows).to_csv(RESULTS / "calibration_diagnostics.csv", index=False)
print("regenerated ceiling_classifier.csv and calibration_diagnostics.csv")
