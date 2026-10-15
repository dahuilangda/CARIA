"""Generate the quantile-regression and threshold-sensitivity evidence.

Reproduces, under duplicate-aware grouped five-fold CV (seed 42):
  - quantile random forest (q25/q50/q75) pinball losses vs the
    marginal-quantile baseline for encapsulation efficiency;
  - the Hartigan dip test on the EE distribution;
  - the ceiling classifier ROC-AUC at alternative thresholds
    (80 / 85 / 90 / 93 / 95 / 97).
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
from pathlib import Path
import numpy as np
import pandas as pd
import diptest
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from quantile_forest import RandomForestQuantileRegressor

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
SEED = 42

oof = pd.read_csv(HERE.parent / "results/oof_predictions.csv")
e = oof[(oof.target == "encapsulation_efficiency_percent") & (oof.model == "NestedSelected")].set_index("row_index")
fe = pd.read_csv(DATA / "inputs/formulation_features.csv").set_index("row_index")
grp = pd.read_csv(DATA / "chemical_groups_dupaware.csv").set_index("row_index")
X = fe.reindex(e.index).reset_index(drop=True)
y = e.observed.to_numpy(float)
g = grp.loc[e.index, "structure_dupaware_group"].fillna("NA").astype(str).to_numpy()
keep = [c for c in X.columns if X[c].notna().any() and X[c].nunique(dropna=True) > 1]
imputer = SimpleImputer(strategy="median").fit(X[keep])
Xi = pd.DataFrame(imputer.transform(X[keep]), columns=keep)
folds = list(GroupKFold(5).split(Xi, groups=g))

dip, dip_p = diptest.diptest(y)
print(f"Hartigan dip test: dip={dip:.4f}, p={dip_p:.3f}")

qs = [0.25, 0.5, 0.75]
pin = {q: [] for q in qs}
base = {q: [] for q in qs}
auc_rows = []
for train, test in folds:
    model = RandomForestQuantileRegressor(n_estimators=400, min_samples_leaf=10,
                                          random_state=SEED, n_jobs=8)
    model.fit(Xi.iloc[train], y[train])
    pred = np.atleast_2d(model.predict(Xi.iloc[test], quantiles=qs))
    for index, q in enumerate(qs):
        pin[q].append(np.mean(np.maximum(q * (y[test] - pred[:, index]),
                                         (q - 1) * (y[test] - pred[:, index]))))
    for threshold in [80, 85, 90, 93, 95, 97]:
        if len(set(y[test] >= threshold)) < 2:
            continue
        classifier = HistGradientBoostingClassifier(
            random_state=SEED, max_iter=200, learning_rate=0.05,
            max_leaf_nodes=15, min_samples_leaf=15, l2_regularization=5.0)
        classifier.fit(Xi.iloc[train][keep], (y[train] >= threshold).astype(int))
        proba = classifier.predict_proba(Xi.iloc[test][keep])[:, 1]
        auc_rows.append({"threshold": threshold,
                         "roc_auc": roc_auc_score((y[test] >= threshold).astype(int), proba)})
    for q in qs:
        base[q].append(np.mean(np.maximum(q * (y[test] - y[train].mean()),
                                          (q - 1) * (y[test] - y[train].mean()))))
print("\nQuantile pinball losses (pp):")
for q in qs:
    print(f"  q{int(q*100)}: QRF {np.mean(pin[q]):.2f} vs marginal baseline {np.mean(base[q]):.2f}")
sensitivity = pd.DataFrame(auc_rows).groupby("threshold").roc_auc.mean().round(4)
print("\nCeiling classifier ROC-AUC by threshold:")
print(sensitivity.to_string())
