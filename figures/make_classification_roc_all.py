"""Manuscript figure: four-outcome classification ROC panels.

Reproduces the released classification views (results/classification.csv)
by re-running the identical fold-local protocol — duplicate-aware grouped
five-fold cross-validation, train-fold column screening, HistGradientBoosting
classifier, seed 42 — then verifies every plotted number against the
released table and draws the ROC curves for the four views (one-vs-rest
per class for the multiclass panels; the EE ceiling view is binary):

  (a) drug loading bands   <10 / 10-25 / >25 percent
  (b) encapsulation efficiency ceiling  EE >= 95 percent
  (c) zeta potential sign   anionic / neutral / cationic
  (d) particle size bands   <100 / 100-200 / >200 nm
"""
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold

from modeling.evaluation import classification_target, prepare_target
from modeling.features import load_features

SEED = 42
DATA = ROOT / "data"
RESULTS = ROOT / "results"

PANELS = [
    ("drug_loading_percent", "(a) Drug loading bands"),
    ("encapsulation_efficiency_percent", "(b) Encapsulation efficiency ceiling"),
    ("zeta_potential_mV", "(c) Zeta potential sign"),
    ("particle_size_nm", "(d) Particle size bands"),
]
CLASS_LABELS = {
    "drug_loading_percent": ["low <10%", "moderate 10-25%", "high >25%"],
    "zeta_potential_mV": ["anionic (<-10 mV)", "neutral (-10 to 10 mV)", "cationic (>10 mV)"],
    "particle_size_nm": ["small <100 nm", "medium 100-200 nm", "large >200 nm"],
}
CLASS_COLORS = ["#1F4E9C", "#E8743B", "#4C9F70"]
BINARY_COLOR = "#7B4FA6"


def oof_probabilities(target):
    """Replicate the classification block of modeling/evaluation.py."""
    data = pd.read_csv(DATA / "corrected_dataset.csv").set_index("row_index")
    groups = pd.read_csv(DATA / "chemical_groups_dupaware.csv").set_index("row_index")
    features = load_features(DATA)
    td = prepare_target(data, groups, features, target)
    labels, _ = classification_target(target, td.observed)
    present = np.unique(labels)
    probability = np.full((len(labels), present.size), np.nan)
    for train, test in GroupKFold(5).split(td.features, groups=td.groups):
        columns = [
            column
            for column in td.features.columns
            if td.features.iloc[train][column].notna().any()
            and td.features.iloc[train][column].nunique(dropna=True) > 1
        ]
        model = HistGradientBoostingClassifier(
            random_state=SEED,
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=15,
            l2_regularization=5.0,
        )
        model.fit(td.features.iloc[train][columns], labels[train])
        probability[test] = model.predict_proba(td.features.iloc[test][columns])
    if not np.isfinite(probability).all():
        raise RuntimeError("Nonfinite class probabilities for " + target)
    return td, labels, probability


def verify_against_release(target, labels, probability):
    """Every plotted number must equal the released classification.csv."""
    stored = pd.read_csv(RESULTS / "classification.csv").set_index("target").loc[target]
    metrics = {}
    present = np.unique(labels)
    if present.size == 2:
        metrics["roc_auc"] = roc_auc_score(labels, probability[:, 1])
    else:
        metrics["balanced_accuracy"] = balanced_accuracy_score(
            labels, probability.argmax(1)
        )
        for index, cls in enumerate(present):
            metrics[f"auc_class_{int(cls)}"] = roc_auc_score(
                (labels == int(cls)).astype(int), probability[:, index]
            )
    for name, value in metrics.items():
        released = float(stored[name])
        if abs(round(float(value), 4) - released) > 1e-4:
            raise RuntimeError(
                f"{target}.{name}: recomputed {value:.6f} does not match "
                f"released classification.csv value {released}"
            )
    return metrics


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E6E8EB", "axes.axisbelow": True,
    "savefig.dpi": 300, "pdf.fonttype": 42,
})

stored = pd.read_csv(RESULTS / "classification.csv").set_index("target")
fig, axes = plt.subplots(2, 2, figsize=(6.2, 5.4))
CLASS_DASHES = ["solid", (0, (4, 2)), (0, (1, 1.5))]
for ax, (target, title) in zip(axes.ravel(), PANELS):
    td, labels, probability = oof_probabilities(target)
    metrics = verify_against_release(target, labels, probability)
    row = stored.loc[target]
    present = np.unique(labels)
    if present.size == 2:
        ax.plot(
            *roc_curve(labels, probability[:, 1])[:2],
            color=BINARY_COLOR, linewidth=1.6,
            label=f"EE >= 95% (AUC = {float(row.roc_auc):.2f})",
        )
        headline = f"n = {len(labels)}, ROC-AUC = {float(row.roc_auc):.2f}"
    else:
        for index, cls in enumerate(present):
            fpr, tpr, _ = roc_curve(
                (labels == int(cls)).astype(int), probability[:, index]
            )
            ax.plot(
                fpr, tpr, color=CLASS_COLORS[index], linewidth=1.6,
                linestyle=CLASS_DASHES[index],
                label=f"{CLASS_LABELS[target][index]} "
                      f"(AUC = {float(row[f'auc_class_{int(cls)}']):.2f})",
            )
        headline = (f"n = {len(labels)}, balanced accuracy = "
                    f"{float(row.balanced_accuracy):.2f}")
    ax.plot([0, 1], [0, 1], "--", color="#333940", linewidth=1.0, label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate", fontsize=8.5)
    ax.set_ylabel("True positive rate", fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.set_title(f"{title}\n{headline}", loc="left", fontsize=9)
    ax.legend(loc="lower right", fontsize=7.0, framealpha=0.8)
fig.suptitle(
    "ROC curves for the predeclared classification views\n"
    "(duplicate-aware grouped five-fold cross-validation)",
    x=0.02, ha="left", fontsize=10.5, fontweight="bold",
)
fig.tight_layout(rect=(0, 0, 1, 0.92))
for ext in ["png", "pdf"]:
    fig.savefig(HERE / f"classification_roc_all.{ext}")
plt.close(fig)
print("classification_roc_all.png/pdf written; all numbers verified "
      "against results/classification.csv")
