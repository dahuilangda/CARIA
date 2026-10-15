"""Manuscript Fig. 2: drug-loading parity, RF nested-CV pool vs TabPFN."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

HERE = Path(__file__).resolve().parent
oof = pd.read_csv(HERE.parent / "results/oof_predictions.csv")
dl = oof[oof.target == "drug_loading_percent"]
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E6E8EB", "axes.axisbelow": True,
    "savefig.dpi": 300, "pdf.fonttype": 42,
})
fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.1))
panels = [
    (axes[0], "NestedSelected", "#1F4E9C", "Random forest (nested-CV pool)"),
    (axes[1], "TabPFN", "#E8743B", "TabPFN (final model)"),
]
for ax, model, color, title in panels:
    sub = dl[dl.model == model]
    y = sub.observed.to_numpy(float)
    p = sub.prediction.to_numpy(float)
    r2 = r2_score(y, p)
    rmse = np.sqrt(np.mean((y - p) ** 2))
    mae = np.abs(y - p).mean()
    band = np.linspace(0, 100, 100)
    ax.fill_between(band, 0.8 * band, 1.2 * band, color=color, alpha=0.08, linewidth=0,
                    label="±20% relative error")
    ax.plot([0, 100], [0, 100], "--", color="#333940", linewidth=1.0, label="Identity")
    ax.scatter(y, p, s=13, c=color, alpha=0.3, linewidths=0)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel("Observed drug loading (%)", fontsize=8.5)
    ax.set_ylabel("Predicted drug loading (%)", fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.set_title(f"{title}\nR² = {r2:.3f}   RMSE = {rmse:.2f} pp   MAE = {mae:.2f} pp",
                 loc="left", fontsize=9.5)
    ax.legend(loc="lower right", fontsize=7.0, framealpha=0.8)
fig.suptitle("Drug loading: pooled outer out-of-fold predictions\n"
             "duplicate-aware grouped five-fold cross-validation (n = 859)",
             x=0.02, ha="left", fontsize=11, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.93))
for ext in ["png", "pdf"]:
    fig.savefig(HERE / f"dl_parity.{ext}")
plt.close(fig)
print("dl_parity.png/pdf written")
