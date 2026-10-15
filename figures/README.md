# Figures

## classification_roc_all (png / pdf)

Main manuscript figure: ROC curves for the four predeclared classification
views — (a) drug loading bands <10 / 10–25 / >25 %, (b) encapsulation
efficiency ceiling EE >= 95 % (binary), (c) zeta potential sign
anionic / neutral / cationic, (d) particle size bands <100 / 100–200 /
>200 nm. Multiclass panels plot one-vs-rest curves per class.
Duplicate-aware grouped five-fold cross-validation; every plotted
metric is verified against `results/classification.csv` at generation time.

## dl_parity (png / pdf)

Drug-loading regression parity: pooled outer out-of-fold predictions under
duplicate-aware grouped five-fold cross-validation (n = 859).
Left panel: random forest (nested-CV pool), R² = 0.335.
Right panel: TabPFN (final model), R² = 0.360.
Dashed line: identity. Shaded band: ±20% relative error.

## Scripts

- `make_classification_roc_all.py` — regenerates classification_roc_all by
  re-running the released classification protocol (seed 42, train-fold
  column screening) and verifying every metric against
  `results/classification.csv` before plotting.
- `make_dl_parity.py` — generates dl_parity from released OOF predictions
  (`results/oof_predictions.csv`).
- `quantile_and_threshold_evidence.py` — quantile-regression and
  EE-threshold sensitivity evidence (see `docs/RESULTS.md`).
