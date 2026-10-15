# CARIA

CARIA is an end-to-end study of whether the physicochemical properties of
polymeric and lipid nanocarriers can be predicted from literature-extracted
formulation data. An agentic LLM extraction harness (Qwen3-30B via an OpenAI-compatible
endpoint, with context management, goal tracking, error-repair rounds and
formulation-specific validation skills) processed the 1,310-record
literature corpus; after a leakage audit, unit corrections and duplicate
removal, 1,108 records with 90 input features were evaluated
under duplicate-aware grouped cross-validation.

Drug loading is predictable at moderate accuracy (TabPFN, R² 0.360;
three-band classification, balanced accuracy 0.64), with the drug:polymer
feed ratio as the dominant signal. Encapsulation efficiency is answered
by a high-EE classification view (ROC-AUC 0.63) plus quantile regression;
zeta potential and particle size are reported as band-level classifications
only — the honest scope of what this corpus supports. All numbers in this
repository are pooled out-of-fold estimates under predeclared, hash-anchored
protocols; every prediction row, fold allocation and paired test ships with
the release.

## Repository layout

- `data/` — the corrected 1,108-record dataset, its cell-level provenance,
  grouping keys, the fixed input snapshot, the audit ledgers, and the
  external PLGA corpus.
- `modeling/`, `results/` — the grouped evaluation pipeline and its frozen
  protocol, metrics and per-row out-of-fold predictions.
- `figures/` — publication figures regenerated from the released predictions.
- `extraction/` — the agentic LLM extraction harness (prompt, schema,
  vision-capable client) and the literature discovery pipeline.
- `analysis/` — ablations, special studies, and root-cause analyses.
- `docs/` — the full results ladder and data dictionary.

## Quick start

```bash
pip install -r requirements.txt
python data/build_clean_dataset.py
python data/evaluate_clean_dataset.py --device cpu --tabpfn-model-path /path/to/checkpoint --output-dir reproduced
python data/generate_derived_tables.py
python figures/make_classification_roc_all.py
python figures/make_dl_parity.py
python -m unittest discover -s tests
```

The TabPFN checkpoint (the Prior Labs release
`tabpfn-v3.5-20260909.safetensors`, sha256 recorded in `results/protocol.json`)
is downloaded automatically by the `tabpfn` package on first use; if your
network blocks the download, fetch it from the Prior Labs Hugging Face
repository and pass `--tabpfn-model-path`.
Use `--no-tabpfn` to evaluate the conventional models without checkpoint weights.
The released results include row identifiers, inner/outer allocations, candidate
scores and out-of-fold predictions; figures are generated from these predictions.
Deterministic reproduction requires the pinned environment in `results/protocol.json`;
small floating-point drift (up to ~1e-2 in R²) is expected across thread counts
and devices, while fold allocations are exact.

## Headline results (pooled outer OOF, duplicate-aware grouped five-fold folds)

Every outcome is answered by a predeclared classification view with direct
pharmaceutical meaning — the four ROC panels of the manuscript (table:
`results/classification.csv`, figure script:
`figures/make_classification_roc_all.py`):

| Outcome | Classification view | n | Headline | One-vs-rest AUC |
|---|---|---:|---|---|
| Drug loading | band <10 / 10–25 / >25 % | 859 | balanced accuracy 0.643 | 0.863 / 0.799 / 0.869 |
| Encapsulation efficiency | ceiling EE >= 95% | 803 | ROC-AUC 0.627 | — |
| Zeta potential | sign anionic / neutral / cationic | 572 | balanced accuracy 0.388 | 0.578 / 0.624 / 0.587 |
| Particle size | band <100 / 100–200 / >200 nm | 1035 | balanced accuracy 0.378 | 0.641 / 0.597 / 0.553 |

For the quantitative view, a separately specified **TabPFN** model (a
pre-trained tabular foundation model, Nature 2025) reaches drug-loading
R² 0.360 with MAE 7.73 percentage points (training-mean baseline: 11.92 pp)
under the same folds. For encapsulation efficiency the release additionally
ships **conditional quantile regression** (q25/q50/q75 all beat the
quantile baselines on grouped folds; q75 pinball loss 4.58 vs 7.93 pp;
evidence script `figures/quantile_and_threshold_evidence.py`). The ceiling
classifier answers the go/no-go question "does this formulation reach
EE >= 95%?" — 95% being the predeclared top of the commercial specification
band (>=90-95%), not a discovered distributional mode. The complete
regression ladder with all baselines and diagnostics is tabulated in
`docs/RESULTS.md` and `results/` (Supporting Table S1 of the manuscript).

The analysis cohort applies explicitly recorded screening and duplicate rules.
These are internal flags, not source-paper validation. Grouping uses cached
chemical identities and duplicate proxies; publication independence is unverified.
Predictions evaluate formulation properties and do not validate carrier ranking.

## Fresh literature extraction

The agentic harness in `extraction/` runs against any OpenAI-compatible,
vision-capable endpoint. Copy `.env.example` to `.env`, fill in
`EXTRACTION_BASE_URL` / `EXTRACTION_API_KEY` / `EXTRACTION_MODEL`
(`TAVILY_API_KEY` enables the PubMed + Tavily discovery channel), then:

```bash
pip install -r requirements-extraction.txt
python extraction/literature.py "your query" --cache-dir fulltext_xml
python extraction/markdown_to_json.py --input-dir fulltext_xml --output-dir records
```
