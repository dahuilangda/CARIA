# Data

All values in this folder are released under CC-BY 4.0 (data); the code is
MIT-licensed. Verify file integrity with `SHA256SUMS`.

## Primary dataset

**`corrected_dataset.csv`** (1,108 records × 5 columns) — the analysis
cohort. Columns: `row_index` (stable identifier), `drug_loading_percent`,
`encapsulation_efficiency_percent`, `zeta_potential_mV`,
`particle_size_nm`. Derived from the 1,310-record extraction snapshot by
four predeclared removal rules (see `build_clean_dataset.py` and
`row_selection.csv`); every value changed relative to the raw extraction
has cell-level provenance in `cell_provenance.csv`.

## Supporting files

| File | Description |
|---|---|
| `corrected_dataset_full_1310.csv` | full pre-clean snapshot (all 1,310 records) for auditability |
| `row_selection.csv` | per-row selection ledger (all four removal rules flagged) |
| `near_duplicate_pairs_audit.csv` | 134 audited cross-file near-duplicate pairs |
| `cell_provenance.csv` | 48 cell-level provenance entries (30 declared-unit conversions and 18 analysis-domain masks) |
| `chemical_groups_dupaware.csv` | fused chemical + duplicate-aware grouping keys |
| `drug_vocab_enriched.csv` | 400-entry normalized parent-name vocabulary |
| `extraction_metadata_audit.json` | declared extraction metadata audit |
| `material_mentions.csv` | record-level evidence for manuscript Table 1 |
| `inputs/formulation_features.csv` | fixed input feature snapshot (90 features + `row_index`) |
| `inputs/feature_sources.csv` | provenance of each feature block |
| `SHA256SUMS` | checksums for the files above |

## Notes

- 24 cohort rows have all four target values missing; they are retained
  because their non-target endpoints are valid for other analyses.
- EE/DL are percentages in [0, 100]; size in nm; zeta in mV. Values above
  100 (EE/DL) and at or above 34,700 nm (size) are physically impossible
  and are masked to missing.
