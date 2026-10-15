# Data definitions

| File | Contents |
|---|---|
| `data/corrected_dataset_full_1310.csv` | Full analysis snapshot before row screening: `row_index`, `split_group` (source-file identifier; not a verified PMID/DOI), the four target columns and `formulation_composition_excipients_concentration_percent_w_v__std` |
| `data/corrected_dataset.csv` | Selected formulation cohort |
| `data/row_selection.csv` | Row-level flags and inclusion decisions |
| `data/material_mentions.csv` | Record-level evidence for manuscript Table 1 |
| `data/cell_provenance.csv` | 30 declared-unit changes and 18 analysis-domain masks; source verification pending |
| `data/inputs/formulation_features.csv` | Fixed numeric, input-only feature snapshot keyed by `row_index` |
| `data/inputs/feature_sources.csv` | Feature-block definitions |
| `data/chemical_groups_dupaware.csv` | Cached chemical and duplicate-aware group keys |
| `data/near_duplicate_pairs_audit.csv` | 134 internally matched pairs; not verified publication duplicates |

`row_index` is the stable formulation identifier. EE and DL are percentages;
MAE/RMSE for these targets are percentage points. Size is nm and zeta potential mV.
EE/DL evaluation requires finite values in [0, 100]; size requires positive finite
values; zeta requires finite values. Missing values remain missing until fold-local
preprocessing. No outcome, quality score or source identifier is a predictor.

The fixed 90-feature snapshot contains reported and component-derived feed inputs,
features derived from 19 conservatively tagged process quantities (16 value
columns and 19 missingness flags retained), molecular
descriptors and carrier-family indicators. ESOL is a calculated solubility descriptor,
not a measured value: 0.16 − 0.63 logP − 0.0062 MW + 0.066 rotatable bonds −
0.74 aromatic heavy-atom fraction ([Delaney, 2004](https://doi.org/10.1021/ci034243x)).
There is no learned cross-corpus feed-ratio filling.

Screening excludes DL > EE, numerically close EE/DL under NumPy's default
`isclose`, and the second member of each listed duplicate pair. The first two
flags are heuristic consistency screens, not universal physical constraints or
proof of extraction error. These outcome-based screens limit interpretation to
the selected cohort. Both endpoints remain available in the full snapshot.

Duplicate-aware grouping combines cached exact chemical identities with the
listed pair links. Unavailable chemical keys are stored empty in
`chemical_groups_dupaware.csv` and share one group, labelled `NA_GROUP` at
grouping time (`modeling/evaluation.py`); they do not establish new-chemical
or publication-level independence. Source-file identifiers
and legacy deduplication keys are not verified PMIDs/DOIs.


## Extraction provenance files
| File | Meaning |
|---|---|
| `drug_vocab_enriched.csv` | the 400-entry normalized parent-name vocabulary with PubChem-resolved synonyms and descriptors (the "400-entry vocabulary" in the manuscript) |
| `extraction_metadata_audit.json` | declared extraction metadata: all 1,310 rows declare `Qwen/Qwen3-30B-A3B-Thinking-2507-FP8`; 12 truncated; 836 below the review threshold of 50 (rule-based score, never an exclusion) |
| `cell_provenance.csv` / `row_selection.csv` | cell-level correction ledger and per-row selection flags behind the 1,108-record cohort |


## Cohort selection rules (all four recorded in row_selection.csv)

1. `dl_greater_than_ee` — DL exceeds EE (8 rows)
2. `ee_dl_close` — EE and DL numerically equal to two decimals (41 rows)
3. `duplicate_pair_second_member` — second member of an audited
   near-duplicate pair (90 rows)
4. `ee_dl_ratio_inconsistent` — reported DL exceeds 1.5× the
   encapsulated-mass ceiling implied by reported EE and feed ratio (63 rows)

Additionally, physically impossible target cells are masked to missing
(EE > 100%, DL > 100%, size ≥ 34,700 nm; 18 declared cells in
`data/cell_provenance.csv` — 8 DL, 2 EE and 8 size — plus 11 further EE
values in (100, 106.8] masked by the predeclared rule in
`build_clean_dataset.py`; the two sets do not overlap: 29 masked target
cells in total). The union of all removal
rules leaves 1,108 records; the 1,171-record intermediate cohort (before
the label-basis rule) is also available in the audit archive.

## Feature snapshot columns

`inputs/formulation_features.csv` (91 columns = 90 features + row_index)
contains: feed-ratio and nominal-feed quantities (with missingness
indicators), cached molecular-descriptor summaries (mean/min/max of
MolLogP, MW, TPSA, HBD, HBA, rotatable bonds, aromatic proportion,
FractionCSP3, formal charge, ESOL-style log S), conservatively parsed preparation numeric
values, family/type indicators, and associated missingness flags. Column
names are snake_case with RDKit descriptor names and standard unit symbols
preserved (`drugext_MolLogP_mean`, `process_numeric__stirring_temperature_C`,
`..._sonication_power_W`, `..._dialysis_MWCO_Da`, `..._pH`); eleven columns
derived from carrier-family labels contain the family name as suffix
(e.g. `carrier_family__polyester`). Known cosmetic mismatch: the `feature_sources.csv` row for the cellulose-derivative indicator writes `carrier_family__cellulose derivative` (space) while the actual column is `carrier_family__cellulose_derivative`; the sources file is kept byte-identical to the hash pinned in `results/protocol.json` for the completed evaluation run.
