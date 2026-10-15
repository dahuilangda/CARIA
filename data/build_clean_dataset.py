"""Build the analysis cohort and its row-selection ledger."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_dataset(data_dir: Path) -> pd.DataFrame:
    full = pd.read_csv(data_dir / "corrected_dataset_full_1310.csv")
    pairs = pd.read_csv(data_dir / "near_duplicate_pairs_audit.csv")
    if full.row_index.duplicated().any():
        raise ValueError("Formulation identifiers must be unique")
    if not set(pairs.row_a).union(pairs.row_b).issubset(set(full.row_index)):
        raise ValueError("Duplicate-pair identifiers are absent from the corpus")
    ee = full["encapsulation_efficiency_percent"]
    dl = full["drug_loading_percent"]
    paired = ee.notna() & dl.notna()
    greater = paired & (dl > ee)
    equal = paired & np.isclose(dl, ee)
    duplicate = full.row_index.isin(pairs.row_b)
    # Label-basis consistency: with r = drug/polymer feed ratio, the
    # encapsulated drug per polymer mass is EE% x r / 100 under every
    # reported definition of DL (drug over polymer, or drug over particle).
    # A reported DL above that implied ceiling cannot coexist with the
    # reported EE on any denominator; 1.5x tolerance absorbs extraction
    # noise in EE and r. (The reverse, DL below the ceiling, is legitimate:
    # particles may contain non-polymer mass.)
    features = pd.read_csv(data_dir / "inputs/formulation_features.csv")
    ratio = features.set_index("row_index")["input_drug_polymer_ratio"]
    r = pd.to_numeric(ratio, errors="coerce").reindex(full.row_index)
    triple = paired & r.notna() & (r > 0)
    implied_polymer_drug = ee * r  # percent: encapsulated drug per polymer mass
    inconsistent = triple & (dl > 1.5 * implied_polymer_drug)
    # Physically impossible target values are masked (cell-level), keeping
    # the row for its other endpoints: EE and DL are percentages bounded by
    # 100; nanocarrier sizes at or above 34,700 nm are not nanoscale.
    over_100_ee = ee > 100
    over_100_dl = dl > 100
    size_full = full["particle_size_nm"]
    oversized = size_full >= 34_700
    full = full.copy()
    full.loc[over_100_ee, "encapsulation_efficiency_percent"] = np.nan
    full.loc[over_100_dl, "drug_loading_percent"] = np.nan
    full.loc[oversized, "particle_size_nm"] = np.nan
    selected = ~(greater | equal | duplicate | inconsistent)
    ledger = pd.DataFrame(
        {
            "row_index": full.row_index,
            "dl_greater_than_ee": greater,
            "ee_dl_close": equal,
            "duplicate_pair_second_member": duplicate,
            "ee_dl_ratio_inconsistent": inconsistent,
            "included": selected,
        }
    )
    columns = [
        "row_index",
        "drug_loading_percent",
        "encapsulation_efficiency_percent",
        "zeta_potential_mV",
        "particle_size_nm",
    ]
    clean = full.loc[selected, columns].copy()
    remaining = set(clean.row_index)
    if any(a in remaining and b in remaining for a, b in zip(pairs.row_a, pairs.row_b)):
        raise ValueError("An audited duplicate pair remains in the analysis cohort")
    clean.to_csv(data_dir / "corrected_dataset.csv", index=False)
    ledger.to_csv(data_dir / "row_selection.csv", index=False)
    return clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parent
    )
    args = parser.parse_args()
    clean = build_dataset(args.data_dir)
    print(f"Analysis cohort: {len(clean)} records")


if __name__ == "__main__":
    main()
