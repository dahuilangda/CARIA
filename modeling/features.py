"""Load the released input-only snapshot using explicit formulation identifiers."""

from pathlib import Path

import numpy as np
import pandas as pd


def load_features(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    snapshot = pd.read_csv(data_dir / "inputs/formulation_features.csv").set_index(
        "row_index"
    )
    rows = pd.read_csv(data_dir / "corrected_dataset.csv").row_index
    if snapshot.index.duplicated().any() or rows.duplicated().any():
        raise ValueError("Formulation identifiers must be unique")
    if not set(rows).issubset(snapshot.index):
        raise ValueError("Missing feature rows")
    features = snapshot.loc[rows].apply(pd.to_numeric, errors="raise")
    if np.isinf(features.to_numpy(float)).any():
        raise ValueError("Nonfinite feature values")
    return features
