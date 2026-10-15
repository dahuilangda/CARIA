"""Group-separated nested evaluation of input-only formulation features."""

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
from importlib.metadata import version

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modeling.features import load_features

CONFIGS = [
    "rf_depth8",
    "rf_leaf15",
    "extra_trees",
    "hist_gradient_boosting",
    "ridge",
    "mean",
]
TARGETS = [
    "drug_loading_percent",
    "encapsulation_efficiency_percent",
    "zeta_potential_mV",
    "particle_size_nm",
]


def estimator(config: str, seed: int) -> Pipeline:
    common = {"random_state": seed}
    if config == "rf_depth8":
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=1,
            max_features=1.0,
            n_jobs=8,
            **common,
        )
    elif config == "rf_leaf15":
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=15,
            max_features=0.5,
            n_jobs=8,
            **common,
        )
    elif config == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=1,
            max_features=1.0,
            n_jobs=8,
            **common,
        )
    elif config == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.05,
            early_stopping=False,
            max_leaf_nodes=7,
            min_samples_leaf=30,
            l2_regularization=20.0,
            **common,
        )
    elif config == "ridge":
        model = Ridge(alpha=1000.0)
    elif config == "mean":
        model = DummyRegressor(strategy="mean")
    else:
        raise ValueError(config)
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if config == "ridge":
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps + [("model", model)])




CEILING_THRESHOLD = 95.0
SIZE_BANDS = (100.0, 200.0)
ZETA_NEUTRAL_BAND = 10.0


def classification_target(target: str, observed: np.ndarray):
    """Return (class labels, task description) for the meaningful
    classification view of a target, or None when the target has no
    established classification in nanomedicine."""
    if target == "encapsulation_efficiency_percent":
        return (observed >= CEILING_THRESHOLD).astype(int), "ceiling EE >= 95%"
    if target == "drug_loading_percent":
        labels = np.select(
            [observed < 10.0, observed <= 25.0],
            [0, 1],
            default=2,
        )
        return labels, "DL band <10 / 10-25 / >25 percent"
    if target == "particle_size_nm":
        labels = np.select(
            [observed < SIZE_BANDS[0], observed <= SIZE_BANDS[1]],
            [0, 1],
            default=2,
        )
        return labels, "size band <100 / 100-200 / >200 nm"
    if target == "zeta_potential_mV":
        labels = np.select(
            [observed < -ZETA_NEUTRAL_BAND, observed > ZETA_NEUTRAL_BAND],
            [0, 2],
            default=1,
        )
        return labels, "zeta sign anionic / neutral / cationic"
    return None


def clip_predictions(prediction: np.ndarray | list[float], target: str) -> np.ndarray:
    p = np.asarray(prediction, float)
    if target in TARGETS[:2]:
        p = np.clip(p, 0.0, 100.0)
    elif target == "particle_size_nm":
        p = np.maximum(p, 0.0)
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite predictions")
    return p


def train_predict(
    config: str,
    X: pd.DataFrame,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    target: str,
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    # Selection of nonconstant columns uses this training fold only.
    keep = X.iloc[train].nunique(dropna=True).gt(1)
    columns = keep.index[keep].tolist()
    if not columns:
        columns = [X.columns[0]]
    model = estimator(config, seed)
    model.fit(X.iloc[train][columns], y[train])
    return clip_predictions(model.predict(X.iloc[test][columns]), target), columns


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y, p)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "mae": float(mean_absolute_error(y, p)),
    }


@dataclass(frozen=True)
class TargetData:
    target: str
    row_ids: np.ndarray
    observed: np.ndarray
    features: pd.DataFrame
    groups: np.ndarray


def prepare_target(
    data: pd.DataFrame, groups: pd.DataFrame, features: pd.DataFrame, target: str
) -> TargetData:
    if target not in TARGETS:
        raise ValueError(f"Unsupported endpoint: {target}")
    valid = np.isfinite(data[target])
    if target in TARGETS[:2]:
        valid &= data[target].between(0, 100)
    elif target == "particle_size_nm":
        valid &= data[target] > 0
    rows = data.index[valid].to_numpy()
    return TargetData(
        target=target,
        row_ids=rows,
        observed=data.loc[rows, target].to_numpy(float),
        features=features.loc[rows].reset_index(drop=True),
        groups=groups.reindex(rows)
        .structure_dupaware_group.fillna("NA_GROUP")
        .astype(str)
        .to_numpy(),
    )


def build_protocol(data_dir: Path, seed: int, tabpfn: bool, checkpoint: Path) -> dict:
    inputs = [
        data_dir / "corrected_dataset.csv",
        data_dir / "chemical_groups_dupaware.csv",
        data_dir / "inputs/formulation_features.csv",
        data_dir / "inputs/feature_sources.csv",
    ]
    protocol = {
        "tabpfn_targets": TARGETS[:2] if tabpfn else [],
        "seed": seed,
        "outer_folds": 5,
        "inner_folds": 3,
        "selection": "Minimum RMSE across complete training-only inner OOF predictions; candidate tie uses declared order.",
        "configurations": CONFIGS,
        "prediction_transform": "Original units; percent clip 0..100, size lower bound0. Identical clipping for inner/outer scoring.",
                "features": "Reported formulation inputs and fixed molecular descriptors; no cross-corpus ratio fill. All feature selection, median fill and Ridge scaling fit within training folds.",
        "missing_structure_groups": "Unavailable structures share NA_GROUP; this is not verified publication grouping.",
        "inputs_sha256": {
            str(path.relative_to(data_dir)): file_hash(path) for path in inputs
        },
        "software": {
            name: version(name)
            for name in ["numpy", "pandas", "scipy", "scikit-learn", "tabpfn"]
        },
        "status": "Internal development diagnostic; source identities and external prospective accuracy are not established.",
        "implementation_sha256": {
            path.name: file_hash(path)
            for path in [Path(__file__), Path(__file__).with_name("features.py")]
        },
        "estimators": {config: str(estimator(config, seed)) for config in CONFIGS},
    }
    if tabpfn:
        protocol["tabpfn_checkpoint"] = {
            "filename": checkpoint.name,
            "sha256": file_hash(checkpoint),
        }
    return protocol


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_finite(predictions: np.ndarray) -> None:
    if not np.isfinite(predictions).all():
        raise ValueError("Evaluation predictions are incomplete or nonfinite")


def require_separated_groups(
    groups: np.ndarray, train: np.ndarray, test: np.ndarray
) -> None:
    if set(groups[train]) & set(groups[test]):
        raise ValueError("Training and evaluation groups overlap")


def record_allocation(
    records: list[dict],
    data: TargetData,
    outer: int,
    inner: int,
    role: str,
    indices: np.ndarray,
) -> None:
    records.extend(
        {
            "target": data.target,
            "outer_fold": outer,
            "inner_fold": inner,
            "role": role,
            "row_index": int(data.row_ids[index]),
            "group": data.groups[index],
        }
        for index in indices
    )


def record_columns(
    records: list[dict],
    target: str,
    outer: int,
    inner: int,
    config: str,
    columns: list[str],
) -> None:
    records.append(
        {
            "target": target,
            "outer_fold": outer,
            "inner_fold": inner,
            "config": config,
            "columns": json.dumps(columns),
        }
    )


def select_configuration(
    data: TargetData,
    train: np.ndarray,
    outer: int,
    seed: int,
    tables: dict[str, list[dict]],
) -> tuple[str, float]:
    splits = list(GroupKFold(3).split(train, groups=data.groups[train]))
    scores = []
    for config in CONFIGS:
        predictions = np.full(len(train), np.nan)
        for inner, (inner_train, validation) in enumerate(splits):
            fit_rows, valid_rows = train[inner_train], train[validation]
            require_separated_groups(data.groups, fit_rows, valid_rows)
            predictions[validation], columns = train_predict(
                config,
                data.features,
                data.observed,
                fit_rows,
                valid_rows,
                data.target,
                seed,
            )
            record_columns(
                tables["selected_feature_columns"],
                data.target,
                outer,
                inner,
                config,
                columns,
            )
            tables["inner_oof_predictions"].extend(
                {
                    "target": data.target,
                    "outer_fold": outer,
                    "inner_fold": inner,
                    "config": config,
                    "row_index": int(data.row_ids[index]),
                    "observed": float(data.observed[index]),
                    "prediction": float(value),
                }
                for index, value in zip(valid_rows, predictions[validation])
            )
            if config == CONFIGS[0]:
                record_allocation(
                    tables["fold_allocations"], data, outer, inner, "train", fit_rows
                )
                record_allocation(
                    tables["fold_allocations"],
                    data,
                    outer,
                    inner,
                    "validation",
                    valid_rows,
                )
        require_finite(predictions)
        score = metrics(data.observed[train], predictions)["rmse"]
        scores.append(score)
        tables["inner_scores"].append(
            {
                "target": data.target,
                "outer_fold": outer,
                "config": config,
                "inner_oof_rmse": score,
                "n": len(train),
            }
        )
    return CONFIGS[int(np.argmin(scores))], min(scores)


def predict_classical_models(
    data: TargetData,
    train: np.ndarray,
    test: np.ndarray,
    selected: str,
    outer: int,
    seed: int,
    tables: dict[str, list[dict]],
) -> dict[str, np.ndarray]:
    predictions = {}
    for name, config in [
        ("NestedSelected", selected),
        ("Mean", "mean"),
        ("Ridge", "ridge"),
    ]:
        predictions[name], columns = train_predict(
            config, data.features, data.observed, train, test, data.target, seed
        )
        record_columns(
            tables["selected_feature_columns"], data.target, outer, -1, name, columns
        )
    return predictions


def predict_tabpfn(
    data: TargetData,
    train: np.ndarray,
    test: np.ndarray,
    checkpoint: Path,
    seed: int,
    device: str,
) -> tuple[np.ndarray, list[str], bool]:
    from tabpfn import TabPFNRegressor

    keep = data.features.iloc[train].nunique(dropna=True).gt(1)
    columns = keep.index[keep].tolist()
    model = TabPFNRegressor(
        ignore_pretraining_limits=True,
        random_state=seed,
        n_jobs=8,
        device=device,
        model_path=checkpoint,
    )
    model.fit(data.features.iloc[train][columns], data.observed[train])
    predictions = clip_predictions(
        model.predict(data.features.iloc[test][columns]), data.target
    )
    resolved = any(
        hasattr(model, attribute) for attribute in ["model_paths_", "model_path_"]
    )
    return predictions, columns, resolved


def record_outer_predictions(
    data: TargetData,
    outer: int,
    test: np.ndarray,
    predictions: dict[str, np.ndarray],
    tables: dict[str, list[dict]],
) -> None:
    for name, prediction in predictions.items():
        values = prediction
        tables["fold_metrics"].append(
            {
                "target": data.target,
                "outer_fold": outer,
                "model": name,
                "n": len(test),
                **metrics(data.observed[test], values),
            }
        )
        tables["oof_predictions"].extend(
            {
                "target": data.target,
                "outer_fold": outer,
                "model": name,
                "row_index": int(data.row_ids[index]),
                "group": data.groups[index],
                "observed": float(data.observed[index]),
                "prediction": float(value),
            }
            for index, value in zip(test, values)
        )

def evaluate_target(
    data: TargetData,
    tabpfn: bool,
    seed: int,
    device: str,
    checkpoint: Path,
    protocol: dict,
    tables: dict[str, list[dict]],
) -> None:
    predictions = {
        name: np.full(len(data.observed), np.nan)
        for name in ["NestedSelected", "Mean", "Ridge"]
        + (["TabPFN"] if tabpfn else [])
    }
    splits = GroupKFold(5).split(data.features, groups=data.groups)
    for outer, (train, test) in enumerate(splits):
        require_separated_groups(data.groups, train, test)
        record_allocation(tables["fold_allocations"], data, outer, -1, "train", train)
        record_allocation(tables["fold_allocations"], data, outer, -1, "test", test)
        selected, score = select_configuration(data, train, outer, seed, tables)
        values = predict_classical_models(
            data, train, test, selected, outer, seed, tables
        )
        tables["selected_configs"].append(
            {
                "target": data.target,
                "outer_fold": outer,
                "selected_config": selected,
                "inner_oof_rmse": score,
            }
        )
        if tabpfn:
            values["TabPFN"], columns, resolved = predict_tabpfn(
                data, train, test, checkpoint, seed, device
            )
            record_columns(
                tables["selected_feature_columns"],
                data.target,
                outer,
                -1,
                "TabPFN",
                columns,
            )
            protocol["tabpfn_model_path_setting"] = checkpoint.name
            if resolved:
                protocol["tabpfn_resolved_model"] = checkpoint.name
        for name, prediction in values.items():
            predictions[name][test] = prediction
        record_outer_predictions(data, outer, test, values, tables)
        print(data.target, "outer fold", outer, "selected", selected, flush=True)
    # classification views with direct pharmaceutical meaning: for EE the
    # question is whether a formulation reaches the top of the commercial
    # specification band (>= 95%); for size and zeta potential the meaningful
    # targets are the standard nanomedicine bands (EPR-accessible / long-
    # circulation / rapid-clearance; anionic / neutral / cationic).
    classification = classification_target(data.target, data.observed)
    if classification is not None:
        labels, task = classification
        probability = np.full((len(data.observed), np.unique(labels).size), np.nan)
        for outer, (train, test) in enumerate(
            GroupKFold(5).split(data.features, groups=data.groups)
        ):
            fold_columns = [
                column
                for column in data.features.columns
                if data.features.iloc[train][column].notna().any()
                and data.features.iloc[train][column].nunique(dropna=True) > 1
            ]
            model = HistGradientBoostingClassifier(
                random_state=seed,
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=15,
                l2_regularization=5.0,
            )
            model.fit(data.features.iloc[train][fold_columns], labels[train])
            probability[test] = model.predict_proba(
                data.features.iloc[test][fold_columns]
            )
        require_finite(probability)
        present = np.unique(labels)
        entry = {"target": data.target, "task": task, "n": len(labels),
                 "class_counts": json.dumps(
                     {str(int(c)): int((labels == c).sum()) for c in present})}
        if len(present) == 2:
            entry["roc_auc"] = round(
                float(roc_auc_score(labels, probability[:, 1])), 4)
        else:
            entry["balanced_accuracy"] = round(
                float(balanced_accuracy_score(labels, probability.argmax(1))), 4)
            for index, c in enumerate(present):
                binary = (labels == c).astype(int)
                if 0 < binary.sum() < len(labels):
                    entry[f"auc_class_{int(c)}"] = round(
                        float(roc_auc_score(binary, probability[:, index])), 4)
        tables["classification"].append(entry)
    for name, values in predictions.items():
        require_finite(values)
        tables["metrics"].append(
            {
                "target": data.target,
                "model": name,
                "n": len(data.observed),
                "groups": len(set(data.groups)),
                "missing_structure_group_rows": int(np.sum(data.groups == "NA_GROUP")),
                **metrics(data.observed, values),
            }
        )


def write_tables(output: Path, tables: dict[str, list[dict]]) -> None:
    for name, records in tables.items():
        pd.DataFrame(records).to_csv(output / (name + ".csv"), index=False)


def verify_run_inputs(data_dir: Path, protocol: dict, checkpoint: Path | None) -> None:
    for filename, expected in protocol["inputs_sha256"].items():
        if file_hash(data_dir / filename) != expected:
            raise RuntimeError("Input changed during evaluation: " + filename)
    for filename, expected in protocol["implementation_sha256"].items():
        if file_hash(Path(__file__).with_name(filename)) != expected:
            raise RuntimeError("Implementation changed during evaluation")
    if (
        checkpoint is not None
        and file_hash(checkpoint) != protocol["tabpfn_checkpoint"]["sha256"]
    ):
        raise RuntimeError("Checkpoint changed during evaluation")


def evaluate(
    data_dir: str | Path,
    out: str | Path,
    targets: list[str],
    tabpfn: bool = True,
    seed: int = 42,
    device: str = "auto",
    model_path: str | Path | None = None,
) -> pd.DataFrame:
    data_dir, output = Path(data_dir), Path(out)
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(data_dir / "corrected_dataset.csv").set_index("row_index")
    groups = pd.read_csv(data_dir / "chemical_groups_dupaware.csv").set_index(
        "row_index"
    )
    if groups.index.duplicated().any():
        raise ValueError("Duplicate group identifiers")
    features = load_features(data_dir)
    checkpoint = Path(
        model_path or Path.home() / ".cache/tabpfn/tabpfn-v3.5-20260909.safetensors"
    )
    if tabpfn and not checkpoint.is_file():
        raise ValueError(
            "TabPFN checkpoint unavailable; supply --tabpfn-model-path or run --no-tabpfn"
        )
    protocol = build_protocol(data_dir, seed, tabpfn, checkpoint)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    tables = {
        name: []
        for name in [
            "metrics",
            "oof_predictions",
            "fold_allocations",
            "inner_scores",
            "selected_configs",
            "fold_metrics",
            "inner_oof_predictions",
            "selected_feature_columns",
            "classification",
        ]
    }
    for target in targets:
        prepared = prepare_target(data, groups, features, target)
        evaluate_target(
            prepared,
            tabpfn and target in TARGETS[:2],
            seed,
            device,
            checkpoint,
            protocol,
            tables,
        )
        write_tables(output, tables)
    verify_run_inputs(data_dir, protocol, checkpoint if tabpfn else None)
    protocol["completed"] = True
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    return pd.DataFrame(tables["metrics"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "results")
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=TARGETS)
    parser.add_argument("--no-tabpfn", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tabpfn-model-path", type=Path)
    args = parser.parse_args()
    print(
        evaluate(
            args.data_dir,
            args.output_dir,
            args.targets,
            not args.no_tabpfn,
            device=args.device,
            model_path=args.tabpfn_model_path,
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
