"""Read GraphLand CSV/YAML data and align every file by node ID."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
import yaml

from .catalog import dataset_identity


TASK_METRICS = {"binary_classification": "AP", "multiclass_classification": "accuracy", "regression": "R2"}


@dataclass
class Dataset:
    path: Path
    base_name: str
    tiny_index: int | None
    info: dict
    features: pd.DataFrame
    targets: np.ndarray
    masks: dict[str, np.ndarray]

    @property
    def task(self) -> str:
        return self.info["task"]


def read_indexed(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col="node_id")
    if not frame.index.is_unique or not pd.api.types.is_integer_dtype(frame.index.dtype):
        raise ValueError(f"{path}: node IDs must be unique integers")
    return frame


def align(frame: pd.DataFrame, index: pd.Index, path: Path) -> pd.DataFrame:
    if len(frame) != len(index) or not frame.index.isin(index).all():
        raise ValueError(f"{path}: node IDs differ from features.csv")
    return frame.reindex(index)


def load_dataset(path: Path) -> Dataset:
    base, tiny_index = dataset_identity(path.name)
    with (path / "info.yaml").open() as f:
        info = yaml.safe_load(f)
    if info["task"] not in TASK_METRICS or info["metric"] != TASK_METRICS[info["task"]]:
        raise ValueError(f"{path}: unsupported task/metric combination")
    features = read_indexed(path / "features.csv")
    numerical = info["numerical_features_names"]
    categorical = info["categorical_features_names"]
    expected = numerical + categorical
    if not len(features) or not expected or len(set(expected)) != len(expected) or set(features) != set(expected):
        raise ValueError(f"{path}: feature columns disagree with metadata")
    if not set(info["fraction_features_names"]).issubset(numerical):
        raise ValueError(f"{path}: fraction features must be a subset of numerical features")
    if np.isinf(features.to_numpy(dtype=np.float32)).any():
        raise ValueError(f"{path}: infinite feature values")
    targets_df = align(read_indexed(path / "targets.csv"), features.index, path / "targets.csv")
    if list(targets_df) != [info["target_name"]]:
        raise ValueError(f"{path}: unexpected target columns")
    targets = targets_df.iloc[:, 0].to_numpy(dtype=np.float64)
    if np.isinf(targets).any():
        raise ValueError(f"{path}: infinite targets")
    labeled = ~np.isnan(targets)
    if info["task"] == "binary_classification" and not np.isin(targets[labeled], [0, 1]).all():
        raise ValueError(f"{path}: binary labels must be 0 or 1")
    if info["task"] == "multiclass_classification" and not (targets[labeled] == np.floor(targets[labeled])).all():
        raise ValueError(f"{path}: multiclass labels must be integers")
    split_path = path / "split_masks_RL.csv"
    splits = align(read_indexed(split_path), features.index, split_path)
    if list(splits) != ["train", "val", "test"]:
        raise ValueError(f"{path}: unexpected RL mask columns")
    # Do not coerce arbitrary strings to bool: bool('False') is True.
    if any(splits[name].dtype != bool for name in splits):
        raise ValueError(f"{path}: RL masks must contain only True/False")
    if not (splits.to_numpy().sum(axis=1) == 1).all():
        raise ValueError(f"{path}: RL masks must partition every node exactly once")
    masks = {name: splits[name].to_numpy() & labeled for name in splits}
    if not masks["train"].any():
        raise ValueError(f"{path}: no labeled training nodes")
    return Dataset(path, base, tiny_index, info, features, targets, masks)


def load_adjacency(dataset: Dataset) -> sparse.csr_matrix:
    """Simple undirected graph: remove self-loops and deduplicate neighbors."""
    edges = pd.read_csv(dataset.path / "edgelist.csv", dtype={"source": np.int64, "target": np.int64})
    if list(edges) != ["source", "target"]:
        raise ValueError(f"{dataset.path}: unexpected edge columns")
    u = dataset.features.index.get_indexer(edges["source"])
    v = dataset.features.index.get_indexer(edges["target"])
    if (u < 0).any() or (v < 0).any():
        raise ValueError(f"{dataset.path}: edges reference unknown nodes")
    n = len(dataset.features)
    adjacency = sparse.coo_matrix((np.ones(len(u), dtype=bool), (u, v)), shape=(n, n)).tocsr()
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.setdiag(False)
    adjacency.eliminate_zeros()
    adjacency.sort_indices()
    return adjacency
