"""Full/tiny GraphLand evaluation and RF-50+NFA threshold accounting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from sklearn.decomposition import PCA
import torch

from baselines.catalog import discover_datasets
from baselines.data import Dataset, load_adjacency, load_dataset
from baselines.features import RawFeatures, preprocess_features
from baselines.run import score_predictions
from nanographpfn import TaskType


Suites = dict[str, list[Path]]

TINY_DATASETS = ("artnet-views", "city-reviews", "tolokers-2", "twitch-views")
AVAZU_PCA_COMPONENTS = 64
TASK_TYPES: dict[str, TaskType] = {"binary_classification": "binclass", "multiclass_classification": "multiclass",
              "regression": "regression"}


def evaluation_paths(data_dir: str | Path, full_names: list[str] | None = None,
                     tiny_names: list[str] | None = None) -> Suites:
    tiny_names = list(TINY_DATASETS if tiny_names is None else tiny_names)
    if not tiny_names or not set(tiny_names).issubset(TINY_DATASETS):
        raise ValueError(f"Tiny dataset names must be drawn from {TINY_DATASETS}")
    suites = {
        "full": discover_datasets(Path(data_dir), full_names, "full"),
        "tiny": discover_datasets(Path(data_dir), tiny_names, "tiny"),
    }
    expected = {f"{name}-t-{index}" for name in tiny_names for index in range(5)}
    if {p.name for p in suites["tiny"]} != expected:
        raise ValueError("Tiny evaluation requires exactly variants -t-0 through -t-4 for each selected dataset")
    return suites


def load_baselines(path: str | Path, suites: Suites, seed: int = 42) -> dict[str, Any]:
    """Select the exact evaluation population; missing/duplicate runs are errors."""
    contents = Path(path).read_bytes()
    report = json.loads(contents)
    selected = {}
    for suite, paths in suites.items():
        records = {}
        for dataset in paths:
            matches = [row for row in report["results"]
                       if row["model"] == "rf50_nfa" and row["dataset"] == dataset.name
                       and row["seed"] == seed and row["split"] == "RL"]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one RF-50+NFA seed={seed} RL result for {dataset.name}")
            row = matches[0]
            value = row.get("scores", {}).get("test")
            if row["status"] != "ok" or value is None or not math.isfinite(value):
                raise ValueError(f"Invalid RF-50+NFA test score for {dataset.name}")
            if row["variant"] != suite or row["n_estimators"] != 50 or row["nfa"] is not True:
                raise ValueError(f"Inconsistent RF-50+NFA metadata for {dataset.name}")
            records[dataset.name] = row
        selected[suite] = {
            "score": float(np.mean([r["scores"]["test"] for r in records.values()])),
            "datasets": records,
        }
    return {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(contents).hexdigest(),
            "model": "rf50_nfa", "seed": seed, "split": "RL", "score_split": "test",
            "aggregation": "equal_weight_mean_of_raw_dataset_scores", "suites": selected}


@dataclass
class PreparedGraph:
    """CPU evaluation inputs, with context nodes first.

    x: float32 [N, F], ordinal categories and baseline numerical transforms.
       Numerical columns are standardized on context later, by model.prepare().
    edge_index: int64 [2, E], source then destination.
    context_y: float32 [C], standardized for regression or class indices.
    feature_cardinalities: int64 [F]; zero means numerical.
    test_positions: [N_test] indices into query predictions [N-C, ...].
    test_targets: [N_test] targets in the original scoring units.
    """

    name: str
    x: torch.Tensor
    edge_index: torch.Tensor
    context_y: torch.Tensor
    feature_cardinalities: torch.Tensor
    task_type: TaskType
    num_classes: int | None
    test_positions: np.ndarray
    test_targets: np.ndarray
    target_mean: float
    target_scale: float
    metric: str
    details: dict[str, Any]


@dataclass
class PreparedFeatures:
    values: np.ndarray  # float32 [N, F]
    cardinalities: np.ndarray  # int64 [F]
    count_before_pca: int
    pca_details: dict[str, Any] | None


@dataclass
class PreparedTargets:
    context: np.ndarray  # [C], standardized targets or class indices
    test: np.ndarray  # [N_test], original regression/binary units or multiclass indices
    mean: float
    scale: float
    num_classes: int | None


def validate_evaluation_dataset(dataset: Dataset, baseline: dict[str, Any] | None) -> None:
    train = dataset.masks["train"]
    test = dataset.masks["test"]
    if not test.any() or (dataset.task == "regression" and test.sum() < 2):
        raise ValueError(f"{dataset.path}: test set is too small to score")
    if baseline is not None:
        counts = {key: int(mask.sum()) for key, mask in dataset.masks.items()}
        if (baseline["metric"] != dataset.info["metric"] or baseline["task"] != dataset.task
                or baseline["num_nodes"] != len(train) or baseline["labeled_split_counts"] != counts
                or baseline["num_raw_features"] != dataset.features.shape[1]):
            raise ValueError(f"{dataset.path}: dataset metadata no longer matches the baseline report")


def encode_ordinal_features(dataset: Dataset) -> tuple[RawFeatures, np.ndarray]:
    """Fit category codes/modes on all nodes, preserving the original column order."""
    columns = list(dataset.features)
    fractions = set(dataset.info["fraction_features_names"])
    categorical = set(dataset.info["categorical_features_names"])
    values = dataset.features.to_numpy(dtype=np.float32, copy=True)
    cardinalities = np.zeros(len(columns), dtype=np.int64)
    numeric_indices, fraction_indices, cat_indices = [], [], []
    for index, name in enumerate(columns):
        if name in categorical:
            cat_indices.append(index)
            observed = np.isfinite(values[:, index])
            if observed.any():
                vocabulary, inverse, counts = np.unique(values[observed, index], return_inverse=True,
                                                        return_counts=True)
                values[observed, index] = inverse
                values[~observed, index] = counts.argmax()
                cardinalities[index] = len(vocabulary)
            else:
                values[:, index] = 0
                cardinalities[index] = 1
        else:
            (fraction_indices if name in fractions else numeric_indices).append(index)
    # Same numeric transforms as the baseline. Keep categories ordinal, matching
    # nanoprior; the forest's one-hot representation is not suitable for this model.
    raw = RawFeatures(values, numeric_indices, fraction_indices, cat_indices, columns)
    return raw, cardinalities


def prepare_features(dataset: Dataset) -> PreparedFeatures:
    """Apply baseline transforms, drop context constants, then optionally reduce Avazu with PCA."""
    train = dataset.masks["train"]
    raw, cardinalities = encode_ordinal_features(dataset)
    features = preprocess_features(raw, dataset.base_name)
    keep = np.ptp(features[train], axis=0) > 0
    if not keep.any():
        # Keep a zero feature for a graph with entirely constant context inputs.
        keep[0] = True
    features, cardinalities = features[:, keep], cardinalities[keep]
    num_features_before_pca = features.shape[1]
    pca_details = None
    if dataset.base_name == "avazu-ctr" and num_features_before_pca > AVAZU_PCA_COMPONENTS:
        # GraphPFN uses 64-component PCA for high-dimensional inputs. Apply that
        # reduction to Avazu here, fitting on all nodes without using labels,
        # like the numeric transforms above. Cap rank for small sampled graphs.
        components = min(AVAZU_PCA_COMPONENTS, len(features) - 1)
        pca = PCA(n_components=components, svd_solver="randomized", random_state=0)
        features = np.asarray(pca.fit_transform(features), dtype=np.float32, order="C")
        cardinalities = np.zeros(components, dtype=np.int64)
        pca_details = {
            "n_components": components, "fit_on": "all_nodes", "svd_solver": "randomized",
            "random_state": 0, "whiten": False,
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        }
    return PreparedFeatures(features, cardinalities, num_features_before_pca, pca_details)


def prepare_targets(dataset: Dataset) -> PreparedTargets:
    """Fit target statistics/class vocabulary on labeled context nodes only."""
    train, test = dataset.masks["train"], dataset.masks["test"]
    context_y = dataset.targets[train]
    test_targets = dataset.targets[test]
    target_mean, target_scale, num_classes = 0.0, 1.0, None
    if dataset.task == "regression":
        target_mean = float(context_y.mean())
        target_scale = float(context_y.std())
        if target_scale < 1e-12:
            target_scale = 1.0
        context_y = (context_y - target_mean) / target_scale
    else:
        classes = np.unique(context_y)
        if not np.isin(test_targets, classes).all():
            raise ValueError(f"{dataset.path}: test labels contain a class absent from context")
        num_classes = len(classes)
        context_y = np.searchsorted(classes, context_y)
        if dataset.task == "binary_classification":
            # AP must retain the dataset's original positive class (label 1).
            if not np.array_equal(classes, [0, 1]):
                raise ValueError(f"{dataset.path}: binary context must contain labels 0 and 1")
        else:
            test_targets = np.searchsorted(classes, test_targets)
    return PreparedTargets(context_y, test_targets, target_mean, target_scale, num_classes)


def prepare_graph(path: str | Path, baseline: dict[str, Any] | None = None) -> PreparedGraph:
    """Prepare features/targets, then reorder nodes and edges into the model contract."""
    dataset = load_dataset(Path(path))
    validate_evaluation_dataset(dataset, baseline)
    features, targets = prepare_features(dataset), prepare_targets(dataset)
    train, test = dataset.masks["train"], dataset.masks["test"]
    order = np.concatenate((np.flatnonzero(train), np.flatnonzero(~train)))
    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(len(order))
    adjacency = load_adjacency(dataset).tocoo()
    edges = np.stack((inverse_order[adjacency.col], inverse_order[adjacency.row]))
    context_size = int(train.sum())
    details = {
        "num_nodes": len(features.values), "num_edges": edges.shape[1], "num_features": features.values.shape[1],
        "num_context": context_size, "num_test": int(test.sum()), "num_classes": targets.num_classes,
        "num_features_before_pca": features.count_before_pca, "pca": features.pca_details,
    }
    return PreparedGraph(
        name=dataset.path.name, x=torch.from_numpy(np.ascontiguousarray(features.values[order])),
        edge_index=torch.from_numpy(np.ascontiguousarray(edges)),
        context_y=torch.from_numpy(targets.context.astype(np.float32)),
        feature_cardinalities=torch.from_numpy(features.cardinalities.copy()),
        task_type=TASK_TYPES[dataset.task], num_classes=targets.num_classes,
        test_positions=inverse_order[np.flatnonzero(test)] - context_size, test_targets=targets.test,
        target_mean=targets.mean, target_scale=targets.scale, metric=dataset.info["metric"], details=details,
    )


class GraphEvaluator:
    def __init__(self, suites: Suites, baselines: dict[str, Any], device: str | torch.device,
                 amp: bool = True) -> None:
        self.suites, self.baselines = suites, baselines
        self.device = torch.device(device)
        self.amp = amp and self.device.type == "cuda"
        self.cache: dict[tuple[str, str], PreparedGraph] = {}

    def get_graph(self, suite: str, path: Path) -> PreparedGraph:
        key = (suite, path.name)
        if key not in self.cache:
            record = self.baselines["suites"][suite]["datasets"][path.name]
            self.cache[key] = prepare_graph(path, record)
        return self.cache[key]

    @torch.inference_mode()
    def evaluate_graph(self, model: torch.nn.Module, graph: PreparedGraph) -> dict[str, Any]:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
            predictions = model(
                graph.x.to(self.device), graph.context_y.to(self.device), graph.edge_index.to(self.device),
                graph.task_type, num_classes=graph.num_classes,
                feature_cardinalities=graph.feature_cardinalities.to(self.device),
            )
        predictions = predictions.float().cpu()
        if not torch.isfinite(predictions).all():
            raise ValueError(f"Nonfinite predictions for {graph.name}")
        predictions = predictions[graph.test_positions]
        if graph.task_type == "regression":
            predictions = predictions.numpy().astype(np.float64) * graph.target_scale + graph.target_mean
        elif graph.task_type == "binclass":
            predictions = predictions.softmax(-1)[:, 1].numpy()
        else:
            predictions = predictions.argmax(-1).numpy()
        score, note = score_predictions(graph.metric, graph.test_targets, predictions)
        if score is None or not math.isfinite(score):
            raise ValueError(f"Invalid score for {graph.name}: {note}")
        record = {**graph.details, "metric": graph.metric, "score": score,
                  "seconds": time.perf_counter() - start}
        if self.device.type == "cuda":
            record.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
        return record

    def evaluate(self, model: torch.nn.Module, log: Callable[[str], None] = print) -> dict[str, Any]:
        model.eval()
        start = time.perf_counter()
        results = {}
        for suite, paths in self.suites.items():
            records = {}
            for path in paths:
                log(f"Evaluating {suite}/{path.name}")
                try:
                    records[path.name] = self.evaluate_graph(model, self.get_graph(suite, path))
                except torch.OutOfMemoryError as exc:
                    raise RuntimeError(
                        f"Evaluation OOM on {suite}/{path.name}. Lower EVAL_NODE_BATCH_SIZE or "
                        "EVAL_TOKEN_BATCH_SIZE in nanographpfn.py and rerun preflight; "
                        f"no datasets were silently skipped. Original allocation error: {exc}"
                    ) from exc
                record = records[path.name]
                memory = (f", peak GPU {record['peak_allocated_bytes'] / 2**30:.2f} GiB"
                          if "peak_allocated_bytes" in record else "")
                log(f"  {record['metric']}={record['score']:.6f}, {record['seconds']:.2f}s{memory}")
            results[suite] = {"score": float(np.mean([r["score"] for r in records.values()])),
                              "datasets": records, "evaluation_elapsed_seconds": time.perf_counter() - start}
        return results


class ScoreTracker:
    """Best suite mean at ONE epoch and first observed RF threshold crossing."""

    def __init__(self, baselines: dict[str, Any], state: dict[str, Any] | None = None) -> None:
        self.state = state if state is not None else {
            suite: {"baseline_score": data["score"], "best_score": None, "best_epoch": None,
                    "best_dataset_scores": None, "baseline_reached": False,
                    "time_to_baseline_seconds": None, "wall_time_to_baseline_seconds": None,
                    "baseline_reached_epoch": None}
            for suite, data in baselines["suites"].items()
        }

    def update(self, results: dict[str, Any], epoch: int, train_seconds: float, wall_seconds: float) -> list[str]:
        improved = []
        for suite, result in results.items():
            score = result["score"]
            if not math.isfinite(score):
                raise ValueError(f"Nonfinite {suite} score")
            item = self.state[suite]
            if item["best_score"] is None or score > item["best_score"]:
                item.update(best_score=score, best_epoch=epoch,
                            best_dataset_scores={name: row["score"] for name, row in result["datasets"].items()})
                improved.append(suite)
            if not item["baseline_reached"] and score >= item["baseline_score"]:
                item.update(baseline_reached=True, time_to_baseline_seconds=train_seconds,
                            wall_time_to_baseline_seconds=result.get("wall_seconds", wall_seconds),
                            baseline_reached_epoch=epoch)
        return improved
