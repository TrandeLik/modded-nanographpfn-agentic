"""Run all four fixed random forests: python3 -m baselines.run --n-jobs 8.

Use --list to inspect the plan without installing the scientific dependencies.
Defaults: all available full/tiny GraphLand datasets, RL split, seed 42.
"""

import argparse
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import platform
import sys
from time import perf_counter
import uuid

from .catalog import MODELS, REPO_ROOT, dataset_identity, discover_datasets
from .results import utc_now, write_json


PREPROCESSING = {
    "reference": "https://github.com/yandex-research/graphland/blob/main/dataset.py",
    "feature_fit_scope": "all_nodes_transductive",
    "categorical": "one_hot_drop_if_binary",
    "numerical": "quantile_normal_n_quantiles_min_1000_n_nodes_subsample_none_random_state_0",
    "fraction": "quantile_normal_for_avazu-ctr_and_artnet-views_otherwise_identity",
    "missing_values": "most_frequent_after_transform_all_nodes",
    "all_missing_columns": "retain_and_fill_zero",
    "regression_targets": "standard_scaler_fit_labeled_train_inverse_before_scoring",
    "nfa": {
        "hops": 1, "input": "raw_features_before_normalization_or_imputation",
        "graph": "undirected_unique_neighbors_no_self_loops",
        "numerical": ["mean", "min", "max"], "categorical": "one_hot_neighbor_frequencies",
        "missing_neighbors": "ignore_missing_per_feature_empty_neighborhood_is_nan",
        "derived_feature_types": "numeric_statistics_inherit_source_type_category_frequencies_are_fractions",
        "tiny_graph": "use_only_edges_and_features_in_the_tiny_dataset",
        "labels_used": False,
    },
}


def dependency_versions() -> dict:
    from . import data, features  # noqa: F401 -- verify scientific imports before creating results

    packages = ("numpy", "scipy", "pandas", "scikit-learn", "PyYAML")
    return {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in packages}}


def score_predictions(metric, targets, predictions):
    from sklearn.metrics import accuracy_score, average_precision_score, r2_score

    if len(targets) == 0:
        return None, "No labeled evaluation nodes"
    if metric == "R2" and len(targets) < 2:
        return None, "R2 requires at least two labeled evaluation nodes"
    functions = {"accuracy": accuracy_score, "AP": average_precision_score, "R2": r2_score}
    return float(functions[metric](targets, predictions)), None


def predict(model, features, task, target_scaler):
    import numpy as np

    if task == "binary_classification":
        positive = np.flatnonzero(model.classes_ == 1)
        if not len(positive):
            return np.zeros(features.shape[0], dtype=np.float64)
        return model.predict_proba(features)[:, positive[0]]
    predictions = model.predict(features)
    if target_scaler is not None:
        predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()
    return predictions


def record_header(path, model_name, seed):
    base, index = dataset_identity(path.name)
    return {
        "dataset": path.name, "base_dataset": base,
        "variant": "full" if index is None else "tiny", "tiny_index": index,
        "model": model_name, **MODELS[model_name], "seed": seed, "split": "RL",
    }


def error_record(path, model_name, seed, exc):
    return {**record_header(path, model_name, seed), "status": "error",
            "error": {"type": type(exc).__name__, "message": str(exc)}}


def evaluate_dataset(path, model_names, seed, n_jobs):
    """Yield each result immediately; share feature preparation between forests."""
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    from .data import load_adjacency, load_dataset
    from .features import add_nfa, encode_features, preprocess_features

    start = perf_counter()
    dataset = load_dataset(path)
    raw = encode_features(dataset)
    load_seconds = perf_counter() - start
    train = dataset.masks["train"]
    target_scaler = None
    y_train = dataset.targets[train]
    if dataset.task == "regression":
        target_scaler = StandardScaler()
        y_train = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    else:
        y_train = y_train.astype(np.int64)
    dataset_details = {
        "task": dataset.task, "metric": dataset.info["metric"],
        "num_nodes": len(dataset.targets), "num_raw_features": dataset.features.shape[1],
        "num_unlabeled_nodes": int(np.isnan(dataset.targets).sum()),
        "labeled_split_counts": {name: int(mask.sum()) for name, mask in dataset.masks.items()},
        "shared_load_seconds": load_seconds,
    }
    # Plain RFs run first so their results survive a later NFA preparation failure.
    for use_nfa in (False, True):
        group = [name for name in model_names if MODELS[name]["nfa"] == use_nfa]
        if not group:
            continue
        print(f"{path.name}: preparing {'NFA' if use_nfa else 'plain'} features", flush=True)
        start = perf_counter()
        try:
            if use_nfa:
                adjacency = load_adjacency(dataset)
                augmented = add_nfa(raw, adjacency)
                features = preprocess_features(augmented, dataset.base_name)
                del augmented, adjacency
            else:
                features = preprocess_features(raw, dataset.base_name)
        except Exception as exc:
            for name in group:
                yield {**error_record(path, name, seed, exc), **dataset_details}
            continue
        prepare_seconds = perf_counter() - start
        for name in group:
            print(f"{path.name}: fitting {name} ({features.shape[1]} features)", flush=True)
            record = {**record_header(path, name, seed), **dataset_details,
                      "num_model_features": features.shape[1], "shared_feature_seconds": prepare_seconds}
            estimator = RandomForestRegressor if dataset.task == "regression" else RandomForestClassifier
            model = estimator(n_estimators=MODELS[name]["n_estimators"], random_state=seed, n_jobs=n_jobs)
            record["model_params"] = model.get_params()
            try:
                start = perf_counter()
                model.fit(features[train], y_train)
                record["fit_seconds"] = perf_counter() - start
                scores, predict_seconds, notes = {}, {}, []
                for split in ("val", "test"):
                    mask = dataset.masks[split]
                    start = perf_counter()
                    predictions = (predict(model, features[mask], dataset.task, target_scaler)
                                   if mask.any() else np.empty(0))
                    predict_seconds[split] = perf_counter() - start
                    value, note = score_predictions(dataset.info["metric"], dataset.targets[mask], predictions)
                    scores[split] = value
                    if note:
                        notes.append(f"{split}: {note}")
                record.update(status="ok", scores=scores, predict_seconds=predict_seconds, notes=notes)
                if dataset.task != "regression":
                    record["training_classes"] = model.classes_.tolist()
            except Exception as exc:
                record.update(status="error", error={"type": type(exc).__name__, "message": str(exc)})
            finally:
                del model
            yield record
        del features


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--datasets", nargs="+", help="Base names select full + tiny; exact tiny names select one")
    parser.add_argument("--variants", choices=("all", "full", "tiny"), default="all")
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-jobs", type=int, default=8,
        help="RF workers; datasets/models run sequentially (default: 8)",
    )
    parser.add_argument("--output", type=Path, help="JSON file; default: baselines/results/<timestamp>-<id>.json")
    parser.add_argument("--list", action="store_true", help="Print the plan without fitting or importing sklearn")
    args = parser.parse_args(argv)
    if args.n_jobs == 0 or args.n_jobs < -1:
        parser.error("n-jobs must be -1 (all CPUs) or a positive integer")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be between 0 and 2**32 - 1")
    if len(set(args.models)) != len(args.models):
        parser.error("Duplicate models")
    try:
        args.paths = discover_datasets(args.data_dir.resolve(), args.datasets, args.variants)
    except ValueError as exc:
        parser.error(str(exc))
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        args.output = REPO_ROOT / "baselines" / "results" / f"{stamp}-{uuid.uuid4().hex[:8]}.json"
    args.output = args.output.resolve()
    if not args.list and args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}; choose a new --output")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    expected = len(args.paths) * len(args.models)
    print(f"{len(args.paths)} datasets x {len(args.models)} models = {expected} fits; RL; seed {args.seed}")
    if args.list:
        for path in args.paths:
            print(path.name)
        print("Models: " + ", ".join(args.models))
        return 0
    try:
        versions = dependency_versions()
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        print(f"Missing baseline dependency: {exc}\nInstall: python3 -m pip install -r baselines/requirements.txt",
              file=sys.stderr)
        return 2
    report = {
        "schema_version": 1, "status": "running", "started_at": utc_now(), "finished_at": None,
        "seed": args.seed, "split": "RL", "n_jobs": args.n_jobs, "transductive": True,
        "datasets": [p.name for p in args.paths], "models": args.models, "expected_runs": expected,
        "preprocessing": PREPROCESSING, "versions": versions,
        "score_units": "raw sklearn values; AP/accuracy in [0,1], R2 may be negative; no x100 scaling",
        "results": [],
    }
    start = perf_counter()
    write_json(args.output, report)
    print(f"Saving incremental results to {args.output}", flush=True)

    def save(record):
        report["results"].append(record)
        report["elapsed_seconds"] = perf_counter() - start
        write_json(args.output, report)
        outcome = record.get("scores", record.get("error"))
        print(f"  {record['dataset']} / {record['model']}: {record['status']} {outcome}", flush=True)

    try:
        for path in args.paths:
            completed = set()
            try:
                for record in evaluate_dataset(path, args.models, args.seed, args.n_jobs):
                    save(record)
                    completed.add(record["model"])
            except Exception as exc:
                for name in args.models:
                    if name not in completed:
                        save(error_record(path, name, args.seed, exc))
    except KeyboardInterrupt:
        report.update(status="interrupted", finished_at=utc_now(), elapsed_seconds=perf_counter() - start)
        write_json(args.output, report)
        return 130
    errors = sum(record["status"] == "error" for record in report["results"])
    report.update(status="completed_with_errors" if errors else "completed", finished_at=utc_now(),
                  elapsed_seconds=perf_counter() - start, successful_runs=len(report["results"]) - errors,
                  failed_runs=errors)
    write_json(args.output, report)
    print(f"Saved {len(report['results'])} results ({errors} errors) to {args.output}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
