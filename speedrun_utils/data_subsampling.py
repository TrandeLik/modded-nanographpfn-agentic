#!/usr/bin/env python3
"""Create small, connected GraphLand datasets in the original CSV/YAML format.

From the repository root:
    python3 speedrun_utils/data_subsampling.py
    python3 speedrun_utils/data_subsampling.py --datasets artnet-exp city-roads-M
    python3 speedrun_utils/data_subsampling.py --output-dir /tmp/graphland-small

Requires PyYAML; graph processing uses only the Python standard library.
By default, create five independent samples per available GraphLand dataset,
with a uniformly drawn size of 500--1,000 nodes and at most 100 raw features.
At least 50% of each sample's nodes must have labels (configurable with
--min-labeled-fraction). Retry up to 100 neighborhoods at the same size.
Random frontier growth produces weakly connected induced subgraphs. This is
neighborhood sampling, not uniform sampling over all possible connected graphs.
Samples may overlap. Edges retain their directions, duplicates, and self-loops.

RL split proportions are measured over labeled source nodes and reapportioned
to sampled labeled nodes with integer rounding. Classification is stratified,
with priority given to representing sampled classes in training. Unlabeled
nodes remain in training, as in the original GraphLand masks. RH/TH splits are
not copied. Each output includes subsampling.json with seeds, selected original
node IDs, retained features, and counts. Existing output folders are refused.
"""

from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile

import yaml


GRAPHLAND_DATASETS = (
    "artnet-exp", "artnet-views", "avazu-ctr", "city-reviews",
    "city-roads-L", "city-roads-M", "hm-categories", "hm-prices",
    "pokec-regions", "tolokers-2", "twitch-views", "web-fraud",
    "web-topics", "web-traffic",
)
SPLITS = ("train", "val", "test")
REPO_ROOT = Path(__file__).resolve().parents[1]


def missing(value: str) -> bool:
    return value.strip().lower() in {"", "nan"}


def sample_seed(seed: int, name: str, index: int) -> int:
    """Stable across Python processes, dataset order, and subset selection."""
    key = f"{seed}\0{name}\0{index}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def apportion(total: int, weights: list[int]) -> list[int]:
    """Largest-remainder rounding, with totals conserved exactly."""
    denominator = sum(weights)
    if denominator <= 0:
        raise ValueError("The source RL split has no labeled nodes")
    counts = [total * weight // denominator for weight in weights]
    order = sorted(range(len(weights)), key=lambda i: (-(total * weights[i] % denominator), i))
    for i in order[:total - sum(counts)]:
        counts[i] += 1
    return counts


@dataclass
class Source:
    path: Path
    info: dict
    node_ids: list[int]
    index: dict[int, int]
    targets: list[str]
    target_header: list[str]
    feature_header: list[str]
    split_counts: list[int]


def load_source(path: Path) -> Source:
    with (path / "info.yaml").open() as f:
        info = yaml.safe_load(f)
    if info["task"] not in {"regression", "binary_classification", "multiclass_classification"}:
        raise ValueError(f"{path}: unsupported task {info['task']!r}")
    node_ids, targets = [], []
    with (path / "targets.csv").open(newline="") as f:
        reader = csv.reader(f)
        target_header = next(reader)
        if target_header != ["node_id", info["target_name"]]:
            raise ValueError(f"{path}: unexpected target columns {target_header}")
        for row in reader:
            if len(row) != 2:
                raise ValueError(f"{path}: invalid target row")
            node_ids.append(int(row[0]))
            targets.append(row[1])
    index = {node_id: i for i, node_id in enumerate(node_ids)}
    if not node_ids or len(index) != len(node_ids):
        raise ValueError(f"{path}: empty or duplicate node IDs")
    with (path / "features.csv").open(newline="") as f:
        feature_header = next(csv.reader(f))
    expected = info["numerical_features_names"] + info["categorical_features_names"]
    if (feature_header[0] != "node_id" or len(set(feature_header)) != len(feature_header)
            or len(expected) != len(feature_header) - 1 or set(expected) != set(feature_header[1:])):
        raise ValueError(f"{path}: feature columns disagree with info.yaml")
    if not set(info.get("fraction_features_names", [])).issubset(info["numerical_features_names"]):
        raise ValueError(f"{path}: fraction features must be numerical features")

    split_counts = [0, 0, 0]
    seen = bytearray(len(node_ids))
    with (path / "split_masks_RL.csv").open(newline="") as f:
        reader = csv.reader(f)
        if next(reader) != ["node_id", *SPLITS]:
            raise ValueError(f"{path}: unexpected RL split columns")
        for row in reader:
            if len(row) != 4 or any(v not in {"True", "False"} for v in row[1:]):
                raise ValueError(f"{path}: invalid RL mask row")
            i = index[int(row[0])]
            mask = [v == "True" for v in row[1:]]
            if seen[i] or sum(mask) != 1:
                raise ValueError(f"{path}: RL masks must partition nodes exactly once")
            seen[i] = 1
            if not missing(targets[i]):
                split_counts[mask.index(True)] += 1
    if not all(seen) or not all(split_counts):
        raise ValueError(f"{path}: missing nodes or empty labeled RL partition")
    return Source(path, info, node_ids, index, targets, target_header, feature_header, split_counts)


def load_adjacency(source: Source) -> tuple[list[array], int]:
    """Compact weak adjacency; duplicate neighbors do not affect sampling."""
    adjacency = [array("I") for _ in source.node_ids]
    edge_count = 0
    with (source.path / "edgelist.csv").open(newline="") as f:
        reader = csv.reader(f)
        if next(reader) != ["source", "target"]:
            raise ValueError(f"{source.path}: unexpected edge columns")
        for row in reader:
            if len(row) != 2:
                raise ValueError(f"{source.path}: invalid edge row")
            u, v = source.index[int(row[0])], source.index[int(row[1])]
            if u != v:
                adjacency[u].append(v)
                adjacency[v].append(u)
            edge_count += 1
    return adjacency, edge_count


def connected_components(adjacency: list[array]) -> list[array]:
    seen = bytearray(len(adjacency))
    components = []
    for start in range(len(adjacency)):
        if seen[start]:
            continue
        component = array("I", [start])
        seen[start] = 1
        for u in component:
            for v in adjacency[u]:
                if not seen[v]:
                    seen[v] = 1
                    component.append(v)
        components.append(component)
    return components


def sample_nodes(adjacency: list[array], components: list[array], size: int, rng: random.Random) -> list[int]:
    """Uniform seed among nodes in sufficiently large components; random growth."""
    eligible = [component for component in components if len(component) >= size]
    if not eligible:
        raise ValueError(f"No weakly connected component contains {size} nodes")
    offset = rng.randrange(sum(map(len, eligible)))
    for component in eligible:
        if offset < len(component):
            start = component[offset]
            break
        offset -= len(component)
    discovered = bytearray(len(adjacency))
    discovered[start] = 1
    frontier, selected = [start], []
    while len(selected) < size:
        i = rng.randrange(len(frontier))
        u = frontier[i]
        frontier[i] = frontier[-1]
        frontier.pop()
        selected.append(u)
        if len(selected) == size:
            break
        for v in adjacency[u]:
            if not discovered[v]:
                discovered[v] = 1
                frontier.append(v)
    return sorted(selected)


def assign_splits(targets: list[str], task: str, source_counts: list[int], rng: random.Random) -> list[int]:
    """Return split indices; unlabeled nodes use GraphLand's training mask."""
    labeled = [i for i, target in enumerate(targets) if not missing(target)]
    quotas = apportion(len(labeled), source_counts)
    if not all(quotas):
        raise ValueError("Sample has too few labeled nodes for nonempty RL splits")
    assignment = [0] * len(targets)
    if task == "regression":
        rng.shuffle(labeled)
        offset = 0
        for split, count in enumerate(quotas):
            for i in labeled[offset:offset + count]:
                assignment[i] = split
            offset += count
        return assignment

    groups = defaultdict(list)
    for i in labeled:
        groups[targets[i]].append(i)
    if len(groups) < 2:
        raise ValueError("Classification sample must contain at least two classes")
    classes = sorted(groups)
    sizes = [len(groups[label]) for label in classes]
    allocated = [[0, 0, 0] for _ in classes]
    remaining = sizes.copy()
    capacity = quotas.copy()

    def reserve(c: int, split: int) -> None:
        allocated[c][split] += 1
        remaining[c] -= 1
        capacity[split] -= 1

    # Cover classes in training when feasible; cover validation/test as well
    # when each eligible class has at least three nodes and quotas allow it.
    if capacity[0] >= len(classes):
        for c in range(len(classes)):
            reserve(c, 0)
        eligible = [c for c, size in enumerate(sizes) if size >= 3]
        for split in (1, 2):
            if capacity[split] >= len(eligible):
                for c in eligible:
                    reserve(c, split)

    # Fill the largest class/split deficit while respecting both margins.
    # Thus rounding cannot change the required global train/val/test counts.
    while any(remaining):
        _, c, split = max(
            (sizes[c] * quotas[split] - allocated[c][split] * len(labeled), -c, -split)
            for c in range(len(classes)) if remaining[c]
            for split in range(3) if capacity[split]
        )
        c, split = -c, -split
        reserve(c, split)
    for c, label in enumerate(classes):
        members = groups[label]
        rng.shuffle(members)
        offset = 0
        for split, count in enumerate(allocated[c]):
            for i in members[offset:offset + count]:
                assignment[i] = split
            offset += count
    return assignment


@dataclass
class Sample:
    name: str
    seed: int
    nodes: list[int]
    feature_columns: list[int]
    targets: list[str]
    assignments: list[int]


def make_samples(
    source: Source, adjacency: list[array], components: list[array], args: argparse.Namespace
) -> list[Sample]:
    largest = max(map(len, components))
    if largest < args.min_nodes:
        raise ValueError(f"{source.path}: largest component has only {largest} nodes")
    samples = []
    for index in range(args.num_subsamples):
        seed = sample_seed(args.seed, source.path.name, index)
        rng = random.Random(seed)
        size = rng.randint(args.min_nodes, min(args.max_nodes, largest))
        # Retry neighborhoods with insufficient labels or unusable splits
        # without changing the requested sample size.
        for _ in range(100):
            nodes = sample_nodes(adjacency, components, size, rng)
            targets = [source.targets[i] for i in nodes]
            labeled_count = sum(not missing(target) for target in targets)
            if labeled_count / size < args.min_labeled_fraction:
                continue
            try:
                assignments = assign_splits(targets, source.info["task"], source.split_counts, rng)
            except ValueError:
                continue
            break
        else:
            raise ValueError(
                f"{source.path}: could not sample {size} nodes with at least "
                f"{args.min_labeled_fraction:.0%} labeled nodes and usable RL splits in 100 attempts"
            )
        columns = list(range(1, len(source.feature_header)))
        if len(columns) > args.max_features:
            columns = sorted(rng.sample(columns, args.max_features))
        samples.append(Sample(f"{source.path.name}-t-{index}", seed, nodes, columns, targets, assignments))
    return samples


def write_samples(source: Source, samples: list[Sample], edge_count: int, args: argparse.Namespace) -> list[dict]:
    """Read each large feature/edge file once for all samples; stage outputs."""
    staging = Path(tempfile.mkdtemp(prefix=f".{source.path.name}-subsamples-", dir=args.output_dir))
    manifests = []
    try:
        memberships = defaultdict(dict)
        for s, sample in enumerate(samples):
            (staging / sample.name).mkdir()
            for new_id, i in enumerate(sample.nodes):
                memberships[source.node_ids[i]][s] = new_id
        has_nans = [False] * len(samples)
        numerical_columns = [
            [c for c in sample.feature_columns if source.feature_header[c] in source.info["numerical_features_names"]]
            for sample in samples
        ]
        with ExitStack() as stack:
            writers = [csv.writer(stack.enter_context((staging / sample.name / "features.csv").open("w", newline="")))
                       for sample in samples]
            for writer, sample in zip(writers, samples):
                writer.writerow(["node_id", *[source.feature_header[c] for c in sample.feature_columns]])
            rows = [[] for _ in samples]
            seen = bytearray(len(source.node_ids))
            with (source.path / "features.csv").open(newline="") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) != len(source.feature_header):
                        raise ValueError(f"{source.path}: invalid feature row")
                    node_id = int(row[0])
                    i = source.index[node_id]
                    if seen[i]:
                        raise ValueError(f"{source.path}: duplicate feature node")
                    seen[i] = 1
                    for s, new_id in memberships.get(node_id, {}).items():
                        rows[s].append([new_id, *[row[c] for c in samples[s].feature_columns]])
                        has_nans[s] |= any(missing(row[c]) for c in numerical_columns[s])
            if not all(seen):
                raise ValueError(f"{source.path}: features are missing nodes")
            for writer, sample_rows in zip(writers, rows):
                writer.writerows(sorted(sample_rows, key=lambda row: row[0]))

        output_edges = [0] * len(samples)
        with ExitStack() as stack:
            writers = [csv.writer(stack.enter_context((staging / sample.name / "edgelist.csv").open("w", newline="")))
                       for sample in samples]
            for writer in writers:
                writer.writerow(["source", "target"])
            with (source.path / "edgelist.csv").open(newline="") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    from_samples = memberships.get(int(row[0]))
                    if not from_samples:
                        continue
                    to_samples = memberships.get(int(row[1]), {})
                    for s, u in from_samples.items():
                        if s in to_samples:
                            writers[s].writerow([u, to_samples[s]])
                            output_edges[s] += 1

        for s, sample in enumerate(samples):
            directory = staging / sample.name
            with (directory / "targets.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(source.target_header)
                writer.writerows(enumerate(sample.targets))
            with (directory / "split_masks_RL.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["node_id", *SPLITS])
                writer.writerows([i, *[split == part for part in range(3)]]
                                 for i, split in enumerate(sample.assignments))
            feature_names = [source.feature_header[c] for c in sample.feature_columns]
            info = dict(source.info)
            info["dataset_name"] = sample.name
            for key in ("numerical_features_names", "categorical_features_names", "fraction_features_names"):
                info[key] = [name for name in source.info.get(key, []) if name in feature_names]
            info["has_unlabeled_nodes"] = any(map(missing, sample.targets))
            info["has_nans_in_numerical_features"] = has_nans[s]
            with (directory / "info.yaml").open("w") as f:
                yaml.safe_dump(info, f, sort_keys=False)
            split_counts = Counter(SPLITS[split] for target, split in zip(sample.targets, sample.assignments)
                                   if not missing(target))
            manifest = {
                "dataset_name": sample.name,
                "source_dataset": source.path.name,
                "base_seed": args.seed,
                "sample_seed": sample.seed,
                "min_labeled_fraction": args.min_labeled_fraction,
                "num_nodes": len(sample.nodes),
                "num_features": len(feature_names),
                "num_edges": output_edges[s],
                "num_unlabeled_nodes": sum(map(missing, sample.targets)),
                "source_num_nodes": len(source.node_ids),
                "source_num_edges": edge_count,
                "source_labeled_split_counts": dict(zip(SPLITS, source.split_counts)),
                "labeled_split_counts": {name: split_counts[name] for name in SPLITS},
                "original_node_ids": [source.node_ids[i] for i in sample.nodes],
                "feature_names": feature_names,
            }
            if source.info["task"] != "regression":
                manifest["class_counts_by_split"] = {
                    name: dict(sorted(Counter(target for target, split in zip(sample.targets, sample.assignments)
                                               if split == part and not missing(target)).items()))
                    for part, name in enumerate(SPLITS)
                }
            with (directory / "subsampling.json").open("w") as f:
                json.dump(manifest, f, indent=2)
                f.write("\n")
            manifests.append(manifest)
        for sample in samples:
            destination = args.output_dir / sample.name
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite {destination}")
            (staging / sample.name).rename(destination)
    finally:
        shutil.rmtree(staging)
    return manifests


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, help="Default: same directory as --input-dir")
    parser.add_argument(
        "--datasets", nargs="+", choices=GRAPHLAND_DATASETS, help="Default: all available GraphLand datasets"
    )
    parser.add_argument("--num-subsamples", type=int, default=5)
    parser.add_argument("--min-nodes", type=int, default=500)
    parser.add_argument("--max-nodes", type=int, default=1000)
    parser.add_argument("--max-features", type=int, default=100)
    parser.add_argument(
        "--min-labeled-fraction", type=float, default=0.5,
        help="Minimum fraction of sampled nodes with targets, between 0 and 1 (default: 0.5)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if not (1 <= args.min_nodes <= args.max_nodes) or args.num_subsamples < 1 or args.max_features < 1:
        parser.error("Require 1 <= min-nodes <= max-nodes and positive num-subsamples/max-features")
    if not 0 <= args.min_labeled_fraction <= 1:
        parser.error("Require 0 <= min-labeled-fraction <= 1")
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve() if args.output_dir else args.input_dir
    if args.datasets is None:
        args.datasets = [name for name in GRAPHLAND_DATASETS if (args.input_dir / name / "info.yaml").is_file()]
    if not args.datasets or len(set(args.datasets)) != len(args.datasets):
        parser.error("Provide at least one available dataset, with no duplicates")
    for name in args.datasets:
        for filename in ("info.yaml", "features.csv", "targets.csv", "edgelist.csv", "split_masks_RL.csv"):
            if not (args.input_dir / name / filename).is_file():
                parser.error(f"Missing input: {args.input_dir / name / filename}")
        for i in range(args.num_subsamples):
            output = args.output_dir / f"{name}-t-{i}"
            if output.exists():
                parser.error(f"Refusing to overwrite {output}; choose another --output-dir")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for name in args.datasets:
        print(f"{name}: reading source graph", flush=True)
        source = load_source(args.input_dir / name)
        adjacency, edge_count = load_adjacency(source)
        print(f"{name}: {len(adjacency):,} nodes, {edge_count:,} stored edges; finding weak components", flush=True)
        components = connected_components(adjacency)
        samples = make_samples(source, adjacency, components, args)
        del adjacency, components
        print(f"{name}: writing {len(samples)} induced subgraphs", flush=True)
        for manifest in write_samples(source, samples, edge_count, args):
            counts = manifest["labeled_split_counts"]
            print(f"  {manifest['dataset_name']}: {manifest['num_nodes']} nodes, "
                  f"{manifest['num_features']} features, {manifest['num_edges']:,} edges, "
                  f"labeled RL={counts['train']}/{counts['val']}/{counts['test']}, "
                  f"unlabeled={manifest['num_unlabeled_nodes']}", flush=True)
            total += 1
    print(f"Created {total} subsamples in {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
