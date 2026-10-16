"""Run with: python3 -m unittest discover -s speedrun_utils -p 'test_*.py'."""

from array import array
from argparse import Namespace
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import yaml

from speedrun_utils import data_subsampling as sampling


class SamplingTests(unittest.TestCase):
    def test_minimum_labeled_fraction_retries_and_accepts_exact_boundary(self):
        # Two connected neighborhoods: 45% and exactly 50% labeled.
        adjacency = [array("I") for _ in range(40)]
        for start in (0, 20):
            for u in range(start, start + 19):
                adjacency[u].append(u + 1)
                adjacency[u + 1].append(u)
        source = sampling.Source(
            path=Path("city-roads-M"), info={"task": "regression"},
            node_ids=list(range(40)), index={i: i for i in range(40)},
            targets=["1"] * 9 + [""] * 11 + ["1"] * 10 + ["NaN"] * 10,
            target_header=["node_id", "label"], feature_header=["node_id", "f0"], split_counts=[1, 1, 8],
        )
        args = Namespace(seed=42, num_subsamples=5, min_nodes=20, max_nodes=20,
                         max_features=100, min_labeled_fraction=0.5)
        components = sampling.connected_components(adjacency)
        with patch.object(sampling, "sample_nodes", wraps=sampling.sample_nodes) as grow:
            samples = sampling.make_samples(source, adjacency, components, args)
            self.assertGreater(grow.call_count, 5)  # A poor neighborhood really was retried.
        for sample in samples:
            self.assertEqual(sample.nodes, list(range(20, 40)))
            self.assertEqual(sum(not sampling.missing(t) for t in sample.targets), 10)
        self.assertEqual(samples, sampling.make_samples(source, adjacency, components, args))

        args.min_labeled_fraction = 0.51
        with patch.object(sampling, "sample_nodes", wraps=sampling.sample_nodes) as grow:
            with self.assertRaisesRegex(ValueError, "at least 51% labeled.*100 attempts"):
                sampling.make_samples(source, adjacency, components, args)
            self.assertEqual(grow.call_count, 100)

    def test_rejects_invalid_labeled_fraction(self):
        for value in ("-0.1", "1.1", "nan", "inf"):
            with self.subTest(value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                sampling.parse_args(["--min-labeled-fraction", value])

    def test_connected_growth_excludes_small_components(self):
        adjacency = [array("I") for _ in range(32)]
        for u in range(29):
            adjacency[u].append(u + 1)
            adjacency[u + 1].append(u)
        adjacency[30].append(31)
        adjacency[31].append(30)
        components = sampling.connected_components(adjacency)
        self.assertEqual(sorted(map(len, components)), [2, 30])
        for seed in range(20):
            selected = sampling.sample_nodes(adjacency, components, 15, random.Random(seed))
            self.assertEqual(len(set(selected)), 15)
            self.assertLess(max(selected), 30)
            self.assertEqual(selected, list(range(selected[0], selected[0] + 15)))
        with self.assertRaisesRegex(ValueError, "No weakly connected component"):
            sampling.sample_nodes(adjacency, components, 33, random.Random(0))

    def test_split_quotas_stratification_and_unlabeled_nodes(self):
        targets = ["0"] * 72 + ["1"] * 23 + ["rare"] + ["", "NaN"]
        for task in ("regression", "multiclass_classification"):
            for seed in range(10):
                assignments = sampling.assign_splits(targets, task, [10, 10, 80], random.Random(seed))
                counts = Counter(split for label, split in zip(targets, assignments) if not sampling.missing(label))
                self.assertEqual([counts[i] for i in range(3)], [10, 9, 77])
                self.assertEqual(assignments[-2:], [0, 0])
                if task != "regression":
                    self.assertEqual(assignments[95], 0)  # A singleton class goes into training.
                    for split in range(3):
                        labels = {label for label, part in zip(targets, assignments) if part == split}
                        self.assertTrue({"0", "1"}.issubset(labels))
        with self.assertRaises(ValueError):
            sampling.assign_splits([""] * 50, "regression", [10, 10, 80], random.Random(0))
        with self.assertRaises(ValueError):
            sampling.assign_splits(["0"] * 50, "binary_classification", [10, 10, 80], random.Random(0))

    @staticmethod
    def write_csv(path, header, rows):
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    @staticmethod
    def read_csv(path):
        with path.open(newline="") as f:
            return list(csv.reader(f))

    def test_end_to_end_preserves_values_edges_metadata_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input" / "artnet-exp"
            source.mkdir(parents=True)
            ids = [10 + i * 3 for i in range(120)]
            features = [f"f{i}" for i in range(105)]
            info = {
                "dataset_name": "artnet-exp", "task": "binary_classification", "metric": "AP",
                "graph_is_directed": True, "has_unlabeled_nodes": True,
                "has_nans_in_numerical_features": True, "target_name": "label",
                "numerical_features_names": features[:102], "categorical_features_names": features[102:],
                "fraction_features_names": features[:3],
            }
            (source / "info.yaml").write_text(yaml.safe_dump(info))
            feature_rows = [[node_id, *["" if j == 0 else str(i * 1000 + j) for j in range(105)]]
                            for i, node_id in enumerate(ids)]
            target_rows = [[node_id, "" if i % 11 == 0 else str(i % 2)] for i, node_id in enumerate(ids)]
            edges = [[ids[i + 1], ids[i]] for i in range(119)]  # One-way edges, connectivity must be weak.
            edges += [[node_id, node_id] for node_id in ids]  # Preserve self-loops.
            edges += [[ids[i + 1], ids[i]] for i in range(119)]  # Preserve repeated edges.
            masks = []
            labeled_i = 0
            for node_id, target in target_rows:
                part = 0
                if target:
                    part = 0 if labeled_i % 10 == 0 else 1 if labeled_i % 10 == 1 else 2
                    labeled_i += 1
                masks.append([node_id, *[part == j for j in range(3)]])
            self.write_csv(source / "features.csv", ["node_id", *features], reversed(feature_rows))
            self.write_csv(source / "targets.csv", ["node_id", "label"], target_rows)
            self.write_csv(source / "edgelist.csv", ["source", "target"], edges)
            self.write_csv(source / "split_masks_RL.csv", ["node_id", "train", "val", "test"], masks)
            self.write_csv(source / "split_masks_TH.csv", ["node_id", "train", "val", "test"], masks)
            before = {p.name: p.read_bytes() for p in source.iterdir()}
            common = ["--input-dir", str(source.parent), "--datasets", "artnet-exp", "--min-nodes", "60",
                      "--max-nodes", "80", "--seed", "42"]
            for destination in (root / "first", root / "second"):
                with redirect_stdout(io.StringIO()):
                    sampling.main([*common, "--output-dir", str(destination)])
            self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})
            self.assertEqual(len(list((root / "first").iterdir())), 5)
            selections = []
            for i in range(5):
                output = root / "first" / f"artnet-exp-t-{i}"
                manifest = json.loads((output / "subsampling.json").read_text())
                self.assertEqual(manifest["min_labeled_fraction"], 0.5)
                self.assertGreaterEqual(1 - manifest["num_unlabeled_nodes"] / manifest["num_nodes"], 0.5)
                original = manifest["original_node_ids"]
                selections.append(tuple(original))
                self.assertTrue(60 <= len(original) <= 80)
                self.assertEqual(len(original), len(set(original)))
                self.assertEqual(original, list(range(original[0], original[0] + 3 * len(original), 3)))
                mapping = {node_id: j for j, node_id in enumerate(original)}
                self.assertFalse((output / "split_masks_TH.csv").exists())
                self.assertEqual(set(p.name for p in output.iterdir()), {
                    "features.csv", "targets.csv", "edgelist.csv", "split_masks_RL.csv",
                    "info.yaml", "subsampling.json",
                })
                actual_edges = self.read_csv(output / "edgelist.csv")
                expected_edges = [[str(mapping[u]), str(mapping[v])] for u, v in edges if u in mapping and v in mapping]
                self.assertEqual(actual_edges[1:], expected_edges)
                rows = self.read_csv(output / "features.csv")
                self.assertEqual(len(rows[0]), 101)
                columns = [int(name[1:]) for name in rows[0][1:]]
                for j, row in enumerate(rows[1:]):
                    source_row = feature_rows[ids.index(original[j])]
                    self.assertEqual(row, [str(j), *[str(source_row[c + 1]) for c in columns]])
                metadata = yaml.safe_load((output / "info.yaml").read_text())
                self.assertEqual(metadata["dataset_name"], output.name)
                self.assertTrue(metadata["graph_is_directed"])
                self.assertEqual(set(metadata["numerical_features_names"] + metadata["categorical_features_names"]),
                                 set(rows[0][1:]))
                self.assertEqual(metadata["has_nans_in_numerical_features"], "f0" in rows[0])
                targets = self.read_csv(output / "targets.csv")[1:]
                masks = self.read_csv(output / "split_masks_RL.csv")[1:]
                counts = [0, 0, 0]
                for j, (target, mask) in enumerate(zip(targets, masks)):
                    self.assertEqual(target, [str(j), target_rows[ids.index(original[j])][1]])
                    self.assertEqual(mask[0], str(j))
                    self.assertEqual(mask[1:].count("True"), 1)
                    if target[1]:
                        counts[mask[1:].index("True")] += 1
                    else:
                        self.assertEqual(mask[1:], ["True", "False", "False"])
                self.assertEqual(counts, sampling.apportion(sum(counts),
                                 [manifest["source_labeled_split_counts"][part] for part in sampling.SPLITS]))
                for p in output.iterdir():
                    self.assertEqual(p.read_bytes(), (root / "second" / output.name / p.name).read_bytes())
            self.assertEqual(len(set(selections)), 5)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                sampling.main([*common, "--output-dir", str(root / "first")])


if __name__ == "__main__":
    unittest.main()
