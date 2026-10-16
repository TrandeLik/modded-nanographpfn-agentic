import csv
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import numpy as np
    from scipy import sparse
    import pandas as pd
    import sklearn
    import yaml
except ImportError:
    HAS_DEPENDENCIES = False
else:
    HAS_DEPENDENCIES = True
    from baselines.data import load_dataset, load_adjacency
    from baselines.features import RawFeatures, encode_features, add_nfa, neighbor_statistics, preprocess_features
    from baselines.run import evaluate_dataset, score_predictions, predict
    from baselines.catalog import MODELS


@unittest.skipUnless(HAS_DEPENDENCIES, "Install baselines/requirements.txt to run numerical/RF tests")
class NumericalTests(unittest.TestCase):
    def fixture(self, root, task="binary_classification", name="artnet-exp-t-0"):
        d = root / name
        d.mkdir()
        metrics = {"binary_classification": "AP", "multiclass_classification": "accuracy", "regression": "R2"}
        info = {
            "dataset_name": name, "task": task,
            "metric": metrics[task],
            "target_name": "y", "graph_is_directed": True,
            "numerical_features_names": ["amount", "empty", "fraction"],
            "fraction_features_names": ["fraction"], "categorical_features_names": ["category"],
        }
        (d / "info.yaml").write_text(yaml.safe_dump(info))
        ids = [10, 20, 30, 40, 50, 60, 70, 80]
        amounts = [1, "", 5, 9, 3, 4, 8, 2]
        categories = [0, 1, 0, 2, 1, 1, 2, 0]
        targets = {
            "binary_classification": [0, 1, 0, 1, 1, 0, 1, ""],
            "multiclass_classification": [0, 1, 2, 1, 2, 0, 2, ""],
            "regression": [1, 3, 5, 6, 9, 12, 14, ""],
        }[task]
        features = [[node, amounts[i], "", i / 10, categories[i]] for i, node in enumerate(ids)]
        masks = [[node, i in (0, 1, 2, 7), i in (3, 4), i in (5, 6)] for i, node in enumerate(ids)]
        edges = [[ids[i], ids[i + 1]] for i in range(7)] + [[20, 30], [30, 20], [10, 10]]
        files = {
            "features.csv": (["node_id", "amount", "empty", "fraction", "category"], features),
            "targets.csv": (["node_id", "y"], list(reversed(list(zip(ids, targets))))),
            "split_masks_RL.csv": (["node_id", "train", "val", "test"], list(reversed(masks))),
            "edgelist.csv": (["source", "target"], edges),
        }
        for filename, (header, rows) in files.items():
            with (d / filename).open("w", newline="") as f:
                writer = csv.writer(f); writer.writerow(header); writer.writerows(rows)
        return d

    def test_alignment_and_simple_undirected_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = load_dataset(self.fixture(Path(tmp)))
            np.testing.assert_array_equal(data.targets[:7], [0, 1, 0, 1, 1, 0, 1])
            self.assertEqual({k: int(v.sum()) for k, v in data.masks.items()}, {"train": 3, "val": 2, "test": 2})
            self.assertFalse(data.masks["train"][-1])
            adjacency = load_adjacency(data)
            self.assertEqual(adjacency.nnz, 14)  # Seven unique edges, in both directions.
            self.assertEqual(adjacency.diagonal().sum(), 0)
            np.testing.assert_array_equal(adjacency.toarray(), adjacency.T.toarray())
            self.assertEqual(np.diff(adjacency.indptr).tolist(), [1, 2, 2, 2, 2, 2, 2, 1])
            masks = pd.read_csv(data.path / "split_masks_RL.csv")
            masks["train"] = masks["train"].astype(object)
            masks.loc[0, "train"] = "bad"
            masks.to_csv(data.path / "split_masks_RL.csv", index=False)
            with self.assertRaisesRegex(ValueError, "True/False"):
                load_dataset(data.path)

    def test_nfa_ignores_missing_values_and_handles_isolated_nodes(self):
        adjacency = sparse.csr_matrix(np.array([
            [0, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
        ], dtype=bool))
        values = np.array([[2, np.nan], [4, 7], [np.nan, 9], [100, 13], [np.nan, np.nan]], dtype=np.float32)
        mean, minimum, maximum = neighbor_statistics(values, adjacency)
        np.testing.assert_allclose(mean[0], [4, 8])
        np.testing.assert_allclose(minimum[0], [4, 7])
        np.testing.assert_allclose(maximum[0], [4, 9])
        for block in (mean, minimum, maximum):
            np.testing.assert_allclose(block[1], [2, np.nan], equal_nan=True)
            self.assertTrue(np.isnan(block[3:]).all())

    def test_nfa_is_raw_uses_all_neighbor_features_and_never_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = load_dataset(self.fixture(Path(tmp)))
            raw = encode_features(data)
            adjacency = load_adjacency(data)
            augmented = add_nfa(raw, adjacency)
            # Node 20 sees raw amounts 1 and 5; the missing own value is excluded.
            for stat, value in (("mean", 3), ("min", 1), ("max", 5)):
                self.assertEqual(augmented.values[1, augmented.names.index(f"nfa_{stat}__amount")], value)
            # Node 30 sees node 40's category even though node 40 is in validation.
            freq = augmented.names.index("nfa_frequency__category_2.0")
            self.assertEqual(augmented.values[2, freq], 0.5)
            # Its other neighbor (node 20) has a missing amount: ignore before imputation.
            self.assertEqual(augmented.values[2, augmented.names.index("nfa_mean__amount")], 9)
            # Pandas may expose a read-only NumPy view; replace the targets.
            data.targets = np.full_like(data.targets, 999)
            np.testing.assert_allclose(encode_features(data).values, raw.values, equal_nan=True)
            np.testing.assert_allclose(
                add_nfa(encode_features(data), adjacency).values, augmented.values, equal_nan=True
            )

    def test_graphland_preprocessing_modes_fraction_policy_and_empty_features(self):
        values = np.array([[1, 0.0, np.nan], [2, 0.25, np.nan], [3, 0.5, np.nan], [4, 1.0, np.nan]], dtype=np.float32)
        raw = RawFeatures(values, [0, 2], [1], [], ["number", "fraction", "empty"])
        ordinary = preprocess_features(raw, "city-roads-M")
        special = preprocess_features(raw, "avazu-ctr")
        np.testing.assert_array_equal(ordinary[:, 1], values[:, 1])
        self.assertLess(ordinary[0, 0], -5)
        self.assertGreater(ordinary[-1, 0], 5)
        np.testing.assert_allclose(ordinary[:, 0], special[:, 1], atol=1e-5)
        self.assertTrue(np.isfinite(ordinary).all())
        np.testing.assert_array_equal(ordinary[:, 2], 0)
        self.assertTrue(np.isnan(raw.values[:, 2]).all())  # Input stays raw.
        mode = RawFeatures(np.array([[3], [1], [np.nan], [1], [1]], dtype=np.float32), [], [0], [], ["fraction"])
        self.assertEqual(preprocess_features(mode, "city-roads-M")[2, 0], 1)

    def test_encoding_drops_binary_category_and_uses_tiny_dataset_base_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = load_dataset(self.fixture(Path(tmp), name="artnet-views-t-2"))
            data.features["category"] = [0, 1, 0, 1, 0, 1, 0, 1]
            raw = encode_features(data)
            self.assertEqual(len(raw.categorical), 1)
            self.assertEqual(data.base_name, "artnet-views")
            transformed = preprocess_features(raw, data.base_name)
            col = raw.fraction[0]
            self.assertLess(transformed[0, col], -5)
            self.assertGreater(transformed[-1, col], 5)

    def test_metrics_use_probability_ranking_and_original_target_units(self):
        self.assertAlmostEqual(score_predictions("AP", [0, 1, 1, 0], [.1, .4, .35, .8])[0], 7 / 12)
        self.assertAlmostEqual(score_predictions("accuracy", [2, 4, 4], [2, 2, 4])[0], 2 / 3)
        self.assertAlmostEqual(score_predictions("R2", [1, 2, 3], [1, 2, 2])[0], 0.5)
        self.assertLess(score_predictions("R2", [1, 2], [10, 10])[0], 0)
        self.assertIsNone(score_predictions("R2", [1], [1])[0])
        self.assertIsNone(score_predictions("accuracy", [], [])[0])

    def test_regression_scaler_fits_only_training_targets(self):
        fitted_y = []

        class ConstantRegressor:
            def __init__(self, **kwargs):
                self.params = kwargs

            def get_params(self):
                return self.params

            def fit(self, x, y):
                fitted_y.append(y.copy())

            def predict(self, x):
                return np.zeros(len(x))  # Training mean in standardized units.

        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(Path(tmp), task="regression", name="city-roads-M-t-0")
            with patch("sklearn.ensemble.RandomForestRegressor", ConstantRegressor), redirect_stdout(io.StringIO()):
                records = list(evaluate_dataset(path, ["rf50", "rf50_nfa"], 42, 1))
            for y in fitted_y:
                np.testing.assert_allclose(y, (np.array([1, 3, 5]) - 3) / np.std([1, 3, 5]))
            for record in records:
                self.assertEqual(record["status"], "ok", record)
                self.assertAlmostEqual(record["scores"]["val"], score_predictions("R2", [6, 9], [3, 3])[0])
                self.assertAlmostEqual(record["scores"]["test"], score_predictions("R2", [12, 14], [3, 3])[0])

    def test_all_four_forests_on_synthetic_classification_and_regression(self):
        for task in ("binary_classification", "multiclass_classification", "regression"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as tmp:
                path = self.fixture(Path(tmp), task=task)
                with redirect_stdout(io.StringIO()):
                    records = list(evaluate_dataset(path, list(MODELS), 42, 1))
                self.assertEqual(len(records), 4)
                for record in records:
                    self.assertEqual(record["status"], "ok", record)
                    self.assertEqual(record["labeled_split_counts"]["train"], 3)
                    self.assertEqual(record["model_params"]["n_estimators"], MODELS[record["model"]]["n_estimators"])
                    self.assertEqual(record["model_params"]["random_state"], 42)
                    self.assertTrue(all(np.isfinite(value) for value in record["scores"].values()))
                self.assertGreater(records[2]["num_model_features"], records[0]["num_model_features"])


if __name__ == "__main__":
    unittest.main()
