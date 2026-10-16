import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch
import yaml

from graph_eval import GraphEvaluator, prepare_graph


def write_dataset(root, name, n=96, width=80):
    path = root / name
    path.mkdir()
    rng = np.random.default_rng(17)
    ids = np.arange(n) * 3 + 101
    names = [f"fraction_{index}" for index in range(width)]
    values = rng.uniform(size=(n, width)).astype(np.float32)
    frame = pd.DataFrame(values, columns=names)
    frame.insert(0, "node_id", ids)
    frame["category"] = np.arange(n) % 5
    frame["constant"] = 1.0
    frame.to_csv(path / "features.csv", index=False)
    pd.DataFrame({"node_id": ids, "target": rng.normal(size=n)}).to_csv(path / "targets.csv", index=False)
    train, val, test = (np.arange(n) % 3 == index for index in (1, 0, 2))
    pd.DataFrame({"node_id": ids, "train": train, "val": val, "test": test}).to_csv(
        path / "split_masks_RL.csv", index=False)
    pd.DataFrame({"source": ids[:-1], "target": ids[1:]}).to_csv(path / "edgelist.csv", index=False)
    info = {"task": "regression", "metric": "R2", "target_name": "target",
            "numerical_features_names": names + ["constant"], "fraction_features_names": names,
            "categorical_features_names": ["category"]}
    (path / "info.yaml").write_text(yaml.safe_dump(info))
    return path, train, test


class EvaluationPCATests(unittest.TestCase):
    def test_avazu_pca_is_numeric_preserves_graph_and_ignores_query_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, train, test = write_dataset(Path(tmp), "avazu-ctr")
            # Raising the cutoff provides the existing pre-PCA features for an
            # independent transform and checks that topology/order stay intact.
            with patch("graph_eval.AVAZU_PCA_COMPONENTS", 1000):
                original = prepare_graph(path)
            fit_inputs = []

            class RecordingPCA(PCA):
                def fit_transform(self, features, y=None):
                    fit_inputs.append(features.copy())
                    return super().fit_transform(features)

            with patch("graph_eval.PCA", RecordingPCA):
                graph = prepare_graph(path)
            self.assertEqual(tuple(graph.x.shape), (96, 64))
            self.assertEqual(graph.details["num_features_before_pca"], 81)
            self.assertEqual(graph.details["pca"]["n_components"], 64)
            self.assertGreater(graph.details["pca"]["explained_variance_ratio_sum"], 0)
            self.assertLessEqual(graph.details["pca"]["explained_variance_ratio_sum"], 1.000001)
            self.assertEqual(graph.x.dtype, torch.float32)
            self.assertTrue(torch.isfinite(graph.x).all())
            self.assertTrue((graph.feature_cardinalities == 0).all())
            torch.testing.assert_close(graph.edge_index, original.edge_index, atol=0, rtol=0)
            torch.testing.assert_close(graph.context_y, original.context_y, atol=0, rtol=0)
            np.testing.assert_array_equal(graph.test_positions, original.test_positions)
            np.testing.assert_array_equal(graph.test_targets, original.test_targets)
            order = np.concatenate((np.flatnonzero(train), np.flatnonzero(~train)))
            # Verify all-node fitting, after existing transforms, before the
            # context/query reorder. Project independently with the learned basis.
            np.testing.assert_array_equal(fit_inputs[0][order], original.x.numpy())
            reference = PCA(n_components=64, svd_solver="randomized", random_state=0).fit(fit_inputs[0])
            expected = reference.transform(fit_inputs[0])[order]
            # Randomized fit_transform uses approximate U*S, whereas transform
            # projects X onto V; allow the small approximation residual.
            np.testing.assert_allclose(graph.x.numpy(), expected, atol=2e-3, rtol=2e-4)
            targets = pd.read_csv(path / "targets.csv")
            targets.loc[~train, "target"] += 1000
            targets.to_csv(path / "targets.csv", index=False)
            changed = prepare_graph(path)
            torch.testing.assert_close(changed.x, graph.x, atol=0, rtol=0)
            torch.testing.assert_close(changed.context_y, graph.context_y, atol=0, rtol=0)
            np.testing.assert_allclose(changed.test_targets, graph.test_targets + 1000)
            self.assertEqual(len(changed.test_targets), test.sum())
            json.dumps(graph.details, allow_nan=False)

    def test_other_datasets_and_narrow_avazu_do_not_use_pca(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wide, _, _ = write_dataset(root, "artnet-views")
            narrow, _, _ = write_dataset(root, "avazu-ctr", width=63)
            with patch("graph_eval.PCA", side_effect=AssertionError("PCA must not run")):
                for path, width in ((wide, 81), (narrow, 64)):
                    graph = prepare_graph(path)
                    self.assertEqual(graph.x.shape[1], width)
                    self.assertIsNone(graph.details["pca"])
                    self.assertEqual(int(graph.feature_cardinalities[-1]), 5)

    def test_small_avazu_variant_caps_components_to_centered_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _, _ = write_dataset(Path(tmp), "avazu-ctr-t-0", n=12)
            graph = prepare_graph(path)
            self.assertEqual(tuple(graph.x.shape), (12, 11))
            self.assertEqual(graph.details["pca"]["n_components"], 11)
            self.assertTrue(torch.isfinite(graph.x).all())

    def test_evaluator_caches_pca_and_reports_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _, _ = write_dataset(Path(tmp), "avazu-ctr")
            evaluator = GraphEvaluator({"full": [path]}, {"suites": {"full": {"datasets": {path.name: None}}}},
                                       device="cpu")
            with patch("graph_eval.prepare_graph", wraps=prepare_graph) as prepare:
                first = evaluator.get_graph("full", path)
                self.assertIs(first, evaluator.get_graph("full", path))
                prepare.assert_called_once()

            class ZeroModel(torch.nn.Module):
                def forward(self, x, context_y, *args, **kwargs):
                    return x.new_zeros(len(x) - len(context_y))

            result = evaluator.evaluate_graph(ZeroModel(), first)
            self.assertEqual(result["num_features"], 64)
            self.assertEqual(result["pca"], first.details["pca"])
            json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
