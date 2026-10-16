"""Numerical and multiprocessing checks for the standalone graph prior.

Run with: python -m unittest discover -s tests -v
These tests require the project's PyTorch and NumPy dependencies.
"""

import random
import unittest
from dataclasses import replace
from multiprocessing.connection import Listener
from unittest.mock import patch

import numpy as np
import torch

import nanoprior as prior


class PriorTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assert_same_task(self, first, second):
        for name in ("x", "y", "edge_index", "feature_cardinalities"):
            self.assertTrue(torch.equal(getattr(first, name), getattr(second, name)), name)
        for name in ("sep", "task_type", "num_classes", "metadata"):
            self.assertEqual(getattr(first, name), getattr(second, name), name)


class GraphTests(PriorTestCase):
    def test_simple_graph_and_largest_component(self):
        # Components: {1, 3, 5}, {0, 2}, isolated 4 and 6. Include duplicates,
        # both orientations, and a loop to exercise canonicalization.
        source = np.array([1, 3, 3, 5, 1, 0, 2, 6])
        target = np.array([3, 1, 5, 3, 3, 2, 0, 6])
        edges = prior._simple_undirected_edges(source, target, 7)
        edges, retained = prior._largest_component(edges, 7)
        self.assertEqual(retained.tolist(), [1, 3, 5])
        self.assertEqual(set(map(tuple, edges.t().tolist())), {(0, 1), (1, 0), (1, 2), (2, 1)})
        self.assertEqual(edges.shape[1], 4)

    def test_component_tie_and_empty_graph(self):
        edges = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        _, retained = prior._largest_component(edges, 4)
        self.assertEqual(retained.tolist(), [0, 1])
        empty = torch.empty((2, 0), dtype=torch.int64)
        edges, retained = prior._largest_component(empty, 4)
        self.assertEqual(retained.tolist(), [0])
        self.assertEqual(tuple(edges.shape), (2, 0))

    def test_degree_propensities_affect_endpoints(self):
        weights = np.array([0.7, 0.1, 0.1, 0.1])
        first, second = prior._sample_distinct_pairs(weights, 20000, prior._RandomState.from_seed(1))
        self.assertTrue(np.all(first != second))
        self.assertLess(abs(np.mean(first == 0) - 0.7), 0.015)
        # Conditional second-endpoint distribution has renormalized weights.
        self.assertLess(abs(np.mean(second[first == 1] == 0) - 0.7 / 0.9), 0.04)

    def test_extreme_hub_does_not_require_rejection_loop(self):
        weights = np.array([1 - 1e-10, 5e-11, 5e-11])
        first, second = prior._sample_distinct_pairs(weights, 1000, prior._RandomState.from_seed(2))
        self.assertTrue(np.all(first != second))
        self.assertTrue(np.all(second < len(weights)))

    def test_node_permutation_keeps_edges_aligned(self):
        features = torch.tensor([[10.0], [20.0], [30.0]])
        labels = torch.tensor([100, 200, 300])
        edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        order = torch.tensor([2, 0, 1])
        new_x, new_y, new_edges = prior._permute_nodes(features, labels, edges, order)
        self.assertEqual(new_x[:, 0].tolist(), [30, 10, 20])
        self.assertEqual(new_y.tolist(), [300, 100, 200])
        old_pairs = {(int(features[s]), int(features[t])) for s, t in edges.t()}
        new_pairs = {(int(new_x[s]), int(new_x[t])) for s, t in new_edges.t()}
        self.assertEqual(old_pairs, new_pairs)


class AggregationTests(PriorTestCase):
    def setUp(self):
        self.graph = prior._Graph(torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]), 4)
        self.values = torch.tensor([[1.0], [2.0], [4.0], [8.0]])

    def test_mean_min_max_and_isolates(self):
        expected = {
            "mean": [2.0, 2.5, 2.0, 0.0],
            "min": [2.0, 1.0, 2.0, 0.0],
            "max": [2.0, 4.0, 2.0, 0.0],
        }
        for kind, reference in expected.items():
            with self.subTest(kind=kind):
                actual = prior._aggregate(self.graph, self.values, kind, prior._RandomState.from_seed(0))
                torch.testing.assert_close(actual[:, 0], torch.tensor(reference))

    def test_gcn_normalization(self):
        expected = torch.tensor([2, 5, 2, 0], dtype=torch.float32) / np.sqrt(2)
        actual = prior._aggregate(self.graph, self.values, "gcn", prior._RandomState.from_seed(0))
        torch.testing.assert_close(actual[:, 0], expected)

    def test_source_to_destination_direction(self):
        graph = prior._Graph(torch.tensor([[0, 2], [1, 1]]), 3)
        actual = prior._aggregate(graph, self.values[:3], "mean", prior._RandomState.from_seed(0))
        torch.testing.assert_close(actual[:, 0], torch.tensor([0.0, 2.5, 0.0]))

    def test_every_aggregation_is_equivariant_to_node_permutation(self):
        values = torch.cat((self.values, self.values.square()), dim=1)
        order = torch.tensor([2, 0, 3, 1])
        permuted, _, edges = prior._permute_nodes(values, torch.zeros(4), self.graph.edges, order)
        graph = prior._Graph(edges, 4)
        for kind in prior.AGGREGATIONS:
            with self.subTest(kind=kind):
                original = prior._aggregate(self.graph, values, kind, prior._RandomState.from_seed(10))
                actual = prior._aggregate(graph, permuted, kind, prior._RandomState.from_seed(10))
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual, original[order], rtol=1e-5, atol=1e-5)

    def test_attention_uses_only_neighbors_and_self(self):
        # Node 3 is isolated. Changing it cannot affect nodes 0, 1, or 2.
        changed = self.values.clone()
        changed[3] = 1000
        first = prior._attention_neighbors(self.graph, self.values, prior._RandomState.from_seed(4))
        second = prior._attention_neighbors(self.graph, changed, prior._RandomState.from_seed(4))
        torch.testing.assert_close(first[:3], second[:3], rtol=0, atol=0)


class AttributeTests(PriorTestCase):
    def test_target_coordinate_is_never_selected_as_feature(self):
        values = torch.arange(60, dtype=torch.float32).reshape(3, 20)
        for seed in range(10):
            for effect in (False, True):
                for clique in (False, True):
                    features, target = prior._choose_observed(
                        [values], 6, effect, clique, prior._RandomState.from_seed(seed)
                    )
                    self.assertFalse(any(torch.equal(features[:, i], target) for i in range(6)))

    def test_multiclass_encoding_preserves_ties_and_requested_count(self):
        raw = torch.arange(100, dtype=torch.float32).repeat_interleave(2)
        labels = prior._make_target(raw, "multiclass", 32, prior._RandomState.from_seed(5))
        self.assertEqual(labels.dtype, torch.int64)
        self.assertEqual(labels.unique(sorted=True).tolist(), list(range(32)))
        self.assertTrue(torch.equal(labels[::2], labels[1::2]))

    def test_constant_target_is_rejected(self):
        with self.assertRaises(prior._InvalidSample):
            prior._make_target(torch.ones(50), "binclass", 2, prior._RandomState.from_seed(0))

    def test_regression_conversion_does_not_fit_statistics_on_query(self):
        raw = torch.tensor([1.0, 3.0, 10000.0])
        converted = prior._make_target(raw, "regression", None, prior._RandomState.from_seed(0))
        torch.testing.assert_close(converted, raw, rtol=0, atol=0)

    def test_context_covers_all_classes_with_rare_classes(self):
        labels = torch.tensor([0] * 30 + [1] * 10 + [2])
        order = prior._split_order(labels, 5, "multiclass", prior._RandomState.from_seed(0))
        self.assertEqual(order.unique().numel(), len(labels))
        self.assertEqual(labels[order[:5]].unique(sorted=True).tolist(), [0, 1, 2])

    def test_feature_scaling_uses_context_and_drops_context_constants(self):
        config = prior.PriorConfig(n_features=(1, 3), categorical_probability=0)
        features = torch.tensor([[1.0, 5.0], [3.0, 5.0], [100.0, 8.0]])
        first, cardinalities = prior._process_features(features, 2, config, prior._RandomState.from_seed(0))
        changed = features.clone()
        changed[2] = torch.tensor([1e20, -100.0])
        second, _ = prior._process_features(changed, 2, config, prior._RandomState.from_seed(0))
        self.assertEqual(tuple(first.shape), (3, 1))
        torch.testing.assert_close(first[:2, 0], torch.tensor([-1.0, 1.0]))
        torch.testing.assert_close(first[:2], second[:2], rtol=0, atol=0)
        self.assertEqual(cardinalities.tolist(), [0])


class SamplingTests(PriorTestCase):
    def setUp(self):
        self.config = prior.PriorConfig(
            n_nodes=64, n_features=(2, 8), n_groups=(2, 4), avg_degree=(4, 12),
            context_fraction=(0.3, 0.5), max_num_classes=5,
        )

    def assert_valid_graph(self, task):
        n_nodes = len(task.x)
        self.assertLessEqual(n_nodes, self.config.n_nodes)
        pairs = set(map(tuple, task.edge_index.t().tolist()))
        self.assertEqual(len(pairs), task.edge_index.shape[1])
        self.assertTrue(all(s != t and (t, s) in pairs for s, t in pairs))
        self.assertTrue(all(0 <= s < n_nodes and 0 <= t < n_nodes for s, t in pairs))
        _, retained = prior._largest_component(task.edge_index, n_nodes)
        self.assertEqual(len(retained), n_nodes)

    def test_end_to_end_all_task_types(self):
        for task_type in prior.TASK_TYPES:
            for seed in (0, 11, 42):
                with self.subTest(task_type=task_type, seed=seed):
                    task = prior.sample_task(replace(self.config, task_type=task_type), seed=seed)
                    self.assert_valid_graph(task)
                    self.assertEqual(task.task_type, task_type)
                    self.assertTrue(torch.isfinite(task.x).all())
                    self.assertTrue(torch.isfinite(task.y).all())
                    self.assertFalse(task.x.requires_grad)
                    self.assertEqual(task.x.device.type, "cpu")
                    self.assertGreaterEqual(task.sep, 2)
                    self.assertLess(task.sep, len(task.x))
                    self.assertTrue(torch.equal(task.context_y, task.y[:task.sep]))
                    self.assertTrue(torch.equal(task.query_y, task.y[task.sep:]))
                    if task_type == "regression":
                        self.assertAlmostEqual(task.context_y.mean().item(), 0, places=5)
                        self.assertAlmostEqual(task.context_y.std(correction=0).item(), 1, places=5)
                    else:
                        self.assertEqual(task.y.dtype, torch.int64)
                        self.assertEqual(task.context_y.unique(sorted=True).tolist(), list(range(task.num_classes)))
                    for column, count in enumerate(task.feature_cardinalities):
                        if count:
                            values = task.x[:, column]
                            self.assertTrue(torch.equal(values, values.round()))
                            self.assertEqual(values.unique(sorted=True).tolist(), list(range(int(count))))

    def test_split_uses_retained_component_size(self):
        edges = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
        config = replace(self.config, task_type="regression", context_fraction=(0.5, 0.5))
        features = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        labels = torch.tensor([0.0, 1.0, 3.0, 6.0])
        with patch.object(prior, "_sample_graph", return_value=(edges, 4, {})), patch.object(
            prior, "_sample_attributes", return_value=(features, labels, {})
        ):
            task = prior.sample_task(config, seed=1)
        self.assertEqual(len(task.x), 4)
        self.assertEqual(task.sep, 2)

    def test_reproducibility_and_global_rng_isolation(self):
        torch_before = torch.get_rng_state().clone()
        numpy_before = np.random.get_state()
        python_before = random.getstate()
        first = prior.sample_task(self.config, seed=42)
        second = prior.sample_task(self.config, seed=42)
        self.assert_same_task(first, second)
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        self.assertEqual(python_before, random.getstate())

    def test_bounded_retries_and_programming_errors(self):
        config = replace(self.config, task_type="regression", max_attempts=3)
        with patch.object(prior, "_sample_graph", side_effect=prior._InvalidSample("too small")) as sampler:
            with self.assertRaisesRegex(prior.SamplingError, "regression.*seed=7.*3 attempts.*too small"):
                prior.sample_task(config, seed=7)
            self.assertEqual(sampler.call_count, 3)
        with patch.object(prior, "_sample_graph", side_effect=RuntimeError("bug")) as sampler:
            with self.assertRaisesRegex(RuntimeError, "bug"):
                prior.sample_task(config, seed=7)
            self.assertEqual(sampler.call_count, 1)

    def test_loader_resume_and_reiteration_preserve_tasks(self):
        options = dict(config=self.config, num_tasks=4, seed=23)
        torch_before = torch.get_rng_state().clone()
        loader = prior.make_dataloader(**options, num_workers=0)
        direct = list(loader)
        for first, second in zip(direct, loader):
            self.assert_same_task(first, second)
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        resumed = list(prior.make_dataloader(self.config, num_tasks=2, seed=23, start_index=2))
        for first, second in zip(direct[2:], resumed):
            self.assert_same_task(first, second)
        self.assertEqual([task.metadata["task_index"] for task in resumed], [2, 3])
        self.assertEqual(len({task.metadata["seed"] for task in direct}), 4)

    def test_worker_count_and_resume_preserve_tasks(self):
        # PyTorch's default CPU tensor transfer needs a local socket. Check it
        # before starting workers, otherwise restricted sandboxes can hang.
        try:
            listener = Listener()
        except PermissionError as error:
            self.skipTest(f"Local sockets required for PyTorch worker transfer are blocked: {error}")
        else:
            listener.close()
        options = dict(config=self.config, num_tasks=4, seed=23)
        torch_before = torch.get_rng_state().clone()
        direct = list(prior.make_dataloader(**options, num_workers=0))
        parallel_loader = prior.make_dataloader(**options, num_workers=2)
        parallel_loader.timeout = 30
        parallel = list(parallel_loader)
        self.assertEqual(len(direct), len(parallel))
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        for first, second in zip(direct, parallel):
            self.assert_same_task(first, second)
        resumed_loader = prior.make_dataloader(self.config, num_tasks=2, seed=23, start_index=2, num_workers=1)
        resumed_loader.timeout = 30
        for first, second in zip(direct[2:], resumed_loader):
            self.assert_same_task(first, second)

    def test_invalid_configuration_fails_before_sampling(self):
        for settings in (
            {"n_nodes": 1}, {"n_features": (8, 2)}, {"context_fraction": (0.1, 1.0)},
            {"max_attempts": 0}, {"task_type": "unknown"}, {"avg_degree": (0, 10)},
            {"scm_width_mean": (0, 10)}, {"scm_depth_mean": (4, 2)}, {"scm_noise_mean": (0.1, float("inf"))},
            {"scm_graph_ratio": (-0.1, 1)}, {"scm_graph_ratio": (0.7, 0.2)},
            {"scm_dropout_scale": 1.0}, {"scm_dropout_beta": (0, 1)},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                prior.PriorConfig(**settings)
        with self.assertRaises(ValueError):
            prior.make_dataloader(self.config, num_tasks=1, num_workers=-1)

    def test_scm_configuration_can_disable_graph_mixing(self):
        config = replace(self.config, task_type="regression", scm_graph_ratio=(0, 0),
                         scm_dropout_scale=0, scm_depth_mean=(1, 1), scm_width_mean=(5, 5))
        for seed in range(3):
            task = prior.sample_task(config, seed=seed)
            self.assertEqual(task.metadata["scm"]["graph_ratio"], 0)
            self.assertTrue(torch.isfinite(task.x).all())
            self.assertTrue(torch.isfinite(task.y).all())


if __name__ == "__main__":
    unittest.main()
