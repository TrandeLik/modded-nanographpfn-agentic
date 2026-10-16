"""DGL parity tests run against real kernels when the project env is installed."""

import importlib.util
import sys
import unittest
from unittest.mock import patch

import torch

import nanographpfn as model_module
from nanographpfn import (
    GraphAttention,
    NanoGraphPFNModel,
    dgl_neighbor_attention,
    make_dgl_graph,
)


def reference_neighbor_attention(q, k, v, edges):
    """Small test oracle; repeated edges contribute separately to the softmax."""
    outputs = []
    for node in range(len(q)):
        neighbors = edges[0, edges[1] == node]
        if len(neighbors):
            scores = (q[node].float() * k[neighbors].float()).sum(-1) / q.shape[-1] ** 0.5
            outputs.append((scores.softmax(0)[..., None] * v[neighbors].float()).sum(0).to(v.dtype))
        else:
            outputs.append(q[node] * 0 + k[node] * 0 + v[node] * 0)
    return torch.stack(outputs)


def model_fixture():
    torch.manual_seed(21)
    model = NanoGraphPFNModel(num_layers=2, num_heads=2, embed_dim=8, hidden_dim=16, max_classes=3).eval()
    for block in model.layers:
        torch.nn.init.normal_(block.graph_attention.output.weight, std=0.1)
        torch.nn.init.normal_(block.graph_mlp[-1].weight, std=0.1)
    x = torch.randn(9, 3)
    edges = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 0]])
    return model, x, edges


class GraphBackendRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_small_graph_training_and_inference_do_not_import_dgl(self):
        model, x, edges = model_fixture()
        with patch.dict(sys.modules, {"dgl": None}):
            expected = model(x, torch.tensor([0, 1, 2]), edges, "multiclass")
            expected.square().sum().backward()
            with torch.inference_mode():
                actual = model(x, torch.tensor([0, 1, 2]), edges, "multiclass")
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    def test_missing_dgl_is_explicit_for_large_graphs(self):
        model, x, edges = model_fixture()
        with patch.dict(sys.modules, {"dgl": None}), patch.object(model_module, "MAX_SDPA_GRAPH_SIZE", 8):
            with self.assertRaisesRegex(RuntimeError, "Large-graph attention requires DGL.*uv sync"):
                model(x, torch.tensor([0, 1, 2]), edges, "multiclass")

    def test_graph_attention_switches_only_above_existing_cutoff(self):
        module = GraphAttention(4, 1)
        edges = torch.tensor([[0, 1], [1, 0]])
        graph = object()
        with patch.object(model_module, "make_dgl_graph", return_value=graph) as build, \
                patch.object(model_module, "dgl_neighbor_attention", side_effect=lambda q, k, v, g: v) as attend:
            module(torch.randn(2048, 1, 4), edges)
            build.assert_not_called()
            attend.assert_not_called()
            module(torch.randn(2049, 1, 4), edges)
            build.assert_called_once_with(edges, 2049)
            self.assertIs(attend.call_args.args[-1], graph)

    def test_large_graph_topology_is_reused_across_layers(self):
        # A torch stand-in tests dispatch without requiring DGL. Kernel parity is
        # checked separately below with real DGL, never with this stand-in.
        model, x, edges = model_fixture()
        y = torch.tensor([0, 1, 2])
        expected = model(x, y, edges, "multiclass")
        graph = object()

        def attend(q, k, v, topology):
            self.assertIs(topology, graph)
            return reference_neighbor_attention(q, k, v, edges)

        with patch.object(model_module, "MAX_SDPA_GRAPH_SIZE", 8), \
                patch.object(model_module, "make_dgl_graph", return_value=graph) as build, \
                patch.object(model_module, "dgl_neighbor_attention", side_effect=attend) as attention:
            actual = model(x, y, edges, "multiclass")
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            build.assert_called_once_with(edges, len(x))
            self.assertEqual(attention.call_count, model.num_layers)


class DGLAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("dgl") is None:
            raise unittest.SkipTest("DGL is not installed; run uv sync for real-kernel tests")
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(12)

    def check_values_and_gradients(self, device, autocast=False):
        module = GraphAttention(12, 3).to(device)
        torch.nn.init.normal_(module.output.weight, std=0.1)
        # Asymmetric edges, an explicit self loop, and trailing isolated nodes.
        edges = torch.tensor([[0, 1, 2, 2, 3, 3], [1, 2, 0, 3, 2, 3]], device=device)
        graph = make_dgl_graph(edges, 6)
        self.assertEqual(graph.num_nodes(), 6)
        self.assertEqual(graph.num_edges(), edges.shape[1])
        src, dst = graph.edges()
        torch.testing.assert_close(torch.stack((src, dst)), edges)
        mask = torch.zeros(6, 6, dtype=torch.bool, device=device)
        mask[edges[1], edges[0]] = True
        isolated = ~mask.any(1)
        indices = torch.where(isolated)[0]
        mask[indices, indices] = True
        x = torch.randn(6, 3, 12, device=device, requires_grad=True)
        inputs = (x, *module.parameters())
        with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
            dense = module(x, edges, (mask, isolated))
            actual = module(x, edges, graph=graph)
        tolerance = dict(atol=2e-2, rtol=2e-2) if autocast else dict(atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(actual, dense, **tolerance)
        actual_grads = torch.autograd.grad(actual.float().square().sum(), inputs)
        dense_grads = torch.autograd.grad(dense.float().square().sum(), inputs)
        for actual_grad, dense_grad in zip(actual_grads, dense_grads, strict=True):
            torch.testing.assert_close(actual_grad, dense_grad, **tolerance)

    def test_cpu_values_and_gradients_match_dense(self):
        self.check_values_and_gradients("cpu")

    def test_cpu_autocast_values_and_gradients(self):
        self.check_values_and_gradients("cpu", autocast=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_values_and_gradients(self):
        self.check_values_and_gradients("cuda")
        if torch.cuda.is_bf16_supported():
            self.check_values_and_gradients("cuda", autocast=True)

    def test_empty_graph_and_duplicate_edges(self):
        for edges in (torch.empty(2, 0, dtype=torch.long),
                      torch.tensor([[0, 0, 1, 2], [2, 2, 2, 0]])):
            with self.subTest(edges=edges.shape[1]):
                q, k, v = (torch.randn(5, 3, 2, 4, requires_grad=True) for _ in range(3))
                actual = dgl_neighbor_attention(q, k, v, make_dgl_graph(edges, 5))
                expected = reference_neighbor_attention(q, k, v, edges)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                actual_grads = torch.autograd.grad(actual.square().sum(), (q, k, v))
                expected_grads = torch.autograd.grad(expected.square().sum(), (q, k, v))
                for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
                    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-5)

    def test_dgl_predictions_match_dense_for_all_tasks(self):
        model, x, edges = model_fixture()
        for task, y in (("binclass", torch.tensor([0, 1, 0])),
                        ("multiclass", torch.tensor([0, 1, 2])),
                        ("regression", torch.tensor([-1., 0., 1.]))):
            with self.subTest(task=task):
                kwargs = {"num_classes": 2 if task == "binclass" else 3}
                expected = model(x, y, edges, task, **kwargs)
                with patch.object(model_module, "MAX_SDPA_GRAPH_SIZE", 8):
                    actual = model(x, y, edges, task, **kwargs)
                    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                    actual.square().sum().backward()
                    self.assertGreater(float(model.layers[0].graph_attention.qkv.weight.grad.norm()), 0)
                    model.zero_grad(set_to_none=True)


if __name__ == "__main__":
    unittest.main()
