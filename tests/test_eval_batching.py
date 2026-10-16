"""Evaluation batching preserves full context/graph semantics and training gradients."""

import copy
import importlib.util
import unittest
import weakref
from unittest.mock import patch

import torch

import nanographpfn as model_module
from nanographpfn import NanoGraphPFNModel, TransformerEncoderLayer, make_neighbor_mask


def fixture():
    torch.manual_seed(31)
    model = NanoGraphPFNModel(num_layers=2, num_heads=2, embed_dim=16, hidden_dim=32, max_classes=3).eval()
    for layer in model.layers:
        torch.nn.init.normal_(layer.graph_attention.output.weight, std=0.1)
        torch.nn.init.normal_(layer.graph_mlp[-1].weight, std=0.1)
    x = torch.randn(19, 6)
    # Directed edges crossing node batch boundaries, plus two isolated nodes.
    nodes = torch.arange(17)
    edges = torch.stack((nodes.repeat(2), torch.cat(((nodes + 1) % 17, (nodes + 5) % 17))))
    return model, x, edges


class EvaluationBatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def check_predictions(self, dgl=False):
        model, x, edges = fixture()
        original_x = x.clone()
        for autocast in (False, True):
            for task, y in (("binclass", torch.arange(7) % 2),
                            ("multiclass", torch.arange(7) % 3),
                            ("regression", torch.linspace(-1, 1, 7))):
                with self.subTest(task=task, autocast=autocast), torch.inference_mode(), \
                        torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast), \
                        patch.object(model_module, "MAX_SDPA_GRAPH_SIZE", 8 if dgl else 2048):
                    kwargs = {"num_classes": 2 if task == "binclass" else 3}
                    with patch.object(model_module, "EVAL_NODE_BATCH_SIZE", 0), \
                            patch.object(model_module, "EVAL_TOKEN_BATCH_SIZE", 0):
                        expected = model(x, y, edges, task, **kwargs)
                    # Non-divisible batches exercise the final partial slices.
                    with patch.object(model_module, "EVAL_NODE_BATCH_SIZE", 5), \
                            patch.object(model_module, "EVAL_TOKEN_BATCH_SIZE", 3), \
                            patch.object(model_module, "make_dgl_graph",
                                         wraps=model_module.make_dgl_graph) as build:
                        actual = model(x, y, edges, task, **kwargs)
                    self.assertEqual(build.call_count, int(dgl))
                    self.assertEqual(actual.dtype, expected.dtype)
                    tolerance = dict(atol=2e-2, rtol=2e-2) if autocast else dict(atol=2e-6, rtol=2e-5)
                    torch.testing.assert_close(actual, expected, **tolerance)
        torch.testing.assert_close(x, original_x, atol=0, rtol=0)

    def test_dense_predictions_match_unbatched_for_all_tasks_and_autocast(self):
        self.check_predictions()

    @unittest.skipUnless(importlib.util.find_spec("dgl"), "DGL is not installed")
    def test_dgl_predictions_match_unbatched_for_all_tasks_and_autocast(self):
        self.check_predictions(dgl=True)

    def test_training_and_gradient_enabled_evaluation_remain_unbatched(self):
        for training, gradients in ((True, True), (False, True), (True, False)):
            with self.subTest(training=training, gradients=gradients):
                model, x, edges = fixture()
                model.train(training)
                reference = copy.deepcopy(model)
                y = torch.linspace(-1, 1, 7)
                actual_x = x.clone().requires_grad_(gradients)
                reference_x = x.clone().requires_grad_(gradients)
                shapes = []
                handles = [module.register_forward_pre_hook(
                    lambda module, args: shapes.append(tuple(args[0].shape)))
                    for module in model.modules() if isinstance(module, torch.nn.LayerNorm)]
                with torch.set_grad_enabled(gradients):
                    with patch.object(model_module, "EVAL_NODE_BATCH_SIZE", 0), \
                            patch.object(model_module, "EVAL_TOKEN_BATCH_SIZE", 0):
                        expected = reference(reference_x, y, edges, "regression")
                    with patch.object(model_module, "EVAL_NODE_BATCH_SIZE", 2), \
                            patch.object(model_module, "EVAL_TOKEN_BATCH_SIZE", 1):
                        actual = model(actual_x, y, edges, "regression")
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    if gradients:
                        actual_grads = torch.autograd.grad(actual.square().sum(),
                                                          (actual_x, *model.parameters()), allow_unused=True)
                        expected_grads = torch.autograd.grad(expected.square().sum(),
                                                            (reference_x, *reference.parameters()), allow_unused=True)
                        for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
                            if expected_grad is None:
                                self.assertIsNone(actual_grad)
                            else:
                                torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)
                for handle in handles:
                    handle.remove()
                # Five norms per block, plus the regression decoder's norm.
                self.assertEqual(shapes, [(19, 7, 16)] * (5 * model.num_layers) + [(12, 32)])

    def test_avazu_normalization_and_projections_are_bounded_before_allocation(self):
        layer = TransformerEncoderLayer(num_heads=4, embed_dim=256, hidden_dim=768).eval().to("meta")
        node_operations = ("feature_norm", "feature_qkv", "feature_output", "mlp_norm", "mlp_in",
                           "graph_mlp_norm", "graph_mlp.0")
        token_operations = ("context_norm", "context_qkv", "context_output", "graph_attention_norm",
                            "graph_attention.qkv")
        shapes = {name: [] for name in node_operations + token_operations}
        handles = [module.register_forward_pre_hook(
            lambda module, args, name=name: shapes[name].append(tuple(args[0].shape)))
            for name, module in layer.named_modules() if name in shapes]
        x = torch.empty(76269, 65, 256, device="meta")
        edges = torch.empty(2, 0, dtype=torch.long, device="meta")
        with torch.inference_mode(), \
                patch.object(model_module.F, "scaled_dot_product_attention",
                             side_effect=lambda q, k, v: torch.empty_like(q)), \
                patch.object(model_module, "dgl_neighbor_attention", side_effect=lambda q, k, v, graph: v):
            result = layer(x, 7626, edges, None, graph=object())
        self.assertEqual(result.shape, x.shape)
        for handle in handles:
            handle.remove()
        for name in node_operations:
            self.assertGreater(len(shapes[name]), 1)
            self.assertEqual(sum(shape[0] for shape in shapes[name]), 76269)
            self.assertTrue(all(n <= model_module.EVAL_NODE_BATCH_SIZE and (t, d) == (65, 256)
                                for n, t, d in shapes[name]))
        for name in token_operations:
            self.assertGreater(len(shapes[name]), 1)
            self.assertEqual(sum(shape[1] for shape in shapes[name]), 65)
            self.assertTrue(all(t <= model_module.EVAL_TOKEN_BATCH_SIZE and (n, d) == (76269, 256)
                                for n, t, d in shapes[name]))

    def test_context_temporaries_are_released_before_feedforward(self):
        layer = TransformerEncoderLayer(num_heads=2, embed_dim=16, hidden_dim=32).eval()
        x = torch.randn(19, 7, 16)
        edges = torch.tensor([[0, 1], [1, 0]])
        temporaries = []
        handle = layer.context_qkv.register_forward_hook(
            lambda module, args, output: temporaries.append(weakref.ref(output)))
        sdpa = model_module.F.scaled_dot_product_attention

        def attention(q, k, v, **kwargs):
            output = sdpa(q, k, v, **kwargs)
            if q.shape == (7, 2, 19, 8) and k.shape[-2] == 7:
                temporaries.append(weakref.ref(output))
            return output

        def check_released(module, args):
            self.assertEqual(len(temporaries), 2)
            self.assertTrue(all(ref() is None for ref in temporaries))

        feedforward_handle = layer.mlp_norm.register_forward_pre_hook(check_released)
        with torch.inference_mode(), patch.object(model_module, "EVAL_NODE_BATCH_SIZE", 0), \
                patch.object(model_module, "EVAL_TOKEN_BATCH_SIZE", 0), \
                patch.object(model_module.F, "scaled_dot_product_attention", side_effect=attention):
            layer(x, 7, edges, make_neighbor_mask(edges, 19))
        handle.remove()
        feedforward_handle.remove()

    @unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("dgl"), "CUDA/DGL unavailable")
    def test_cuda_batching_reduces_peak_memory_and_preserves_predictions(self):
        torch.manual_seed(31)
        model = NanoGraphPFNModel(num_layers=2, num_heads=4, embed_dim=64, hidden_dim=192, max_classes=3).eval().cuda()
        for layer in model.layers:
            torch.nn.init.normal_(layer.graph_attention.output.weight, std=0.05)
            torch.nn.init.normal_(layer.graph_mlp[-1].weight, std=0.05)
        n = 8193
        x, y = torch.randn(n, 16, device="cuda"), torch.randn(128, device="cuda")
        nodes = torch.arange(n, device="cuda")
        edges = torch.stack((nodes.repeat(2), torch.cat(((nodes + 1) % n, (nodes + 5) % n))))
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        def measure(node_batch, token_batch):
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype), \
                    patch.object(model_module, "EVAL_NODE_BATCH_SIZE", node_batch), \
                    patch.object(model_module, "EVAL_TOKEN_BATCH_SIZE", token_batch):
                torch.cuda.synchronize()
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                result = model(x, y, edges, "regression").float().cpu()
                torch.cuda.synchronize()
                return result, torch.cuda.max_memory_allocated() - baseline

        # Warm up both execution patterns before comparing live tensor peaks.
        measure(0, 0)
        measure(model_module.EVAL_NODE_BATCH_SIZE, model_module.EVAL_TOKEN_BATCH_SIZE)
        expected, full_peak = measure(0, 0)
        actual, batched_peak = measure(model_module.EVAL_NODE_BATCH_SIZE, model_module.EVAL_TOKEN_BATCH_SIZE)
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
        self.assertLess(batched_peak, full_peak)


if __name__ == "__main__":
    unittest.main()
