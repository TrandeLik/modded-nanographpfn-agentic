"""Parameter ownership and matrix-update semantics for Muon/schedule-free AdamW."""

import contextlib
import io
import unittest

import torch

from nanographpfn import NanoGraphPFNModel
from train_nano import Config, Muon, make_optimizers, parse_args, zeropower_via_newtonschulz5


class OptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_parameter_ownership_is_complete_and_qkv_is_explicit(self):
        model = NanoGraphPFNModel(num_layers=2, num_heads=2, embed_dim=8, hidden_dim=24, max_classes=3)
        optimizers = make_optimizers(model, Config())
        muon = [parameter for group in optimizers.muon.param_groups for parameter in group["params"]]
        adamw = [parameter for group in optimizers.adamw.param_groups for parameter in group["params"]]
        self.assertFalse(set(muon) & set(adamw))
        self.assertEqual(set(muon) | set(adamw), set(model.parameters()))
        self.assertEqual(len(muon) + len(adamw), len(list(model.parameters())))
        qkv = {parameter for group in optimizers.muon.param_groups if group["is_qkv"] for parameter in group["params"]}
        self.assertEqual(len(qkv), 3 * model.num_layers)
        for layer in model.layers:
            for projection in (layer.feature_qkv, layer.context_qkv, layer.graph_attention.qkv):
                self.assertIn(projection.weight, qkv)
                self.assertIn(projection.bias, set(adamw))
            for projection in (layer.feature_output, layer.context_output, layer.graph_attention.output):
                self.assertIn(projection.weight, set(muon))
                self.assertNotIn(projection.weight, qkv)
                self.assertIn(projection.bias, set(adamw))
            self.assertEqual(layer.mlp_in.weight.shape, layer.feature_qkv.weight.shape)
            self.assertIn(layer.mlp_in.weight, set(muon))
            self.assertNotIn(layer.mlp_in.weight, qkv)
            self.assertIn(layer.feature_norm.weight, set(adamw))
        self.assertIn(model.feature_encoder.weight, set(adamw))
        self.assertIn(model.regression_decoder[0].weight, set(adamw))

    def test_same_shape_mlp_and_qkv_receive_different_correct_updates(self):
        torch.manual_seed(32)
        initial = torch.randn(24, 8)
        gradient = torch.randn_like(initial)
        mlp = torch.nn.Parameter(initial.clone())
        qkv = torch.nn.Parameter(initial.clone())
        mlp.grad, qkv.grad = gradient.clone(), gradient.clone()
        optimizer = Muon([mlp, qkv], qkv_params=[qkv], lr=0.01, momentum=0, backend_steps=5)
        whole_update = zeropower_via_newtonschulz5(gradient, steps=5) * (24 ** 0.5)
        qkv_update = torch.cat([zeropower_via_newtonschulz5(part, steps=5)
                               for part in gradient.chunk(3)]) * (8 ** 0.5)
        optimizer.step()
        torch.testing.assert_close(mlp, initial - 0.01 * whole_update)
        torch.testing.assert_close(qkv, initial - 0.01 * qkv_update)
        self.assertFalse(torch.allclose(mlp, qkv))

    def test_non_nesterov_momentum_carries_update_through_zero_gradient(self):
        for is_qkv in (False, True):
            with self.subTest(is_qkv=is_qkv):
                parameter = torch.nn.Parameter(torch.zeros(6, 2))
                optimizer = Muon([parameter], qkv_params=[parameter] if is_qkv else [],
                                 lr=0.01, momentum=0.9, nesterov=False)
                parameter.grad = torch.arange(1, 13, dtype=parameter.dtype).reshape_as(parameter)
                optimizer.step()
                before = parameter.detach().clone()
                expected_buffer = parameter.grad * 0.9
                if is_qkv:
                    update = torch.cat([zeropower_via_newtonschulz5(part, steps=5)
                                        for part in expected_buffer.chunk(3)])
                    scale = 2 ** 0.5
                else:
                    update = zeropower_via_newtonschulz5(expected_buffer, steps=5)
                    scale = 6 ** 0.5
                parameter.grad.zero_()
                optimizer.step()
                torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"], expected_buffer)
                torch.testing.assert_close(parameter, before - 0.01 * scale * update)
                self.assertFalse(torch.equal(parameter, before))
                self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0)

    def test_descriptive_architecture_flags_and_no_short_aliases(self):
        _, config = parse_args(["--num-layers", "2", "--num-heads", "2", "--embed-dim", "16",
                                "--hidden-dim", "48", "--max-classes", "8"])
        self.assertEqual((config.num_layers, config.num_heads, config.embed_dim, config.hidden_dim, config.max_classes),
                         (2, 2, 16, 48, 8))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--h", "48"])


if __name__ == "__main__":
    unittest.main()
