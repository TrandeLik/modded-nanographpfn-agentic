"""Numerical and lifecycle checks for optional execution accelerators."""
import copy
import unittest

import torch

from nanographpfn import NanoGraphPFNModel
from nanoprior import PriorConfig, make_dataloader, sample_task
from train_nano import Muon, task_loss, zeropower_via_newtonschulz5


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
class AccelerationTests(unittest.TestCase):
    def test_batched_muon_independent_norms_qkv_and_skipped_gradients(self):
        torch.manual_seed(37)
        shapes = [(16, 8), (16, 8), (8, 16), (8, 16), (24, 8), (24, 8)]
        for nesterov in (True, False):
            left = [torch.nn.Parameter(torch.randn(shape, device='cuda')) for shape in shapes]
            right = [torch.nn.Parameter(p.detach().clone()) for p in left]
            reference = Muon(left, qkv_params=left[-2:], cuda_graph=False, batched=False, nesterov=nesterov)
            accelerated = Muon(right, qkv_params=right[-2:], cuda_graph=True, batched=True, nesterov=nesterov)
            for step in range(3):
                for index, (p, q) in enumerate(zip(left, right)):
                    # Different scales detect accidental normalization across matrices.
                    gradient = torch.randn_like(p) * (10 ** (index - 2))
                    p.grad = None if step == 1 and index == 1 else gradient.clone()
                    q.grad = None if p.grad is None else gradient.clone()
                before_left = [p.detach().clone() for p in left]
                before_right = [p.detach().clone() for p in right]
                reference.step()
                accelerated.step()
                errors, rounding_errors = [], []
                for index, (p, q, a, b) in enumerate(zip(left, right, before_left, before_right)):
                    expected, actual = p - a, q - b
                    if expected.norm() > 0:
                        buffer = reference.state[p]['momentum_buffer']
                        effective = p.grad.add(buffer, alpha=0.95) if nesterov else buffer
                        if index >= len(left) - 2:
                            raw = torch.cat([zeropower_via_newtonschulz5(g, steps=5)
                                             for g in effective.chunk(3)])
                            scale = p.shape[1] ** 0.5
                        else:
                            raw = zeropower_via_newtonschulz5(effective, steps=5)
                            scale = max(p.shape) ** 0.5
                        raw_delta = raw * (-0.0003 * scale)
                        errors.append((actual - expected).square().sum())
                        rounding_errors.append((raw_delta - expected).square().sum())
                    else:
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(accelerated.state[q]['momentum_buffer'],
                                               reference.state[p]['momentum_buffer'], rtol=0, atol=0)
                # Batch GEMMs choose a different BF16 accumulation path. Require
                # less change than the existing eager-vs-compiled Muon rounding,
                # rather than widening an arbitrary fixed tolerance.
                self.assertLess(float(torch.stack(errors).sum()), float(torch.stack(rounding_errors).sum()))

    def test_pinned_tasks_async_transfer_preserve_values_and_query_isolation(self):
        prior = PriorConfig(n_nodes=32, n_features=(4, 4), task_type='regression')
        task = next(iter(make_dataloader(prior, num_tasks=1, seed=12, pin_memory=True)))
        for tensor in (task.x, task.y, task.edge_index, task.feature_cardinalities):
            self.assertTrue(tensor.is_pinned())
        model = NanoGraphPFNModel(num_layers=1, num_heads=2, embed_dim=8, hidden_dim=16).cuda()
        seen = []
        handle = model.register_forward_pre_hook(lambda m, args: seen.append(args[1].detach().clone()))
        blocking = task_loss(model, task, torch.device('cuda'), async_transfer=False)
        asynchronous = task_loss(model, task, torch.device('cuda'), async_transfer=True)
        handle.remove()
        torch.testing.assert_close(asynchronous, blocking, rtol=0, atol=0)
        for observed in seen:
            torch.testing.assert_close(observed.cpu(), task.context_y, rtol=0, atol=0)

    def test_muon_capture_replay_skip_hyperparameters_and_reload(self):
        torch.manual_seed(18)
        for nesterov in (True, False):
            original = [torch.randn(16, 8, device='cuda'), torch.randn(24, 8, device='cuda')]
            eager_params = [torch.nn.Parameter(p.clone()) for p in original]
            graph_params = [torch.nn.Parameter(p.clone()) for p in original]
            eager = Muon(eager_params, qkv_params=[eager_params[1]], cuda_graph=False, nesterov=nesterov)
            captured = Muon(graph_params, qkv_params=[graph_params[1]], cuda_graph=True, nesterov=nesterov)
            for step in range(6):
                if step == 3:
                    for optimizer in (eager, captured):
                        for group in optimizer.param_groups:
                            group['lr'] *= 0.5
                if step == 4:
                    captured.load_state_dict(copy.deepcopy(eager.state_dict()))
                for index, (left, right) in enumerate(zip(eager_params, graph_params)):
                    gradient = torch.randn_like(left)
                    left.grad = None if step == 2 and index == 1 else gradient.clone()
                    right.grad = None if left.grad is None else gradient.clone()
                eager.step()
                captured.step()
                for left, right in zip(eager_params, graph_params):
                    torch.testing.assert_close(right, left, rtol=1e-6, atol=1e-7)
                    torch.testing.assert_close(captured.state[right]['momentum_buffer'],
                                               eager.state[left]['momentum_buffer'], rtol=0, atol=0)
                self.assertEqual(captured.state_dict().keys(), eager.state_dict().keys())

    def test_compiled_blocks_values_gradients_shapes_and_state_names(self):
        torch.manual_seed(24)
        eager = NanoGraphPFNModel(num_layers=1, num_heads=2, embed_dim=16, hidden_dim=32).cuda()
        # Nonzero adapters exercise the whole graph path in both directions.
        torch.nn.init.normal_(eager.layers[0].graph_attention.output.weight, std=0.1)
        torch.nn.init.normal_(eager.layers[0].graph_mlp[-1].weight, std=0.1)
        compiled = copy.deepcopy(eager)
        compiled.compile_training_blocks()
        self.assertEqual(list(eager.state_dict()), list(compiled.state_dict()))
        outputs = []
        handles = [model.register_forward_hook(lambda module, args, output: outputs.append(output.detach()))
                   for model in (eager, compiled)]
        for nodes, features, kind in ((32, 4, 'binclass'), (40, 7, 'regression'), (48, 5, 'multiclass')):
            task = sample_task(PriorConfig(n_nodes=nodes, n_features=(features, features), task_type=kind), seed=6)
            eager.zero_grad(set_to_none=True)
            compiled.zero_grad(set_to_none=True)
            expected = task_loss(eager, task, torch.device('cuda'), amp=False)
            actual = task_loss(compiled, task, torch.device('cuda'), amp=False)
            torch.testing.assert_close(outputs[-1], outputs[-2], rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            expected.backward()
            actual.backward()
            for left, right in zip(eager.parameters(), compiled.parameters()):
                if left.grad is None:
                    self.assertIsNone(right.grad)
                else:
                    torch.testing.assert_close(right.grad, left.grad, rtol=3e-4, atol=3e-5)
        # Actual training uses BF16. Compare gradient vectors with a relative
        # norm bound so near-zero coordinates do not dominate the comparison.
        eager.zero_grad(set_to_none=True)
        compiled.zero_grad(set_to_none=True)
        expected = task_loss(eager, task, torch.device('cuda'))
        actual = task_loss(compiled, task, torch.device('cuda'))
        torch.testing.assert_close(outputs[-1], outputs[-2], rtol=0.02, atol=0.005)
        torch.testing.assert_close(actual, expected, rtol=0.005, atol=0.003)
        expected.backward()
        actual.backward()
        left = torch.cat([p.grad.flatten() for p in eager.parameters() if p.grad is not None])
        right = torch.cat([p.grad.flatten() for p in compiled.parameters() if p.grad is not None])
        self.assertLess(float((right - left).norm() / left.norm()), 0.02)
        for handle in handles:
            handle.remove()
        compiled.eval()
        with torch.inference_mode():
            expected = task_loss(eager.eval(), task, torch.device('cuda'), amp=False)
            actual = task_loss(compiled, task, torch.device('cuda'), amp=False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class ScheduleFreeForeachTests(unittest.TestCase):
    def test_foreach_matches_scalar_equations_and_eval_train_weights(self):
        import schedulefree
        torch.manual_seed(44)
        left = [torch.nn.Parameter(torch.randn(3, 4)), torch.nn.Parameter(torch.randn(4))]
        right = [torch.nn.Parameter(p.detach().clone()) for p in left]
        scalar = schedulefree.AdamWScheduleFree(left, lr=0.001, warmup_steps=3, foreach=False)
        vectorized = schedulefree.AdamWScheduleFree(right, lr=0.001, warmup_steps=3, foreach=True)
        scalar.train()
        vectorized.train()
        for step in range(5):
            for index, (p, q) in enumerate(zip(left, right)):
                grad = torch.randn_like(p)
                p.grad = None if step == 2 and index == 1 else grad.clone()
                q.grad = None if p.grad is None else grad.clone()
            scalar.step()
            vectorized.step()
            for p, q in zip(left, right):
                torch.testing.assert_close(q, p)
                for key in ('z', 'exp_avg_sq'):
                    torch.testing.assert_close(vectorized.state[q][key], scalar.state[p][key])
            scalar.eval()
            vectorized.eval()
            for p, q in zip(left, right):
                torch.testing.assert_close(q, p)
            scalar.train()
            vectorized.train()


if __name__ == '__main__':
    unittest.main()
