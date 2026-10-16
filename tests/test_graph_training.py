import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import PropertyMock, patch

import numpy as np
import pandas as pd
import torch
import yaml

from graph_eval import GraphEvaluator, ScoreTracker, evaluation_paths, load_baselines, prepare_graph
from nanographpfn import GraphAttention, NanoGraphPFNModel, TransformerEncoderLayer
from nanoprior import PriorConfig, make_dataloader, sample_task
from train_nano import Config, main, make_optimizers, parse_args, save_checkpoint, task_loss, train_epoch


def small_model():
    return NanoGraphPFNModel(num_layers=2, num_heads=2, embed_dim=16, hidden_dim=32, max_classes=8)


def fixture(root, name, regression=True):
    path = root / name
    path.mkdir()
    n = 16
    ids = np.arange(n) * 3 + 101
    values = np.linspace(-2, 3, n)
    pd.DataFrame({"node_id": ids, "numeric": values, "category": np.arange(n) % 3,
                  "constant": np.ones(n)}).to_csv(path / "features.csv", index=False)
    targets = (values * 2 + 10 if regression else (np.arange(n) % 2).astype(float))
    targets[-1] = np.nan
    # Reverse target rows to test node-ID alignment, not implicit CSV row order.
    pd.DataFrame({"node_id": ids, "target": targets}).iloc[::-1].to_csv(path / "targets.csv", index=False)
    train = np.arange(n) < 6
    train[-1] = True  # GraphLand puts unlabeled nodes in train; they are NOT context.
    val = (np.arange(n) >= 6) & (np.arange(n) < 9)
    test = ~(train | val)
    pd.DataFrame({"node_id": ids, "train": train, "val": val, "test": test}).to_csv(
        path / "split_masks_RL.csv", index=False)
    pd.DataFrame({"source": ids[:-1], "target": ids[1:]}).to_csv(path / "edgelist.csv", index=False)
    info = {"task": "regression" if regression else "binary_classification", "metric": "R2" if regression else "AP",
            "target_name": "target", "numerical_features_names": ["numeric", "constant"],
            "fraction_features_names": [], "categorical_features_names": ["category"]}
    (path / "info.yaml").write_text(yaml.safe_dump(info))
    return {"dataset": name, "model": "rf50_nfa", "seed": 42, "split": "RL", "status": "ok",
            "variant": "tiny" if "-t-" in name else "full", "n_estimators": 50, "nfa": True,
            "scores": {"val": 0.9, "test": -1e6}, "task": info["task"], "metric": info["metric"],
            "num_nodes": n, "num_raw_features": 3, "labeled_split_counts": {"train": 6, "val": 3, "test": 6}}


def fixture_suites(root):
    data = root / "data"
    data.mkdir()
    rows = []
    for name, regression in (("artnet-views", True), ("tolokers-2", False)):
        rows.append(fixture(data, name, regression))
        for index in range(5):
            rows.append(fixture(data, f"{name}-t-{index}", regression))
    baseline = root / "baseline.json"
    baseline.write_text(json.dumps({"results": rows}))
    return data, baseline


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(12)

    def test_dense_attention_matches_neighbor_reference_values_and_gradients_with_isolates(self):
        module = GraphAttention(12, 3)
        edges = torch.tensor([[0, 1, 2, 2, 3], [1, 2, 0, 3, 2]])
        x = torch.randn(5, 3, 12, requires_grad=True)
        dense = module(x, edges)
        dense_grad = torch.autograd.grad(dense.square().sum(), x)[0]
        q, k, v = module.project(x)
        aggregates = []
        for node in range(len(x)):
            neighbors = edges[0, edges[1] == node]
            if len(neighbors):
                scores = (q[node] * k[neighbors]).sum(-1) / module.head_dim ** 0.5
                aggregates.append((scores.softmax(0)[..., None] * v[neighbors]).sum(0))
            else:
                aggregates.append(v[node] * 0)
        expected = module.output(torch.stack(aggregates).flatten(-2))
        expected_grad = torch.autograd.grad(expected.square().sum(), x)[0]
        torch.testing.assert_close(dense, expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(dense_grad, expected_grad, atol=1e-6, rtol=1e-5)

    def test_node_permutation_and_edges_affect_predictions(self):
        model = small_model().eval()
        task = sample_task(PriorConfig(n_nodes=32, n_features=(4, 4), task_type="regression"), seed=5)
        n, sep = len(task.x), task.sep
        order = torch.cat((torch.randperm(sep), sep + torch.randperm(n - sep)))
        inverse = torch.argsort(order)
        with torch.no_grad():
            original = model(task.x, task.context_y, task.edge_index, task.task_type)
            reordered = model(task.x[order], task.y[order][:sep], inverse[task.edge_index], task.task_type)
            disconnected = model(task.x, task.context_y, torch.empty(2, 0, dtype=torch.long), task.task_type)
        torch.testing.assert_close(reordered[inverse[sep:] - sep], original, atol=2e-6, rtol=2e-5)
        self.assertGreater(float((original - disconnected).abs().max()), 1e-4)

    def test_query_labels_are_never_model_inputs(self):
        model = small_model()
        task = sample_task(PriorConfig(n_nodes=32, n_features=(4, 4), task_type="regression"), seed=5)
        outputs = []
        handle = model.register_forward_hook(lambda module, args, output: outputs.append(output.detach().clone()))
        first = task_loss(model, task, torch.device("cpu"))
        changed = replace(task, y=task.y.clone())
        changed.y[task.sep:] += 100
        second = task_loss(model, changed, torch.device("cpu"))
        handle.remove()
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        self.assertNotEqual(float(first.detach()), float(second.detach()))
        with self.assertRaisesRegex(ValueError, "ONLY context"):
            model(task.x, task.y, task.edge_index, task.task_type)

    def test_all_task_types_train_their_heads_and_graph_adapters(self):
        model = small_model()
        for kind in ("binclass", "multiclass", "regression"):
            model.zero_grad(set_to_none=True)
            task = sample_task(PriorConfig(n_nodes=32, n_features=(4, 6), task_type=kind, max_num_classes=8), seed=8)
            loss = task_loss(model, task, torch.device("cpu"))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            decoder = model.regression_decoder if kind == "regression" else model.classification_decoder
            self.assertGreater(float(decoder[-1].weight.grad.abs().sum()), 0)
            # Check the first backward pass with the model's actual initialization.
            for index, layer in enumerate(model.layers):
                for name, weight in (("feature_output", layer.feature_output.weight),
                                     ("context_output", layer.context_output.weight),
                                     ("graph_qkv", layer.graph_attention.qkv.weight),
                                     ("graph_output", layer.graph_attention.output.weight),
                                     ("graph_mlp_in", layer.graph_mlp[0].weight),
                                     ("graph_mlp_out", layer.graph_mlp[-1].weight)):
                    with self.subTest(task=kind, layer=index, parameter=name):
                        self.assertIsNotNone(weight.grad)
                        self.assertTrue(torch.isfinite(weight.grad).all())
                        self.assertGreater(float(weight.grad.abs().sum()), 0)

    def test_continuous_prior_does_not_restart_at_epoch_boundaries(self):
        model = small_model()
        optimizers = make_optimizers(model, Config())
        prior = PriorConfig(n_nodes=32, n_features=(4, 4), max_num_classes=8)
        iterator = iter(make_dataloader(prior, num_tasks=4, seed=2))
        first = train_epoch(model, optimizers, iterator, 2, torch.device("cpu"))
        optimizers.adamw.eval()
        second = train_epoch(model, optimizers, iterator, 2, torch.device("cpu"))
        self.assertEqual(first["next_task_index"], 2)
        self.assertEqual(second["next_task_index"], 4)


class FeatureAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(12)

    def test_cuda_batch_limit_and_avazu_shape(self):
        layer = TransformerEncoderLayer(num_heads=4, embed_dim=256, hidden_dim=768).to("meta")
        for nodes, batches in ((65535, [65535]), (65536, [32768, 32768]),
                               (76269, [32768, 32768, 10733])):
            with self.subTest(nodes=nodes):
                x = torch.empty(nodes, 65, 256, device="meta")
                # Exercise CUDA dispatch without allocating Avazu-sized tensors.
                with patch.object(torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True), \
                        patch("nanographpfn.F.scaled_dot_product_attention",
                              side_effect=lambda q, k, v: torch.empty_like(q)) as attention:
                    result = layer.feature_attention(x)
                self.assertEqual(result.shape, x.shape)
                self.assertEqual([call.args[0].shape[0] for call in attention.call_args_list], batches)
                for call in attention.call_args_list:
                    for tensor in call.args:
                        self.assertEqual(tensor.shape[1:], (4, 65, 64))

    def test_large_batch_splitting_preserves_values_and_gradients(self):
        layer = TransformerEncoderLayer(num_heads=2, embed_dim=8, hidden_dim=16).double()
        x = torch.randn(76269, 3, 8, dtype=torch.float64, requires_grad=True)
        inputs = (x, *layer.feature_norm.parameters(), *layer.feature_qkv.parameters(),
                  *layer.feature_output.parameters())
        sdpa = torch.nn.functional.scaled_dot_product_attention
        with patch("nanographpfn.F.scaled_dot_product_attention", wraps=sdpa) as attention:
            expected = layer.feature_attention(x)
        attention.assert_called_once()  # Large CPU batches keep the ordinary path.
        gradient = torch.randn_like(expected)
        expected_grads = torch.autograd.grad(expected, inputs, grad_outputs=gradient)
        # Use real CPU attention to validate the CUDA dispatch branch's math.
        with patch.object(torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True):
            actual = layer.feature_attention(x)
        actual_grads = torch.autograd.grad(actual, inputs, grad_outputs=gradient)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(actual_grad, expected_grad, atol=1e-8, rtol=1e-8)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_large_batch_fused_attention_forward_and_backward(self):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        layer = TransformerEncoderLayer(num_heads=2, embed_dim=32, hidden_dim=64).cuda()
        x = torch.randn(76269, 3, 32, device="cuda", requires_grad=True)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        indices = torch.tensor([0, 32767, 32768, 65534, 65535, 65536, 76268], device="cuda")
        # Disallow the math fallback so the regression tests real fused kernels.
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]), \
                torch.autocast("cuda", dtype=dtype):
            actual = layer.feature_attention(x)
            expected = layer.feature_attention(x[indices])
        torch.testing.assert_close(actual[indices], expected, atol=2e-2, rtol=2e-2)
        self.assertTrue(torch.isfinite(actual).all())
        actual.float().square().mean().backward()
        for gradient in (x.grad, layer.feature_norm.weight.grad, layer.feature_qkv.weight.grad,
                         layer.feature_output.weight.grad, layer.feature_output.bias.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0)


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_real_selection_and_baselines(self):
        root = Path(__file__).resolve().parents[1]
        suites = evaluation_paths(root / "data")
        self.assertEqual(len(suites["full"]), 10)
        self.assertEqual(len(suites["tiny"]), 20)
        baseline = load_baselines(root / "baselines/results/graphland_rf.json", suites)
        self.assertAlmostEqual(baseline["suites"]["full"]["score"], 0.5111094821985549)
        self.assertAlmostEqual(baseline["suites"]["tiny"]["score"], 0.2773684572726378)

    def test_context_scaling_label_masks_and_test_only_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root, "artnet-views")
            path = root / "artnet-views"
            first = prepare_graph(path)
            self.assertEqual(first.details["num_context"], 6)
            self.assertEqual(first.details["num_test"], 6)
            self.assertEqual(first.details["num_features"], 2)
            self.assertAlmostEqual(float(first.context_y.mean()), 0, places=6)
            self.assertAlmostEqual(float(first.context_y.std(correction=0)), 1, places=6)
            targets = pd.read_csv(path / "targets.csv")
            # Change val/test labels, leaving context labels and unlabeled nodes intact.
            selected = targets["node_id"].between(101 + 6 * 3, 101 + 14 * 3)
            targets.loc[selected, "target"] += 10000
            targets.to_csv(path / "targets.csv", index=False)
            changed = prepare_graph(path)
            torch.testing.assert_close(changed.context_y, first.context_y, rtol=0, atol=0)
            self.assertEqual(changed.target_mean, first.target_mean)
            self.assertEqual(changed.target_scale, first.target_scale)
            torch.testing.assert_close(changed.x, first.x, rtol=0, atol=0)
            self.assertEqual(len(changed.test_targets), 6)

    def test_baseline_coverage_is_not_silently_reduced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, path = fixture_suites(root)
            suites = evaluation_paths(data, tiny_names=["artnet-views", "tolokers-2"])
            payload = json.loads(path.read_text())
            payload["results"].pop()
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "exactly one"):
                load_baselines(path, suites)

    def test_evaluation_uses_full_graph_forward_in_inference_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, baseline = fixture_suites(root)
            suites = evaluation_paths(data, tiny_names=["artnet-views", "tolokers-2"])
            evaluator = GraphEvaluator(suites, load_baselines(baseline, suites), "cpu")
            model = small_model()
            calls = []

            def observe(module, args, output):
                x, context_y, edges, task_type = args
                calls.append(task_type)
                self.assertFalse(module.training)
                self.assertTrue(torch.is_inference_mode_enabled())
                self.assertFalse(output.requires_grad)
                self.assertEqual(len(x), 16)  # Includes validation and unlabeled nodes.
                self.assertEqual(len(context_y), 6)
                self.assertEqual(edges.shape, (2, 30))
                self.assertEqual(len(output), 10)  # All non-context nodes, before scoring.

            handle = model.register_forward_hook(observe)
            try:
                results = evaluator.evaluate(model, log=lambda message: None)
            finally:
                handle.remove()
            self.assertEqual(calls.count("regression"), 6)
            self.assertEqual(calls.count("binclass"), 6)
            self.assertEqual(set(results), {"full", "tiny"})
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_best_scores_and_first_crossings_are_independent(self):
        tracker = ScoreTracker({"suites": {"full": {"score": .5}, "tiny": {"score": .3}}})
        for epoch, (full, tiny) in enumerate(((.4, .3), (.6, .2), (.55, .4)), start=1):
            tracker.update({"full": {"score": full, "datasets": {"f": {"score": full}}},
                            "tiny": {"score": tiny, "datasets": {"t": {"score": tiny}}}}, epoch, epoch * 10, epoch * 20)
        self.assertEqual(tracker.state["full"]["best_score"], .6)
        self.assertEqual(tracker.state["tiny"]["best_score"], .4)
        self.assertEqual(tracker.state["full"]["time_to_baseline_seconds"], 20)
        self.assertEqual(tracker.state["tiny"]["time_to_baseline_seconds"], 10)
        self.assertEqual(tracker.state["full"]["wall_time_to_baseline_seconds"], 40)
        fresh = ScoreTracker({"suites": {"full": {"score": 1}}})
        fresh.update({"full": {"score": -2, "datasets": {}}}, 1, 2, 3)
        self.assertIsNone(fresh.state["full"]["time_to_baseline_seconds"])
        json.dumps(fresh.state, allow_nan=False)


class TrainingIntegrationTests(unittest.TestCase):
    def arguments(self, root, data, baseline):
        return ["--device", "cpu", "--prior-workers", "0", "--epochs", "2", "--steps", "3",
                "--num-layers", "2", "--num-heads", "2", "--embed-dim", "16", "--hidden-dim", "32", "--max-classes", "8",
                "--prior-nodes", "32", "--prior-max-features", "5", "--max-train-mins", "0",
                "--data-dir", str(data), "--baseline-report", str(baseline), "--experiments-dir", str(root / "runs"),
                "--tiny-datasets", "artnet-views", "tolokers-2"]

    def reports(self, root):
        return sorted((root / "runs").glob("*/report.json"))

    def test_training_continues_after_both_baselines_and_resume_matches(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            data, baseline = fixture_suites(root)
            args = self.arguments(root, data, baseline)
            self.assertEqual(main(args), 0)
            report_path = self.reports(root)[0]
            report = json.loads(report_path.read_text())
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["completed_epochs"], 2)
            self.assertEqual(report["next_task_index"], 6)
            self.assertEqual(report["stop_reason"], "epochs_completed")
            self.assertEqual(report["scores"]["full"]["baseline_reached_epoch"], 1)
            self.assertEqual(report["scores"]["tiny"]["baseline_reached_epoch"], 1)
            full_run = torch.load(report_path.parent / "final.pt", weights_only=True)
            # Run one epoch and resume, verifying schedule-free optimizer weights too.
            self.assertEqual(main(args + ["--epochs", "1"]), 0)
            one_epoch_path = next(p for p in self.reports(root) if json.loads(p.read_text())["completed_epochs"] == 1)
            self.assertEqual(main(args + ["--resume", str(one_epoch_path.parent / "last.pt")]), 0)
            resumed_path = next(p for p in self.reports(root) if json.loads(p.read_text())["resumed_from"])
            resumed = torch.load(resumed_path.parent / "final.pt", weights_only=True)
            self.assertEqual(resumed["next_task_index"], full_run["next_task_index"])
            for name, tensor in full_run["model"].items():
                torch.testing.assert_close(resumed["model"][name], tensor, rtol=0, atol=0)

    def test_time_limit_still_evaluates_and_writes_final_report(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            data, baseline = fixture_suites(root)
            args = self.arguments(root, data, baseline)
            self.assertEqual(main(args + ["--max-train-mins", "0.000000001", "--skip-preflight",
                                          "--eval-every", "100"]), 0)
            path = self.reports(root)[0]
            report = json.loads(path.read_text())
            self.assertEqual(report["completed_epochs"], 1)
            self.assertEqual(report["stop_reason"], "training_time_limit")
            self.assertIsNotNone(report["best_full_score"])
            self.assertIsNotNone(report["best_tiny_score"])
            self.assertTrue((path.parent / "final.pt").exists())

    def test_evaluation_interval_and_final_epoch(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            data, baseline = fixture_suites(root)
            args = self.arguments(root, data, baseline)
            # Five epochs exercises two intervals and an off-interval final eval.
            with patch.object(GraphEvaluator, "evaluate", autospec=True,
                              side_effect=GraphEvaluator.evaluate) as evaluate:
                self.assertEqual(main(args + ["--epochs", "5", "--eval-every", "2", "--skip-preflight"]), 0)
            self.assertEqual(evaluate.call_count, 3)
            path = self.reports(root)[0]
            report = json.loads(path.read_text())
            records = [json.loads(line) for line in (path.parent / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([r["epoch"] for r in records], [1, 2, 3, 4, 5])
            self.assertEqual([r["epoch"] for r in records if r["evaluation"] is not None], [2, 4, 5])
            self.assertEqual(report["last_evaluation"], records[-1]["evaluation"])
            self.assertEqual(report["completed_epochs"], 5)
            self.assertEqual(report["evaluation"]["every_epochs"], 2)
            for suite in ("full", "tiny"):
                self.assertEqual(report["scores"][suite]["baseline_reached_epoch"], 2)
                self.assertEqual(report["scores"][suite]["time_to_baseline_seconds"], records[1]["train_seconds"])
                best = torch.load(path.parent / f"best_{suite}.pt", weights_only=True)
                self.assertIn(best["epoch"], [2, 4, 5])
            final = torch.load(path.parent / "final.pt", weights_only=True)
            self.assertEqual(final["epoch"], 5)
            self.assertEqual(final["config"]["eval_every"], 2)

    def test_resume_from_epoch_without_evaluation_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            data, baseline = fixture_suites(root)
            args = self.arguments(root, data, baseline) + ["--epochs", "4", "--eval-every", "3", "--skip-preflight"]
            unevaluated_path = root / "epoch_two.pt"

            def keep_checkpoint(path, model, optimizers, config, prior_config, tracker, epoch, *remaining):
                save_checkpoint(path, model, optimizers, config, prior_config, tracker, epoch, *remaining)
                if path.name == "last.pt" and epoch == 2:
                    unevaluated_path.write_bytes(path.read_bytes())

            with patch("train_nano.save_checkpoint", side_effect=keep_checkpoint):
                self.assertEqual(main(args), 0)
            checkpoint = torch.load(unevaluated_path, weights_only=True)
            self.assertEqual(checkpoint["next_task_index"], 6)
            self.assertIsNone(checkpoint["scores"]["full"]["best_score"])
            self.assertIsNone(checkpoint["scores"]["tiny"]["best_score"])
            full_path = self.reports(root)[0]
            full_run = torch.load(full_path.parent / "final.pt", weights_only=True)
            self.assertEqual(main(args + ["--resume", str(unevaluated_path)]), 0)
            resumed_path = next(p for p in self.reports(root) if json.loads(p.read_text())["resumed_from"])
            resumed = torch.load(resumed_path.parent / "final.pt", weights_only=True)
            records = [json.loads(line) for line in (resumed_path.parent / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([r["epoch"] for r in records if r["evaluation"] is not None], [3, 4])
            self.assertEqual(resumed["next_task_index"], full_run["next_task_index"])
            for name, tensor in full_run["model"].items():
                torch.testing.assert_close(resumed["model"][name], tensor, rtol=0, atol=0)
            for suite in ("full", "tiny"):
                self.assertEqual(resumed["scores"][suite]["best_score"], full_run["scores"][suite]["best_score"])

    def test_evaluation_interval_must_be_positive(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for value in ("0", "-1"):
                with self.assertRaises(SystemExit) as error:
                    parse_args(["--eval-every", value])
                self.assertEqual(error.exception.code, 2)

    def test_preflight_never_trains_or_counts_random_model_scores(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            data, baseline = fixture_suites(root)
            with patch("train_nano.train_epoch", side_effect=AssertionError("must not train")):
                self.assertEqual(main(self.arguments(root, data, baseline) + ["--preflight-only"]), 0)
            path = self.reports(root)[0]
            report = json.loads(path.read_text())
            self.assertEqual(report["status"], "preflight_completed")
            self.assertIsNone(report["best_full_score"])
            self.assertIsNone(report["time_to_baseline_tiny_seconds"])
            self.assertFalse((path.parent / "final.pt").exists())
            self.assertTrue((path.parent / "preflight.json").exists())

    def test_preflight_oom_stops_before_training_with_explicit_failure(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            data, baseline = fixture_suites(root)
            with patch.object(GraphEvaluator, "evaluate_graph", side_effect=torch.OutOfMemoryError("oom")), \
                    patch("train_nano.train_epoch", side_effect=AssertionError("must not train")):
                self.assertEqual(main(self.arguments(root, data, baseline)), 1)
            report = json.loads(self.reports(root)[0].read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["completed_epochs"], 0)
            self.assertIn("full/artnet-views", report["error"])
            self.assertIn("EVAL_NODE_BATCH_SIZE", report["error"])
            self.assertIn("Original allocation error: oom", report["error"])
            self.assertIsNone(report["best_full_score"])


if __name__ == "__main__":
    unittest.main()
