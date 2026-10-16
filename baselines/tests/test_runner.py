from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from baselines.catalog import FILES, MODELS, REPO_ROOT, dataset_identity, discover_datasets
from baselines.results import write_json
from baselines import run


class RunnerTests(unittest.TestCase):
    def fixture(self, root, name):
        directory = root / name
        directory.mkdir()
        for filename in FILES:
            (directory / filename).touch()
        return directory

    def test_discovery_variants_and_exact_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("artnet-exp", "artnet-exp-t-10", "artnet-exp-t-2", "city-roads-M", "cora"):
                self.fixture(root, name)
            self.assertEqual([p.name for p in discover_datasets(root)],
                             ["artnet-exp", "artnet-exp-t-2", "artnet-exp-t-10", "city-roads-M"])
            self.assertEqual(len(discover_datasets(root, ["artnet-exp"])), 3)
            self.assertEqual(len(discover_datasets(root, ["artnet-exp"], "tiny")), 2)
            self.assertEqual(len(discover_datasets(root, ["artnet-exp-t-2"])), 1)
            self.assertEqual(len(discover_datasets(root, variants="full")), 2)
            with self.assertRaises(ValueError):
                discover_datasets(root, ["twitch-views"])
            with self.assertRaises(ValueError):
                discover_datasets(root, ["artnet-exp", "artnet-exp"])
            with self.assertRaises(ValueError):
                dataset_identity("cora")
            (root / "artnet-exp-t-2" / "features.csv").unlink()
            with self.assertRaisesRegex(ValueError, "Missing dataset file"):
                discover_datasets(root)

    def test_local_suite_plan(self):
        if not (REPO_ROOT / "data" / "city-roads-M-t-4").exists():
            self.skipTest("Local GraphLand data not available")
        paths = discover_datasets(REPO_ROOT / "data")
        self.assertEqual(len(paths), 60)
        self.assertEqual(sum(dataset_identity(p.name)[1] is None for p in paths), 10)
        self.assertEqual(len(paths) * len(MODELS), 240)

    def test_strict_json_preserves_previous_checkpoint_on_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            write_json(path, {"scores": {"val": None, "test": -0.25}})
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                write_json(path, {"score": float("nan")})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_list_does_not_import_dependencies_or_write_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, "artnet-exp")
            path = root / "results.json"
            with patch.object(run, "dependency_versions", side_effect=AssertionError("Imported dependencies")):
                with redirect_stdout(io.StringIO()) as output:
                    status = run.main(["--data-dir", str(root), "--list", "--output", str(path)])
            self.assertEqual(status, 0)
            self.assertIn("4 fits", output.getvalue())
            self.assertFalse(path.exists())

    def test_missing_dependencies_fail_before_writing_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, "artnet-exp")
            output = root / "results.json"
            with patch.object(run, "dependency_versions", side_effect=ModuleNotFoundError("numpy")), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as stderr:
                status = run.main(["--data-dir", str(root), "--output", str(output)])
            self.assertEqual(status, 2)
            self.assertIn("baselines/requirements.txt", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_failure_is_recorded_without_losing_completed_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, "artnet-exp")
            output = root / "scores.json"

            def evaluate(path, models, seed, n_jobs):
                yield {**run.record_header(path, "rf50", seed), "status": "ok", "metric": "AP",
                       "scores": {"val": 0.6, "test": 0.7}}
                # The runner must already have persisted the first completed fit.
                self.assertEqual(len(json.loads(output.read_text())["results"]), 1)
                raise RuntimeError("fixture failure")

            with patch.object(run, "dependency_versions", return_value={"test": "fixture"}), \
                    patch.object(run, "evaluate_dataset", side_effect=evaluate), redirect_stdout(io.StringIO()):
                status = run.main(["--data-dir", str(root), "--output", str(output)])
            report = json.loads(output.read_text())
            self.assertEqual(status, 1)
            self.assertEqual(report["status"], "completed_with_errors")
            self.assertEqual(report["successful_runs"], 1)
            self.assertEqual(report["failed_runs"], 3)
            self.assertEqual({r["model"] for r in report["results"]}, set(MODELS))
            self.assertEqual(report["results"][0]["scores"]["test"], 0.7)
            self.assertTrue(all(r["error"]["message"] == "fixture failure" for r in report["results"][1:]))
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run.parse_args(["--data-dir", str(root), "--output", str(output)])

    def test_success_and_interruption_status(self):
        for interrupt in (False, True):
            with self.subTest(interrupt=interrupt), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.fixture(root, "artnet-exp")
                output = root / "scores.json"

                def evaluate(path, models, seed, n_jobs):
                    for name in models:
                        yield {**run.record_header(path, name, seed), "status": "ok", "scores": {"test": 0.5}}
                        if interrupt:
                            raise KeyboardInterrupt()

                with patch.object(run, "dependency_versions", return_value={"test": "fixture"}), \
                        patch.object(run, "evaluate_dataset", side_effect=evaluate), redirect_stdout(io.StringIO()):
                    status = run.main(["--data-dir", str(root), "--output", str(output)])
                report = json.loads(output.read_text())
                self.assertEqual(status, 130 if interrupt else 0)
                self.assertEqual(report["status"], "interrupted" if interrupt else "completed")
                self.assertEqual(len(report["results"]), 1 if interrupt else 4)
                self.assertIsNotNone(report["finished_at"])


if __name__ == "__main__":
    unittest.main()
