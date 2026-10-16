"""Train nanoGraphPFN on nanoprior; evaluate full/tiny GraphLand test splits.

Nano backbone and Muon/schedule-free training scheme derived from record 3 of
borawhocodess/modded-nanotabpfn (Apache-2.0). Importing this module is safe.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import random
import sys
import time
import uuid
from typing import Any, TypedDict

import numpy as np
import schedulefree
import torch
import torch.nn.functional as F

from baselines.catalog import REPO_ROOT
from baselines.results import write_json
from graph_eval import AVAZU_PCA_COMPONENTS, GraphEvaluator, ScoreTracker, Suites, evaluation_paths, load_baselines
from nanographpfn import ARCHITECTURE_FIELDS, NanoGraphPFNModel, PackedQKVLinear
from nanoprior import PRIOR_VERSION, GraphTask, PriorConfig, make_dataloader


CHECKPOINT_VERSION = 2
Log = Callable[[str], None]


@dataclass
class Config:
    experiments_dir: str = str(REPO_ROOT / "workdir/experiments")
    data_dir: str = str(REPO_ROOT / "data")
    baseline_report: str = str(REPO_ROOT / "baselines/results/graphland_rf.json")
    baseline_seed: int = 42
    seed: int = 11
    batch_size: int = 1
    lr: float = 0.001
    steps: int = 64
    epochs: int = 1000
    eval_every: int = 1
    num_heads: int = 4
    embed_dim: int = 256
    hidden_dim: int = 768
    num_layers: int = 6
    max_classes: int = 32
    max_train_mins: float = 80.0
    prior_workers: int = 4
    prior_nodes: int = 1000
    prior_min_features: int = 4
    prior_max_features: int = 100
    device: str = "cuda"
    cpu_threads: int = 1
    cuda_graph_muon: bool = True
    compile_blocks: bool = True
    compile_projections: bool = True
    batched_muon: bool = True
    async_transfer: bool = True


class EpochMetrics(TypedDict):
    seconds: float
    loss: float
    task_losses: dict[str, float]
    task_counts: dict[str, int]
    next_task_index: int


def zeropower_via_newtonschulz5(gradient: torch.Tensor, steps: int = 10, eps: float = 1e-7) -> torch.Tensor:
    """Approximate the polar factor of a matrix [out_channels, in_channels]."""
    linear, cubic, quintic = 3.4445, -4.7750, 2.0315
    matrix = gradient.bfloat16()
    matrix /= matrix.norm() + eps
    transpose = gradient.size(0) > gradient.size(1)
    if transpose:
        matrix = matrix.T
    for _ in range(steps):
        gram = matrix @ matrix.T
        cubic_term = gram @ matrix
        matrix = linear * matrix + cubic * cubic_term + quintic * gram @ cubic_term
    if transpose:
        matrix = matrix.T
    return matrix.to(gradient.dtype)


def batched_zeropower_via_newtonschulz5(gradient: torch.Tensor, steps: int = 10,
                                      eps: float = 1e-7) -> torch.Tensor:
    """Independent polar factors for [B, rows, columns], never a joint norm."""
    matrix = gradient.bfloat16()
    matrix = matrix / (matrix.norm(dim=(-2, -1), keepdim=True) + eps)
    transpose = gradient.size(-2) > gradient.size(-1)
    if transpose:
        matrix = matrix.transpose(-2, -1)
    for _ in range(steps):
        gram = matrix @ matrix.transpose(-2, -1)
        cubic_term = gram @ matrix
        matrix = 3.4445 * matrix + -4.7750 * cubic_term + 2.0315 * gram @ cubic_term
    if transpose:
        matrix = matrix.transpose(-2, -1)
    return matrix.to(gradient.dtype)


class Muon(torch.optim.Optimizer):
    """Muon for matrix weights, with explicitly marked packed QKV projections.

    QKV groups orthogonalize three [D, D] matrices independently. Other matrices,
    including MLP weights shaped [3D, D], are orthogonalized as a whole.
    """

    def __init__(self, params: Iterable[torch.nn.Parameter], *,
                 qkv_params: Iterable[torch.nn.Parameter] = (), lr: float = 3e-4,
                 momentum: float = 0.95, nesterov: bool = True, backend_steps: int = 5,
                 cuda_graph: bool = True, batched: bool = True) -> None:
        params = list(params)
        qkv = set(qkv_params)
        if not qkv.issubset(params) or any(parameter.ndim != 2 for parameter in params):
            raise ValueError("Muon requires matrix parameters; QKV parameters must be a subset")
        groups = [{"params": [parameter for parameter in params if parameter not in qkv], "is_qkv": False},
                  {"params": [parameter for parameter in params if parameter in qkv], "is_qkv": True}]
        super().__init__(groups, dict(lr=lr, momentum=momentum, nesterov=nesterov, backend_steps=backend_steps))
        self.orthogonalize = (torch.compile(zeropower_via_newtonschulz5)
                              if any(p.is_cuda for p in params) else zeropower_via_newtonschulz5)
        self.cuda_graph = cuda_graph and bool(params) and all(p.is_cuda for p in params)
        self.batched = batched and bool(params) and all(p.is_cuda for p in params)
        self.orthogonalize_batch = (torch.compile(batched_zeropower_via_newtonschulz5)
                                   if self.batched else batched_zeropower_via_newtonschulz5)
        self._graph = None
        self._graph_signature = None
        self._static_gradients = []

    @torch.no_grad()
    def step(self) -> None:
        if not self.cuda_graph:
            self._step()
            return
        active = [p for group in self.param_groups for p in group['params'] if p.grad is not None]
        if not active:
            return
        # Changes in active parameters, optimizer state, or hyperparameters need
        # a new capture. In particular, load_state_dict replaces momentum tensors.
        for p in active:
            if 'momentum_buffer' not in self.state[p]:
                self.state[p]['momentum_buffer'] = torch.zeros_like(p)
        signature = (self.batched, tuple((id(p), p.data_ptr(), self.state[p]['momentum_buffer'].data_ptr(),
                            p.grad.shape, p.grad.stride(), p.grad.dtype) for p in active),
                     tuple((tuple(id(p) for p in g['params']), g['lr'], g['momentum'],
                            g['nesterov'], g['backend_steps'], g['is_qkv'])
                           for g in self.param_groups))
        if signature != self._graph_signature:
            self._capture(active)
            self._graph_signature = signature
        for p, gradient in zip(active, self._static_gradients):
            gradient.copy_(p.grad)
        self._graph.replay()

    @torch.no_grad()
    def _capture(self, active: list[torch.nn.Parameter]) -> None:
        device = active[0].device
        if any(p.device != device for p in active):
            raise ValueError('Muon CUDA graph requires parameters on one device')
        gradients = [p.grad for p in active]
        self._static_gradients = [g.clone() for g in gradients]
        parameters = [p.clone() for p in active]
        buffers = [self.state[p]['momentum_buffer'].clone() for p in active]

        def restore():
            for p, saved, buffer in zip(active, parameters, buffers):
                p.copy_(saved)
                self.state[p]['momentum_buffer'].copy_(buffer)

        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        try:
            for p, gradient in zip(active, self._static_gradients):
                p.grad = gradient
            # Compilation and library initialization must finish before capture.
            # Restore every warmup/capture update: replay is the ONLY real step.
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._step()
                    restore()
            torch.cuda.current_stream(device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                self._step()
            self._graph = graph
        finally:
            torch.cuda.current_stream(device).wait_stream(stream)
            restore()
            for p, gradient in zip(active, gradients):
                p.grad = gradient

    @torch.no_grad()
    def _step(self) -> None:
        if self.batched:
            self._step_batched()
            return
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(gradient)
                momentum_buffer = state["momentum_buffer"]
                momentum_buffer.mul_(group["momentum"]).add_(gradient)
                if group["nesterov"]:
                    gradient = gradient.add(momentum_buffer, alpha=group["momentum"])
                else:
                    gradient = momentum_buffer
                if group["is_qkv"]:
                    update = torch.cat([self.orthogonalize(part, steps=group["backend_steps"])
                                        for part in gradient.chunk(3, dim=0)])
                    scale = gradient.size(1) ** 0.5
                else:
                    update = self.orthogonalize(gradient, steps=group["backend_steps"])
                    scale = max(gradient.shape) ** 0.5
                parameter.add_(update, alpha=-group["lr"] * scale)

    @torch.no_grad()
    def _step_batched(self) -> None:
        for group in self.param_groups:
            buckets = {}
            for p in group['params']:
                if p.grad is not None:
                    buckets.setdefault((p.shape, p.dtype, p.device), []).append(p)
            for parameters in buckets.values():
                gradients = [p.grad for p in parameters]
                for p in parameters:
                    if 'momentum_buffer' not in self.state[p]:
                        self.state[p]['momentum_buffer'] = torch.zeros_like(p)
                buffers = [self.state[p]['momentum_buffer'] for p in parameters]
                torch._foreach_mul_(buffers, group['momentum'])
                torch._foreach_add_(buffers, gradients)
                gradients = (torch._foreach_add(gradients, buffers, alpha=group['momentum'])
                             if group['nesterov'] else buffers)
                batch = torch.stack(gradients)
                if group['is_qkv']:
                    batch = batch.reshape(-1, parameters[0].shape[1], parameters[0].shape[1])
                    scale = parameters[0].shape[1] ** 0.5
                else:
                    scale = max(parameters[0].shape) ** 0.5
                update = self.orthogonalize_batch(batch, steps=group['backend_steps'])
                updates = update.reshape(len(parameters), *parameters[0].shape).unbind(0)
                torch._foreach_add_(parameters, updates, alpha=-group['lr'] * scale)


@dataclass
class Optimizers:
    """Matrix updates plus schedule-free AdamW for encoders, heads, biases and norms.

    AdamW must switch to eval weights before evaluation/checkpointing and back
    to train weights before optimizer steps. Muon has no mode-dependent weights.
    """

    muon: Muon
    adamw: schedulefree.AdamWScheduleFree

    def zero_grad(self) -> None:
        self.muon.zero_grad(set_to_none=True)
        self.adamw.zero_grad(set_to_none=True)

    def step(self) -> None:
        self.muon.step()
        self.adamw.step()

    def state_dict(self) -> dict[str, Any]:
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.muon.load_state_dict(state["muon"])
        self.adamw.load_state_dict(state["adamw"])


def make_optimizers(model: NanoGraphPFNModel, config: Config) -> Optimizers:
    matrices = {parameter for parameter in model.layers.parameters() if parameter.ndim == 2}
    qkv = {module.weight for module in model.layers.modules() if isinstance(module, PackedQKVLinear)}
    # Traverse model parameters once to keep optimizer state ordering deterministic.
    muon = [parameter for parameter in model.parameters() if parameter in matrices]
    adamw = [parameter for parameter in model.parameters() if parameter not in matrices]
    return Optimizers(
        Muon(muon, qkv_params=qkv, lr=0.1 * config.lr, momentum=0.95,
             cuda_graph=config.cuda_graph_muon, batched=config.batched_muon),
        schedulefree.AdamWScheduleFree(adamw, lr=config.lr, weight_decay=0.0, warmup_steps=1000, foreach=True),
    )


def task_loss(model: NanoGraphPFNModel, task: GraphTask, device: torch.device, amp: bool = True,
              validate_on_cpu: bool = True, async_transfer: bool = True) -> torch.Tensor:
    """Predict query labels only; the prior has already standardized numerical inputs."""
    if validate_on_cpu:
        model.validate_tensor_values(task.x, task.context_y, task.edge_index)
    targets = task.y.to(device, non_blocking=async_transfer)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
        predictions = model(
            task.x.to(device, non_blocking=async_transfer), targets[:task.sep],
            task.edge_index.to(device, non_blocking=async_transfer), task.task_type,
            num_classes=task.num_classes,
            feature_cardinalities=task.feature_cardinalities.to(device, non_blocking=async_transfer),
            features_normalized=True,
            inputs_validated=validate_on_cpu,
        )
    target = targets[task.sep:]
    if task.task_type == "regression":
        return F.mse_loss(predictions.float(), target.float())
    return F.cross_entropy(predictions.float(), target.long())


def train_epoch(model: NanoGraphPFNModel, optimizers: Optimizers, iterator: Iterator[GraphTask],
                steps: int, device: torch.device, amp: bool = True, async_transfer: bool = True) -> EpochMetrics:
    if steps < 1:
        raise ValueError("An epoch must contain at least one step")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    model.train()
    optimizers.adamw.train()
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    next_index = 0
    for _ in range(steps):
        task = next(iterator)
        optimizers.zero_grad()
        loss = task_loss(model, task, device, amp, async_transfer=async_transfer)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite {task.task_type} loss at task {task.metadata['task_index']}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizers.step()
        sums[task.task_type] = sums.get(task.task_type, 0) + loss.detach()
        counts[task.task_type] = counts.get(task.task_type, 0) + 1
        next_index = task.metadata["task_index"] + 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    losses = {key: float(value.cpu()) / counts[key] for key, value in sums.items()}
    mean = sum(losses[key] * counts[key] for key in counts) / steps
    return {"seconds": seconds, "loss": mean, "task_losses": losses,
            "task_counts": counts, "next_task_index": next_index}


def save_checkpoint(path: Path, model: NanoGraphPFNModel, optimizers: Optimizers, config: Config,
                    prior_config: PriorConfig, tracker: ScoreTracker, epoch: int, next_task_index: int,
                    train_seconds: float, wall_seconds: float, baselines: dict[str, Any], suites: Suites) -> None:
    numpy_state = np.random.get_state()
    checkpoint = {
        "format_version": CHECKPOINT_VERSION, "arch": model.arch, "model": model.state_dict(),
        "optimizers": optimizers.state_dict(),
        "config": asdict(config), "prior_config": asdict(prior_config), "prior_version": PRIOR_VERSION,
        "scores": tracker.state, "epoch": epoch, "next_task_index": next_task_index,
        "train_seconds": train_seconds, "wall_seconds": wall_seconds,
        "baseline_sha256": baselines["sha256"],
        "suite_names": {key: [p.name for p in paths] for key, paths in suites.items()},
        "rng": {"torch": torch.get_rng_state(), "python": random.getstate(),
                "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
    }
    temporary = path.with_suffix(".tmp")
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def restore_checkpoint(checkpoint: dict[str, Any], model: NanoGraphPFNModel, optimizers: Optimizers,
                       config: Config, prior_config: PriorConfig, baselines: dict[str, Any], suites: Suites) -> None:
    if checkpoint["format_version"] != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint format; start a new run")
    if checkpoint["arch"] != model.arch:
        raise ValueError("Checkpoint architecture does not match the requested model")
    if checkpoint["prior_version"] != PRIOR_VERSION or checkpoint["prior_config"] != asdict(prior_config):
        raise ValueError("Checkpoint prior differs from current prior")
    for key in ("seed", "steps", "lr", "batch_size"):
        if checkpoint["config"][key] != getattr(config, key):
            raise ValueError(f"Resume requires the original {key}")
    if (checkpoint["baseline_sha256"] != baselines["sha256"]
            or checkpoint["suite_names"] != {key: [p.name for p in paths] for key, paths in suites.items()}):
        raise ValueError("Resume requires the same baseline report and dataset selections")
    model.load_state_dict(checkpoint["model"])
    optimizers.load_state_dict(checkpoint["optimizers"])
    torch.set_rng_state(checkpoint["rng"]["torch"])
    random.setstate(checkpoint["rng"]["python"])
    numpy_state = checkpoint["rng"]["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    if torch.cuda.is_available() and checkpoint["rng"]["cuda"]:
        torch.cuda.set_rng_state_all(checkpoint["rng"]["cuda"])


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, Config]:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    defaults = Config()
    parser.add_argument("--name", default="")
    for key, value in asdict(defaults).items():
        help_text = {
            "eval_every": "Evaluate both suites every N epochs and at training end (default: 1)",
            "cuda_graph_muon": "Replay the Muon CUDA step as a graph (default: enabled on CUDA)",
            "compile_blocks": "Compile dynamic-shape MLP/norm training blocks (default: enabled on CUDA)",
            "compile_projections": "Compile attention norm/projection/QKV unpacking; keep SDPA eager",
            "batched_muon": "Batch same-shape Muon matrices on CUDA with independent normalization",
            "async_transfer": "Pin prior tasks and use nonblocking GPU copies",
        }.get(key)
        if isinstance(value, bool):
            parser.add_argument("--" + key.replace("_", "-"), action=argparse.BooleanOptionalAction,
                                default=value, help=help_text)
        else:
            parser.add_argument("--" + key.replace("_", "-"), type=type(value), default=value, help=help_text)
    parser.add_argument("--full-datasets", nargs="+", help="Full GraphLand names; default: all available full datasets")
    parser.add_argument("--tiny-datasets", nargs="+", help="Subset of the four approved tiny dataset families")
    parser.add_argument("--list-eval", action="store_true", help="Print selected datasets and RF test thresholds, then exit")
    parser.add_argument("--preflight-only", action="store_true", help="Evaluate all selected graphs once without training")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip the initial evaluation memory check")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", type=Path, help="Resume a trusted local last.pt/best_*.pt checkpoint")
    args = parser.parse_args(argv)
    config = Config(**{key: getattr(args, key) for key in asdict(defaults)})
    if min(getattr(config, key) for key in ARCHITECTURE_FIELDS) < 1 or config.embed_dim % config.num_heads:
        parser.error("architecture dimensions must be positive and embed-dim divisible by num-heads")
    if min(config.epochs, config.steps, config.cpu_threads) < 1 or config.batch_size != 1:
        parser.error("epochs/steps/cpu-threads must be positive; only batch-size=1 is supported")
    if config.eval_every < 1:
        parser.error("eval-every must be positive")
    if not math.isfinite(config.lr) or config.lr <= 0 or not math.isfinite(config.max_train_mins):
        parser.error("learning rate must be finite and positive, time limit must be finite")
    if config.seed < 0 or config.prior_workers < 0 or config.max_train_mins < 0:
        parser.error("seed/prior-workers/max-train-mins cannot be negative; 0 disables the time limit")
    if args.preflight_only and args.skip_preflight:
        parser.error("--preflight-only and --skip-preflight conflict")
    return args, config



@dataclass
class TrainingState:
    """Mutable training progress shared by the loop, checkpoints and final report."""

    model: NanoGraphPFNModel
    optimizers: Optimizers
    prior_config: PriorConfig
    tracker: ScoreTracker
    epoch: int = 0
    next_task_index: int = 0
    train_seconds: float = 0.0
    wall_offset: float = 0.0
    wall_start: float | None = None

    @property
    def wall_seconds(self) -> float:
        elapsed = time.perf_counter() - self.wall_start if self.wall_start is not None else 0.0
        return self.wall_offset + elapsed


def configure_runtime(config: Config, no_amp: bool) -> tuple[torch.device, bool]:
    """Validate hardware, choose precision and seed the main process."""
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use a CUDA host for training/preflight; --device cpu is for small checks.")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices are cpu and cuda")
    if device.type == "cuda":
        device = torch.device("cuda", device.index if device.index is not None else 0)
        torch.cuda.set_device(device)
    amp = not no_amp and device.type == "cuda"
    if amp and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This training scheme requires bfloat16 support; use --no-amp to run float32")
    torch.set_num_threads(config.cpu_threads)
    torch.set_float32_matmul_precision("high")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    return device, amp


def initialize_training(config: Config, device: torch.device, baselines: dict[str, Any], suites: Suites,
                        resume: Path | None, preflight_only: bool) -> TrainingState:
    prior_config = PriorConfig(n_nodes=config.prior_nodes,
                               n_features=(config.prior_min_features, config.prior_max_features),
                               max_num_classes=config.max_classes)
    model = NanoGraphPFNModel(**{name: getattr(config, name) for name in ARCHITECTURE_FIELDS}).to(device)
    if (config.compile_blocks or config.compile_projections) and device.type == 'cuda':
        model.compile_training_blocks(mlps=config.compile_blocks, projections=config.compile_projections)
    state = TrainingState(model, make_optimizers(model, config), prior_config, ScoreTracker(baselines))
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        restore_checkpoint(checkpoint, model, state.optimizers, config, prior_config, baselines, suites)
        state.tracker = ScoreTracker(baselines, checkpoint["scores"])
        state.epoch, state.next_task_index = checkpoint["epoch"], checkpoint["next_task_index"]
        state.train_seconds, state.wall_offset = checkpoint["train_seconds"], checkpoint["wall_seconds"]
        if state.epoch >= config.epochs and not preflight_only:
            raise ValueError("Checkpoint already completed the requested epochs; increase --epochs")
        if config.max_train_mins and state.train_seconds >= config.max_train_mins * 60 and not preflight_only:
            raise ValueError("Checkpoint already exhausted the training-time cap; increase or disable --max-train-mins")
    return state


def create_run_directory(config: Config, name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%y%m%d-%H%M%S")
    suffix = "".join(char if char.isalnum() or char in "-_" else "_" for char in name.strip())
    run_id = f"{timestamp}-{uuid.uuid4().hex[:8]}" + (f"-{suffix}" if suffix else "")
    output = Path(config.experiments_dir) / run_id
    output.mkdir(parents=True, exist_ok=False)
    source_dir = output / "source"
    source_dir.mkdir()
    for filename in ("train_nano.py", "nanographpfn.py", "graph_eval.py", "nanoprior.py"):
        (source_dir / filename).write_bytes((REPO_ROOT / filename).read_bytes())
    return output


def make_logger(output: Path) -> Log:
    def log(message: str) -> None:
        print(message, flush=True)
        with (output / "log.txt").open("a") as stream:
            stream.write(message + "\n")
    return log


def create_report(config: Config, state: TrainingState, evaluator: GraphEvaluator, output: Path,
                  selection: dict[str, Any], resume: Path | None) -> dict[str, Any]:
    report = {
        "schema_version": 2, "status": "running", "run_id": output.name, "completed_epochs": state.epoch,
        "config": asdict(config), "prior_config": asdict(state.prior_config), "prior_version": PRIOR_VERSION,
        "amp": evaluator.amp,
        "evaluation": {"split": "RL", "score_split": "test", "selection": selection,
                       "aggregation": "equal_weight_mean_of_raw_dataset_scores",
                       "frequency": "both_suites_every_n_epochs_and_at_training_end",
                       "every_epochs": config.eval_every,
                       "graph": "all_nodes_all_edges_undirected_no_self_loops",
                       "preprocessing": ("ordinal_categories_numeric_baseline_transforms_all_nodes_"
                                         "context_standardization"),
                       "avazu_pca_components": AVAZU_PCA_COMPONENTS},
        "baseline": evaluator.baselines, "scores": state.tracker.state,
        "timing": {"primary": "training_seconds_excluding_evaluation_and_preflight",
                   "wall": "elapsed_since_training_start_including_evaluation_excluding_preflight_and_resume_downtime",
                   "crossing": "first_evaluated_epoch_with_suite_mean_greater_than_or_equal_to_RF50_NFA"},
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "numpy": np.__version__, "cuda": torch.version.cuda},
        "device": str(evaluator.device), "parameters": sum(p.numel() for p in state.model.parameters()),
        "resumed_from": str(resume) if resume else None,
    }
    if evaluator.device.type == "cuda":
        props = torch.cuda.get_device_properties(evaluator.device)
        report["gpu"] = {"name": props.name, "total_memory_bytes": props.total_memory}
    return report


def run_preflight(state: TrainingState, evaluator: GraphEvaluator, output: Path, log: Log) -> None:
    """Check every selected dataset without counting untrained scores as progress."""
    log("Preflight: evaluating every selected graph before any training steps.")
    state.optimizers.adamw.eval()
    results = evaluator.evaluate(state.model, log)
    write_json(output / "preflight.json", {"status": "completed", "device": str(evaluator.device), "results": results})
    log("Preflight passed.")


def save_training_checkpoint(path: Path, state: TrainingState, config: Config,
                             evaluator: GraphEvaluator, wall_seconds: float) -> None:
    save_checkpoint(path, state.model, state.optimizers, config, state.prior_config, state.tracker,
                    state.epoch, state.next_task_index, state.train_seconds, wall_seconds,
                    evaluator.baselines, evaluator.suites)


def run_training(config: Config, state: TrainingState, evaluator: GraphEvaluator,
                 output: Path, report: dict[str, Any], log: Log) -> None:
    """Consume one continuous prior stream; evaluate and checkpoint at epoch boundaries."""
    loader = make_dataloader(state.prior_config, num_tasks=(config.epochs - state.epoch) * config.steps,
                             seed=config.seed, start_index=state.next_task_index, num_workers=config.prior_workers,
                             pin_memory=config.async_transfer and evaluator.device.type == 'cuda')
    state.wall_start = time.perf_counter()
    iterator = iter(loader)
    try:
        for epoch in range(state.epoch + 1, config.epochs + 1):
            metrics = train_epoch(state.model, state.optimizers, iterator, config.steps, evaluator.device, evaluator.amp,
                                  async_transfer=config.async_transfer)
            state.epoch = epoch
            state.train_seconds += metrics["seconds"]
            state.next_task_index = metrics["next_task_index"]
            time_limit_reached = bool(config.max_train_mins and state.train_seconds >= config.max_train_mins * 60)
            # Checkpoints always contain schedule-free evaluation weights,
            # including epochs where scoring is skipped.
            state.optimizers.adamw.eval()
            results = None
            if epoch % config.eval_every == 0 or epoch == config.epochs or time_limit_reached:
                evaluation_wall_start = state.wall_seconds
                results = evaluator.evaluate(state.model, log)
                for result in results.values():
                    result["wall_seconds"] = evaluation_wall_start + result["evaluation_elapsed_seconds"]
            wall_seconds = state.wall_seconds
            improved = (state.tracker.update(results, epoch, state.train_seconds, wall_seconds)
                        if results is not None else [])
            record = {"epoch": epoch, "training": metrics, "train_seconds": state.train_seconds,
                      "wall_seconds": wall_seconds, "evaluation": results}
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            scores = (f"full_test={results['full']['score']:.6f}, tiny_test={results['tiny']['score']:.6f}"
                      if results is not None else "evaluation skipped")
            log(f"Epoch {epoch}/{config.epochs}: loss={metrics['loss']:.5f}, train={state.train_seconds:.2f}s, {scores}")
            save_training_checkpoint(output / "last.pt", state, config, evaluator, wall_seconds)
            for suite in improved:
                save_training_checkpoint(output / f"best_{suite}.pt", state, config, evaluator, wall_seconds)
            if results is not None:
                report["last_evaluation"] = results
            report["completed_epochs"] = epoch
            # Baseline crossings are recorded and NEVER stop training.
            if time_limit_reached:
                report["stop_reason"] = "training_time_limit"
                break
        else:
            report["stop_reason"] = "epochs_completed"
        save_training_checkpoint(output / "final.pt", state, config, evaluator, wall_seconds)
        report["status"] = "completed"
    finally:
        del iterator


def finalize_report(report: dict[str, Any], state: TrainingState, output: Path,
                    invocation_start: float, log: Log) -> None:
    report.update(train_seconds=state.train_seconds, next_task_index=state.next_task_index,
                  wall_seconds=state.wall_seconds, invocation_seconds=time.perf_counter() - invocation_start)
    for suite in ("full", "tiny"):
        item = state.tracker.state[suite]
        report[f"best_{suite}_score"] = item["best_score"]
        report[f"time_to_baseline_{suite}_seconds"] = item["time_to_baseline_seconds"]
    write_json(output / "report.json", report)
    log(f"Report: {output / 'report.json'}")


def main(argv: list[str] | None = None) -> int:
    invocation_start = time.perf_counter()
    args, config = parse_args(argv)
    suites = evaluation_paths(config.data_dir, args.full_datasets, args.tiny_datasets)
    baselines = load_baselines(config.baseline_report, suites, config.baseline_seed)
    selection = {key: {"datasets": [p.name for p in paths], "rf50_nfa_test_score": baselines["suites"][key]["score"]}
                 for key, paths in suites.items()}
    if args.list_eval:
        print(json.dumps(selection, indent=2))
        return 0
    device, amp = configure_runtime(config, args.no_amp)
    state = initialize_training(config, device, baselines, suites, args.resume, args.preflight_only)
    output = create_run_directory(config, args.name)
    log = make_logger(output)
    evaluator = GraphEvaluator(suites, baselines, device, amp=amp)
    report = create_report(config, state, evaluator, output, selection, args.resume)
    log(f"Run {output.name}: {report['parameters']:,} parameters; {selection}")
    return_code = 0
    try:
        if not args.skip_preflight:
            run_preflight(state, evaluator, output, log)
        if args.preflight_only:
            report.update(status="preflight_completed", stop_reason="preflight_only")
        else:
            run_training(config, state, evaluator, output, report, log)
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      stop_reason=type(exc).__name__, error=str(exc))
        log(f"Run ended: {type(exc).__name__}: {exc}")
        return_code = 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        finalize_report(report, state, output, invocation_start, log)
    return return_code


if __name__ == "__main__":
    sys.exit(main())
