"""Profile the real pretraining path without evaluation, checkpoints, or DGL (<2049 nodes).

Example: .venv/bin/python speedrun_utils/profile_pretrain.py --device cuda --workers 4
Use --trace to collect an additional instrumented pass. Throughput is measured
separately, without profiler overhead or per-stage CUDA synchronization.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanographpfn import MAX_SDPA_GRAPH_SIZE, NanoGraphPFNModel
from nanoprior import PriorConfig, make_dataloader
from train_nano import Config, make_optimizers, task_loss


@contextmanager
def block_ranges(model):
    """Temporary method wrappers; never alter saved model or parameter names."""
    originals = []
    try:
        for layer in model.layers:
            for name in ('feature_attention', 'context_attention', 'feedforward',
                         'graph_update', 'graph_feedforward'):
                attribute = ('_compiled_' + name if getattr(layer, '_compiled_' + name, None) is not None
                             else name)
                method = getattr(layer, attribute)
                def wrapped(*args, _method=method, _name=name, **kwargs):
                    with record_function('block/' + _name):
                        return _method(*args, **kwargs)
                originals.append((layer, attribute, attribute in layer.__dict__, method))
                setattr(layer, attribute, wrapped)
        yield
    finally:
        for layer, name, instance_attribute, method in originals:
            if instance_attribute:
                setattr(layer, name, method)
            else:
                delattr(layer, name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--nodes', type=int, default=1000)
    parser.add_argument('--min-features', type=int, default=4)
    parser.add_argument('--max-features', type=int, default=100)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=768)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--profile-steps', type=int, default=5)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--compile-blocks', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--compile-projections', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--batched-muon', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--async-transfer', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--pin-memory', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--cuda-graph-muon', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--validate-on-cpu', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--log-recompiles', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('workdir/profile-pretrain'))
    args = parser.parse_args(argv)
    if min(args.steps, args.profile_steps) < 1 or args.warmup < 0:
        parser.error('steps/profile-steps must be positive and warmup nonnegative')
    if args.nodes > MAX_SDPA_GRAPH_SIZE:
        parser.error('Use <=2048 nodes for profiling without DGL')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use --device cpu for CPU-only measurements')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if args.log_recompiles:
        torch._logging.set_logs(recompiles=True)
    device = torch.device(args.device)
    prior = PriorConfig(n_nodes=args.nodes, n_features=(args.min_features, args.max_features))
    model = NanoGraphPFNModel(num_layers=args.layers, num_heads=args.heads,
                              embed_dim=args.embed_dim, hidden_dim=args.hidden_dim).to(device).train()
    if (args.compile_blocks or args.compile_projections) and device.type == 'cuda':
        model.compile_training_blocks(mlps=args.compile_blocks, projections=args.compile_projections)
    optimizers = make_optimizers(model, Config(cuda_graph_muon=args.cuda_graph_muon, batched_muon=args.batched_muon))
    optimizers.adamw.train()
    total = args.warmup + args.steps + (args.profile_steps if args.trace else 0)
    loader = make_dataloader(prior, num_tasks=total, seed=args.seed, num_workers=args.workers,
                             pin_memory=args.pin_memory and device.type == 'cuda')
    iterator = iter(loader)
    shapes = []
    losses = []

    def step():
        with record_function('train/prior_wait'):
            task = next(iterator)
        shapes.append({'nodes': len(task.x), 'features': task.x.shape[1], 'context': task.sep,
                       'edges': task.edge_index.shape[1], 'task': task.task_type})
        with record_function('train/zero_grad'):
            optimizers.zero_grad()
        with record_function('train/transfer_forward_loss'):
            loss = task_loss(model, task, device, validate_on_cpu=args.validate_on_cpu,
                             async_transfer=args.async_transfer)
        # Match train_epoch, including its current per-step finite-loss sync.
        with record_function('train/check_loss'):
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at task {task.metadata["task_index"]}: {shapes[-1]}')
        with record_function('train/backward'):
            loss.backward()
        with record_function('train/clip_grad'):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        with record_function('train/muon'):
            optimizers.muon.step()
        with record_function('train/adamw'):
            optimizers.adamw.step()
        losses.append(loss.detach())

    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    warmup_start = time.perf_counter()
    for _ in range(args.warmup):
        step()
    synchronize()
    warmup_seconds = time.perf_counter() - warmup_start
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(args.steps):
        step()
    synchronize()
    seconds = time.perf_counter() - start
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'torch': torch.__version__, 'device': str(device),
              'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
              'arguments': {**vars(args), 'output': str(args.output)}, 'prior': asdict(prior),
              'benchmark_seconds': seconds, 'steps_per_second': args.steps / seconds,
              'warmup_seconds': warmup_seconds,
              'mean_loss': torch.stack(losses[args.warmup:]).mean().item(),
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
              'benchmark_shapes': shapes[args.warmup:].copy()}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    if args.trace:
        activities = [ProfilerActivity.CPU]
        if device.type == 'cuda':
            activities.append(ProfilerActivity.CUDA)
        with block_ranges(model), profile(activities=activities, record_shapes=True,
                                          profile_memory=True) as prof:
            for _ in range(args.profile_steps):
                step()
                prof.step()
            synchronize()
        prof.export_chrome_trace(str(args.output / 'trace.json'))
        # Kineto can emit device-side events with the same name as host ranges.
        # Keep the native table intact (2.4 needs device events for percentages),
        # but restrict the JSON range summary to host events.
        averages = prof.key_averages()
        host_averages = [e for e in averages if e.device_type == torch.autograd.DeviceType.CPU]
        sort = 'self_cuda_time_total' if device.type == 'cuda' else 'self_cpu_time_total'
        (args.output / 'operators.txt').write_text(averages.table(sort_by=sort, row_limit=60))
        ranges = [{'name': e.key, 'cpu_total_ms': e.cpu_time_total / 1000,
                   'device_total_ms': e.device_time_total / 1000, 'calls': e.count}
                  for e in host_averages if e.key.startswith(('train/', 'block/'))]
        report['profile_ranges'] = ranges
        report['profile_note'] = ('Ranges overlap/nest and must not be summed. CPU totals include launch/wait time; '
                                  'device totals attribute kernels, not wall time. Backward attribution can appear '
                                  'under autograd operators rather than its enclosing range. Native operator table '
                                  'includes device events; do not sum its rows. Trace pass uses fresh tasks.')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
