"""A small CPU prior for node classification and regression.

The pipeline is: degree-corrected SBM -> largest component -> random MLP/GNN
causal model -> observed features/target -> context/query split.

All implementation lives in this file. PyTorch and NumPy are the only external
dependencies. Nothing is sampled at import time, and training is not imported.
See README.md for the distributions, intentional differences from GraphPFN,
and examples of direct sampling and multiprocessing.

Adapted from the GraphPFN prior algorithms (Apache-2.0):
https://github.com/yandex-research/graphpfn/tree/main/paper/lib/graphpfn/prior
Copyright (c) 2026 Yandex LLC. See LICENSE and NOTICE.
This is a standalone adaptation, not an exact reproduction of that prior.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal, TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


TaskType = Literal["binclass", "multiclass", "regression"]
TASK_TYPES = ("binclass", "multiclass", "regression")
AGGREGATIONS = ("mean", "min", "max", "gcn", "attention")
ACTIVATIONS = ("identity", "tanh", "leaky_relu", "elu", "silu", "relu", "softplus", "sin", "rbf", "square")
PRIOR_VERSION = 2
Choice = TypeVar("Choice")


# Configuration and the public data contract.


@dataclass(frozen=True)
class PriorConfig:
    """Bounds are inclusive. Each task draws fresh hyperparameters.

    ``n_nodes`` is the size BEFORE largest-component extraction. The returned
    task is usually smaller. ``avg_degree`` controls the expected number of
    proposed edges; removing duplicates/self-loops changes the realized degree.
    A fixed ``task_type`` is useful for experiments; None draws each of the
    three task types with equal probability.
    """

    n_nodes: int = 1000
    n_features: tuple[int, int] = (4, 100)
    n_groups: tuple[int, int] = (2, 8)
    avg_degree: tuple[float, float] = (1.5, 100.0)
    offdiagonal_scale: float = 0.1
    context_fraction: tuple[float, float] = (0.05, 0.5)
    task_type: TaskType | None = None
    max_num_classes: int = 32
    categorical_probability: float = 0.2
    max_categories: int = 256
    max_attempts: int = 32
    # Bounds on the log-uniform mean of a positive-normal SCM draw, not hard
    # bounds on the final width/depth/std. Cause/depth/width offsets are 1/2/4.
    scm_cause_mean: tuple[float, float] = (1.0, 12.0)
    scm_depth_mean: tuple[float, float] = (1.0, 6.0)
    scm_width_mean: tuple[float, float] = (5.0, 130.0)
    scm_noise_mean: tuple[float, float] = (0.0001, 0.3)
    scm_weight_mean: tuple[float, float] = (0.01, 10.0)
    scm_graph_ratio: tuple[float, float] = (0.0, 1.0)  # 11 evenly spaced choices
    scm_dropout_scale: float = 0.9
    scm_dropout_beta: tuple[float, float] = (0.1, 5.0)
    scm_activation_scale: tuple[float, float] = (0.1, 10.0)

    def __post_init__(self) -> None:
        for name, minimum in (("n_nodes", 8), ("max_num_classes", 3), ("max_categories", 2), ("max_attempts", 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("n_features", "n_groups", "avg_degree", "context_fraction", "scm_cause_mean",
                     "scm_depth_mean", "scm_width_mean", "scm_noise_mean", "scm_weight_mean",
                     "scm_dropout_beta", "scm_activation_scale"):
            bounds = getattr(self, name)
            if len(bounds) != 2 or not all(math.isfinite(v) for v in bounds):
                raise ValueError(f"{name} must contain two finite bounds")
            if not 0 < bounds[0] <= bounds[1]:
                raise ValueError(f"{name} must satisfy 0 < lower <= upper")
        for name in ("n_features", "n_groups"):
            if any(type(value) is not int for value in getattr(self, name)):
                raise ValueError(f"{name} bounds must be integers")
        if self.n_groups[0] > self.n_nodes // 4:
            raise ValueError("n_nodes must allow at least four nodes per group")
        if self.avg_degree[0] > self.n_nodes - 1:
            raise ValueError("minimum avg_degree cannot exceed n_nodes - 1")
        if self.context_fraction[1] >= 1:
            raise ValueError("context_fraction must leave query nodes")
        if not math.isfinite(self.offdiagonal_scale) or self.offdiagonal_scale < 0:
            raise ValueError("offdiagonal_scale must be finite and nonnegative")
        if not 0 <= self.categorical_probability <= 1:
            raise ValueError("categorical_probability must be in [0, 1]")
        if self.task_type is not None and self.task_type not in TASK_TYPES:
            raise ValueError(f"task_type must be one of {TASK_TYPES}, or None")
        if (len(self.scm_graph_ratio) != 2 or not all(math.isfinite(value) for value in self.scm_graph_ratio)
                or not 0 <= self.scm_graph_ratio[0] <= self.scm_graph_ratio[1] <= 1):
            raise ValueError("scm_graph_ratio must satisfy 0 <= lower <= upper <= 1")
        if not math.isfinite(self.scm_dropout_scale) or not 0 <= self.scm_dropout_scale < 1:
            raise ValueError("scm_dropout_scale must be in [0, 1)")


@dataclass
class GraphTask:
    """One graph, with no batch dimension. All tensors are on the CPU.

    x: float32 [N, F]; categorical columns contain integer codes as floats.
    y: int64 [N] for classification, float32 [N] for regression.
    edge_index: int64 [2, E], source then destination; both directions stored.
    sep: nodes [:sep] are context; nodes [sep:] are queries.
    feature_cardinalities: int64 [F]; zero means numerical.

    Only ``context_y`` may be supplied as observed labels to a model. The full
    ``y`` is ground truth, including query labels for the training loss.
    Numerical x columns are already standardized on context nodes and clipped
    to [-100, 100]. Pass features_normalized=True when calling the model.
    """

    x: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor
    sep: int
    task_type: TaskType
    num_classes: int | None
    feature_cardinalities: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def context_y(self) -> torch.Tensor:
        return self.y[: self.sep]

    @property
    def query_y(self) -> torch.Tensor:
        return self.y[self.sep :]

    def pin_memory(self) -> GraphTask:
        """Let DataLoader's pinning thread stage the custom task container."""
        return replace(self, x=self.x.pin_memory(), y=self.y.pin_memory(),
                       edge_index=self.edge_index.pin_memory(),
                       feature_cardinalities=self.feature_cardinalities.pin_memory())


class SamplingError(RuntimeError):
    """Sampling exhausted its retry budget; the message describes failures."""


class _InvalidSample(Exception):
    """An expected numerical or statistical rejection, not a programming error."""


@dataclass
class _RandomState:
    """Local RNGs: sampling never seeds or consumes the caller's global RNGs."""

    numpy: np.random.Generator
    torch: torch.Generator

    @classmethod
    def from_seed(cls, seed: int) -> _RandomState:
        numpy_seed, torch_seed = np.random.SeedSequence(seed).spawn(2)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(torch_seed.generate_state(1, dtype=np.uint64)[0]))
        return cls(np.random.default_rng(numpy_seed), generator)

    def integer(self, low: int, high: int) -> int:
        return int(self.numpy.integers(low, high + 1))

    def coin(self) -> bool:
        return bool(self.numpy.random() < 0.5)

    def choice(self, values: tuple[Choice, ...]) -> Choice:
        return values[self.integer(0, len(values) - 1)]

    def log_uniform(self, low: float, high: float) -> float:
        return float(np.exp(self.numpy.uniform(np.log(low), np.log(high))))

    def normal(self, *shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=self.torch, dtype=torch.float32)

    def uniform(self, *shape: int) -> torch.Tensor:
        return torch.rand(shape, generator=self.torch, dtype=torch.float32)

    def permutation(self, size: int) -> torch.Tensor:
        return torch.randperm(size, generator=self.torch)


# Degree-corrected stochastic block model.


def _partition_nodes(n_nodes: int, n_groups: int, rng: _RandomState) -> np.ndarray:
    """Random unequal group sizes with at least four nodes in each group."""
    minimum = max(4, int(0.1 * n_nodes / n_groups))
    remainder = n_nodes - (minimum - 1) * n_groups
    cuts = rng.numpy.choice(np.arange(1, remainder), n_groups - 1, replace=False)
    return np.diff(np.r_[0, np.sort(cuts), remainder]) + minimum - 1


def _sample_block_edge_counts(
    sizes: np.ndarray, avg_degree: float, offdiagonal_scale: float, rng: _RandomState
) -> np.ndarray:
    """Poisson edge counts for the upper triangle, including within-group edges.

    Group-pair mass is proportional to affinity * size_i * size_j, following
    GraphPFN. Its total is N * avg_degree / 2 before graph simplification.
    """
    n_groups = len(sizes)
    uniform_affinities = rng.coin()
    diagonal = np.ones(n_groups) if uniform_affinities else rng.numpy.random(n_groups)
    affinities = np.diag(diagonal)
    shared_affinity = rng.numpy.random()
    for first in range(n_groups):
        for second in range(first + 1, n_groups):
            affinity = shared_affinity if uniform_affinities else rng.numpy.random() * math.sqrt(
                diagonal[first] * diagonal[second]
            )
            affinities[first, second] = offdiagonal_scale * affinity
    mass = affinities * sizes[:, None] * sizes[None, :]
    expected_counts = mass / mass.sum() * (sizes.sum() * avg_degree / 2)
    return rng.numpy.poisson(expected_counts)


def _sample_distinct_pairs(weights: np.ndarray, count: int, rng: _RandomState) -> tuple[np.ndarray, np.ndarray]:
    """Draw weighted endpoints without replacement within each edge.

    The second endpoint excludes the first. Removing the first endpoint's
    interval from the probability CDF avoids an unbounded rejection loop when
    a hub has almost all the probability mass.
    """
    first = rng.numpy.choice(len(weights), count, p=weights)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    interval_start = cumulative[first] - weights[first]
    draws = rng.numpy.random(count) * (1.0 - weights[first])
    draws += np.where(draws >= interval_start, weights[first], 0.0)
    second = np.searchsorted(cumulative, np.minimum(draws, np.nextafter(1.0, 0.0)), side="right")
    return first, second


def _simple_undirected_edges(source: np.ndarray, target: np.ndarray, n_nodes: int) -> torch.Tensor:
    """Deduplicate unordered pairs, then store both directions in sorted order."""
    keep = source != target
    low = np.minimum(source[keep], target[keep])
    high = np.maximum(source[keep], target[keep])
    pairs = np.unique(low * n_nodes + high)
    low, high = pairs // n_nodes, pairs % n_nodes
    directed = np.sort(np.r_[low * n_nodes + high, high * n_nodes + low])
    return torch.from_numpy(np.stack((directed // n_nodes, directed % n_nodes)).astype(np.int64))


def _largest_component(edge_index: torch.Tensor, n_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return relabeled edges and old node IDs of the largest component.

    Ties are resolved by the smallest old node ID. All node attributes are
    generated AFTER this step, so there are no feature/label arrays to slice.
    """
    neighbors = [[] for _ in range(n_nodes)]
    for source, target in edge_index.t().tolist():
        neighbors[source].append(target)
    visited = [False] * n_nodes
    largest = []
    for start in range(n_nodes):
        if visited[start]:
            continue
        component = [start]
        visited[start] = True
        for node in component:
            for neighbor in neighbors[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    component.append(neighbor)
        if len(component) > len(largest):
            largest = component
    retained = torch.tensor(sorted(largest), dtype=torch.int64)
    old_to_new = torch.full((n_nodes,), -1, dtype=torch.int64)
    old_to_new[retained] = torch.arange(len(retained))
    keep = (old_to_new[edge_index[0]] >= 0) & (old_to_new[edge_index[1]] >= 0)
    return old_to_new[edge_index[:, keep]], retained


def _sample_graph(config: PriorConfig, rng: _RandomState) -> tuple[torch.Tensor, int, dict]:
    n_groups = rng.integer(config.n_groups[0], min(config.n_groups[1], config.n_nodes // 4))
    sizes = _partition_nodes(config.n_nodes, n_groups, rng)
    avg_degree = rng.log_uniform(config.avg_degree[0], min(config.avg_degree[1], config.n_nodes - 1))
    counts = _sample_block_edge_counts(sizes, avg_degree, config.offdiagonal_scale, rng)
    degree_distribution = "zipf" if rng.coin() else "uniform"
    if degree_distribution == "zipf":
        propensity = rng.numpy.zipf(rng.numpy.uniform(2.0, 3.0), config.n_nodes).astype(np.float64)
    else:
        propensity = rng.numpy.random(config.n_nodes) + np.finfo(np.float64).eps
    offsets = np.r_[0, np.cumsum(sizes)]
    weights = [propensity[offsets[i] : offsets[i + 1]].copy() for i in range(n_groups)]
    weights = [values / values.sum() for values in weights]
    source_parts, target_parts = [], []
    for first in range(n_groups):
        for second in range(first, n_groups):
            count = int(counts[first, second])
            if first == second:
                source, target = _sample_distinct_pairs(weights[first], count, rng)
            else:
                source = rng.numpy.choice(sizes[first], count, p=weights[first])
                target = rng.numpy.choice(sizes[second], count, p=weights[second])
            source_parts.append(source + offsets[first])
            target_parts.append(target + offsets[second])
    edges = _simple_undirected_edges(np.concatenate(source_parts), np.concatenate(target_parts), config.n_nodes)
    edges, retained = _largest_component(edges, config.n_nodes)
    n_nodes = len(retained)
    if n_nodes < 4:
        raise _InvalidSample("largest component has fewer than four nodes")
    # Group indices must not determine context membership or appear as features.
    old_to_new = rng.permutation(n_nodes)
    edges = old_to_new[edges]
    metadata = {
        "requested_nodes": config.n_nodes,
        "n_groups": n_groups,
        "requested_avg_degree": avg_degree,
        "degree_distribution": degree_distribution,
    }
    return edges, n_nodes, metadata


# Graph operations. Sparse matrices avoid allocating an N x N x hidden array.


@dataclass
class _Graph:
    edges: torch.Tensor
    n_nodes: int
    degree: torch.Tensor = field(init=False)
    adjacency: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        source, target = self.edges
        self.degree = torch.bincount(target, minlength=self.n_nodes).float()
        # Rows are destinations: adjacency @ x sums source-node features.
        self.adjacency = torch.sparse_coo_tensor(
            torch.stack((target, source)), torch.ones(len(source), dtype=torch.float32),
            (self.n_nodes, self.n_nodes), check_invariants=True,
        ).coalesce()

    def sum_neighbors(self, values: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(self.adjacency, values)


def _aggregate(graph: _Graph, values: torch.Tensor, kind: str, rng: _RandomState) -> torch.Tensor:
    degrees = graph.degree.clamp_min(1).unsqueeze(1)
    if kind == "mean":
        return graph.sum_neighbors(values) / degrees
    if kind == "gcn":
        return graph.sum_neighbors(values / degrees.sqrt()) / degrees.sqrt()
    if kind == "attention":
        return _attention_neighbors(graph, values, rng)
    if kind not in ("min", "max"):
        raise ValueError(f"Unknown aggregation: {kind}")
    fill = math.inf if kind == "min" else -math.inf
    result = torch.full_like(values, fill)
    # Chunking bounds temporary message storage even for dense graphs.
    for edge_chunk in graph.edges.split(4096, dim=1):
        source, target = edge_chunk
        indices = target[:, None].expand(-1, values.shape[1])
        result.scatter_reduce_(0, indices, values[source], reduce="amin" if kind == "min" else "amax")
    result[graph.degree == 0] = 0
    return result


def _attention_neighbors(graph: _Graph, values: torch.Tensor, rng: _RandomState) -> torch.Tensor:
    """One-head random neighborhood attention, including a self contribution."""
    width = values.shape[1]
    query, key, value = (values @ (rng.normal(width, 3 * width) / math.sqrt(width))).chunk(3, dim=1)
    nodes = torch.arange(graph.n_nodes)
    edges = torch.cat((graph.edges, torch.stack((nodes, nodes))), dim=1)
    score_parts = []
    for source, target in edges.split(4096, dim=1):
        score_parts.append((key[source] * query[target]).sum(dim=1) / math.sqrt(width))
    scores = torch.cat(score_parts)
    source, target = edges
    maximum = torch.full((graph.n_nodes,), -math.inf, dtype=torch.float32)
    maximum.scatter_reduce_(0, target, scores, reduce="amax")
    weights = torch.exp(scores - maximum[target])
    normalizer = torch.zeros(graph.n_nodes, dtype=torch.float32).index_add_(0, target, weights)
    weights = weights / normalizer[target]
    attention = torch.sparse_coo_tensor(
        torch.stack((target, source)), weights, graph.adjacency.shape, check_invariants=True,
    ).coalesce()
    return torch.sparse.mm(attention, value) @ (rng.normal(width, width) / math.sqrt(width))


def _structural_inputs(graph: _Graph, use_degree: bool, use_pagerank: bool, rng: _RandomState) -> torch.Tensor:
    columns = []
    if use_degree:
        columns.append(torch.log1p(graph.degree))
    if use_pagerank:
        damping = float(rng.numpy.uniform(0.6, 0.9))
        rank = torch.full((graph.n_nodes, 1), 1 / graph.n_nodes, dtype=torch.float32)
        for _ in range(30):
            new_rank = (1 - damping) / graph.n_nodes + damping * graph.sum_neighbors(
                rank / graph.degree.clamp_min(1).unsqueeze(1)
            )
            if (new_rank - rank).abs().sum().item() < 1e-6:
                rank = new_rank
                break
            rank = new_rank
        columns.append(torch.log(rank[:, 0] + 1e-6))
    return torch.stack(columns, dim=1) if columns else torch.empty((graph.n_nodes, 0), dtype=torch.float32)


# Random neural structural causal model (SCM).


def _meta_positive(mean_bounds: tuple[float, float], rng: _RandomState) -> float:
    """GraphPFN-style positive normal draw with log-uniform mean/relative std."""
    mean = rng.log_uniform(*mean_bounds)
    std = mean * rng.log_uniform(0.01, 1.0)
    for _ in range(100):
        value = float(rng.numpy.normal(mean, std))
        if value > 0:
            return value
    raise _InvalidSample("positive normal sampler exhausted its attempts")


def _sample_causes(n_nodes: int, width: int, rng: _RandomState) -> torch.Tensor:
    strategy = rng.choice(("normal", "uniform", "mixed"))
    if strategy == "normal":
        values = rng.normal(n_nodes, width)
        if rng.coin():
            mean = rng.normal(width)
            values = values * (rng.normal(width) * mean).abs() + mean
        return values
    if strategy == "uniform":
        return rng.uniform(n_nodes, width)
    columns = []
    for _ in range(width):
        kind = rng.choice(("normal", "uniform", "categorical", "zipf"))
        if kind == "normal":
            column = rng.normal(n_nodes)
        elif kind == "uniform":
            column = rng.uniform(n_nodes)
        elif kind == "categorical":
            probabilities = rng.uniform(rng.integer(2, 20))
            column = torch.multinomial(probabilities, n_nodes, replacement=True, generator=rng.torch).float()
            column = (column - column.mean()) / column.std(correction=0).clamp_min(1e-6)
        else:
            values = rng.numpy.zipf(rng.numpy.uniform(2.0, 4.0), n_nodes)
            column = torch.from_numpy(np.minimum(values, 10).astype(np.float32))
            column = column - column.mean()
        columns.append(column)
    return torch.stack(columns, dim=1)


def _random_weights(
    n_inputs: int, n_outputs: int, std: float, dropout: float, block_dropout: bool, rng: _RandomState
) -> torch.Tensor:
    if block_dropout:
        n_blocks = rng.integer(1, math.ceil(math.sqrt(min(n_inputs, n_outputs))))
        block_rows, block_cols = n_inputs // n_blocks, n_outputs // n_blocks
        keep_fraction = n_blocks * block_rows * block_cols / (n_inputs * n_outputs)
        weights = torch.zeros((n_inputs, n_outputs), dtype=torch.float32)
        for block in range(n_blocks):
            rows = slice(block * block_rows, (block + 1) * block_rows)
            cols = slice(block * block_cols, (block + 1) * block_cols)
            weights[rows, cols] = rng.normal(block_rows, block_cols) * std / math.sqrt(keep_fraction)
        return weights
    keep = rng.uniform(n_inputs, n_outputs) >= dropout
    return rng.normal(n_inputs, n_outputs) * keep * std / math.sqrt(1 - dropout)


def _activate(values: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "identity":
        return values
    if kind == "tanh":
        return torch.tanh(values)
    if kind == "leaky_relu":
        return F.leaky_relu(values)
    if kind == "elu":
        return F.elu(values)
    if kind == "silu":
        return F.silu(values)
    if kind == "relu":
        return F.relu(values)
    if kind == "softplus":
        return F.softplus(values)
    if kind == "sin":
        return torch.sin(values)
    if kind == "rbf":
        return torch.exp(-values.square())
    if kind == "square":
        return values.square()
    raise ValueError(f"Unknown activation: {kind}")


def _choose_observed(
    states: list[torch.Tensor], n_features: int, target_is_effect: bool, in_clique: bool, rng: _RandomState
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select distinct latent coordinates for features and the scalar target."""
    hidden = torch.cat(states, dim=1)
    available = hidden.shape[1] - int(target_is_effect)
    n_selected = n_features + int(not target_is_effect)
    if in_clique:
        start = rng.integer(0, available - n_selected)
        selected = start + rng.permutation(n_selected)
    else:
        selected = rng.permutation(available)[:n_selected]
    if target_is_effect:
        feature_indices, target_index = selected, hidden.shape[1] - 1
    else:
        feature_indices, target_index = selected[1:], int(selected[0])
    return hidden[:, feature_indices], hidden[:, target_index]


def _sample_attributes(graph: _Graph, n_features: int, rng: _RandomState,
                       config: PriorConfig) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Draw an SCM and return observed features [N, F], raw target [N] and its settings."""
    causal = rng.coin()
    n_causes = 1 + round(_meta_positive(config.scm_cause_mean, rng)) if causal else n_features
    n_layers = 2 + round(_meta_positive(config.scm_depth_mean, rng))
    width = 4 + round(_meta_positive(config.scm_width_mean, rng))
    if causal:
        width = max(width, 2 * n_features + 1)
    ratio_low, ratio_high = config.scm_graph_ratio
    graph_ratio = ratio_low + (ratio_high - ratio_low) * (rng.integer(0, 10) / 10)
    aggregation = rng.choice(AGGREGATIONS)
    use_degree, use_pagerank = rng.coin(), rng.coin()
    noise_std = _meta_positive(config.scm_noise_mean, rng)
    weight_std = _meta_positive(config.scm_weight_mean, rng)
    dropout = min(0.99, config.scm_dropout_scale * rng.numpy.beta(
        rng.numpy.uniform(*config.scm_dropout_beta), rng.numpy.uniform(*config.scm_dropout_beta),
    ))
    block_dropout, per_feature_noise = rng.coin(), rng.coin()
    shared_activation = rng.choice(ACTIVATIONS) if rng.coin() else None
    causes = _sample_causes(graph.n_nodes, n_causes, rng)
    structural = _structural_inputs(graph, use_degree, use_pagerank, rng)
    values = torch.cat((causes, structural), dim=1)
    states = []
    output_widths = [width] * n_layers + ([] if causal else [1])
    for layer_index, output_width in enumerate(output_widths):
        if layer_index:
            activation = shared_activation or rng.choice(ACTIVATIONS)
            values = _activate(values * rng.log_uniform(*config.scm_activation_scale), activation)
        weights = _random_weights(
            values.shape[1], output_width, weight_std,
            dropout if layer_index else 0.0, block_dropout, rng,
        )
        bias = (2 * rng.uniform(output_width) - 1) / math.sqrt(values.shape[1])
        values = values @ weights + bias
        graph_channels = rng.uniform(output_width) < graph_ratio
        if graph_channels.any():
            messages = _aggregate(graph, values, aggregation, rng)
            values = torch.where(graph_channels, messages, values)
        if layer_index:
            noise_scale = noise_std * rng.normal(output_width).abs() if per_feature_noise else noise_std
            values = values + rng.normal(graph.n_nodes, output_width) * noise_scale
            states.append(values)
        if not torch.isfinite(values).all():
            raise _InvalidSample("nonfinite SCM state")
    target_is_effect, in_clique = rng.coin(), rng.coin()
    if causal:
        features, target = _choose_observed(states, n_features, target_is_effect, in_clique, rng)
    else:
        features, target = causes, states[-1][:, 0]
    metadata = {
        "causal": causal, "n_layers": n_layers, "hidden_width": width,
        "graph_ratio": graph_ratio, "aggregation": aggregation,
        "use_degree": use_degree, "use_pagerank": use_pagerank,
    }
    return features, target, metadata


# Targets, context membership, and observed feature processing.


def _make_target(raw: torch.Tensor, task_type: TaskType, num_classes: int | None, rng: _RandomState) -> torch.Tensor:
    # Float64 statistics avoid overflow on finite but large SCM outputs.
    values = raw.double()
    spread = values.std(correction=0)
    if spread <= 1e-12 or not torch.isfinite(spread):
        raise _InvalidSample("constant or numerically degenerate target")
    if task_type == "regression":
        # Regression scaling happens after splitting, using context only.
        return raw.clone()
    values = (values - values.mean()) / spread
    if task_type == "binclass":
        boundary = torch.quantile(values, float(rng.numpy.uniform(0.01, 0.99)))
        labels = (values > boundary).long()
    else:
        assert num_classes is not None
        # Distinct rank boundaries prevent empty classes from repeated cuts.
        # Tied raw target values remain tied; never invent a label from node ID.
        unique_values = values.unique(sorted=True)
        if len(unique_values) < num_classes:
            raise _InvalidSample("too few distinct target values for the requested classes")
        boundary_indices = rng.permutation(len(unique_values) - 1)[: num_classes - 1]
        boundaries = unique_values[boundary_indices].sort().values
        labels = torch.searchsorted(boundaries, values.contiguous(), right=False)
    assert num_classes is not None
    if labels.unique().numel() != num_classes:
        raise _InvalidSample("target conversion lost a class")
    return rng.permutation(num_classes)[labels]


def _split_order(labels: torch.Tensor, sep: int, task_type: TaskType, rng: _RandomState) -> torch.Tensor:
    """Return old node IDs in new order, with context first.

    Classification reserves one random member of each class for context.
    Remaining context nodes are random. Query classes need not all be present,
    but every query class is guaranteed to occur in context.
    """
    order = rng.permutation(len(labels))
    if task_type == "regression":
        return order
    reserved = []
    for label in labels.unique():
        reserved.append(order[labels[order] == label][0])
    reserved = torch.stack(reserved)
    if len(reserved) > sep:
        raise _InvalidSample("context is too small to contain every class")
    available = torch.ones(len(labels), dtype=torch.bool)
    available[reserved] = False
    remaining = order[available[order]]
    context = torch.cat((reserved, remaining[: sep - len(reserved)]))
    context = context[rng.permutation(sep)]
    return torch.cat((context, remaining[sep - len(reserved) :]))


def _permute_nodes(
    features: torch.Tensor, labels: torch.Tensor, edges: torch.Tensor, order: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    old_to_new = torch.empty_like(order)
    old_to_new[order] = torch.arange(len(order))
    return features[order], labels[order], old_to_new[edges]


def _process_features(
    features: torch.Tensor, sep: int, config: PriorConfig, rng: _RandomState
) -> tuple[torch.Tensor, torch.Tensor]:
    features = features.clone()
    cardinalities = torch.zeros(features.shape[1], dtype=torch.int64)
    if rng.numpy.random() < config.categorical_probability:
        column_probability = rng.numpy.random()
        for column in range(features.shape[1]):
            if rng.numpy.random() >= column_probability:
                continue
            count = min(max(round(rng.numpy.gamma(1, 10)), 2), config.max_categories)
            indices = torch.randint(len(features), (count - 1,), generator=rng.torch)
            boundaries = features[indices, column].sort().values
            codes = torch.searchsorted(boundaries, features[:, column].contiguous(), right=False)
            _, codes = codes.unique(return_inverse=True)
            count = int(codes.max()) + 1
            features[:, column] = rng.permutation(count)[codes].float()
            cardinalities[column] = count
    # Drop columns constant in context, as they provide no within-context signal.
    context = features[:sep]
    keep = context.amax(dim=0) > context.amin(dim=0)
    features, cardinalities = features[:, keep], cardinalities[keep]
    if features.shape[1] < config.n_features[0]:
        raise _InvalidSample("too few nonconstant context features")
    numerical = cardinalities == 0
    if numerical.any():
        values = features[:, numerical].double()
        context = values[:sep]
        values = (values - context.mean(dim=0)) / context.std(dim=0, correction=0).clamp_min(1e-12)
        features[:, numerical] = values.clamp(-100, 100).float()
    order = rng.permutation(features.shape[1])
    return features[:, order].contiguous(), cardinalities[order]


def _validate_task(task: GraphTask) -> None:
    """Check the public contract. Only expected sample failures are retried."""
    n_nodes = len(task.x)
    if task.x.ndim != 2 or task.y.shape != (n_nodes,):
        raise AssertionError("invalid feature or target shape")
    if task.x.dtype != torch.float32 or task.feature_cardinalities.shape != (task.x.shape[1],):
        raise AssertionError("invalid feature dtype or metadata")
    if not 2 <= task.sep < n_nodes:
        raise _InvalidSample("empty query or fewer than two context nodes")
    if not torch.isfinite(task.x).all() or not torch.isfinite(task.y).all():
        raise _InvalidSample("nonfinite observed features or targets")
    if task.task_type == "regression":
        if task.y.dtype != torch.float32 or task.num_classes is not None:
            raise AssertionError("invalid regression target contract")
        if task.context_y.double().std(correction=0) <= 1e-12:
            raise _InvalidSample("constant context regression target")
    else:
        if task.y.dtype != torch.int64 or task.num_classes is None:
            raise AssertionError("invalid classification target contract")
        expected = torch.arange(task.num_classes)
        if not torch.equal(task.context_y.unique(sorted=True), expected):
            raise _InvalidSample("context does not cover all classes")
    if task.edge_index.shape[0] != 2 or task.edge_index.dtype != torch.int64:
        raise AssertionError("invalid edge tensor")
    if task.edge_index.numel() == 0 or task.edge_index.min() < 0 or task.edge_index.max() >= n_nodes:
        raise AssertionError("edges reference an absent node")


@torch.no_grad()
def sample_task(config: PriorConfig | None = None, *, seed: int = 0) -> GraphTask:
    """Sample one CPU task without touching global random state.

    The task type is fixed before retries, so failures do not silently change
    the requested task mixture. Numerical failures resample the other task
    parameters, up to ``max_attempts``. Programming errors are never swallowed.
    """
    config = PriorConfig() if config is None else config
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = _RandomState.from_seed(seed)
    task_type = config.task_type or rng.choice(TASK_TYPES)
    rejections: Counter[str] = Counter()
    for attempt in range(1, config.max_attempts + 1):
        try:
            edges, n_nodes, graph_metadata = _sample_graph(config, rng)
            context_fraction = float(rng.numpy.uniform(*config.context_fraction))
            sep = max(2, min(n_nodes - 1, int(n_nodes * context_fraction)))
            num_classes = None
            if task_type == "binclass":
                num_classes = 2
            elif task_type == "multiclass":
                limit = min(config.max_num_classes, sep)
                if limit < 3:
                    raise _InvalidSample("component is too small for multiclass context")
                num_classes = rng.integer(3, limit)
            n_features = rng.integer(*config.n_features)
            features, raw_target, scm_metadata = _sample_attributes(_Graph(edges, n_nodes), n_features, rng, config)
            labels = _make_target(raw_target, task_type, num_classes, rng)
            order = _split_order(labels, sep, task_type, rng)
            features, labels, edges = _permute_nodes(features, labels, edges, order)
            features, cardinalities = _process_features(features, sep, config, rng)
            target_mean, target_scale = 0.0, 1.0
            if task_type == "regression":
                context = labels[:sep].double()
                target_mean = context.mean().item()
                target_scale = context.std(correction=0).item()
                if target_scale <= 1e-12:
                    raise _InvalidSample("constant context regression target")
                labels = ((labels.double() - target_mean) / target_scale).float()
            task = GraphTask(
                features, labels.contiguous(), edges.contiguous(), sep, task_type, num_classes, cardinalities,
                metadata={
                    "prior_version": PRIOR_VERSION, "seed": seed, "attempts": attempt,
                    "rejections": dict(rejections), "config": asdict(config),
                    "graph": graph_metadata, "scm": scm_metadata,
                    "target_mean": target_mean, "target_scale": target_scale,
                },
            )
            _validate_task(task)
            return task
        except _InvalidSample as error:
            rejections[str(error)] += 1
    raise SamplingError(
        f"Failed to sample {task_type} with seed={seed} after {config.max_attempts} attempts: {dict(rejections)}"
    )


# CPU worker loader. Map-style indexing makes seeds independent of scheduling.


class _TaskDataset(Dataset[GraphTask]):
    def __init__(self, config: PriorConfig, seed: int, start_index: int, num_tasks: int) -> None:
        self.config = config
        self.seed = seed
        self.start_index = start_index
        self.num_tasks = num_tasks

    def __len__(self) -> int:
        return self.num_tasks

    def __getitem__(self, index: int) -> GraphTask:
        if not 0 <= index < self.num_tasks:
            raise IndexError(index)
        task_index = self.start_index + index
        seed = int(np.random.SeedSequence([self.seed, task_index]).generate_state(1, dtype=np.uint64)[0])
        task = sample_task(self.config, seed=seed)
        task.metadata["task_index"] = task_index
        return task


def _worker_init(worker_id: int) -> None:
    # Multiprocessing plus multithreaded BLAS can otherwise oversubscribe CPUs.
    torch.set_num_threads(1)


def make_dataloader(
    config: PriorConfig | None = None, *, num_tasks: int, seed: int = 0,
    start_index: int = 0, num_workers: int = 0, prefetch_factor: int = 2, pin_memory: bool = False,
) -> DataLoader[GraphTask]:
    """Yield individual GraphTasks; variable component sizes are never padded.

    Use a main guard when num_workers > 0 (workers use spawn). To resume, save
    the next CONSUMED task index and pass it as start_index. Prefetched tasks
    are regenerated with the same seeds. Iterating this loader again replays
    the same index range; a new range needs a new loader.
    """
    for name, value, minimum in (
        ("num_tasks", num_tasks, 1), ("seed", seed, 0), ("start_index", start_index, 0),
        ("num_workers", num_workers, 0), ("prefetch_factor", prefetch_factor, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    dataset = _TaskDataset(PriorConfig() if config is None else config, seed, start_index, num_tasks)
    worker_options = {}
    if num_workers:
        worker_options = {"multiprocessing_context": "spawn", "prefetch_factor": prefetch_factor}
    # DataLoader also needs a private generator for its own worker base seeds.
    loader_generator = torch.Generator().manual_seed(seed % (2**64))
    return DataLoader(
        dataset, batch_size=None, num_workers=num_workers, worker_init_fn=_worker_init,
        generator=loader_generator, pin_memory=pin_memory, **worker_options,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--nodes", type=int, default=1000)
    parser.add_argument("--task", choices=TASK_TYPES, default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = PriorConfig(n_nodes=args.nodes, task_type=args.task)
    loader = make_dataloader(config, num_tasks=args.samples, seed=args.seed, num_workers=args.workers)
    start = time.perf_counter()
    for task in loader:
        degree = torch.bincount(task.edge_index[0], minlength=len(task.x))
        report = {
            "task_index": task.metadata["task_index"], "seed": task.metadata["seed"],
            "task_type": task.task_type, "nodes": len(task.x), "features": task.x.shape[1],
            "undirected_edges": task.edge_index.shape[1] // 2, "context_nodes": task.sep,
            "mean_degree": degree.float().mean().item(), "max_degree": int(degree.max()),
            "categorical_features": int((task.feature_cardinalities > 0).sum()),
            "num_classes": task.num_classes, "attempts": task.metadata["attempts"],
            "rejections": task.metadata["rejections"],
            "scm": task.metadata["scm"],
        }
        if task.num_classes is not None:
            report["class_counts"] = torch.bincount(task.y, minlength=task.num_classes).tolist()
            source, target = task.edge_index
            report["edge_label_agreement"] = (task.y[source] == task.y[target]).float().mean().item()
        print(json.dumps(report))
    print(json.dumps({"samples": args.samples, "seconds_including_worker_startup": time.perf_counter() - start}))


if __name__ == "__main__":
    _main()
