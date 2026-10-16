"""NanoTabPFN backbone with GraphPFN-style graph adapters and LimiX-style regression.

One graph per call; context nodes must come first. Only context labels are inputs.
Training and inference share the same full-graph forward pass.
See NOTICE for the upstream architecture references.

Shapes: N nodes, F input features, T=F+1 tokens (last token is the label),
D embedding channels, H heads, K=D/H channels per head, C context nodes.
Edges are int64 [2, E], with source nodes in row 0 and destinations in row 1.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TYPE_CHECKING

import torch
from torch import Tensor, nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from dgl import DGLGraph

TaskType = Literal["binclass", "multiclass", "regression"]
GraphMask = tuple[Tensor, Tensor]  # bool [N, N] neighbor mask, bool [N] isolates
QKV = tuple[Tensor, Tensor, Tensor]
ARCHITECTURE_FIELDS = ("num_layers", "num_heads", "embed_dim", "hidden_dim", "max_classes")


class PackedQKVLinear(nn.Linear):
    """Project [..., D] to [..., 3D]; Muon treats Q, K, V as separate matrices."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__(embed_dim, 3 * embed_dim)


# Only graphs above this cutoff need DGL, in both training and evaluation.
MAX_SDPA_GRAPH_SIZE = 2048
# Evaluation workspace limits; set either to 0 to disable that batching axis.
# Training and gradient-enabled calls always use whole tensors.
EVAL_NODE_BATCH_SIZE = 2048
EVAL_TOKEN_BATCH_SIZE = 4


def apply_in_batches(operation: Callable[..., Tensor], x: Tensor, *, dim: int,
                     batch_size: int, **kwargs: Any) -> Tensor:
    """Apply a shape-preserving operation with bounded inference temporaries.

    The chosen axis must be independent for this operation. Slice before calling
    it so normalization and projections also fit inside the batch. Use the first
    result's dtype, which can differ from x under autocast.
    """
    if torch.is_grad_enabled() or batch_size == 0 or x.shape[dim] <= batch_size:
        return operation(x, **kwargs)
    output = None
    for start in range(0, x.shape[dim], batch_size):
        size = min(batch_size, x.shape[dim] - start)
        batch = operation(x.narrow(dim, start, size), **kwargs)
        if output is None:
            output = batch.new_empty(x.shape)
        output.narrow(dim, start, size).copy_(batch)
        del batch  # Release this result before computing the next one.
    assert output is not None
    return output


def make_neighbor_mask(edges: Tensor, n_nodes: int) -> GraphMask:
    """Incoming-neighbor SDPA mask and nodes whose aggregate must be zero."""
    mask = torch.zeros((n_nodes, n_nodes), device=edges.device, dtype=torch.bool)
    mask[edges[1], edges[0]] = True
    isolated = ~mask.any(-1)
    # Give SDPA a valid row, then zero that aggregate before projection.
    # Avoid nonzero/where's dynamically sized CUDA output and host sync.
    mask.diagonal().logical_or_(isolated)
    return mask, isolated


def make_dgl_graph(edges: Tensor, n_nodes: int) -> DGLGraph:
    """Convert only the graph adapter's topology, retaining isolated nodes."""
    try:
        import dgl
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Large-graph attention requires DGL 2.4.0 with PyTorch 2.4.0 and Python 3.12.9. "
            "Run `uv sync` to install the compatible environment from pyproject.toml."
        ) from exc
    return dgl.graph((edges[0], edges[1]), num_nodes=n_nodes, device=edges.device)


def dgl_neighbor_attention(q: Tensor, k: Tensor, v: Tensor, graph: DGLGraph) -> Tensor:
    """Exact incoming-neighbor attention using GraphPFN's fused DGL operations.

    Q/K/V and output are [N, T, H, K]. Scores are [E, T, H, 1]; no per-edge value vectors or
    dense adjacency are materialized. Keep float32 aggregation under AMP.
    No self loops are added; isolates aggregate zero.
    """
    from dgl import ops

    if graph.num_edges() == 0:
        # Keep all three projections connected to autograd for edgeless graphs.
        return q * 0 + k * 0 + v * 0
    with torch.autocast(q.device.type, enabled=False):
        scores = ops.u_dot_v(graph, k.float(), q.float()) * (q.shape[-1] ** -0.5)
        weights = ops.edge_softmax(graph, scores, norm_by="dst")
        out = ops.u_mul_e_sum(graph, v.float(), weights)
    return out.to(v.dtype)


class TargetEncoder(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        # The second input distinguishes observed labels from unknown labels.
        self.linear = nn.Linear(2, embed_dim)

    def forward(self, context_y: Tensor, n_nodes: int) -> Tensor:
        """Encode observed labels [C] into label tokens [N, D]."""
        y = context_y.float()
        values = y.mean().expand(n_nodes).clone()
        values[:len(y)] = y
        missing = torch.arange(n_nodes, device=y.device) >= len(y)
        return self.linear(torch.stack((values, missing.to(values.dtype)), dim=-1))


class GraphAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = PackedQKVLinear(embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)

    def project(self, x: Tensor) -> QKV:
        """Project [N, T, D] to Q/K/V, each [N, T, H, K]."""
        n, tokens, _ = x.shape
        return self.qkv(x).reshape(n, tokens, 3, self.num_heads, self.head_dim).unbind(2)

    def forward(self, x: Tensor, edges: Tensor, mask: GraphMask | None = None, *,
                graph: DGLGraph | None = None) -> Tensor:
        """Attend over incoming graph neighbors, preserving [N, T, D]."""
        return self.attend(*self.project(x), edges, mask, graph=graph)

    def attend(self, q: Tensor, k: Tensor, v: Tensor, edges: Tensor, mask: GraphMask | None = None, *,
               graph: DGLGraph | None = None) -> Tensor:
        """Keep SDPA/DGL eager while projection and QKV unpacking can compile."""
        if mask is None and graph is None:
            if len(q) <= MAX_SDPA_GRAPH_SIZE:
                mask = make_neighbor_mask(edges, len(q))
            else:
                graph = make_dgl_graph(edges, len(q))
        if mask is not None:
            neighbor_mask, isolated = mask
            out = F.scaled_dot_product_attention(
                q.permute(1, 2, 0, 3), k.permute(1, 2, 0, 3), v.permute(1, 2, 0, 3),
                attn_mask=neighbor_mask,
            ).permute(2, 0, 1, 3)
            out = out.masked_fill(isolated[:, None, None, None], 0)
        else:
            assert graph is not None
            out = dgl_neighbor_attention(q, k, v, graph)
        return self.output(out.flatten(-2))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, num_heads: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.feature_qkv = PackedQKVLinear(embed_dim)
        self.context_qkv = PackedQKVLinear(embed_dim)
        self.feature_output = nn.Linear(embed_dim, embed_dim)
        self.context_output = nn.Linear(embed_dim, embed_dim)
        self.feature_norm = nn.LayerNorm(embed_dim)
        self.context_norm = nn.LayerNorm(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp_in = nn.Linear(embed_dim, hidden_dim)
        self.mlp_out = nn.Linear(hidden_dim, embed_dim)
        self.graph_attention_norm = nn.LayerNorm(embed_dim)
        self.graph_attention = GraphAttention(embed_dim, num_heads)
        self.graph_mlp_norm = nn.LayerNorm(embed_dim)
        self.graph_mlp = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim), nn.GELU(), nn.Linear(2 * embed_dim, embed_dim),
        )
        self._compiled_feedforward = None
        self._compiled_graph_feedforward = None
        self._compiled_project_features = None
        self._compiled_project_context = None
        self._compiled_project_graph = None

    def project_features(self, x: Tensor) -> QKV:
        """Unpack within the compiled region to fuse QKV backward concatenation."""
        n, tokens, _ = x.shape
        qkv = self.feature_qkv(self.feature_norm(x)).reshape(n, tokens, 3, self.num_heads, self.head_dim)
        return qkv.permute(2, 0, 3, 1, 4).unbind(0)

    def feature_attention(self, x: Tensor) -> Tensor:
        """[N, T, D] -> [N, T, D]; attention Q/K/V are [N, H, T, K]."""
        n, tokens, embed_dim = x.shape
        projection = (self._compiled_project_features if self.training and torch.is_grad_enabled()
                      and self._compiled_project_features is not None else self.project_features)
        q, k, v = projection(x)
        # PyTorch 2.4 fused CUDA attention cannot launch more than 65,535 batches.
        # Nodes are independent batches here; each call still sees every feature.
        if q.is_cuda and n > 65535:
            out = torch.cat([
                F.scaled_dot_product_attention(qi, ki, vi)
                for qi, ki, vi in zip(q.split(32768), k.split(32768), v.split(32768))
            ], dim=0)
        else:
            out = F.scaled_dot_product_attention(q, k, v)
        return self.feature_output(out.transpose(1, 2).reshape(n, tokens, embed_dim))

    def project_context(self, x: Tensor) -> QKV:
        """Project [N, T, D] to Q/K/V, each [T, H, N, K]."""
        n, tokens, _ = x.shape
        qkv = self.context_qkv(self.context_norm(x)).reshape(n, tokens, 3, self.num_heads, self.head_dim)
        return qkv.permute(2, 1, 3, 0, 4).unbind(0)

    def context_attention(self, x: Tensor, context_size: int) -> Tensor:
        """All N nodes attend to the first C context nodes, independently per token."""
        # Keep QKV local so inference releases it before the MLP/graph adapters.
        projection = (self._compiled_project_context if self.training and torch.is_grad_enabled()
                      and self._compiled_project_context is not None else self.project_context)
        q, k, v = projection(x)
        update = F.scaled_dot_product_attention(q, k[:, :, :context_size], v[:, :, :context_size])
        return self.context_output(update.permute(2, 0, 1, 3).flatten(-2))

    def feedforward(self, x: Tensor) -> Tensor:
        return self.mlp_out(F.gelu(self.mlp_in(self.mlp_norm(x))))

    def graph_update(self, x: Tensor, edges: Tensor, graph_mask: GraphMask | None,
                     graph: DGLGraph | None) -> Tensor:
        if self.training and torch.is_grad_enabled() and self._compiled_project_graph is not None:
            return self.graph_attention.attend(*self._compiled_project_graph(x), edges, graph_mask, graph=graph)
        return self.graph_attention(self.graph_attention_norm(x), edges, graph_mask, graph=graph)

    def project_graph(self, x: Tensor) -> QKV:
        return self.graph_attention.project(self.graph_attention_norm(x))

    def graph_feedforward(self, x: Tensor) -> Tensor:
        return self.graph_mlp(self.graph_mlp_norm(x))

    def forward(self, x: Tensor, context_size: int, edges: Tensor, graph_mask: GraphMask | None,
                graph: DGLGraph | None = None) -> Tensor:
        """One shared block sequence for training and evaluation: [N, T, D] -> [N, T, D]."""
        node_batch = 0 if self.training else EVAL_NODE_BATCH_SIZE
        token_batch = 0 if self.training else EVAL_TOKEN_BATCH_SIZE
        compiled = self.training and torch.is_grad_enabled() and self._compiled_feedforward is not None
        feedforward = self._compiled_feedforward if compiled else self.feedforward
        graph_feedforward = self._compiled_graph_feedforward if compiled else self.graph_feedforward
        # Features interact within a node; context/graph attention interacts
        # across nodes independently for each feature token. Keep every neighbor.
        x = x + apply_in_batches(self.feature_attention, x, dim=0, batch_size=node_batch)
        x = x + apply_in_batches(self.context_attention, x, dim=1, batch_size=token_batch, context_size=context_size)
        x = x + apply_in_batches(feedforward, x, dim=0, batch_size=node_batch)
        x = x + apply_in_batches(self.graph_update, x, dim=1, batch_size=token_batch,
                                 edges=edges, graph_mask=graph_mask, graph=graph)
        return x + apply_in_batches(graph_feedforward, x, dim=0, batch_size=node_batch)


class NanoGraphPFNModel(nn.Module):
    def __init__(self, num_layers: int = 6, num_heads: int = 4, embed_dim: int = 256,
                 hidden_dim: int = 768, max_classes: int = 32) -> None:
        super().__init__()
        if min(num_layers, num_heads, embed_dim, hidden_dim, max_classes) < 1 or embed_dim % num_heads:
            raise ValueError("Positive architecture dimensions and embedding size divisible by num_heads required")
        self.num_layers, self.num_heads = num_layers, num_heads
        self.embed_dim, self.hidden_dim, self.max_classes = embed_dim, hidden_dim, max_classes
        self.feature_encoder = nn.Linear(1, embed_dim)
        self.classification_encoder = TargetEncoder(embed_dim)
        self.regression_encoder = TargetEncoder(embed_dim)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(num_heads, embed_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.classification_decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, max_classes),
        )
        self.regression_decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1),
        )

    def compile_training_blocks(self, *, mlps: bool = True, projections: bool = True) -> None:
        """Compile tensor-only MLP and projection regions; SDPA and evaluation stay eager.

        Dynamic node/token dimensions avoid one graph per prior sample.
        Projection regions include QKV unpacking so backward packing can fuse.
        """
        for block in self.layers:
            if mlps:
                block._compiled_feedforward = torch.compile(block.feedforward, dynamic=True, fullgraph=True)
                block._compiled_graph_feedforward = torch.compile(block.graph_feedforward, dynamic=True, fullgraph=True)
            if projections:
                for name in ('project_features', 'project_context', 'project_graph'):
                    setattr(block, '_compiled_' + name, torch.compile(getattr(block, name), dynamic=True, fullgraph=True))

    @staticmethod
    def validate_tensor_values(x: Tensor, context_y: Tensor, edges: Tensor) -> None:
        """Value checks can run on prior CPU tensors before their device transfer."""
        if not torch.isfinite(x).all() or not torch.isfinite(context_y).all():
            raise ValueError("Features and observed labels must be finite")
        if edges.numel() and (edges.min() < 0 or edges.max() >= len(x)):
            raise ValueError("Edge endpoint outside graph")

    @property
    def arch(self) -> dict[str, int]:
        return {key: getattr(self, key) for key in ARCHITECTURE_FIELDS}

    def prepare(self, x: Tensor, context_y: Tensor, edges: Tensor, task_type: TaskType,
                feature_cardinalities: Tensor | None, features_normalized: bool,
                inputs_validated: bool = False) -> tuple[Tensor, Tensor]:
        """Validate [N, F] inputs and [C] labels; return features and label tokens [N, D].

        The prior supplies context-standardized features. Evaluation supplies
        preprocessed features that still need context-only standardization here.
        """
        if x.ndim != 2 or context_y.ndim != 1 or not 0 < len(context_y) < len(x):
            raise ValueError("Expected features [N,F] and ONLY context labels [sep], with 0 < sep < N")
        if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype != torch.long:
            raise ValueError("Expected int64 edge_index [2,E]")
        if task_type not in {"binclass", "multiclass", "regression"}:
            raise ValueError(f"Unknown task type: {task_type}")
        if not inputs_validated:
            self.validate_tensor_values(x, context_y, edges)
        if feature_cardinalities is None:
            feature_cardinalities = torch.zeros(x.shape[1], device=x.device, dtype=torch.long)
        if feature_cardinalities.shape != (x.shape[1],):
            raise ValueError("Feature cardinalities must have shape [F]")
        if not features_normalized:
            numerical = feature_cardinalities == 0
            # Nanoprior already performs this transformation for synthetic tasks.
            context = x[:len(context_y)].float()
            mean, scale = context.mean(0), context.std(0, correction=0).clamp_min(1e-12)
            x = torch.where(numerical[None], ((x - mean) / scale).clamp(-100, 100), x)
        encoder = self.regression_encoder if task_type == "regression" else self.classification_encoder
        return x, encoder(context_y, len(x))

    def decode(self, embeddings: Tensor, task_type: TaskType, num_classes: int | None) -> Tensor:
        """Query label tokens [N-C, D] -> regression [N-C] or logits [N-C, classes]."""
        if task_type == "regression":
            return self.regression_decoder(embeddings).squeeze(-1)
        count = self.max_classes if num_classes is None else num_classes
        if not 1 <= count <= self.max_classes:
            raise ValueError(f"Class count {count} exceeds model capacity {self.max_classes}")
        return self.classification_decoder(embeddings)[..., :count]

    def forward(self, x: Tensor, context_y: Tensor, edge_index: Tensor, task_type: TaskType, *,
                num_classes: int | None = None, feature_cardinalities: Tensor | None = None,
                features_normalized: bool = False, inputs_validated: bool = False) -> Tensor:
        """Features [N, F], observed labels [C], edges [2, E] -> query predictions.

        Context nodes must come first. Classification returns [N-C, classes]
        logits; regression returns [N-C] standardized targets. No query labels
        enter the model. Internal embeddings are [N, F+1, D].
        """
        x, y_embedding = self.prepare(
            x, context_y, edge_index, task_type, feature_cardinalities, features_normalized, inputs_validated,
        )
        num_nodes, context_size = len(x), len(context_y)
        embedding = torch.cat((self.feature_encoder(x[..., None]), y_embedding[:, None]), dim=1)
        graph_mask = None
        graph = None
        if num_nodes <= MAX_SDPA_GRAPH_SIZE:
            graph_mask = make_neighbor_mask(edge_index, num_nodes)
        else:
            graph = make_dgl_graph(edge_index, num_nodes)
        for block in self.layers:
            embedding = block(embedding, context_size, edge_index, graph_mask, graph)
        return self.decode(embedding[context_size:, -1], task_type, num_classes)
