# modded-nanographpfn

Small graph PFN, based on **record 3** of
[borawhocodess/modded-nanotabpfn](https://github.com/borawhocodess/modded-nanotabpfn).
Training consumes fresh synthetic graphs from `nanoprior.py`, with binary,
multiclass, and regression tasks. The nano backbone has GraphPFN-style
neighborhood attention and residual MLP adapters after every transformer block.
Separate target encoders and classification/regression heads share the backbone;
the scalar regression decoder follows LimiX-16M. All parameters train from
scratch. No pretrained model or tabular prior download is needed.

## Agentic research

Start with [RESEARCH_INSTRUCTIONS.md](RESEARCH_INSTRUCTIONS.md) for the lead's
role and workflow. The shared [protocol](workdir/research/PROTOCOL.md),
[report](workdir/research/REPORT.md), and [queue](workdir/research/QUEUE.md)
define fair comparisons, the team's single-GPU policy, and experiment tracking.
Validation-based evaluation is pending the user's evaluator changes; these
instructions do not start experiments or change the evaluation code.

## Training scheme

The starting point was Carter Prince's SDPA/bfloat16 record, dated February 4, 2026.
Upstream reports 10.10 minutes of training on one NVIDIA L40S; this result
is historical tabular performance, not a result for this graph model.

The training scheme retains six pre-norm transformer layers, four attention heads,
256-dimensional embeddings, a 768-dimensional MLP, learning rate 0.001,
batch size 1, 64 steps per epoch, and seed 11. It combines Muon for transformer
matrices with schedule-free AdamW for the remaining parameters. Attention
uses explicit QKV projections, PyTorch SDPA, and learned output projections
for feature, context, and graph attention, with bfloat16 autocast for training
and inference. The graph model has 9,119,009 parameters with the default
32-class head. Graph adapters use standard `nn.Linear` random initialization,
so graph structure affects predictions from the first forward pass.
Regression uses MSE on context-standardized
targets; classification uses cross-entropy over the task's actual classes.

Optimizer ownership follows the model structure: matrices inside `model.layers`
use Muon; encoders, decoders, biases, and normalization parameters use
schedule-free AdamW. `PackedQKVLinear` explicitly identifies projections whose
Q, K, and V matrices are orthogonalized separately. MLP matrices are processed
as whole matrices, including weights shaped `[3D, D]`. This corrects the earlier
shape heuristic that incorrectly treated those MLP weights as QKV. The two
optimizers are named `optimizers.muon` and `optimizers.adamw`.

On CUDA, Muon batches same-shape matrices with an independent norm for every
matrix (and every Q/K/V slice), and replays its step through a CUDA graph.
Transformer MLPs and attention norm/projection/QKV-unpacking regions use
`torch.compile` with dynamic node/token dimensions. SDPA and evaluation stay
eager. Disable individual optimizations with `--no-cuda-graph-muon`,
`--no-batched-muon`, `--no-compile-blocks`, or `--no-compile-projections`.
Graph warmup restores weights and momentum before the
first real update; captures are rebuilt after optimizer-state or hyperparameter
changes. Checkpoint parameter names and optimizer state formats are unchanged.
Prior input validation runs on CPU before transfer; loss finiteness is still
checked before every backward pass.
The prior loader prefetches tasks into pinned memory, and GPU copies are
nonblocking; labels are copied once and sliced into context and query views.
Only the context slice enters the model. `--no-async-transfer` disables pinning
and nonblocking copies. Schedule-free AdamW explicitly uses its built-in foreach
path, which was already the installed package's default; its equations are unchanged.

Profile training without evaluation or checkpoints using:

```bash
.venv/bin/python speedrun_utils/profile_pretrain.py --warmup 20 --steps 200 --trace
```

This writes throughput, shapes, peak allocated memory, operator timings, and a
Chrome trace under `workdir/profile-pretrain/`. Throughput excludes compilation
warmup and trace collection. Use `--no-compile-blocks --no-compile-projections
--no-cuda-graph-muon --no-batched-muon --no-validate-on-cpu --no-async-transfer
--no-pin-memory` for a comparison run with the same task sequence. Graphs
up to 2,048 nodes can be profiled without DGL. `--log-recompiles` reports compiler
guard failures; the default dynamic compilation does not require data padding.

Training uses one continuous prior iterator across epochs. Its next consumed
task index is saved with optimizer states and random states for resuming.
The defaults are 4,000 epochs, 64 steps per epoch, and an 80-minute training-time
cap. Both evaluation suites run after every completed epoch by default.
Set `--eval-every 100` to evaluate at epochs 100, 200, and so on. Training always
evaluates after its final epoch, including an epoch that reaches the time cap.
Preflight remains a separate initial evaluation; `--skip-preflight` skips it.
Checkpoints and training metrics are saved every epoch; skipped evaluations
appear as `null` in `metrics.jsonl`. Best scores and baseline crossings update
only on evaluation epochs. On resume, the interval uses the total epoch number.
Baseline attainment never stops training.
Set `--max-train-mins 0` to run through the configured epoch count without a
time cap. Primary timing includes training and waiting for prior tasks, and
excludes evaluation and preflight.

## Setup and launch

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
`uv sync` from this directory. The project environment includes dependencies
for training, prior sampling, GraphLand subsampling, random forest baselines,
and the results notebook. The baseline's scikit-learn version is pinned to
1.7.2 in both the project manifest and the standalone requirements file.

Inspect the dataset selections and RF-50+NFA test-score thresholds:

```bash
uv sync
./run.sh --list-eval
```

The project uses Python 3.12.9, PyTorch 2.4.0, and DGL 2.4.0 from the
[DGL CUDA 12.4 wheel index](https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html),
matching DGL's PyTorch ABI. These DGL wheels target Linux x86-64. `uv sync`
recreates an existing Python 3.13 environment with the pinned Python version.
Production training targets one NVIDIA A6000 with CUDA/bfloat16 support;
prior sampling, preprocessing, subsampling, and
random forests run on CPU. `--device cpu --prior-workers 0` is available for
small integration checks. Four CPU prior workers are used by default.

Before training, run the actual inference memory preflight on the GPU host:

```bash
./run.sh --name a6000-preflight --preflight-only
./run.sh --name graph-run
```

For evaluation every 100 epochs:

```bash
./run.sh --name graph-run --eval-every 100
```

Normal training also runs preflight by default. `--preflight-only` evaluates
every selected graph with the configured model and records per-dataset peak
GPU allocated/reserved memory in `preflight.json`, without optimizer steps.
These initial scores do not count as trained-model best scores or threshold
crossings. After a successful separate check, `--skip-preflight` avoids repeating
it. An OOM fails explicitly with the dataset name and original allocation error;
it never silently drops a dataset. If needed, lower the evaluation batch sizes
described below and rerun preflight. To explicitly evaluate a smaller selection:

```bash
./run.sh --preflight-only --full-datasets artnet-views city-reviews tolokers-2 twitch-views
```

Evaluation uses the same full-graph forward pass as training, with gradients
disabled. All graph nodes and edges remain available, including unlabeled nodes,
and all context labels are retained. Each layer has one forward implementation.
During evaluation, `apply_in_batches()` limits temporary allocations by slicing
the inputs before normalization and projection, then writing results into a
preallocated output:

- Feature attention and both MLPs process at most **2,048 nodes** per batch.
  Feature attention still sees every feature token for each node.
- Context and graph attention process at most **four feature tokens** per batch.
  Every batch retains all nodes, context labels, and graph neighbors.

These limits are `EVAL_NODE_BATCH_SIZE` and `EVAL_TOKEN_BATCH_SIZE` near the top
of `nanographpfn.py`; set either to `0` to disable batching along that axis.
Smaller batches reduce temporary memory at the cost of more kernel launches.
Batching requires both `model.eval()` and disabled gradients (`inference_mode()`
or `no_grad()`). Training and gradient-enabled calls use full tensors. Context
attention's temporary QKV tensors are released before subsequent operations
during inference. This batching changes execution only; it retains the same
model operations and full graph/context inputs.

The full graph embeddings still reside on the GPU, so these limits do not cap
total memory use. Run preflight with the intended model and dataset selection.
The separate `predict_chunked()` path and its CLI flags remain removed.
For calls with node batching disabled, feature attention retains the PyTorch 2.4
CUDA launch workaround: above 65,535 nodes, SDPA processes at most 32,768 nodes
per call, with every feature token retained.

Only neighborhood attention switches to DGL for graphs with **more than 2,048
nodes**, in both training and evaluation. Its fused
dot-product, incoming-edge softmax, and weighted-sum operations follow
[GraphPFN's graph adapter](https://github.com/yandex-research/graphpfn/blob/main/paper/lib/graphpfn/model.py#L325).
The topology is built once per model call and reused across layers;
all nodes, edges, and existing self loops are retained, and no self loops
are added. Aggregation stays in float32 under autocast. DGL stores scalar scores
per edge/token/head instead of per-edge value vectors.

Graphs with at most 2,048 nodes use dense masked SDPA in both training and evaluation.
DGL is imported lazily for large graphs; an unavailable or incompatible DGL
installation produces an explicit error. Feature/context attention, MLPs,
encoders, heads, the prior, data loading, preprocessing, and baselines retain
their current implementations.

Run settings live in `Config` and are exposed as CLI flags (`--help`). Outputs
go to a unique directory under `workdir/experiments/`: `log.txt`, source snapshots,
`metrics.jsonl`, `last.pt`, `best_full.pt`, `best_tiny.pt`, and `final.pt`.
`report.json` is written atomically when the run ends, after the final evaluation
for successful runs. Failures/interruption produce an explicitly marked partial
report; they do not masquerade as completed runs.

Architecture options use descriptive names: `--num-layers`, `--num-heads`,
`--embed-dim`, `--hidden-dim`, and `--max-classes`. The model constructor uses the
same names with underscores. The old single-letter flags are not supported.
Checkpoints and run reports use format/schema version 2. Earlier runs are not
supported; there are no key migrations or compatibility aliases.
Checkpoints from before feature/context attention output projections were added
lack those parameters and cannot be loaded directly into the current model.

Resume with `--resume /path/to/last.pt`, keeping the original architecture,
prior configuration, seed, steps, learning rate, baseline file, and dataset
selection. The resumed invocation gets a new output directory. Epoch counts
are total counts, not additional epochs. Increase or disable the training-time
cap if the checkpoint already reached it. Prior worker count can change without
changing the task sequence. Checkpoints contain schedule-free evaluation
weights; the saved optimizer state restores training weights on the next epoch.
Resume applies to checkpoints produced by this codebase version.

## Full and tiny evaluation

There are exactly two real-data suites, both using GraphLand RL **test** masks:

- **Full:** all 10 available GraphLand datasets by default. `--full-datasets`
  can explicitly restrict this list.
- **Tiny:** only `artnet-views`, `city-reviews`, `tolokers-2`, and `twitch-views`,
  using every existing `-t-0` through `-t-4` variant (20 graphs). `--tiny-datasets`
  can restrict these four families; all five variants are required per family.

Binary classification uses AP, multiclass uses accuracy, and regression uses
R². Suite scores are equal-weight means of the raw dataset metrics, as in the
baseline notebook. The best suite score is the best mean at a single epoch,
not the mean of each dataset's independently best epoch. Test scores explicitly
drive checkpoint selection and threshold reporting for this benchmark protocol.

Only labeled RL training nodes provide context labels. Feature transforms fit
all nodes without labels, matching the transductive setting. Numerical/fraction
transforms and missing-value handling follow the baseline; categorical features
use ordinal codes to match nanoprior, rather than the forest's one-hot encoding.
Context-constant feature columns are dropped, and numeric features are
standardized using context rows. For `avazu-ctr`, after the numeric transforms
and constant-column removal, PCA reduces the features to **64 components** if
more than 64 columns remain. This adopts the dimensionality used for
high-dimensional datasets in [GraphPFN, Appendix D](https://arxiv.org/html/2509.21489v4#A4)
and applies it specifically to Avazu here. PCA fits all nodes without labels,
uses deterministic randomized SVD (`random_state=0`, no whitening), and returns
numerical features for context standardization. Small Avazu variants use at
most `num_nodes - 1` components. Each graph's evaluation record reports the
input/output feature counts and PCA settings, including retained variance.
The prepared features are cached across evaluation epochs. Other datasets and
the RF baselines keep their existing preprocessing.

Regression target scaling also uses context only; predictions are converted
back to dataset units before R² scoring.
The model consumes graph edges directly; NFA features are used by the RF
comparator, not appended to the model inputs.

Thresholds are loaded from `baselines/results/graphland_rf.json` (override with
`--baseline-report`), selecting `rf50_nfa`, seed 42, RL, and test scores for the
exact selected datasets. The default thresholds in the current report are
**0.5111094821985549** for full and **0.2773684572726378** for tiny.
Missing, duplicate, unsuccessful, or incompatible baseline records are errors.
This report is an existing local artifact; a fresh checkout needs to generate
it with the baseline runner described below.

The final report exposes these four requested summary fields:

```json
{
  "best_full_score": null,
  "best_tiny_score": null,
  "time_to_baseline_full_seconds": null,
  "time_to_baseline_tiny_seconds": null
}
```

The example values are placeholders. Completed runs contain measured best
scores; a time remains `null` if that suite never reaches its RF-50+NFA threshold.
The full and tiny entries under `scores` have identical statistics: best score,
best epoch, dataset scores at that epoch, baseline threshold, first reaching
epoch, training seconds to threshold, and wall seconds to threshold. Equality
counts as reaching the baseline. Crossing times are first observed at an epoch's
evaluation; they are not interpolated. Wall time includes prior startup,
evaluation, and checkpointing since training started, excluding preflight and
downtime between resumed invocations.

## Verification

```bash
uv run python -m unittest discover -s tests -v
uv run python -m unittest discover -s baselines/tests -v
uv run python -m unittest discover -s speedrun_utils -p 'test_*.py' -v
```

Graph integration tests cover attention values/gradients, full-graph evaluation
with gradients disabled, nonzero graph adapters, node permutations, query-label
isolation, mixed-task training, independent best scores/first crossings,
uninterrupted-versus-resumed training, final reporting at the time cap, and
preflight OOM behavior. Small CPU
training runs and real-dataset preprocessing are exercised; A6000 peak memory
and training throughput require running preflight/training on the GPU host.
`tests/test_eval_batching.py` compares batched and unbatched predictions across
all tasks, dense/DGL attention, and autocast; verifies unchanged training
gradients and bounded projection shapes for Avazu; and includes a CUDA peak-memory
regression check. CUDA measurements are skipped when no GPU is available.
`tests/test_optimizers.py` checks complete, disjoint optimizer ownership and
different updates for same-shaped MLP and packed-QKV matrices. Test sources in
`tests/` and `baselines/tests/` are tracked; generated caches remain ignored.
`tests/test_dgl_attention.py` also checks the size cutoff, lazy import, topology
reuse, and DGL parity with dense attention and a small per-node reference,
including gradients, directed edges, isolated nodes, empty graphs, duplicate
edges, and autocast.
Real DGL kernel tests are skipped when DGL is absent; CUDA checks require a GPU.

## Code layout

- `train_nano.py`: configuration, explicit optimizer ownership, and training.
  `main()` calls runtime setup, training initialization, preflight, the training
  loop, and final reporting. `TrainingState` holds counters and resume state.
  Safe to import; execution is guarded by `main`.
- `nanographpfn.py`: nano transformer, graph adapters, regression/classification
  encoders and decoders, and a shared full-graph forward pass for training and inference.
  Public methods document tensor shapes and the context/query boundary; block
  modules have names such as `feature_norm`, `context_qkv`, and `mlp_in`.
- `graph_eval.py`: separate validation, ordinal encoding, feature/PCA processing,
  target preparation, and graph reordering; full/tiny suites, baseline matching,
  test scoring, and best-score/time-to-baseline statistics. `PreparedGraph`
  documents the evaluator/model tensor contract.
- `nanoprior.py`: standalone degree-corrected SBM and neural graph prior,
  CPU worker loader, and sampling diagnostics.
- `run.sh`: launcher for nanoGraphPFN.
- `pyproject.toml`: shared dependencies for all project workflows.
- `speedrun_utils/data_subsampling.py`: connected GraphLand subsampling for
  small speedrun evaluation datasets.
- `baselines/`: fixed RF-50, RF-500, RF-50+NFA, and RF-500+NFA runners and tests.
- `LICENSE`: Apache License 2.0, retained for the upstream code.

## Standalone graph prior

`nanoprior.py` samples synthetic node classification and regression tasks.
It needs only NumPy and PyTorch, and does not import the training script,
download data, or use a GPU. All generation code is in this one file.

```bash
uv run python nanoprior.py --samples 10 --seed 42 --workers 2
uv run python nanoprior.py --samples 3 --task regression
uv run python -m unittest discover -s tests -v
```

The command prints one JSON record per task with realized sizes, degree
statistics, task type, class counts, edge label agreement for classification,
SCM parameters, and rejection counts. Its final record gives the total time,
including worker startup. `--nodes` changes the number of nodes before
component extraction. It does not save datasets or train a model.

The generation pipeline follows the graph-first approach in
[GraphPFN](https://github.com/yandex-research/graphpfn/tree/main/paper/lib/graphpfn/prior):

1. Sample one degree-corrected stochastic block model (SBM). Unequal communities
   have random connection affinities; node-specific endpoint probabilities
   follow either a Zipf or a uniform distribution, normalized within each group.
   Poisson edge counts are sampled for group pairs. Within-group endpoints are
   sampled without replacement within each edge. Duplicate edges are removed.
2. Extract the largest connected component and renumber/shuffle its nodes.
   The requested size is **1,000 nodes**, matching the configured speedrun
   prior's row count. Returned tasks can have fewer nodes. No edges are added
   to force connectivity, and there is no preferential-attachment stage.
3. Generate attributes with a random neural structural causal model (SCM).
   Each channel uses either its local linear value or a graph aggregation.
   Observations are sampled from hidden states, or the random inputs become
   features and the network's final scalar output becomes the target.
4. Convert the target to the requested task, then choose context nodes and
   reorder nodes consistently across features, targets, and edges. Convert some
   feature columns to categorical codes; drop context-constant columns; scale
   numerical columns using context statistics and clip to [-100, 100].
5. Validate and return the task. Expected invalid samples are retried up to a
   fixed limit. Exhaustion raises `SamplingError` with the seed and reasons;
   unexpected programming errors propagate immediately.

The main defaults are deliberately visible in `PriorConfig`:

| Quantity | Distribution or policy |
| --- | --- |
| Nodes before extraction | Fixed at 1,000 |
| Requested features | Uniform integer, 4–100; constants may be dropped, retaining at least 4 |
| Communities | Uniform integer, 2–8; at least 4 nodes per community |
| Proposed average degree | Log-uniform, 1.5–100; upper bound capped at N−1 |
| Between-community affinity scale | 0.1 |
| Task type | Equal probabilities for binary, multiclass, regression |
| Multiclass count | Uniform integer, 3–32, capped by the realized context size |
| Context fraction | Uniform, 0.05–0.5, calculated after component extraction |
| Categorical conversion | Enabled for 20% of tasks; a random fraction of columns converted |
| Categorical cardinality | Gamma-derived, capped at 256 |
| Graph channel probability | Uniform choice of 0.0, 0.1, …, 1.0 |
| Aggregation | Uniform choice of mean, min, max, GCN, neighborhood attention |
| Degree and PageRank SCM inputs | Independently enabled with probability 0.5 |

The SCM settings are exposed as `PriorConfig.scm_*` fields: latent cause/depth/width
mean bounds, weight and noise mean bounds, graph mixing range, dropout scale/beta
bounds, and activation scale. Positive-normal mean bounds control a sampled
log-uniform mean; they are not hard limits on the final width, depth, or standard
deviation. Defaults retain the previous sampling distributions and draw order.
The prior configuration contract is version 2. For example, set
`scm_graph_ratio=(0.0, 0.0)` to disable message-passing channels.
With zero graph channels and no structural inputs, attributes are independent
of topology. The graph channel probability does not directly prescribe label
homophily. Graph simplification and component extraction change realized
degree, which is reported separately from the proposed degree.

This is a **GraphPFN-style baseline, not exact upstream reproduction**. It
uses one SBM, a smaller activation set, a single log-uniform degree range,
and pure PyTorch graph operations. Mixed latent input distributions have equal
weights. Neighborhood attention uses random projections scaled by input width.
Classification uses distinct rank boundaries and reserves one random node of
each class for context; the remaining context nodes are chosen randomly. This
avoids repeated rejection of rare classes in small components. Query nodes
need not contain every class. Task type is held fixed across retries; other
parameters are redrawn, so validity rejection still conditions their realized
distributions. No ExtraTrees predictability filter is applied.

Numerical feature and regression-target scaling fit context rows only.
Categorical columns retain integer codes in a float tensor, with cardinalities
returned separately. Categorical boundaries and synthetic class boundaries
are part of dataset generation and may use all generated nodes. No missing
features or extra unlabeled nodes are simulated.

Direct sampling:

```python
from nanoprior import PriorConfig, sample_task

task = sample_task(PriorConfig(task_type="multiclass"), seed=42)
x = task.x                       # float32 [N, F], CPU
edges = task.edge_index           # int64 [2, E], source -> destination
context_y = task.context_y        # y[:sep]; labels visible to the model
query_y = task.query_y            # y[sep:]; labels for the loss only
```

Every undirected edge appears in both directions, without duplicates or
self-loops. Attention adds self contributions internally. Classification
targets are int64; regression targets are float32, standardized on context.
`feature_cardinalities` is zero for numerical columns and the number of codes
for categorical columns. `metadata` records the seed, prior version, config,
sampled graph/SCM parameters, and rejection history. For regression,
`target_mean` and `target_scale` convert returned targets back to the original
SCM target units: `raw_y = task.y * target_scale + target_mean`.

CPU workers yield one graph at a time, without padding or stacking unrelated
tasks. Use the usual main guard for multiprocessing:

```python
import torch
from nanoprior import PriorConfig, make_dataloader

def main():
    torch.set_num_threads(1)
    loader = make_dataloader(
        PriorConfig(), num_tasks=100, seed=42, num_workers=4,
    )
    for task in loader:
        # Consume task.x, task.edge_index, task.context_y, and task.query_y.
        next_task_index = task.metadata["task_index"] + 1

if __name__ == "__main__":
    main()
```

Task seeds depend on `(seed, task_index)`, not worker scheduling. Generation
does not consume Python, NumPy, or PyTorch global random state. Workers use one
PyTorch thread each and a bounded prefetch queue. For exact comparisons, use
the same dependency versions, CPU/thread configuration, and prior version.
To resume, pass the next consumed index as `start_index` to a new loader with
the same seed/config. Reiterating an existing loader replays its original
index range; prefetched but unconsumed tasks do not advance a saved index.

Verification on October 14, 2026 used Python 3.13.16, PyTorch 2.14.1, and
NumPy 2.5.4. All 23 prior tests passed, including CPU-worker transfer,
reproducibility across worker counts, and resuming from a saved task index.
The suite ran outside the sandbox to allow PyTorch's local multiprocessing
sockets. The 15 baseline tests and 5 subsampling tests also passed.
A separate audit of seeds 0–199 for each task type generated 600 valid tasks
with the default 1,000 requested nodes. Retained components had 372–1,000
nodes; 584 tasks succeeded on the first attempt and 16 on the second.
The audit checked connectivity, edge symmetry and uniqueness, finite tensors,
categorical codes, context class coverage, and context regression scaling.
These checks establish sampling behavior, not downstream model accuracy.

## GraphLand subsampling

Run `python3 speedrun_utils/data_subsampling.py` to create five samples of each
available GraphLand dataset under `data/<dataset>-t-0` through `-t-4`. The
standalone script needs PyYAML; it does not require PyTorch or a GPU.

With seed 42, each sample draws a size from 500–1,000 nodes, grows a weakly
connected neighborhood, and retains all original edges between those nodes.
At least 50% of each sample's nodes must have labels. Neighborhoods below
this threshold are retried at the same size, with an error after 100 failed
attempts. Override the threshold with `--min-labeled-fraction` (default `0.5`).
Samples can overlap. Up to 100 original feature columns are selected randomly
per sample. The original CSV/YAML layout is preserved and feature metadata is
updated. `subsampling.json` records seeds, source node IDs, features, and counts.

Only RL masks are produced: labeled nodes are reassigned using the source
train/validation/test proportions (approximately 10%/10%/80%), with integer
rounding and stratification for classification. Unlabeled nodes count toward
the sample size and retain GraphLand's training-mask convention. Each sampled
class is represented in training when the training quota permits it.

Existing destinations are refused. Use `--output-dir /tmp/graphland-small` for
a separate run, `--datasets artnet-exp city-roads-M` for a subset, or `--help`
for size, feature, sample-count, and seed options. Run the utility tests with
`python3 -m unittest discover -s speedrun_utils -p 'test_*.py'`.

## Random forest baselines

The CPU runner evaluates `rf50`, `rf500`, `rf50_nfa`, and `rf500_nfa` on all
available full GraphLand datasets and their `-t-N` subsamples. The current data
directory contains 10 full and 50 tiny datasets: 240 fits with one seed (42).
Each forest uses scikit-learn defaults except `n_estimators`, `random_state`,
and `n_jobs`; there is no hyperparameter search. Every model fits only labeled
RL training nodes, and reports separate validation and test scores.

Set up a standalone environment (Python 3.10 or newer; no GPU or PFN training
dependencies required), then run all four baselines:

```bash
python3 -m venv .venv-baselines
.venv-baselines/bin/python -m pip install -r baselines/requirements.txt
.venv-baselines/bin/python -m baselines.run --n-jobs 8 --output baselines/results/graphland_rf.json
```

Inspect the run plan without fitting models or installing scientific packages:

```bash
python3 -m baselines.run --list
```

`--datasets city-roads-M` selects that full graph and its tiny versions;
`--datasets city-roads-M-t-2` selects just that subsample. `--variants full`
or `--variants tiny` restricts graph sizes. `--models rf50 rf50_nfa` restricts
methods. Models and datasets run sequentially; `--n-jobs` controls RF workers.

Preprocessing follows the transductive defaults of the official
[GraphLand loader](https://github.com/yandex-research/graphland/blob/main/dataset.py):
one-hot categorical encoding with `drop="if_binary"`, normal-quantile numerical
transforms, and mode imputation after transformation. Fraction columns are
quantile-transformed only for `avazu-ctr` and `artnet-views`; tiny variants
inherit their parent dataset's policy. These feature transforms fit on all
nodes, without using labels. Regression target standardization fits only
labeled training nodes; predictions are converted back before scoring. Values
already logged in the source datasets are not logged again.

NFA follows [Appendix B.2 of the GraphLand paper](https://papers.nips.cc/paper_files/paper/2025/file/8f54de761abc405145d8b4df8bef3f18-Paper-Datasets_and_Benchmarks_Track.pdf):
append one-hop numerical neighbor means/minima/maxima and one-hot categorical
frequencies to original features. Aggregation uses raw values before
normalization or imputation, ignores missing numerical neighbor values, and
uses a simple undirected graph without self-neighbors. It includes all nodes'
features, including validation, test, and unlabeled nodes, but no labels.
Each tiny graph uses only its own CSV files. NFA is computed once per dataset
and reused between the two forest sizes.

For the derived columns, numerical statistics inherit the source feature's
numerical/fraction type, and category frequencies are fraction features. Empty
neighborhood aggregates remain missing until imputation. All-missing columns
are retained and filled with zero, to handle tiny-graph edge cases consistently.
The 100-feature sampling limit applies to stored raw columns; one-hot encoding
and NFA may increase model input width beyond 100.

Scores use `info.yaml`: average precision (`AP`) for binary classification,
accuracy for multiclass classification, and `R2` for regression. JSON stores
raw scores, without multiplying by 100; R² can be negative. Empty evaluation
sets and R² sets with fewer than two labeled nodes get `null` plus an explanation.

The JSON includes individual dataset/model scores, labeled split sizes,
feature counts, all RF parameters, dependency versions, preprocessing choices,
and timings. Load/feature times are shared between models and must not be
summed as independent fit costs. The file is replaced atomically after each
fit; failures are recorded and other jobs continue. A completed run with
errors exits nonzero. Interruptions retain completed results. Existing output
files are refused; omit `--output` to use a unique timestamped filename under
`baselines/results/`. No fitted models are saved.

Plot the average test score of each method across full and tiny datasets in
[`baselines/plot_results.ipynb`](baselines/plot_results.ipynb):

```bash
uv run jupyter lab baselines/plot_results.ipynb
```

For a separate plotting environment, install `baselines/requirements-notebook.txt`
instead of the full project dependencies.

The notebook reads JSON reports from `baselines/results/`, shows dataset counts,
and weights each dataset equally. Set `SCORE_SPLIT = "val"` for validation scores.
The mean combines raw AP, accuracy, and R² values; it is not an accuracy percentage.
The per-dataset section shows each method's full score beside its mean across
the corresponding tiny versions, with one panel per base dataset and a score table.

Run tests (small synthetic graphs and forests, not the benchmark):

```bash
.venv-baselines/bin/python -m unittest discover -s baselines/tests -v
```

Numerical/RF tests explicitly skip when their dependencies are unavailable;
discovery and JSON checkpoint tests also run with the standard-library Python.

## Source and verification

The run 3 source was recovered from the code embedded in the
[published run 3 log](https://github.com/borawhocodess/modded-nanotabpfn/blob/main/records/20260204carter/260205-225951-d60000eb-carter-log.txt).
See also the [run 3 pull request](https://github.com/borawhocodess/modded-nanotabpfn/pull/4).

Direct Git downloads were unavailable in the preparation environment.
The web-rendered log removed indentation, so formatting was restored using
the corresponding blocks from upstream's
[`b0f29b7406e048125bbe939a6365eeb70acabc90` source](https://raw.githubusercontent.com/borawhocodess/modded-nanotabpfn/b0f29b7406e048125bbe939a6365eeb70acabc90/train_nano.py).
Before the graph integration, all 560 nonblank source lines matched the published
run 3 code after whitespace normalization, excluding the added provenance header.
The current trainer and model deliberately extend and restructure that initial
snapshot; they are no longer a verbatim reproduction of the tabular record.

The original training dependency bounds come from the
[upstream project manifest](https://raw.githubusercontent.com/borawhocodess/modded-nanotabpfn/main/pyproject.toml).
The shared manifest also includes dependencies for the graph tools and notebook,
and pins scikit-learn to the version used by the standalone baselines.
The historical run logged Python 3.13.9, PyTorch 2.9.1+cu128, and CUDA 12.8.
The original lockfile was not retrieved. The current `uv.lock` records the
project's newly resolved environment, not those exact historical dependency
versions.

Initial source-preparation checks covered Python syntax, correspondence with
the published source, and shell syntax. Subsequent graph integration checks
include CPU training and inference as described above. No long GPU pretraining
result is claimed.
