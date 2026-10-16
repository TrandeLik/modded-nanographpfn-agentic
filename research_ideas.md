# Research ideas

## 1. Architecture

### 1.1. TabICLv2-style backbone and graph adapters

- Explore a TabICLv2-style architecture: column-wise distribution embedding, within-node feature interaction, feature pooling, and node-level in-context learning (ICL).
- Compare attention orderings and the placement of feature pooling.
- Evaluate query-aware scalable softmax (QASSMax).
- Compare graph adapters placed in the feature block, before pooling, as learnable neighborhood feature aggregation (NFA); after pooling; inside the ICL block; and at combinations of these locations.
- Compare graph aggregation over feature-indexed representations with aggregation over pooled node representations.

Reference: [TabICLv2](https://arxiv.org/pdf/2602.11139).

### 1.2. Numerical feature embeddings

- Compare learned Fourier embeddings using sine and cosine, as in TabFM.
- Compare context-relative empirical CDF (ECDF) and midrank encodings, combined with standardized values and Fourier features, as in TabPFN-3.5.
- Compare LimiX-2M-style RaBEL radial-basis-function embeddings.
- Explore combinations of magnitude, rank, periodic, and localized numerical representations.

References: [TabFM](https://arxiv.org/pdf/2609.37959), [TabPFN-3.5](https://arxiv.org/pdf/2609.17895v1), [LimiX-2M](https://arxiv.org/abs/2606.04485).

### 1.3. Universal Classifier architecture

- Explore schema-invariant feature encoding and query–anchor similarity prediction.
- Support varying feature counts and class cardinalities without a fixed maximum-class prediction head.
- Investigate adaptation of the architecture to node classification and regression.

Reference: [The Universal Classifier for Graph Learning](https://arxiv.org/pdf/2609.36302).

### 1.4. Ephris-style architecture

- Explore induced attention for per-feature refinement.
- Combine induced attention with graph layers during feature refinement.
- Explore graph-aware ICL through sparse message passing and learned global nodes.
- Compare dense context attention, sparse graph ICL, and combinations of the two.
- Investigate how graph information is broadcast back into feature representations.

Reference: [Ephris: Message Passing Does More with Less for In-Context Learning on Graphs](https://arxiv.org/pdf/2609.37057).

### 1.5. Looped transformers

- Repeatedly apply shared transformer blocks to increase effective depth without increasing parameter count.
- Explore weight sharing in feature blocks, ICL blocks, graph adapters, and combinations of these modules.
- Vary the number of recurrent iterations during training and inference.
- Investigate whether additional inference iterations improve predictions and how graph oversmoothing depends on iteration count.

Reference: [LoopICL](https://arxiv.org/pdf/2609.36108).

### 1.6. Other architectural mechanisms
- Investigate architectural mechanisms from the additional linked papers.

References:

- [OpenReview: 63YdJdtZZ0](https://openreview.net/forum?id=63YdJdtZZ0).
- [Architecture Alignment With Sparse Priors in Tabular Foundation Models](https://arxiv.org/pdf/2609.36883).
- [OpenReview: WIVg3A3mBq](https://openreview.net/forum?id=WIVg3A3mBq).
- [Thinking in Depth: Retrospective Inference for Tabular Foundation Models](https://arxiv.org/pdf/2610.10317v1).

### 1.7. Unordered categorical representations

- Compare separate category embedding vectors, multiple independent hash embeddings (https://arxiv.org/abs/2608.27413), and category-specific rotation-based embeddings.
- Explore other representations that do not impose numerical order on categories.
- For rotation-based embeddings, assign category-specific angles or rotations without deriving a monotonic angle from the category ID.
- Support high cardinality, rare categories, and categories absent from the labeled context.
- Investigate invariance to consistent category renaming within a task.

### 1.8. Joint handling of fractional features

- Add a module that processes related fractional features jointly rather than treating each as an independent numerical column.
- Explore groups of proportions, ratios, or composition components connected through shared denominators, sum constraints, or other dependencies.
- Compare joint group representations with independent numerical embeddings.
- Pass fractional features both as numerical and fractional features.
- Note that there are fractional features in GraphLand

### 1.9. Graph-aware QASSMax

- Explore query-aware attention scaling conditioned on the size of the actual attention neighborhood.
- Compare ordinary softmax, global-length QASSMax, and Ephris-style neighborhood-aware scaling.
- Explore graph descriptors beyond neighborhood size as inputs to the scaling mechanism.
- Distinguish scaling for local graph attention, context attention, and attention involving global nodes.

### 1.10. Separate recurrence rates for graph and ICL computation

- Loop graph adapters and ICL blocks independently.
- Compare multiple graph propagation steps per ICL step with multiple ICL refinement steps per graph step.
- Investigate whether propagation distance and prediction-rule refinement require different computation budgets.

### 1.11. Feature grouping and representation-rank expansion

- Explore TabICLv2-style repeated feature grouping, in which overlapping groups of features are embedded jointly.
- Compare independent scalar embeddings with grouped embeddings and other approaches that increase the effective rank and expressiveness of representations.
- Vary group size, overlap, and grouping strategy, and investigate their interaction with numerical embeddings and graph adapters.
- Examine representation rank before and after feature interaction, pooling, and graph propagation.
- It is known that representations in TFMs are low-rank (for example, after embedding layer all representations share the same vector up to scaling) and it is significant bottleneck

Reference: [TabICLv2](https://arxiv.org/pdf/2602.11139).

### 1.12. Positional encodings

- Explore positional encodings inspired by TabICLv2 and alternative ways to distinguish feature tokens and positions within feature groups.
- Compare learned, fixed, and rotary encodings, including their placement in feature and ICL blocks.
- Investigate interactions between positional encodings, feature grouping, and representation rank.
- Explore generalization across feature counts and sensitivity to feature and node permutations.

Reference: [TabICLv2](https://arxiv.org/pdf/2602.11139).

### 1.13. Parameter-efficient ensembling

- Explore ensembles with a shared backbone and member-specific lightweight adapters, heads, modulation parameters or TabM like approaches.
- Compare sharing the feature encoder, graph modules, and ICL blocks at different levels.
- Explore independent member initialization and task sampling to encourage useful prediction diversity.
- Compare prediction averaging and learned combination of ensemble members.
- Investigate the tradeoff between ensemble diversity, predictive quality, parameter count, memory, and inference cost.

### 1.14. Early exit and adaptive inference depth

- Add prediction heads at intermediate blocks or recurrent iterations and allow inference to stop before the final depth.
- Explore stopping criteria based on confidence, prediction stability, or a learned halting mechanism.
- Compare task-level, graph-level, and node-level exit decisions.
- Combine early exit with intermediate supervision and looped architectures.
- Investigate the tradeoff between prediction quality and inference cost, including the effect of node-level exits on graph propagation.

### 1.15. Learning when to trust the graph

- Explore task-conditioned combinations of tabular predictions, local graph propagation, and global context attention.
- Generate matched task families with informative, irrelevant, and misleading observed edges, including controlled edge corruption or rewiring while retaining node features and targets.
- Explore controlling degree distributions and other graph statistics across these variants to distinguish graph reliability from structural differences.
- Investigate whether labeled context can guide graph-reliance decisions at the task, node, or feature level.
- Compare learned gating or prediction mixtures with fixed combinations and graph-branch dropout.
- Measure both gains from informative topology and degradation under irrelevant or corrupted topology.

Reference: [Ephris, Appendix G](https://arxiv.org/html/2609.37057v1).

### 1.16. PFN target encodings

- Explore how observed context targets are embedded and supplied to the PFN backbone, including separate classification and regression encoders versus shared representations.
- For classification, compare scalar class-ID projection with projected one-hot labels, learned class embeddings, and randomly assigned per-task class codes.
- Investigate equivariance to class-label permutations and generalization across class counts, keeping target representations aligned with the output decoder.
- For regression, compare linear scalar projection with MLP, Fourier, and radial-basis-function target embeddings, and combinations of context-normalized magnitudes and context-relative ranks.
- Compare learned query-target mask embeddings with the current context-mean placeholder plus missing-label indicator, preserving an explicit distinction between observed and unknown targets.
- Compare separate target tokens, additive target embeddings, and other ways of conditioning feature, ICL, and graph blocks on context targets.
- Use only observed context targets as encoder inputs and to compute target-normalization statistics; keep query targets hidden.

## 2. Prior design

### 2.1. TabICLv2 prior and its extension to graphs

- Explore the synthetic data-generation mechanisms of TabICLv2 and extend them to graph tasks.
- Model dependencies among latent variables, observed node features, topology, and targets.
- Explore graph-conditioned transformations within structural causal models (SCMs).
- Distinguish the SCM dependency graph from the observed graph connecting data instances.

Reference: [TabICLv2](https://arxiv.org/pdf/2602.11139).

### 2.2. Prior filtering

- Explore TabICLv2-style filtering based on predictive utility and causal dependence.
- Compare filtering using raw features, features plus simple graph descriptors such as degree, and NFA-augmented features.
- Compare these filters with no predictive filtering.
- Investigate acceptance criteria that retain tasks whose signal requires graph information.
- Explore filtering cost, acceptance rate, and the resulting distribution of task difficulty.

### 2.3. Target generation beyond scalar binning

- Compare scalar thresholding and binning with class assignments based on proximity to latent variables or latent prototypes.
- Explore density-based assignments and clustering over several latent neurons or features.
- Control class count, imbalance, noise, and separability.
- Generate nominal class structure without relying exclusively on ordered intervals of one scalar target.

### 2.4. Categorical-variable generation

- Generate realistic category frequencies, cardinalities, rare categories, and dependencies on other features and targets.
- Generate categories by clustering several latent features.
- Introduce discrete variables during generation so that categories can directly affect downstream features and targets.
- Explore unordered categorical mechanisms beyond scalar discretization followed by category-ID shuffling.

### 2.5. Fractional-feature generation

- Generate related groups of proportions, ratios, or composition components.
- Model shared denominators, sum constraints, and other dependencies where applicable.
- Vary group size, dependence strength, sparsity, and the frequency of boundary values.
- Align generated fractional-feature structure with the joint modules explored in the architecture.
- Note that there are fractional features in GraphLand

### 2.6. Graph-generation models

- Explore the Ephris prior and other random graph models.
- Prioritize graphs constructed from latent node features, including latent variables that also influence observed features or targets.
- Explore mixtures of group-based connectivity, heterogeneous node degrees, and ordering-based connectivity.
- Generate bipartite graphs and project them into observed graphs where appropriate.
- Calibrate realized graph statistics to cover real-world distributions, including degree heterogeneity, clustering, connectivity, assortativity, and feature/label–topology relationships.

Reference: [Ephris](https://arxiv.org/pdf/2609.37057).

### 2.7. Additional prior mechanisms

- Investigate the generation mechanisms in the supplied additional prior reference and their adaptation to graphs.

Reference: [OpenReview: fhcw14RF94](https://openreview.net/forum?id=fhcw14RF94).

### 2.8. NFA-style task generation

- Explore neighborhood feature aggregation (NFA) as an additional synthetic task-generation mechanism alongside graph layers in the prior.
- Aggregate latent variables or base node features using neighborhood statistics such as numerical means, minima, maxima, and categorical frequencies.
- Combine own-node values and neighborhood summaries as inputs to sampled target-generating functions or downstream SCM nodes, including random MLPs, trees, and other nonlinear mechanisms.
- Explore keeping neighborhood summaries internal to the generator so that prediction requires recovering relational information from the observed graph and node features.
- Vary aggregation operators, hop counts, neighborhood sampling, and the relative contribution of local and aggregated variables to the target.
- Compare NFA-based generation, graph-layer-based generation, and mixtures or compositions of the two.

Reference: [GraphLand, Appendix B.2](https://papers.nips.cc/paper_files/paper/2025/file/8f54de761abc405145d8b4df8bef3f18-Paper-Datasets_and_Benchmarks_Track.pdf).

## 3. Pretraining and training

### 3.1. Larger graphs in the final training stages

- Increase graph sizes during the last training stages or epochs.
- Compare fixed-size training with gradual and staged size curricula.
- Explore stage-specific graph-size distributions, context counts, degree distributions, and learning rates.
- Investigate transfer from small training graphs to large inference graphs.

### 3.2. Tabular pretraining followed by graph adaptation

- Pretrain the tabular backbone with graph modules disabled, then activate graph adapters and train on graph tasks.
- Compare freezing the tabular backbone with joint training after graph activation.
- Compare staged training with training the complete model from scratch.
- Explore adapter initialization and optimizer-state handling at the transition.

### 3.3. Batch size and exponential moving averages

- Explore effective batch size measured as independent synthetic tasks per optimizer update.
- Distinguish task batch size, node count, and gradient accumulation.
- Explore exponential moving averages (EMA) of weights, including decay, start time, and update frequency.
- Compare EMA with existing optimizer averaging and combinations of the two.

### 3.4. Resample context masks on the same synthetic task

- Reuse a synthetic graph with different observed-label subsets.
- Train adaptation to changing amounts and locations of available label evidence.
- Explore amortizing prior-generation cost across multiple context masks.
- Compare one context mask per graph with multiple independent masks.

### 3.5. Graph-branch dropout

- Occasionally disable graph adapters during pretraining so the tabular path remains independently useful.
- Compare task-level disabling of all adapters with selective adapter dropout.
- Explore graph-branch dropout during both joint training and staged tabular-to-graph adaptation.

### 3.6. Intermediate supervision

- Apply auxiliary prediction losses at selected depths or recurrent iteration counts.
- Explore whether intermediate supervision improves optimization and enables shorter inference runs.
- Vary auxiliary-loss strength and placement.
- Investigate the tradeoff between early prediction quality and final-depth performance.

### 3.7. JEPA-style loss for representation geometry

- Explore a joint-embedding predictive architecture (JEPA) objective as an auxiliary loss or a pretraining stage to improve the geometric structure of learned representations.
- Predict target-encoder representations of masked features, nodes, or subgraphs from visible context rather than reconstructing raw inputs.
- Compare applying the objective to feature tokens, pooled node embeddings, and graph-refined representations.
- Explore masking strategies, predictor architectures, target-encoder EMA, and loss weighting relative to the supervised PFN objective.
- Investigate representation collapse, neighborhood structure in embedding space, and transfer to downstream prediction tasks.

### 3.8. Regression objectives

- Compare MSE with quantile regression using the pinball loss and a grid of predicted conditional quantiles.
- Explore Adaptive Quantile Regression (AdaQR), with jointly learned input-dependent smoothing.
- Compare Gaussian-smoothed target-binning objectives, including soft binning and HL-Gaussian.
- Investigate how regression objectives interact with target normalization, heteroskedastic noise, and graph-dependent targets.
- Explore point predictions and uncertainty estimates derived from the predicted distributions or quantiles.

Reference: [Revisiting Tabular Deep Regression: Objectives and the Limits of Global Early Stopping](https://arxiv.org/pdf/2610.13909v1).

## 4. Evaluation recipe

### 4.1. Feature augmentation at inference

- Explore adding random features and graph-derived features to the model inputs at inference.
- Compare structural descriptors such as node degree, neighborhood statistics, and other graph features, individually and in combination with random features.
- Investigate how the number, scale, and construction of added features affect predictions, including averaging predictions across different random feature realizations.

### 4.2. Context subsampling

- Explore context subsampling inspired by the Universal Classifier and graph-aware context selection.
- Compare random context subsets with subsets selected using graph proximity, structural coverage, or feature similarity to query nodes.
- Explore query-specific contexts and prediction aggregation across multiple context subsets.
- Investigate the tradeoff between context size, prediction quality, and inference cost.

Reference: [The Universal Classifier for Graph Learning](https://arxiv.org/pdf/2609.36302).

## 5. Other ideas

### 5.1. Reasoning nodes and features

- Add auxiliary reasoning nodes, feature tokens, or latent workspace representations for intermediate computation before prediction.
- Explore how these representations interact with observed nodes, feature tokens, and labeled context across multiple processing steps.
- Vary their number, connectivity, and update schedule to investigate whether additional reasoning capacity improves predictions.

### 5.2. Pretraining for inference-time compute scaling

- Pretrain the model to use variable inference budgets, including different numbers of recurrent steps, reasoning nodes or tokens, and context views.
- Explore training schedules and objectives that encourage additional inference computation to improve prediction quality.
- Investigate generalization to inference budgets larger than those used during pretraining and compatibility with early exit.

### 5.3. Ideas from nanoGPT speedrun and slowrun

- Review nanoGPT speedrun and nanoGPT slowrun experiments for transferable architecture, optimization, initialization, training-schedule, and implementation ideas.
- Explore adaptations of promising mechanisms to tabular and graph PFN pretraining.
- Investigate both improvements in training efficiency and improvements in predictive quality from longer training or additional computation.
