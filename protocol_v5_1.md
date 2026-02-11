# Cognitive Architecture for Abstract Reasoning
## Research Protocol v5.1

**Principal Investigator:** Tim Antonson
**Research Partner:** Claude Opus 4.6 (Anthropic)
**Affiliation:** PURE Cognitive LLC (planned)
**Date:** February 11, 2026
**Status:** Competition-Focused Build — Architecture Pivot to PCLT Agent
**Version Note:** v5.1 incorporates ADR-005 corrections from pre-implementation verification of reference architecture specifications against actual papers and codebases. Key changes from v5.0: loss function corrected to standard cross-entropy (not Stablemax), Think block layer count corrected to 4 (not 2), outer ACT loop added, FFN expansion corrected to 2× (not 4×), SlotSSM dynamics corrected to Mamba-based (not S4), SOLD performance claim qualified. See ADR-005 for full discrepancy analysis.

---

## 1. What This Study Is

This is a **competition-focused architecture build** that produces publication-quality science as a direct byproduct.

**The mission:** Build a 25–34M parameter cognitive agent that wins ARC-AGI-3 — the interactive reasoning benchmark launching March 25, 2026 — and claims the $700K ARC-AGI-2 grand prize by implementing validated human cognitive principles as native architectural features rather than bolt-on modules. The agent validates its computational substrate on ARC-AGI-1 and ARC-AGI-2 as calibration benchmarks, with TRM (7M, 45% on ARC-AGI-1), URM (14M, 53.8%), and Loop-ViT (18M, 65.8%) as the efficiency baselines to beat and exceed.

**The architectural thesis:** A Predictive Coding Looped Transformer (PCLT) provides a unified computational substrate where **prediction error is the single signal** that simultaneously drives world-model learning, curiosity-based exploration, adaptive computation depth, and anomaly detection across environments. Combined with slot-based object-centric perception, graph-structured exploration, dual-process planning, and three-tier memory, this architecture natively implements all eight biological cognitive principles at a parameter budget (~25–34M) that fits competition constraints and achieves real-time interactive reasoning at 2000 FPS.

**Why PCLT instead of a standard recursive transformer:** The previous protocol (v4.0–v4.1) proposed bolting cognitive modules onto a TRM-style recursive transformer core. Deep architecture research (February 10–11, 2026) revealed three fatal problems with that approach:

1. **The TC⁰ ceiling.** Merrill and Sabharwal (TACL 2023, ACL 2025) proved fixed-depth transformers are contained in TC⁰ complexity — provably unable to solve linear equations, graph connectivity, or simulate finite-state machines. This includes SSMs like Mamba. The escape route is depth that grows with input: looped transformers with effective depth scaling with iteration count escape TC⁰ into problems provably inexpressible at fixed depth.

2. **Biological principles were bolt-ons, not native.** A standard transformer with added modules for working memory, predictive coding, etc. is an engineering workaround. The PCLT makes these principles native to the computational substrate: the loop IS feedback-dominant processing, prediction error IS predictive coding, k-Winners-Take-All IS sparse coding, slot attention IS working memory, entropy-based halting IS metabolic efficiency.

3. **ARC-AGI-3 demands an agent, not a grid predictor.** The previous protocol treated the agent layer as an afterthought (Phase 10 of 11). ARC-AGI-3 is fundamentally an interactive environment requiring exploration, world-model learning, planning, and episodic memory — the agent architecture IS the architecture, not a wrapper around a static predictor.

**Dual output:** Winning the competition IS the publication. A cognitive architecture that achieves competitive or state-of-the-art performance on ARC-AGI benchmarks, with progressive ablation data showing which principles contribute and how they interact, is a stronger paper than any ablation study conducted in isolation.

**The fundamental question:** Can a 25–34M parameter agent implementing validated cognitive principles — where prediction error serves as the unified learning, exploration, and adaptation signal — win ARC-AGI-3, claim the ARC-AGI-2 $700K prize, and outperform systems with 30,000× more parameters on interactive reasoning tasks that frontier AI currently scores 0% on?

**Why this matters:** ARC-AGI-3 resets the entire competitive landscape. Frontier models (GPT-5.2, o3, Gemini) score 0% on interactive environments. The best preview agent scored 12.58%. Humans score near 100%. This is not a benchmark where scale has any demonstrated advantage — it is a benchmark where architectural principles determine everything.

---

## 2. The Competitive Landscape

We are entering a well-developed competition ecosystem with a brand-new benchmark format. Understanding what exists — what works, what fails, and why — is essential to building something that wins.

### 2.1 ARC-AGI Benchmark Evolution

**ARC-AGI-1** (Chollet, 2019; arXiv:1911.01547): Approaching saturation for engineered systems. GPT-5.2 achieved 90.5% (December 2025). Loop-ViT achieves 65.8% at 18M parameters. Remains valuable as a substrate calibration benchmark — our PCLT substrate must beat URM's 53.8% here to validate the architecture before building the agent layer.

**ARC-AGI-2** (arXiv:2505.11831, May 2025): Significantly harder. Pure LLMs score 0%. Best verified: 54% by Poetiq + Gemini 3 Pro at $30.57/task. Under compute constraints ($0.20/task), NVARC achieved 24.03%. The $700K grand prize for >85% remains unclaimed alongside ARC-AGI-3.

**ARC-AGI-3** (preview July 2025, full launch March 25, 2026): Our primary competition target. Interactive video-game-like environments replacing static grid puzzles.

- **Format:** 1,000+ levels across 150+ hand-crafted environments. Each environment is a novel interactive world with rules the agent must discover through exploration.
- **Grid specification:** 64x64 grids, 16 colors, 7 discrete actions (directional movement, interaction, coordinate-click, undo, reset).
- **Five tested capabilities:** Exploration, Planning, Memory, Goal Acquisition, Alignment.
- **Scoring:** Action efficiency relative to human baselines. 200+ human participants establish reference performance.
- **Current performance ceiling:** Frontier AI scored 0% at preview. Best preview agent (StochasticGoose): 12.58%. Humans scored near 100%.
- **Developer Toolkit:** Released January 29, 2026. Python package with local execution engine at 2,000 FPS, agent templates, hosted API. Three public environments: LS20 (map navigation), VC33 (object manipulation), FT09 (pattern matching).
- **Official resources:** github.com/arcprize/ARC-AGI-3-Agents.
- **Competition rules (compute budget, hardware, prizes) NOT yet announced.** ARC Prize confirmed continuing ARC-AGI-2 Grand Prize ($700K) alongside ARC-AGI-3.

### 2.2 ARC-AGI-3 Preview Results — Critical Lessons

**StochasticGoose** (Dries Smit, Tufa Labs) — 1st place, 12.58%: CNN-based action-change predictor trained via supervised learning. Learned which actions modify game state, sampled them preferentially. Completed 18 levels across 2 games but required 255,964 actions. Key insight: predicting which actions are meaningful dramatically filters the search space.

**Graph-Based Explorer** (Rudakov et al., arXiv:2512.24156) — effectively matched 1st place after bug fix: Zero learning. Pure systematic state-space enumeration. Directed graph where nodes are unique states (image hash), edges are action transitions. Action selection: untested actions first, then shortest path to states with untested actions. Solved median 30/52 levels. Key insight: systematic exploration dominates LLM-guided exploration. Limitation: linear scaling with state-space size, zero cross-level transfer.

**All LLM-based agents:** 0%. Frontier models including GPT-4 scored exactly 0%.

**Three critical lessons:** (1) Systematic exploration beats random or LLM-guided exploration. (2) Learning which actions change state dramatically filters the action space. (3) Preview environments were vulnerable to brute-force — ARC Prize confirmed hardening for launch. Winning requires genuine understanding, not exhaustive search.

### 2.3 Small-Model Substrate Baselines (ARC-AGI-1/2)

These establish the performance floor our PCLT substrate must exceed before building the agent layer.

| System | Parameters | Architecture | ARC-AGI-1 | ARC-AGI-2 | Key Innovation |
|---|---|---|---|---|---|
| Loop-ViT (Feb 2026) | 18M | Looped conv + self-attention, entropy-based dynamic exit | 65.8% | — | Weight-tied hybrid block, dynamic halting |
| URM (Dec 2025) | 14M | 4-layer looped transformer + ConvSwiGLU + TBPTL + outer ACT | 53.8% | 16% | Truncated backprop through loops, adaptive computation |
| TRM (Oct 2025) | 7M | Recursive transformer, 16-step refinement | 45% | 8% | Final-cycle-only gradients, puzzle embeddings |
| HRM (Jun 2025) | 27M | Dual-timescale hierarchical modules | 40.3% | — | Slow planner + fast worker (hierarchy debunked) |
| CompressARC | 76K | MDL neural program compression | 20% | ~4% | No pretraining, no dataset |
| GPT-5.2 | >1T | Frontier LLM | 90.5% | ~52.9% | Pure scale |

The formula: (architecture x test-time adaptation x data quality) leads to performance, NOT (model size) leads to performance. Every top small-model performer uses recursive/iterative computation.

### 2.4 World Model and Agent Architecture Baselines

| System | Parameters | Domain | Key Features | Relevance |
|---|---|---|---|---|
| DreamerV3 (Nature 2025) | 12M–200M | 150+ diverse tasks | RSSM, categorical stochastic vars, fixed hyperparameters | Reference world model; 12M config achieves comparable performance |
| SOLD (ICML 2025) | ~12M | Relational manipulation | Slot-based dynamics, outperforms DreamerV3 by >2x on relational variants requiring inter-object reasoning | Object-centric world model validation |
| SlotSSMs (NeurIPS 2024) | <5M | 3D visual reasoning | Per-slot Mamba-based SSM transitions + Slot Mixer cross-slot attention, O(n) complexity | Slot dynamics at our scale |
| AXIOM (2025) | Bayesian | Novel game mastery | Object-centric Gaussian mixture, masters games in 5K–10K interactions | Active inference for rapid rule learning |

### 2.5 What Francois Chollet Says Is Missing

Core thesis from June 2025 AI Startup School: Current AI reasoning is tied to model knowledge, but human reasoning is not bound to knowledge. The gap is separating knowledge from reasoning — reasoning about genuinely novel problems using only core cognitive priors.

ARC Prize 2025 Technical Report (arXiv:2601.10904): Accuracy gap is bottlenecked by engineering. Efficiency gap is bottlenecked by science and new ideas. ARC-AGI-3's interactive format targets capabilities "fundamentally missing" from current systems.

### 2.6 Our Differentiation

No existing approach combines a biologically-native computational substrate with an interactive agent architecture.

| System | Substrate | Agent Layer | Bio Principles Native | Progressive Ablation |
|---|---|---|---|---|
| StochasticGoose | CNN | Supervised action predictor | 0 | No |
| Graph-Based Explorer | None (training-free) | Graph search | 0 | N/A |
| TRM/URM/Loop-ViT | Looped transformer | None (grid predictor) | 1–2 | No |
| DreamerV3 | RSSM | Actor-critic | 0 | No |
| MIRAGE | Frozen meta-trained TF | Symbolic schema engine | 4–5 | No |
| **This study** | **PCLT** | **Full agent** | **8 native** | **Yes** |

### 2.7 The HRM Cautionary Tale

HRM (Wang et al., arXiv:2506.21734) claimed hierarchical brain-inspired architecture was central to performance. Independent analysis demonstrated an 8-layer single-module network performs comparably. Hierarchy contributed minimally.

**Binding methodological commitment:** No cognitive principle will be claimed as contributing to performance unless both forward and reverse ablation demonstrate statistically significant improvement. Every claim must be backed by ablation evidence, not architectural narrative.

---

## 3. Research Questions and Hypotheses

### 3.1 Primary Research Question

Can a 25–34M parameter PCLT-based agent — where prediction error serves as the unified signal for learning, exploration, planning, and adaptation — win ARC-AGI-3, achieve prize-competitive performance on ARC-AGI-2, and outperform architecturally equivalent baselines on abstract reasoning benchmarks?

### 3.2 Hypotheses

**H1 (PCLT Substrate Dominance):** The PCLT substrate (looped transformer + predictive coding loss + slot attention + kWTA sparsity + entropy-based halting) will exceed Loop-ViT's 65.8% on ARC-AGI-1 and exceed 25% on ARC-AGI-2 at 22M or fewer parameters, establishing a new state-of-the-art for small-model abstract reasoning and validating the biological-native computational core before agent-layer development.

**H2 (Prediction Error Unification):** Using prediction error as the unified signal for world-model learning, exploration, adaptive computation, and anomaly detection will produce a more sample-efficient agent than equivalent architectures using separate modules for each function. Measured by fewer interactions to learn novel environment rules compared to ablated variants with separate curiosity modules.

**H3 (Object-Centric World Model):** Slot-based world model (Mamba-based SlotSSM dynamics over 4–8 entity slots) will learn environment rules faster than a monolithic state-vector world model of equivalent parameters.

**H4 (Dual-Process Planning):** System 1 (policy MLP) + System 2 (MCTS in latent space) with learned switching will achieve higher action efficiency than either alone. System 1 handles >90% of actions at full speed; System 2 engages only when prediction uncertainty warrants deliberation.

**H5 (Three-Tier Memory Transfer):** Working memory (slots) + episode memory (DND) + rule memory (environment-type abstractions) will enable cross-level transfer, measured by performance on later levels exceeding earlier levels within environment types.

**H6 (Graph-Structured Exploration):** Prediction-error-driven exploration augmented by explicit state graph tracking will solve more levels with fewer actions than either approach alone.

**H7 (Adaptive Computation):** Entropy-based dynamic halting (Loop-ViT pattern) will allocate more compute iterations to novel situations and fewer to familiar ones, producing both speed (2000 FPS average) and accuracy.

**H8 (Synergy):** The combined effect of all components will exceed the sum of individual effects. Prediction error unification predicts superadditive interaction.

**H9 (Win ARC-AGI-3):** The full PCLT agent will win or place top-3 on the ARC-AGI-3 leaderboard, demonstrating that cognitive architecture principles at 25–34M parameters can solve interactive reasoning tasks that frontier AI (>1T parameters) scores 0% on. Cognitive-ablated variants will perform measurably worse, confirming that architectural principles — not just scale or search — drive performance.

**H10 (Radical Efficiency):** The PCLT agent will outperform frontier models (>1T parameters) on ARC-AGI-3 using 25–34M parameters — a 30,000× parameter efficiency advantage — demonstrating that architectural principles matter more than scale for genuine reasoning.

### 3.3 Falsification Criteria

- H1 falsified if PCLT substrate does not exceed 53.8% on ARC-AGI-1 at 22M or fewer parameters. (Note: 53.8% is the minimum viability threshold. The target is >65.8%.)
- H2 falsified if prediction error as exploration signal performs equivalently or worse than separate RND/ICM module.
- H3 falsified if slot-based world model learns rules no faster than monolithic world model.
- H4 falsified if System 1 alone matches dual-process action efficiency, or System 2 never activates.
- H5 falsified if no measurable learning curve across levels within environment types.
- H6 falsified if adding learned prediction to graph exploration does not reduce total actions.
- H7 falsified if dynamic halting does not correlate with novelty, or fixed iterations match performance.
- H8 falsified if combined effect is less than or equal to sum of individual effects.
- H9 falsified if agent achieves 0% on ARC-AGI-3, or cognitive variants perform equivalently to ablated. (Note: 0% is the falsification floor. The target is 1st place.)
- H10 falsified if frontier models consistently outperform our agent on ARC-AGI-3.

---

## 4. Architecture Specification

### 4.1 Design Philosophy: Prediction Error as Unified Signal

The architecture is built around a single organizing principle: prediction error drives everything. The brain minimizes prediction error (free energy) across all levels of processing — perception, action, learning, and attention allocation. The PCLT implements this:

- **World model training:** Prediction errors between predicted and observed next-states train the dynamics model.
- **Exploration signal:** Prediction error magnitude = epistemic uncertainty. High error means unknown rule means explore there.
- **Adaptive computation:** Persistent prediction errors trigger System 2 engagement. Low errors mean System 1 handles it.
- **Anomaly detection:** Sudden error spikes after low-error periods indicate rule changes across levels, triggering re-exploration.
- **Halting criterion:** Entropy-based dynamic exit halts looped computation when prediction error converges — metabolic efficiency.

This is not a metaphor. These are the same tensor computations repurposed for different functional roles — no separate curiosity module, no separate anomaly detector, no separate halting network. One signal, five uses.

### 4.2 The PCLT Substrate

The computational core processes observations and produces actions through a looped predictive coding architecture: Encode, Think (iterative with PC loss), Decode.

**Grid Encoder + Slot Attention (~3–5M):**
- 16-dimensional color embedding per cell (discrete grid, no CNN needed)
- Patch-based spatial grouping (4x4 patches yielding 256 patch tokens for 64x64 grid)
- 4–8 Slot Attention iterations decompose patches into object-centric slots
- Each slot represents one entity: agent, objects, goal markers, background regions
- Natively implements parallel streams (each slot processes independently) and capacity-constrained working memory (4–8 items)
- Reference: Locatello et al. (NeurIPS 2020), OCRA (Webb et al., NeurIPS 2023)

**Think Block — Looped Transformer (~8–12M):**
- 4-layer transformer block with weight tying (verified from URM: 4 layers, NOT 2), iterated via two-level loop structure
- Inner loop: 8 iterations with TBPTL gradient strategy (first 2 forward-only, remaining 6 with per-step loss). Loss: standard cross-entropy at each backprop iteration, summed. (ADR-005: NOT Stablemax CE — neither URM nor Loop-ViT uses Stablemax.)
- Outer loop: Adaptive Computation Time (ACT) with up to 16 steps, each running the full 8-step inner loop. Learned halting probability per step. (ADR-005: this outer ACT loop was missing from v5.0 initial draft.)
- ConvSwiGLU FFN (from URM): fused W_up projection split into gate and value, SiLU gating, 1D depthwise conv (k=2) after gating, second SiLU, down-projection. Hidden expansion 2× (m = d_model). (ADR-005: expansion is 2×, NOT 4×.)
- At each iteration t, block generates predictions of state at t+1
- Predictive coding auxiliary loss: prediction error (predicted vs. actual state at t+1) added to training loss (Phase 1 — disabled in Phase 0)
- Entropy-based dynamic exit: halt when internal state crystallizes into low-entropy attractor (Loop-ViT pattern, parameter-free, τ=0.05). Inference-only — not used during training. (Phase 1 — disabled in Phase 0)
- Natively implements feedback-dominant processing (each loop is feedback), predictive coding (errors drive refinement), dual-process (few iterations = System 1, many = System 2), and metabolic efficiency (stop when error minimized)
- Reference: Loop-ViT (arXiv:2602.02156), URM (arXiv:2512.14693), TRM (arXiv:2510.04871)

**Slow Planner (~3–5M):**
- Updates every 4th Think iteration (temporal hierarchy)
- Maintains compressed representation of reasoning trajectory
- Sets subgoals, evaluates plan progress, triggers re-exploration
- Natively implements multiple memory systems (slow module as semantic/compressed)
- Reference: HRM dual-timescale design (arXiv:2506.21734)

**k-Winners-Take-All Sparsity:**
- After each Think iteration, only top 5% of activations retained
- Natively implements sparse coding at 2–5% activity
- Creates interference-resistant representations for rapid environment switching
- Reference: Self-Ablating Transformers (2025), NGC-Learn lateral inhibition

### 4.3 The Agent Layer

Built on top of the PCLT substrate, the agent layer handles environment interaction.

**World Model — SlotSSM Dynamics (~4–6M):**
- Independent Mamba-based selective SSM transitions per slot (object-centric dynamics). (ADR-005: Mamba, NOT S4 — SlotSSMs paper uses input-dependent selective parameterization.)
- Slot Mixer attention layer for inter-object interaction
- Action-conditioned via FiLM (Feature-wise Linear Modulation)
- Prediction heads: next-state slot prediction, reward/done MLPs on aggregated slots
- Trained by the same prediction error that drives PCLT convergence
- Reference: SlotSSMs (NeurIPS 2024), SOLD (ICML 2025), R2I (ICLR 2024)

**Exploration System (~1–2M):**
- Prediction-error curiosity: world model prediction error magnitude is intrinsic reward. In deterministic grid environments, prediction error = pure epistemic uncertainty (no noisy TV problem).
- NovelD frontier-based priority: intrinsic reward = difference in novelty between consecutive states, producing breadth-first exploration at the boundary of known/unknown regions.
- Graph-structured state tracking: directed graph with nodes = unique states (hash), edges = action transitions. Provides systematic coverage guarantee and shortest-path navigation to unexplored frontiers.
- Hypothesis-driven exploration: after initial broad exploration, form hypotheses about rules and design targeted action sequences to test them.
- Reference: NovelD (NeurIPS 2021), Graph-Based Explorer (arXiv:2512.24156), AXIOM (2025)

**Dual-Process Planning:**
- System 1 (every frame, <0.1ms): Policy MLP or SSM head on latent state (~2–3M). Single forward pass. Handles routine movement and cached plans. Trained via imagination rollouts in world model.
- System 2 (on-demand, 5–50ms): MCTS with 50–200 simulations in latent space using learned world model. Batched on GPU. Learned value function evaluates leaf nodes.
- Switching criterion: three signals trigger System 2: (1) world model prediction error exceeds threshold, (2) policy entropy exceeds threshold, (3) environment-type change detection. Meta-controller network (~0.5–1M) learns when to switch.
- Amortized planning: MCTS results cached and fed back as training targets for System 1, progressively expanding the set of situations handled at full speed.
- Reference: DreamerV3 actor-critic, MuZero MCTS, E-MCTS (Oren et al.)

**Three-Tier Memory:**
- Working memory (per-step): PCLT's 4–8 slot attention slots. Current entities. Refreshes each observation. No additional parameters.
- Episode memory (per-level): Differentiable Neural Dictionary (DND) storing (state embedding, Q-value) pairs. O(log n) kd-tree retrieval. Non-parametric storage + ~1–2M retrieval networks.
- Rule memory (per-environment-type): abstract pattern storage indexed by environment-type embeddings. Retrieved via cosine similarity. FiLM modulation adapts policy based on retrieved rules. Enables cross-level transfer.
- Reference: Neural Episodic Control, ArcMemo (arXiv:2504.20109), CAVIA meta-learning

### 4.4 Parameter Budget

| Component | Parameters | Biological Principle |
|---|---|---|
| Grid encoder + Slot Attention (4–8 slots) | 3–5M | Parallel streams, working memory |
| Think block (4 layers, looped 8x inner / up to 16x outer ACT) | 8–12M | Feedback, predictive coding, dual-process, efficiency |
| Slow planner module | 3–5M | Multiple memory systems |
| World model (SlotSSM dynamics) | 4–6M | World-model learning via prediction error |
| Policy + value heads (System 1) | 2–3M | Fast action selection |
| Memory retrieval networks (DND interface) | 1–2M | Episode/rule memory |
| Meta-controller (System 2 trigger) | 0.5–1M | Uncertainty detection, computation routing |
| **Total** | **~22–34M** | **All 8 principles native** |

### 4.5 Mapping Eight Biological Principles to Architecture

| Principle | How PCLT Implements It | Where |
|---|---|---|
| Feedback-dominant (10:1) | Each loop iteration is a feedback pass | Think block loop |
| Predictive coding | Prediction error between iterations is auxiliary loss; same error drives exploration | Think block + World model |
| Sparse coding (2–5%) | kWTA retains top 5% of activations | Think block nonlinearity |
| Parallel streams | Slot Attention decomposes input into independent slots | Encoder |
| Working memory (3–4) | 4–8 capacity-limited slots with competitive attention | Slot Attention |
| Multiple memory systems | Working (slots) + Episode (DND) + Rule (abstractions) + Slow planner | Three-tier memory + Planner |
| Dual-process reasoning | Few iterations = System 1, many = System 2; policy MLP vs. MCTS | Think block + Planning |
| Metabolic efficiency | Entropy-based dynamic exit; stop when error minimized | Halting criterion |

### 4.6 Baseline Architecture

For fair scientific comparison:
- **PCLT substrate baseline:** Same looped transformer WITHOUT predictive coding loss, WITHOUT slot attention (flat encoding), WITHOUT kWTA, WITHOUT dynamic halting (fixed iterations). Matched parameters.
- **Standard transformer baseline:** Non-recursive transformer of matched parameter count (lower bound).
- **External calibration:** Published TRM (45%), URM (53.8%), Loop-ViT (65.8%) on ARC-AGI-1.
- **Agent baseline (ARC-AGI-3):** Graph-based explorer (training-free, ~30/52 levels) as systematic-exploration floor to beat.

All variants receive same training data, compute budget, hyperparameter tuning effort, and random seeds.

---

## 5. Progressive Build Sequence

The build has two tracks: Substrate Track (Phases 0–2, validated on ARC-AGI-1/2) and Agent Track (Phases 3–5, validated on ARC-AGI-3 public environments). Phase 6 is competition optimization.

### Track A: Substrate Development (ARC-AGI-1/2 Validation)

**Phase 0: PCLT Substrate Core**
- Looped transformer block (4 layers, weight-tied, 8 inner iterations with TBPTL + outer ACT loop up to 16 steps). (ADR-005: 4 layers verified from URM, NOT 2.)
- ConvSwiGLU FFN (from URM): fused projection, SiLU gating, 1D depthwise conv k=2, 2× hidden expansion
- TBPTL gradient strategy: 2 forward-only + 6 backprop inner iterations, standard cross-entropy loss summed at each backprop step. (ADR-005: NOT Stablemax CE.)
- Flat grid encoding (12 tokens, 512-dim embeddings)
- Per-task puzzle embeddings (100× model LR via main optimizer with LR multiplier). (ADR-005: SignSGD unverified — TRM codebase check needed. Use main optimizer as default.)
- Learned initial states, post-norm RMSNorm, RoPE positional encoding
- AdamAtan2 optimizer, LR=1e-4, weight decay=0.1, EMA on model parameters
- Standard cross-entropy loss (no PC auxiliary loss yet)
- Fixed iteration count (no dynamic halting yet)
- ~9–12M parameters
- Gate: at least 53.8% on ARC-AGI-1 (match URM)
- This validates the looped substrate before adding biological principles

**Phase 1: Biological Principles — Native Integration**
- ADD: Slot Attention encoder (4–8 slots, replacing flat encoding). Validates parallel streams and working memory.
- ADD: Predictive coding auxiliary loss on Think block iterations. Validates predictive coding.
- ADD: k-Winners-Take-All sparsity (top 5%). Validates sparse coding.
- ADD: Entropy-based dynamic exit (Loop-ViT pattern). Validates metabolic efficiency and dual-process (variable depth).
- ADD: Slow planner module (every 4th iteration). Validates multiple memory systems and feedback-dominant processing.
- ~15–22M parameters
- Gate: exceeds Phase 0 by at least 5% absolute on ARC-AGI-1
- Each sub-component ablated individually (forward ablation)
- If any sub-component hurts performance: debug, redesign, skip (1 week max each)

**Phase 2: Test-Time Adaptation + Full Substrate Validation**
- ADD: Full-parameter TTT on few-shot examples. Leave-one-out training data, All-Outputs loss, augmentation voting.
- Evaluate on ARC-AGI-1 AND ARC-AGI-2.
- ~15–22M parameters (same model, TTT applied at inference)
- Gate: at least 60% on ARC-AGI-1 with TTT (target: exceed Loop-ViT's 65.8%)
- Gate: at least 15% on ARC-AGI-2 with TTT (target: >25%, stretch: >85% for $700K prize)
- Substrate is LOCKED after this phase. Agent track builds on this.

### Track B: Agent Development (ARC-AGI-3 Validation)

**Phase 3: World Model + Environment Interface**
- ADD: ARC-AGI-3 environment interface (official SDK integration)
- ADD: SlotSSM world model (independent per-slot Mamba-based SSM dynamics + Slot Mixer). Action-conditioned via FiLM. (ADR-005: Mamba, NOT S4.)
- ADD: Prediction error as intrinsic reward for exploration
- Train world model on public environments (LS20, VC33, FT09)
- ~19–28M parameters (substrate + world model)
- Gate: world model prediction error drops below 10% on known rules within 500 interactions on public environments
- Gate: prediction error drops measurably as environment rules are discovered

**Phase 4: Exploration + Planning**
- ADD: Graph-structured state tracker (hash-based nodes, action edges)
- ADD: NovelD frontier-based exploration priority
- ADD: System 1 policy head (MLP on latent state)
- ADD: System 2 MCTS in latent space (50–200 sims)
- ADD: Meta-controller for System 1/2 switching
- Train actor-critic via imagination rollouts in world model
- ~22–32M parameters
- Gate: solves at least 1 public environment level with fewer actions than graph-based explorer baseline
- Gate: System 2 activates on novel situations, not routine actions
- Gate: agent achieves 2000 FPS average throughput

**Phase 5: Memory + Transfer + Full Agent Integration**
- ADD: Episode memory (DND with kd-tree retrieval)
- ADD: Rule memory (environment-type abstractions, FiLM modulation)
- ADD: Cross-level transfer mechanism (later levels use earlier discoveries)
- ADD: Hypothesis-driven exploration (targeted rule testing after initial survey)
- All components active simultaneously
- ~24–34M parameters (final agent)
- Gate: measurable learning curve across levels within environment types
- Gate: performance on later levels exceeds earlier levels (transfer working)
- Gate: exceeds StochasticGoose's preview score on public environments (target: 1st place at launch)

### Track C: Competition Optimization

**Phase 6: ARC-AGI-3 Competition Optimization**
- Reverse ablation: remove each component, measure degradation
- Pairwise interaction tests for strong theoretical predictions
- Synthetic environment generation for meta-training generalization
- Profile and optimize for competition compute constraints
- Submission preparation and testing
- ARC-AGI-2 submission (fallback competition target)
- Paper drafting with ablation tables

**Critical design rules (apply to ALL phases):**
- Train from scratch at each phase for clean attribution.
- Regression testing: previous phases' tests must still pass.
- 1-week maximum per sub-component. If it doesn't improve after debug/redesign, skip it.
- Competition decision: after each gate, ask "does this improve competition performance?" If no, consider shipping without it.
- HRM Cautionary Principle: no claim without forward AND reverse ablation evidence.

---

## 6. Training Methodology

### 6.1 Data Strategy

**Substrate training (Phases 0–2) — ARC-AGI-1/2:**

| Source | Size | Quality | Access |
|---|---|---|---|
| ARC-AGI-1 training set | 400 tasks | Gold standard | Public |
| RE-ARC | Unlimited (procedural) | High | github.com/michaelhodel/re-arc |
| BARC | 400K synthetic | Medium-high | github.com/xu3kev/BARC |

Augmentation: 8 geometric transforms x color permutations yielding up to 3.2M effective training examples from 400 base tasks.

**Agent training (Phases 3–5) — ARC-AGI-3:**

| Source | Purpose |
|---|---|
| ARC-AGI-3 public environments (LS20, VC33, FT09) | Primary development and validation |
| Procedural grid-world generation | Meta-training for generalization to 147+ unseen environments |
| World-model imagination rollouts | DreamerV3-style training from imagined trajectories |

### 6.2 Test-Time Adaptation

**Substrate TTT (ARC-AGI-1/2):** Full-parameter fine-tuning per task (NOT LoRA — at 15–22M params the model IS LoRA-scale). Leave-one-out training data from demonstration pairs. All-Outputs loss. Augmented inference with hierarchical voting.

**Agent TTT (ARC-AGI-3):** World model updates in real time during interaction (online predictive coding). Episode and rule memory accumulate during gameplay. No gradient-based adaptation needed — prediction error drives online learning natively. CAVIA-style meta-learned initialization enables rapid adaptation to new environment types.

### 6.3 Fair Comparison Principles

Compute-matched, data-identical, hyperparameter-fair (N=20 random search), minimum 5 random seeds, TTT-fair across all variants.

### 6.4 Training Infrastructure

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti, 16GB GDDR7 |
| CPU | Intel Core Ultra 9 285K, 24-core |
| RAM | 64GB DDR5 @ 6400MHz |
| Storage | 2TB PCIe 4.0 SSD |
| OS | Windows 11 Pro |
| Supplementary | Kaggle 30 hrs/week P100/T4 |

---

## 7. Evaluation Framework

### 7.1 Dual-Track Evaluation

**Track A — Substrate (ARC-AGI-1/2):**

| Benchmark | Metric | Baselines | Target |
|---|---|---|---|
| ARC-AGI-1 (120 tasks) | pass@2 accuracy | TRM 45%, URM 53.8%, Loop-ViT 65.8% | >65.8% (beat SOTA) |
| ARC-AGI-2 (120 tasks) | pass@2, $/task | NVARC 24% at $0.20/task | >25% (stretch: >85% for $700K prize) |

**Track B — Agent (ARC-AGI-3):**

| Benchmark | Metric | Baselines | Target |
|---|---|---|---|
| ARC-AGI-3 public environments | Action efficiency vs. humans | StochasticGoose 12.58% | **1st place** |
| ARC-AGI-3 public environments | Level completion rate | Graph-based 30/52 levels | >40/52 levels |
| ARC-AGI-3 public environments | Adaptation speed | No established baseline | Measurable cross-level transfer |

### 7.2 Metrics

**Competition:** Action efficiency vs. humans (primary), level completion rate, $/task.

**Substrate:** Task accuracy (pass@2), $/task, parameter efficiency.

**Agent-specific:** World model prediction accuracy over time, exploration efficiency (actions per discovered rule), planning depth utilization (System 1 vs. System 2 frequency), memory transfer effectiveness (later vs. earlier level performance), adaptation speed.

**Calibration:** ECE, selective prediction accuracy.

### 7.3 Success Criteria

| Benchmark | Target | Stretch | Justification |
|---|---|---|---|
| ARC-AGI-1 | >65.8% | >75% | Beat Loop-ViT SOTA at fewer parameters |
| ARC-AGI-2 | >25% at $0.20/task | >85% ($700K prize) | Competitive at contest compute; grand prize if substrate is strong enough |
| ARC-AGI-3 | **1st place** | >50% score | Win the competition. Current best: 12.58%. Humans: ~100% |

---

## 8. Ablation Protocol

### 8.1 Build IS the Ablation

Progressive build generates ablation data. Each phase measures marginal contribution.

### 8.2 Three Approaches

**Forward ablation (primary):** Phase 0 through 2 (substrate) and Phase 3 through 5 (agent). Each phase's additions measured against previous baseline.

**Reverse ablation (validation):** From full architecture, remove one component at a time.

**Pairwise interaction tests:** (1) Slot Attention x Predictive Coding, (2) World Model x Graph Exploration, (3) Dual-Process x Meta-Controller, (4) Episode Memory x Rule Memory.

### 8.3 Competition-Optimized Process

Train and measure. If improvement: keep. If not: debug, redesign, skip. Maximum 1 week per component.

---

## 9. Execution Plan

### 9.1 Phase Gate Criteria

| Phase | Gate Criteria |
|---|---|
| 0 (PCLT Core) | Converges; at least 53.8% on ARC-AGI-1; eval pipeline works; ~9-12M params |
| 1 (Bio Principles) | Each sub-component non-degenerate; exceeds Phase 0 by at least 5% |
| 2 (TTT) | TTT functional; at least 60% on ARC-AGI-1 (target >65.8%); at least 15% on ARC-AGI-2 (target >25%) |
| 3 (World Model) | Prediction error drops with interaction; <10% on known rules in 500 steps |
| 4 (Exploration + Planning) | Beats graph baseline on at least 1 level; System 2 activates properly; 2000 FPS |
| 5 (Full Agent) | Learning curve across levels; transfer measurable; exceeds preview SOTA |
| 6 (Competition) | Reverse ablation done; submission ready |

### 9.2 Timeline

| Effort | Activity |
|---|---|
| Week 1 | Infrastructure + Phase 0: PCLT substrate core. Gate: at least 53.8% on ARC-AGI-1 |
| Week 2 | Phase 1 (bio principles) + Phase 2 (TTT + validation). Gate: at least 60% |
| Week 3 | Phase 3: World model + environment interface |
| Week 4 | Phase 4: Exploration + dual-process planning |
| Week 5 | Phase 5: Memory + transfer + full agent |
| Week 6 | Phase 6: Ablation, optimization, submission prep |
| Week 7 | Competition submission + paper drafting |

**Critical path:** Phase 0 is the foundation. Substrate must be validated before agent development.

### 9.3 Open-Source Arsenal

**Substrate:** Loop-ViT (github.com/WenjieShu/LoopViT), URM (github.com/UbiquantAI/URM), TRM (github.com/SamsungSAILMontreal/TinyRecursiveModels), SlotSSMs (github.com/JindongJiang/SlotSSMs), TorchDEQ (github.com/locuslab/torchdeq), OCRA (github.com/Shanka123/OCRA).

**Agent:** ARC-AGI-3 Agents (github.com/arcprize/ARC-AGI-3-Agents), ARC-AGI Toolkit (github.com/arcprize/ARC-AGI), DreamerV3 (github.com/danijar/dreamerv3), Graph-Based Explorer (github.com/dolphin-in-a-coma/arc-agi-3-just-explore).

**Data:** RE-ARC (github.com/michaelhodel/re-arc), BARC (github.com/xu3kev/BARC), ARC-DSL (github.com/michaelhodel/arc-dsl), arckit (github.com/mxbi/arckit).

---

## 10. Risk Management

### 10.1 Competition Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| ARC-AGI-3 rules impose unexpected constraints | Medium | High | Modular architecture. ARC-AGI-2 as fallback |
| Phase 0 substrate cannot match URM's 53.8% | Medium | Critical | Study Loop-ViT/URM codebases; use URM hyperparameters as starting point |
| Preview environments not representative of full benchmark | Medium | High | Meta-training on procedural environments |

### 10.2 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| PC loss destabilizes training | Medium | High | Start at low weight (0.01), gradually increase |
| Slot attention collapses | High | High | Diversity loss. Gate check for non-degenerate assignment |
| World model too slow for 2000 FPS | Medium | Medium | Profile early. Mamba-based SlotSSM provides efficient O(n) per-step computation |
| Online learning insufficient for rule discovery | Medium | High | Fallback: gradient-based fine-tuning per environment |

### 10.3 Scientific Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Cognitive principles don't help | Medium | Critical | Legitimate finding. Report honestly |
| HRM-style debunking | High | High | HRM Cautionary Principle binding. No claim without ablation |
| Kitchen sink critique | High | High | Progressive build IS the defense |

### 10.4 Complexity Creep

Start simple. Each component's initial implementation is the simplest thing that could demonstrate the principle. If the full agent is slower to develop than timeline allows, ship the best-performing subset. A substrate that wins ARC-AGI-1/2 without the agent layer is still a top publication.

---

## 11. Publication Strategy

### 11.1 Three Scenarios

**Scenario A — Win ARC-AGI-3 (and potentially ARC-AGI-2 $700K prize):** Lead with competition results. PCLT Agent Wins ARC-AGI-3 with YM Parameters While Frontier AI Scores 0%.

**Scenario B — Strong substrate, modest agent:** Lead with substrate. Predictive Coding Looped Transformer Achieves X% on ARC-AGI-1.

**Scenario C — Null results on cognitive principles:** Lead with ablation methodology. Systematic Evaluation of Bio-Inspired Principles: What Works, What Doesn't, and Why.

### 11.2 Primary Venue: ICLR 2027

Submission approximately October 2026. Receptive to bio-inspired work. Protection for non-SOTA results.

### 11.3 Mandatory Citations

Chollet (2019), ARC Prize Technical Report (arXiv:2601.10904), Loop-ViT (arXiv:2602.02156), URM (arXiv:2512.14693), TRM (arXiv:2510.04871), Merrill and Sabharwal (TACL 2023), Graph-Based Explorer (arXiv:2512.24156), DreamerV3 (Nature 2025), SlotSSMs (NeurIPS 2024), SOLD (ICML 2025), Locatello et al. (NeurIPS 2020), Chalk et al. (2018), HRM debunking (arXiv:2510.00355), MIRAGE (arXiv:2507.18868), NovelD (NeurIPS 2021), Mamba (Gu & Dao, 2024).

---

## 12. Connection to the PURE Research Program

Study 1's discovery that Fear dominance dramatically outperforms rule-based scaffolding is directly implemented: prediction error IS the architectural citizen of uncertainty. Study 4 augments existing LLMs with memory; this study builds multiple memory types from scratch. Study 6 is a sibling implementation of biological vision neuroscience. Study 8 provides cross-modal convergence evidence.

---

## 13. Ethical Considerations

Small neural networks trained from scratch on public datasets. No proprietary data, no personal information, no deception, no deployment-risk scale. All code, weights, and submissions released under MIT License.

---

## 14. Version History

| Version | Date | Changes |
|---|---|---|
| 1.0–4.1 | Feb 5–10, 2026 | TRM-informed recursive transformer approach. Archived. |
| 5.0 | Feb 11, 2026 | Ground-up rebuild. Architecture pivoted to PCLT substrate + interactive agent. Prediction error as unified signal. 7 phases (Substrate Track + Agent Track + Competition). New baselines: Loop-ViT 65.8%, URM 53.8%, StochasticGoose 12.58%, graph-based explorer 30/52 levels. Parameter budget 25–50M. New hypotheses H1–H10. HRM Cautionary Principle preserved. See ADR-004 for pivot rationale. |
| 5.1 | Feb 11, 2026 | ADR-005 corrections from pre-implementation verification. Loss: standard CE, not Stablemax. Think block: 4 layers, not 2. Outer ACT loop (up to 16 steps) added. FFN expansion: 2×, not 4×. SlotSSM dynamics: Mamba-based, not S4. SOLD claim qualified. Parameter estimates revised downward (~22-34M total, ~9-12M substrate). Puzzle embedding optimizer flagged as unverified (SignSGD → default to main optimizer). Hypotheses H1, H9, H10 rewritten to target WINNING ARC-AGI-3 and claiming ARC-AGI-2 $700K prize, not just matching baselines. Success criteria, evaluation targets, and Phase 2/5 gates updated accordingly. |

---

*This protocol was designed using the PURE Method applied to competition-focused systems research. It prioritizes winning ARC-AGI-3 while producing rigorous, publication-quality scientific data through progressive ablation of human-inspired cognitive principles natively implemented in a Predictive Coding Looped Transformer substrate. All reference architecture specifications have been verified against source papers and codebases per ADR-005.*
