# Cognitive Architecture for Abstract Reasoning
## Research Protocol v4.1

**Principal Investigator:** Tim Antonson
**Research Partner:** Claude Opus 4.6 (Anthropic)
**Affiliation:** PURE Cognitive LLC (planned)
**Date:** February 10, 2026
**Status:** Competition-Focused Build — Architecture Design Phase
**Version Note:** v4.1 incorporates corrections from ADR-002 (TRM codebase analysis) and ADR-003 (TRM/URM corrections). Key changes: puzzle embeddings replace LoRA, final-cycle-only gradients replace deep supervision, Phase 0.5 added for URM's TBPTL + ConvSwiGLU, all hyperparameters corrected.

---

## 1. What This Study Is

This is a **competition-focused architecture build** that produces publication-quality science as a direct byproduct.

**The mission:** Build a small-parameter (7–50M) cognitive architecture that wins ARC-AGI-3 — the interactive reasoning benchmark launching March 25, 2026 — by implementing validated human cognitive principles as architectural features rather than emergent properties. The architecture validates on ARC-AGI-1 and ARC-AGI-2 as calibration benchmarks, with TRM (7M parameters, 40% on ARC-AGI-1) and URM (14M parameters, 53.8% on ARC-AGI-1) as the efficiency baseline to beat at every development stage.

**Dual output:** Winning the competition IS the publication. A cognitive architecture that achieves competitive or state-of-the-art performance on ARC-AGI benchmarks, with progressive ablation data showing which principles contribute and how they interact, is a stronger paper than any ablation study conducted in isolation. If we win, the paper writes itself. If we don't win but demonstrate clear architectural advantages, the ablation data is independently publishable.

This is NOT a behavioral experiment on existing LLMs (Studies 1–5). This is NOT a 33-month research study building toward eventual evaluation. This is an aggressive, competition-optimized build that moves fast, measures everything, and iterates based on what wins.

**The fundamental question:** Can a neural architecture implementing validated human cognitive principles — at 7–50M parameters — outperform systems with 100–1000× more parameters on tasks requiring genuine abstraction, and specifically can it solve interactive reasoning environments that frontier AI currently scores 0% on?

**Why this matters:** The AI research community is undergoing a paradigm shift from scaling maximalism toward architectural innovation. Reuters reported in 2024 that AI companies are no longer seeing exponential capability growth from scaling alone. NeurIPS 2025 runner-up research showed RLVR "doesn't expand reasoning capabilities beyond what's already present in base models." François Chollet, creator of ARC-AGI, explicitly states that "log-linear scaling is insufficient" and "new ideas are still needed, like how to separate knowledge and reasoning." GPT-5.2 achieves 90.5% on ARC-AGI-1 but struggles on ARC-AGI-2 (~52.9%), and the entire field scores 0% on ARC-AGI-3's interactive environments. Scale alone cannot solve abstract reasoning — architectural principles can.

**Why ARC-AGI-3 is the right target:** ARC-AGI-3 replaces static grid puzzles with interactive video-game-like environments testing exploration, planning, memory, goal acquisition, and alignment across 1,000+ levels in 150+ hand-crafted environments. Frontier models scored 0% at the July 2025 preview. The best preview agent (StochasticGoose) scored 12.58%. This is a reset of the entire competitive landscape — nobody has cracked interactive abstract reasoning. Our architecture's cognitive principles (dual-process reasoning for perception vs. planning, episodic memory for tracking interactions, working memory for holding environment state, metacognitive monitoring for strategy switching, global workspace for coordinating all modules) map directly to the five capabilities ARC-AGI-3 explicitly tests. We are not retrofitting an architecture to fit a benchmark — the benchmark demands exactly what we are building.

**Why these specific principles:** The eight cognitive principles implemented in this study are not arbitrary selections from neuroscience textbooks. Independent analysis of two completely different sensory modalities — biological vision and biological auditory processing — reveals that evolution converged on the same core computational architecture in both systems: hierarchical abstraction with progressive transformation, parallel processing from the earliest stages, feedback-dominant connectivity, predictive coding as an operating principle, sparse and efficient coding, nonlinear compression, population coding with heterogeneity, and massive plasticity governed by critical periods (documented in our foundational research: "Biological Vision Architecture" and "Human Auditory Processing: A Comprehensive Neuroscience Reference"). This cross-modal convergence is strong evidence that these are domain-general computational principles — optimal solutions to information processing under metabolic constraints — not modality-specific features. The unifying framework is Barlow's efficient coding hypothesis extended by Bayesian inference: maximize mutual information between input and neural representation while minimizing metabolic expenditure (Laughlin, 1981; Chalk et al., 2018). If evolution independently arrived at the same architecture for processing light and sound — two physically different signal types with different statistical structures — these principles should transfer to a third domain: abstract reasoning in interactive environments.

**Why the competition frame matters for science:** Publication-optimized research builds conservatively, documents everything, and guarantees a paper regardless of competitiveness. Competition-optimized research makes aggressive architectural choices, iterates fast on what works, and treats ablation data as a byproduct of optimization rather than its goal. These frames are not in tension — progressive ablation (add one principle at a time, measure contribution) is good development strategy for competition too, because it tells you which principles pull their weight and which are dead weight to drop or redesign. The difference is mindset: a publication frame treats a non-contributing principle as a "null result" to report; a competition frame treats it as a module to rip out and replace with something that works.

---

## 2. The Competitive Landscape

We are entering a well-developed competition ecosystem. Understanding what exists — what works, what fails, and why — is essential to building something that wins. This section reflects deep competitive intelligence research conducted in February 2026.

### 2.1 ARC-AGI Benchmark Evolution

**ARC-AGI-1** (Chollet, 2019; arXiv:1911.01547): Approaching saturation for engineered systems. GPT-5.2 achieved 90.5% (December 2025). TRM achieves 45% at 7M parameters. Contamination concerns are emerging. Remains valuable as a calibration benchmark — our architecture must beat TRM's 45% here to validate that cognitive principles add value beyond recursive refinement alone.

**ARC-AGI-2** (arXiv:2505.11831, May 2025): Significantly harder tasks requiring in-context symbol definition and multi-step compositional reasoning. Pure LLMs score 0%. Best verified score: 54% by Poetiq + Gemini 3 Pro at $30.57/task versus human ~60% at ~$17/task. Under ARC Prize 2025 contest compute constraints ($0.20/task on Kaggle 4× L4 GPUs), NVARC achieved 24.03%. The $700K grand prize for >85% remains unclaimed and continues alongside ARC-AGI-3.

**ARC-AGI-3** (preview July 2025, full launch March 25, 2026): **Our primary competition target.** Interactive video-game-like environments replacing static grid puzzles. Key details:

- **Format:** 1,000+ levels across 150+ hand-crafted environments. Each environment is a novel interactive world with rules the agent must discover through exploration.
- **Five tested capabilities:** Exploration (systematic discovery of environment rules), Planning (multi-step goal decomposition), Memory (tracking state across interactions), Goal Acquisition (understanding objectives from context), and Alignment (adapting behavior to match intended goals).
- **Scoring:** Action efficiency relative to human baselines. 200+ human participants establish reference performance. Systems are scored on how efficiently they achieve goals compared to humans — not just whether they succeed, but how many actions they waste.
- **Current performance ceiling:** Frontier AI scored 0% at the July 2025 preview. Best preview agent (StochasticGoose): 12.58%. Humans scored near 100%.
- **Developer Toolkit:** Released January 29, 2026. Python package with local execution engine running at 2,000 FPS, agent templates, hosted API. Three public environments available: LS20 (map navigation), VC33 (object manipulation), FT09 (pattern matching).
- **Official resources:** github.com/arcprize/ARC-AGI-3-Agents (agent templates and examples).
- **Competition rules (compute budget, hardware constraints, prizes) have NOT yet been announced.** ARC Prize confirmed continuing ARC-AGI-2 Grand Prize ($700K for 85%) alongside ARC-AGI-3.

**Why ARC-AGI-3 is our strategic opportunity:** Every other benchmark has established leaders with years of iteration. ARC-AGI-3 resets the playing field entirely. The interactive format demands exactly the capabilities our cognitive architecture provides: exploration requires dual-process reasoning (hypothesize with System 2, test with System 1), planning requires working memory and metacognitive monitoring, memory is explicitly tested, and goal acquisition requires the kind of flexible in-context learning that architectural principles — not scale — provide.

### 2.2 Every Competitive Approach Dissected

#### 2.2.1 Tiny Recursive Model (TRM) — 1st Place Paper Award, ARC Prize 2025

**Our primary calibration benchmark.** arXiv:2510.04871. Code: github.com/SamsungSAILMontreal/TinyRecursiveModels. Pre-trained checkpoints: HuggingFace arcprize/trm_arc_prize_verification.

**Performance:** 7M parameters. 40% on ARC-AGI-1 pass@1. 8% on ARC-AGI-2.

**Architecture in detail (CORRECTED per ADR-002 — TRM codebase analysis):**
- Single 2-layer Transformer with recursive refinement: 16 outer supervision steps, T=3 inner cycles per step, n=6 latent reasoning steps per cycle
- Three states maintained: input embedding x (fixed), answer state y (progressively refined), and latent states z_H and z_L (implicit chain of thought, never directly supervised)
- **Same-resolution latent states:** z_H and z_L at the SAME resolution, updated by a single weight-shared L_level module (NOT multi-scale/downsampled as paper language suggests — ADR-002)
- 512 dimensions per token, **post-norm** (add then normalize — NOT pre-norm), functional RMSNorm (no learnable parameters), no bias terms, Rotary Positional Encoding
- **Final-cycle-only gradients:** Cycles 0 to T-2 run under torch.no_grad(). Backpropagation only through the final cycle. Latent states detached between supervision steps. (NOT deep supervision at every step — ADR-002)
- **Learned initial states** (H_init, L_init — not zeros), **input injection** (input embeddings added to z_H at every z_L update), **stablemax CE loss**
- 12-token vocabulary: PAD=0, EOS=1, colors 2-11 (NOT 11 tokens)
- **Puzzle embeddings:** Each puzzle × augmentation gets a unique [1, 512] learned vector, zero-initialized, trained with SignSGD at 100× model LR. These are pre-trained lookup vectors — NOT test-time training. Roye-Azar ablation (arXiv 2512.11847): replacing correct puzzle ID with blank/random yields ZERO accuracy.
- **AdamATan2 optimizer** (atan2 instead of sqrt for 2nd moment), LR=1e-4, constant after warmup, weight decay=0.1, beta2=0.95
- Training: ~1000× data augmentation (8 geometric transforms × color permutations + translational augmentation), ~1M effective samples from 400 base tasks, ~750K optimizer steps
- **Zero pretraining required** — randomly initialized, trained only on ARC data

**Critical context:** The author explicitly disclaims biological inspiration: "ultimately has nothing to do with the human brain." TRM uses exactly one architectural mechanism — recursive refinement with final-cycle gradient flow — and achieves strong results through that single mechanism implemented well. The paper's language about "deep supervised refinement" is misleading — actual codebase shows loss computed only on final cycle output, not at every step.

**What we learn from TRM (CORRECTED per ADR-002):**
1. Recursive refinement is the single most validated mechanism for ARC. We must incorporate it.
2. Final-cycle-only gradients (not deep supervision) make recursive refinement trainable efficiently.
3. Same-resolution dual latent states provide a form of working memory. Our explicit working memory should improve on this.
4. Puzzle embeddings are a neural hash map over known puzzles — they do NOT generalize to new puzzles.
5. 7M parameters is sufficient. Our cognitive enhancements should improve performance without dramatically increasing parameter count.
6. No pretraining needed — ARC-specific training from random initialization works.

#### 2.2.2 Universal Reasoning Model (URM) — TRM Successor (ADDED in v4.1)

arXiv:2512.14693 (December 2025).

**Performance:** ~14M parameters. **53.8% on ARC-AGI-1 pass@1** — a 13.8pp improvement over TRM's 40%.

**Key innovations over TRM:**

1. **TBPTL (Truncated Backpropagation Through Loops):** With 8 inner-loop iterations, the first 2 run forward-only and the remaining 6 participate in backpropagation with loss at each step. This provides better gradient flow than TRM's final-cycle-only approach without the instability of full BPTT. **Ablation: removing TBPTL drops accuracy from 53.8% to exactly 40.0%** (TRM level). Full BPTT also produces 40.0% — TBPTL's moderate truncation is strictly better than both extremes.

2. **ConvSwiGLU FFN:** Replaces standard FFN with SwiGLU activation (output = xW1 * silu(xW_gate) @ W2) plus depthwise convolutions for local spatial awareness. Roughly doubles parameter count to ~14M. **Ablation: removing ConvSwiGLU drops accuracy from 53.8% to 45.3%.**

**What we learn from URM:**
1. TBPTL is the single highest-impact change available — +13.8pp from gradient strategy alone.
2. ConvSwiGLU adds spatial awareness through convolutions, important for grid-based tasks.
3. Both innovations are adopted for our Phase 0.5 before adding cognitive modules, ensuring we build on the strongest available baseline.
4. VRAM impact is minimal: full BPTT (16 cycles) uses ~1.7GB, TBPTL ~1.3GB — both fit easily on RTX 5070 Ti (16GB).

#### 2.2.3 NVARC — 1st Place Score, ARC Prize 2025 Kaggle Competition

Code: github.com/1ytic/NVARC.

**Performance:** 24.03% on ARC-AGI-2 private evaluation at $0.20/task.

**Architecture and pipeline:**
- Qwen-2-VL 4B parameters as base model, with vocabulary aggressively reduced to just 16 tokens (digits 0–9 plus formatting characters)
- Synthetic data pipeline: ~700 human-described puzzle descriptions expanded to 260K puzzle descriptions via Claude/GPT-4o decomposition, then code generation with consensus verification — at least 8 out of 20 independent implementations must produce identical outputs to be included
- Filtered to ~100K verified high-quality puzzles
- Augmentation: 8 geometric transforms × 10! color permutations = 3.2M effective training puzzles
- Per-task test-time training with LoRA adapters

**What we learn from NVARC:**
1. Synthetic data quality is the hidden multiplier. Consensus verification (8/20 agreement) is a powerful quality filter.
2. The 4B parameter model only wins through massive data engineering, not architectural innovation. At our parameter budget, we need architectural innovation instead.
3. Vocabulary reduction is clever — forces the model to focus on spatial reasoning rather than language modeling.
4. Their augmentation pipeline (geometric × color permutations) is standard and we should adopt it.

#### 2.2.4 MindsAI — Highest Ever ARC-AGI-1 Score

**Performance:** 55.5% on ARC-AGI-1 (2024, highest ever). 15.4% on ARC-AGI-2 (2025).

**Key innovations:**
- T5-series encoder-decoder architecture
- Test-Time Fine-Tuning (TTFT): fine-tune the entire model on each task's demonstration pairs at test time. This alone provides a ~300% performance boost.
- AIRV (Augment, Inference, Reverse-Augmentation, Vote): generate 96 predictions per problem under different geometric augmentations, then reverse the augmentations and apply majority voting. Provides ~260% additional boost.
- 2025 additions: tokenizer dropout and augmentation ensembles for further robustness.

**What we learn:** Test-time adaptation is the single most important technique in ARC competition. MindsAI's TTFT + AIRV demonstrates that the combination of per-task adaptation and augmented verification is more valuable than any architectural innovation they make to the base model.

#### 2.2.5 CompressARC — 3rd Place Paper Award, ARC Prize 2025

arXiv:2512.06104. Code: github.com/iliao2345/CompressARC.

**Performance:** 76K parameters. 20% on ARC-AGI-1 (34% with extended time). ~4% on ARC-AGI-2.

**Architecture:**
- Minimum Description Length (MDL) principle: find the shortest neural program that outputs the entire puzzle (all demonstration pairs plus test outputs)
- 76K-parameter neural network decoder taking latent z ~ N(0,I) as input
- No encoder — encoding is performed via gradient descent, optimizing the decoder weights at inference time to minimize description length
- "Multitensors": buckets of tensors with different dimension subsets for capturing combinatorial structure
- No pretraining, no dataset, no search — just 2,000 gradient steps per puzzle (~20 minutes on RTX 4070)

**What we learn:** Extreme compression and information-theoretic principles produce meaningful abstract reasoning at tiny parameter counts. CompressARC proves that the right inductive bias matters more than scale by a factor of 10,000×. The MDL principle connects to our efficiency framing — finding the simplest explanation that accounts for the data.

#### 2.2.6 Poetiq — Highest ARC-AGI-2 Score (Unrestricted Compute)

Code: github.com/poetiq-ai/poetiq-arc-agi-solver.

**Performance:** 54% on ARC-AGI-2 using Gemini 3 Pro at $30.57/task.

**Approach:**
- Pure inference-time orchestration — no training at all
- Iterative refinement loop: generate solution → verify against demonstrations → analyze feedback on failures → refine solution → repeat
- LLM-agnostic (tested with Gemini, GPT-5.1, Claude Opus 4.5, Grok 4)
- Baseline Gemini 3 Pro without Poetiq: 31% at $0.81/task. With Poetiq harness: 54% at 38× the cost.

**What we learn:** Refinement loops are universally effective — even pure orchestration over frozen LLMs extracts massive performance gains. The 38× cost increase for 23% absolute improvement shows diminishing returns on compute-heavy approaches. Under competition compute constraints ($0.20/task), Poetiq's approach is not viable — reinforcing that architectural efficiency wins competitions.

#### 2.2.7 SOAR — 2nd Place Paper Award, ARC Prize 2025

arXiv:2506.xxxxx. Code: github.com/flowersteam/SOAR.

**Performance:** 52% on ARC-AGI-1.

**Approach:**
- Self-improving evolutionary program synthesis
- Virtuous cycle: LLM generates Python programs → tests against demonstrations → refines failures → collects both successes AND failures → relabels failed programs as correct for whatever output they accidentally produce → fine-tunes LLM on expanded solution set → repeats
- After iterations, 7B models match 123B models from before SOAR training
- Released 5M ARC solutions publicly

**What we learn:** Self-improvement through evolutionary search over programs is viable. The relabeling trick (failed programs become training data for whatever they accidentally solve) is clever data augmentation. SOAR operates at program synthesis level, complementary to our neural architecture approach.

#### 2.2.8 Other Critical Competitors

**Jeremy Berman:** 79.6% on ARC-AGI-1, 29.4% on ARC-AGI-2 with Grok 4. Uses evolutionary test-time compute — evolving natural language instructions rather than code. Performance scales with test-time compute budget.

**Ryan Greenblatt:** 50% on ARC-AGI-1 with GPT-4o. Generates ~8,000 Python programs per task, selects by execution verification. Performance scales log-linearly with sample count.

**GPT-5.2 (December 2025):** 90.5% on ARC-AGI-1, ~52.9% on ARC-AGI-2. First model to exceed 90% on ARC-AGI-1. Demonstrates that frontier models can partially solve ARC through scale, but still fail on the hardest tasks and score 0% on ARC-AGI-3.

**MIRAGE (arXiv:2507.18868):** 1.19M parameters. >99% on ALL SCAN compositional generalization splits. Neuroscience-inspired dual-process reasoning with a frozen meta-trained transformer (System 1) + symbolic schema engine modeling hippocampus-PFC loop (System 2). Our closest methodological relative — validates that multi-principle cognitive integration produces exceptional compositional generalization. Not evaluated on ARC-AGI.

### 2.3 The Test-Time Training Paradigm

Test-time training (TTT) / test-time fine-tuning (TTFT) is the single most important technique separating competitive from non-competitive approaches. Every winner uses it. Static frozen models fail catastrophically on ARC.

**Canonical TTT pipeline (Akyürek et al., arXiv:2411.07279):**

1. **Phase 1 (offline):** Fine-tune base LLM on synthetic ARC tasks (RE-ARC, BARC, custom generators)
2. **Phase 2 (per task):** Construct training data from demonstration pairs using leave-one-out (LOO): for N demonstrations, create N training examples where each example uses N-1 demonstrations as context and the held-out pair as target. Apply geometric augmentations + color permutations → up to 250 training examples per task.
3. **Phase 3 (per task):** Optimize task-specific adapters using "All Outputs" loss — loss computed on test output AND demonstration outputs simultaneously, not just the test prediction. This is critical: it prevents overfitting to the few test examples by also requiring consistency with demonstrations. The adapter mechanism varies by scale: LoRA for large models (4B+), but **full-parameter fine-tuning for small models (7-14M)** where the entire model is LoRA-scale.
4. **Phase 4 (inference):** Augmented inference with hierarchical voting — generate predictions under multiple geometric transformations, vote within augmentation groups (top-3), then vote globally (top-2).

**Critical constraint:** Under competition rules, compute budget is approximately $0.20/task (Kaggle 4× L4 GPUs, 12 hours for ~120 tasks). This eliminates heavyweight reasoning approaches and forces a strategy: move complexity offline into synthetic data and pre-training, run lightweight online TTT per task.

**TTT failure modes to avoid:**
- Overfitting to 2–7 demonstration pairs (addressed by augmentation and All-Outputs loss)
- Augmentation destroying task-relevant information (some color permutations destroy semantics)
- Logarithmic returns on additional compute (diminishing gains beyond optimal TTT steps)
- Distribution shift between offline training data and test tasks
- Shared adapters degrade performance 24% vs. per-task adapters — per-task is mandatory

**For our architecture (CORRECTED per ADR-003):** TTT is not an evaluation methodology — it is a core architectural feature. The recursive refinement core (inspired by TRM) already implements a form of test-time adaptation through its iterative improvement loop. We extend this with per-task full-parameter fine-tuning of the cognitive modules (NOT LoRA — at 7-14M parameters the model IS LoRA-scale), creating a three-level adaptation: fast adaptation through recursive refinement (within inference), medium adaptation through puzzle embeddings (known tasks), and slow adaptation through gradient-based full-parameter fine-tuning (new tasks).

### 2.4 Patterns That Separate Winners from Losers

**What always works:**

1. **Test-time adaptation is non-negotiable.** Every competitive approach uses some form of per-task adaptation. Pure frozen models fail catastrophically — even GPT-5.2 at >1T parameters scores near 0% on ARC-AGI-2 without task-specific adaptation.

2. **Induction + transduction ensembling.** Analysis of ARC Prize 2025 results shows: 26 tasks solved ONLY by inductive approaches (program synthesis), 35 tasks solved ONLY by transductive approaches (direct pattern prediction), 19 tasks solved by both. Ensembling both approaches nears human-level coverage. This maps directly to our dual-process architecture: System 2 (induction/program synthesis) + System 1 (transduction/pattern prediction).

3. **Recursive refinement dominates.** TRM's 16-step improvement, Poetiq's generate-verify-refine, Berman's evolutionary refinement, MindsAI's AIRV — every winning approach iteratively improves predictions. Single-shot prediction is not competitive.

4. **Synthetic data quality is the hidden multiplier.** NVARC's 100K verified puzzles + 3.2M augmentations, SOAR's 5M self-generated solutions, RE-ARC's unlimited procedurally generated training examples. The quality gate (consensus verification, execution verification) matters more than quantity.

**What always fails:**

- Pure LLM prompting without adaptation: near-zero on ARC-AGI-2
- Brute-force DSL search alone: ceiling ~20%
- Neural embedding / vector arithmetic approaches: only 2%
- DreamCoder-style program induction: peak 4.5%
- Static pre-trained models without test-time adaptation: universal failure
- Pure scaling without architectural innovation: only log-linear improvements at enormous cost

### 2.5 The Efficiency Frontier: Small Models Can Absolutely Win

The relationship between model size and ARC performance is NOT linear. The data:

| System | Parameters | ARC-AGI-1 | ARC-AGI-2 |
|---|---|---|---|
| CompressARC | 76K | 20% | ~4% |
| TRM | 7M | 45% | 8% |
| MIRAGE | 1.19M | (SCAN >99%) | — |
| NVARC | 4B | — | 24% |
| GPT-4 (no adaptation) | >1T | ~5% | ~0% |
| GPT-5.2 | >1T | 90.5% | ~52.9% |
| o3-medium | — | — | <3% |

**The formula is (architecture × test-time adaptation × data quality) → performance, NOT (model size) → performance.**

For a 5M–50M parameter budget, the evidence strongly supports competitiveness:
- TRM proves 7M parameters with the right architecture beats 1T+ parameter models that lack adaptation
- CompressARC proves 76K parameters with the right inductive bias (MDL) beats billion-parameter models
- Under Kaggle's compute constraints, smaller models are advantageous — they can run more TTT iterations and augmentation passes within the time budget

For ARC-AGI-3's interactive format specifically: the 2,000 FPS local execution engine shifts the bottleneck from grid prediction accuracy to real-time decision-making speed. A compact model (7–50M parameters) running at high FPS with working memory and planning capabilities may have a structural advantage over large models requiring seconds per inference.

### 2.6 What François Chollet Says Is Missing

Chollet left Google in November 2024 and co-founded Ndea with Zapier's Mike Knoop, focusing on deep learning-guided program synthesis. His position, articulated at the June 2025 AI Startup School:

> "When faced with a new task, your programmer-like meta-learner will synthesize on the fly a program or model that is adapted to the task. And this program will blend deep learning sub-modules for System 1 sub-problems, like perception, and algorithmic modules for System 2 sub-problems."

His analysis of o3: acknowledged it as "a form of deep learning-guided program search" but identified two critical limitations — programs are natural language (not executable or verifiable), and the system cannot autonomously improve its program generation ability.

Core thesis: **Current AI reasoning is tied to model knowledge, but human reasoning capability is not bound to knowledge.** The gap is the ability to separate knowledge from reasoning — to reason about genuinely novel problems using only core cognitive priors.

The ARC Prize 2025 Technical Report (arXiv:2601.10904, January 2026) identifies "refinement loops" as the dominant 2025 paradigm but notes these remain domain-specific. Chollet emphasizes: the accuracy gap is bottlenecked by engineering; the **efficiency gap is bottlenecked by science and new ideas.** ARC-AGI-3's interactive format targets capabilities that are "fundamentally missing" from current systems.

**For our architecture:** Chollet's analysis validates the bio-inspired approach. Global workspace coordinating specialized processors, dual-process reasoning, structured memory systems are the architectural features he believes will ultimately solve ARC. His emphasis on "skill-acquisition efficiency" — intelligence as the ability to learn new skills quickly from minimal examples — aligns perfectly with our test-time adaptive architecture in the 5M–50M parameter range.

### 2.7 Our Differentiation — Why Cognitive Architecture Can Win

**No existing approach systematically implements the full set of validated cognitive principles.** Across 90+ papers submitted to ARC Prize competitions, no submission has integrated more than 4–5 cognitive principles, and no submission has used progressive ablation to quantify individual principle contributions.

| System | Cognitive Principles | Progressive Ablation? | ARC-AGI-3 Ready? |
|---|---|---|---|
| TRM | 1 (recursive refinement) | N/A | No (static grid only) |
| NVARC | 0 (engineering pipeline) | N/A | No |
| MindsAI | 1 (test-time adaptation) | No | No |
| CompressARC | 1 (MDL compression) | N/A | No |
| MIRAGE | 4-5 (dual-process, memory, schemas) | No | No |
| Poetiq | 0 (orchestration only) | N/A | Potentially |
| **This study** | **8 + 3 vision principles** | **Yes** | **Yes — architecture designed for interactive environments** |

**Our three strongest competitive advantages:**

1. **ARC-AGI-3 architectural alignment.** Our cognitive principles map 1:1 to ARC-AGI-3's tested capabilities. No competitor has this alignment because no competitor built their architecture around cognitive science — they built around ARC-AGI-1/2's static grid format.

2. **Dual-process reasoning as architectural feature, not emergent property.** Every winning approach implements some form of induction/transduction duality through engineering (separate pipelines, ensembling). We implement it as a first-class architectural component with learned routing, producing cleaner integration and the ability to dynamically allocate compute between System 1 and System 2 based on task demands.

3. **Principled cognitive motivation with ablation evidence.** We can answer WHY each component helps, not just THAT it helps. This is the paper that TRM and NVARC cannot write — they win through engineering excellence but cannot explain which architectural principles drive their success.

### 2.8 The HRM Cautionary Tale

**The single most important lesson from the literature.** HRM (Wang et al., arXiv:2506.21734) built a 27M-parameter model with two coupled recurrent modules — a slow abstract planner and a fast detailed worker — inspired by hierarchical cortical processing. It achieved 40.3% on ARC-AGI-1 and the authors claimed the hierarchical brain-inspired architecture was central to performance.

Independent analysis by the ARC Prize Foundation and a separate paper (arXiv:2510.00355, "Hierarchical Reasoning Models: Perspectives and Misconceptions") demonstrated that **an 8-layer single-module network performs comparably.** The hierarchical structure contributed minimally — deep supervision and test-time training, not the brain-inspired architecture, drove performance. HRM's headline cognitive claim was functionally debunked.

**Binding methodological commitment:** No cognitive principle will be claimed as contributing to performance unless both forward and reverse ablation demonstrate statistically significant improvement with that principle present. Every claim in our results must be backed by ablation evidence, not architectural narrative. If a principle doesn't help, we report that honestly — or we redesign it until it does.

---

## 3. Research Questions and Hypotheses

### 3.1 Primary Research Question

**Can a 7–50M parameter neural architecture implementing validated human cognitive principles win ARC-AGI-3 and outperform architecturally equivalent baselines on abstract reasoning benchmarks?**

"Architecturally equivalent" means: same parameter count, same training data, same training compute budget. The only variable is the presence or absence of cognitively-inspired architectural components.

### 3.2 Hypotheses

**H1 (Cumulative Architecture):** A neural architecture implementing all eight cognitive principles will outperform both (a) a standard transformer of equivalent parameter count and (b) TRM at 7M parameters on ARC-AGI-1, with performance advantage increasing as more principles are integrated.

**H2 (Perceptual Advantage):** Incorporating biological vision principles (hierarchical feature extraction, predictive coding, foveated attention) into the perceptual front-end will improve performance on spatial reasoning tasks compared to flat grid encoding.

**H3 (Dual-Process Dominance):** The dual-process module (System 1 transduction + System 2 induction with learned routing) will be the single largest contributor to ARC performance, consistent with the competitive landscape showing induction/transduction ensembling as the most validated pattern.

**H4 (Working Memory Bottleneck):** Constraining information flow through a slot-based working memory (4–7 slots) will improve systematic generalization by forcing the model to form abstract representations rather than memorizing surface patterns. Effect will be strongest on ARC-AGI-2 tasks requiring multi-rule composition.

**H5 (Metacognitive Advantage):** A model with explicit confidence estimation and recursive refinement (building on TRM's proven mechanism) will achieve better calibration and adaptive compute allocation than equivalent models without metacognitive monitoring.

**H6 (Structural Transfer):** Compositional inductive biases (structural alignment, relational binding) will produce superior systematic generalization on novel compositions, measurable on both SCAN and ARC spatial composition tasks.

**H7 (Efficiency Thesis):** The full cognitive architecture will achieve equivalent performance to a standard transformer using ≤50% of the training data, demonstrating superior sample efficiency consistent with Chollet's "skill-acquisition efficiency" thesis.

**H8 (Synergy):** The combined effect of all cognitive principles will exceed the sum of their individual effects. Theoretical basis: Chalk et al. (2018) proved that efficient coding, predictive coding, and sparse coding emerge jointly from a single optimization, predicting superadditive interaction between P1/P7/P8. More broadly, the cognitive principles implement facets of the same underlying computation (efficient information processing under resource constraints), so they should interact constructively.

**H9 (Interactive Reasoning — ARC-AGI-3 Specific):** The cognitive architecture will achieve non-zero performance on ARC-AGI-3 interactive environments, with the combination of dual-process reasoning, episodic memory, and metacognitive monitoring producing measurably more efficient exploration strategies (fewer wasted actions) than ablated variants lacking these components. This is the competition hypothesis — it is tested by submitting to ARC-AGI-3.

**H10 (Test-Time Adaptation Amplification):** Cognitive architectural features will amplify the effectiveness of test-time training — per-task full-parameter adaptation will yield larger performance gains on the cognitive architecture than on the baseline transformer, because cognitive modules provide better-structured representations for TTT to fine-tune.

### 3.3 What Would Falsify These Hypotheses

- **H1 falsified if:** Full architecture does not statistically outperform the matched-parameter baseline on any benchmark after fair training, OR does not exceed TRM's 40% on ARC-AGI-1.
- **H2 falsified if:** Biological vision front-end performs equivalently or worse than standard grid encoding on spatial reasoning tasks.
- **H3 falsified if:** Dual-process module is not the top-contributing principle in ablation analysis.
- **H4 falsified if:** Working memory bottleneck reduces performance rather than improving generalization.
- **H5 falsified if:** Metacognitive monitoring does not improve calibration (ECE equivalent to baseline).
- **H6 falsified if:** Structural transfer mechanisms do not improve compositional generalization on ARC spatial composition tasks.
- **H7 falsified if:** Architecture requires equal or more training data to achieve baseline performance.
- **H8 falsified if:** Combined effect ≤ sum of individual effects (no emergent synergy).
- **H9 falsified if:** Architecture achieves 0% on ARC-AGI-3, or cognitive ablated variants perform equivalently (no advantage from cognitive principles in interactive settings).
- **H10 falsified if:** TTT provides equal or greater performance gains on the baseline than on the cognitive architecture.

---

## 4. Architecture Specification

### 4.1 Design Philosophy: TRM-Informed Cognitive Core

The architecture is built around a **TRM-informed recursive transformer core** enhanced with cognitive principles as first-class architectural features. We do not fork TRM's code (this would create IP complications and not demonstrate novelty), but we study its design choices deeply and build principled improvements on top of its proven foundation.

**Core design decisions derived from competitive intelligence:**

1. **Recursive refinement is the backbone.** TRM's 16-step iterative improvement is the single most effective mechanism for ARC. Our architecture implements recursive refinement as the core computation loop, with cognitive modules enhancing what happens within each refinement step.

2. **Deep supervised refinement.** Loss at every output prediction step, not just the final one. This is what makes recursive refinement trainable and is non-negotiable.

3. **Test-time adaptation is architectural, not optional.** Per-task puzzle embeddings (Phase 0) and full-parameter fine-tuning (Phase 0.5+) are baked into the inference pipeline from day one. NO LoRA — at our parameter scale the model IS LoRA-scale.

4. **Target parameter range: 7–15M for ARC-AGI-1/2, expandable to 50M for ARC-AGI-3.** TRM proves 7M is sufficient for 40% (URM reaches 53.8% at 14M) on ARC-AGI-1. Our cognitive enhancements should improve performance at similar or modestly higher parameter counts. ARC-AGI-3's interactive format may benefit from additional capacity for exploration and memory modules.

5. **Induction + transduction ensembling.** System 2 (inductive program synthesis) and System 1 (transductive pattern prediction) run on every task, outputs ensembled through augmentation-consistency voting.

### 4.2 The Eight Cognitive Principles — Competition Priority Order

Principles are ordered by competitive impact based on our competitive landscape analysis. This ordering determines the progressive build sequence.

| Priority | # | Cognitive Principle | Competition Value | Architectural Component |
|---|---|---|---|---|
| **CRITICAL** | P5 | Dual-Process Reasoning | Most validated pattern in ARC research; induction/transduction duality IS dual-process | System 1 (fast feedforward transduction) + System 2 (slow recurrent induction) with learned routing |
| **HIGH** | P3 | Multiple Memory Types | Explicitly required by ARC-AGI-3; episodic memory for interaction tracking | Episodic + semantic + working memory with distinct read/write dynamics |
| **HIGH** | P7 | Constrained Attention (Working Memory) | Directly determines performance on hard compositional tasks; TRM's same-resolution dual latent states are a partial implementation | Slot-based information bottleneck (4–7 slots) with heterogeneous capacity |
| **HIGH** | P8 | Metacognitive Monitoring | Already the pattern behind every winning refinement loop; enables adaptive compute | Confidence estimation + recursive halt/continue + TRM-style iterative refinement |
| **STRATEGIC** | P1 | Global Workspace Theory | Untested in ARC but provides principled coordination mechanism; natural fit for ARC-AGI-3 multi-module coordination | Modular architecture with shared workspace + learned gating |
| **MODERATE** | P6 | Structural Transfer | ARC premise is testing Core Knowledge Prior transfer; should be structural priors, not learned | Compositional binding + relational attention + geometric equivariances |
| **MODERATE** | P2 | Staged Learning | Improves training efficiency; SOAR validates curriculum approach | Progressive layer freezing + curriculum from simple to complex |
| **LOWEST** | P4 | Emotional Valuation | Weak for static ARC; moderate for ARC-AGI-3 exploration/exploitation tradeoffs | Learned resource allocation signals (urgency/novelty/confidence) |

**Key precedents for each principle:**

| # | Key Precedent | Citation |
|---|---|---|
| P5 | MIRAGE dual-process for SCAN >99%; VSA dual-process for ARC-style tasks; induction+transduction ensembling (arXiv:2411.02272) | arXiv:2507.18868 |
| P3 | ArcMemo lifelong LLM memory for ARC (Runner-up Paper 2025); Titans surprise-driven writes; MIRAGE episodic buffer | arXiv:2504.20109 |
| P7 | Locatello et al. Slot Attention (NeurIPS 2020); TRM same-resolution dual latent states; Slot Mixture Module (ICLR 2024) | arXiv:2006.15055 |
| P8 | TRM iterative refinement; URM TBPTL; MIND introspection network (ICLR 2025 Oral); Graves ACT; PonderNet | arXiv:2510.04871, arXiv:2512.14693 |
| P1 | Goyal & Bengio global workspace (ICLR 2022); Bengio "The Consciousness Prior" | arXiv:2103.01197 |
| P6 | MIRAGE schema engine; ARC Core Knowledge Priors (objectness, topology, counting, geometry) | arXiv:1911.01547 |
| P2 | SOAR alternating search/learning phases; critical period biology (Hensch, 2005) | arXiv:2506.xxxxx |
| P4 | Nature Communications 2025 separable affective prediction errors; metabolic efficiency (Levy & Calvert, 2021; Lennie, 2003) | — |

#### 4.2.1 Deepened Theoretical Rationale from Cross-Modal Neuroscience

Independent neuroscience research into the human auditory system (documented in "Why the Auditory System Is Built the Way It Is: Efficient Coding, Bayesian Inference, and Metabolic Constraints") reveals that several of our cognitive principles are not merely "brain-inspired design choices" but are information-theoretically optimal or metabolically necessary solutions to specific computational problems. This provides a stronger theoretical foundation than biological analogy alone.

**P1 (Global Workspace) — Active Gating, Not Passive Bottleneck.** The auditory thalamus (medial geniculate body) implements a push-pull gating mechanism: cortex simultaneously excites thalamic relay neurons directly and inhibits them via the thalamic reticular nucleus (TRN), creating attention-modulated filtering before information reaches cortex (Sherman & Guillery, 2002). Our global workspace includes a learned gating component that actively filters what gets broadcast based on current processing context — an attention gate on workspace access, not just a capacity limit. This is architecturally distinct from the capacity constraint of slot attention (P7): P1 gating determines relevance, P7 determines capacity.

**P2 (Staged Learning) — Stability-Plasticity Dilemma.** Grossberg's (1980) stability-plasticity dilemma proves that too much plasticity causes catastrophic forgetting while too much stability causes entrenchment. The biological solution is a high-plasticity exploration phase (critical period) followed by progressive restriction. The molecular mechanism — PV+ interneuron maturation triggering perineuronal net formation (Hensch, 2005; Pizzorusso et al., 2002) — maps to progressive layer freezing from lower to higher layers, matching the biological pattern where primary sensory areas stabilize before association areas.

**P4 (Emotional Valuation) — Metabolic Efficiency Rationale.** Communication between neural populations costs approximately 35 times more energy than computation within a population (Levy & Calvert, 2021), and each action potential costs ~2.4 × 10⁹ ATP molecules (Lennie, 2003). Emotional valuation signals implement a learned routing cost function that directs computational resources toward high-value stimuli. This reframes P4 from "mimicking the amygdala" to "implementing a learned resource allocation mechanism justified by information-theoretic efficiency under fixed compute budgets." For ARC-AGI-3 specifically, this becomes an exploration/exploitation controller — deciding what to investigate, what to ignore, and when to commit to a plan.

**P7 (Working Memory) — Formal Optimality of Heterogeneous Capacity.** Ganguli & Simoncelli (2014) proved that for optimal population encoding, cell density should be proportional to prior probability. This suggests slot attention should support heterogeneous slot resolution — more slots at standard resolution for common patterns, with at least one high-capacity slot for rare but informationally critical features. Additionally, "mutually exclusive biophysical constraints" from auditory neuroscience strengthens the case for modular specialization: certain computational goals (fast pattern matching vs. slow deliberative reasoning vs. stable long-term storage) require incompatible representations that a single monolithic network handles poorly.

### 4.3 Biological Vision Principles for the Perceptual Front-End

The perceptual module derives directly from our foundational biological vision neuroscience research (documented in "Biological Vision Architecture: A First-Principles Neuroscience Foundation for AI Design").

**Principle V1: Predictive Coding as the Core Perceptual Loop**

Each cortical level computes precision-weighted prediction errors: `ξ_l = Π_l · ε_l` where `ε_l` is the raw error (observed minus predicted) and `Π_l` is a learned precision weight reflecting confidence. For ARC grids: as the reasoning module develops a hypothesis about the input→output transformation, it sends a top-down prediction to the perceptual module, which returns precision-weighted error maps highlighting where the prediction fails.

**Principle V2: Hierarchical Sparse Representation**

Progressive abstraction from simple features to complex invariant representations:

| Layer | Biological Analog | Grid Processing | Output |
|---|---|---|---|
| L1 | V1 (edge detection) | Cell-level color boundaries, single-cell features | Edge maps, color region boundaries |
| L2 | V2 (contour integration) | Local patterns: small shapes, lines, blocks | Local structure descriptors |
| L3 | V4/IT (object recognition) | Composite structures: larger objects, repeating patterns, symmetry | Abstract structural descriptions |
| L4 | Decision/Workspace | Global composition: spatial relationships between L3 objects | Workspace-ready representations |

**Principle V3: Foveated Attention with Task-Driven Priority**

Working memory slots serve as "fixation points" — high-resolution processing around each slot's attended region, with coarse monitoring of the rest. Priority map: `P(x,y) = α·S_BU(x,y) + (1-α)·S_TD(x,y)` balancing bottom-up saliency and top-down task relevance.

**Cross-modal validation:** Independent auditory neuroscience research confirms that predictive coding, hierarchical sparse representation, and attention-driven priority allocation also govern biological hearing, validating V1–V3 as domain-general perceptual computation principles.

### 4.4 The Progressive Build Sequence — Competition Priority Order

This is the heart of both the development strategy and the ablation design. We build in order of competitive impact, measuring each principle's contribution against TRM as baseline.

```
Phase 0: BACKBONE + TRM-STYLE RECURSIVE CORE
  └── Small recursive transformer (~7-10M base parameters)
  └── TRM-style recursive refinement (up to 16 steps)
  └── Deep supervised loss at every refinement step
  └── Multi-scale latent states (z_H, z_L)
  └── Standard flat grid encoding
  └── Per-task puzzle embeddings (SignSGD, 100× model LR) — NOT LoRA (ADR-003)
  └── Baseline measurement: TARGET ≥ TRM's 40% minimum, ≥45% target on ARC-AGI-1
  └── This phase must match or approach TRM before we add cognitive modules

Phase 1: DUAL-PROCESS REASONING (P5) — CRITICAL
  └── Add System 1 fast feedforward path (transductive pattern prediction)
  └── Add System 2 slow recurrent path (inductive rule synthesis)
  └── Learned routing between paths based on task difficulty/uncertainty
  └── Ensemble both paths via augmentation-consistency voting
  └── Measure: Does dual-process improve over recursive core alone?
  └── Diagnostic: Does routing correlate with task difficulty?
  └── TRM comparison: Does P5 alone explain TRM's remaining failure modes?

Phase 2: MULTIPLE MEMORY TYPES (P3) — HIGH
  └── Add episodic memory (stores specific input-output examples and interaction history)
  └── Add semantic memory (consolidated abstractions from multiple tasks)
  └── Working memory (capacity-limited) added in Phase 3
  └── Measure: Does memory separation improve few-shot learning?
  └── ARC-AGI-3 specific: Does episodic memory track interaction history effectively?

Phase 3: WORKING MEMORY BOTTLENECK (P7) — HIGH
  └── Add slot-based attention layer (4-7 slots, heterogeneous capacity)
  └── Information must flow through capacity-limited workspace
  └── Tight coupling with perceptual module (slots drive attention, attention feeds slots)
  └── Measure: Does forced abstraction improve generalization?
  └── Diagnostic: Are slots used non-degenerately? Do different tasks use different slot patterns?

Phase 4: METACOGNITIVE MONITORING (P8) — HIGH
  └── Add explicit confidence estimation head
  └── Add learned halt/continue mechanism (adaptive computation time)
  └── Enhances TRM's recursive refinement with principled stopping criterion
  └── Measure: Does calibration improve? Does the model allocate more compute to harder tasks?
  └── TRM comparison: Does explicit metacognition improve on TRM's fixed 16-step refinement?

Phase 5: GLOBAL WORKSPACE (P1) — STRATEGIC
  └── Add shared workspace with learned gating (relevance filtering)
  └── Specialized sub-networks communicate through workspace bottleneck
  └── Measure: Does coordinated module communication improve over ad-hoc connections?
  └── ARC-AGI-3 specific: Does workspace coordination improve multi-module planning?

Phase 6: PERCEPTION (V1-V3) — MODERATE
  └── Replace flat grid encoding with hierarchical perceptual front-end
  └── Add predictive coding loop (top-down predictions, precision-weighted error)
  └── Add foveated processing (attention-driven resolution allocation)
  └── Measure: Does biological perception improve spatial reasoning vs. flat encoding?
  └── Diagnostic: Are prediction errors concentrated at task-relevant features?

Phase 7: STRUCTURAL TRANSFER (P6) — MODERATE
  └── Add relational attention / structural alignment
  └── Embed geometric equivariances (rotation, reflection invariance)
  └── Compositional binding (objects bound to roles, not positions)
  └── Measure: Does compositional generalization improve?

Phase 8: STAGED LEARNING (P2) — MODERATE
  └── Modify training curriculum: simple → complex transformations
  └── Progressive layer freezing (lower layers stabilize first)
  └── Measure: Does staged learning improve final performance and training efficiency?

Phase 9: EMOTIONAL VALUATION (P4) — LOWEST
  └── Add multi-dimensional value signals (urgency, novelty, confidence)
  └── Signals modulate processing priority and resource allocation
  └── ARC-AGI-3: exploration/exploitation controller
  └── Measure: Does valuation improve on ambiguous tasks or exploration efficiency?

Phase 10: FULL INTEGRATION + ARC-AGI-3 AGENT
  └── All principles active simultaneously
  └── ARC-AGI-3 agent wrapper (see Section 4.6)
  └── Full benchmark evaluation across ARC-AGI-1/2/3
  └── Compare to sum-of-parts prediction (synergy test)
  └── Competition submission preparation
```

**Critical design choice:** At each phase, we train from scratch for clean attribution. However, in competition mode, if a principle doesn't improve performance within 2–3 training runs, we have three options: (1) debug and fix the implementation, (2) redesign the component based on failure analysis, or (3) skip it and move to the next principle. We do NOT spend weeks on a non-contributing component. The ablation data from the attempt is still valuable for publication.

**Phase 0 gate is the most important:** If our TRM-style baseline cannot match TRM's published 40% on ARC-AGI-1, we have an implementation problem that must be resolved before adding complexity. Every cognitive module is measured against this foundation.

### 4.5 Baseline Architecture

The baseline must be as fair as possible for scientific comparison:

- **Architecture:** TRM-style recursive transformer WITHOUT cognitive modifications (our Phase 0)
- **Parameters:** Matched to full cognitive architecture
- **Training:** Same data, same compute budget, same hyperparameter tuning effort, same random seeds
- **TTT:** Same per-task adaptation pipeline (puzzle embeddings for known tasks, full-parameter fine-tuning for new tasks)
- **Evaluation:** Same benchmarks, same metrics, same number of evaluation runs

Additionally, we compare against published TRM results as an external baseline, and against a standard (non-recursive) transformer of matched parameter count as a lower bound.

### 4.6 ARC-AGI-3 Agent Architecture

ARC-AGI-3 requires an agent that interacts with environments through actions, not a model that predicts output grids from input grids. This requires additional architectural components on top of the cognitive core.

**Environment Interface:**
- Observation encoder: processes current environment state (visual observation) through the perceptual front-end (V1-V3)
- Action decoder: maps from workspace state to action space (environment-specific discrete actions)
- Reward/feedback processor: interprets environment feedback to update internal state

**Exploration Module (built on P5 + P4):**
- System 2 generates hypotheses about environment rules ("if I push this object, it might move right")
- System 1 executes actions to test hypotheses quickly
- P4 (emotional valuation as exploration/exploitation controller) determines whether to explore new hypotheses or exploit known rules
- Scoring metric is action efficiency vs. humans — systematic, hypothesis-driven exploration outperforms random exploration

**Planning System (built on P7 + P8):**
- Working memory holds current goal state, subgoal decomposition, and environment model
- Metacognitive monitoring tracks progress toward goals and triggers replanning when stuck
- Multi-step planning through recursive refinement: plan → execute → observe → refine plan

**Episodic Interaction Memory (built on P3):**
- Records (state, action, outcome) tuples from each interaction step
- Enables the agent to remember what it has tried and what worked
- Supports within-episode learning: agent gets better at each environment as it accumulates experience
- Critical for the "Memory" capability ARC-AGI-3 explicitly tests

**Global Workspace Coordination (P1):**
- Coordinates perception → planning → action selection → memory update cycle
- Learned gating determines which module gets broadcast priority at each step
- Natural fit for ARC-AGI-3's multi-module coordination requirements

**Integration with ARC-AGI-3 Developer Toolkit:**
- Build on official agent templates from github.com/arcprize/ARC-AGI-3-Agents
- Local execution engine provides 2,000 FPS — our compact model (7–50M params) can run many decision cycles per second
- Three public environments (LS20, VC33, FT09) available for development and validation before competition launch

---

## 5. Training Methodology

### 5.1 Data Strategy: Leverage the Existing Ecosystem

We do NOT build our own synthetic data pipeline. The ARC research community has already produced high-quality, verified training data that would take months to replicate. Our competitive advantage is architecture, not data engineering. We use existing data and focus our effort on the architecture.

**Primary training data sources:**

| Source | Size | Quality | Access |
|---|---|---|---|
| **ARC-AGI-1 training set** | 400 tasks | Gold standard (hand-crafted by Chollet) | Public |
| **RE-ARC** (github.com/michaelhodel/re-arc) | Unlimited (procedural generators for all 400 ARC-AGI-1 tasks) | High — 1,000 verified examples per task, unlimited generation | Public |
| **BARC** (github.com/xu3kev/BARC, huggingface.co/barc0) | 400K synthetic tasks | Medium-high — LLM-generated via GPT-4 descriptions, execution-verified | Public |
| **ARC-DSL** (github.com/michaelhodel/arc-dsl) | 160 grid primitives + hand-crafted solvers for all 400 training tasks | Reference implementation | Public |
| **SOAR solutions** (github.com/flowersteam/SOAR) | 5M ARC solutions | Variable — self-generated, not all verified | Public |

**Augmentation strategy (following NVARC's proven approach):**
- 8 geometric transforms: 4 rotations × 2 reflections
- Color permutations: random permutation of color palette per task, with awareness of the "task blurring" problem (some permutations destroy semantics — we follow NVARC's filtering approach)
- Combined: up to 3.2M effective training examples from 400 base tasks

**For ARC-AGI-3:** Use the official Developer Toolkit's environments for training. Three public environments (LS20, VC33, FT09) available for development. Additional environments expected before competition launch.

### 5.2 Test-Time Training — Core Architectural Feature

TTT is not an evaluation methodology bolted on after training. It is a core part of how the architecture operates.

**Two-level adaptation:**

1. **Fast adaptation (within inference):** Recursive refinement through 16+ steps. The model iteratively improves its predictions through the refinement loop. This happens at inference time on every task, with no gradient computation required.

2. **Slow adaptation (per task):** Full-parameter fine-tuning on each task's demonstration pairs (NOT LoRA — at 7-14M params the model IS LoRA-scale).
   - Construct training data using leave-one-out (LOO): for N demonstrations, create N training examples
   - Apply geometric augmentations to expand training set to ~250 examples per task
   - Optimize task-specific model parameters using All-Outputs loss (loss on test output AND demonstration outputs)
   - Full-parameter fine-tuning: per-task adaptation (NOT shared across tasks — 24% degradation with shared adapters)
   - Budget: optimize for competition compute constraints (~$0.20/task)

**For ARC-AGI-3:** TTT extends to within-episode learning. The agent adapts its behavior within each environment based on accumulated (state, action, outcome) experience stored in episodic memory. This is a natural extension of per-task TTT to interactive settings.

### 5.3 Fair Comparison Principles

The study's scientific credibility requires fair comparison:

**Compute-matched training:** Every architecture variant receives the same total training FLOPs. If cognitive additions increase parameter count, they receive fewer training steps to equalize.

**Data-identical training:** All variants train on the same data in the same order (fixed random seeds).

**Hyperparameter fairness:** Each variant receives the same hyperparameter search budget (random search, N=20 trials). No hand-tuning the cognitive architecture more carefully than the baseline.

**Multiple seeds:** Every configuration trained with minimum 5 random seeds. Report mean ± standard deviation. No cherry-picking.

**TTT fairness:** All variants (baseline AND cognitive) receive the same TTT pipeline. The comparison isolates architectural contribution from TTT contribution.

### 5.4 Training Infrastructure

**Hardware (CyberPowerPC Gamer Supreme — SLC8200BSTV11):**

| Component | Specification | Relevance |
|---|---|---|
| **GPU** | NVIDIA GeForce RTX 5070 Ti, 16GB GDDR7 | Primary training accelerator. Supports 7–50M parameter models with batch sizes 16–64. FP16 Tensor Cores double effective memory. |
| **CPU** | Intel Core Ultra 9 285K, 24-core, 3.7/5.7 GHz | Data loading, augmentation pipeline, evaluation. |
| **RAM** | 64GB DDR5 @ 6400MHz (expandable to 192GB) | All datasets fit in memory. |
| **Storage** | 2TB PCIe 4.0 SSD | Fast checkpoints. |
| **OS** | Windows 11 Pro | PyTorch + CUDA. WSL2 available. |
| **Cooling** | Liquid CPU, 1000W PSU | Sustained GPU loads. |

**Framework:** PyTorch with CUDA. Custom architecture implementations. Mixed-precision training (torch.cuda.amp). Flash attention where supported. Gradient accumulation for larger effective batch sizes.

**Free supplementary compute:** Kaggle provides 30 hours/week of P100/T4 GPU time with 9-hour background execution sessions — sufficient for additional training runs, hyperparameter searches, and competition submission testing.

### 5.5 Training Protocol Per Phase

For each architecture phase:

1. **Initialize** model weights randomly (fixed seed per run)
2. **Hyperparameter search:** Random search over learning rate, weight decay, batch size, scheduler (N=20 trials, short runs)
3. **Full training:** Best hyperparameters, 5 random seeds, full training duration
4. **TTT evaluation:** Apply per-task adaptation (puzzle embeddings for known tasks, full-parameter fine-tuning for new tasks) and evaluate with TTT pipeline
5. **TRM comparison:** Compare phase performance against TRM baseline
6. **Gate check:** Does this phase pass its gate criteria? (see Section 9.2)
7. **Competition decision:** Does this phase improve competition performance? If no after debugging, consider skipping.
8. **Logging:** Record all metrics, training curves, hyperparameters, ablation data

### 5.6 Competition Compute Budget Planning

Under anticipated ARC Prize compute constraints (~$0.20/task, Kaggle 4× L4 GPUs, 12 hours for ~120 tasks):

**Budget allocation per task:**
- Model loading + initialization: ~5% of budget
- Per-task TTT (full-parameter fine-tuning): ~40% of budget
- Augmented inference (multiple geometric transforms): ~40% of budget
- Ensembling + verification: ~15% of budget

**Optimization targets:**
- Speculative decoding: 4.7× inference speedup (documented in ARChitects approach)
- Prefix caching: 5.8× speedup in scoring
- Small model advantage: 7–15M parameter model runs many more iterations within budget than 4B model

---

## 6. Evaluation Framework

### 6.1 Primary Target: ARC-AGI-3

ARC-AGI-3 (launching March 25, 2026) is our competition target. Evaluation metrics specific to ARC-AGI-3:

- **Action efficiency:** Primary scoring metric. Ratio of agent's actions to human baseline actions for the same level. Lower is better — measures how efficiently the agent achieves goals compared to humans.
- **Level completion rate:** Percentage of levels successfully completed across all environments.
- **Exploration efficiency:** Actions spent gathering information vs. actions spent executing known solutions. Hypothesis-driven exploration should outperform random exploration.
- **Adaptation speed:** How quickly the agent's performance improves within each environment as it accumulates experience.
- **Cross-environment transfer:** Does performance on later environments benefit from experience with earlier ones?

**Development evaluation:** Three public environments (LS20, VC33, FT09) for development. Full evaluation on 1,000+ levels across 150+ environments at competition submission.

### 6.2 Calibration Benchmarks: ARC-AGI-1 and ARC-AGI-2

These are NOT the competition target but serve essential roles:

**ARC-AGI-1 Public Evaluation (120 tasks):**
- Primary calibration benchmark — our Phase 0 must match TRM's 45%
- Direct comparison to: TRM (45%), HRM (40.3%), CompressARC (20%), MindsAI (55.5%), GPT-5.2 (90.5%)
- Scoring: pass@2 (two attempts per task)
- Every phase measured against TRM here. If adding a cognitive principle doesn't improve over TRM baseline, that principle needs redesign.

**ARC-AGI-2 Public Evaluation (120 tasks):**
- Harder benchmark testing compositional reasoning and in-context symbol definition
- Comparison to: NVARC (24% at $0.20/task), Poetiq (54% at $30.57/task), Berman (29.4%), GPT-5.2 (~52.9%)
- Cost efficiency metric: $/task following ARC Prize reporting standards
- Scoring: pass@2

### 6.3 Secondary Benchmarks

**SCAN (Simplified Commands for Abstract Navigation):** Tests systematic compositional generalization. Validates H6 (Structural Transfer). MIRAGE achieves >99% on all splits with 1.19M parameters, setting the ceiling. Our contribution on SCAN is demonstrating competitive compositional generalization as a byproduct of principled cognitive design, not through SCAN-specific engineering.

**BIG-Bench Hard (selected subtasks):** Logical deduction, boolean expressions, tracking shuffled objects, causal judgment. Provides broader test of reasoning capability beyond grid-based tasks.

### 6.4 Metrics

**Competition metrics (ARC-AGI-3):**
- Action efficiency vs. human baselines (primary)
- Level completion rate
- $/task compute cost

**Performance metrics (ARC-AGI-1/2):**
- Task accuracy (pass@2)
- Systematic generalization score (SCAN)
- Few-shot learning efficiency (learning curve)
- Cost efficiency ($/task)

**Calibration metrics:**
- Expected Calibration Error (ECE)
- Selective prediction accuracy (accuracy on tasks the model chooses to attempt)

**Efficiency metrics:**
- Parameter efficiency: performance / parameter count
- Sample efficiency: training examples needed to reach baseline performance
- Compute efficiency: FLOPs per task at inference
- TTT amplification: performance gain from TTT on cognitive architecture vs. baseline

**Cognitive behavior metrics:**
- Adaptive computation: inference time correlation with task difficulty
- Working memory utilization: slot activation patterns across task types
- Dual-process routing: System 1 vs. System 2 selection frequency and accuracy
- Exploration strategy: hypothesis-driven vs. random action patterns (ARC-AGI-3)

### 6.5 Success Criteria

Competition success criteria are separate from scientific success criteria.

**Competition targets:**

| Benchmark | Target | Justification |
|---|---|---|
| ARC-AGI-1 | >50% | Must beat TRM (45%) to demonstrate cognitive principles add value |
| ARC-AGI-2 | >10% (at $0.20/task) | Must be competitive at contest compute constraints |
| ARC-AGI-3 | Top-3 on leaderboard | Competition target — specific threshold TBD when rules announced |

**Scientific success criteria (pre-registered):**

| Hypothesis | Success Criterion | Justification |
|---|---|---|
| H1 (Cumulative) | Full architecture outperforms Phase 0 by ≥10% absolute on ARC-AGI-1 | Meaningful practical improvement from cognitive principles |
| H2 (Perception) | Biological perception improves spatial ARC tasks by ≥5% over flat encoding | Isolated perceptual contribution |
| H3 (Dual-Process) | P5 is the largest single-principle contributor in ablation ranking | Consistent with competitive landscape evidence |
| H4 (Working Memory) | Constrained attention improves OOD generalization by ≥15% relative | Working memory's contribution to abstraction |
| H5 (Metacognition) | ECE < 5% (vs. typical 15–25% for small models) | Meaningful calibration improvement |
| H6 (Structural Transfer) | ≥90% on SCAN add-jump AND ≥5% ARC spatial composition improvement | Dual criterion prevents SCAN-only optimization |
| H7 (Efficiency) | Match baseline with ≤50% training data | 2× sample efficiency |
| H8 (Synergy) | Full architecture effect > Σ(individual effects) by ≥10% | Detectable emergent synergy |
| H9 (Interactive) | Non-zero ARC-AGI-3 performance; cognitive variants outperform ablated | Competition hypothesis |
| H10 (TTT Amplification) | TTT gains larger on cognitive architecture than baseline | Architectural features enhance adaptation |

---

## 7. Ablation Protocol

### 7.1 Competition Mindset: Build IS the Ablation

The progressive build sequence (Phase 0 → Phase 10) naturally generates ablation data. Each phase measures the marginal contribution of one cognitive principle. This serves dual purposes: it tells us which principles to keep, optimize, or remove for competition performance, AND it produces the ablation tables required for publication.

**The mindset difference from v3.1:** Previously, ablation was the study's goal — we built progressively in order to measure contributions. Now, ablation is the byproduct of competitive development — we build progressively because it's good engineering (tells us what works), and the ablation data is the publication material.

### 7.2 Three Ablation Approaches

**Forward ablation (primary):** The progressive build. Phase 0 → Phase 1 → ... → Phase 10. Measures each principle's marginal contribution given all previously added principles.

**Reverse ablation (validation):** Starting from the full architecture, remove one principle at a time. Measures each principle's contribution to the full system. Differences between forward and reverse attribution indicate interaction effects.

**Pairwise interaction tests (targeted):** For principles with strong theoretical interaction predictions, test 2×2 combinations. Priority pairs:

1. **Dual-Process × Working Memory** (strong positive predicted): Working memory provides the bottleneck that makes System 2 deliberation valuable.
2. **Metacognition × Dual-Process** (strong positive predicted): Metacognition needs meaningful decisions — when to escalate to System 2.
3. **Episodic Memory × Metacognition** (positive predicted for ARC-AGI-3): Memory of past interactions informs metacognitive decisions about exploration strategy.
4. **Global Workspace × Dual-Process** (interaction predicted): Workspace coordination may amplify or modulate dual-process routing.

### 7.3 Competition-Optimized Ablation Process

For each principle added:

1. **Train and measure.** Does it improve ARC-AGI-1 over previous phase? Over TRM?
2. **If yes:** Keep. Move to next phase.
3. **If no (first attempt):** Debug implementation. Check for training instabilities, degenerate solutions, hyperparameter sensitivity.
4. **If no (after debugging):** Redesign component. Try alternative implementation informed by failure analysis.
5. **If no (after redesign):** Document finding. Skip principle for competition build. Include in publication as null result. Move to next principle.
6. **Maximum time per principle:** 1 week. After 1 week without improvement, invoke option 5.

This ensures we never spend excessive time on non-contributing components while still generating honest ablation data.

---

## 8. Comparison Protocol

### 8.1 Direct Comparisons

| System | Parameters | Comparison Purpose | Source |
|---|---|---|---|
| **Phase 0 (TRM-style baseline)** | ~7-10M | Internal fair comparison | Internal |
| **Standard transformer** | ~7-10M | Lower bound (no recursion, no cognitive principles) | Internal |
| **TRM published** | 7M | External calibration — must beat 40% on ARC-AGI-1 | arXiv:2510.04871 |
| **HRM** | 27M | Brain-inspired cautionary comparison | arXiv:2506.21734 |
| **CompressARC** | 76K | Extreme efficiency reference | arXiv:2512.06104 |
| **MIRAGE** | 1.19M | Cognitive integration comparison (on SCAN) | arXiv:2507.18868 |
| **NVARC** | 4B | Competition score benchmark (ARC-AGI-2) | github.com/1ytic/NVARC |
| **MindsAI** | — | Competition score benchmark (ARC-AGI-1) | Published results |
| **Berman** | — | Competition score benchmark (ARC-AGI-2) | Published results |
| **ARC-AGI-3 leaderboard** | Various | Competition standing | ARC Prize 2026 |

### 8.2 Cognitive Integration Depth Comparison

| System | Cognitive Principles | Ablation? | ARC-AGI-3? |
|---|---|---|---|
| TRM | 1 (recursive refinement) | N/A | No |
| CompressARC | 1 (MDL) | N/A | No |
| HRM | 1-2 (debunked hierarchy) | No | No |
| MIRAGE | 4-5 (dual-process, memory, schemas) | No | No |
| **This study** | **8 + 3 vision** | **Yes — forward, reverse, pairwise** | **Yes** |

### 8.3 Scaling Analysis

Evaluate at three parameter counts to test whether cognitive advantages hold across scales:
- **Minimal:** ~7M (TRM-matched)
- **Medium:** ~20M
- **Maximum:** ~50M (VRAM limit on 16GB RTX 5070 Ti)

If cognitive advantage grows with scale → principles become more valuable with capacity. If advantage shrinks → principles primarily help at small scale (still valuable for efficiency argument).

---

## 9. Execution Plan

### 9.1 Development Environment: Claude Code

This study uses Claude Code for implementation — a command-line tool providing persistent file system access, git integration, test execution, and full codebase awareness. This eliminates the manual session-bridging burden of Claude.ai-based development.

**Zero-context-loss strategy:**
- CLAUDE.md file at project root with architecture overview, current phase, known issues, and build decisions
- Makefile with phase-gated test targets (test_phase0, test_phase1, etc.)
- Each phase has explicit pass/fail criteria testable from command line
- Git commits at every phase gate with descriptive messages
- BUILD_STATE.md tracks current phase, blocking issues, and next actions

### 9.2 Phase Gate Criteria

Each phase must satisfy gate criteria before proceeding:

| Phase | Gate Criteria |
|---|---|
| 0 (Recursive Core) | Trains to convergence; matches or approaches TRM's 40% on ARC-AGI-1; TTT pipeline functional; evaluation produces interpretable numbers |
| 1 (Dual-Process) | Both System 1 and System 2 paths activated; routing is non-degenerate; performance improves over Phase 0 |
| 2 (Memory Types) | Episodic and semantic memories contain different information; retrieval produces relevant results |
| 3 (Working Memory) | Slot attention produces interpretable assignments; slots actually used (not bypassed); training stable |
| 4 (Metacognition) | Confidence scores correlate with accuracy; halt/continue activates at varying points |
| 5 (Global Workspace) | Gating network admits non-trivial information; module coordination measurable |
| 6 (Perception) | Hierarchical features differ from flat encoding; prediction errors concentrate at relevant features |
| 7 (Structural Transfer) | Relational binding produces role-dependent representations |
| 8 (Staged Learning) | Curriculum produces phase transitions; performance trajectory differs from standard training |
| 9 (Emotional Valuation) | Value signals vary across tasks; measurable downstream modulation |
| 10 (Full + Agent) | All principles active; ARC-AGI-3 agent functional; competition submission ready |

### 9.3 Code Architecture

```
study7/
├── CLAUDE.md                      # Zero-context-loss project overview
├── BUILD_STATE.md                 # Current phase, issues, next actions
├── Makefile                       # Phase-gated test targets
├── config/
│   ├── experiment_config.yaml     # All hyperparameters, paths, settings
│   └── competition_config.yaml    # Competition-specific constraints
├── data/
│   ├── arc/                       # ARC-AGI-1/2 datasets
│   ├── re_arc/                    # RE-ARC procedural generators
│   ├── barc/                      # BARC synthetic tasks
│   ├── scan/                      # SCAN benchmark
│   └── arc_agi_3/                 # ARC-AGI-3 environments
├── models/
│   ├── backbone.py                # Phase 0: Recursive transformer core
│   ├── dual_process.py            # Phase 1: System 1/2 routing (P5)
│   ├── memory_systems.py          # Phase 2: Episodic + semantic (P3)
│   ├── working_memory.py          # Phase 3: Slot-based attention (P7)
│   ├── metacognition.py           # Phase 4: Confidence + halt/continue (P8)
│   ├── global_workspace.py        # Phase 5: Workspace + gating (P1)
│   ├── perception.py              # Phase 6: Biological vision (V1-V3)
│   ├── structural_transfer.py     # Phase 7: Relational binding (P6)
│   ├── staged_learning.py         # Phase 8: Curriculum manager (P2)
│   ├── emotional_valuation.py     # Phase 9: Value signals (P4)
│   ├── full_architecture.py       # Phase 10: Integration module
│   └── arc_agi3_agent.py          # ARC-AGI-3 agent wrapper
├── training/
│   ├── trainer.py                 # Main training loop
│   ├── ttt.py                     # Test-time adaptation (puzzle embeddings + full-param fine-tuning)
│   ├── hyperparam_search.py       # Random search for HP tuning
│   └── curriculum.py              # Staged learning curriculum
├── evaluation/
│   ├── arc_evaluator.py           # ARC-AGI-1/2 evaluation (pass@2)
│   ├── arc_agi3_evaluator.py      # ARC-AGI-3 interactive evaluation
│   ├── scan_evaluator.py          # SCAN compositional generalization
│   ├── bigbench_evaluator.py      # BIG-Bench Hard tasks
│   ├── calibration.py             # ECE and confidence metrics
│   ├── efficiency.py              # FLOPs, memory, cost efficiency
│   └── cognitive_metrics.py       # Adaptive compute, slots, routing
├── analysis/
│   ├── ablation_analysis.py       # Forward/reverse/pairwise ablation
│   ├── scaling_analysis.py        # Cross-scale comparison
│   ├── statistical_tests.py       # Significance testing, effect sizes
│   └── visualization.py           # Publication-ready plots
├── competition/
│   ├── kaggle_submission.py       # Kaggle submission pipeline
│   ├── arc_agi3_submission.py     # ARC-AGI-3 competition submission
│   └── compute_budget.py          # Budget tracking and optimization
├── scripts/
│   ├── run_phase.py               # Run a single phase
│   ├── run_full_experiment.py     # Run all phases sequentially
│   ├── run_ablation.py            # Reverse ablation and interactions
│   └── generate_report.py         # Analysis report
└── logs/                          # Experiment logs, checkpoints, results
```

Standard interface for all cognitive modules:
```python
class CognitiveComponent:
    def __init__(self, config):
        """Initialize with configuration."""
    def forward(self, workspace_state):
        """Process workspace state and return modified state."""
    def get_diagnostics(self):
        """Return component-specific diagnostic metrics."""
    def get_ttt_params(self):
        """Return parameters eligible for test-time adaptation (full-parameter, NOT LoRA)."""
    def adapt_to_task(self, examples):
        """Adapt model to a new task using few-shot input-output examples."""
```

### 9.4 Timeline

Timeline is owned by Tim. Claude owns technical strategy. The timeline below estimates effort, not calendar time.

| Effort | Activity |
|---|---|
| Week 1 | Infrastructure: data loaders (RE-ARC, BARC, ARC-AGI-3 toolkit), evaluation pipeline, logging, TTT pipeline |
| Week 1 | Phase 0: TRM-style recursive core. Gate: ≥40% minimum, ≥45% target on ARC-AGI-1. |
| Week 2 | Phase 1 (Dual-Process) + Phase 2 (Memory Types) |
| Week 3 | Phase 3 (Working Memory) + Phase 4 (Metacognition) |
| Week 4 | Phase 5 (Global Workspace) + Phase 6 (Perception) |
| Week 5 | Phase 7 (Structural Transfer) + Phase 8 (Staged Learning) + Phase 9 (Emotional Valuation) |
| Week 5 | Phase 10: Full integration + ARC-AGI-3 agent |
| Week 6 | Reverse ablation, pairwise interactions, scaling analysis |
| Week 6 | ARC-AGI-3 agent optimization on public environments |
| Week 7 | Competition submission preparation, paper drafting |
| Ongoing | ARC-AGI-3 competition participation (rules TBD) |

**Total estimated effort:** 7+ weeks from start of implementation.

**Critical path:** Phase 0 is the foundation. If the TRM-style baseline takes longer than expected, everything shifts. Phases 7–9 (lower priority principles) can be deferred or parallelized if timeline pressure emerges.

### 9.5 Open-Source Arsenal — External Dependencies

Critical codebases to study and build on:

| Resource | Purpose | URL |
|---|---|---|
| TRM | Architecture reference, pre-trained checkpoints | github.com/SamsungSAILMontreal/TinyRecursiveModels |
| CompressARC | MDL approach reference | github.com/iliao2345/CompressARC |
| NVARC | Full winning pipeline reference | github.com/1ytic/NVARC |
| RE-ARC | Unlimited verified training data | github.com/michaelhodel/re-arc |
| BARC | 400K synthetic tasks | github.com/xu3kev/BARC |
| ARC-DSL | 160 grid primitives | github.com/michaelhodel/arc-dsl |
| arckit | Python tools for ARC tasks | github.com/mxbi/arckit |
| ARC-AGI-3 Agents | Official agent templates | github.com/arcprize/ARC-AGI-3-Agents |
| Poetiq solver | Refinement loop reference | github.com/poetiq-ai/poetiq-arc-agi-solver |
| SOAR | Self-improving synthesis + 5M solutions | github.com/flowersteam/SOAR |

Key papers (all available on arXiv):
- ARC Prize 2025 Technical Report: arXiv:2601.10904
- ARC-AGI-2: arXiv:2505.11831
- TRM: arXiv:2510.04871
- TTT effectiveness: arXiv:2411.07279
- Induction + transduction ensembling: arXiv:2411.02272
- Chollet "On the Measure of Intelligence": arXiv:1911.01547
- HRM debunking: arXiv:2510.00355

---

## 10. Risk Management

### 10.1 Competition-Specific Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| ARC-AGI-3 competition rules impose constraints we haven't designed for | Medium | High | Monitor ARC Prize announcements. Architecture is modular — can adapt to constraints. ARC-AGI-2 Grand Prize ($700K) continues as fallback target. |
| ARC-AGI-3 launch delayed beyond March 25, 2026 | Low | Medium | Continue ARC-AGI-1/2 development. Architecture validates on static benchmarks regardless. |
| Competition compute budget tighter than anticipated $0.20/task | Medium | Medium | Small model (7-15M) is inherently compute-efficient. Optimization targets (speculative decoding, prefix caching) provide headroom. |
| Competitor publishes similar cognitive architecture approach before us | Low | High | Speed of execution. Our progressive ablation methodology is unique regardless. Publish ablation results even if competition placement is not top-3. |
| Phase 0 cannot match TRM's 45% baseline | Medium | Critical | This is the #1 technical risk. Mitigation: study TRM codebase in detail, use their pre-trained checkpoints for verification, ensure our recursive refinement implementation is faithful to their design. |

### 10.2 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Training instability with novel cognitive components | High | Medium | Gradient monitoring, layer norm, gradient clipping. Conservative learning rates. Start simple — each component's initial implementation is the simplest thing that could demonstrate the principle. |
| Component collapses to trivial solution (e.g., always routes to System 1) | High | High | Explicit diversity losses. Phase gate criteria check for degeneracy. Architectural constraints (minimum System 2 activation rate during training). |
| Insufficient VRAM for larger configurations | Medium | Medium | Gradient accumulation, mixed precision (FP16). Reduce batch size. 50M is hard upper bound. |
| Cognitive modules slow inference below competition time budget | Medium | High | Profile early. Aggressive pruning of non-contributing components. Competition build may use fewer principles than the full 8. |

### 10.3 Scientific Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Cognitive principles don't help at this scale | Medium | Critical | Legitimate finding. Report honestly. Analyze whether it's the principles or the implementation. TRM comparison isolates the question. |
| "Kitchen sink" critique — too many components | High | High | Progressive build IS the defense. Ablation tables show each component's measured contribution. Non-contributing components are documented or removed. |
| HRM precedent — cognitive claims debunked | High | High | HRM Cautionary Principle is binding. No claim without forward AND reverse ablation evidence. |
| Emotional valuation perceived as unfounded | Medium | Medium | Ground in metabolic efficiency rationale (Levy & Calvert, 2021). If ablation shows no contribution, report as null result — this is our most novel and highest-risk principle. |

### 10.4 Complexity Creep — The Most Important Risk

Eight cognitive principles, each with its own implementation. The temptation will be to make each component more sophisticated.

**Mitigation:** Start simple. Each component's initial implementation is the simplest thing that could possibly demonstrate the principle. Five attention slots, not a sophisticated gating mechanism. A basic confidence head, not a full introspection network. We add sophistication only after the simple version proves its worth in ablation.

**Competition-specific mitigation:** If the full 8-principle architecture is slower to develop than the timeline allows, we ship the best-performing subset. A 4-principle architecture that wins is better than an 8-principle architecture that isn't ready. The ablation data tells us which principles to keep.

### 10.5 Theoretical Confidence Tiers

Not all principles rest on equally firm foundations. This calibration protects against reviewer challenges.

**Tier 1 — Well-Established:**
- Sparse coding (P7 partial): Information-theoretically optimal AND metabolically necessary. Validated across vision (Olshausen & Field, 1996), auditory (Smith & Lewicki, 2006), and theory (Chalk et al., 2018).
- Predictive coding (P1 partial, V1): Hierarchical prediction error processing confirmed across auditory and visual systems.
- Parallel processing / modular specialization (P1): Demonstrated at molecular level — mutually exclusive biophysical constraints force parallel streams.
- Feedback-dominant architecture (P1 partial): Corticothalamic projections outnumber thalamocortical by ~10:1.

**Tier 2 — Strong Evidence, Active Refinement:**
- Dual-process (P5): Behavioral evidence strong (Kahneman). Neural implementation in fast/slow pathways demonstrated. MIRAGE validates computationally.
- Multiple memory types (P3): Behaviorally well-established. Neural substrates identified. Computational implementations promising.
- Critical periods / staged learning (P2): Molecular cascade well-established (Hensch). Bayesian interpretation theoretically motivated.

**Tier 3 — Theoretically Motivated, Higher Risk:**
- Emotional valuation (P4): Neuroscience evidence for separable affective prediction errors is strong, but no precedent for integrating into abstract reasoning architectures.
- Metacognitive monitoring as explicit architecture (P8): The behavior is validated; whether explicit implementation beats emergent metacognition is open. TRM achieves strong results through recursive refinement without explicit metacognition.

---

## 11. Analysis Plan

### 11.1 Primary Analysis: Progressive Build Curve

Plot performance (y-axis) against phase number (x-axis) for each benchmark. This is the study's signature visualization — it shows whether adding cognitive principles progressively improves competition performance.

Include confidence intervals from multiple seeds. TRM baseline as horizontal reference line. Mark which phases exceed TRM.

### 11.2 Ablation Analysis

For each cognitive principle, report:
- Forward marginal contribution (with 95% CI)
- Reverse marginal contribution (with 95% CI)
- Difference between forward and reverse (interaction indicator)
- Rank ordering by contribution size (hypothesis H3 predicts P5 is largest)

### 11.3 Statistical Testing

- **Per-phase comparisons:** Two-tailed paired t-tests with Holm-Bonferroni correction
- **Full architecture vs. Phase 0:** Two-tailed independent t-test, Cohen's d effect size
- **Full architecture vs. TRM:** Two-tailed independent t-test, Cohen's d
- **Synergy test (H8):** Sum of individual forward contributions vs. actual full performance. Bootstrap 95% CI.
- **TTT amplification (H10):** Compare TTT gain on cognitive architecture vs. baseline
- **Scaling interaction:** Two-way ANOVA {baseline, full} × {7M, 20M, 50M}

### 11.4 Reporting Standards

- Mean ± standard deviation across seeds
- Effect sizes with confidence intervals
- Full hyperparameter configurations in appendix
- Training curves for all variants
- Negative results reported with equal prominence
- Code and trained model weights released under MIT License
- Competition submission code published on GitHub

---

## 12. Publication Strategy

### 12.1 The Competition IS the Publication

If we achieve competitive performance on ARC-AGI-3 and/or exceed TRM on ARC-AGI-1, the paper writes itself. "Cognitive Architecture Achieves [X]% on ARC-AGI-3 with [Y]M Parameters: Systematic Ablation of Human-Inspired Principles" is a compelling title at any top venue.

**Two publication scenarios:**

**Scenario A — Competitive performance (top-3 on ARC-AGI-3 or >50% on ARC-AGI-1):** Lead with competition results. Frame cognitive principles as the architecture that enabled the result. Progressive ablation shows which principles drove performance. Story: "Principled cognitive architecture design, not just engineering optimization, can win abstract reasoning competitions."

**Scenario B — Modest performance (>45% on ARC-AGI-1 but not top-3 on ARC-AGI-3):** Lead with ablation methodology. Frame as: "First systematic measurement of how individual cognitive principles contribute to abstract reasoning." Performance is respectable (beats TRM's 40%), and the ablation data is the primary contribution. This paper is publishable regardless of competition placement.

### 12.2 Primary Venue Target: ICLR 2027

Submissions expected ~October 2026. Strongest target for three reasons:

1. **Demonstrated receptivity to bio-inspired work.** ICLR 2025 awarded Oral presentations to three bio-inspired papers.
2. **Explicit protection for non-SOTA results.** "A lack of state-of-the-art results does not by itself constitute grounds for rejection."
3. **Community alignment.** ICLR bridges ML engineering and theoretical motivation.

### 12.3 Alternative Venues

| Venue | Submission | Notes |
|---|---|---|
| NeurIPS 2026 (Sydney, Dec 6-12) | ~May 2026 | "Neuroscience and cognitive science" in scope. Position Paper track available. |
| CogSci 2027 | ~Feb 2027 | Theme TBD. Non-archival 6-page, allows subsequent ICLR submission. |
| AAAI 2027 | ~Aug 2026 | "Cognitive Modeling and Cognitive Systems" track. |
| NeurIPS 2026 NeuroAI Workshop | ~Aug 2026 | Lower bar, non-archival, builds community visibility. |

### 12.4 Framing Principles

1. **Lead with competition results.** "Achieving X% on ARC-AGI-3 with Y parameters." Competition success is the hook; cognitive science is the how.
2. **Lead with efficiency, not biology.** Frame cognitive principles as principled solutions to the efficiency problem — maximum reasoning per unit compute. The brain operates under ~20W with 35× communication-to-computation cost ratio. Our architecture operates under analogous GPU constraints.
3. **Make progressive ablation the methodological headline.** "First systematic evaluation of how individual cognitive principles contribute to abstract reasoning capability."
4. **Connect to Chollet's framework.** Position this study as the principled cognitive architecture ARC was designed to test.
5. **Use ARC-AGI-3 results to demonstrate interactive reasoning capability.** This is the frontier — 0% for all current AI. Any non-zero result with clear cognitive principle contribution is significant.

### 12.5 Mandatory Citations

| Citation | Relevance |
|---|---|
| Chollet, "On the Measure of Intelligence" (2019) | Theoretical foundation for ARC-AGI |
| ARC Prize 2025 Technical Report (arXiv:2601.10904) | Competition landscape |
| TRM (arXiv:2510.04871) | Primary calibration baseline |
| HRM debunking (arXiv:2510.00355) | Why rigorous ablation matters |
| Goyal & Bengio, Global Workspace (ICLR 2022) | P1 precedent |
| Locatello et al., Slot Attention (NeurIPS 2020) | P7 foundation |
| Graves, Adaptive Computation Time (2016) | P8 foundation |
| Chalk et al. (2018) | Unified efficient/predictive/sparse coding theory |
| Ganguli & Simoncelli (2014) | Optimal heterogeneous population coding |
| Laughlin (1981) | Infomax principle |
| Smith & Lewicki, Nature (2006) | Efficient auditory coding |
| MIRAGE (arXiv:2507.18868) | Closest cognitive integration approach |
| TTT effectiveness (arXiv:2411.07279) | Test-time training methodology |
| Induction + transduction (arXiv:2411.02272) | Dual-process validation for ARC |

---

## 13. Connection to the PURE Research Program

### 13.1 Relationship to Other Studies

| Study | Relationship to This Study |
|---|---|
| Study 1 (PURE Method for LLM Enhancement) | Study 1 tests if prompting frameworks improve existing LLMs. This study asks if we can build better reasoning systems instead. Study 1's finding that Fear identification (anticipatory failure awareness) explains 100× more variance than Rules or Constraints informs our emphasis on metacognitive monitoring (P8). |
| Study 4 (Cognitive Memory for LLMs) | Study 4 augments existing LLMs with memory. This study builds multiple memory types into the architecture from scratch. |
| Study 5 (Temporal Performance Variation) | Study 5 documents when LLMs fail. This study builds architectures designed not to fail in those ways. |
| Study 6 (Biological Vision) | Sibling study independently implementing the same biological vision neuroscience for desktop screen observation. Cross-study comparison opportunity. |
| Study 8 (Biological Auditory Processing) | This study's theoretical foundations draw on auditory neuroscience for cross-modal convergence evidence and metabolic efficiency arguments. |

### 13.2 How Universal Trinity Theory Informs Architecture Design

The Universal Trinity Theory (Rules, Constraints, Fear) maps to architectural design:

- **Rules** → The cognitive principles themselves. Design rules we're implementing.
- **Constraints** → Hardware limitations, parameter budgets, competition compute constraints. Force efficient design.
- **Fear/Uncertainty** → The model's metacognitive uncertainty. Principle 8 makes the Trinity's third element a first-class architectural citizen. Study 1's discovery that Fear dominance (η² = 0.1987) dramatically outperforms rule-based scaffolding is directly validated by our competitive landscape analysis: metacognitive monitoring (uncertainty-aware refinement) is the pattern behind every winning approach.

### 13.3 Bidirectional Contribution

1. **Validation of cognitive principles:** If cognitive architecture wins or competes on ARC-AGI, this validates the theoretical foundation underlying all PURE studies.
2. **Research platform:** A working cognitive architecture becomes a platform for testing additional hypotheses about AI cognition.
3. **Competition credibility:** An ARC-AGI competition result provides external validation of the PURE research program's architectural principles.

---

## 14. Ethical Considerations

This study trains small neural networks from scratch on public benchmark datasets. It does not use proprietary or private data, train on personal information, create systems designed for deception, or produce models at a scale that poses deployment risk.

The research aligns with responsible AI development: building more capable systems through architectural insight rather than scale, which is inherently more interpretable, efficient, and accessible.

All code, trained model weights, and competition submissions released under MIT License on GitHub.

---

## 15. Version History

| Version | Date | Changes |
|---|---|---|
| 1.0 | February 5, 2026 | Initial protocol design. Complete research framework. |
| 1.1 | February 5, 2026 | Updated Research Partner to Claude Opus 4.6. |
| 1.2 | February 5, 2026 | Expanded perceptual front-end with biological vision neuroscience principles. |
| 1.3 | February 5, 2026 | Corrected theoretical lineage — Study 6 and Study 7 are independent sibling implementations. |
| 2.0 | February 5, 2026 | Version bump reflecting protocol maturity. |
| 3.0 | February 6, 2026 | Major update incorporating comprehensive literature review. Expanded competitive landscape (MIRAGE, SOAR, VSA, ARC-AGI evolution). HRM Cautionary Principle adopted. Publication strategy added. |
| 3.1 | February 6, 2026 | Theoretical foundation deepened with cross-modal neuroscience (auditory system convergence evidence, metabolic efficiency quantification, Chalk et al. unified coding theory). |
| 4.0 | February 7, 2026 | **Fundamental transformation from research study to competition-focused build.** Changes: (1) Reframed mission — win ARC-AGI-3 while producing publication-quality ablation data. (2) Massively expanded competitive landscape with deep competitive intelligence: TRM architecture dissection, NVARC pipeline, MindsAI TTFT+AIRV, CompressARC MDL, Poetiq refinement, SOAR evolutionary synthesis, Berman evolutionary NL instructions, efficiency frontier analysis, test-time training paradigm decoded. (3) ARC-AGI-3 elevated from "aspirational" to primary competition target with dedicated agent architecture specification (exploration module, planning system, episodic interaction memory, global workspace coordination). (4) Principles reordered by competition priority: P5 (Critical) → P3/P7/P8 (High) → P1 (Strategic) → P6/P2 (Moderate) → P4 (Lowest). (5) Architecture redesigned around TRM-informed recursive transformer core (7-15M params) with cognitive enhancements. (6) TTT elevated from evaluation method to core architectural feature (two-level adaptation: fast recursive refinement + slow per-task LoRA). (7) Data strategy transformed to "leverage existing ecosystem" — RE-ARC, BARC, ARC-DSL, NVARC augmentation approach. (8) Progressive build phases reordered for competition priority. (9) TRM established as calibration benchmark at every phase. (10) Two new hypotheses: H9 (Interactive Reasoning) and H10 (TTT Amplification). (11) Execution plan updated for Claude Code development environment with zero-context-loss strategy. (12) Competition submission pipeline and compute budget planning added. (13) Publication strategy reframed — competition results ARE the publication material. (14) Open-source arsenal catalogued (10 codebases, 7 key papers). (15) Updated competition leaderboard with GPT-5.2, Berman, and Chollet's AGI roadmap. |

---

*This protocol was designed using the PURE Method (Preparation, Understanding, Reliability, Execution) applied to the specific requirements of competition-focused systems research in novel neural architecture design. It prioritizes winning ARC-AGI-3 while producing rigorous, publication-quality scientific data through progressive ablation of human-inspired cognitive principles.*
