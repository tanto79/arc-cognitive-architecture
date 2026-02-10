# PHASES.md — Cognitive Architecture for Abstract Reasoning

All phases build progressively. Each phase adds one cognitive principle (or principle group)
to the architecture. Gate criteria must pass before proceeding to next phase. Every phase
is measured against TRM (7M params, 40% ARC-AGI-1) and URM (14M params, 53.8% ARC-AGI-1) as calibration baselines.

**Competition rule:** If a principle doesn't improve performance within 1 week of effort
(after debugging and one redesign attempt), skip it. Document the finding. Move on.
A 4-principle architecture that wins beats an 8-principle architecture that isn't ready.

---

## Phase 0: RECURSIVE TRANSFORMER CORE (Foundation)

**What:** TRM-style recursive transformer. This is the foundation everything else builds on.

**Architecture (CORRECTED per ADR-002/003):**
- Small recursive transformer (~7-10M base parameters)
- Recursive refinement: 16 outer supervision steps, T=3 inner cycles per step, n=6 latent reasoning steps per cycle
- **Final-cycle-only gradients** (NOT deep supervision): cycles 0 to T-2 run under torch.no_grad(), backprop only through last cycle. Latent states detached between supervision steps.
- Three maintained states: input embedding x (fixed), answer state y (progressively refined), latent states z_H and z_L (implicit chain of thought)
- **Same-resolution latent states:** z_H and z_L at SAME resolution with weight-shared update module (NOT 4× downsampled)
- 512 dims/token, **post-norm** (add then normalize), functional RMSNorm (no learnable params), no bias, Rotary Positional Encoding
- **Learned initial states** (H_init, L_init — not zeros)
- **Input injection:** input embeddings added to z_H at every z_L update step
- **Stablemax CE loss** (not standard softmax)
- Per-task puzzle embeddings: one [1, 512] vector per puzzle × augmentation, zero-initialized, trained with SignSGD at 100× model LR (NOT LoRA — see ADR-002/003)
- **AdamATan2 optimizer**, LR=1e-4, constant after warmup, WD=0.1, beta2=0.95
- Data augmentation: 8 geometric transforms × color permutations + translational augmentation (~1M effective samples)
- Grid encoding: 12 learned embeddings (PAD=0, EOS=1, colors 2-11), 512-dim

**IMPORTANT:** TRM's puzzle embeddings are NOT test-time training. They are pre-trained lookup vectors that only work for puzzles seen during training. Real TTT for new puzzles is added in Phase 0.5.

**Training data:** RE-ARC procedural generators + ARC-AGI-1 training set (400 tasks). ~1M effective samples through augmentation.

**Gate criteria:**
- [ ] Trains to convergence on ARC-AGI-1 training set
- [ ] Puzzle embedding pipeline functional (embeddings improve over no-embedding baseline)
- [ ] Evaluation pipeline produces interpretable numbers
- [ ] **≥40% on ARC-AGI-1** (MINIMUM — implementation not fatally broken)
- [ ] **≥45% on ARC-AGI-1** (TARGET — TRM-equivalent)
- [ ] Infrastructure stable: data loading, logging, checkpointing all working
- [ ] VRAM < 15GB during training

**Phase 0 test specifications** (`tests/test_phases/test_phase_00.py`):
```
test_model_instantiation:    Model creates without error, param count 7-15M
test_forward_pass_shapes:    Input [B, seq, 512] → output [B, seq, 12] at final cycle
test_gradient_final_cycle:   Gradient flows only through last cycle, cycles 0-1 have no grad
test_same_res_latents:       z_H and z_L have identical spatial dimensions
test_puzzle_embed_creation:  Puzzle embeddings created, one per puzzle × augmentation
test_puzzle_embed_effect:    Correct puzzle embedding improves prediction vs random/zero embedding
test_deterministic_seeding:  Same seed → same outputs (within floating point tolerance)
test_vram_under_limit:       Peak VRAM < 15GB during training step (GPU required)
test_evaluation_pipeline:    Evaluator produces accuracy dict with correct keys
test_data_augmentation:      8 geometric transforms produce 8 distinct grids
```

**Competition significance:** This IS our TRM reproduction. If we can't match TRM's 40%, we have an implementation problem. Everything else is built on this.

**Key references:** TRM: arXiv:2510.04871. URM: arXiv:2512.14693. Code: github.com/SamsungSAILMontreal/TinyRecursiveModels

**Status:** NOT STARTED
**Result:** —

---

## Phase 0.5: TBPTL UPGRADE (URM Innovations — HIGH IMPACT)

**What:** Adopt the two highest-impact findings from URM (Universal Reasoning Model, arXiv 2512.14693, December 2025) that together improve ARC-AGI-1 from 40% to 53.8%.

**Architecture changes from Phase 0:**
- **TBPTL (Truncated Backpropagation Through Loops):** With 8 inner-loop iterations, first 2 run forward-only, remaining 6 get full backprop with per-step loss. URM ablation: removing TBPTL drops 53.8% → 40.0% exactly.
- **ConvSwiGLU FFN:** Replace standard FFN with SwiGLU activation + depthwise convolutions. Adds ~7M params (total ~14M). URM ablation: removing ConvSwiGLU drops 53.8% → 45.3%.
- **Real test-time adaptation:** Full-parameter fine-tuning on few-shot examples for genuinely new puzzles. At 14M params FP16, model is ~28MB — fine-tuning per puzzle takes seconds.

**Gate criteria:**
- [ ] **≥50% on ARC-AGI-1** with puzzle embeddings (TBPTL + ConvSwiGLU)
- [ ] Real TTT functional: non-zero accuracy on held-out puzzles never seen during training
- [ ] TBPTL gradient flow correct: exactly n_backprop iterations have gradients
- [ ] VRAM < 15GB during training

**Key reference:** URM: arXiv:2512.14693

**Status:** NOT STARTED
**Result:** —

---

## Phase 1: DUAL-PROCESS REASONING (P5) — CRITICAL

**What:** System 1 (fast transductive pattern prediction) + System 2 (slow inductive rule synthesis) with learned routing.

**Why CRITICAL:** Most validated pattern in ALL of ARC research. Analysis shows 26 tasks solved ONLY by induction, 35 ONLY by transduction, 19 by both. Every top competitor ensembles both. This is the single largest expected performance gain.

**Architecture:**
- System 1: Fast feedforward path. Takes input grid, produces direct output prediction. Low latency, handles familiar patterns.
- System 2: Slow recurrent path with more refinement steps. Handles novel/complex tasks requiring deliberate reasoning.
- Learned router: Lightweight network that scores task difficulty/novelty and allocates compute between System 1 and System 2.
- Ensemble: Both paths run on every task. Outputs combined via augmentation-consistency voting (predictions stable across geometric transforms are preferred).

**Gate criteria:**
- [ ] Both System 1 and System 2 paths activated during inference (non-degenerate routing)
- [ ] Router doesn't always pick one path (check activation statistics)
- [ ] Performance improves over Phase 0 on ARC-AGI-1
- [ ] Inference time correlates with task difficulty (System 2 used more on harder tasks)

**Hypothesis tested:** H3 (P5 is largest single-principle contributor)

**Status:** NOT STARTED
**Result:** —

---

## Phase 2: MULTIPLE MEMORY TYPES (P3) — HIGH

**What:** Episodic memory (stores specific examples) + semantic memory (consolidated abstractions). Working memory added in Phase 3.

**Why HIGH:** ARC-AGI-3 explicitly lists "Memory" as a required capability. MIRAGE's episodic buffer for variable bindings validates the approach. ArcMemo (Runner-up Paper 2025) implements lifelong LLM memory specifically for ARC.

**Architecture:**
- Episodic memory: Key-value store of (input, output, transformation) tuples from demonstration pairs and (for ARC-AGI-3) interaction history. Content-addressable retrieval.
- Semantic memory: Slower-updating store of consolidated abstract patterns (e.g., "rotation transforms," "color mapping rules"). Updated during training, read-only during inference.
- Distinct read/write dynamics: episodic writes frequently (every example), semantic writes rarely (consolidation from episodic).

**Gate criteria:**
- [ ] Episodic and semantic memories contain measurably different information
- [ ] Memory retrieval produces relevant results for test tasks
- [ ] Performance improves over Phase 1 (or at minimum doesn't degrade)

**Hypothesis tested:** H9 (partial — memory component for interactive reasoning)

**Status:** NOT STARTED
**Result:** —

---

## Phase 3: WORKING MEMORY BOTTLENECK (P7) — HIGH

**What:** Slot-based attention (4-7 slots) forcing information through a capacity-limited bottleneck.

**Why HIGH:** Working memory capacity directly determines performance on hard compositional tasks in ARC-AGI-2. TRM's multi-scale latent states are a partial implementation — explicit slot attention should improve on this.

**Architecture:**
- Slot attention mechanism: 4-7 learnable slot vectors
- Heterogeneous slot capacity: most slots at standard resolution, 1-2 high-capacity slots
- Tight coupling with perception: slots drive foveated attention, attention feeds slots
- Information bottleneck: all reasoning must flow through slot representations

**Gate criteria:**
- [ ] Slot attention produces interpretable, non-degenerate assignments
- [ ] Slots are actually used (information flows through them, not around them)
- [ ] Different tasks produce different slot activation patterns
- [ ] Training is stable (slot attention can be finicky)

**Hypothesis tested:** H4 (working memory improves OOD generalization)

**Status:** NOT STARTED
**Result:** —

---

## Phase 4: METACOGNITIVE MONITORING (P8) — HIGH

**What:** Explicit confidence estimation + learned halt/continue mechanism enhancing TRM's recursive refinement.

**Architecture:**
- Confidence estimation head: predicts probability of correct answer at each refinement step
- Halt/continue mechanism: learned decision to stop refining or continue (Adaptive Computation Time)
- Instead of fixed 16 steps, adaptively allocate 2-16+ steps based on confidence
- ECE as training objective component

**Gate criteria:**
- [ ] Confidence scores have non-trivial correlation with actual accuracy
- [ ] Halt/continue activates at varying points (not always max or min steps)
- [ ] Harder tasks get more refinement steps (adaptive compute)
- [ ] ECE improves over Phase 0's implicit confidence

**Hypothesis tested:** H5 (metacognitive advantage in calibration)

**Status:** NOT STARTED
**Result:** —

---

## Phase 5: GLOBAL WORKSPACE (P1) — STRATEGIC

**What:** Shared workspace with learned gating for cross-module communication.

**Why STRATEGIC:** No competitor implements this. Untested competitive advantage. For ARC-AGI-3, coordinating perception → planning → action → memory is architecturally natural through a global workspace.

**Architecture:**
- Shared bottleneck workspace: all modules read from and write to
- Learned gating network: scores each module's proposed broadcast, admits only relevant info
- Distinct from P7: P1 gating determines RELEVANCE, P7 determines CAPACITY
- Based on Goyal & Bengio (ICLR 2022)

**Gate criteria:**
- [ ] Gating network admits non-trivial information
- [ ] Module coordination is measurable
- [ ] Performance improves or at minimum doesn't degrade

**Hypothesis tested:** H8 (partial — synergy through coordinated processing)

**Status:** NOT STARTED
**Result:** —

---

## Phase 6: BIOLOGICAL PERCEPTION (V1-V3) — MODERATE

**What:** Replace flat grid encoding with hierarchical perceptual front-end inspired by biological vision.

**Architecture:**
- V1 (Predictive Coding): bidirectional connections, top-down predictions, precision-weighted error
- V2 (Hierarchical Sparse): L1 edges → L2 local patterns → L3 objects → L4 workspace
- V3 (Foveated Attention): task-driven priority allocation, coupled with working memory slots

**Gate criteria:**
- [ ] Hierarchical features differ measurably from flat encoding
- [ ] Prediction errors concentrate at task-relevant features
- [ ] Performance on spatial ARC tasks improves over flat encoding

**Hypothesis tested:** H2 (perceptual advantage)

**Status:** NOT STARTED
**Result:** —

---

## Phase 7: STRUCTURAL TRANSFER (P6) — MODERATE

**What:** Compositional binding + relational attention + geometric equivariances.

**Gate criteria:**
- [ ] Relational binding produces role-dependent representations
- [ ] Compositional generalization improves on ARC spatial tasks

**Hypothesis tested:** H6 (structural transfer)

**Status:** NOT STARTED
**Result:** —

---

## Phase 8: STAGED LEARNING (P2) — MODERATE

**What:** Training curriculum from simple → complex + progressive layer freezing.

**Gate criteria:**
- [ ] Curriculum produces measurable phase transitions in training dynamics
- [ ] Performance trajectory differs from standard (non-staged) training
- [ ] Final performance improves or training efficiency improves

**Hypothesis tested:** H7 (partial — sample efficiency)

**Status:** NOT STARTED
**Result:** —

---

## Phase 9: EMOTIONAL VALUATION (P4) — LOWEST PRIORITY

**What:** Learned resource allocation signals modulating processing priority.

**Gate criteria:**
- [ ] Value signals vary across tasks
- [ ] Signals measurably modulate downstream processing
- [ ] Some improvement on ambiguous tasks or exploration efficiency

**Hypothesis tested:** H8 (partial — synergy), H9 (partial — exploration efficiency)

**Status:** NOT STARTED
**Result:** —

---

## Phase 10: FULL INTEGRATION + ARC-AGI-3 AGENT — COMPETITION

**What:** All principles active. ARC-AGI-3 agent wrapper. Competition submission.

**Architecture:**
- Full cognitive architecture: all passing phases integrated
- ARC-AGI-3 agent wrapper: observation encoder → cognitive core → action decoder
- Exploration module (P5 + P4): hypothesis-driven environment exploration
- Planning system (P7 + P8): working memory holds goals, metacognition tracks progress
- Episodic interaction memory (P3): records (state, action, outcome) tuples
- Global workspace coordination (P1): orchestrates perception → planning → action → memory

**Gate criteria:**
- [ ] All active principles functional simultaneously (no conflicts)
- [ ] ARC-AGI-3 agent wrapper functional with Developer Toolkit
- [ ] Non-zero performance on ARC-AGI-3 public environments
- [ ] Competition submission pipeline tested

**Hypothesis tested:** H1 (cumulative), H8 (synergy), H9 (interactive reasoning)

**Status:** NOT STARTED
**Result:** —

---

## Post-Phase: Ablation & Competition

**Reverse ablation:** Remove one principle at a time from full architecture, measure impact.
**Pairwise interactions:** Test priority pairs (P5×P7, P8×P5, P3×P8, P1×P5).
**Scaling analysis:** Run at 7M, 20M, 50M parameter counts.
**Competition optimization:** Profile and optimize for competition compute budget.
**Paper drafting:** Compile ablation results into publication.
