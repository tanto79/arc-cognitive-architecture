PHASES — Cognitive Architecture for Abstract Reasoning
Track A: Substrate Development (ARC-AGI-1/2 Validation)
Phase 0: PCLT Substrate Core (~9-12M params)
4-layer looped transformer block (weight-tied, ADR-005 verified)
Inner loop: 8 iterations with TBPTL (2 forward-only + 6 backprop with per-step CE loss)
Outer loop: ACT with up to 16 steps, learned halting probability
ConvSwiGLU FFN (fused projection, SiLU gating, 1D depthwise conv k=2, second SiLU, 2× expansion)
Standard cross-entropy loss (NOT Stablemax — ADR-005)
Flat grid encoding, puzzle embeddings, learned initial states, RoPE, EMA
Gate: >=53.8% ARC-AGI-1 (match URM). Target: exceed URM, approach Loop-ViT 65.8%.
Status: Not started
Result: —
Phase 1: Biological Principles — Native Integration (~15-22M params)
ADD: Slot Attention (4-8 slots), PC auxiliary loss, kWTA (5%), dynamic exit (τ=0.05, inference-only), slow planner
Each sub-component ablated individually
Gate: Exceeds Phase 0 by >=5% absolute on ARC-AGI-1
Status: Not started
Result: —
Phase 2: TTT + Full Substrate Validation (~15-22M params)
ADD: Full-parameter TTT on few-shot examples
Evaluate on ARC-AGI-1 AND ARC-AGI-2
Gate: >=60% ARC-AGI-1 with TTT (target >65.8%), >=15% ARC-AGI-2 (target >25%, stretch >85% for $700K)
SUBSTRATE LOCKED after this phase
Status: Not started
Result: —
Track B: Agent Development (ARC-AGI-3 Validation)
Phase 3: World Model + Environment Interface (~19-28M params)
ADD: ARC-AGI-3 SDK (pip install arc-agi), Mamba-based SlotSSM world model, PE as intrinsic reward
Gate: World model PE <10% on known rules within 500 interactions
Status: Not started
Result: —
Phase 4: Exploration + Planning (~22-32M params)
ADD: Graph explorer, NovelD, System 1 policy, System 2 MCTS, meta-controller
Gate: Beat graph baseline on >=1 level, 2000 FPS
Status: Not started
Result: —
Phase 5: Memory + Transfer + Full Agent (~24-34M params)
ADD: Episode memory (DND), rule memory, cross-level transfer
Gate: Learning curve across levels, exceed preview SOTA, target 1st place
Status: Not started
Result: —
Track C: Competition Optimization
Phase 6: Competition Optimization
Reverse ablation, pairwise interactions, submission prep
Gate: Submission ready
Status: Not started
Result: —
Baselines
System	Params	ARC-AGI-1	ARC-AGI-2	ARC-AGI-3
Loop-ViT	18M	65.8%	14.2%	—
URM	14M	53.8%	16%	—
TRM	7M	45%	8%	—
StochasticGoose	CNN	—	—	12.58%
Graph-Based Explorer	0	—	—	~12.58%
Frontier AI (GPT-5.2)	>1T	90.5%	~52.9%	0%
