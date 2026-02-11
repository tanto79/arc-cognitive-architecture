# ADR-005: Reference Architecture Verification — Corrected Implementation Specifications

**Date:** February 11, 2026
**Status:** Accepted (binding for implementation)
**Triggered by:** Pre-implementation verification of Protocol v5.0 specifications against actual papers and codebases
**Purpose:** Provide Claude Code with verified, implementation-ready specifications. All discrepancies between Protocol v5.0 assumptions and verified reality are documented with corrections.

---

## Context

Protocol v4.0 suffered from a mid-build discovery problem: Claude Code's Sub-task 0.4 (TRM Codebase Study) revealed that the protocol had the gradient strategy, latent state resolution, TTT mechanism, normalization order, and loss function all wrong — leading to ADR-002 and ADR-003 emergency corrections.

To prevent this from recurring, all reference architecture specifications were verified against the actual papers and available codebases *before* Claude Code begins Phase 0 implementation. This ADR documents seven discrepancies found and provides corrected specifications.

**Sources verified:**
- Loop-ViT: arXiv:2602.02156, Feb 2, 2026 (HTML full text)
- URM: arXiv:2512.14693v2, Dec 24, 2025 (HTML full text + PDF)
- TRM: arXiv:2510.04871, Oct 2025 (HTML full text)
- ARC-AGI-3: arcprize.org, GitHub repos, 30-day learnings blog post
- SlotSSMs: arXiv:2406.12272, NeurIPS 2024
- SOLD: arXiv:2410.08822
- Graph-Based Explorer: arXiv:2512.24156

**Codebases identified:**
- TRM: https://github.com/SamsungSAILMontreal/TinyRecursiveModels (6.1K stars, MIT)
- URM: https://github.com/UbiquantAI/URM (official, latest)
- Loop-ViT: https://github.com/WenjieShu/LoopViT (claimed in paper; may not yet be populated)
- SlotSSMs: https://github.com/JindongJiang/SlotSSMs
- Graph Explorer: https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore
- ARC-AGI Toolkit: https://github.com/arcprize/ARC-AGI
- ARC-AGI-3 Agents: https://github.com/arcprize/ARC-AGI-3-Agents

---

## Discrepancy Summary

| # | Protocol v5.0 Assumption | Verified Reality | Severity | Action |
|---|--------------------------|------------------|----------|--------|
| D1 | URM uses Stablemax CE loss | Standard cross-entropy | **HIGH** | Fix base.yaml, all training code |
| D2 | URM changes from TRM are ConvSwiGLU + TBPTL only | Also 4 layers (not 2), standard UT loop, outer ACT loop | **HIGH** | Revise Phase 0 architecture |
| D3 | ConvSwiGLU uses separate W1 + W_gate projections | Fused single W_up, split into gate and value | Low | Functionally equivalent; use fused for efficiency |
| D4 | Loop-ViT ConvGLU applies conv uniformly | Heterogeneous: conv on image tokens only, task tokens bypass | Medium | Note for Phase 1 reference |
| D5 | Puzzle embedding always uses 100× LR via SignSGD | 100× confirmed for ARC; SignSGD unverified; zero-init unverified | Medium | Verify from TRM codebase |
| D6 | SlotSSMs uses S4 transitions | Uses Mamba-based SSMs | Medium | Fix protocol for Phase 3+ |
| D7 | SOLD outperforms DreamerV3 >2× broadly | >2× only on "Distinct" relational variants | Low | Correct protocol language |

---

## Correction 1: Loss Function (HIGH SEVERITY)

### What was assumed
Protocol v5.0 and base.yaml specified `loss_type: "stablemax_ce"` based on earlier ADR-002 analysis of the TRM codebase.

### What the papers say
Neither URM nor Loop-ViT mentions Stablemax anywhere. Both use **standard per-pixel cross-entropy loss**. The TRM paper's codebase may use a custom loss variant, but neither successor paper adopted it.

### Corrected specification
```yaml
# base.yaml correction
loss_type: "cross_entropy"  # Standard CE. NOT Stablemax.
```

**Implementation:** `torch.nn.CrossEntropyLoss()` applied per-pixel on the output grid prediction. If numerical stability is needed, use `torch.nn.CrossEntropyLoss()` which already uses log-sum-exp internally. Do not implement a custom Stablemax variant.

---

## Correction 2: Phase 0 Layer Count and Loop Structure (HIGH SEVERITY)

### What was assumed
Protocol v5.0 specified the looped transformer as "2-4 layer block" and base.yaml set `transformer_layers: 2` (inheriting from TRM's 2-layer design).

### What the papers say

**URM uses 4 layers per loop block, NOT 2.** Additionally, URM wraps the 8-step inner loop in an **outer Adaptive Computation Time (ACT) loop of up to 16 steps**. The full computation path is:

```
Outer loop (ACT, up to S=16 steps):
  For each outer step s:
    Inner loop (TBPTL, T=8 iterations):
      For each inner step t:
        z_{t+1} = LoopBlock_4layers(z_t)    # 4-layer transformer block
        if t >= 2: compute loss L_t          # TBPTL: first 2 forward-only
    Compute halting probability p_s
    If cumulative halt ≥ threshold: stop outer loop
```

**Loop-ViT** uses a simpler structure — a single-level loop of T iterations (trained at T=12, inference T varies 4-28) through a hybrid conv+attention block. No outer ACT loop.

### Corrected specification

**Phase 0 should use URM's 4-layer block with the inner TBPTL loop.** The outer ACT loop is an important feature but adds complexity; we should implement it but can defer testing its contribution to ablation.

```yaml
# base.yaml corrections
model:
  transformer_layers: 4               # CORRECTED: 4 layers per loop block (was 2)
  
  # Inner loop (TBPTL)
  loop_iterations: 8                  # Total inner loop iterations
  tbptl_forward_only: 2               # First 2 under torch.no_grad()
  tbptl_backprop: 6                   # Remaining 6 with per-step loss
  
  # Outer loop (ACT) — NEW, was missing entirely
  act_enabled: true                   # Adaptive Computation Time outer loop
  act_max_steps: 16                   # Maximum outer loop steps
  act_halt_threshold: 0.99            # Cumulative halting probability threshold
```

**Parameter impact:** Going from 2 → 4 layers approximately doubles the transformer block parameters. With d=512, 4 layers, 8 heads:
- Attention per layer: ~1.05M (Q, K, V, O projections)
- FFN per layer: ~1.05M (with 2× expansion ConvSwiGLU)
- Per layer total: ~2.1M
- 4 layers: ~8.4M
- Plus embeddings, output head, puzzle embeddings: ~2-3M
- **Estimated total: ~12M parameters** (within our 12-15M Phase 0 target)

Note: the weight-tied loop means the 4-layer block parameters are shared across all 8 inner iterations and all ACT outer steps. Parameter count does NOT scale with iteration count.

---

## Correction 3: ConvSwiGLU Exact Implementation

### What was assumed
Protocol referenced ConvSwiGLU with separate W1 and W_gate matrices and vague positioning of the depthwise convolution.

### Verified URM implementation

```python
# URM ConvSwiGLU FFN — verified from paper
class ConvSwiGLU(nn.Module):
    def __init__(self, d_model, hidden_mult=2):
        super().__init__()
        m = d_model * hidden_mult  # hidden dim = 2 * d_model = 1024
        self.w_up = nn.Linear(d_model, 2 * m, bias=False)    # fused gate+value
        self.w_down = nn.Linear(m, d_model, bias=False)
        self.dw_conv = nn.Conv1d(m, m, kernel_size=2, padding=1, groups=m)  # depthwise
        self.act = nn.SiLU()
    
    def forward(self, x):
        # x: [B, T, D]
        up = self.w_up(x)                        # [B, T, 2*m]
        gate, value = up.chunk(2, dim=-1)         # each [B, T, m]
        h = self.act(gate) * value                # SwiGLU gating: [B, T, m]
        h = h.transpose(1, 2)                     # [B, m, T] for conv
        h = self.dw_conv(h)[:, :, :x.size(1)]     # depthwise conv, trim padding
        h = h.transpose(1, 2)                     # [B, T, m]
        h = self.act(h)                           # second SiLU after conv
        return self.w_down(h)                     # [B, T, D]
```

**Key details verified from paper:**
- W_up is a **single fused projection** of size [D, 2m], output split into gate and value
- m = D (hidden expansion = 2×, NOT the typical 8/3× SwiGLU expansion)
- Depthwise conv is 1D with **kernel size k=2**, applied AFTER gating
- A **second SiLU activation** is applied after the convolution (easy to miss)
- Conv operates along the **sequence/token dimension**, not spatial

**Discrepancy note (D3):** Using separate W1 and W_gate matrices instead of fused W_up is functionally equivalent (same parameter count, same computation) but fused is more memory-efficient. Use fused for implementation.

---

## Correction 4: Loop-ViT Heterogeneous ConvGLU (for Phase 1 reference)

### Why this matters
Loop-ViT's ConvGLU differs from URM's in a non-obvious way: it **only applies convolution to image patch tokens**, while task/rule tokens bypass the convolution entirely. This heterogeneous design preserves abstract information while adding spatial inductive bias only where appropriate.

### Verified Loop-ViT ConvGLU structure

```python
# Loop-ViT Heterogeneous ConvGLU — verified from paper
def conv_glu_forward(self, z, n_task_tokens):
    gate, value = self.linear1(z).chunk(2, dim=-1)
    
    # Split into task tokens and image tokens
    g_task = gate[:, :n_task_tokens, :]       # bypass conv
    g_img = gate[:, n_task_tokens:, :]        # apply conv
    
    # Reshape image tokens to 2D grid, apply 3x3 depthwise conv
    g_img = rearrange(g_img, 'b (h w) c -> b c h w', h=H, w=W)
    g_img = self.dw_conv_2d(g_img)            # 3x3 depthwise, groups=channels
    g_img = rearrange(g_img, 'b c h w -> b (h w) c')
    
    gate = torch.cat([g_task, g_img], dim=1)  # reassemble
    return self.linear2(F.silu(gate) * value)
```

**Key differences from URM ConvSwiGLU:**
- **2D** depthwise conv (3×3 kernel) vs. URM's **1D** conv (kernel size 2)
- Task tokens **bypass** convolution entirely
- Activation is **SiLU** (same as URM)
- No second SiLU after conv (unlike URM)

### Decision for our PCLT implementation
**Phase 0:** Use URM-style ConvSwiGLU (1D, uniform across all tokens). This is simpler and matches our baseline target.

**Phase 1 (when adding Slot Attention):** Consider switching to Loop-ViT-style heterogeneous ConvGLU if we have both slot tokens and image patch tokens. This would be a natural ablation candidate — test uniform vs. heterogeneous conv.

---

## Correction 5: Puzzle Embedding Details — Partially Verified

### What is confirmed
- Puzzle embedding dimension: 512 (matches d_model) ✅
- LR ratio: 100× model LR for ARC tasks (puzzle_emb_lr = 1e-2, model_lr = 1e-4) ✅
- Weight decay: 0.1 ✅

### What is NOT confirmed from papers alone
- **Initialization:** Zero-init is assumed from our earlier TRM codebase analysis (ADR-002) but not stated in either paper
- **Optimizer for embeddings:** SignSGD was specified in ADR-002/003 but neither URM nor TRM paper text confirms this. URM uses AdamAtan2 for everything and does not carve out a separate optimizer for embeddings
- **Embedding shape:** [num_puzzles, 512] is inferred, not explicitly stated

### Action required
Before Phase 0 implementation, Claude Code should verify these three details from the TRM codebase at https://github.com/SamsungSAILMontreal/TinyRecursiveModels. Specifically check:
1. `puzzle_embedding` initialization in model setup
2. Whether a separate optimizer (SignSGD) or the main optimizer handles puzzle embeddings
3. Exact embedding dimensions and how they're injected into the forward pass

**If the codebase shows AdamAtan2 (not SignSGD) for puzzle embeddings:** Use AdamAtan2 with 100× LR. The separate optimizer adds complexity for potentially no benefit.

**Corrected base.yaml (conservative):**
```yaml
puzzle_embed_dim: 512
puzzle_embed_lr_mult: 100           # 100x model LR for ARC tasks
puzzle_embed_init: "zeros"          # VERIFY from TRM codebase
puzzle_embed_optimizer: "same"      # Use same optimizer as model, with LR multiplier
                                    # VERIFY: TRM may use SignSGD separately
```

---

## Correction 6: SlotSSMs Use Mamba, Not S4

### What was assumed
Protocol v5.0 specified "per-slot S4 transitions" for the world model dynamics.

### Verified from paper
SlotSSMs (arXiv:2406.12272) uses **Mamba-based selective state space models**, not S4. The distinction matters:
- **S4:** Fixed parameterization, HiPPO initialization, frequency-domain computation
- **Mamba:** Input-dependent (selective) parameterization, hardware-aware scan, more expressive

The per-slot dynamics use block-diagonal transition matrices A_t = diag({A(s^k_t)}) where each slot has its own Mamba SSM transition, conditioned on that slot's current state.

### Corrected specification
```yaml
# base.yaml correction (Phase 3)
world_model:
  dynamics_type: "slot_mamba"         # CORRECTED: Mamba-based SSM, NOT S4
```

**Implementation note:** Use the `mamba-ssm` package (pip install mamba-ssm) or implement a simplified selective scan. The key feature is input-dependent A and B matrices per slot.

---

## Correction 7: SOLD Performance Claim

### What was assumed
Protocol stated SOLD "outperforms DreamerV3 by >2× on relational reasoning tasks."

### Verified from paper
The >2× advantage applies **specifically to "Distinct" task variants** in the Shapes2D/3D benchmarks — these require relational reasoning between objects (e.g., "push the red cube to the left of the blue sphere"). On "Specific" task variants (fixed target identification), SOLD only narrowly surpasses DreamerV3.

### Corrected protocol language
"SOLD outperforms DreamerV3 by >2× on relational reasoning task variants requiring inter-object reasoning, with smaller advantages on non-relational variants."

No implementation change needed — this is a documentation correction.

---

## Verified Specifications — Complete Phase 0 Reference

These specifications are verified and ready for Claude Code implementation:

### Model Architecture
```
Component              | Specification          | Source    | Confidence
-----------------------|------------------------|-----------|------------
Looped transformer     | 4-layer block, shared  | URM paper | ✅ VERIFIED
Attention heads        | 8                      | URM paper | ✅ VERIFIED
d_model                | 512                    | URM paper | ✅ VERIFIED
Head dimension         | 64 (= 512/8)           | Inferred  | HIGH
FFN type               | ConvSwiGLU             | URM paper | ✅ VERIFIED
FFN hidden dim         | 1024 (2× d_model)      | URM paper | ✅ VERIFIED
FFN conv kernel        | 1D, k=2, depthwise     | URM paper | ✅ VERIFIED
FFN activation         | SiLU (×2: gate + post-conv) | URM paper | ✅ VERIFIED
Normalization          | RMSNorm, post-norm     | URM/Loop-ViT | ✅ VERIFIED
Positional encoding    | RoPE                   | URM/Loop-ViT | ✅ VERIFIED
Loss                   | Cross-entropy          | Both papers | ✅ VERIFIED
Initial states         | Learned                | TRM paper  | HIGH
Input injection        | Every iteration        | TRM paper  | HIGH
Vocab size             | 12 (PAD+EOS+10 colors) | TRM paper  | ✅ VERIFIED
```

### Loop Structure
```
Component              | Specification          | Source    | Confidence
-----------------------|------------------------|-----------|------------
Inner loop iterations  | 8                      | URM paper | ✅ VERIFIED
TBPTL forward-only     | 2 iterations           | URM paper | ✅ VERIFIED
TBPTL backprop         | 6 iterations           | URM paper | ✅ VERIFIED
TBPTL loss             | Per-step CE, summed    | URM paper | ✅ VERIFIED
Outer ACT loop         | Up to 16 steps         | URM paper | ✅ VERIFIED
ACT halting            | Learned probability    | URM paper | HIGH
```

### Training
```
Component              | Specification          | Source    | Confidence
-----------------------|------------------------|-----------|------------
Optimizer              | AdamAtan2              | URM paper | ✅ VERIFIED
Model LR               | 1e-4 (ARC-AGI-1)      | URM paper | ✅ VERIFIED
Model LR (ARC-AGI-2)  | 3e-4                   | URM paper | ✅ VERIFIED
Weight decay           | 0.1                    | URM paper | ✅ VERIFIED
Betas                  | NOT SPECIFIED          | —         | ⚠️ UNKNOWN
LR schedule            | Constant after warmup  | TRM paper | HIGH
Gradient clipping      | 1.0                    | TRM paper | HIGH
Puzzle embed LR        | 1e-2 (100× model)     | URM paper | ✅ VERIFIED
EMA                    | Applied                | URM paper | ✅ VERIFIED
Mixed precision        | Assumed yes            | —         | REASONABLE
Max steps              | ~750K (ARC-AGI-1)      | TRM paper | HIGH
```

### Data
```
Component              | Specification          | Source    | Confidence
-----------------------|------------------------|-----------|------------
Training data          | ARC-AGI-1 + RE-ARC + BARC | Both papers | ✅ VERIFIED
Augmentation           | D4 (8 transforms) + color perm + translational | Both papers | ✅ VERIFIED
Effective augmentation | ~1000× per example     | TRM paper | HIGH
Grid canvas            | 30×30                  | ARC spec  | ✅ VERIFIED
Pass@K evaluation      | Pass@2                 | Both papers | ✅ VERIFIED
```

### Estimated Parameters (~12M)
```
Component              | Parameters (approx)
-----------------------|--------------------
Token embedding        | 12 × 512 = 6K
Positional (RoPE)      | 0 (computed, not stored)
Attention (4 layers)   | 4 × (4 × 512² + bias) ≈ 4.2M
ConvSwiGLU (4 layers)  | 4 × (512×1024 + 1024×2 + 1024×512) ≈ 4.2M
Layer norms            | ~16K
Output head            | 512 × 12 ≈ 6K
Puzzle embeddings      | ~400 × 512 ≈ 200K (varies with dataset)
ACT halting network    | ~100K
Initial states         | ~1K
TOTAL                  | ~8.7M (core) + embeddings ≈ ~9-12M
```

---

## ARC-AGI-3 Verified Specifications (Phase 3+)

### SDK
```
Package:        pip install arc-agi (current v0.9.1)
Depends on:     arcengine (auto-installed)
Main class:     arc_agi.Arcade
Env creation:   arc.make("ls20", render_mode="terminal")
Step:           obs = env.step(GameAction.ACTION1)
Action space:   env.action_space (varies per game)
```

### Environment
```
Grid:           64×64
Colors:         16
Actions:        7 (RESET, UP, DOWN, LEFT, RIGHT, INTERACT, CLICK(x,y))
                Note: UNDO (ACTION7) added in v0.9.2
                Not all actions available in every game
Observation:    FrameDataRaw (grid state, levels completed, etc.)
Scoring:        Action count vs. human baseline, per-level, mean across games
FPS:            Up to 2000 locally (without rendering)
```

### Competition Timeline
```
SDK released:     Jan 29, 2026
Full launch:      March 25, 2026
Rules announced:  March 25, 2026 (compute budget, hardware, submission format TBD)
Public games:     3 (LS20, VC33, FT09)
Full benchmark:   150+ environments, 1000+ levels
```

### Preview Competition Results
```
1st: StochasticGoose  | 12.58% | CNN-based RL
3rd: Graph Explorer    | ~12.58% | Zero-learning, graph enumeration
Frontier AI (GPT-5.2) | 0% at launch
```

---

## Required Updates to Bootstrap Files

### base.yaml changes
1. `loss_type: "stablemax_ce"` → `loss_type: "cross_entropy"`
2. `transformer_layers: 2` → `transformer_layers: 4`
3. ADD `act_enabled: true`, `act_max_steps: 16`, `act_halt_threshold: 0.99`
4. `ffn_hidden_mult: 4` → `ffn_hidden_mult: 2` (URM uses 2× expansion, not 4×)
5. `puzzle_embed_optimizer: "same"` (use main optimizer with LR multiplier, not separate SignSGD — to be verified from TRM codebase)
6. `world_model.dynamics_type: "slot_ssm"` → `"slot_mamba"`

### CLAUDE.md changes
1. Update Architecture Overview to reference 4-layer block
2. Add mention of outer ACT loop
3. Fix loss function reference
4. Fix SlotSSM → Mamba reference

### Protocol v5.0 changes
1. Section on URM: correct layer count and loop structure
2. Section on loss: remove Stablemax, specify standard CE
3. Section on SlotSSMs: Mamba not S4
4. Section on SOLD: qualify the >2× claim

---

## Remaining Unknowns (to verify from TRM codebase)

These items could not be fully verified from the papers and require checking the TRM source code at https://github.com/SamsungSAILMontreal/TinyRecursiveModels:

1. **Puzzle embedding initialization** — assumed zero-init, need confirmation
2. **Puzzle embedding optimizer** — assumed same as model (AdamAtan2 with 100× LR), may be SignSGD
3. **AdamAtan2 betas** — not specified in either paper, check codebase for defaults
4. **EMA decay rate** — confirmed EMA is used, decay rate not specified
5. **Warmup steps** — assumed 1000, need confirmation
6. **Batch size** — not specified in URM paper
7. **Input injection mechanism** — confirmed it happens, exact implementation (additive? concatenative?) needs verification

**Action:** Claude Code should verify items 1-7 from the TRM codebase as the FIRST sub-task of Phase 0, before writing any model code. This is a focused code-reading task (not open-ended research) with 7 specific questions to answer. Results should be logged in a session handoff note.

---

## Consequences

1. Phase 0 architecture uses 4-layer looped transformer block (not 2), increasing parameter count by ~4M but staying within the 12-15M target.
2. Standard cross-entropy loss simplifies implementation (no custom loss function needed).
3. The outer ACT loop adds meaningful adaptive computation but also implementation complexity. It should be built from the start but can be ablated.
4. ConvSwiGLU FFN expansion factor is 2× (not 4×), which actually reduces parameters per layer.
5. Phase 3+ world model should reference Mamba-based SSMs, not S4.
6. The 7 remaining unknowns are scoped as a focused codebase-reading task for Claude Code's first sub-task, with specific questions — not open-ended research.

---

## References

- URM: arXiv:2512.14693v2 — Gao et al., "Universal Reasoning Model" (2025)
- Loop-ViT: arXiv:2602.02156 — Shu et al., "LoopViT: Scaling Visual ARC with Looped Transformers" (2026)
- TRM: arXiv:2510.04871 — Jolicoeur-Martineau et al., "Less is More: Recursive Reasoning with Tiny Networks" (2025)
- SlotSSMs: arXiv:2406.12272 — Jiang et al., "Slot State Space Models" (NeurIPS 2024)
- SOLD: arXiv:2410.08822 — "Slot Object-Centric Latent Dynamics" (ICML 2025)
- Graph Explorer: arXiv:2512.24156 — "Graph-Based Exploration for ARC-AGI-3"
- ARC-AGI-3: https://arcprize.org/arc-agi/3/, https://github.com/arcprize/ARC-AGI
