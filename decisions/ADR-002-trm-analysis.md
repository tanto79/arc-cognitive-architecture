# ADR-002: TRM Codebase Analysis

**Status:** Accepted
**Date:** 2026-02-10
**Context:** Sub-task 0.4 — Study TRM (TinyRecursiveModels) reference implementation to inform Phase 0 backbone design.
**Source:** github.com/SamsungSAILMontreal/TinyRecursiveModels, cloned to `C:/Projects/TRM-reference`

---

## 1. Recursive Refinement Implementation

### Two-Level Hierarchy (z_H, z_L)

TRM maintains two latent states of **equal resolution**:
- `z_H` — high-level state, shape `[batch, seq_len + puzzle_emb_len, hidden_size]`
- `z_L` — low-level state, shape `[batch, seq_len + puzzle_emb_len, hidden_size]`

**Critical finding:** Both z_H and z_L are the **same spatial resolution**. Our protocol described z_L as 4× downsampled (`seq_len//4`). This is incorrect — TRM uses full resolution for both.

### Iteration Structure

The refinement loop is nested:

```
for H_step in range(H_cycles):        # outer: default 3
    for L_step in range(L_cycles):     # inner: default 6
        z_L = L_level(z_L, z_H + input_embeddings)
    z_H = L_level(z_H, z_L)
```

Total forward passes of L_level per inference: `H_cycles * (L_cycles + 1)` = 3 × 7 = **21 passes**.

### Weight Sharing

The **same** `L_level` module (2 transformer layers) is used for both:
- z_L updates: `L_level(z_L, z_H + input_embeddings)`
- z_H updates: `L_level(z_H, z_L)`

This is a single `TinyRecursiveReasoningModel_ACTV1ReasoningModule` containing 2 `TinyRecursiveReasoningModel_ACTV1Block` layers. Each block has: non-causal self-attention → post-norm → SwiGLU FFN → post-norm.

### Input Injection

Input embeddings are injected at **every** L_level call for z_L via addition to z_H:
```python
z_L = L_level(z_L, z_H + input_embeddings)  # input_injection = z_H + input_embeddings
```

The reasoning module adds input_injection to hidden_states before the transformer layers:
```python
def forward(self, hidden_states, input_injection, **kwargs):
    hidden_states = hidden_states + input_injection
    for layer in self.layers:
        hidden_states = layer(hidden_states=hidden_states, **kwargs)
    return hidden_states
```

z_H updates do NOT receive input embeddings directly — only z_L:
```python
z_H = L_level(z_H, z_L)  # input_injection = z_L only
```

### Initial States

z_H and z_L are initialized from **learned vectors** (not zeros):
```python
self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(hidden_size), std=1))
self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(hidden_size), std=1))
```

These are broadcast across batch and sequence dimensions via `torch.where(reset_flag, self.H_init, carry.z_H)`.

---

## 2. Deep Supervision / Gradient Flow

### NOT True Deep Supervision

**Critical finding:** TRM does NOT compute loss at every refinement step. The first `H_cycles - 1` outer iterations run under `torch.no_grad()`. Only the **last** outer iteration has gradient:

```python
# H_cycles-1 without grad
with torch.no_grad():
    for _H_step in range(self.config.H_cycles-1):
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings)
        z_H = self.L_level(z_H, z_L)
# 1 with grad
for _L_step in range(self.config.L_cycles):
    z_L = self.L_level(z_L, z_H + input_embeddings)
z_H = self.L_level(z_H, z_L)
```

This means gradient flows through only **7 L_level passes** (6 z_L updates + 1 z_H update), not all 21. The first 14 passes function as a "warm-up" that evolves the latent states without training signal.

### Implication for Our Design

Our protocol's "deep supervised loss at every step" is not what TRM does. We should either:
1. Replicate TRM's approach (no-grad warm-up + grad on last cycle) as the baseline
2. Experiment with true deep supervision as an enhancement AFTER matching TRM's baseline

---

## 3. Loss Function

### Stablemax Cross-Entropy

TRM uses a custom `stablemax_cross_entropy` loss instead of standard softmax CE:
```python
# Stablemax: numerically stable alternative to softmax
# Replaces exp() with (x - max(x))^2 normalization
```

### Loss Composition

Total loss = `lm_loss + 0.5 * (q_halt_loss + q_continue_loss)`

- **lm_loss**: Stablemax CE over token predictions, per-example normalized by valid (non-ignored) token count
- **q_halt_loss**: BCE loss — predicts whether the current output is fully correct (binary: all tokens match)
- **q_continue_loss**: BCE loss for the continue action's Q-value (when `no_ACT_continue=True`, this is simplified to just using `q_halt_logits > 0` as the halt criterion)

### Label Handling

- `IGNORE_LABEL_ID = -100` (standard PyTorch convention)
- Per-example loss normalization: each example's loss is divided by its number of valid tokens, then averaged across the batch
- This prevents long sequences from dominating the loss

---

## 4. Data Augmentation Pipeline

### Dihedral Group (8 transforms)
Identity, rot90, rot180, rot270, flip_lr, flip_ud, transpose, anti-diagonal transpose.

### Color Permutation
- 9! / (9-k)! permutations of non-black colors (black/color 0 is never permuted)
- Combined with 8 dihedral transforms ≈ 1000 augmentations per puzzle

### Translational Augmentation
Grids are placed at **random positions** within the 30×30 padded grid, not always top-left aligned. This teaches position-invariance.

### Grid Encoding
- All grids padded to 30×30 → flattened to 900 tokens
- Each input-output example is a pair of 900-token sequences
- **Vocab: 12 tokens** — PAD=0, EOS=1, digits 2-11 (representing colors 0-9)
- EOS markers placed at grid row boundaries

### Pre-processing
Data is pre-processed into `.npy` files with ~1000 augmentations per puzzle, stored as memory-mapped arrays. Puzzles are grouped (all augmentations of one puzzle = one group) for batch sampling.

---

## 5. Vocabulary Difference

| | TRM | Our Protocol |
|---|---|---|
| Colors 0-9 | Tokens 2-11 | Tokens 0-9 |
| Padding | Token 0 | Token 10 |
| EOS | Token 1 | Not used |
| Total vocab | 12 | 11 |

**Decision needed:** Whether to add EOS tokens. TRM uses them as grid-row delimiters. Our current GridEncoder uses 11 tokens (0-9 colors + 10 padding). We may need to add EOS (token 11) for TRM fidelity.

---

## 6. Model Architecture Details

### Hyperparameters (from config)

| Parameter | TRM Value | Our base.yaml |
|---|---|---|
| hidden_size | 512 | 512 |
| num_heads | 8 | 8 |
| expansion | 4.0 | 4.0 |
| H_cycles | 3 | — |
| L_cycles | 6 | — |
| L_layers | 2 | 2 |
| halt_max_steps | 16 | 16 (max_refinement_steps) |
| pos_encodings | rope | rope |
| rope_theta | 10000 | 10000 |
| rms_norm_eps | 1e-5 | 1e-5 |
| forward_dtype | bfloat16 | bfloat16 |
| puzzle_emb_len | 16 | — |
| puzzle_emb_ndim | 4096 | — |
| seq_len | 900 | 900 |

### Post-Norm (not Pre-Norm)

TRM blocks use **post-normalization**:
```python
hidden_states = rms_norm(hidden_states + self.self_attn(...), eps)
hidden_states = rms_norm(hidden_states + self.mlp(...), eps)
```

This differs from the more common pre-norm used in modern transformers (LLaMA, etc.). Our base.yaml specifies `norm_type: rmsnorm` but does not specify pre/post. We should use post-norm to match TRM.

### RMSNorm Without Learnable Parameters

TRM's `rms_norm` is a plain function with no learnable scale/bias:
```python
def rms_norm(hidden_states, variance_epsilon):
    variance = hidden_states.float().square().mean(-1, keepdim=True)
    return (hidden_states * torch.rsqrt(variance + variance_epsilon)).to(input_dtype)
```

### No Bias

All linear layers use `bias=False` except the Q-head (which has bias initialized to -5 for conservative initial halting).

---

## 7. Optimizer and Training

| Parameter | TRM Value | Our base.yaml |
|---|---|---|
| Optimizer | AdamATan2 | AdamW |
| Learning rate | 1e-4 | 1e-3 |
| LR schedule | Warmup → constant | Cosine |
| lr_min_ratio | 1.0 (no decay) | — |
| Weight decay | 0.1 | 0.01 |
| Beta1 | 0.9 | 0.9 |
| Beta2 | 0.95 | 0.999 |
| Global batch size | 768 | 64 |
| Puzzle emb LR | 1e-2 | — |
| Puzzle emb optimizer | SignSGD | — |

**Key differences:**
1. **AdamATan2 vs AdamW**: AdamATan2 replaces the division by sqrt(v) with atan2(m, v), providing more stable updates. We should start with AdamW but note this as a potential improvement.
2. **LR 10× lower**: TRM uses 1e-4, we specified 1e-3. With larger batch size this may be appropriate (linear scaling rule), but we should verify.
3. **Constant LR after warmup**: TRM does NOT use cosine decay — lr_min_ratio=1.0 means the LR stays at 1e-4 forever after warmup.
4. **Beta2=0.95**: Much lower than the standard 0.999, closer to settings used for LLM training.
5. **Batch size 768**: 12× our planned 64. TRM uses global batching across GPUs.

---

## 8. Puzzle Embeddings (TTT Mechanism)

TRM's "test-time training" is NOT LoRA adapters. It uses **per-puzzle learned embeddings**:

- Each puzzle gets a learned embedding vector of dimension `puzzle_emb_ndim=4096`
- This is reshaped into `puzzle_emb_len=16` tokens of `hidden_size=512` dimensions
- These 16 tokens are **prepended** to the input sequence
- During training, puzzle embeddings are updated via `CastedSparseEmbeddingSignSGD_Distributed` — a custom distributed SignSGD optimizer with lr=1e-2 and weight_decay=1e-2
- The embedding table is stored as a non-parameter buffer (nn.Buffer) — not part of the standard optimizer
- At each forward pass, relevant embeddings are copied to a local buffer with `requires_grad=True`, gradients flow back, and SignSGD updates the master copy

**Implication:** Our protocol describes TTT as "per-task LoRA with All-Outputs loss." This is fundamentally different from TRM's approach. For Phase 0 fidelity, we should implement puzzle embeddings. LoRA-based TTT can be explored as an enhancement in later phases.

---

## 9. Evaluation Pipeline

### Augmentation Voting

During evaluation:
1. Apply all 8 dihedral transforms to input
2. Run model on each augmented input
3. Inverse-transform predictions back to original orientation
4. Hash each predicted grid
5. Vote: weight each prediction by `sigmoid(q_halt_logits)` (confidence)
6. Select the grid with highest weighted vote count

### Fixed Steps at Evaluation

During evaluation, ACT always runs for `halt_max_steps` (16) iterations regardless of Q-values. This ensures consistent batching (all examples in a batch take the same number of steps).

### Pass@K

Evaluation supports pass@K — select top-K distinct predictions by confidence and check if any matches.

---

## 10. Parameter Count

TRM's total parameter count with default config:
- `L_level`: 2 transformer blocks, each with:
  - Attention: QKV proj (512 → 1536) + O proj (512 → 512) = ~1.05M
  - SwiGLU FFN: gate_up (512 → 2×1536) + down (1536 → 512) = ~2.36M
  - Per block: ~3.4M
- 2 blocks × 3.4M = **~6.8M** for the shared reasoning module
- Token embedding: 12 × 512 = 6K
- LM head: 512 × 12 = 6K
- Q head: 512 × 2 + 2 = ~1K
- RoPE: non-learnable
- H_init, L_init: 1K
- **Total (excl. puzzle embeddings): ~7M parameters**

Puzzle embeddings add `num_puzzles × 4096` parameters but these are sparse and separate.

---

## 11. Decisions for Phase 0 Backbone

Based on this analysis, the following adjustments to our base.yaml / backbone design are recommended:

### Must Match TRM (for faithful reproduction)
1. **z_L same resolution as z_H** — not 4× downsampled
2. **Weight sharing** — single L_level module for both z_L and z_H updates
3. **Post-norm** — add-then-normalize, not pre-norm
4. **No-grad warm-up** — H_cycles-1 iterations without gradient
5. **Input injection** — add input embeddings to z_H at every z_L update step
6. **Learned initial states** — H_init, L_init buffers, not zeros
7. **RMSNorm without learnable params** — functional, not nn.Module with scale

### Should Match TRM (strong recommendation)
8. **Vocab 12 with EOS** — add EOS token for grid row boundaries
9. **Puzzle embeddings** as TTT mechanism (not LoRA) for Phase 0
10. **LR=1e-4** with warmup → constant schedule (not cosine)
11. **Weight decay=0.1**, beta2=0.95

### Can Diverge (acceptable differences)
12. **AdamW instead of AdamATan2** — simpler, well-understood, can upgrade later
13. **Batch size 64** — limited by single GPU, compensate with gradient accumulation
14. **Standard softmax CE** — can switch to stablemax if training is unstable
15. **No ACT for initial Phase 0** — fixed halt_max_steps, add ACT as enhancement

---

## 12. Config Updates Required

```yaml
# Changes to configs/base.yaml for TRM fidelity:
model:
  num_colors: 12          # was 11 — add EOS token
  padding_idx: 0          # was 10 — match TRM convention
  eos_idx: 1              # new
  color_offset: 2         # new — colors 0-9 map to tokens 2-11
  H_cycles: 3             # new
  L_cycles: 6             # new
  norm_position: post     # clarify post-norm
  multi_scale: false      # z_L is NOT downsampled
  puzzle_emb_ndim: 4096   # new
  puzzle_emb_len: 16      # new

training:
  optimizer: adamw
  lr: 1e-4                # was 1e-3
  lr_schedule: constant   # was cosine
  weight_decay: 0.1       # was 0.01
  beta2: 0.95             # was 0.999
  puzzle_emb_lr: 1e-2     # new
```

---

## References

- TRM paper: "Tiny Recursive Models" (Samsung AI Lab Montreal)
- ACT: "Adaptive Computation Time for Recurrent Neural Networks" (Graves, 2016)
- PQN: "Parallelized Q-Networks" (arXiv:2407.04811)
- AdamATan2: Custom optimizer variant using atan2 instead of sqrt for second moment
