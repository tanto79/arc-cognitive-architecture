# ADR-004: Architecture Pivot from TRM-Informed Recursive Transformer to PCLT Agent

**Date:** February 11, 2026
**Status:** Accepted
**Supersedes:** Protocol v4.0–v4.1 architecture (TRM-informed recursive transformer with bolt-on cognitive modules)
**Decision:** Replace the TRM-informed recursive transformer core with a Predictive Coding Looped Transformer (PCLT) substrate and build an interactive agent layer for ARC-AGI-3.

---

## Context

Protocol v4.0–v4.1 specified a TRM-style recursive transformer (~7–15M parameters) as the backbone, with eight cognitive principles added as separate modules across 11 phases (0–10). The agent layer for ARC-AGI-3 was treated as the final phase (Phase 10). Claude Code implementation had reached Phase 0 (baseline) but no further.

On February 10–11, 2026, two comprehensive architecture research sessions were conducted:

1. **Research 1 (Feb 10):** Evaluated whether the transformer core should be replaced with a biologically-native architecture. Analyzed eight candidate architectures (PCNs, Active Predictive Coding, Energy-Based Transformers, DEQs, Mamba/SSMs, SlotSSMs, Hymba, Looped Transformers) mapped against eight biological principles.

2. **Research 2 (Feb 11):** Evaluated how to win ARC-AGI-3 as an interactive agent. Analyzed world model architectures (DreamerV3, SOLD, SlotSSMs), exploration strategies (graph-based, prediction-error curiosity, NovelD, AXIOM), planning approaches (MCTS, dual-process), and memory systems (DND, ArcMemo, CAVIA).

---

## Decision Rationale

### Three fatal problems with the v4.1 approach

**1. The TC⁰ ceiling makes fixed-depth transformers provably insufficient.**

Merrill & Sabharwal (TACL 2023, ACL 2025) proved that fixed-depth transformers are contained in TC⁰ complexity class — provably unable to solve linear equations, graph connectivity, Horn-clause satisfiability, or simulate finite-state machines. This is not a practical limitation but a mathematical proof. SSMs including Mamba share this ceiling (Merrill et al. 2024).

The escape route is depth that grows with input size. Looped transformers achieve this because effective depth scales with iteration count, which adapts at inference time. This is precisely what TRM, URM, and Loop-ViT already do — but the v4.1 protocol did not frame it as the TC⁰ escape or leverage it architecturally.

**2. Biological principles were bolt-ons, not native.**

The v4.1 approach added cognitive modules to a standard transformer backbone:
- Slot attention added as a separate encoder (Phase 3)
- Predictive coding added as a separate perceptual loop (Phase 6)
- Sparse coding not addressed architecturally
- Dynamic halting added as a separate metacognitive module (Phase 4)

This created seven documented mismatches between transformer computation and biological processing: feedforward vs. feedback-dominant, no capacity constraints vs. constraints-as-features, conflated vs. separated memory, full processing vs. predictive coding, dense vs. sparse coding, single mechanism vs. parallel streams, no resource constraint vs. metabolic constraint.

The PCLT makes these principles native to the substrate itself:
- The loop IS feedback-dominant processing
- Prediction error IS predictive coding
- k-Winners-Take-All IS sparse coding
- Slot attention IS working memory with capacity constraints
- Entropy-based halting IS metabolic efficiency
- Variable loop depth IS dual-process reasoning

**3. ARC-AGI-3 demands an agent, not a grid predictor with an afterthought wrapper.**

The v4.1 protocol devoted 10 of 11 phases to building a static grid predictor, with the agent layer crammed into Phase 10. ARC-AGI-3 is fundamentally different from ARC-AGI-1/2:
- Interactive environments requiring exploration and rule discovery
- 7 discrete actions, not grid-to-grid prediction
- Scoring on action efficiency vs. humans, not prediction accuracy
- 1000+ levels across 150+ environments requiring memory and transfer
- 2000 FPS real-time constraint
- Frontier AI scores 0% — this is not a scale problem

The agent architecture (world model, exploration, planning, memory) IS the architecture. It cannot be an afterthought.

### Empirical evidence supporting the pivot

**Looped architectures dominate small-model ARC performance:**
- Loop-ViT (18M, looped conv + self-attention): 65.8% on ARC-AGI-1 — current small-model SOTA
- URM (14M, enhanced recursive transformer): 53.8%
- TRM (7M, recursive transformer): 45%

**Object-centric world models outperform monolithic ones on relational reasoning:**
- SOLD (slot-based dynamics) outperforms DreamerV3 by >2x on relational manipulation tasks
- SlotSSMs handle 2560+ timestep sequences with linear complexity

**Graph-based exploration already matches ARC-AGI-3 preview winners:**
- Training-free graph explorer solved 30/52 levels, matching StochasticGoose's 12.58% after bug fix
- Every LLM-based agent scored 0%

**Prediction error as exploration signal eliminates the noisy TV problem:**
- In deterministic grid environments (which ARC-AGI-3 uses), prediction error = pure epistemic uncertainty
- No separate curiosity module needed

---

## What Changes

| Aspect | v4.1 (Retired) | v5.0 (New) |
|---|---|---|
| Core substrate | TRM-style recursive transformer | PCLT (Predictive Coding Looped Transformer) |
| Bio principles | Bolt-on modules added across 11 phases | Native to substrate architecture |
| Agent layer | Phase 10 afterthought | Co-equal development track (Phases 3–5) |
| Phase count | 11 (0–10) | 7 (0–6) in two tracks |
| Unified signal | None — separate modules for each function | Prediction error drives learning, exploration, halting, anomaly detection |
| Validation strategy | ARC-AGI-1 only as primary | Dual-track: ARC-AGI-1/2 for substrate, ARC-AGI-3 for agent |
| Parameter budget | 7–50M | 25–50M |
| Primary competition target | ARC-AGI-3 (aspirational) | ARC-AGI-3 (primary, architecture designed for it) |
| Substrate baselines | TRM 45% | TRM 45%, URM 53.8%, Loop-ViT 65.8% |
| Agent baselines | None | StochasticGoose 12.58%, graph-based 30/52 levels |

## What Is Preserved

- **HRM Cautionary Principle (ADR-001):** Binding. No claim without forward AND reverse ablation evidence.
- **Progressive build and ablation methodology:** Each phase measures marginal contribution.
- **Competition-focused mindset:** Win first, publish the ablation data.
- **Fair comparison principles:** Compute-matched, data-identical, multiple seeds.
- **1-week maximum per component:** Skip non-contributing principles after debug/redesign.
- **ARC-AGI-1/2 as calibration benchmarks:** Substrate must beat established baselines before agent development.
- **Eight cognitive principles:** Same principles, now natively implemented rather than bolted on.

## What Is Archived

- Protocol v4.0 and v4.1
- Claude Code Playbook v2.0, v2.1, v2.2
- ADR-002 (TRM codebase analysis) — findings incorporated into v5.0 but TRM is now a baseline, not the core
- ADR-003 (TRM/URM corrections) — corrections incorporated into v5.0 baselines
- All Phase 0 Claude Code implementation work from v4.1

## Risks of This Pivot

| Risk | Mitigation |
|---|---|
| PCLT substrate is unproven on ARC tasks | Loop-ViT validates looped architecture at 65.8%. Phase 0 gate requires matching URM before proceeding. |
| Starting over loses Phase 0 implementation progress | Phase 0 was baseline only. Clean start avoids carrying forward architectural assumptions that no longer apply. |
| More complex architecture may be harder to debug | Simpler phase structure (7 vs. 11). Each component independently validated in literature. |
| Agent layer adds scope | Agent IS the competition entry. Without it, we cannot compete on ARC-AGI-3. |

---

## Decision

Accepted. Protocol v4.1 is archived. Protocol v5.0 governs all future development. Claude Code implementation restarts from Phase 0 with PCLT substrate.

**Signed:** Tim Antonson, Principal Investigator
**Date:** February 11, 2026
