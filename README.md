# Cognitive Architecture for Abstract Reasoning
## Research Protocol & Design Documentation

**Principal Investigator:** Tim Antonson, PURE Cognitive LLC  
**Research Partner:** Claude Opus 4.6 (Anthropic)  
**Date:** February 2026  
**Status:** Pre-implementation — Architecture design complete, implementation beginning

---

## What This Is

This repository contains the **design documentation** for a cognitive architecture that implements human cognitive principles as native architectural features in a small neural network (25–34M parameters) targeting the [ARC-AGI](https://arcprize.org/) abstract reasoning benchmarks, including ARC-AGI-3's interactive environments where frontier AI currently scores 0%.

This is **not** the implementation code. This repository exists to publicly timestamp our research hypotheses, architecture design, analysis plan, and methodological commitments **before** empirical results are obtained — serving the same scientific purpose as pre-registration while allowing the iterative flexibility that systems engineering research requires.

## The Research Question

> Can a neural architecture implementing validated human cognitive principles — at 25–34M parameters — outperform systems with 100–1000× more parameters on tasks requiring genuine abstraction, win the ARC-AGI-3 competition, achieve prize-competitive performance on ARC-AGI-2, and claim the $700K ARC-AGI-2 grand prize?

## Architecture Overview

The architecture is a **Predictive Coding Looped Transformer (PCLT)** — a looped transformer substrate where prediction error serves as the unified signal driving learning, exploration, adaptive computation, and anomaly detection. The substrate is informed by [Loop-ViT](https://arxiv.org/abs/2602.02156) (65.8% ARC-AGI-1 at 18M params) and [URM](https://arxiv.org/abs/2512.14693) (53.8% at 14M params), with an interactive agent layer for ARC-AGI-3.

All eight cognitive principles are native to the architecture rather than bolted on:

| # | Principle | PCLT Implementation |
|---|-----------|-------------------|
| 1 | Feedback-dominant processing | Loop iteration IS recurrent feedback |
| 2 | Predictive coding | Prediction error between iterations IS PC loss |
| 3 | Sparse coding | k-Winners-Take-All retains top 5% of activations |
| 4 | Parallel streams | Slot Attention decomposes input into independent object slots |
| 5 | Working memory bottleneck | 4–8 capacity-limited competitive attention slots |
| 6 | Multiple memory types | Working (slots) + Episode (DND) + Rule (abstractions) |
| 7 | Dual-process reasoning | Few iterations = System 1, many = System 2; policy MLP vs MCTS |
| 8 | Metabolic efficiency | Entropy-based dynamic exit halts when prediction error converges |

### Why PCLT (not standard recursive transformer)

The previous protocol (v4.0–v4.1) used a TRM-style recursive transformer with cognitive modules added sequentially. Three problems forced a complete redesign:

1. **TC⁰ ceiling:** Fixed-depth transformers provably cannot solve linear equations, graph connectivity, or FSM simulation — core ARC reasoning patterns. Looped transformers escape this ceiling via depth scaling.
2. **Bolt-on vs native:** Adding cognitive modules to a standard transformer made them optimization afterthoughts. In PCLT, the loop IS feedback, prediction error IS predictive coding, slots ARE working memory.
3. **Agent-as-afterthought:** ARC-AGI-3 is an interactive environment where frontier AI (GPT-5.2, Claude 4.0) scores 0%. The agent cannot be crammed into the final phases — it must be the architecture.

This pivot is documented in `decisions/ADR-004-pclt-architecture-pivot.md`.

### Dual-Track Build

The build proceeds in two tracks validated on separate benchmarks:

- **Track A (Phases 0–2): Substrate Development** — Validated on ARC-AGI-1/2. Must beat URM's 53.8%, target Loop-ViT's 65.8%. Substrate LOCKED after Phase 2.
- **Track B (Phases 3–5): Agent Development** — Validated on ARC-AGI-3 interactive environments. World model (Mamba-based SlotSSMs), graph exploration, dual-process planning, episodic + rule memory.
- **Track C (Phase 6): Competition Optimization** — Ablation, optimization, submission.

### Verified Implementation Specifications (ADR-005)

Before implementation, every specification was verified against the actual URM, Loop-ViT, and ARC-AGI-3 SDK codebases. Seven discrepancies were found and corrected, including: loss function (standard CE, not Stablemax), Think block layer count (4, not 2), missing outer ACT loop, FFN expansion ratio (2×, not 4×), and SlotSSM dynamics type (Mamba, not S4). All corrections are documented in `decisions/ADR-005-reference-architecture-verification.md`.

## Research Hypotheses (stated before results)

**H1:** The full PCLT substrate will exceed Loop-ViT's 65.8% on ARC-AGI-1 AND exceed 25% on ARC-AGI-2, establishing new state-of-the-art at the 25M parameter scale.

**H3:** Prediction error as unified signal (learning + exploration + halting + anomaly detection) will outperform architectures using separate mechanisms for each function.

**H5:** Metacognitive monitoring (entropy-based halting) will produce computation depth that correlates with task difficulty (measured by human solve time).

**H8:** The full architecture will exceed the sum of individual principle contributions (synergy > sum of parts) due to native integration.

**H9:** The complete PCLT agent will win or place top-3 on the ARC-AGI-3 leaderboard.

**H10:** The cognitive architecture will outperform frontier models at 30,000× parameter efficiency on ARC-AGI-3 environments.

See `protocol_v5_1.md` for the complete set of 10 hypotheses and falsification criteria.

## Methodology: HRM Cautionary Principle

Following the debunking of the Hierarchical Reasoning Model's cognitive claims (arXiv:2510.00355), we adopt a strict methodological commitment: **no cognitive principle is claimed as contributing without both forward AND reverse ablation evidence.** Forward ablation (add principle to baseline) and reverse ablation (remove principle from full architecture) must both show statistically significant effects. This is documented in `decisions/ADR-001-hrm-cautionary-principle.md`.

## Architecture Evolution Trail

This research design went through a significant pivot, documented in the ADR (Architecture Decision Record) trail:

1. **ADR-001** (HRM Cautionary Principle): Established the ablation methodology after HRM's cautionary example
2. **ADR-004** (PCLT Architecture Pivot): Retired the TRM-based approach due to TC⁰ ceiling, bolt-on principles, and agent-as-afterthought problems. Adopted PCLT with prediction error as unified signal.
3. **ADR-005** (Reference Architecture Verification): Pre-implementation verification against actual codebases found 7 discrepancies and corrected all specifications before writing any code.

The previous ADR-002 (TRM codebase analysis) and ADR-003 (TRM/URM corrections) are superseded by ADR-004 and ADR-005 but remain in the repository for historical completeness.

## Documents in This Repository

| Document | Description |
|----------|-------------|
| `protocol_v5_1.md` | Complete research protocol: competitive landscape, PCLT architecture specification, training methodology, evaluation framework, analysis plan, publication strategy |
| `PHASES.md` | All 7 build phases with gate criteria across 3 tracks |
| `decisions/ADR-001-hrm-cautionary-principle.md` | Methodological commitment to ablation-backed claims |
| `decisions/ADR-004-pclt-architecture-pivot.md` | Why we pivoted from TRM to PCLT architecture |
| `decisions/ADR-005-reference-architecture-verification.md` | Pre-implementation verification — 7 corrections from actual codebases |

## Target Benchmarks

- **ARC-AGI-1:** Substrate calibration. Phase 0 gate ≥53.8% (match URM). Target: exceed Loop-ViT's 65.8%.
- **ARC-AGI-2:** Substrate validation + prize target. Phase 2 gate ≥15%. Target: >25%. Stretch: >85% ($700K grand prize).
- **ARC-AGI-3:** Primary competition target (launches March 25, 2026). Interactive environments requiring sequential exploration and rule discovery where frontier AI scores 0%. Target: 1st place.

## Planned Publication

Primary venue: **ICLR 2027** (submissions ~October 2026). The progressive ablation methodology and competition results constitute the paper regardless of competition placement. A dual-track build with separate substrate and agent validation provides publishable results even if individual components underperform.

## License

Documentation in this repository is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Implementation code (in a separate repository) will be released under MIT License upon publication.

---

*This repository was created to publicly timestamp research design decisions before empirical results are obtained. Git commit history documents the evolution of the research protocol, including the pivot from TRM-based to PCLT-based architecture (ADR-004) and pre-implementation verification (ADR-005).*
