# Cognitive Architecture for Abstract Reasoning
## Research Protocol & Design Documentation

**Principal Investigator:** Tim Antonson, PURE Cognitive LLC  
**Research Partner:** Claude Opus 4.6 (Anthropic)  
**Date:** February 2026  
**Status:** Pre-implementation — Architecture design complete, implementation beginning

---

## What This Is

This repository contains the **design documentation** for a cognitive architecture that implements human cognitive principles as architectural features in a small neural network (7–50M parameters) targeting the [ARC-AGI](https://arcprize.org/) abstract reasoning benchmarks.

This is **not** the implementation code. This repository exists to publicly timestamp our research hypotheses, architecture design, analysis plan, and methodological commitments **before** empirical results are obtained — serving the same scientific purpose as pre-registration while allowing the iterative flexibility that systems engineering research requires.

## The Research Question

> Can a neural architecture implementing validated human cognitive principles — at 7–50M parameters — outperform systems with 100–1000× more parameters on tasks requiring genuine abstraction?

## Architecture Overview

The architecture implements eight cognitive principles derived from cross-modal neuroscience research (biological vision and auditory processing converge on the same computational strategies), built on a recursive transformer core informed by the [Tiny Recursive Model](https://arxiv.org/abs/2510.04871) (TRM) and [Universal Reasoning Model](https://arxiv.org/abs/2512.14693) (URM):

| Phase | Principle | Cognitive Science Basis |
|-------|-----------|------------------------|
| 0 | Recursive Transformer Core | TRM-style iterative refinement |
| 0.5 | TBPTL + Real TTT | URM gradient strategy + test-time adaptation |
| 1 | Dual-Process Reasoning (P5) | Kahneman; validated as most impactful pattern in ARC research |
| 2 | Multiple Memory Types (P3) | Tulving's episodic/semantic distinction |
| 3 | Working Memory Bottleneck (P7) | Slot attention; capacity-limited processing |
| 4 | Metacognitive Monitoring (P8) | Confidence estimation + adaptive computation |
| 5 | Global Workspace (P1) | Baars/Dehaene; cross-module coordination |
| 6 | Biological Perception (V1-V3) | Hierarchical + predictive coding |
| 7 | Structural Transfer (P6) | Gentner; relational binding + equivariances |
| 8 | Staged Learning (P2) | Critical periods; progressive freezing |
| 9 | Emotional Valuation (P4) | Damasio; learned resource allocation |
| 10 | Full Integration + Agent | All principles + ARC-AGI-3 interactive agent |

Each principle is added progressively with forward and reverse ablation to measure its individual contribution — ensuring every architectural claim is backed by empirical evidence.

## Research Hypotheses (stated before results)

**H1:** The full cognitive architecture will statistically outperform a matched-parameter baseline lacking cognitive organization.

**H3:** Dual-process reasoning (P5) will be the single largest contributor to performance.

**H5:** Metacognitive monitoring will produce better-calibrated confidence estimates than implicit confidence from the baseline.

**H8:** The full architecture will exceed the sum of individual principle contributions (synergy > sum of parts).

**H9:** The architecture will achieve non-zero performance on ARC-AGI-3 interactive environments.

**H10:** Cognitive features will amplify the effectiveness of test-time adaptation.

See `protocol_v4_1.md` for the complete set of hypotheses and falsification criteria.

## Methodology: HRM Cautionary Principle

Following the debunking of the Hierarchical Reasoning Model's cognitive claims (arXiv:2510.00355), we adopt a strict methodological commitment: **no cognitive principle is claimed as contributing without both forward AND reverse ablation evidence.** Forward ablation (add principle to baseline) and reverse ablation (remove principle from full architecture) must both show statistically significant effects. This is documented in `ADR-001-hrm-cautionary-principle.md`.

## Key Design Corrections (ADR-002 & ADR-003)

During implementation preparation, systematic analysis of the TRM codebase revealed 15 discrepancies between the published paper's descriptions and the actual implementation. These corrections, plus adoption of innovations from the URM (which achieves 53.8% vs TRM's 40% on ARC-AGI-1), are documented in ADR-002 and ADR-003. Key corrections include: puzzle embeddings (not LoRA) for task conditioning, final-cycle-only gradients (not deep supervision at every step), and same-resolution latent states (not multi-scale).

## Documents in This Repository

| Document | Description |
|----------|-------------|
| `protocol_v4_1.md` | Complete research protocol: competitive landscape, architecture specification, training methodology, evaluation framework, analysis plan, publication strategy |
| `PHASES.md` | All 11 build phases with gate criteria and test specifications |
| `decisions/ADR-001-hrm-cautionary-principle.md` | Methodological commitment to ablation-backed claims |
| `decisions/ADR-002-trm-analysis.md` | TRM codebase analysis — 15 corrections to paper assumptions |
| `decisions/ADR-003-trm-urm-corrections.md` | Master corrections document + URM adoption decisions |

## Target Benchmarks

- **ARC-AGI-1:** Calibration benchmark. Phase 0 target ≥40%, Phase 0.5 target ≥50%.
- **ARC-AGI-2:** Secondary benchmark. Target >10% at $0.20/task.
- **ARC-AGI-3:** Primary competition target (launches March 25, 2026). Interactive environments where frontier AI currently scores 0%.

## Planned Publication

Primary venue: **ICLR 2027** (submissions ~October 2026). The progressive ablation methodology and competition results constitute the paper regardless of competition placement.

## License

Documentation in this repository is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Implementation code (in a separate repository) will be released under MIT License upon publication.

---

*This repository was created to publicly timestamp research design decisions before empirical results are obtained. Git commit history documents the evolution of the research protocol.*
