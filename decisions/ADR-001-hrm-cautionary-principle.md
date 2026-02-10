ADR-001: HRM Cautionary Principle — Binding Methodological Commitment
Date: 2026-02-07
Status: ACCEPTED (binding)
Context: Protocol v3.0+ / v4.0
Decision
No cognitive principle will be claimed as contributing to performance unless BOTH forward
and reverse ablation demonstrate statistically significant improvement with that principle
present.
Context
HRM (Wang et al., arXiv:2506.21734) built a 27M-parameter model with two coupled recurrent
modules — a slow abstract planner and a fast detailed worker — inspired by hierarchical
cortical processing. The authors claimed the brain-inspired hierarchy was central to
performance, achieving 40.3% on ARC-AGI-1.
Independent analysis by the ARC Prize Foundation and a separate paper (arXiv:2510.00355,
"Hierarchical Reasoning Models: Perspectives and Misconceptions") demonstrated that an
8-layer single-module network performs comparably. The hierarchical structure contributed
minimally — deep supervision and test-time training, not the brain-inspired architecture,
drove performance. HRM's headline cognitive claim was functionally debunked.
Consequences
Every claim about a cognitive principle's contribution must be backed by ablation evidence,
not architectural narrative.
Forward ablation (build progressively) AND reverse ablation (remove from full) must agree.
If a principle doesn't contribute, we report that finding honestly in the paper.
In competition context: non-contributing principles are redesigned or removed, not just
documented as null results.
This principle protects us from the "#1 anticipated reviewer concern" for multi-component
architectures — the "kitchen sink" critique.
References
arXiv:2510.00355 — "Hierarchical Reasoning Models: Perspectives and Misconceptions"
arXiv:2506.21734 — Original HRM paper
Protocol v4.0, Section 2.8 and Section 10.3
