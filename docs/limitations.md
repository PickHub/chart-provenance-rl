# Limitations

- The confirmatory hypothesis failed. Same-data SFT saturated the atomic online-training
  prompts, and the naive and minimal rewards never differed on 9,600 sampled rollouts.
  The run cannot determine whether the reward definitions would diverge under genuine
  citation uncertainty.
- The requested A10 is blocked. Azure ML exposes the SKU only for MIR, and raw quota
  is 8 of 36 required cores. All nine required training runs completed on a 16 GB T4.
- The charts are synthetic, small, value-labeled, and English-only. They do not cover
  natural chart noise, arbitrary fonts, complex legends, accessibility issues, or
  external chart domains.
- The primary OOD claim is only held-out program composition. Renderer and layout
  sets are stress tests, not independent broad generalization claims.
- Program-defined minimal provenance is a benchmark convention. Argmax and count
  require the full compared series because proving absence matters. Other reasonable
  user notions of sufficient evidence may differ.
- Evidence exact match operates on visible semantic labels. It does not evaluate
  bounding boxes, attention, image-region causality, explanation quality, or whether
  the model internally relied on cited entities.
- The minimal reward and hidden joint endpoint share the same program-defined
  provenance labels. Their agreement tests reward specification on this benchmark, not
  independent causal visual grounding.
- The offline intervention diagnostic tests executable source-table dependencies, not
  VLM evidence dependence. Evidence-RL's CED addresses a different causal question.
- Deterministic JSON repair can improve format validity but cannot establish better
  reasoning. Raw and repaired evaluations are reported separately.
- QLoRA, a frozen vision tower, two completions, and 100-step default budgets are
  resource-driven. Negative findings may be specific to this small regime.
- Azure child durations include time queued behind other jobs on the single-node
  cluster. They are completion evidence, not pure trainer throughput.
- Renderer-family statistics can identify style. The audit prevents style overlap in
  the holdout, but it cannot prove absence of every visual artifact.
- The question-only baseline is a cheap template-majority probe, not a competitive
  language-only model.
- External transfer was not run because the core matched comparison failed.
