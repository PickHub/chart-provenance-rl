# Limitations

- No real model training or evaluation has run. There is no empirical support yet for
  the hypothesis.
- The requested A10 is blocked. Azure ML exposes the SKU only for MIR, and raw quota
  is 8 of 36 required cores. The validated T4 has 16 GB and may fail the preregistered
  three-seed throughput criterion.
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
- The offline intervention diagnostic tests executable source-table dependencies, not
  VLM evidence dependence. Evidence-RL's CED addresses a different causal question.
- Deterministic JSON repair can improve format validity but cannot establish better
  reasoning. Raw and repaired evaluations are reported separately.
- QLoRA, a frozen vision tower, two completions, and 100-step default budgets are
  resource-driven. Negative findings may be specific to this small regime.
- Renderer-family statistics can identify style. The audit prevents style overlap in
  the holdout, but it cannot prove absence of every visual artifact.
- The question-only baseline is a cheap template-majority probe, not a competitive
  language-only model.
- External transfer is deferred until the core matched comparison succeeds.
