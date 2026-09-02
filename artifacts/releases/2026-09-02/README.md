# Confirmatory result bundle

This bundle contains no chart images, model weights, prompts, answers, or generated
completion text. Record IDs and artifact hashes preserve lineage without publishing
example content.

The preregistered hypothesis failed. See
[`docs/confirmatory-result-2026-09-02.md`](../../../docs/confirmatory-result-2026-09-02.md).

## Files

- `confirmatory-report.json`: authoritative fail-closed gate output.
- `t4-throughput.json`: Azure parent/child job, compute, timing, source, dataset, run,
  and adapter lineage.
- `dataset-audit.json`: split, leakage, factor-balance, renderer, and content-hash audit.
- `*.question-metrics.jsonl`: question-level metrics for each method or probe across
  seeds 17, 29, and 43.
- `*.rollout-diagnostics.jsonl`: non-independent training diagnostics for the two
  confirmatory reward arms.
- `manipulation-check.json`: post-hoc descriptive summary of reward variance and
  reward-definition activation.
- `SHA256SUMS`: digest manifest for the release files.

Rollout rows are diagnostics, not independent statistical samples. Question is the
inferential unit, averaged across seeds before the paired bootstrap and randomization
test.
