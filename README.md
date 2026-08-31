# Chart provenance reward misspecification

Can a recall-only evidence reward distinguish minimal provenance from a coverage
hedge that cites every chart entity? This artifact provides a controlled benchmark and
matched online group-relative experiment for measuring that reward indifference. It
generates charts from typed tables and executable programs, asks for an answer plus
visible semantic evidence, and compares recall with minimal-provenance F1.

The intended result is a controlled comparison of reward definitions, not a generic
chart demo, independent causal visual-grounding evidence, or a claim that a model has
already improved.

## Status

| Layer | Status |
| --- | --- |
| Deterministic bar, line, and table-like chart generator | Implemented |
| Executable programs, minimal provenance sets, OOD and stress splits | Implemented |
| Hard JSON parser, three reward arms, attacks, metrics, and statistics | Implemented |
| Qwen3-VL QLoRA SFT, online GRPO, inference, and checkpoint resume | Implemented, model construction still requires GPU validation |
| Azure ML v2 components and one-seed pipeline | SDK schema-validated |
| Real SFT/GRPO runs and scientific results | **Not run** |

The CPU smoke output is synthetic evaluator validation. It is not model evidence.

## Quick start

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --frozen --group dev
uv run chart-provenance smoke
uv run pytest
```

Generate the full deterministic benchmark:

```bash
uv run chart-provenance generate --output data/generated
uv run chart-provenance audit --dataset data/generated
```

Generated release datasets and result bundles should be placed under
`artifacts/releases/`. That path is deliberately not ignored.

## Task and output contract

Each prompt contains one chart image, a question, and the output contract. Source
tables, programs, answers, minimal-provenance annotations, record hashes, and split
metadata are never included in model input. Visible series/category labels may appear
in the question because they identify chart entities.

```json
{"answer":"42","evidence":[{"series":"Revenue","category":"Q3"}]}
```

Evidence names human-visible series and category labels. It never uses hidden IDs,
coordinates, or filenames. The parser rejects prose, malformed or duplicate-key JSON,
extra fields, wrong types, unknown chart labels, repeated normalized citations, and
more than eight raw citations. Values are canonicalized with `Decimal`; evidence is
whitespace-normalized and deduplicated before duplicate rejection.

Each chart has exactly eight visible semantic cells. The common citation cap is also
eight, so cite-all remains available to every reward arm. Valid minimal sets usually
contain one to three cells; comparison and aggregate programs can require more.
Questions with duplicate values can store multiple valid minimal sets.

## Reward intervention

All GRPO arms share the model, SFT initialization, data, parser, citation cap, rollout
count, temperature, and optimization budget. Only the scalar reward changes.

Let `EM` be answer exact match and score evidence against the best valid minimal set:

```text
answer-only = EM
naive       = EM * (0.5 + 0.5 * recall)
minimal     = EM * (0.5 + 0.5 * F1)
```

Malformed output and citation-cap violations receive zero. Both evidence rewards are
answer-gated and lie in `[0, 1]`. The naive arm omits precision, so an exact minimal set
and every full-recall superset tie at reward 1.0. It therefore permits and can reinforce
coverage hedging under citation uncertainty, but does not by itself force cite-all.
The minimal arm resolves that indifference through F1 precision. A separate
overcitation subtraction is not used because it would change reward scale too.

## Benchmark design

Atomic training programs are lookup, comparison, argmax, and count. Difference of two
lookups and sum over selected categories are held out as the primary program-
composition OOD split. Secondary stress splits cover:

- renderer family holdout;
- duplicate-value decoys and alternative witnesses;
- label shuffles;
- category and series order perturbations;
- answer-correct/citation-wrong attacks.

All images, source tables, records, and manifests are content-addressed or hashed.
Train and held-out splits use distinct grammar/config seeds. Audits verify executable
answers, sidecars, content hashes, disjoint table/image content, prompt/filename
leakage, held-out program families, renderer families, nuisance-factor contingency
tables, answer entropy, and duplicate behavior. Every rendered label retains its
bounding box, foreground/background colors, and contrast ratio; generation fails if a
label clips or overlaps another label.

See [methodology](docs/methodology.md), the
[machine-readable experiment matrix](configs/experiment_matrix.yml), and the
[preregistration](docs/preregistration.md).

## Evaluation

The question is the statistical unit. Rollout rows are never treated as independent
examples. Report:

- answer exact match;
- evidence precision, recall, and F1 conditioned on a correct answer;
- joint exact pass;
- valid JSON and valid schema rates;
- overcitation and cite-all rates;
- cite-all attack success;
- training reward versus hidden joint/precision divergence.

Main comparisons average each question across three seeds before a paired
question-level bootstrap confidence interval and sign randomization test.
`chart-provenance confirm` consumes SFT, naive, minimal, neutral-image, shuffled-image,
dataset-audit, throughput, and rollout artifacts. It reports every preregistered
criterion and cannot return `pass` when an artifact is missing or malformed. Dataset
and adapter-weight hashes, run-manifest hashes, seed, reward arm, source SFT, and image
mode prevent relabeling one artifact as another.

The executable provenance-set necessity/sufficiency diagnostic perturbs source-table
values. It checks program dependencies only. It does not show that the VLM causally
used an image region.

## Model and training

Actual training is restricted to
[`Qwen/Qwen3-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) at
revision `89644892e4d85e24eaac8bacfd4f463576704203` (Apache-2.0). The Hugging Face API
was checked for the exact SHA, license, `transformers` library, and
`image-text-to-text` task.

The implementation uses 4-bit NF4 double quantization, float16 compute, rank-8 LoRA on
`q_proj` and `v_proj`, a frozen vision tower, SDPA, images no larger than 512 pixels,
two completions per prompt, length-unbiased Dr.GRPO loss with unscaled group rewards,
zero reference-model KL, and no vLLM. TRL `0.26.2` is pinned; its upstream tag
commit `8c26b7d0e5f5ddc0ddea0d6b519ab30e437b8731` contains the Qwen3-VL QLoRA GRPO
notebook used to lock the API.

TRL 0.26.2 converts trainable QLoRA parameters to bfloat16 inside both trainer
constructors. T4 does not provide native bfloat16, so this artifact converts trainable
adapter parameters back to float16 after trainer construction and fails if the dtype
check does not hold.

Qwen3-VL's processor cap is set through
`image_processor.size["longest_edge"]`, not an inert `max_pixels` attribute. Model
smoke validates the resulting patch grid against the 512 by 384 pixel budget.

Run model construction separately when GPU dependencies and weights are present:

```bash
uv sync --frozen --extra gpu
uv run --extra gpu chart-provenance model-smoke
```

See [MODEL_CARD.md](MODEL_CARD.md).

## Azure

The requested `Standard_NV36ads_A10_v5` is not currently usable. The live Azure ML
size API exposes it as `MIR` only, not `AmlCompute`. The raw-VM quota request was
partially fulfilled at 8 of the required 36 family cores. The sole validated path is
the existing scale-to-zero `gpu-t4-dedicated` cluster
(`Standard_NC4as_T4_v3`, 16 GB).

No A10 cluster asset is included. The preflight fails if T4 loses AmlCompute support
or if A10 availability changes, forcing an explicit protocol review.

See [exact Azure commands and blockers](docs/azure.md).

## Related work and claim boundary

[Evidence-RL](https://arxiv.org/abs/2608.08021) is the closest prior work. It uses
counterfactual evidence disentanglement during GRPO to reward answer dependence on
target image regions. This project does not claim counterfactual evidence dependence
or generic evidence-grounded RL as new. It studies explicit semantic provenance
outputs, recall-only cite-all incentives, minimal-set precision, and held-out chart
program composition.

Citation-aware deep-search RL
([CaRR/C-GRPO](https://aclanthology.org/2026.acl-long.950/)), evidence-aware radiology
RL ([ESC-RL/GEAR](https://aclanthology.org/2026.acl-long.1718/)),
[RewardMap](https://arxiv.org/abs/2510.02240), and graph-reasoning
[MuSe](https://aclanthology.org/2026.acl-long.476/) are adjacent. Precision-aware
evidence rewards themselves are not claimed as novel. See
[related work](docs/related-work.md).

## License and limitations

Code is Apache-2.0. Deterministically generated synthetic datasets are released under
CC0-1.0. Model weights retain the Qwen Apache-2.0 license. Pillow's bundled Aileron
Regular font is used through `ImageFont.load_default`; Aileron is SIL OFL-1.1.

Read [DATA_CARD.md](DATA_CARD.md), [license notes](docs/licenses.md), and
[limitations](docs/limitations.md) before using or citing the artifact.
