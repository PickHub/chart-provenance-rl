# Preregistration

Protocol frozen for the first real model run. Infrastructure smoke outputs do not count
as outcomes.

## Research question

Under matched online group-relative training, does an answer-gated
minimal-provenance F1 reward resolve the recall-only indifference between minimal and
overcomplete evidence while preserving answer accuracy and improving joint held-out
program-composition performance?

## Confirmatory contrast and endpoint

The single primary endpoint is held-out-composition joint exact pass: the answer must
match exactly and predicted evidence must equal one valid minimal provenance set.

The confirmatory contrast is:

```text
GRPO minimal-provenance minus GRPO naive-recall
```

For each question, average joint pass over seeds 17, 29, and 43. Compute the mean
paired question-level difference, a 95% paired bootstrap interval with 10,000
resamples, and a two-sided paired sign-randomization test with 10,000 draws. Rollout
rows are not independent samples.

The primary criterion passes only when the mean contrast is positive, the 95%
bootstrap lower bound is above zero, and the two-sided randomization p-value is below
0.05.

## Hypothesis

The minimal reward will produce a positive primary contrast. Recall-only reward permits
full-recall supersets to tie with exact minimal evidence, making coverage hedging
risk-dominant under citation uncertainty. The mechanism prediction is at least a
10-point higher naive cite-all rate than the matched minimal arm and greater naive
train-reward/hidden-joint divergence. Naive change from SFT is descriptive. Answer EM
is a guardrail.

## Fixed intervention

All three GRPO arms start from the exact same seed-specific SFT adapter and share:

- Qwen3-VL-2B-Instruct at revision
  `89644892e4d85e24eaac8bacfd4f463576704203`;
- train records, order, steps, optimizer, learning rate, seed, and checkpoint schedule;
- two stochastic completions per prompt at temperature 1.0;
- the same strict parser, eight-cell action space, and citation cap of eight;
- 4-bit NF4 double quantization, float16 compute, rank-8 `q_proj`/`v_proj` LoRA,
  frozen vision tower, SDPA, Dr.GRPO sequence normalization, unscaled group rewards,
  zero reference-model KL, and no vLLM.

Rewards are:

```text
answer-only = EM
naive       = EM * (0.5 + 0.5 * best-set recall)
minimal     = EM * (0.5 + 0.5 * best-set F1)
```

F1 is the primary minimality intervention. No extra overcitation subtraction is used.

## Data and splits

The default generator produces 96 atomic train questions, 24 atomic validation
questions, 48 primary held-out composition questions, and 24 questions in each
secondary stress split. The primary OOD split contains only difference and
sum-selected programs. Training contains lookup, comparison, argmax, and count.
Grammar/config seeds, source table hashes, image hashes, and record hashes are
disjoint across splits.

Within each sufficiently sized split, generator audits require program kinds to span
chart, renderer, category-family, series-family, target-series-position, and title
levels. COUNT answers must include 1, 2, and 3. Render audits require eight visible
semantic values with non-clipped, non-overlapping labels and contrast of at least 4.5.

Secondary analyses include renderer holdout, duplicate decoys, label shuffles, order
perturbations, answer-correct/citation-wrong probes, image shuffling, a question-only
baseline, and cite-all attacks.

## Baselines

Run prompted frozen, SFT, SFT with deterministic JSON repair, answer-only GRPO,
naive-recall GRPO, and minimal-provenance GRPO. Main three-seed inference is required
for naive versus minimal. If T4 throughput cannot support all secondary arms at three
seeds, answer-only is secondary and may be reduced only after recording that decision;
naive versus minimal remains three-seed.

## Guardrails and mechanism checks

- Answer harm margin: minimal-reward answer EM may be at most 3 percentage points
  below naive-recall and SFT on held-out composition.
- Mechanism floor: naive-recall cite-all rate must be at least 10 percentage points
  above minimal-provenance, and naive train-reward/hidden-joint divergence must exceed
  minimal divergence. Naive versus SFT change is descriptive.
- Report evidence precision, recall, F1 conditioned on correct answers, valid
  JSON/schema, overcitation, cite-all, attack success, and training-reward versus
  hidden-joint divergence.
- Average rollout completions within training question before arm/seed summaries;
  rollout rows remain non-independent diagnostics.
- The executable provenance-set necessity/sufficiency diagnostic is descriptive. It
  does not test model evidence dependence.

## Leakage threshold

Stop confirmatory interpretation if any of these occur:

- question-only answer EM exceeds 10% on held-out composition;
- shuffled-image answer EM exceeds 25% or retains more than half of unshuffled answer
  EM;
- any table, image, or record content hash crosses train/held-out splits;
- an answer, program field, hidden provenance annotation/ID, or content hash appears
  in a model prompt or filename. Visible labels named by the natural-language question
  are allowed.

Renderer-family classification is reported as an artifact diagnostic but is not itself
label leakage because renderer family is not an answer target.

## Prespecified failure criteria

The primary hypothesis fails if any of the following hold:

1. Minimal-reward GRPO does not beat SFT on joint held-out composition performance.
2. Citation improvement comes with answer EM harm beyond the 3-point margin.
3. Naive cite-all is not at least 10 points above minimal, or naive
   train-reward/hidden-joint divergence does not exceed minimal.
4. A leakage threshold is exceeded.
5. Online T4 throughput cannot support three naive/minimal seeds under the fixed
   budget.

All failures and incomplete runs will be reported. Hyperparameter changes after
outcome inspection create a new exploratory protocol.
