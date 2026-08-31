# Preregistration

Protocol frozen for the first real model run. Infrastructure smoke outputs do not count
as outcomes.

## Research question

Under matched online GRPO, does an answer-gated minimal-provenance F1 reward reduce
recall-only cite-all behavior while preserving answer accuracy and improving joint
held-out program-composition performance?

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

## Hypothesis

The minimal reward will produce a positive primary contrast. The mechanism prediction
is lower cite-all/overcitation and higher evidence precision than the naive arm. Answer
EM is a guardrail.

## Fixed intervention

All three GRPO arms start from the exact same seed-specific SFT adapter and share:

- Qwen3-VL-2B-Instruct at revision
  `89644892e4d85e24eaac8bacfd4f463576704203`;
- train records, order, steps, optimizer, learning rate, seed, and checkpoint schedule;
- two stochastic completions per prompt at temperature 1.0;
- the same strict parser, eight-cell action space, and citation cap of eight;
- 4-bit NF4 double quantization, float16 compute, rank-8 `q_proj`/`v_proj` LoRA,
  frozen vision tower, SDPA, zero reference-model KL, and no vLLM.

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
- Mechanism floor: naive-recall must increase cite-all rate by at least 10 percentage
  points over SFT. Otherwise the proposed reward-hacking mechanism is unsupported.
- Report evidence precision, recall, F1 conditioned on correct answers, valid
  JSON/schema, overcitation, cite-all, attack success, and training-reward versus
  hidden-joint divergence.
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
3. Naive reward does not increase cite-all by at least 10 points versus SFT.
4. A leakage threshold is exceeded.
5. Online T4 throughput cannot support three naive/minimal seeds under the fixed
   budget.

All failures and incomplete runs will be reported. Hyperparameter changes after
outcome inspection create a new exploratory protocol.
