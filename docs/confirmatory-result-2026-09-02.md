# Confirmatory result: the reward intervention did not activate

The preregistered hypothesis failed. Minimal-provenance GRPO did not outperform
naive-recall GRPO or SFT on held-out program compositions. The more important
diagnostic is that the two reward definitions never disagreed on any sampled training
rollout, so this run did not create a behavioral intervention capable of testing the
proposed mechanism.

## Result

Qwen3-VL-2B-Instruct was trained with 100 SFT steps followed by 100 online
group-relative steps for each reward arm on seeds 17, 29, and 43. All nine required SFT,
naive-recall, and minimal-provenance runs completed on one 16 GB T4.

| Method | Answer EM | Joint exact | Valid JSON | Cite-all |
| --- | ---: | ---: | ---: | ---: |
| Prompted frozen | 0.00% | 0.00% | 0.00% | 0.00% |
| SFT | 2.78% | 2.78% | 100.00% | 0.00% |
| SFT + JSON repair | 2.78% | 2.78% | 100.00% | 0.00% |
| Answer-only GRPO | 2.78% | 2.78% | 100.00% | 0.00% |
| Naive-recall GRPO | 2.78% | 2.78% | 100.00% | 0.00% |
| Minimal-provenance GRPO | 2.78% | 2.78% | 100.00% | 0.00% |

The primary minimal-minus-naive joint-exact contrast was `0.0000`, with a paired
question-level bootstrap 95% interval of `[0.0000, 0.0000]` and a two-sided
sign-randomization `p = 1.0`.

Per-seed joint exact was identical for SFT, naive, and minimal:

| Seed | SFT | Naive recall | Minimal provenance |
| ---: | ---: | ---: | ---: |
| 17 | 2.08% | 2.08% | 2.08% |
| 29 | 4.17% | 4.17% | 4.17% |
| 43 | 2.08% | 2.08% | 2.08% |

## Manipulation check

The online training distribution was saturated before GRPO:

- all 96 global-step-zero rollouts were hidden-joint exact;
- mean hidden joint exact across atomic training rollouts was 99.79%;
- zero of 9,600 rollout rows contained overcitation;
- zero rollout rows received different naive-recall and minimal-F1 rewards;
- only 10 of 2,400 two-completion groups per arm had nonzero reward variance, 0.42%;
- the varying-group counts were 2, 3, and 5 of 800 for seeds 17, 29, and 43,
  identically in both arms.

The two objectives therefore supplied the same sampled reward signal. Adapter files are
not byte-identical and a few generated strings differ, but every preregistered scored
metric is identical between SFT, naive, and minimal. This is not evidence that
minimality rewards are ineffective. It is evidence that this experiment failed to
activate the distinction it intended to test.

## Generalization failure

The SFT policy learned the output contract and nearly solved the 96 atomic prompts used
again for online training. It did not generalize to the 48 held-out difference and
sum-selected questions:

| Held-out program | Correct / seed-question pairs | Answer EM |
| --- | ---: | ---: |
| Difference | 4 / 72 | 5.56% |
| Sum selected | 0 / 72 | 0.00% |

All correct held-out answers also had exact evidence, but there were only four correct
seed-question pairs. Evidence precision and recall conditioned on those four cases
should not be generalized.

## Controls and integrity

- neutral-image answer EM: 0.00%;
- shuffled-image answer EM: 0.00%;
- unshuffled minimal-model answer EM: 2.78%;
- cross-split record, image, and table hash overlap: zero;
- answer leakage: zero;
- all required lineage-bound 100-step T4 runs: complete;
- source commit used for training:
  `22889199a62835ea2adc5bfee5abf72d628e1002`;
- pinned model revision:
  `89644892e4d85e24eaac8bacfd4f463576704203`.

The blank and shuffled controls pass their preregistered thresholds, but the unshuffled
accuracy is too low to support a strong visual-dependence claim.

## Interpretation

This study demonstrates a reproducible single-GPU VLM post-training stack, strict
reward and artifact lineage, online group-relative optimization, and a fail-closed
analysis. It does not demonstrate reward-hacking mitigation or improved visual
reasoning.

The negative result identifies two coupled design failures:

1. SFT and GRPO reused the same 96 atomic prompts, leaving almost no exploration signal.
2. The primary test required unseen arithmetic compositions that none of the online
   reward arms practiced.

Azure child durations in the throughput artifact include queue time because three
pipelines shared one node. They prove feasibility and completion, not pure trainer
throughput.

## Follow-up

A new exploratory protocol should:

1. use disjoint SFT and GRPO prompt sets;
2. validate reward-definition disagreement on a training-only pilot before evaluating;
3. increase rollout diversity without supervising cite-all behavior;
4. reserve new, uninspected generator seeds for held-out composition evaluation;
5. retain matched initialization, parser, optimizer, and budgets across reward arms.

That follow-up would test whether recall-only reward induces coverage hedging under
genuine uncertainty. Reusing the current held-out outcomes as a new confirmatory test
would not be valid.

## Evidence

The passage-free and image-free result bundle is in
[`artifacts/releases/2026-09-02`](../artifacts/releases/2026-09-02/README.md).
`confirmatory-report.json` is the authoritative preregistered gate output.
