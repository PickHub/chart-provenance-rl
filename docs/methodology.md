# Methodology

## Generator

Each example starts with a typed `TableData`: four visible category labels, two visible
series labels, and finite `Decimal` values. The generator renders that table as a bar,
line, or table-like PNG using Pillow. Numeric values are printed visibly so questions
do not depend on inaccessible pixel coordinates. Pillow's bundled Aileron Regular
font is loaded with `ImageFont.load_default`.

Images are at most 512 by 384 pixels. PNG bytes determine the image filename. Canonical
table JSON determines the table hash. Canonical record JSON determines the record ID.
Semantic sidecars use the record hash as their filename; filenames contain no labels,
answers, operations, or positions.

Program, chart type, renderer family, title, category-label family, series-label
family, and target-series position use separate deterministic balanced schedules.
Audits emit contingency tables and require every sufficiently represented program to
span every nuisance-factor level. COUNT thresholds are balanced to produce answers 1,
2, and 3 rather than a constant count.

Each text element is drawn on a known background and retains a padded bounding box,
foreground/background colors, WCAG contrast ratio, role, and semantic cell reference
for values. Generation rejects placements below 4.5 contrast or without a non-clipped,
non-overlapping position. Dataset audit verifies exactly eight uniquely referenced
visible value labels plus all categories and series legends.

## Executable programs

| Program | Answer | Minimal provenance |
| --- | --- | --- |
| Lookup | one cell value | selected cell; duplicate witnesses may provide alternatives |
| Comparison | yes/no | both compared cells |
| Argmax | winning category label(s) | all cells in the compared series |
| Count | integer | all cells in the counted series |
| Difference | left lookup minus right lookup | both operands |
| Sum selected | numeric sum | selected operands |

The evaluator recomputes every answer and provenance set from the table/program pair.
For duplicates or equivalent witnesses it chooses the best valid minimal set, never a
single arbitrary annotation.

## Model boundary

Model input is built from only the chart image, system output contract, and natural
language question. The training data adapter explicitly selects these fields. Reward
metadata remains in separate dataset columns passed to the deterministic TRL reward
callable.

## Splits

The primary OOD axis is program composition. Difference and sum-selected are absent
from training. Style/layout, duplicate values, label shuffles, and order perturbations
are secondary stress tests rather than additional co-primary OOD claims.

Split-specific grammar and table configuration seeds are fixed in
`generation/config.py`. Audits check record, table, and image hash intersections.
Repeated generic wording is allowed and reported; it carries no table value or answer.

## Reward and attacks

The eight-cell evidence universe and cap of eight deliberately permit cite-all.
Correct-answer cite-all and an exact minimal set receive the same naive-recall reward,
while cite-all receives lower minimal F1 unless all cells are genuinely required. The
naive reward therefore permits coverage hedging rather than deterministically causing
it. The attack generator also emits oracle-minimal, answer-only, citation-wrong,
malformed, and cap-violation outputs.

Format failures and cap violations score zero in every arm. Both evidence arms gate on
answer exact match. This isolates recall versus precision rather than confounding the
comparison with different answer gating.

The answer-only ablation is `schema_valid * answer_exact`: it removes evidence quality
credit but retains the shared parser, citation cap, and action space.

GRPO uses `beta=0`. With an active PEFT adapter, TRL's implicit reference path disables
that adapter and would compare against the base model rather than the shared SFT
initialization. Removing that KL term keeps the matched reward comparison from being
pulled toward a different policy.

TRL 0.26.2's original `grpo` sequence normalization creates a completion-length bias,
which would confound an intervention that changes citation count. Every arm therefore
uses its supported `dr_grpo` loss with `scale_rewards="none"`. This remains online
group-relative policy optimization while removing length and group-difficulty scaling
from the reward comparison.

## Hidden evaluation and statistics

Deterministic inference produces one row per method, seed, and question. Evaluation
rejects duplicate prediction rows. Training rollout diagnostics are retained for
reward-divergence plots but are never used as independent observations.

Prediction rows carry the dataset-manifest hash plus adapter stage, seed, reward arm,
source-SFT adapter hash, run-manifest hash, adapter config/weight artifact hash, and
image mode. Scoring validates those fields against the named method. Confirmatory
analysis requires all methods to match the audited 48 held-out record IDs, verifies
that naive/minimal adapters originate from the corresponding SFT weights, and verifies
that neutral/shuffled probes use the exact same minimal weights as normal inference.

Rollout diagnostics average completions within each training question before averaging
questions within arm and seed. The resulting train-reward/hidden-joint divergence is
a mechanism diagnostic only; raw rollout rows never enter question-level inference.
On resume, diagnostics at or beyond the restored checkpoint are discarded before new
rows append. Confirmatory analysis requires exact arm/seed groups, all audited training
question IDs, and at least 100 committed global steps per group.

Evidence metrics are conditioned on answer correctness. Joint exact is the hidden
primary endpoint. Main paired inference first averages each question across the three
seeds, then bootstraps and randomizes question-level paired differences.

## Executable provenance diagnostic

The offline diagnostic perturbs cited and uncited source-table values, then reruns the
stored program. It asks whether a provenance set covers program dependencies. It
does not inspect VLM activations or establish causal use of image regions. That
distinction separates it from Evidence-RL's representation-level CED reward.

The minimal reward and hidden joint endpoint use the same program-defined provenance
labels. The endpoint measures controlled reward misspecification, not an independent
grounding construct.

Held-out minimal-GRPO inference runs in three image modes: normal, neutral white image,
and a deterministic derangement of held-out images. Neutral and shuffled answer EM are
the actual model leakage probes; the template-majority question-only baseline is only
descriptive.
