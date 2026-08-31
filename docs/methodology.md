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
Correct-answer cite-all receives full naive-recall reward and lower minimal F1 unless
all cells are genuinely required. The attack generator also emits oracle-minimal,
answer-only, citation-wrong, malformed, and cap-violation outputs.

Format failures and cap violations score zero in every arm. Both evidence arms gate on
answer exact match. This isolates recall versus precision rather than confounding the
comparison with different answer gating.

GRPO uses `beta=0`. With an active PEFT adapter, TRL's implicit reference path disables
that adapter and would compare against the base model rather than the shared SFT
initialization. Removing that KL term keeps the matched reward comparison from being
pulled toward a different policy.

## Hidden evaluation and statistics

Deterministic inference produces one row per method, seed, and question. Evaluation
rejects duplicate prediction rows. Training rollout diagnostics are retained for
reward-divergence plots but are never used as independent observations.

Evidence metrics are conditioned on answer correctness. Joint exact is the hidden
primary endpoint. Main paired inference first averages each question across the three
seeds, then bootstraps and randomizes question-level paired differences.

## Executable provenance diagnostic

The offline diagnostic perturbs cited and uncited source-table values, then reruns the
stored program. It asks whether a provenance set covers program dependencies. It
does not inspect VLM activations or establish causal use of image regions. That
distinction separates it from Evidence-RL's representation-level CED reward.
