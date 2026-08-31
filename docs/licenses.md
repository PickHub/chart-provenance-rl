# License and provenance

## Artifact

Source code and documentation are licensed under Apache License 2.0 in
[`LICENSE`](../LICENSE).

Deterministically generated synthetic tables, chart images, sidecars, and benchmark
manifests produced by this repository are dedicated under CC0-1.0 as described in
[`DATA_LICENSE.md`](../DATA_LICENSE.md). They contain no collected personal data.

## Model

`Qwen/Qwen3-VL-2B-Instruct` is used only at immutable revision
`89644892e4d85e24eaac8bacfd4f463576704203`. Its model card reports Apache-2.0. The
artifact does not redistribute model weights in Git.

## Rendering

Pillow is permissively licensed under HPND. `ImageFont.load_default(size=...)` in
Pillow 12.3.0 loads its bundled subset of Aileron Regular when FreeType is available.
Aileron is licensed under SIL Open Font License 1.1. No font file is copied into this
repository or fetched by the generator.

## Training stack

Transformers, TRL, PEFT, Accelerate, and Qwen3-VL are Apache-2.0 projects. PyTorch uses
BSD-style terms. Azure SDK packages use MIT terms. Each dependency retains its own
license; see the exact versions in `uv.lock` and `azure/environment/conda.yml`.

Users publishing adapters, generated data, or result bundles must preserve upstream
model and dependency notices as applicable.
