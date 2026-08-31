# Model card

## Base model

The only preregistered model is
[`Qwen/Qwen3-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct):

```text
revision: 89644892e4d85e24eaac8bacfd4f463576704203
license: Apache-2.0
library: transformers
pipeline: image-text-to-text
```

These fields were verified against the live Hugging Face API. Azure preparation
downloads that exact snapshot and writes SHA-256 and size metadata for every model
file.

## Planned derivatives

The pipeline can produce:

- one completion-only SFT adapter per seed;
- answer-only, naive-recall, and minimal-provenance online-GRPO adapters initialized
  from that exact SFT adapter.

No adapter is published and no run result exists yet.

## Adaptation

- 4-bit NF4 with double quantization;
- float16 compute;
- LoRA rank 8, alpha 32, dropout 0.1;
- language `q_proj` and `v_proj` targets;
- frozen Qwen3-VL vision tower;
- SDPA, no FlashAttention 2;
- no vLLM;
- zero reference-model KL during GRPO;
- two GRPO completions, maximum 256 completion tokens.

TRL 0.26.2 internally casts QLoRA adapter parameters to bfloat16. The training entry
points restore them to float16 after trainer construction and assert the resulting
dtype before optimizer creation.

The implementation is sized for a 16 GB T4 fallback. The originally requested 24 GB
A10 is unavailable in the workspace.

## Intended use

Adapters are research artifacts for comparing deterministic provenance rewards on the
included synthetic chart benchmark. They are not intended for factual decision
support, accessibility, finance, medical use, or arbitrary visual documents.

## Evaluation

The primary endpoint is held-out-composition joint exact pass. Required secondary
metrics and statistical procedures are fixed in
[`docs/preregistration.md`](docs/preregistration.md).

## Limitations

Results, once available, apply only to the pinned model, adapter regime, synthetic
generator, and compute budget. Explicit citation does not prove internal visual
grounding. See [`docs/limitations.md`](docs/limitations.md).

## License

Base and adapter use remains subject to the Qwen Apache-2.0 license. Code in this
repository is Apache-2.0.
