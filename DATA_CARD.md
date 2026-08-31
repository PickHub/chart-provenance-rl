# Data card

## Summary

The benchmark is generated from scratch. It contains typed synthetic tables,
deterministically rendered chart images, natural-language questions, executable
programs, exact answers, and one or more valid minimal semantic provenance sets.

No web data, personal data, documents, receipts, screenshots, or human annotations are
collected.

## Schema

Each record includes hidden evaluator fields:

- schema and record version;
- split, chart type, renderer family, and generator seeds;
- relative content-addressed image path and SHA-256;
- canonical typed source table and table SHA-256;
- executable program;
- canonical answer;
- valid minimal provenance sets.

Model adapters expose only the image, system output contract, and question.

## Composition

The default build contains:

| Split | Count | Purpose |
| --- | ---: | --- |
| train | 96 | atomic programs |
| validation | 24 | atomic development checks |
| test_composition | 48 | held-out difference and selected-sum programs |
| each stress split | 24 | renderer, duplicate, label, order, or citation probe |

All charts have four categories and two series. Values are finite decimals. Images are
bar, line, or table-like PNGs no larger than 512 by 384.

## Generation and reproducibility

`chart-provenance generate` uses fixed split-specific seeds. Canonical JSON and PNG
bytes are hashed. Running the same pinned environment produces byte-identical
manifests and images. `chart-provenance audit` checks sidecars, executable labels,
content hashes, split overlap, prompt leakage, filenames, and split design.

The generator uses Pillow 12.3.0 and its bundled Aileron Regular subset. It does not
download a font.

## Intended use

The data is for controlled research on answer-plus-provenance reward definitions,
evaluation, and chart program composition. It is not intended to estimate broad chart
understanding or production reliability.

## Risks

Synthetic regularity may create shortcuts. Stress splits and cheap leakage probes
reduce but do not eliminate that risk. Value labels simplify visual extraction. The
small evidence universe intentionally makes cite-all possible and is not representative
of long-document citation tasks.

## License

Generated synthetic data is CC0-1.0. See [DATA_LICENSE.md](DATA_LICENSE.md).
