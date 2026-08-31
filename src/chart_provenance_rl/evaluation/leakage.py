from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean

from PIL import Image, ImageStat

from chart_provenance_rl.schema import BenchmarkRecord


def question_only_predictions(
    train_records: list[BenchmarkRecord],
    evaluation_records: list[BenchmarkRecord],
) -> dict[str, str]:
    template_answers: dict[str, Counter[str]] = defaultdict(Counter)
    global_answers: Counter[str] = Counter()
    for record in train_records:
        template_answers[_template_key(record)][record.answer] += 1
        global_answers[record.answer] += 1
    if not global_answers:
        raise ValueError("Question-only baseline requires training records")
    global_answer = global_answers.most_common(1)[0][0]
    return {
        record.record_id: (
            template_answers[_template_key(record)].most_common(1)[0][0]
            if template_answers[_template_key(record)]
            else global_answer
        )
        for record in evaluation_records
    }


def image_shuffle_map(records: list[BenchmarkRecord], *, shift: int = 1) -> dict[str, str]:
    if len(records) < 2:
        raise ValueError("Image shuffle requires at least two records")
    normalized_shift = shift % len(records)
    if normalized_shift == 0:
        raise ValueError("Image shuffle must be a derangement")
    image_paths = [record.image_path for record in records]
    return {
        record.record_id: image_paths[(index + normalized_shift) % len(records)]
        for index, record in enumerate(records)
    }


def renderer_family_classifier(
    root: Path,
    records: list[BenchmarkRecord],
) -> dict[str, float | int]:
    if len(records) < 2:
        raise ValueError("Renderer classifier requires at least two records")
    features = [_image_features(root / record.image_path) for record in records]
    correct = 0
    evaluated = 0
    for held_out_index, record in enumerate(records):
        centroids: dict[str, tuple[float, ...]] = {}
        grouped: dict[str, list[tuple[float, ...]]] = defaultdict(list)
        for index, other in enumerate(records):
            if index != held_out_index:
                grouped[other.renderer_family].append(features[index])
        for family, family_features in grouped.items():
            centroids[family] = tuple(
                fmean(feature[dimension] for feature in family_features)
                for dimension in range(len(family_features[0]))
            )
        if record.renderer_family not in centroids:
            continue
        predicted = min(
            centroids,
            key=lambda family: _squared_distance(features[held_out_index], centroids[family]),
        )
        correct += int(predicted == record.renderer_family)
        evaluated += 1
    return {
        "evaluated_images": evaluated,
        "accuracy": correct / evaluated if evaluated else 0.0,
    }


def _template_key(record: BenchmarkRecord) -> str:
    question = record.question.casefold()
    labels = sorted(
        (*record.table.categories, *record.table.series),
        key=len,
        reverse=True,
    )
    for label in labels:
        question = question.replace(label.casefold(), "<label>")
    return re.sub(r"[-+]?\d+(?:\.\d+)?", "<number>", question)


def _image_features(path: Path) -> tuple[float, ...]:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((64, 64))
        statistics = ImageStat.Stat(rgb)
        means = tuple(float(value) for value in statistics.mean)
        deviations = tuple(float(value) for value in statistics.stddev)
    return means + deviations


def _squared_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(
        (left_value - right_value) ** 2 for left_value, right_value in zip(left, right, strict=True)
    )
