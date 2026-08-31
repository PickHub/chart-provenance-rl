from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont

from chart_provenance_rl.constants import DEFAULT_IMAGE_SIZE
from chart_provenance_rl.decimal_utils import canonical_decimal
from chart_provenance_rl.schema import (
    ChartType,
    EvidenceRef,
    RenderText,
    TableData,
    TextRole,
)

PALETTES: dict[str, tuple[str, str, str, str]] = {
    "classic": ("#2f5597", "#c55a11", "#ffffff", "#1f1f1f"),
    "pastel": ("#719ec7", "#e6a57e", "#fffdf7", "#323232"),
    "high_contrast": ("#0057b8", "#f4b400", "#ffffff", "#000000"),
    "monochrome": ("#333333", "#a0a0a0", "#f7f7f7", "#111111"),
}
MIN_TEXT_CONTRAST = Decimal("4.5")


@dataclass(frozen=True)
class RenderResult:
    image_bytes: bytes
    text_elements: tuple[RenderText, ...]


class _Fonts:
    def __init__(self) -> None:
        self.title = ImageFont.load_default(size=24)
        self.label = ImageFont.load_default(size=16)
        self.small = ImageFont.load_default(size=13)


class _TextRegistry:
    def __init__(
        self,
        draw: ImageDraw.ImageDraw,
        image_size: tuple[int, int],
    ) -> None:
        self.draw = draw
        self.image_size = image_size
        self.elements: list[RenderText] = []

    def draw_centered(
        self,
        role: TextRole,
        text: str,
        positions: tuple[tuple[float, float], ...],
        foreground: str,
        background: str,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        evidence: EvidenceRef | None = None,
    ) -> RenderText:
        if contrast_ratio(foreground, background) < MIN_TEXT_CONTRAST:
            raise ValueError(
                f"Insufficient text contrast for {role.value} {text!r}: "
                f"{foreground} on {background}"
            )
        for center_x, top in positions:
            placement = self._placement(text, center_x, top, font)
            if self._fits(placement) and not any(
                boxes_overlap(placement, element.bbox) for element in self.elements
            ):
                self.draw.rectangle(placement, fill=background)
                text_box = self.draw.textbbox((0, 0), text, font=font)
                text_x = round(center_x - (text_box[2] - text_box[0]) / 2 - text_box[0])
                text_y = round(top + 2 - text_box[1])
                self.draw.text(
                    (text_x, text_y),
                    text,
                    fill=foreground,
                    font=font,
                )
                element = RenderText(
                    role=role,
                    text=text,
                    bbox=placement,
                    foreground=foreground,
                    background=background,
                    contrast_ratio=contrast_ratio(foreground, background),
                    evidence=evidence,
                )
                self.elements.append(element)
                return element
        raise ValueError(f"Unable to place visible {role.value} label {text!r}")

    def _placement(
        self,
        text: str,
        center_x: float,
        top: float,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    ) -> tuple[int, int, int, int]:
        text_box = self.draw.textbbox((0, 0), text, font=font)
        width = text_box[2] - text_box[0]
        height = text_box[3] - text_box[1]
        left = round(center_x - width / 2) - 2
        top_coordinate = round(top)
        return (
            left,
            top_coordinate,
            round(left + width + 4),
            round(top_coordinate + height + 4),
        )

    def _fits(self, box: tuple[int, int, int, int]) -> bool:
        width, height = self.image_size
        left, top, right, bottom = box
        return left >= 0 and top >= 0 and right <= width and bottom <= height


def render_chart(
    table: TableData,
    chart_type: ChartType,
    renderer_family: str,
) -> RenderResult:
    try:
        palette = PALETTES[renderer_family]
    except KeyError as error:
        raise ValueError(f"Unknown renderer family: {renderer_family}") from error
    image = Image.new("RGB", DEFAULT_IMAGE_SIZE, color=palette[2])
    draw = ImageDraw.Draw(image)
    fonts = _Fonts()
    labels = _TextRegistry(draw, DEFAULT_IMAGE_SIZE)
    labels.draw_centered(
        TextRole.TITLE,
        table.title,
        ((DEFAULT_IMAGE_SIZE[0] / 2, 8),),
        palette[3],
        palette[2],
        fonts.title,
    )
    if chart_type is ChartType.BAR:
        _draw_bar(draw, labels, table, palette, fonts)
    elif chart_type is ChartType.LINE:
        _draw_line(draw, labels, table, palette, fonts)
    else:
        _draw_table(draw, labels, table, palette, fonts)
    output = BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return RenderResult(
        image_bytes=output.getvalue(),
        text_elements=tuple(labels.elements),
    )


def contrast_ratio(foreground: str, background: str) -> Decimal:
    foreground_luminance = _relative_luminance(foreground)
    background_luminance = _relative_luminance(background)
    lighter = max(foreground_luminance, background_luminance)
    darker = min(foreground_luminance, background_luminance)
    return ((lighter + Decimal("0.05")) / (darker + Decimal("0.05"))).quantize(Decimal("0.001"))


def boxes_overlap(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> bool:
    return not (
        left[2] <= right[0] or right[2] <= left[0] or left[3] <= right[1] or right[3] <= left[1]
    )


def _draw_bar(
    draw: ImageDraw.ImageDraw,
    labels: _TextRegistry,
    table: TableData,
    palette: tuple[str, str, str, str],
    fonts: _Fonts,
) -> None:
    left, top, right, bottom = 58, 82, 454, 300
    _draw_legend(draw, labels, table, palette, fonts)
    _draw_axes(draw, left, top, right, bottom, palette[3])
    maximum = _chart_maximum(table)
    category_width = (right - left) / len(table.categories)
    bar_spacing = min(46.0, category_width / 2.2)
    value_labels: list[tuple[float, float, Decimal, EvidenceRef]] = []
    for category_index, category in enumerate(table.categories):
        center = left + category_width * (category_index + 0.5)
        labels.draw_centered(
            TextRole.CATEGORY,
            category,
            ((center, 312),),
            palette[3],
            palette[2],
            fonts.label,
        )
        for series_index, series in enumerate(table.series):
            value = table.values[series_index][category_index]
            height = float(value / maximum) * (bottom - top - 24)
            value_center = center + (series_index - 0.5) * bar_spacing
            x0 = value_center - bar_spacing * 0.34
            x1 = value_center + bar_spacing * 0.34
            y0 = bottom - height
            draw.rectangle(
                (x0, y0, x1, bottom),
                fill=palette[series_index],
                outline=palette[3],
            )
            value_labels.append(
                (
                    value_center,
                    y0,
                    value,
                    EvidenceRef(series=series, category=category),
                )
            )
    for value_center, y0, value, evidence in value_labels:
        labels.draw_centered(
            TextRole.VALUE,
            canonical_decimal(value),
            (
                (value_center, max(top, y0 - 20)),
                (value_center, min(bottom - 22, y0 + 5)),
                (value_center, max(top, y0 - 39)),
            ),
            palette[3],
            palette[2],
            fonts.small,
            evidence,
        )


def _draw_line(
    draw: ImageDraw.ImageDraw,
    labels: _TextRegistry,
    table: TableData,
    palette: tuple[str, str, str, str],
    fonts: _Fonts,
) -> None:
    left, top, right, bottom = 60, 82, 452, 300
    _draw_legend(draw, labels, table, palette, fonts)
    _draw_axes(draw, left, top, right, bottom, palette[3])
    maximum = _chart_maximum(table)
    spacing = (right - left) / max(1, len(table.categories) - 1)
    for category_index, category in enumerate(table.categories):
        x = left + spacing * category_index
        labels.draw_centered(
            TextRole.CATEGORY,
            category,
            ((x, 312),),
            palette[3],
            palette[2],
            fonts.label,
        )
    plotted_series: list[tuple[int, str, list[tuple[float, float]]]] = []
    for series_index, series in enumerate(table.series):
        points: list[tuple[float, float]] = []
        for category_index, value in enumerate(table.values[series_index]):
            x = left + spacing * category_index
            y = bottom - float(value / maximum) * (bottom - top - 24)
            points.append((x, y))
        plotted_series.append((series_index, series, points))
        draw.line(points, fill=palette[series_index], width=4)
        for x, y in points:
            draw.ellipse(
                (x - 5, y - 5, x + 5, y + 5),
                fill=palette[series_index],
                outline=palette[3],
            )
    for series_index, series, points in plotted_series:
        for (x, y), category, value in zip(
            points,
            table.categories,
            table.values[series_index],
            strict=True,
        ):
            preferred_offset = -22 if series_index == 0 else 8
            labels.draw_centered(
                TextRole.VALUE,
                canonical_decimal(value),
                (
                    (x, y + preferred_offset),
                    (x, y + (8 if preferred_offset < 0 else -22)),
                    (x, y - 41),
                    (x, y + 27),
                ),
                palette[3],
                palette[2],
                fonts.small,
                EvidenceRef(series=series, category=category),
            )


def _draw_table(
    draw: ImageDraw.ImageDraw,
    labels: _TextRegistry,
    table: TableData,
    palette: tuple[str, str, str, str],
    fonts: _Fonts,
) -> None:
    left, top, right = 24, 72, 488
    row_height = 68
    columns = len(table.categories) + 1
    column_width = (right - left) / columns
    rows = len(table.series) + 1
    header_background = "#e8e8e8"
    for row in range(rows):
        for column in range(columns):
            x0 = left + column * column_width
            y0 = top + row * row_height
            x1 = x0 + column_width
            y1 = y0 + row_height
            fill = palette[2] if row and column else header_background
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline=palette[3], width=2)
    for index, category in enumerate(table.categories, start=1):
        labels.draw_centered(
            TextRole.CATEGORY,
            category,
            ((left + (index + 0.5) * column_width, top + 22),),
            palette[3],
            header_background,
            fonts.label,
        )
    for series_index, series in enumerate(table.series, start=1):
        row_top = top + series_index * row_height
        labels.draw_centered(
            TextRole.LEGEND,
            series,
            ((left + column_width / 2, row_top + 22),),
            palette[3],
            header_background,
            fonts.label,
        )
        for category_index, value in enumerate(
            table.values[series_index - 1],
            start=1,
        ):
            category = table.categories[category_index - 1]
            labels.draw_centered(
                TextRole.VALUE,
                canonical_decimal(value),
                (
                    (
                        left + (category_index + 0.5) * column_width,
                        row_top + 22,
                    ),
                ),
                palette[3],
                palette[2],
                fonts.label,
                EvidenceRef(series=series, category=category),
            )


def _draw_axes(
    draw: ImageDraw.ImageDraw,
    left: int,
    top: int,
    right: int,
    bottom: int,
    color: str,
) -> None:
    draw.line((left, top, left, bottom), fill=color, width=2)
    draw.line((left, bottom, right, bottom), fill=color, width=2)


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    labels: _TextRegistry,
    table: TableData,
    palette: tuple[str, str, str, str],
    fonts: _Fonts,
) -> None:
    for index, series in enumerate(table.series):
        center = 330 + index * 105
        element = labels.draw_centered(
            TextRole.LEGEND,
            series,
            ((center, 44),),
            palette[3],
            palette[2],
            fonts.small,
        )
        swatch_right = element.bbox[0] - 5
        draw.rectangle(
            (swatch_right - 13, 46, swatch_right, 59),
            fill=palette[index],
            outline=palette[3],
        )


def _chart_maximum(table: TableData) -> Decimal:
    maximum = max(value for row in table.values for value in row)
    return maximum * 11 / 10 if maximum > 0 else Decimal(1)


def _relative_luminance(color: str) -> Decimal:
    if not color.startswith("#") or len(color) != 7:
        raise ValueError(f"Expected an RGB hex color, received {color!r}")
    channels = [Decimal(int(color[index : index + 2], 16)) / Decimal(255) for index in (1, 3, 5)]
    linear = [
        channel / Decimal("12.92")
        if channel <= Decimal("0.04045")
        else ((channel + Decimal("0.055")) / Decimal("1.055")) ** Decimal("2.4")
        for channel in channels
    ]
    return (
        Decimal("0.2126") * linear[0]
        + Decimal("0.7152") * linear[1]
        + Decimal("0.0722") * linear[2]
    )
