from __future__ import annotations

from decimal import Decimal
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont

from chart_provenance_rl.constants import DEFAULT_IMAGE_SIZE
from chart_provenance_rl.decimal_utils import canonical_decimal
from chart_provenance_rl.schema import ChartType, TableData

PALETTES: dict[str, tuple[str, str, str, str]] = {
    "classic": ("#2f5597", "#c55a11", "#ffffff", "#1f1f1f"),
    "pastel": ("#719ec7", "#e6a57e", "#fffdf7", "#323232"),
    "high_contrast": ("#0057b8", "#f4b400", "#ffffff", "#000000"),
    "monochrome": ("#333333", "#a0a0a0", "#f7f7f7", "#111111"),
}


def render_chart(
    table: TableData,
    chart_type: ChartType,
    renderer_family: str,
) -> bytes:
    try:
        palette = PALETTES[renderer_family]
    except KeyError as error:
        raise ValueError(f"Unknown renderer family: {renderer_family}") from error
    image = Image.new("RGB", DEFAULT_IMAGE_SIZE, color=palette[2])
    draw = ImageDraw.Draw(image)
    fonts = _fonts()
    draw.text((18, 14), table.title, fill=palette[3], font=fonts.title)
    if chart_type is ChartType.BAR:
        _draw_bar(draw, table, palette, fonts)
    elif chart_type is ChartType.LINE:
        _draw_line(draw, table, palette, fonts)
    else:
        _draw_table(draw, table, palette, fonts)
    output = BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


class _Fonts:
    def __init__(self) -> None:
        self.title = ImageFont.load_default(size=24)
        self.label = ImageFont.load_default(size=16)
        self.small = ImageFont.load_default(size=13)


def _fonts() -> _Fonts:
    return _Fonts()


def _draw_bar(
    draw: ImageDraw.ImageDraw,
    table: TableData,
    palette: tuple[str, str, str, str],
    fonts: _Fonts,
) -> None:
    left, top, right, bottom = 52, 72, 490, 310
    _draw_axes(draw, left, top, right, bottom, palette[3])
    maximum = _chart_maximum(table)
    category_width = (right - left) / len(table.categories)
    bar_width = min(40.0, category_width / (len(table.series) + 1))
    for category_index, category in enumerate(table.categories):
        center = left + category_width * (category_index + 0.5)
        for series_index, _series in enumerate(table.series):
            value = table.values[series_index][category_index]
            height = float(value / maximum) * (bottom - top - 28)
            x0 = center + (series_index - (len(table.series) - 1) / 2) * bar_width - bar_width * 0.4
            x1 = x0 + bar_width * 0.8
            y0 = bottom - height
            draw.rectangle((x0, y0, x1, bottom), fill=palette[series_index], outline=palette[3])
            _center_text(
                draw,
                canonical_decimal(value),
                (x0 + x1) / 2,
                max(top, y0 - 18),
                palette[3],
                fonts.small,
            )
        _center_text(draw, category, center, bottom + 12, palette[3], fonts.label)
    _draw_legend(draw, table, palette, fonts)


def _draw_line(
    draw: ImageDraw.ImageDraw,
    table: TableData,
    palette: tuple[str, str, str, str],
    fonts: _Fonts,
) -> None:
    left, top, right, bottom = 52, 72, 490, 310
    _draw_axes(draw, left, top, right, bottom, palette[3])
    maximum = _chart_maximum(table)
    spacing = (right - left) / max(1, len(table.categories) - 1)
    for category_index, category in enumerate(table.categories):
        x = left + spacing * category_index
        _center_text(draw, category, x, bottom + 12, palette[3], fonts.label)
    for series_index, _series in enumerate(table.series):
        points: list[tuple[float, float]] = []
        for category_index, value in enumerate(table.values[series_index]):
            x = left + spacing * category_index
            y = bottom - float(value / maximum) * (bottom - top - 28)
            points.append((x, y))
        draw.line(points, fill=palette[series_index], width=4)
        for (x, y), value in zip(points, table.values[series_index], strict=True):
            draw.ellipse(
                (x - 5, y - 5, x + 5, y + 5), fill=palette[series_index], outline=palette[3]
            )
            _center_text(
                draw,
                canonical_decimal(value),
                x,
                max(top, y - 22 - series_index * 15),
                palette[3],
                fonts.small,
            )
    _draw_legend(draw, table, palette, fonts)


def _draw_table(
    draw: ImageDraw.ImageDraw,
    table: TableData,
    palette: tuple[str, str, str, str],
    fonts: _Fonts,
) -> None:
    left, top, right = 24, 82, 488
    row_height = 52
    columns = len(table.categories) + 1
    column_width = (right - left) / columns
    rows = len(table.series) + 1
    for row in range(rows):
        for column in range(columns):
            x0 = left + column * column_width
            y0 = top + row * row_height
            x1 = x0 + column_width
            y1 = y0 + row_height
            fill = palette[2] if row and column else "#e8e8e8"
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline=palette[3], width=2)
    for index, category in enumerate(table.categories, start=1):
        _center_text(
            draw,
            category,
            left + (index + 0.5) * column_width,
            top + 16,
            palette[3],
            fonts.label,
        )
    for series_index, series in enumerate(table.series, start=1):
        y = top + series_index * row_height + 16
        _center_text(
            draw,
            series,
            left + column_width / 2,
            y,
            palette[3],
            fonts.label,
        )
        for category_index, value in enumerate(table.values[series_index - 1], start=1):
            _center_text(
                draw,
                canonical_decimal(value),
                left + (category_index + 0.5) * column_width,
                y,
                palette[3],
                fonts.label,
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
    table: TableData,
    palette: tuple[str, str, str, str],
    fonts: _Fonts,
) -> None:
    for index, series in enumerate(table.series):
        x = 260 + index * 112
        draw.rectangle((x, 42, x + 14, 56), fill=palette[index], outline=palette[3])
        draw.text((x + 18, 41), series, fill=palette[3], font=fonts.small)


def _center_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    center_x: float,
    y: float,
    color: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    draw.text((center_x - width / 2, y), text, fill=color, font=font)


def _chart_maximum(table: TableData) -> Decimal:
    maximum = max(value for row in table.values for value in row)
    return maximum * 11 / 10 if maximum > 0 else Decimal(1)
