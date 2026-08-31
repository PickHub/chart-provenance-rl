from decimal import Decimal, InvalidOperation


def parse_decimal(value: str | int | Decimal) -> Decimal:
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value).strip())
    except InvalidOperation as error:
        raise ValueError(f"Invalid decimal value: {value!r}") from error
    if not parsed.is_finite():
        raise ValueError(f"Decimal value must be finite: {value!r}")
    return parsed


def canonical_decimal(value: str | int | Decimal) -> str:
    parsed = parse_decimal(value)
    if parsed == 0:
        return "0"
    normalized = parsed.normalize()
    rendered = format(normalized, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def canonical_answer(value: str) -> str:
    stripped = " ".join(value.strip().split())
    if not stripped:
        return ""
    try:
        return canonical_decimal(stripped)
    except ValueError:
        return stripped.casefold()
