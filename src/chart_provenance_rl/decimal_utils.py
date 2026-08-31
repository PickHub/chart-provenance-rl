from decimal import Decimal, DecimalException


def parse_decimal(value: str | int | Decimal) -> Decimal:
    rendered = str(value).strip()
    if len(rendered) > 256:
        raise ValueError("Decimal input exceeds 256 characters")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(rendered)
    except DecimalException as error:
        raise ValueError(f"Invalid decimal value: {value!r}") from error
    if not parsed.is_finite():
        raise ValueError(f"Decimal value must be finite: {value!r}")
    if parsed and abs(parsed.adjusted()) > 100:
        raise ValueError(f"Decimal exponent is outside the supported range: {value!r}")
    return parsed


def canonical_decimal(value: str | int | Decimal) -> str:
    parsed = parse_decimal(value)
    if parsed == 0:
        return "0"
    try:
        normalized = parsed.normalize()
    except DecimalException as error:
        raise ValueError(f"Unable to normalize decimal value: {value!r}") from error
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
