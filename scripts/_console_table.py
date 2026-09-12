"""Small dependency-free console table renderer for diagnostic scripts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def print_table(
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    max_cell_width: int = 72,
) -> None:
    """Print a readable ASCII table while keeping long medical text bounded."""

    print(f"\n{title}")
    if not rows:
        print("(kayıt yok)")
        return
    rendered = [
        [_render_cell(value, max_cell_width=max_cell_width) for value in row]
        for row in rows
    ]
    widths = [
        max(len(str(header)), *(len(row[index]) for row in rendered))
        for index, header in enumerate(headers)
    ]
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    print(separator)
    print("| " + " | ".join(str(header).ljust(widths[index]) for index, header in enumerate(headers)) + " |")
    print(separator)
    for row in rendered:
        print("| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |")
    print(separator)


def _render_cell(value: Any, *, max_cell_width: int) -> str:
    text = "-" if value is None or value == "" else str(value)
    text = " ".join(text.split())
    if len(text) <= max_cell_width:
        return text
    return text[: max(1, max_cell_width - 3)].rstrip() + "..."
