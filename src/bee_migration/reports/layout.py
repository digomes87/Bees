from collections.abc import Sequence

RULE = "=" * 80
UNDERLINE = "-" * 40


def banner(title: str) -> str:
    return "\n".join([RULE, title, RULE])


def section(title: str, lines: Sequence[str]) -> str:
    return "\n".join([title, UNDERLINE, *lines])


def document(parts: Sequence[str]) -> str:
    return "\n\n".join(parts) + "\n"
