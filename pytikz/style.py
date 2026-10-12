"""Named TikZ styles (``\\tikzset{name/.style={...}}``)."""

from __future__ import annotations

from typing import Any

from .options import Options

__all__ = ["Style"]


class Style:
    """A named, reusable option list.

    ``Style("block", "draw=#1, fill=#1!15", default="Blue")`` defines a style
    with a parameter; use it as ``style="block"`` or ``style="block=Red"``
    (or ``block("Red")``).
    """

    __slots__ = ("name", "options", "default")

    def __init__(self, name: str, *args: Any, default: str | None = None, **kwargs: Any):
        if not name or any(ch in name for ch in ",=[]{}"):
            raise ValueError(f"invalid style name {name!r}")
        self.name = name
        self.options = Options(*args, **kwargs)
        self.default = default

    def definitions(self) -> list[str]:
        """The ``\\tikzset`` entries for this style."""
        lines = [f"{self.name}/.style={{{self.options.inner()}}}"]
        if self.default is not None:
            lines.append(f"{self.name}/.default={self.default}")
        return lines

    def __call__(self, argument: Any) -> str:
        """``block("Red")`` -> ``"block=Red"`` for parameterised styles."""
        return f"{self.name}={argument}"

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Style({self.name!r}, {self.options.inner()!r})"
