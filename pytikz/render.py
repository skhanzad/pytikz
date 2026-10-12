"""State shared by all elements while a picture is being written out."""

from __future__ import annotations

import itertools
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Iterable, Iterator

from .color import active_palette
from .style import Style

if TYPE_CHECKING:  # pragma: no cover
    from .picture import Picture
    from .theme import Theme

__all__ = ["RenderContext", "LazyName", "SubName", "current", "rendering", "global_name"]

_CURRENT: ContextVar["RenderContext | None"] = ContextVar("pytikz_render", default=None)
_GLOBAL_COUNTER = itertools.count(1)


def current() -> "RenderContext | None":
    """The context of the picture being rendered, if any."""
    return _CURRENT.get()


def global_name(prefix: str = "node") -> str:
    """A process-wide unique name for objects named outside of rendering."""
    return f"{prefix}{next(_GLOBAL_COUNTER)}"


class RenderContext:
    """Collects what a picture needs (libraries, styles, names) while rendering."""

    def __init__(self, picture: "Picture"):
        self.picture = picture
        self.theme: "Theme" = picture.theme
        self.libraries: list[str] = []
        self.packages: list[tuple[str, str | None]] = []
        self.styles: dict[str, Style] = {}
        self.probe = False  # True during the discovery pass (no warnings)
        self._names: dict[int, str] = {}
        self._taken: set[str] = set()
        self._claims: dict[str, int] = {}
        self._counters: dict[str, int] = {}

    # -- requirements ---------------------------------------------------------
    def require(self, *libraries: str) -> None:
        """Ask for TikZ libraries (``\\usetikzlibrary``)."""
        for lib in libraries:
            if lib not in self.libraries:
                self.libraries.append(lib)

    def require_package(self, name: str, options: str | None = None) -> None:
        """Ask for a LaTeX package in the standalone document's preamble."""
        if all(p[0] != name for p in self.packages):
            self.packages.append((name, options))

    def define_style(
        self,
        name: str,
        *args: Any,
        default: str | None = None,
        prefix: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Define a style while rendering (components use this) and return
        the name to use for it.

        A style the user defined on the picture with the same name wins, so
        component styles can be customised. If another component already
        defined ``name`` differently (two graphs, one directed), the new
        definition gets its own name: ``"<prefix> <name>"``.
        """
        if name in self.picture.styles:
            return name
        style = Style(name, *args, default=default, **kwargs)
        candidates = [name]
        if prefix:
            candidates.append(f"{prefix} {name}")
        candidates += [f"{name} {k}" for k in range(2, 100)]
        for candidate in candidates:
            existing = self.styles.get(candidate)
            if existing is None:
                style.name = candidate
                self.styles[candidate] = style
                return candidate
            if existing.options == style.options and existing.default == style.default:
                return candidate
        raise RuntimeError(f"too many variants of style {name!r}")

    # -- names ------------------------------------------------------------------
    def reserve(self, names: Iterable[str]) -> None:
        """Names chosen explicitly somewhere in the picture: automatic names
        must avoid them even before those nodes are rendered."""
        self._taken.update(names)

    @property
    def claimed(self) -> set[str]:
        """Explicit names seen so far."""
        return set(self._claims)

    def claim(self, name: str, owner: object | None = None) -> None:
        """Record an explicit name; warn when two different objects use it."""
        key = id(owner) if owner is not None else None
        previous = self._claims.get(name)
        if previous is not None and key is not None and previous != key and not self.probe:
            warnings.warn(f"two different nodes are named {name!r}; references to it are ambiguous",
                          stacklevel=2)
        if key is not None:
            self._claims.setdefault(name, key)
        else:
            self._claims.setdefault(name, 0)
        self._taken.add(name)

    def auto_name(self, obj: object, prefix: str = "n", *, bare_first: bool = False) -> str:
        """A unique name for ``obj`` in this picture: ``prefix1``, ``prefix2``...

        With ``bare_first`` the first object gets ``prefix`` itself (used for
        components: ``nn``, ``nn2``, ...).
        """
        key = id(obj)
        if key not in self._names:
            k = self._counters.get(prefix, 0)
            while True:
                k += 1
                name = prefix if bare_first and k == 1 else f"{prefix}{k}"
                if name not in self._taken:
                    break
            self._counters[prefix] = k
            self._taken.add(name)
            self._names[key] = name
        return self._names[key]


class LazyName:
    """The name of a component: the explicit one, or one allocated per picture
    at render time (``nn``, ``nn2``, ...), so that two components never clash."""

    def __init__(self, owner: object, prefix: str, explicit: str | None = None):
        self.owner = owner
        self.prefix = prefix
        self.explicit = explicit
        self._pinned: str | None = None

    def resolve(self) -> str:
        if self.explicit is not None:
            return self.explicit
        if self._pinned is not None:
            return self._pinned
        ctx = current()
        if ctx is not None:
            return ctx.auto_name(self.owner, self.prefix, bare_first=True)
        # asked for outside of rendering (e.g. to build raw TikZ): fix it for good
        self._pinned = global_name(self.prefix)
        return self._pinned

    def sub(self, suffix: str) -> "SubName":
        """A node name built from this one, e.g. ``sub("-1-2")`` -> ``nn-1-2``."""
        return SubName(self, suffix)

    def __str__(self) -> str:
        return self.resolve()


class SubName:
    """A node name derived from a :class:`LazyName` (usable as a NodeRef target)."""

    def __init__(self, base: LazyName, suffix: str):
        self.base = base
        self.suffix = suffix

    @property
    def name(self) -> str:
        return self.base.resolve() + self.suffix

    def _mark_referenced(self) -> None:  # component nodes are always named
        pass


@contextmanager
def rendering(ctx: RenderContext) -> Iterator[RenderContext]:
    token = _CURRENT.set(ctx)
    try:
        with active_palette(ctx.theme.palette):
            yield ctx
    finally:
        _CURRENT.reset(token)
