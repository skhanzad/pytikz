"""Checks behind ``pytikz doctor``: the external tools pytikz needs, and how
to install whatever is missing on this system."""

from __future__ import annotations

import os
import platform
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from . import toolchain
from .toolchain import distribution, find_program, kpsewhich

__all__ = ["Check", "Requirement", "TEX_REQUIREMENTS", "run_checks", "doctor", "fix_commands", "linux_distro"]


@dataclass(frozen=True)
class Requirement:
    """A TeX file pytikz uses and the packages that provide it."""

    file: str
    texlive: str          # TeX Live / TinyTeX package (tlmgr install ...)
    debian: str           # Debian/Ubuntu apt package
    required: bool
    description: str


TEX_REQUIREMENTS = (
    Requirement("tikz.sty", "pgf", "texlive-pictures", True, "TikZ/PGF"),
    Requirement("standalone.cls", "standalone", "texlive-latex-extra", True, "standalone class"),
    Requirement("xcolor.sty", "xcolor", "texlive-latex-recommended", True, "xcolor"),
    Requirement("amsmath.sty", "amsmath", "texlive-latex-base", True, "amsmath"),
    Requirement("amssymb.sty", "amsfonts", "texlive-base", True, "amssymb"),
    Requirement("bm.sty", "tools", "texlive-latex-base", True, "bm"),
    Requirement("lmodern.sty", "lm", "lmodern", False, "Latin Modern fonts"),
    Requirement("helvet.sty", "psnfss", "texlive-latex-base", False, "helvet"),
    Requirement("uhvr8t.tfm", "helvetic", "texlive-fonts-recommended", False, "Helvetica fonts"),
)

MIN_PGF = (3, 0)


@dataclass
class Check:
    """One line of the report: ``status`` is ``"ok"``, ``"warn"`` or ``"fail"``."""

    name: str
    status: str
    detail: str
    extra: list[str] = field(default_factory=list)


def linux_distro() -> set[str]:
    """IDs of the Linux distribution and the ones it is like (``/etc/os-release``)."""
    try:
        text = Path("/etc/os-release").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()
    ids: set[str] = set()
    for key in ("ID", "ID_LIKE"):
        m = re.search(rf"^{key}=\"?([^\"\n]*)\"?$", text, re.M)
        if m:
            ids.update(m.group(1).lower().split())
    return ids


def _pgf_version() -> tuple[int, ...] | None:
    path = kpsewhich("pgf.revision.tex")
    if not path:
        return None
    try:
        m = re.search(r"\\def\\pgfversion\{([\d.]+)", Path(path).read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return None
    return tuple(int(p) for p in m.group(1).split(".")) if m else None


def _tlmgr_writable(root: str | None) -> bool:
    if not root:
        return False
    return os.access(Path(root) / "tlpkg", os.W_OK)


def fix_commands(missing: list[Requirement], *, need_latex: bool, need_png: bool, need_svg: bool,
                 engine: str = "pdflatex") -> list[str]:
    """Human readable install instructions for this system."""
    lines: list[str] = []
    if not (missing or need_latex or need_png or need_svg):
        return lines
    dist = distribution(engine)
    tl_pkgs = " ".join(r.texlive for r in missing)
    system = platform.system()
    if need_latex or missing:
        if dist["kind"] == "tinytex" or (dist["kind"] == "texlive" and _tlmgr_writable(dist["root"])):
            lines.append(f"pytikz setup                  # runs: tlmgr install {tl_pkgs}")
        elif need_latex:
            lines.append("pytikz setup                  # installs TinyTeX (~55 MB download, no admin rights) "
                         "and everything else needed")
        if dist["kind"] == "miktex":
            lines.append("MiKTeX installs missing packages automatically on first use (allow it when asked), "
                         f"or: miktex packages install {tl_pkgs}")
        elif dist["kind"] == "texlive" and not _tlmgr_writable(dist["root"]) and missing:
            lines.append(f"sudo tlmgr install {tl_pkgs}")
        elif system == "Linux" and (need_latex or dist["kind"] == "texlive-debian"):
            distro = linux_distro()
            if distro & {"debian", "ubuntu"}:
                pkgs = ["texlive-latex-base", "texlive-latex-recommended", "texlive-latex-extra",
                        "texlive-pictures", "texlive-fonts-recommended", "lmodern"]
                if need_latex:
                    lines.append("or with apt:  sudo apt install " + " ".join(pkgs + ["poppler-utils"]))
                    return lines  # the apt command covers PNG/SVG too
                else:
                    debs = sorted({r.debian for r in missing})
                    lines.append("or with apt:  sudo apt install " + " ".join(debs))
            elif distro & {"fedora", "rhel", "centos"}:
                pkgs = "texlive-scheme-basic " + " ".join(f"texlive-{r.texlive}" for r in TEX_REQUIREMENTS)
                lines.append(f"or with dnf:  sudo dnf install {pkgs} poppler-utils")
            elif distro & {"opensuse", "suse"}:
                pkgs = "texlive-scheme-basic " + " ".join(f"texlive-{r.texlive}" for r in TEX_REQUIREMENTS)
                lines.append(f"or with zypper:  sudo zypper install {pkgs} poppler-tools")
            elif distro & {"arch"}:
                lines.append("or with pacman:  sudo pacman -S --needed texlive-basic texlive-latex "
                             "texlive-latexrecommended texlive-latexextra texlive-pictures "
                             "texlive-fontsrecommended poppler")
        elif system == "Darwin" and need_latex:
            lines.append("or with Homebrew:  brew install --cask basictex && brew install poppler")
            lines.append("    then:  sudo tlmgr update --self && sudo tlmgr install "
                         + " ".join(r.texlive for r in TEX_REQUIREMENTS))
        elif system == "Windows" and need_latex:
            lines.append("or install MiKTeX:  winget install MiKTeX.MiKTeX   (packages are installed on first use)")
    if need_png and not need_latex:
        lines.append('pip install "pytikz[png]"     # PNG output via pypdfium2 (or install poppler)')
    if need_svg and not need_latex:
        if system == "Linux" and linux_distro() & {"debian", "ubuntu"}:
            lines.append("sudo apt install poppler-utils   # SVG output")
        elif system == "Darwin":
            lines.append("brew install poppler          # SVG output")
        elif system == "Windows":
            lines.append("scoop install poppler         # SVG output (or: choco install poppler)")
        else:
            lines.append("install poppler (pdftocairo) for SVG output")
    return lines


def run_checks(engine: str = "pdflatex", compile_test: bool = True) -> tuple[list[Check], dict]:
    """Run all checks. Returns the report lines and a summary dict with the
    keys ``latex``, ``missing`` (requirements), ``png`` and ``svg``."""
    from .compiler import converters

    checks: list[Check] = []
    try:
        import numpy

        numpy_version = numpy.__version__
    except ImportError:  # pragma: no cover - numpy is a dependency
        numpy_version = None
    py = platform.python_version()
    checks.append(Check("Python", "ok" if numpy_version else "fail",
                        f"{py}, numpy {numpy_version}" if numpy_version else f"{py}; numpy is missing: pip install numpy",
                        [sys.executable]))

    dist = distribution(engine)
    summary: dict = {"latex": dist["path"] is not None, "missing": [], "png": True, "svg": True}
    if dist["path"] is None:
        checks.append(Check("LaTeX", "fail", f"{engine} not found"))
    else:
        where = dist["path"]
        if not any(Path(p) == Path(where).parent for p in os.environ.get("PATH", "").split(os.pathsep) if p):
            where += "  (not on PATH; found automatically)"
        checks.append(Check("LaTeX", "ok", dist["version"] or engine, [where]))

        if find_program("kpsewhich") is None:
            checks.append(Check("packages", "warn", "kpsewhich not found; cannot check TeX packages"))
        else:
            missing = [r for r in TEX_REQUIREMENTS if kpsewhich(r.file) is None]
            summary["missing"] = missing
            pgf = _pgf_version()
            if any(r.texlive == "pgf" for r in missing):
                checks.append(Check("TikZ", "fail", "TikZ/PGF is not installed"))
            elif pgf is not None and pgf < MIN_PGF:
                checks.append(Check("TikZ", "warn", f"PGF {'.'.join(map(str, pgf))} is old; 3.1 or newer "
                                                    "is recommended"))
            else:
                checks.append(Check("TikZ", "ok", f"PGF {'.'.join(map(str, pgf))}" if pgf else "TikZ/PGF"))
            others = [r for r in TEX_REQUIREMENTS if r.required and r.texlive != "pgf"]
            missing_req = [r for r in others if r in missing]
            checks.append(Check(
                "packages",
                "fail" if missing_req else "ok",
                ("missing: " + ", ".join(r.description for r in missing_req)) if missing_req
                else ", ".join(r.description for r in others),
            ))
            fonts = [r for r in TEX_REQUIREMENTS if not r.required]
            missing_fonts = [r for r in fonts if r in missing]
            checks.append(Check(
                "fonts",
                "warn" if missing_fonts else "ok",
                ("missing (pictures still compile, with fallback fonts): "
                 + ", ".join(r.description for r in missing_fonts)) if missing_fonts
                else ", ".join(r.description for r in fonts),
            ))
            if dist["kind"] == "miktex" and missing:
                checks[-1].extra.append("MiKTeX may install missing packages automatically on first use")

    conv = converters()
    summary["png"], summary["svg"] = bool(conv["png"]), bool(conv["svg"])
    for fmt in ("png", "svg"):
        names = conv[fmt]
        if names:
            more = f"  (also: {', '.join(names[1:])})" if len(names) > 1 else ""
            checks.append(Check(fmt.upper(), "ok", names[0] + more))
        else:
            checks.append(Check(fmt.upper(), "warn", f"no converter found; {fmt.upper()} output is unavailable"))

    if compile_test and dist["path"] is not None:
        checks.append(_compile_check(engine))
    return checks, summary


def _compile_check(engine: str) -> Check:
    from .compiler import TikzCompileError, ToolNotFoundError
    from .picture import Picture

    pic = Picture(engine=engine)
    a = pic.node("$x$", style="block=Green", minimum_width=1)
    b = pic.node("pytikz", right_of=a, distance=1, style="block")
    pic.arrow(a, b, label="ok")
    start = time.perf_counter()
    try:
        pic._compile("pdf", cache=False)
    except TikzCompileError as err:
        first = err.message.splitlines()[0]
        return Check("test", "fail", f"compiling a small picture failed: {first}")
    except ToolNotFoundError as err:
        return Check("test", "fail", str(err))
    return Check("test", "ok", f"compiled a small picture in {time.perf_counter() - start:.2f} s")


_COLORS = {"ok": "32", "warn": "33", "fail": "31"}
_WORDS = {"ok": "ok", "warn": "warning", "fail": "missing"}


def _paint(text: str, status: str, color: bool) -> str:
    return f"\033[{_COLORS[status]}m{text}\033[0m" if color else text


def doctor(engine: str = "pdflatex", *, compile_test: bool = True, file: TextIO | None = None) -> bool:
    """Print a report of the toolchain and how to fix problems.

    Returns ``True`` when everything needed to render PDFs works.
    """
    out = file or sys.stdout
    color = hasattr(out, "isatty") and out.isatty() and "NO_COLOR" not in os.environ
    toolchain.clear_cache()
    checks, summary = run_checks(engine, compile_test=compile_test)
    print("pytikz doctor\n", file=out)
    width = max(len(c.detail) for c in checks)
    width = min(width, 64)
    for c in checks:
        print(f"  {c.name:<9} {c.detail:<{width}}  {_paint(_WORDS[c.status], c.status, color)}", file=out)
        for extra in c.extra:
            print(f"  {'':<9} {extra}", file=out)
    required_ok = all(c.status != "fail" for c in checks)
    fixes = fix_commands(
        summary["missing"],
        need_latex=not summary["latex"],
        need_png=not summary["png"],
        need_svg=not summary["svg"],
        engine=engine,
    )
    print(file=out)
    if required_ok and not fixes:
        print("Everything pytikz needs is installed.", file=out)
    elif required_ok:
        print("pytikz works; optional extras:", file=out)
    else:
        print("To fix:", file=out)
    for line in fixes:
        print(f"  {line}", file=out)
    return required_ok
