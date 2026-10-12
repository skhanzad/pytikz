"""Compile LaTeX documents to PDF, PNG and SVG.

Uses ``pdflatex`` (or ``lualatex``/``xelatex``) and poppler's ``pdftocairo``
(falling back to ``pdftoppm``/``dvisvgm``). Results are cached by content hash
in ``~/.cache/pytikz`` (override with ``PYTIKZ_CACHE``), so re-rendering an
unchanged picture is instant.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

__all__ = [
    "TikzCompileError",
    "LatexNotFoundError",
    "compile_pdf",
    "compile_to_bytes",
    "render_document",
    "pdf_to_png",
    "pdf_to_svg",
    "latex_available",
]


class LatexNotFoundError(RuntimeError):
    """A required program (LaTeX engine or converter) is not installed."""


class TikzCompileError(RuntimeError):
    """LaTeX failed; ``str(err)`` shows the error and the offending code."""

    def __init__(self, message: str, *, source: str = "", log: str = "", line: int | None = None):
        self.message = message
        self.source = source
        self.log = log
        self.line = line
        super().__init__(self._format())

    def _format(self) -> str:
        text = [self.message]
        if self.line is not None and self.source:
            src = self.source.splitlines()
            lo, hi = max(1, self.line - 3), min(len(src), self.line + 2)
            text.append(f"at line {self.line} of the generated document:")
            for n in range(lo, hi + 1):
                marker = ">" if n == self.line else " "
                text.append(f"  {marker}{n:5d} | {src[n - 1]}")
        return "\n".join(text)


def _which(*names: str) -> str | None:
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None


def latex_available(engine: str = "pdflatex") -> bool:
    """Whether ``engine`` is installed."""
    return _which(engine) is not None


def _cache_dir() -> Path:
    root = os.environ.get("PYTIKZ_CACHE")
    path = Path(root) if root else Path.home() / ".cache" / "pytikz"
    return path


def _cache_get(key: str, suffix: str) -> bytes | None:
    try:
        return (_cache_dir() / f"{key}{suffix}").read_bytes()
    except OSError:
        return None


def _cache_put(key: str, suffix: str, data: bytes) -> None:
    try:
        directory = _cache_dir()
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / f".{key}{suffix}.{os.getpid()}"
        tmp.write_bytes(data)
        tmp.replace(directory / f"{key}{suffix}")
    except OSError:
        pass  # caching is best effort


_FILE_LINE = re.compile(r"^(?:\./)?(?P<file>[^:\n]+?):(?P<line>\d+): (?P<msg>.*)$", re.M)


def _parse_log(log: str, source: str, engine: str) -> TikzCompileError:
    match = None
    for m in _FILE_LINE.finditer(log):
        match = m
        break
    if match is not None:
        message = match.group("msg").strip()
        # Add the continuation of multi-line messages.
        rest = log[match.end():].splitlines()[1:6]
        for extra in rest:
            if not extra.strip() or extra.startswith(("See the", "Type", "l.")):
                break
            message += " " + extra.strip()
        line = int(match.group("line")) if match.group("file").endswith("figure.tex") else None
        return TikzCompileError(f"{engine} error: {message}", source=source, log=log, line=line)
    bang = re.search(r"^! (.*)$", log, re.M)
    if bang:
        line_match = re.search(r"^l\.(\d+)", log[bang.end():], re.M)
        line = int(line_match.group(1)) if line_match else None
        return TikzCompileError(f"{engine} error: {bang.group(1)}", source=source, log=log, line=line)
    tail = "\n".join(log.strip().splitlines()[-15:])
    return TikzCompileError(f"{engine} failed:\n{tail}", source=source, log=log)


def compile_pdf(source: str, *, engine: str = "pdflatex", timeout: float = 120, cache: bool = True) -> bytes:
    """Compile a complete LaTeX document and return the PDF bytes."""
    key = hashlib.sha256(f"{engine}\0{source}".encode()).hexdigest()[:32]
    if cache:
        hit = _cache_get(key, ".pdf")
        if hit is not None:
            return hit
    exe = _which(engine)
    if exe is None:
        raise LatexNotFoundError(
            f"{engine!r} was not found. Install a TeX distribution (TeX Live, MiKTeX or "
            "MacTeX) with the TikZ (pgf) package, or save the .tex file and compile it elsewhere."
        )
    with tempfile.TemporaryDirectory(prefix="pytikz-") as tmp:
        tex = Path(tmp) / "figure.tex"
        tex.write_text(source, encoding="utf-8")
        cmd = [exe, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "figure.tex"]
        # keep TeX from wrapping log lines at 79 characters
        env = {**os.environ, "max_print_line": "10000", "error_line": "254", "half_error_line": "238"}
        try:
            proc = subprocess.run(cmd, cwd=tmp, capture_output=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            raise TikzCompileError(
                f"{engine} did not finish within {timeout:g}s (infinite loop in the picture?)",
                source=source,
            ) from None
        log_path = Path(tmp) / "figure.log"
        log = (
            log_path.read_text(encoding="utf-8", errors="replace")
            if log_path.exists()
            else proc.stdout.decode("utf-8", "replace")
        )
        pdf = Path(tmp) / "figure.pdf"
        if proc.returncode != 0 or not pdf.exists():
            raise _parse_log(log, source, engine)
        data = pdf.read_bytes()
    if cache:
        _cache_put(key, ".pdf", data)
    return data


def pdf_to_png(pdf: bytes, *, dpi: int = 300, transparent: bool = False) -> bytes:
    """Rasterise the first page of a PDF."""
    with tempfile.TemporaryDirectory(prefix="pytikz-") as tmp:
        src = Path(tmp) / "in.pdf"
        src.write_bytes(pdf)
        out = Path(tmp) / "out"
        cairo = _which("pdftocairo")
        if cairo:
            cmd = [cairo, "-png", "-r", str(dpi), "-singlefile"]
            if transparent:
                cmd.append("-transp")
            subprocess.run(cmd + [str(src), str(out)], check=True, capture_output=True)
            return (Path(tmp) / "out.png").read_bytes()
        ppm = _which("pdftoppm")
        if ppm and not transparent:
            subprocess.run(
                [ppm, "-png", "-r", str(dpi), "-singlefile", str(src), str(out)],
                check=True,
                capture_output=True,
            )
            return (Path(tmp) / "out.png").read_bytes()
        magick = _which("magick", "convert")
        if magick:
            target = Path(tmp) / "out.png"
            cmd = [magick, "-density", str(dpi), f"{src}[0]"]
            if not transparent:
                cmd += ["-background", "white", "-alpha", "remove"]
            subprocess.run(cmd + [str(target)], check=True, capture_output=True)
            return target.read_bytes()
    raise LatexNotFoundError("PNG output needs pdftocairo/pdftoppm (poppler) or ImageMagick")


def pdf_to_svg(pdf: bytes) -> bytes:
    """Convert the first page of a PDF to SVG."""
    with tempfile.TemporaryDirectory(prefix="pytikz-") as tmp:
        src = Path(tmp) / "in.pdf"
        src.write_bytes(pdf)
        target = Path(tmp) / "out.svg"
        cairo = _which("pdftocairo")
        if cairo:
            subprocess.run([cairo, "-svg", str(src), str(target)], check=True, capture_output=True)
            return target.read_bytes()
        dvisvgm = _which("dvisvgm")
        if dvisvgm:
            subprocess.run(
                [dvisvgm, "--pdf", "--no-fonts", "-o", str(target), str(src)],
                check=True,
                capture_output=True,
            )
            return target.read_bytes()
    raise LatexNotFoundError("SVG output needs pdftocairo (poppler) or dvisvgm")


def compile_to_bytes(
    source: str,
    fmt: str = "pdf",
    *,
    engine: str = "pdflatex",
    dpi: int = 300,
    transparent: bool = False,
    cache: bool = True,
) -> bytes:
    """Compile ``source`` and return the output in format ``fmt``."""
    fmt = fmt.lower().lstrip(".")
    pdf = compile_pdf(source, engine=engine, cache=cache)
    if fmt == "pdf":
        return pdf
    key = hashlib.sha256(pdf).hexdigest()[:32]
    suffix = f".{dpi}{'t' if transparent else ''}.png" if fmt == "png" else f".{fmt}"
    if cache:
        hit = _cache_get(key, suffix)
        if hit is not None:
            return hit
    if fmt == "png":
        data = pdf_to_png(pdf, dpi=dpi, transparent=transparent)
    elif fmt == "svg":
        data = pdf_to_svg(pdf)
    else:
        raise ValueError(f"unsupported output format {fmt!r}; use pdf, png or svg")
    if cache:
        _cache_put(key, suffix, data)
    return data


def render_document(
    source: str,
    target: str | Path,
    *,
    engine: str = "pdflatex",
    dpi: int = 300,
    transparent: bool = False,
    cache: bool = True,
) -> Path:
    """Compile ``source`` into ``target`` (format from the file extension)."""
    target = Path(target)
    fmt = target.suffix.lower().lstrip(".") or "pdf"
    if not target.suffix:
        target = target.with_suffix(".pdf")
    data = compile_to_bytes(source, fmt, engine=engine, dpi=dpi, transparent=transparent, cache=cache)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target
