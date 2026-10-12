"""Compile LaTeX documents to PDF, PNG and SVG.

LaTeX (``pdflatex``, ``lualatex`` or ``xelatex``) is found on ``PATH`` or in
the usual install locations (see :mod:`pytikz.toolchain`). PDFs are turned
into PNG/SVG by whichever converter is available:

* PNG: poppler (``pdftocairo``/``pdftoppm``), the ``pypdfium2`` Python package
  (``pip install "pytikz-diagrams[png]"``), Ghostscript, MuPDF or ImageMagick.
* SVG: poppler (``pdftocairo``), ``dvisvgm`` (with Ghostscript) or MuPDF.

Results are cached by content hash in ``~/.cache/pytikz`` (override with
``PYTIKZ_CACHE``), so re-rendering an unchanged picture is instant. Run
``pytikz doctor`` to see what is installed and ``pytikz setup`` to install
what is missing.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path
from typing import Callable

from .toolchain import find_program, program_env

__all__ = [
    "ToolNotFoundError",
    "LatexNotFoundError",
    "ConverterNotFoundError",
    "TikzCompileError",
    "compile_pdf",
    "compile_to_bytes",
    "render_document",
    "pdf_to_png",
    "pdf_to_svg",
    "latex_available",
    "converters",
]

SETUP_HINT = "Run `pytikz doctor` to see what is missing and how to install it."


class ToolNotFoundError(RuntimeError):
    """A required external program is not installed."""


class LatexNotFoundError(ToolNotFoundError):
    """No LaTeX engine was found."""


class ConverterNotFoundError(ToolNotFoundError):
    """No program to convert PDF to PNG or SVG was found."""


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

    @property
    def missing_fonts(self) -> bool:
        """True when LaTeX stopped because font metric files are missing."""
        return "Metric (TFM) file not found" in self.log or "Metric (TFM) file not found" in self.message


def latex_available(engine: str = "pdflatex") -> bool:
    """Whether ``engine`` is installed."""
    return find_program(engine) is not None


# ---------------------------------------------------------------- cache
def _cache_dir() -> Path:
    root = os.environ.get("PYTIKZ_CACHE")
    return Path(root) if root else Path.home() / ".cache" / "pytikz"


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


# ---------------------------------------------------------------- LaTeX
_FILE_LINE = re.compile(r"^(?:\./)?(?P<file>[^:\n]+?):(?P<line>\d+): (?P<msg>.*)$", re.M)


def _parse_log(log: str, source: str, engine: str) -> TikzCompileError:
    match = next(_FILE_LINE.finditer(log), None)
    if match is not None:
        message = match.group("msg").strip()
        # Add the continuation of multi-line messages.
        for extra in log[match.end():].splitlines()[1:6]:
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
    exe = find_program(engine)
    if exe is None:
        raise LatexNotFoundError(
            f"{engine} was not found. Run `pytikz setup` to install a small LaTeX distribution "
            "(TinyTeX: a ~55 MB download, no admin rights needed), or `pytikz doctor` for the commands "
            "to install one with your package manager. You can also save the .tex file "
            "(pic.save('figure.tex')) and compile it elsewhere, e.g. on Overleaf."
        )
    with tempfile.TemporaryDirectory(prefix="pytikz-") as tmp:
        tex = Path(tmp) / "figure.tex"
        tex.write_text(source, encoding="utf-8")
        cmd = [exe, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "figure.tex"]
        env = program_env(exe)
        # keep TeX from wrapping log lines at 79 characters
        env.update(max_print_line="10000", error_line="254", half_error_line="238")
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


# ---------------------------------------------------------------- PNG
def _run(cmd: list[str], **kwargs: object) -> None:
    subprocess.run(cmd, check=True, capture_output=True, env=program_env(cmd[0]), **kwargs)  # type: ignore[call-overload]


def _encode_png(pixels: "object", dpi: int) -> bytes:
    """Minimal PNG encoder for an (h, w, 3|4) uint8 array (RGB or RGBA)."""
    import numpy as np

    arr = np.ascontiguousarray(pixels, dtype=np.uint8)
    height, width, channels = arr.shape
    color_type = {3: 2, 4: 6}[channels]
    rows = np.empty((height, width * channels + 1), dtype=np.uint8)
    rows[:, 0] = 0  # filter type "none" for every row
    rows[:, 1:] = arr.reshape(height, -1)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    per_metre = round(dpi / 0.0254)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0))
        + chunk(b"pHYs", struct.pack(">IIB", per_metre, per_metre, 1))
        + chunk(b"IDAT", zlib.compress(rows.tobytes(), 9))
        + chunk(b"IEND", b"")
    )


def _png_pdftocairo(src: Path, out: Path, dpi: int, transparent: bool) -> None:
    cmd = [find_program("pdftocairo"), "-png", "-r", str(dpi), "-singlefile"]  # type: ignore[list-item]
    if transparent:
        cmd.append("-transp")
    _run(cmd + [str(src), str(out.with_suffix(""))])


def _png_pdftoppm(src: Path, out: Path, dpi: int, transparent: bool) -> None:
    _run([find_program("pdftoppm"), "-png", "-r", str(dpi), "-singlefile",  # type: ignore[list-item]
          str(src), str(out.with_suffix(""))])


def _png_pdfium(src: Path, out: Path, dpi: int, transparent: bool) -> None:
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(src))
    try:
        fill = (0, 0, 0, 0) if transparent else (255, 255, 255, 255)
        bitmap = document[0].render(scale=dpi / 72.0, fill_color=fill, rev_byteorder=True)
        out.write_bytes(_encode_png(bitmap.to_numpy(), dpi))
    finally:
        document.close()


def _gs() -> str | None:
    return find_program("gs") or find_program("gswin64c") or find_program("gswin32c")


def _png_ghostscript(src: Path, out: Path, dpi: int, transparent: bool) -> None:
    device = "pngalpha" if transparent else "png16m"
    _run([_gs(), "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", f"-sDEVICE={device}", f"-r{dpi}",  # type: ignore[list-item]
          "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4", "-dFirstPage=1", "-dLastPage=1",
          f"-sOutputFile={out}", str(src)])


def _png_mutool(src: Path, out: Path, dpi: int, transparent: bool) -> None:
    _run([find_program("mutool"), "draw", "-q", "-r", str(dpi), "-c",  # type: ignore[list-item]
          "rgba" if transparent else "rgb", "-o", str(out), str(src), "1"])


def _png_magick(src: Path, out: Path, dpi: int, transparent: bool) -> None:
    cmd = [find_program("magick") or find_program("convert"), "-density", str(dpi), f"{src}[0]"]
    if not transparent:
        cmd += ["-background", "white", "-alpha", "remove"]
    _run(cmd + [str(out)])  # type: ignore[arg-type]


def _has_pdfium() -> bool:
    return importlib.util.find_spec("pypdfium2") is not None


#: PNG converters in order of preference: (name, available?, convert, can do transparency)
_PNG: list[tuple[str, Callable[[], bool], Callable[..., None], bool]] = [
    ("pdftocairo", lambda: find_program("pdftocairo") is not None, _png_pdftocairo, True),
    ("pdftoppm", lambda: find_program("pdftoppm") is not None, _png_pdftoppm, False),
    ("pypdfium2", _has_pdfium, _png_pdfium, True),
    ("ghostscript", lambda: _gs() is not None, _png_ghostscript, True),
    ("mutool", lambda: find_program("mutool") is not None, _png_mutool, True),
    ("imagemagick", lambda: (find_program("magick") or find_program("convert")) is not None, _png_magick, True),
]


def pdf_to_png(pdf: bytes, *, dpi: int = 300, transparent: bool = False) -> bytes:
    """Rasterise the first page of a PDF with the first converter that works."""
    errors = []
    with tempfile.TemporaryDirectory(prefix="pytikz-") as tmp:
        src = Path(tmp) / "in.pdf"
        src.write_bytes(pdf)
        out = Path(tmp) / "out.png"
        for name, available, convert, alpha in _PNG:
            if (transparent and not alpha) or not available():
                continue
            try:
                convert(src, out, dpi, transparent)
                return out.read_bytes()
            except (OSError, subprocess.SubprocessError, ImportError, RuntimeError) as err:
                errors.append(f"{name}: {err}")
    detail = f" (tried: {'; '.join(errors)})" if errors else ""
    raise ConverterNotFoundError(
        "No working PDF-to-PNG converter was found" + detail + ". The easiest fix is "
        '`pip install "pytikz-diagrams[png]"` (installs pypdfium2); poppler, Ghostscript, MuPDF or '
        "ImageMagick also work. " + SETUP_HINT
    )


# ---------------------------------------------------------------- SVG
def _svg_pdftocairo(src: Path, out: Path) -> None:
    _run([find_program("pdftocairo"), "-svg", str(src), str(out)])  # type: ignore[list-item]


def _svg_dvisvgm(src: Path, out: Path) -> None:
    _run([find_program("dvisvgm"), "--pdf", "--no-fonts", "--exact-bbox", "-o", str(out), str(src)])  # type: ignore[list-item]


def _svg_mutool(src: Path, out: Path) -> None:
    _run([find_program("mutool"), "draw", "-q", "-F", "svg", "-o", str(out), str(src), "1"])  # type: ignore[list-item]


_SVG: list[tuple[str, Callable[[], bool], Callable[[Path, Path], None]]] = [
    ("pdftocairo", lambda: find_program("pdftocairo") is not None, _svg_pdftocairo),
    ("dvisvgm", lambda: find_program("dvisvgm") is not None and _gs() is not None, _svg_dvisvgm),
    ("mutool", lambda: find_program("mutool") is not None, _svg_mutool),
]


def pdf_to_svg(pdf: bytes) -> bytes:
    """Convert the first page of a PDF to SVG with the first converter that works."""
    errors = []
    with tempfile.TemporaryDirectory(prefix="pytikz-") as tmp:
        src = Path(tmp) / "in.pdf"
        src.write_bytes(pdf)
        out = Path(tmp) / "out.svg"
        for name, available, convert in _SVG:
            if not available():
                continue
            try:
                convert(src, out)
                data = out.read_bytes()
                if b"<svg" in data[:2000]:
                    return data
                errors.append(f"{name}: produced no SVG")
            except (OSError, subprocess.SubprocessError) as err:
                errors.append(f"{name}: {err}")
    detail = f" (tried: {'; '.join(errors)})" if errors else ""
    raise ConverterNotFoundError(
        "SVG output needs poppler (pdftocairo), dvisvgm with Ghostscript, or MuPDF (mutool)"
        + detail + ". " + SETUP_HINT
    )


def converters() -> dict[str, list[str]]:
    """Names of the available PNG and SVG converters, best first."""
    return {
        "png": [name for name, available, _, _ in _PNG if available()],
        "svg": [name for name, available, _ in _SVG if available()],
    }


# ---------------------------------------------------------------- public API
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
    if fmt not in ("pdf", "png", "svg"):
        raise ValueError(f"unsupported output format {fmt!r}; use pdf, png or svg")
    pdf = compile_pdf(source, engine=engine, cache=cache)
    if fmt == "pdf":
        return pdf
    key = hashlib.sha256(pdf).hexdigest()[:32]
    suffix = f".{dpi}{'t' if transparent else ''}.png" if fmt == "png" else f".{fmt}"
    if cache:
        hit = _cache_get(key, suffix)
        if hit is not None:
            return hit
    data = pdf_to_png(pdf, dpi=dpi, transparent=transparent) if fmt == "png" else pdf_to_svg(pdf)
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
