"""Locate the external programs pytikz uses: LaTeX engines, kpsewhich,
tlmgr and PDF converters.

Programs are looked up on ``PATH`` first, then in the places TeX
distributions install themselves even when ``PATH`` does not mention them
(TinyTeX, MacTeX's ``/Library/TeX/texbin``, TeX Live and MiKTeX defaults,
Homebrew). ``PYTIKZ_TEXBIN`` (a directory) takes precedence over everything.
"""

from __future__ import annotations

import functools
import glob
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

__all__ = [
    "ENGINES",
    "find_program",
    "tex_bin_dirs",
    "tinytex_dir",
    "program_env",
    "clear_cache",
    "first_line",
    "kpsewhich",
    "distribution",
    "config_path",
    "preferred_dirs",
    "load_config",
    "save_config",
]

ENGINES = ("pdflatex", "lualatex", "xelatex")
SYSTEM = platform.system()


def tinytex_dir() -> Path:
    """Where TinyTeX lives by default (the official installers use these)."""
    if SYSTEM == "Darwin":
        return Path.home() / "Library" / "TinyTeX"
    if SYSTEM == "Windows":
        appdata = os.environ.get("APPDATA", "")
        if appdata and re.fullmatch(r"[!-~]+", appdata):  # no spaces or non-ASCII characters
            return Path(appdata) / "TinyTeX"
        return Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "TinyTeX"
    return Path.home() / ".TinyTeX"


def config_path() -> Path:
    """pytikz's settings file (``PYTIKZ_CONFIG`` overrides the location)."""
    override = os.environ.get("PYTIKZ_CONFIG")
    if override:
        return Path(override)
    if SYSTEM == "Windows":
        base = Path(os.environ.get("APPDATA") or Path.home())
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "pytikz" / "config.json"


def load_config() -> dict:
    try:
        data = json.loads(config_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_config(**values: object) -> Path:
    """Merge ``values`` into the settings file and return its path."""
    data = load_config()
    data.update(values)
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def _newest_first(pattern: str) -> list[Path]:
    return [Path(p) for p in sorted(glob.glob(pattern), reverse=True)]


def preferred_dirs() -> list[Path]:
    """Directories searched before ``PATH``: ``PYTIKZ_TEXBIN`` and the TeX
    installation chosen with ``pytikz setup``."""
    dirs: list[Path] = []
    override = os.environ.get("PYTIKZ_TEXBIN")
    if override:
        dirs.append(Path(override))
    configured = load_config().get("texbin")
    if configured:
        dirs.append(Path(configured))
    return dirs


def tex_bin_dirs() -> list[Path]:
    """Candidate directories with TeX binaries, besides ``PATH``."""
    dirs = preferred_dirs()
    tiny = tinytex_dir()
    if SYSTEM == "Windows":
        dirs += [tiny / "bin" / "windows", tiny / "bin" / "win64", tiny / "bin" / "win32"]
        dirs += _newest_first(r"C:\texlive\*\bin\windows") + _newest_first(r"C:\texlive\*\bin\win64")
        dirs += _newest_first(r"C:\texlive\*\bin\win32")
        for root in (os.environ.get("LOCALAPPDATA"), os.environ.get("ProgramFiles")):
            if root:
                dirs.append(Path(root) / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64")
                dirs.append(Path(root) / "MiKTeX" / "miktex" / "bin" / "x64")
    else:
        dirs += _newest_first(str(tiny / "bin" / "*"))
        if SYSTEM == "Darwin":
            dirs.append(Path("/Library/TeX/texbin"))
        dirs += _newest_first("/usr/local/texlive/*/bin/*")
    return dirs


def _tool_dirs() -> list[Path]:
    """Extra places for converters (poppler, Ghostscript, MuPDF)."""
    dirs = []
    if SYSTEM == "Darwin":
        dirs += [Path("/opt/homebrew/bin"), Path("/usr/local/bin"), Path("/opt/local/bin")]
    elif SYSTEM == "Windows":
        home = Path.home()
        dirs += [home / "scoop" / "shims", Path(r"C:\ProgramData\chocolatey\bin")]
        dirs += [Path(p) for p in sorted(glob.glob(r"C:\Program Files\gs\*\bin"), reverse=True)]
    return dirs


def _candidates(directory: Path, name: str) -> list[Path]:
    if SYSTEM == "Windows":
        return [directory / f"{name}{ext}" for ext in (".exe", ".bat", ".cmd", "")]
    return [directory / name]


@functools.lru_cache(maxsize=None)
def find_program(name: str) -> str | None:
    """Full path of ``name``, or ``None`` if it cannot be found."""
    for directory in preferred_dirs():
        for cand in _candidates(directory, name):
            if cand.is_file():
                return str(cand)
    found = shutil.which(name)
    if found:
        return found
    for directory in tex_bin_dirs() + _tool_dirs():
        for cand in _candidates(directory, name):
            if cand.is_file() and os.access(cand, os.X_OK):
                return str(cand)
    return None


def clear_cache() -> None:
    """Forget earlier lookups (after installing something)."""
    find_program.cache_clear()
    kpsewhich.cache_clear()
    distribution.cache_clear()


def program_env(program: str | None) -> dict[str, str]:
    """Environment for running ``program``: its directory comes first on
    ``PATH``, so helpers it calls (kpsewhich, mktexpk, gs) are found too."""
    env = dict(os.environ)
    if program:
        directory = str(Path(program).parent)
        env["PATH"] = directory + os.pathsep + env.get("PATH", "")
    return env


def first_line(*command: str, timeout: float = 30) -> str | None:
    """First line of a command's output, or ``None`` if it fails."""
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                              env=program_env(command[0]))
    except (OSError, subprocess.SubprocessError):
        return None
    text = (proc.stdout or proc.stderr or "").strip()
    return text.splitlines()[0] if text else None


@functools.lru_cache(maxsize=None)
def kpsewhich(filename: str) -> str | None:
    """Where TeX finds ``filename`` (``None`` if it does not)."""
    exe = find_program("kpsewhich")
    if exe is None:
        return None
    try:
        proc = subprocess.run([exe, filename], capture_output=True, text=True, timeout=30, env=program_env(exe))
    except (OSError, subprocess.SubprocessError):
        return None
    path = proc.stdout.strip()
    return path or None


@functools.lru_cache(maxsize=None)
def distribution(engine: str = "pdflatex") -> dict[str, str | None]:
    """What kind of TeX installation provides ``engine``.

    Returns ``{"path", "version", "kind", "root"}`` where ``kind`` is
    ``"tinytex"``, ``"texlive"``, ``"texlive-debian"``, ``"miktex"`` or
    ``None`` when the engine is missing.
    """
    exe = find_program(engine)
    if exe is None:
        return {"path": None, "version": None, "kind": None, "root": None}
    version = first_line(exe, "--version")
    real = os.path.realpath(exe)
    kind = "texlive"
    if "TinyTeX" in real or "TinyTeX" in exe:
        kind = "tinytex"
    elif version and "MiKTeX" in version:
        kind = "miktex"
    elif version and ("Debian" in version or "Ubuntu" in version):
        kind = "texlive-debian"
    root = None
    kpse = find_program("kpsewhich")
    if kpse:
        try:
            root = subprocess.run([kpse, "-var-value", "TEXMFROOT"], capture_output=True, text=True,
                                  timeout=30, env=program_env(kpse)).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            root = None
    return {"path": exe, "version": version, "kind": kind, "root": root}
