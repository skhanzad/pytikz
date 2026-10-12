"""``pytikz setup``: install what pytikz needs, without admin rights.

* No LaTeX: download TinyTeX (https://yihui.org/tinytex/, a small,
  self-contained TeX Live) from its GitHub releases into the user's home
  directory and add the TeX packages pytikz uses.
* LaTeX without some packages: install them with ``tlmgr`` when the TeX
  installation belongs to the user; otherwise print the command to run.
* No PDF-to-PNG converter: ``pip install pypdfium2``.

Nothing is changed before the plan has been shown and confirmed (or
``yes=True`` / ``--yes``).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TextIO

from . import toolchain
from .checks import TEX_REQUIREMENTS, Requirement, doctor, fix_commands, run_checks
from .toolchain import distribution, find_program, program_env, tinytex_dir

__all__ = ["setup", "install_tinytex", "tinytex_release", "tinytex_asset_name", "SetupError"]

REPO = "rstudio/tinytex-releases"
API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
PAGE_LATEST = f"https://github.com/{REPO}/releases/latest"
USER_AGENT = "pytikz-setup"


class SetupError(RuntimeError):
    """Installation failed; the message says what to do instead."""


# ---------------------------------------------------------------- TinyTeX release
def _is_musl() -> bool:
    return bool(list(Path("/lib").glob("ld-musl-*.so.1")) or list(Path("/lib").glob("libc.musl-*.so.1")))


def tinytex_asset_name(tag: str, system: str | None = None, machine: str | None = None,
                       musl: bool | None = None) -> str | None:
    """File name of the TinyTeX-1 bundle for this platform (``None`` if unsupported)."""
    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    if system == "Linux":
        musl = _is_musl() if musl is None else musl
        if machine in ("x86_64", "amd64"):
            arch = "linuxmusl-x86_64" if musl else "linux-x86_64"
        elif machine in ("aarch64", "arm64") and not musl:
            arch = "linux-arm64"
        else:
            return None
        return f"TinyTeX-1-{arch}-{tag}.tar.xz"
    if system == "Darwin":
        return f"TinyTeX-1-darwin-{tag}.tar.xz"
    if system == "Windows":
        return f"TinyTeX-1-windows-{tag}.exe"
    return None


@dataclass
class Release:
    tag: str
    name: str
    url: str
    size: int | None = None
    sha256: str | None = None


def _get(url: str, timeout: float = 30) -> tuple[bytes, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read(), response.geturl()


def tinytex_release(fetch: Callable[[str], tuple[bytes, str]] = _get) -> Release:
    """The latest TinyTeX-1 bundle for this platform (from the GitHub API, or
    from the release page if the API is unavailable)."""
    try:
        body, _ = fetch(API_LATEST)
        data = json.loads(body)
        tag = data["tag_name"]
        name = tinytex_asset_name(tag)
        if name is None:
            raise SetupError(_unsupported())
        for asset in data.get("assets", []):
            if asset.get("name") == name:
                digest = asset.get("digest") or ""
                sha = digest.split(":", 1)[1] if digest.startswith("sha256:") else None
                return Release(tag, name, asset["browser_download_url"], asset.get("size"), sha)
        raise SetupError(f"the latest TinyTeX release ({tag}) has no {name}")
    except SetupError:
        raise
    except Exception:  # noqa: BLE001 - API rate limits, proxies...: fall back to the page
        pass
    _, final_url = fetch(PAGE_LATEST)
    tag = final_url.rstrip("/").rsplit("/", 1)[-1]
    if not tag.startswith("v"):
        raise SetupError("could not find the latest TinyTeX release on GitHub")
    name = tinytex_asset_name(tag)
    if name is None:
        raise SetupError(_unsupported())
    return Release(tag, name, f"https://github.com/{REPO}/releases/download/{tag}/{name}")


def _unsupported() -> str:
    return (f"TinyTeX has no prebuilt bundle for {platform.system()} {platform.machine()}; "
            "install TeX Live (https://tug.org/texlive/) and run `pytikz doctor`")


# ---------------------------------------------------------------- download and unpack
def download(url: str, dest: Path, *, size: int | None = None, sha256: str | None = None,
             file: TextIO | None = None) -> None:
    """Download ``url`` to ``dest`` with a progress line; verify ``sha256``."""
    out = file or sys.stderr
    show = hasattr(out, "isatty") and out.isatty()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=60) as response, open(dest, "wb") as fh:
        total = size or int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = response.read(1 << 16)
            if not chunk:
                break
            fh.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            if show and total:
                print(f"\r  downloading {done / 1e6:5.1f} of {total / 1e6:.1f} MB", end="", file=out, flush=True)
    if show and total:
        print(file=out)
    if sha256 and digest.hexdigest() != sha256.lower():
        raise SetupError(f"checksum mismatch for {url}; please try again")


def _check_member(member: tarfile.TarInfo, base: Path) -> None:
    target = (base / member.name).resolve()
    if base != target and base not in target.parents:
        raise SetupError(f"refusing to extract {member.name!r}: outside the destination")
    if member.issym() or member.islnk():
        link = (target.parent / member.linkname).resolve() if member.issym() else (base / member.linkname).resolve()
        if base != link and base not in link.parents:
            raise SetupError(f"refusing to extract link {member.name!r} -> {member.linkname!r}")
    if member.isdev():
        raise SetupError(f"refusing to extract device file {member.name!r}")


def safe_extract(archive: Path, dest: Path) -> None:
    """Extract a tar archive, refusing paths and links that leave ``dest``."""
    base = dest.resolve()
    with tarfile.open(archive, "r:*") as tar:
        members = tar.getmembers()
        for member in members:
            _check_member(member, base)
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, members=members, filter="data")
        else:  # Python without extraction filters: members were checked above
            tar.extractall(dest, members=members)


def _unpack(archive: Path, target: Path) -> None:
    """Unpack a TinyTeX bundle so that it ends up exactly at ``target``."""
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pytikz-tinytex-", dir=target.parent))
    try:
        if archive.suffix == ".exe":  # Windows: self-extracting archive, unpacks ./TinyTeX
            subprocess.run([str(archive), "-y"], cwd=staging, check=True, capture_output=True)
        else:
            safe_extract(archive, staging)
        roots = [p for p in staging.iterdir() if p.is_dir()]
        if len(roots) != 1:
            raise SetupError(f"unexpected layout of {archive.name}")
        roots[0].rename(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _is_tex_tree(root: Path) -> bool:
    try:
        return any((d / name).exists() for d in (root / "bin").iterdir() if d.is_dir()
                   for name in ("tlmgr", "tlmgr.bat"))
    except OSError:
        return False


def tinytex_bin(root: Path) -> Path:
    """The ``bin/<platform>`` directory of a TeX Live/TinyTeX tree."""
    bins = sorted(p for p in (root / "bin").glob("*") if p.is_dir())
    for directory in bins:
        if any((directory / name).exists() for name in ("pdflatex", "pdflatex.exe")):
            return directory
    if bins:
        return bins[0]
    raise SetupError(f"no TeX binaries found in {root}")


def _tlmgr(bin_dir: Path) -> str:
    for name in ("tlmgr", "tlmgr.bat"):
        if (bin_dir / name).exists():
            return str(bin_dir / name)
    raise SetupError(f"tlmgr not found in {bin_dir}")


def _run(cmd: list[str], out: TextIO, what: str) -> None:
    print(f"  {what}", file=out, flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=program_env(cmd[0]))
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-12:])
        raise SetupError(f"`{' '.join(Path(c).name if i == 0 else c for i, c in enumerate(cmd))}` failed:\n{tail}")


def install_tex_packages(tlmgr: str, packages: list[str], out: TextIO, *, update_self: bool = True) -> None:
    """``tlmgr install`` the given TeX Live packages.

    tlmgr refuses to install anything when its package repository has a
    newer tlmgr, so it updates itself first.
    """
    if not packages:
        return
    if update_self:
        try:
            _run([tlmgr, "update", "--self"], out, "tlmgr update --self")
        except SetupError as err:  # not fatal: the install below may still work
            print(f"  (tlmgr could not update itself: {str(err).splitlines()[0]})", file=out)
    _run([tlmgr, "install", *packages], out, f"tlmgr install {' '.join(packages)}")


def install_tinytex(target: Path | None = None, *, add_to_path: bool = True, release: Release | None = None,
                    file: TextIO | None = None) -> Path:
    """Install TinyTeX plus pytikz's TeX packages; returns the ``bin`` directory."""
    out = file or sys.stdout
    target = Path(target) if target else tinytex_dir()
    if platform.system() != "Windows" and shutil.which("perl") is None:
        raise SetupError("TinyTeX's package manager needs Perl: install it first "
                         "(e.g. `sudo apt install perl`) and run `pytikz setup` again")
    if _is_tex_tree(target):
        print(f"  using the TeX installation already in {target}", file=out)
    elif target.exists() and any(target.iterdir()):
        raise SetupError(f"{target} already exists and is not a TeX installation; remove it or choose "
                         "another --dir")
    else:
        release = release or tinytex_release()
        with tempfile.TemporaryDirectory(prefix="pytikz-download-") as tmp:
            archive = Path(tmp) / release.name
            print(f"  downloading {release.url}", file=out, flush=True)
            download(release.url, archive, size=release.size, sha256=release.sha256)
            print(f"  unpacking into {target}", file=out, flush=True)
            if target.exists():
                target.rmdir()  # empty directory left by an earlier attempt
            _unpack(archive, target)
    bin_dir = tinytex_bin(target)
    tlmgr = _tlmgr(bin_dir)
    if add_to_path:
        try:
            if platform.system() == "Linux":
                user_bin = Path.home() / ".local" / "bin"
                if not user_bin.is_dir():
                    user_bin = Path.home() / "bin"
                user_bin.mkdir(parents=True, exist_ok=True)
                _run([tlmgr, "option", "sys_bin", str(user_bin)], out, f"linking TeX programs into {user_bin}")
                _run([tlmgr, "path", "add"], out, "tlmgr path add")
            elif platform.system() == "Windows" or os.access("/usr/local/bin", os.W_OK):
                _run([tlmgr, "path", "add"], out, "tlmgr path add")
        except SetupError as err:  # PATH is a convenience: pytikz finds TinyTeX anyway
            print(f"  (could not add TinyTeX to PATH: {err})", file=out)
    needed = sorted({r.texlive for r in TEX_REQUIREMENTS})
    install_tex_packages(tlmgr, needed, out)
    return bin_dir


# ---------------------------------------------------------------- Python extras
def _pip_install(package: str, out: TextIO) -> None:
    pip_ok = subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True).returncode == 0
    if pip_ok:
        cmd = [sys.executable, "-m", "pip", "install", package]
    elif shutil.which("uv"):
        cmd = [shutil.which("uv") or "uv", "pip", "install", "--python", sys.executable, package]
    else:
        raise SetupError(f"neither pip nor uv is available to install {package}")
    _run(cmd, out, f"installing {package} into {sys.executable}")
    importlib.invalidate_caches()


# ---------------------------------------------------------------- the command
def _confirm(question: str, yes: bool, out: TextIO) -> bool:
    if yes:
        return True
    try:
        answer = input(f"{question} [Y/n] ")
    except EOFError:
        print("\n(no answer; run `pytikz setup --yes` to proceed without asking)", file=out)
        return False
    return answer.strip().lower() in ("", "y", "yes")


def _tlmgr_for(root: str | None) -> str | None:
    if not root:
        return None
    if not os.access(Path(root) / "tlpkg", os.W_OK):
        return None
    return find_program("tlmgr")


def setup(*, yes: bool = False, engine: str = "pdflatex", tinytex: str | Path | None = None,
          add_to_path: bool = True, png: bool = True, file: TextIO | None = None) -> bool:
    """Install what pytikz needs, after showing the plan and asking.

    Returns ``True`` when pictures can be rendered afterwards.
    """
    out = file or sys.stdout
    toolchain.clear_cache()
    print("pytikz setup: checking what is installed...", file=out, flush=True)
    _, summary = run_checks(engine, compile_test=False)
    dist = distribution(engine)
    missing: list[Requirement] = summary["missing"]
    plan: list[tuple[str, Callable[[], None]]] = []

    custom_dir = Path(tinytex).expanduser() if tinytex else None
    use_tinytex = not summary["latex"] or custom_dir is not None
    tlmgr = _tlmgr_for(dist["root"]) if summary["latex"] else None
    required_missing = [r for r in missing if r.required]
    if summary["latex"] and required_missing and not tlmgr and not use_tinytex:
        if dist["kind"] == "miktex":
            print("\nMiKTeX installs missing packages (" + ", ".join(r.texlive for r in required_missing)
                  + ") automatically the first time they are needed; allow it when asked.", file=out)
        else:
            print("\nYour TeX installation lacks packages pytikz needs, and only an administrator can add "
                  "them:", file=out)
            for line in fix_commands(required_missing, need_latex=False, need_png=False, need_svg=False,
                                     engine=engine):
                print(f"  {line}", file=out)
            print("Alternatively pytikz can install its own small TeX distribution (TinyTeX) for you.", file=out)
            if not _confirm("Install TinyTeX for pytikz?", yes, out):
                return False
            use_tinytex = True

    if use_tinytex:
        target = custom_dir or tinytex_dir()
        resume = _is_tex_tree(target)
        try:
            release = None if resume else tinytex_release()
        except Exception as err:  # noqa: BLE001
            print(f"\nCould not look up TinyTeX: {err}", file=out)
            for line in fix_commands(list(TEX_REQUIREMENTS), need_latex=True, need_png=True, need_svg=True,
                                     engine=engine):
                print(f"  {line}", file=out)
            return False
        path_note = ("its programs are added to your PATH" if add_to_path
                     else "PATH is left unchanged (pytikz finds TinyTeX by itself)")

        def do_tinytex() -> None:
            bin_dir = install_tinytex(target, add_to_path=add_to_path, release=release, file=out)
            if custom_dir is not None or summary["latex"]:
                cfg = toolchain.save_config(texbin=str(bin_dir))
                print(f"  pytikz will use {bin_dir} (saved in {cfg})", file=out)
            toolchain.clear_cache()

        if release is None:
            text = (f"Finish the TeX installation in {target}:\n"
                    f"     add the TeX packages pytikz uses (pgf, standalone, ...); {path_note}")
        else:
            size = f" ({release.size / 1e6:.0f} MB)" if release.size else ""
            text = (f"Download TinyTeX {release.tag}{size} from github.com/{REPO}\n"
                    f"     and install it into {target}\n"
                    f"     with the TeX packages pytikz uses (pgf, standalone, ...);\n"
                    f"     {path_note}")
        plan.append((text, do_tinytex))
    elif missing and tlmgr:
        packages = sorted({r.texlive for r in missing})

        def do_packages() -> None:
            install_tex_packages(tlmgr, packages, out)
            toolchain.clear_cache()

        plan.append((f"Install the TeX packages {', '.join(packages)} with tlmgr", do_packages))

    if png and not summary["png"]:
        plan.append(("Install pypdfium2 (pip) so pictures can be saved as PNG",
                     lambda: _pip_install("pypdfium2", out)))

    if not plan:
        print("Nothing to install.\n", file=out)
        return doctor(engine, file=out)

    print("\nThis will:", file=out)
    for k, (text, _) in enumerate(plan, 1):
        print(f"  {k}. {text}", file=out)
    if not _confirm("Proceed?", yes, out):
        return False
    for text, action in plan:
        print(f"\n{text.splitlines()[0]}", file=out, flush=True)
        try:
            action()
        except (SetupError, OSError, subprocess.SubprocessError) as err:
            print(f"\nSetup failed: {err}", file=out)
            print("`pytikz doctor` shows other ways to install what is missing.", file=out)
            return False
    print(file=out)
    return doctor(engine, file=out)
