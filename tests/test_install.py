"""Tests of the installation helpers: program discovery, doctor, setup and
the pure-Python PNG path. Nothing here downloads anything."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

import pytikz as tz
from pytikz import checks, compiler, installer, toolchain


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTIKZ_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.setenv("PYTIKZ_CACHE", str(tmp_path / "cache"))  # no cached PDFs: LaTeX must run
    monkeypatch.delenv("PYTIKZ_TEXBIN", raising=False)
    # A TinyTeX in the default place (CI installs one before the tests) must not leak in.
    monkeypatch.setattr(toolchain, "tinytex_dir", lambda: tmp_path / "TinyTeX")
    monkeypatch.setattr(installer, "tinytex_dir", lambda: tmp_path / "TinyTeX")
    toolchain.clear_cache()
    yield
    toolchain.clear_cache()


def fake_program(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("#!/bin/sh\necho fake\n", encoding="utf-8")
    path.chmod(0o755)
    return path


# ---------------------------------------------------------------- discovery
def test_texbin_override_wins_over_path(tmp_path, monkeypatch):
    exe = fake_program(tmp_path / "tex", "pdflatex")
    monkeypatch.setenv("PYTIKZ_TEXBIN", str(tmp_path / "tex"))
    toolchain.clear_cache()
    assert toolchain.find_program("pdflatex") == str(exe)


def test_configured_tex_is_preferred(tmp_path):
    exe = fake_program(tmp_path / "chosen", "pdflatex")
    toolchain.save_config(texbin=str(tmp_path / "chosen"))
    toolchain.clear_cache()
    assert toolchain.find_program("pdflatex") == str(exe)
    assert json.loads(toolchain.config_path().read_text(encoding="utf-8"))["texbin"] == str(tmp_path / "chosen")


def test_programs_off_path_are_found_in_known_places(tmp_path, monkeypatch):
    monkeypatch.setattr(toolchain, "tinytex_dir", lambda: tmp_path / ".TinyTeX")
    platform_dir = "windows" if toolchain.SYSTEM == "Windows" else "x86_64-linux"
    exe = fake_program(tmp_path / ".TinyTeX" / "bin" / platform_dir, "pytikz-test-tool")
    toolchain.clear_cache()
    assert toolchain.find_program("pytikz-test-tool") == str(exe)
    assert toolchain.find_program("pytikz-surely-missing-tool") is None


def test_program_env_puts_the_program_first(tmp_path):
    env = toolchain.program_env(str(tmp_path / "bin" / "pdflatex"))
    assert env["PATH"].startswith(str(tmp_path / "bin"))


# ---------------------------------------------------------------- PNG encoding
def test_png_encoder_round_trip():
    pytest.importorskip("PIL")
    from PIL import Image

    rng = np.random.default_rng(0)
    for channels in (3, 4):
        pixels = rng.integers(0, 256, (7, 5, channels), dtype=np.uint8)
        data = compiler._encode_png(pixels, dpi=150)
        image = Image.open(io.BytesIO(data))
        assert np.array_equal(np.asarray(image), pixels)
        assert round(image.info["dpi"][0]) == 150


def test_missing_converters_raise_helpful_errors(monkeypatch):
    monkeypatch.setattr(compiler, "_PNG", [])
    monkeypatch.setattr(compiler, "_SVG", [])
    with pytest.raises(tz.ConverterNotFoundError, match="pytikz-diagrams\\[png\\]"):
        compiler.pdf_to_png(b"%PDF-1.5")
    with pytest.raises(tz.ConverterNotFoundError, match="pytikz doctor"):
        compiler.pdf_to_svg(b"%PDF-1.5")
    assert compiler.converters() == {"png": [], "svg": []}


def test_missing_latex_message_points_to_setup(monkeypatch):
    monkeypatch.setattr(compiler, "find_program", lambda name: None)
    with pytest.raises(tz.LatexNotFoundError, match="pytikz setup"):
        compiler.compile_pdf("\\documentclass{article}", cache=False)


def test_jupyter_display_without_latex(monkeypatch):
    monkeypatch.setattr(compiler, "find_program", lambda name: None)
    pic = tz.Picture()
    pic.node("x")
    bundle = pic._repr_mimebundle_()
    assert list(bundle) == ["text/plain"] and "pytikz setup" in bundle["text/plain"]


# ---------------------------------------------------------------- TinyTeX release lookup
@pytest.mark.parametrize(
    "system, machine, musl, expected",
    [
        ("Linux", "x86_64", False, "TinyTeX-1-linux-x86_64-v2026.10.tar.xz"),
        ("Linux", "aarch64", False, "TinyTeX-1-linux-arm64-v2026.10.tar.xz"),
        ("Linux", "x86_64", True, "TinyTeX-1-linuxmusl-x86_64-v2026.10.tar.xz"),
        ("Linux", "riscv64", False, None),
        ("Darwin", "arm64", False, "TinyTeX-1-darwin-v2026.10.tar.xz"),
        ("Windows", "AMD64", False, "TinyTeX-1-windows-v2026.10.exe"),
        ("FreeBSD", "amd64", False, None),
    ],
)
def test_asset_names(system, machine, musl, expected):
    assert installer.tinytex_asset_name("v2026.10", system, machine, musl) == expected


def test_release_from_api(monkeypatch):
    monkeypatch.setattr(installer, "tinytex_asset_name", lambda tag: f"TinyTeX-1-linux-x86_64-{tag}.tar.xz")
    payload = {
        "tag_name": "v2026.10",
        "assets": [
            {"name": "TinyTeX-1-darwin-v2026.10.tar.xz", "browser_download_url": "https://x/darwin", "size": 1},
            {"name": "TinyTeX-1-linux-x86_64-v2026.10.tar.xz", "browser_download_url": "https://x/linux",
             "size": 53_600_000, "digest": "sha256:" + "ab" * 32},
        ],
    }
    release = installer.tinytex_release(lambda url: (json.dumps(payload).encode(), url))
    assert (release.url, release.size, release.sha256) == ("https://x/linux", 53_600_000, "ab" * 32)


def test_release_falls_back_to_the_release_page(monkeypatch):
    monkeypatch.setattr(installer, "tinytex_asset_name", lambda tag: f"TinyTeX-1-darwin-{tag}.tar.xz")

    def fetch(url):
        if "api.github.com" in url:
            raise OSError("rate limited")
        return b"", "https://github.com/rstudio/tinytex-releases/releases/tag/v2026.10"

    release = installer.tinytex_release(fetch)
    assert release.url.endswith("/download/v2026.10/TinyTeX-1-darwin-v2026.10.tar.xz")


# ---------------------------------------------------------------- unpacking
def make_tar(path: Path, entries: dict[str, str], links: dict[str, str] | None = None) -> Path:
    with tarfile.open(path, "w:xz") as tar:
        for name, text in entries.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755
            tar.addfile(info, io.BytesIO(data))
        for name, target in (links or {}).items():
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            tar.addfile(info)
    return path


def test_unpack_places_the_tree_at_the_target(tmp_path):
    archive = make_tar(tmp_path / "t.tar.xz", {".TinyTeX/bin/x86_64-linux/pdftex": "x",
                                               ".TinyTeX/bin/x86_64-linux/tlmgr": "x"},
                       {".TinyTeX/bin/x86_64-linux/pdflatex": "pdftex"})
    target = tmp_path / "install" / "TinyTeX"
    installer._unpack(archive, target)
    assert (target / "bin" / "x86_64-linux" / "pdflatex").exists()
    assert installer.tinytex_bin(target) == target / "bin" / "x86_64-linux"
    assert installer._is_tex_tree(target)
    assert [p.name for p in target.parent.iterdir()] == ["TinyTeX"]  # staging dir removed


@pytest.mark.parametrize(
    "entries, links",
    [({"../evil.txt": "x"}, None), ({"ok/file": "x"}, {"ok/link": "/etc/passwd"}),
     ({"ok/file": "x"}, {"ok/link": "../../outside"})],
)
def test_unsafe_archives_are_rejected(tmp_path, entries, links):
    archive = make_tar(tmp_path / "bad.tar.xz", entries, links)
    with pytest.raises(installer.SetupError, match="refusing"):
        installer.safe_extract(archive, tmp_path / "dest")


def test_install_refuses_foreign_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(installer.shutil, "which", lambda name: "/usr/bin/perl")
    (tmp_path / "taken").mkdir()
    (tmp_path / "taken" / "notes.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(installer.SetupError, match="not a TeX installation"):
        installer.install_tinytex(tmp_path / "taken", file=io.StringIO())


# ---------------------------------------------------------------- setup and doctor flows
def summary(latex=True, missing=(), png=True, svg=True):
    return [], {"latex": latex, "missing": list(missing), "png": png, "svg": svg}


def no_tex(engine="pdflatex"):
    return {"path": None, "version": None, "kind": None, "root": None}


def test_setup_without_latex_offers_tinytex_and_respects_no(monkeypatch):
    monkeypatch.setattr(installer, "run_checks", lambda *a, **k: summary(latex=False, png=False))
    monkeypatch.setattr(installer, "distribution", no_tex)
    monkeypatch.setattr(installer, "tinytex_release",
                        lambda: installer.Release("v2026.10", "TinyTeX-1-linux-x86_64-v2026.10.tar.xz", "https://x", 53_600_000))
    called = []
    monkeypatch.setattr(installer, "install_tinytex", lambda *a, **k: called.append(1))
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    out = io.StringIO()
    assert installer.setup(file=out) is False
    text = out.getvalue()
    assert "Download TinyTeX v2026.10 (54 MB)" in text and "pypdfium2" in text
    assert not called


def test_setup_needs_yes_when_not_interactive(monkeypatch):
    monkeypatch.setattr(installer, "run_checks", lambda *a, **k: summary(png=False))
    monkeypatch.setattr(installer, "distribution", no_tex)

    def no_stdin(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", no_stdin)
    out = io.StringIO()
    assert installer.setup(file=out) is False
    assert "--yes" in out.getvalue()


def test_setup_with_nothing_to_do_runs_the_doctor(monkeypatch):
    monkeypatch.setattr(installer, "run_checks", lambda *a, **k: summary())
    monkeypatch.setattr(installer, "distribution", no_tex)
    monkeypatch.setattr(installer, "doctor", lambda engine, file: True)
    out = io.StringIO()
    assert installer.setup(file=out) is True
    assert "Nothing to install" in out.getvalue()


def test_fix_commands_for_debian(monkeypatch):
    monkeypatch.setattr(checks, "distribution", no_tex)
    monkeypatch.setattr(checks.platform, "system", lambda: "Linux")
    monkeypatch.setattr(checks, "linux_distro", lambda: {"ubuntu", "debian"})
    lines = checks.fix_commands([], need_latex=True, need_png=True, need_svg=True)
    assert lines[0].startswith("pytikz setup")
    assert any("sudo apt install" in ln and "texlive-pictures" in ln and "poppler-utils" in ln for ln in lines)


def test_fix_commands_for_missing_packages_on_debian_tex(monkeypatch):
    monkeypatch.setattr(checks, "distribution",
                        lambda engine="pdflatex": {"path": "/usr/bin/pdflatex", "version": "(TeX Live 2023/Debian)",
                                                   "kind": "texlive-debian", "root": "/usr/share/texlive"})
    monkeypatch.setattr(checks.platform, "system", lambda: "Linux")
    monkeypatch.setattr(checks, "linux_distro", lambda: {"debian"})
    standalone = next(r for r in checks.TEX_REQUIREMENTS if r.texlive == "standalone")
    lines = checks.fix_commands([standalone], need_latex=False, need_png=False, need_svg=False)
    assert lines == ["or with apt:  sudo apt install texlive-latex-extra"]


def test_doctor_report_lists_every_check(tmp_path):
    out = io.StringIO()
    ok = tz.doctor(file=out, compile_test=False)
    text = out.getvalue()
    for name in ("Python", "LaTeX", "PNG", "SVG"):
        assert name in text
    assert isinstance(ok, bool)
