"""Command line interface: turn pytikz scripts into TikZ, PDF, PNG and SVG.

::

    pytikz render examples/mlp.py -f pdf,png      # compile every picture
    pytikz tex examples/mlp.py                     # print the TikZ code
    pytikz watch examples/mlp.py -f png            # re-render on save
    pytikz doctor                                  # check LaTeX and converters
    pytikz setup                                   # install what is missing

A script's pictures are all the :class:`~pytikz.Picture` objects it
creates; module-level variable names are used for the output files
(``picture<k>`` otherwise). The script is
run with ``__name__ == "__pytikz__"``, so code under
``if __name__ == "__main__":`` is skipped.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import runpy
import sys
import time
import traceback
from pathlib import Path
from typing import Iterator

from . import __version__
from .compiler import LatexNotFoundError, TikzCompileError
from .picture import Picture

FORMATS = ("pdf", "png", "svg", "tex", "tikz")


@contextlib.contextmanager
def _collect() -> Iterator[list[Picture]]:
    created: list[Picture] = []
    original = Picture.__init__

    def tracking_init(self: Picture, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        original(self, *args, **kwargs)
        created.append(self)

    Picture.__init__ = tracking_init  # type: ignore[method-assign]
    try:
        yield created
    finally:
        Picture.__init__ = original  # type: ignore[method-assign]


def load_pictures(script: Path) -> list[tuple[str, Picture]]:
    """Run ``script`` and return ``(name, picture)`` pairs in creation order."""
    script = script.resolve()
    sys.path.insert(0, str(script.parent))
    try:
        with _collect() as created:
            namespace = runpy.run_path(str(script), run_name="__pytikz__")
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(str(script.parent))
    names = {id(v): k for k, v in namespace.items() if isinstance(v, Picture) and not k.startswith("_")}
    out = []
    for k, pic in enumerate(created):
        out.append((names.get(id(pic), f"picture{k + 1}"), pic))
    return out


def _targets(script: Path, outdir: Path | None, name: str, count: int, fmt: str) -> Path:
    stem = script.stem if count == 1 else f"{script.stem}-{name}"
    directory = outdir if outdir is not None else script.parent
    return directory / f"{stem}.{fmt}"


def render_script(script: Path, formats: list[str], outdir: Path | None, dpi: int, transparent: bool,
                  quiet: bool = False) -> list[Path]:
    pictures = load_pictures(script)
    if not pictures:
        raise SystemExit(f"{script}: no Picture objects were created")
    written = []
    for name, pic in pictures:
        for fmt in formats:
            target = _targets(script, outdir, name, len(pictures), fmt)
            t0 = time.perf_counter()
            pic.render(target, dpi=dpi, transparent=transparent)
            written.append(target)
            if not quiet:
                print(f"  {target}  ({time.perf_counter() - t0:.2f}s)", flush=True)
    return written


def _cmd_render(args: argparse.Namespace) -> int:
    formats = _parse_formats(args.format)
    failures = 0
    for script in args.scripts:
        print(f"{script}:", flush=True)
        try:
            render_script(Path(script), formats, Path(args.output) if args.output else None, args.dpi,
                          args.transparent)
        except (TikzCompileError, LatexNotFoundError) as err:
            failures += 1
            print(f"  error: {err}", file=sys.stderr)
        except Exception:  # noqa: BLE001 - report and continue with the next script
            failures += 1
            traceback.print_exc()
    return 1 if failures else 0


def _cmd_tex(args: argparse.Namespace) -> int:
    pictures = load_pictures(Path(args.script))
    for k, (name, pic) in enumerate(pictures):
        if len(pictures) > 1:
            print(f"% ---- {name} ----")
        print(pic.document() if args.standalone else pic.tikz(), end="")
        if k < len(pictures) - 1:
            print()
    return 0


def _cmd_watch(args: argparse.Namespace) -> int:
    script = Path(args.script)
    formats = _parse_formats(args.format)
    outdir = Path(args.output) if args.output else None
    print(f"watching {script} (Ctrl+C to stop)")
    last = None
    try:
        while True:
            try:
                mtime = os.stat(script).st_mtime
            except FileNotFoundError:
                mtime = None
            if mtime != last and mtime is not None:
                last = mtime
                stamp = time.strftime("%H:%M:%S")
                try:
                    render_script(script, formats, outdir, args.dpi, args.transparent, quiet=True)
                    print(f"[{stamp}] rendered")
                except (TikzCompileError, LatexNotFoundError) as err:
                    print(f"[{stamp}] error: {err}", file=sys.stderr)
                except Exception:  # noqa: BLE001
                    print(f"[{stamp}] script failed:", file=sys.stderr)
                    traceback.print_exc()
            time.sleep(0.4)
    except KeyboardInterrupt:
        return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    from .checks import doctor

    return 0 if doctor(args.engine, compile_test=not args.no_compile) else 1


def _cmd_setup(args: argparse.Namespace) -> int:
    from .installer import setup

    ok = setup(yes=args.yes, engine=args.engine, tinytex=args.dir, add_to_path=not args.no_path,
               png=not args.no_png)
    return 0 if ok else 1


def _parse_formats(text: str) -> list[str]:
    formats = [f.strip().lower().lstrip(".") for f in text.split(",") if f.strip()]
    bad = [f for f in formats if f not in FORMATS]
    if bad:
        raise SystemExit(f"unknown format(s) {bad}; choose from {', '.join(FORMATS)}")
    return formats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pytikz", description="Python scripts in, TikZ diagrams out.")
    parser.add_argument("--version", action="version", version=f"pytikz {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def output_options(p: argparse.ArgumentParser, default: str) -> None:
        p.add_argument("-f", "--format", default=default, help=f"comma separated: {', '.join(FORMATS)} (default: {default})")
        p.add_argument("-o", "--output", help="output directory (default: next to the script)")
        p.add_argument("--dpi", type=int, default=300, help="PNG resolution (default: 300)")
        p.add_argument("--transparent", action="store_true", help="transparent PNG background")

    p = sub.add_parser("render", help="compile the pictures of one or more scripts")
    p.add_argument("scripts", nargs="+")
    output_options(p, "pdf")
    p.set_defaults(func=_cmd_render)

    p = sub.add_parser("tex", help="print the TikZ code of a script's pictures")
    p.add_argument("script")
    p.add_argument("--standalone", action="store_true", help="print a complete LaTeX document")
    p.set_defaults(func=_cmd_tex)

    p = sub.add_parser("watch", help="re-render a script whenever it changes")
    p.add_argument("script")
    output_options(p, "pdf,png")
    p.set_defaults(func=_cmd_watch)

    p = sub.add_parser("doctor", help="check LaTeX, TeX packages and converters; explain how to fix problems")
    p.add_argument("--engine", default="pdflatex", choices=["pdflatex", "lualatex", "xelatex"])
    p.add_argument("--no-compile", action="store_true", help="skip the test compilation")
    p.set_defaults(func=_cmd_doctor)

    p = sub.add_parser("setup", help="install what is missing (TinyTeX if there is no LaTeX)")
    p.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")
    p.add_argument("--dir", help="where to install TinyTeX (default: the standard TinyTeX location)")
    p.add_argument("--no-path", action="store_true", help="do not add TinyTeX's programs to PATH")
    p.add_argument("--no-png", action="store_true", help="do not install pypdfium2 for PNG output")
    p.add_argument("--engine", default="pdflatex", choices=["pdflatex", "lualatex", "xelatex"])
    p.set_defaults(func=_cmd_setup)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
