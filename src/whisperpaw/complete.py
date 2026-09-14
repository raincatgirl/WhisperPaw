"""paw-complete — print shell completion code for the paw-* tools.

Usage::

    eval "$(paw-complete bash)"        # add to ~/.bashrc
    paw-complete zsh > ~/.zfunc/_paw    # or as an autoloaded file
    paw-complete fish > ~/.config/fish/completions/paw-read.fish
    paw-complete nu >> ~/.config/nushell/config.nu

The script does not require any environment setup; it just calls into
:mod:`whisperpaw._completions` and prints the result. Exit codes:
``0`` on success, ``2`` on usage error (unknown shell, no argument).
"""
from __future__ import annotations

import argparse
import sys

from whisperpaw import _completions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paw-complete",
        description=(
            "Print shell completion code for the paw-* tools. "
            "Common usage: 'eval \"$(paw-complete bash)\"' or "
            "'paw-complete zsh > ~/.zfunc/_paw'."
        ),
    )
    parser.add_argument(
        "shell",
        nargs="?",
        choices=_completions.SUPPORTED_SHELLS,
        help=(
            "Which shell to emit completions for. One of: "
            + ", ".join(_completions.SUPPORTED_SHELLS)
            + ". Required unless --list-tools is given."
        ),
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="Print the subcommand names we ship completions for, one per line.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2

    if args.list_tools:
        for name in _completions.list_tools():
            print(name)
        return 0

    if not args.shell:
        print(
            "paw-complete: missing required shell argument. "
            "Pass one of: " + ", ".join(_completions.SUPPORTED_SHELLS),
            file=sys.stderr,
        )
        return 2

    sys.stdout.write(_completions.render(args.shell))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
