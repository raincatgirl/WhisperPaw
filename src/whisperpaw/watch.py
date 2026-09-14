"""paw-watch — tail a command's output and read new lines aloud.

Usage::

    paw-watch -- make              # run `make`, speak each new stdout line
    paw-watch --include-stderr -- pytest -q
    paw-watch --max-lines 5 -- seq 10

Design
------
A thin subprocess wrapper that streams a child process's stdout (and
optionally stderr) through the same TTS backend chain that
:mod:`whisperpaw.read` uses. Each complete line of output is spoken as
it arrives; partial lines are buffered until the next chunk or EOF.

Why reuse ``paw-read`` rather than duplicate the backend picker:
``paw-read`` already knows about every OS-native TTS engine, already
maps ``--rate`` / ``--volume`` to per-backend flags, and already has
its own test coverage. Importing it keeps this tool a thin shell and
gives us a single backend to maintain.

Backends (inherited from ``paw-read``):
    ``piper`` (if installed + a voice is available) → ``say`` (macOS) →
    ``spd-say`` → ``espeak`` (Linux) → SAPI (Windows).

Exit codes: 0 ok, 1 backend/TTS error, 2 usage / no command / spawn error.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Iterator

# Reuse the TTS chain from paw-read. Importing it here (rather than
# inside _speak_line) keeps the import graph trivial and means the
# two tools fail together if either is broken.
from whisperpaw import read as _read


#: Lowest acceptable speech rate in words-per-minute.
MIN_RATE: float = _read.MIN_RATE
#: Highest acceptable speech rate in words-per-minute.
MAX_RATE: float = _read.MAX_RATE


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paw-watch",
        description=(
            "Run a command and speak its output line-by-line. "
            "Use -- to separate paw-watch flags from the command, e.g. "
            "'paw-watch --max-lines 5 -- seq 10'."
        ),
    )
    parser.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help=(
            "Command to run. Everything after `--` is passed verbatim. "
            "Example: paw-watch -- echo hello"
        ),
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=200.0,
        help=f"Speech rate, words per minute ({int(MIN_RATE)}–{int(MAX_RATE)}, default: 200).",
    )
    parser.add_argument(
        "--volume",
        type=float,
        default=1.0,
        help="Playback volume, 0.0–1.0 (default: 1.0).",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=0,
        help=(
            "Speak at most this many lines (0 = unlimited, default: 0). "
            "After the limit, further output is dropped, not queued."
        ),
    )
    parser.add_argument(
        "--include-stderr",
        action="store_true",
        help="Also speak lines that arrive on stderr (in addition to stdout).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the announcement line (still speaks).",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    # argparse.REMAINDER includes the `--` separator if present; strip it
    # so the command list is just [argv0, argv1, ...].
    if args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    # Validate after parsing so the user gets a friendly message.
    if not args.cmd:
        print(
            "paw-watch: no command given — pass the command after `--`, "
            "e.g. `paw-watch -- echo hello`",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if not (MIN_RATE <= args.rate <= MAX_RATE):
        print(
            f"paw-watch: --rate must be between {int(MIN_RATE)} and {int(MAX_RATE)} wpm",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if not 0.0 <= args.volume <= 1.0:
        print("paw-watch: --volume must be between 0.0 and 1.0", file=sys.stderr)
        raise SystemExit(2)
    if args.max_lines < 0:
        print("paw-watch: --max-lines must be >= 0", file=sys.stderr)
        raise SystemExit(2)
    return args


# ---------------------------------------------------------------------------
# Line buffering
# ---------------------------------------------------------------------------


class _LineBuffer:
    """Accumulate bytes, yield one complete line per chunk.

    Handles both ``\\n`` and ``\\r\\n`` as line separators. Empty lines
    (after stripping) are dropped — they are not interesting to speak.
    Trailing whitespace on a line is stripped before yielding.
    """

    __slots__ = ("_buf",)

    def __init__(self) -> None:
        self._buf: str = ""

    def feed(self, chunk: str | bytes) -> Iterator[str]:
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8", errors="replace")
        data = self._buf + chunk
        while True:
            # Find the first line terminator: \n or \r\n.
            nl = data.find("\n")
            cr = data.find("\r")
            if cr != -1 and (nl == -1 or cr < nl):
                # CRLF or bare CR
                line = data[:cr]
                data = data[cr + 1:]
                if data.startswith("\n"):
                    data = data[1:]
            elif nl != -1:
                line = data[:nl]
                data = data[nl + 1:]
            else:
                # No complete line yet — keep buffering.
                self._buf = data
                return
            line = line.strip()
            if line:
                yield line

    def flush(self) -> list[str]:
        """Drain the buffer, yielding whatever's left (even without a newline)."""
        leftover = self._buf.strip()
        self._buf = ""
        return [leftover] if leftover else []


# ---------------------------------------------------------------------------
# Subprocess + TTS adapters (so tests can monkey-patch them cleanly)
# ---------------------------------------------------------------------------


def _spawn(cmd: list[str], *, include_stderr: bool) -> subprocess.CompletedProcess:
    """Spawn ``cmd`` and return its CompletedProcess.

    Stdout is captured as text; stderr is captured as text only when
    ``include_stderr`` is true (so we don't pay the cost otherwise).
    """
    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=None,
        stderr=subprocess.STDOUT if include_stderr else subprocess.PIPE,
    )


def _speak_line(line: str, rate: float, volume: float) -> int:
    """Speak a single line via the ``paw-read`` TTS chain.

    Returns the TTS exit code (0 ok, non-zero error). Long lines are
    chunked first so we don't depend on the engine's own input limit.
    """
    chunks = _read.chunk_text(line)
    last = 0
    for chunk in chunks:
        code = _read._speak(
            chunk, rate, volume, backend="auto", piper_voice="auto"
        )
        if code != 0:
            last = code
    return last


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2

    if not args.quiet:
        preview = " ".join(args.cmd)
        if len(preview) > 60:
            preview = preview[:57] + "..."
        print(f"🐾 paw-watch: tailing `{preview}`")

    # Spawn the watched process. FileNotFoundError (binary missing) is
    # mapped to exit 2 — this is a usage error from the user's side.
    try:
        completed = _spawn(args.cmd, include_stderr=args.include_stderr)
    except FileNotFoundError as exc:
        print(f"paw-watch: command not found: {exc.filename or args.cmd[0]}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"paw-watch: could not spawn {args.cmd[0]!r}: {exc}", file=sys.stderr)
        return 2

    # When --include-stderr was set, _spawn redirected stderr into
    # stdout, so completed.stderr is always empty here. We only feed
    # stdout to the line buffer in either case.
    buffer = _LineBuffer()
    spoken = 0
    tts_error = 0
    max_lines = args.max_lines  # 0 == unlimited
    for line in buffer.feed(completed.stdout or ""):
        if max_lines and spoken >= max_lines:
            break
        code = _speak_line(line, args.rate, args.volume)
        spoken += 1
        if code != 0:
            tts_error = code
    # Anything left in the buffer (no trailing newline) is one final line.
    for line in buffer.flush():
        if max_lines and spoken >= max_lines:
            break
        code = _speak_line(line, args.rate, args.volume)
        spoken += 1
        if code != 0:
            tts_error = code

    # Mirror the child's exit code if it failed. Otherwise prefer the
    # TTS error if there was one (more useful for debugging audio).
    if completed.returncode != 0:
        return completed.returncode
    if tts_error:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
