"""paw-watch — tail a command's output and read new lines aloud.

Usage::

    paw-watch -- make              # run `make`, speak each new stdout line
    paw-watch --include-stderr -- pytest -q
    paw-watch --max-lines 5 -- seq 10
    paw-watch --follow -- tail -f /var/log/syslog
    paw-watch --follow --make watch
    paw-watch --dry-run -- pytest -q   # print what would be spoken, no sound

Design
------
A thin subprocess wrapper that streams a child process's stdout (and
optionally stderr) through the same TTS backend chain that
:mod:`whisperpaw.read` uses. Each complete line of output is spoken as
it arrives; partial lines are buffered until the next chunk or EOF.

Two execution modes:

- **batch** (default) — wait for the child to finish, then speak every
  line in order. Best for short-lived commands (``make``, ``pytest``).
- **streaming** (``--follow``) — open the child with ``Popen``, read
  stdout line-by-line, and speak each one as it arrives. Best for
  long-running watchers (``tail -f``, ``make watch``, ``npm run dev``).

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
from pathlib import Path
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
    parser.add_argument(
        "--follow",
        action="store_true",
        help=(
            "Stream the output instead of waiting for the command to finish: "
            "each new line is spoken as it arrives, then the exit code is "
            "mirrored. Useful for `tail -f`, `make watch`, `npm run dev`, etc."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print each line that *would* be spoken to stdout, one per "
            "line, instead of calling the TTS engine. The watched "
            "command still runs. Useful for previewing output, piping "
            "to a pager, or debugging --max-lines / --include-stderr "
            "settings without sound. Exits with the child's exit code "
            "(never a TTS error) so it composes with `set -e` scripts."
        ),
    )
    parser.add_argument(
        "--transcript",
        default=None,
        metavar="PATH",
        help=(
            "Also append every line that is spoken (or, with --dry-run, "
            "that would be spoken) to PATH, one line per row, UTF-8, "
            "opened in append mode. Useful for an accessibility log of "
            "what was announced: ``paw-watch --transcript ~/.local/"
            "share/whisperpaw/session.log -- make`` keeps a record of "
            "every line the screen reader heard. The file is created "
            "if missing; its parent directory must already exist. "
            "A failed write is reported on stderr but does not change "
            "the exit code (the spoken/printed output is the source "
            "of truth)."
        ),
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
    if args.transcript is not None:
        _validate_transcript_path(args.transcript)
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

    When ``include_stderr`` is true we redirect stderr into stdout
    (``stderr=STDOUT``) so the user gets a single ordered stream and we
    never deadlock on a full stderr pipe. In that case we set
    ``stdout=PIPE`` explicitly and skip ``capture_output``, because
    ``capture_output=True`` is incompatible with a custom ``stderr=``.
    """
    if include_stderr:
        return subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=None,
        )
    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=None,
    )


class _StreamProcess:
    """A tiny adapter over ``subprocess.Popen`` for the ``--follow`` path.

    The default ``Popen`` object is already pretty close to what we want,
    but wrapping it lets the tests substitute a deterministic fake without
    having to import ``subprocess`` machinery. Two methods are enough:

    - ``stdout_iter()`` yields one raw line at a time (with the trailing
      ``\\n`` if present) and stops at EOF.
    - ``wait()`` blocks until the child exits and returns the exit code.
    """

    __slots__ = ("_proc",)

    def __init__(self, proc: "subprocess.Popen[str]") -> None:
        self._proc = proc

    def stdout_iter(self):
        if self._proc.stdout is None:
            return
        for line in self._proc.stdout:
            yield line

    def wait(self) -> int:
        return self._proc.wait()


def _popen(cmd: list[str], *, include_stderr: bool) -> _StreamProcess:
    """Start ``cmd`` in streaming mode and return a :class:`_StreamProcess`.

    The child's stdout is opened as a text-mode pipe with line-buffering
    (``bufsize=1``) so each ``for line in proc.stdout`` read returns at
    most one line. Stderr is either captured separately (``include_stderr``
    is false) or merged into stdout so the user gets a single ordered
    stream and we never deadlock on a full stderr pipe.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT if include_stderr else subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    return _StreamProcess(proc)


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


def _emit_line(
    line: str,
    *,
    rate: float,
    volume: float,
    dry_run: bool,
    transcript_write=None,
) -> int:
    """Speak ``line`` (or print it in dry-run mode) and return the TTS code.

    Centralises the "speak vs. print" branch so the batch and streaming
    paths stay symmetric. In dry-run mode the line goes to stdout and we
    return 0 — the dry-run path never reports a TTS error.

    If ``transcript_write`` is supplied (a ``callable[[str], None]``),
    the line is also forwarded to it before we speak/print. The same
    line is written in speak mode and in dry-run mode, because the
    transcript is the record of "what the user heard" — and in
    dry-run mode the line *is* the announcement.
    """
    if transcript_write is not None:
        try:
            transcript_write(line)
        except OSError as exc:
            # A failed write is loud on stderr but not fatal — the
            # spoken/printed output is still the source of truth.
            print(
                f"paw-watch: --transcript write failed: {exc}",
                file=sys.stderr,
            )
    if dry_run:
        print(line)
        return 0
    return _speak_line(line, rate, volume)


def _validate_transcript_path(path: str) -> None:
    """Validate the ``--transcript`` target at parse time.

    We accept the path if the parent directory exists (or is ``""`` /
    ``"."`` — i.e. the current working directory) and is a directory.
    The transcript file itself is created on first write, so it does
    not need to exist yet. A bad path is reported on stderr and exits
    2, matching the other ``--foo must be …`` validations in
    :func:`parse_args`.
    """
    p = Path(path)
    parent = p.parent if str(p.parent) else Path(".")
    if not parent.is_dir():
        print(
            f"paw-watch: --transcript parent directory does not exist: "
            f"{parent}",
            file=sys.stderr,
        )
        raise SystemExit(2)


def open_transcript(path: str | None):
    """Open the transcript file in append mode, or return ``None``.

    Returns a text-mode file handle opened with ``encoding="utf-8"``,
    ``newline=""`` (so the Python runtime doesn't translate ``\n`` to
    the platform default) and a leading BOM-less write. ``None`` is
    returned when no ``--transcript`` was requested so the caller can
    treat the handle uniformly.

    The handle is the caller's to close — :func:`main` wraps the
    speech loop in a ``try/finally`` that always closes it, so a
    failing TTS chain or a KeyboardInterrupt cannot leak an
    unflushed handle on long-running ``--follow`` runs.
    """
    if path is None:
        return None
    # ``newline=""`` lets us write ``\\n`` and get ``\\n`` back on every
    # platform, which keeps the transcript grep-friendly across OSes.
    return open(path, "a", encoding="utf-8", newline="")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _transcript_writer(fh):
    """Return a ``(line) -> None`` closure over an open transcript handle.

    Centralises the per-line write + flush so :func:`_emit_line` doesn't
    have to care about file handles. ``None`` is returned when no
    transcript was requested, so the caller can pass it through
    without an extra branch. The closure swallows the ``OSError`` via
    :func:`_emit_line` (it is reported on stderr but never fatal) and
    the caller still owns ``fh`` for closing.
    """
    if fh is None:
        return None

    def _write(line: str) -> None:
        fh.write(line)
        fh.write("\n")
        fh.flush()

    return _write


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2

    # Open the transcript once for the whole run so a long ``--follow``
    # stream appends in real time and so we always close it cleanly,
    # even on a TTS error or a KeyboardInterrupt mid-loop. We open
    # AFTER parse_args so a bad --transcript path is a clean exit-2
    # before we touch any process.
    try:
        transcript_fh = open_transcript(args.transcript)
    except OSError as exc:
        print(
            f"paw-watch: could not open --transcript {args.transcript!r}: "
            f"{exc}",
            file=sys.stderr,
        )
        return 2
    try:
        return _run(args, transcript_fh)
    finally:
        if transcript_fh is not None:
            transcript_fh.close()


def _run(args: argparse.Namespace, transcript_fh) -> int:
    """Dispatch batch vs. ``--follow`` and own the announcement banner.

    Splits the entry point so the transcript handle is opened /
    closed by :func:`main` (with a single ``try/finally``) while the
    per-mode loops stay in :func:`_run_batch` and
    :func:`_run_streaming`. The ``transcript_write`` closure is
    built here so both modes see the same "what to do with each
    line" contract.
    """
    if not args.quiet:
        mode = "follow" if args.follow else "tail"
        preview = " ".join(args.cmd)
        if len(preview) > 60:
            preview = preview[:57] + "..."
        action = "previewing" if args.dry_run else f"{mode}ing"
        print(f"🐾 paw-watch: {action} `{preview}`")

    writer = _transcript_writer(transcript_fh)
    if args.follow:
        return _run_streaming(args, writer)
    return _run_batch(args, writer)


def _run_batch(args: argparse.Namespace, writer) -> int:
    """Batch path: spawn, collect stdout, speak each line in order."""
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
        code = _emit_line(
            line,
            rate=args.rate,
            volume=args.volume,
            dry_run=args.dry_run,
            transcript_write=writer,
        )
        spoken += 1
        if code != 0:
            tts_error = code
    # Anything left in the buffer (no trailing newline) is one final line.
    for line in buffer.flush():
        if max_lines and spoken >= max_lines:
            break
        code = _emit_line(
            line,
            rate=args.rate,
            volume=args.volume,
            dry_run=args.dry_run,
            transcript_write=writer,
        )
        spoken += 1
        if code != 0:
            tts_error = code

    # Mirror the child's exit code if it failed. Dry-run never reports
    # a TTS error (the helper always returns 0 in that mode), so a
    # failed child is the only failure signal -- which is what `set -e`
    # scripts want.
    if completed.returncode != 0:
        return completed.returncode
    if tts_error:
        return 1
    return 0


def _run_streaming(args: argparse.Namespace, writer) -> int:
    """``--follow`` mode: speak each new stdout line as the child produces it.

    We open the child with :func:`_popen` and iterate ``stdout_iter()``
    line-by-line, feeding each raw line to the same :class:`_LineBuffer`
    the batch path uses. The buffer still does the partial-line
    accumulation, so a write of ``"hello\\nwor"`` followed by ``"ld\\n"``
    yields exactly one spoken line (``"hello world"``).

    After the stream ends (EOF) we call ``wait()`` to collect the exit
    code; if the child exited while we were still draining we already
    have it, otherwise ``wait()`` blocks until the process actually
    finishes. This matches the semantics users expect from ``tail -f``:
    TTS stops as soon as the pipe closes, the process is reaped, and
    its exit code is mirrored back to the shell.
    """
    try:
        proc = _popen(args.cmd, include_stderr=args.include_stderr)
    except FileNotFoundError as exc:
        print(f"paw-watch: command not found: {exc.filename or args.cmd[0]}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"paw-watch: could not spawn {args.cmd[0]!r}: {exc}", file=sys.stderr)
        return 2

    buffer = _LineBuffer()
    spoken = 0
    tts_error = 0
    max_lines = args.max_lines  # 0 == unlimited
    try:
        for raw_line in proc.stdout_iter():
            for line in buffer.feed(raw_line):
                if max_lines and spoken >= max_lines:
                    break
                code = _emit_line(
                    line,
                    rate=args.rate,
                    volume=args.volume,
                    dry_run=args.dry_run,
                    transcript_write=writer,
                )
                spoken += 1
                if code != 0:
                    tts_error = code
            # Check inside the outer loop too so we stop reading the
            # child as soon as we've hit the cap (don't keep draining
            # a long-running process we no longer care about).
            if max_lines and spoken >= max_lines:
                break
        # Trailing fragment without a newline still counts.
        for line in buffer.flush():
            if max_lines and spoken >= max_lines:
                break
            code = _emit_line(
                line,
                rate=args.rate,
                volume=args.volume,
                dry_run=args.dry_run,
                transcript_write=writer,
            )
            spoken += 1
            if code != 0:
                tts_error = code
    finally:
        returncode = proc.wait()

    if returncode != 0:
        return returncode
    if tts_error:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
