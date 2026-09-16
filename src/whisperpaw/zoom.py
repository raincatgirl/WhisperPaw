"""paw-zoom — magnify a region of text (ASCII proof-of-concept).

What this is
------------
This is the **first, ASCII-only milestone** of the magnifier. It accepts
a text source (a positional argument, a ``--file`` path, or stdin),
extracts a rectangular sub-grid from it, and renders that sub-grid
"magnified" — each source cell is repeated ``--zoom`` times in both
directions, so a window of 5×2 cells at zoom 3 becomes a 15×6 block.

Why a text-viewport first?
--------------------------
The real ``paw-zoom`` will eventually capture a portion of the
**visual** screen (Wayland / X11 / Win32 GDI) and magnify pixels.
That needs a cross-platform screen-capture adapter, a per-OS render
loop, and hotkey plumbing — easily a couple of ticks. Before any
of that lands, this milestone nails down the parts that don't
depend on the capture layer:

* the data model (``ZoomConfig``)
* the viewport math (extract a rectangular sub-grid from text)
* the magnification primitive (repeat each cell ``N``×``N``)
* the source resolver (positional → ``--file`` → stdin)
* a clean argparse layer (``paw-complete`` already reads it)

That means the second tick can plug a real screen-capture adapter
in front of this same pipeline without touching the math or the CLI.

Pure stdlib, no third-party deps, no telemetry, no network.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default visible window — 10 source rows × 40 source columns.
DEFAULT_ROWS: int = 10
DEFAULT_COLS: int = 40

#: Default zoom factor — each source cell becomes a 2x2 block.
DEFAULT_ZOOM: int = 2

#: Lowest / highest acceptable zoom factor. Upper bound is just a sanity
#: guard so a stray ``--zoom 9999`` doesn't blow up the user's terminal.
MIN_ZOOM: int = 1
MAX_ZOOM: int = 32

#: The "space" charset (the default): every empty cell stays empty. The
#: ``#`` and ``.`` charsets paint every cell with a fixed fill character,
#: which is sometimes easier to read on a busy terminal.
VALID_CHARSETS: tuple[str, ...] = ("space", "hash", "dot")


@dataclass(frozen=True)
class ZoomConfig:
    """The "what to show" parameters, in one immutable bundle.

    A real screen magnifier would add ``device``, ``cursor_pos`` and
    a render mode here. Keeping them out of the POC means this
    dataclass is purely about the *viewport* of text.
    """

    rows: int = DEFAULT_ROWS
    cols: int = DEFAULT_COLS
    zoom: int = DEFAULT_ZOOM
    fill: str = " "
    row_offset: int = 0
    col_offset: int = 0


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------


def _read_stdin() -> str:
    """Read all of stdin. Returns ``""`` when stdin is a TTY.

    The ``WPAW_ZOOM_STDIN_OVERRIDE`` env var mirrors the pattern set by
    ``paw-read`` and ``paw-watch`` — tests can set it to bypass pytest's
    stdin capture without having to monkey-patch ``sys.stdin``.
    """
    override = os.environ.get("WPAW_ZOOM_STDIN_OVERRIDE")
    if override is not None:
        return override
    if sys.stdin.isatty():
        return ""
    return sys.stdin.read()


def _resolve_source(
    *, text: str | None, file: str | None, stdin_text: str | None
) -> str:
    """Pick the first non-empty source, in priority order.

    Priority: positional ``text`` → ``--file PATH`` → ``stdin``.
    Empty / whitespace-only results from any source are treated as
    "not provided" so the user can run ``paw-zoom ""`` without
    tripping the "no text" error (we still want to *show* an empty
    window in that case — that's why ``render_viewport`` accepts ``""``).
    """
    if text is not None and text.strip():
        return text
    if file is not None:
        from pathlib import Path

        path = Path(file)
        if not path.is_file():
            raise FileNotFoundError(f"paw-zoom: file not found: {file}")
        # Match paw-read's behaviour: strip the trailing newline so the
        # user gets a faithful copy of the file's text content (the
        # trailing \n is a line terminator, not part of the data).
        return path.read_text(encoding="utf-8").rstrip("\n")
    if stdin_text is not None and stdin_text.strip():
        return stdin_text
    raise RuntimeError(
        "paw-zoom: no text to magnify — pass a positional argument, "
        "--file PATH, or pipe via stdin"
    )


# ---------------------------------------------------------------------------
# Region extraction
# ---------------------------------------------------------------------------


def _extract_region(
    source: str,
    *,
    rows: int,
    cols: int,
    row_offset: int,
    col_offset: int,
) -> list[str]:
    """Return a ``rows``×``cols`` sub-grid of ``source`` as a list of strings.

    The source is split on newlines; the sub-grid starts at
    ``(row_offset, col_offset)`` in code-point coordinates and runs
    for ``rows`` lines × ``cols`` code points per line. Offsets past
    the end of the source are clamped (returning padded / empty
    lines) rather than raising, so an out-of-range cursor still
    renders something sensible.

    The function is deliberately *line-based* — multi-line code-point
    cells (grapheme clusters) are out of scope for the POC. The
    next tick (real screen capture) can revisit this if needed.
    """
    # Split, preserving logical lines; an empty source becomes one
    # empty line so the caller still gets ``rows`` rows of output.
    lines = source.split("\n")
    # Drop a single trailing empty line that comes from a source that
    # ends in \n — that empty line isn't a real row of content.
    if lines and lines[-1] == "" and source.endswith("\n"):
        lines = lines[:-1]
    if not lines:
        lines = [""]

    clamped_row = max(0, row_offset)
    clamped_col = max(0, col_offset)
    region: list[str] = []
    for r in range(rows):
        line_idx = clamped_row + r
        if line_idx >= len(lines):
            region.append(" " * cols)
            continue
        line = lines[line_idx]
        # Slice the line at the column offset; pad on the right if
        # the requested window is wider than what's there.
        if clamped_col >= len(line):
            region.append(" " * cols)
            continue
        slice_ = line[clamped_col : clamped_col + cols]
        if len(slice_) < cols:
            slice_ = slice_ + " " * (cols - len(slice_))
        region.append(slice_)
    return region


# ---------------------------------------------------------------------------
# Magnification
# ---------------------------------------------------------------------------


def _magnify(region: list[str], *, zoom: int, fill: str) -> str:
    """Return ``region`` magnified by ``zoom`` in both directions.

    Each cell becomes a ``zoom``×``zoom`` block. Empty input is a
    no-op (we don't pad with ``fill`` — that's the caller's job, see
    :func:`render_viewport`).
    """
    if zoom < MIN_ZOOM or zoom > MAX_ZOOM:
        raise ValueError(
            f"zoom must be between {MIN_ZOOM} and {MAX_ZOOM} (got {zoom})"
        )
    if not region:
        return ""
    out_lines: list[str] = []
    for line in region:
        # Build the magnified row, then repeat it ``zoom`` times.
        wide = "".join(ch * zoom for ch in line)
        for _ in range(zoom):
            out_lines.append(wide)
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Public render entry point
# ---------------------------------------------------------------------------


_CHARSET_FILLS: dict[str, str] = {
    "space": " ",
    "hash": "#",
    "dot": ".",
}


def render_viewport(source: str, cfg: ZoomConfig) -> str:
    """Render the magnified viewport for ``source`` under ``cfg``.

    Empty / short sources are padded with ``cfg.fill`` so the output
    is always exactly ``rows * zoom`` lines of ``cols * zoom`` chars.
    """
    region = _extract_region(
        source,
        rows=cfg.rows,
        cols=cfg.cols,
        row_offset=cfg.row_offset,
        col_offset=cfg.col_offset,
    )
    # If the source was empty, paint the whole region with ``fill`` so
    # the user gets a visible (if blank) window rather than 10 lines
    # of "                          ".
    if not source.strip():
        region = [cfg.fill * cfg.cols for _ in range(cfg.rows)]
    return _magnify(region, zoom=cfg.zoom, fill=cfg.fill)


# ---------------------------------------------------------------------------
# Live / "follow" mode
# ---------------------------------------------------------------------------


#: Default poll interval for ``--live`` (seconds). Small enough that
#: the user sees new lines almost immediately, large enough that we
#: aren't a CPU hog. 250 ms is the same order of magnitude as
#: ``paw-watch``'s 0.1s ``--follow`` cadence.
DEFAULT_LIVE_INTERVAL: float = 0.25


def _read_file_text(path: str) -> str:
    """Read ``path`` as UTF-8 text. Returns ``""`` if the file is empty.

    A small wrapper so the live loop has one obvious call site for
    "re-read the source now". Errors (missing file, permission denied)
    propagate — the caller decides how to surface them.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _tail_and_render(
    path: str,
    cfg: ZoomConfig,
    *,
    interval: float,
    stop_predicate=None,
    clock=None,
    sink=None,
) -> int:
    """Follow ``path`` and re-render the magnified viewport on each change.

    Loops until ``stop_predicate()`` returns truthy (or forever, if not
    given). On every iteration:

    1. Re-read the file from disk.
    2. If the contents changed since the previous iteration, re-render
       the viewport and write it to ``sink`` — a callable taking a
       single ``str`` argument. Defaults to writing the rendered
       viewport + a blank line to ``sys.stdout``. Tests can pass a
       list-collector to inspect each frame.
    3. Sleep ``interval`` seconds using ``clock()`` (defaults to
       :func:`time.sleep`; tests can pass a fake clock that returns
       immediately).

    Returns ``0`` on a clean exit. Errors are surfaced as a single
    stderr line and the loop continues — a transient ENOENT during
    log rotation shouldn't kill the magnifier.

    Note: this is the **text-source** tail (like ``tail -f`` for a log
    file). It is not the real screen-capture tail that v0.2 will
    need. But the loop shape is identical, so this can be reused.
    """
    sleep = clock if clock is not None else time.sleep
    last_text: str | None = None
    last_mtime: float | None = None
    while stop_predicate is None or not stop_predicate():
        try:
            text = _read_file_text(path)
        except FileNotFoundError:
            print(
                f"paw-zoom: --live source not found: {path!r}",
                file=sys.stderr,
            )
            sleep(interval)
            continue
        except OSError as exc:
            print(
                f"paw-zoom: could not read --live source {path!r}: {exc}",
                file=sys.stderr,
            )
            sleep(interval)
            continue
        # Cheap "did it change?" check: mtime. If the mtime moved,
        # something may have been appended (or rotated; we re-read
        # unconditionally so rotation is safe). Avoids re-rendering
        # the same N kB of text on every poll.
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = None
        if text == last_text and mtime == last_mtime:
            sleep(interval)
            continue
        last_text = text
        last_mtime = mtime
        rendered = render_viewport(text, cfg)
        if sink is not None:
            sink(rendered)
        else:
            sys.stdout.write(rendered)
            # Frame separator. A blank line keeps successive frames
            # visually distinct in scrollback.
            sys.stdout.write("\n")
            sys.stdout.flush()
        sleep(interval)
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paw-zoom",
        description=(
            "Magnify a rectangular region of text. The text source is a "
            "positional argument, --file PATH, or stdin (in that priority). "
            "The window is --rows source-rows tall and --cols source-cols wide; "
            "each cell is repeated --zoom times in both directions."
        ),
    )
    parser.add_argument(
        "text",
        nargs=argparse.REMAINDER,
        help=(
            "Text to magnify. If omitted, read from --file or stdin "
            "(in that priority order)."
        ),
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Read text from this UTF-8 file instead of positional args.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_ROWS,
        help=(
            f"How many source rows to show (default: {DEFAULT_ROWS}, must be >= 1)."
        ),
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=DEFAULT_COLS,
        help=(
            f"How many source columns to show (default: {DEFAULT_COLS}, must be >= 1)."
        ),
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        dest="offset",
        help=(
            "Starting row in the source (default: 0). Negative values are clamped to 0."
        ),
    )
    parser.add_argument(
        "--col-offset",
        type=int,
        default=0,
        dest="col_offset",
        help=(
            "Starting column in the source (default: 0). Negative values are clamped to 0."
        ),
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=DEFAULT_ZOOM,
        help=(
            f"Magnification factor: each source cell becomes a "
            f"--zoom×--zoom block (default: {DEFAULT_ZOOM}, range: {MIN_ZOOM}–{MAX_ZOOM})."
        ),
    )
    parser.add_argument(
        "--charset",
        default="space",
        choices=VALID_CHARSETS,
        help=(
            "Fill character for empty cells: 'space' (default, invisible), "
            "'hash' (#), or 'dot' (.)."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the announcement line (still renders).",
    )
    parser.add_argument(
        "--snapshot",
        default=None,
        metavar="PATH",
        help=(
            "Write the rendered viewport to PATH instead of stdout. "
            "The file is created or overwritten. Useful for piping "
            "rendering into scripts, logs, or later processing. The "
            "announcement line (unless --quiet) still goes to stderr."
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Follow --file PATH like 'tail -f': re-render the magnified "
            "viewport every time the source changes. Implies --file. "
            "Press Ctrl-C to stop. The poll interval is --interval "
            f"(default: {DEFAULT_LIVE_INTERVAL:g}s)."
        ),
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_LIVE_INTERVAL,
        metavar="SECS",
        help=(
            "Poll interval in seconds for --live (default: "
            f"{DEFAULT_LIVE_INTERVAL:g}, must be > 0)."
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    # REMAINDER returns a list; join into a single string but keep it
    # None if empty so the resolver can pick --file / stdin.
    if args.text:
        args.text = " ".join(args.text)
    else:
        args.text = None
    # Validate after parsing so the user gets a friendly message
    # and a stable exit code, not a Python traceback.
    if args.rows < 1:
        print("paw-zoom: --rows must be >= 1", file=sys.stderr)
        raise SystemExit(2)
    if args.cols < 1:
        print("paw-zoom: --cols must be >= 1", file=sys.stderr)
        raise SystemExit(2)
    if not (MIN_ZOOM <= args.zoom <= MAX_ZOOM):
        print(
            f"paw-zoom: --zoom must be between {MIN_ZOOM} and {MAX_ZOOM}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if args.interval <= 0:
        print(
            "paw-zoom: --interval must be > 0 seconds",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if args.live and not args.file:
        print(
            "paw-zoom: --live requires --file PATH "
            "(there is no stdin to tail)",
            file=sys.stderr,
        )
        raise SystemExit(2)
    # The --charset choice is enforced by argparse, but the resolved
    # fill char is what the renderer needs. Stash it on the namespace.
    args.fill = _CHARSET_FILLS[args.charset]
    return args


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        source = _resolve_source(
            text=args.text,
            file=args.file,
            stdin_text=_read_stdin(),
        )
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    cfg = ZoomConfig(
        rows=args.rows,
        cols=args.cols,
        zoom=args.zoom,
        fill=args.fill,
        row_offset=args.offset,
        col_offset=args.col_offset,
    )
    if not args.quiet:
        # A short, friendly banner. We deliberately do not show the
        # full text — it could be huge. The output itself is the
        # main thing.
        print(
            f"🐾 paw-zoom: {cfg.rows}×{cfg.cols} window, "
            f"zoom {cfg.zoom}, output {cfg.rows * cfg.zoom}×{cfg.cols * cfg.zoom}"
        )

    if args.live:
        # Live mode: re-render on every change until Ctrl-C.
        # If --snapshot is also set, each frame is written to the
        # file (overwriting the previous one) so an external
        # viewer can `cat` it to see the latest viewport. Without
        # --snapshot, each frame is written to stdout separated
        # by a blank line.
        snapshot_path = args.snapshot

        def _sink(text: str) -> None:
            if snapshot_path is not None:
                try:
                    with open(snapshot_path, "w", encoding="utf-8") as fh:
                        fh.write(text)
                except OSError as exc:
                    print(
                        f"paw-zoom: could not write --snapshot file "
                        f"{snapshot_path!r}: {exc}",
                        file=sys.stderr,
                    )
            else:
                sys.stdout.write(text)
                sys.stdout.write("\n")
                sys.stdout.flush()

        try:
            return _tail_and_render(
                args.file, cfg, interval=args.interval, sink=_sink
            )
        except KeyboardInterrupt:
            # Ctrl-C is a clean exit in --live mode. Don't print
            # a traceback.
            return 0

    try:
        rendered = render_viewport(source, cfg)
    except ValueError as exc:
        # Defensive: ZoomConfig and parse_args already guard this, but
        # the public render_viewport() is callable from anywhere, so
        # we re-validate at the boundary.
        print(f"paw-zoom: {exc}", file=sys.stderr)
        return 2
    if args.snapshot is not None:
        try:
            # Create or overwrite. Text mode preserves the codepoint
            # layout of the rendered viewport, which is what callers
            # expect when they later diff or print the file.
            with open(args.snapshot, "w", encoding="utf-8") as fh:
                fh.write(rendered)
        except OSError as exc:
            print(
                f"paw-zoom: could not write --snapshot file "
                f"{args.snapshot!r}: {exc}",
                file=sys.stderr,
            )
            return 1
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
