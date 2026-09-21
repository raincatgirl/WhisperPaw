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
import dataclasses
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass

from whisperpaw import _screen

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


def _format_line_numbers(
    rendered: str,
    *,
    zoom: int,
    first_source_row: int,
    width: int = 4,
    gutter: str = " │ ",
) -> str:
    """Prepend a per-source-row number to each magnified line in ``rendered``.

    The output of :func:`render_viewport` has exactly ``rows * zoom`` lines,
    where each source row is represented by exactly ``zoom`` consecutive
    magnified lines (the same source row N is repeated ``zoom`` times).
    ``_format_line_numbers`` splits ``rendered`` into those ``zoom``-sized
    groups and prepends ``f"{row+first_source_row:>{width}}{gutter}"`` to
    every line in the group, so the magnified output now reads:

          1 │ xxxxxxxxxx
          1 │ xxxxxxxxxx
          2 │ yyyyyyyyyy
          2 │ yyyyyyyyyy

    instead of:

        xxxxxxxxxx
        xxxxxxxxxx
        yyyyyyyyyy
        yyyyyyyyyy

    The default ``width=4`` right-aligns the row number in a 4-char
    column (so a 1-source-row viewport and a 9999-source-row viewport
    line up cleanly under each other). ``gutter=" │ "`` is the
    standard ``cat -n`` / ``nl`` / editor gutter (a vertical bar
    flanked by spaces, easy to read on a busy terminal). Both are
    overridable for tests and for the rare user who wants a tighter
    or looser gutter. The numbers are 1-based (matching what
    `cat -n` / `nl` / editors show),
    and ``first_source_row`` is what the user would have seen in their
    editor at the start of the magnified viewport. For a vanilla
    ``paw-zoom hello`` (no offset), that's ``1``; for
    ``paw-zoom --offset 12 hello`` it's ``13``.

    The split is line-by-line so the function is pure and trivially
    testable: an empty ``rendered`` returns an empty string (the
    ``--raw`` and the no-frame case both rely on this). A
    ``zoom <= 0`` or a non-multiple-of-zoom line count is treated
    defensively: the function still emits a prefix per line, just
    with whatever numbering ``itertools.count`` produces — the
    caller is responsible for passing a valid ``zoom`` (we already
    validate that at parse time).
    """
    if not rendered:
        return rendered
    if zoom < 1:
        # Defensive: a non-positive zoom can't be grouped, so we
        # fall back to numbering every line sequentially. This
        # should never happen in practice (parse_args rejects
        # ``--zoom 0``), but the library function is callable
        # from anywhere, so we re-validate at the boundary.
        return "\n".join(
            f"{row:>{width}}{gutter}{line}"
            for row, line in enumerate(rendered.split("\n"), start=first_source_row)
        )
    lines = rendered.split("\n")
    out: list[str] = []
    # Group the magnified output into ``zoom``-sized chunks, one
    # chunk per source row. ``itertools.count`` gives us a clean
    # 1-based-per-source-row numbering (start=first_source_row)
    # so the user sees the same numbers `cat -n` would show.
    from itertools import count
    counter = count(first_source_row)
    for start in range(0, len(lines), zoom):
        row = next(counter)
        prefix = f"{row:>{width}}{gutter}"
        out.extend(prefix + line for line in lines[start : start + zoom])
    return "\n".join(out)


def _format_col_ruler(
    rendered: str,
    *,
    zoom: int,
    cols: int,
    first_source_col: int = 1,
) -> str:
    """Prepend a column-number ruler line above the magnified output.

    The ruler is a single line that mirrors the magnified viewport's
    horizontal extent (exactly ``cols * zoom`` code points) and shows
    the 1-based column index of each source column. For each source
    column ``N`` (1-based, starting at ``first_source_col``), the
    last digit of ``first_source_col + N - 1`` is repeated ``zoom``
    times, then all columns are concatenated. The result is a
    familiar editor-style ruler:

        paw-zoom --col-ruler --cols 10 --zoom 2 '0123456789'

    emits

        1122334455667788990
        00112233445566778899
        00112233445566778899
        ...
        00112233445566778899

    (the 10 magnified source columns are numbered 1..10 — the
    ``first_source_col + N - 1`` formula shifts the numbering so
    ``paw-zoom --col-ruler --col-offset 42 ...`` shows columns
    43, 44, 45, ...).

    The ruler is intentionally numeric-only: the last digit of
    the column index is what the user can read at a glance, the
    same convention ``vim`` / ``nano`` / ``less`` use. For
    multi-digit column numbers the last digit is the only one
    that fits in a single magnified cell, so this is the most
    honest representation that doesn't require widening the
    viewport.

    Composes with :func:`_format_line_numbers` — the gutter sits
    on the left of every line, the ruler sits above every line.
    An empty ``rendered`` is a no-op (the ``--raw`` and the
    no-frame case both rely on this), matching
    :func:`_format_line_numbers`'s convention. A non-positive
    ``zoom`` is treated defensively as ``1`` (the caller's
    ``parse_args``-level check has already rejected ``--zoom 0``,
    but the library function is callable from anywhere, so we
    re-validate at the boundary).
    """
    if not rendered:
        return rendered
    if zoom < 1:
        zoom = 1
    width = cols * zoom
    # ``str(index)[-1]`` extracts the last digit of the 1-based
    # column index. For column 10 this yields "0", for column 11
    # "1", etc. — the standard editor-ruler convention.
    ruler_chars: list[str] = []
    for n in range(cols):
        col_index = first_source_col + n
        digit = str(col_index)[-1]
        ruler_chars.append(digit * zoom)
    ruler = "".join(ruler_chars)
    # Defensive: if the ruler is somehow shorter than the
    # magnified width (should never happen with valid ``zoom`` /
    # ``cols``), pad with spaces so the alignment with the
    # viewport below is preserved. If it's longer, truncate
    # (same convention ``less -N`` uses — show what fits,
    # drop the rest).
    if len(ruler) < width:
        ruler = ruler + " " * (width - len(ruler))
    elif len(ruler) > width:
        ruler = ruler[:width]
    return ruler + "\n" + rendered


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
# -------------------------------------------------------------------------


#: Default poll interval for ``--live`` (seconds). Small enough that
#: the user sees new lines almost immediately, large enough that we
#: aren't a CPU hog. 250 ms is the same order of magnitude as
#: ``paw-watch``'s 0.1s ``--follow`` cadence.
DEFAULT_LIVE_INTERVAL: float = 0.25


def _max_seconds_exceeded(
    start: float, max_seconds: float, now: float
) -> bool:
    """Return True if ``max_seconds`` of wall-clock time have passed
    since ``start``.

    A pure predicate that the live loops call once per iteration
    to decide whether the ``--max-seconds`` cap has fired. The
    ``start`` and ``now`` values come from a ``time_fn`` injected
    into the loop (default :func:`time.monotonic`); the helper
    itself is pure so the math is testable without monkey-patching
    the module-level default.

    Conventions:

    * ``max_seconds <= 0`` means "no cap" → always returns ``False``.
      The CLI surfaces the same default (``--max-seconds 0``) and
      the parse-time validator rejects negative values, so the
      loop never sees a negative cap.
    * The check is ``>=``, not ``>``, so a cap of exactly
      ``max_seconds`` is treated as already exceeded (one extra
      iteration would always push the loop past the wall-clock
      window the user asked for, which is the more useful
      behaviour for scripting: "run for 5s" means *at most* 5s).

    The arithmetic is plain subtraction — no tolerance, no drift
    correction, no special handling of clock jumps. ``time.monotonic``
    is the recommended clock and is unaffected by NTP slews, so
    the value of ``now - start`` is the wall-clock time the loop
    actually spent.
    """
    if max_seconds <= 0:
        return False
    return (now - start) >= max_seconds


def _read_file_text(path: str) -> str:
    """Read ``path`` as UTF-8 text. Returns ``""`` if the file is empty.

    A small wrapper so the live loop has one obvious call site for
    "re-read the source now". Errors (missing file, permission denied)
    propagate — the caller decides how to surface them.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _tail_offset(source: str, rows: int) -> int:
    """Return the row offset that puts the last ``rows`` lines in view.

    Mirrors what ``tail -n N`` shows: for a source with more than
    ``rows`` logical lines, the offset is ``total - rows``; for a
    shorter source, the offset is ``0`` (we show the whole thing —
    the renderer pads with ``fill``).

    Trailing empty lines are dropped before counting, so a file
    ending in ``\\n`` doesn't push the last real line off the
    viewport. Empty / whitespace-only source returns ``0`` so the
    caller still emits a visible (blank) window.
    """
    if not source.strip():
        return 0
    lines = source.split("\n")
    if lines and lines[-1] == "" and source.endswith("\n"):
        lines = lines[:-1]
    total = len(lines)
    if total <= rows:
        return 0
    return total - rows


#: Tuple shape returned by :func:`_source_size` — ``(rows, max_cols)``
#: in code-point units. A standalone named alias so callers don't
#: have to remember the positional order; mirrors the
#: ``list[list[str]]`` style the screen-capture adapters return.
SourceSize = tuple[int, int]


def _source_size(source: str) -> SourceSize:
    """Return ``(rows, max_cols)`` of ``source`` in code-point units.

    A source-side analog of the screen-capture ``describe_capture``
    helper. The conventions match the rest of ``paw-zoom``:

    * Lines are split on ``\\n`` (same as :func:`_extract_region`).
    * A single trailing empty line is dropped when the source ends
      in ``\\n`` (same convention as :func:`_tail_offset`), so a
      file ending in a newline reports the *content* line count
      rather than the trailing terminator as a phantom row.
    * An empty / whitespace-only source reports ``(1, 0)`` — one
      row of zero columns, the natural shape for "there's a
      window but nothing in it". This matches what the renderer
      would actually emit: :func:`render_viewport` paints one
      row of fill chars for an empty source.
    * ``max_cols`` is measured in code points (``len(line)``), not
      bytes, so a multi-byte CJK line is counted correctly. The
      rendered viewport is also code-point based, so the two
      numbers mean the same thing.
    """
    if not source:
        return (1, 0)
    lines = source.split("\n")
    if lines and lines[-1] == "" and source.endswith("\n"):
        lines = lines[:-1]
    if not lines:
        return (1, 0)
    return (len(lines), max(len(line) for line in lines))


def _size_to_text(size: SourceSize) -> str:
    """Render a :data:`SourceSize` as a single ``"rows x cols"`` line.

    Parallel to :func:`whisperpaw._screen.describe_to_text` —
    fixed format, predictable layout, easy to grep. ``cols`` is
    always emitted (even when zero) so downstream tooling can
    always split on ``" x "`` to get two integers.
    """
    rows, cols = size
    return f"{rows} x {cols}"


def _size_to_json(size: SourceSize) -> str:
    """Render a :data:`SourceSize` as a single-line parseable JSON object.

    Parallel to :func:`whisperpaw._screen.describe_to_json` — the
    shape ``{"rows": N, "cols": M}`` is fixed so ``jq '.rows'`` and
    friends work without further coercion. Single-line, sorted
    keys, ``ensure_ascii=False``.
    """
    rows, cols = size
    # ``json.dumps`` with ``sort_keys=True`` gives the deterministic
    # key order the rest of the discovery helpers already use.
    return json.dumps({"rows": rows, "cols": cols}, sort_keys=True, ensure_ascii=False)


#: Per-source statistics reported by ``paw-zoom --stats``. A
#: standalone named alias so callers don't have to remember the
#: positional order; mirrors the :data:`SourceSize` style.
#:
#: The five fields, in canonical order:
#:
#: * ``chars`` — total source code points (including newlines).
#:   Code points, not bytes, so a multi-byte CJK source is
#:   counted the same way the renderer would count it.
#: * ``lines`` — logical line count, after dropping a single
#:   trailing empty line (the ``\n``-terminator convention used
#:   by :func:`_source_size` and :func:`_tail_offset`). An empty
#:   source reports ``0`` lines (the "no content" answer; --size
#:   reports ``(1, 0)`` for the same source because it answers
#:   "what would the renderer show?", not "how much content is
#:   there?").
#: * ``non_blank_lines`` — lines whose stripped form is non-empty.
#:   Empty / whitespace-only sources report ``0``.
#: * ``max_line_width`` — longest line width in code points
#:   (same convention as :func:`_source_size`'s ``max_cols``).
#: * ``mean_line_width`` — mean line width in code points,
#:   rounded to 2 decimal places. ``0.0`` for an empty source
#:   (no lines → no mean to compute; matches the other
#:   ``0``-as-natural-zero fields).
SourceStats = tuple[int, int, int, int, float]


def _source_stats(source: str) -> SourceStats:
    """Return per-source statistics for ``source``.

    The text-side analog of :func:`whisperpaw._screen.describe_capture`'s
    *introspection* shape — ``--info`` answers "what would the
    screen-capture pipeline do?"; ``--stats`` answers "what's in the
    source?" Useful for the same reasons ``wc`` is: counting without
    rendering, so a script can branch on "is this a log file or a
    config file?" before piping it through the magnifier.

    Conventions match the rest of the source-introspection helpers:

    * Lines are split on ``\\n`` (same as :func:`_source_size`).
    * A single trailing empty line is dropped when the source ends
      in ``\\n`` (same convention as :func:`_source_size` and
      :func:`_tail_offset`), so a file ending in a newline reports
      the *content* line count rather than the trailing
      terminator as a phantom row.
    * An empty source reports ``(0, 0, 0, 0, 0.0)`` — zero
      everything, the natural "nothing to count" shape. The
      divergent ``(1, 0)`` answer from :func:`_source_size` on
      the same input is the "what would the renderer show?"
      answer; the stats answer is the "how much content is
      there?" answer, and they differ on purpose.
    * ``chars`` is measured in code points (``len(source)``), not
      bytes, so a multi-byte CJK source counts the same way the
      renderer counts it.
    * ``max_line_width`` is measured in code points, same
      convention as :func:`_source_size`'s ``max_cols``.
    * ``mean_line_width`` is rounded to 2 decimal places via
      :func:`round` (banker's rounding — ``0.5`` rounds to the
      nearest even integer, but the test suite uses values that
      don't sit on the half so the behaviour is deterministic).
      The output is always a real ``float`` so JSON serialisation
      stays predictable (``"mean_line_width": 1.5``, never
      ``"mean_line_width": "1.5"``).
    """
    if not source:
        return (0, 0, 0, 0, 0.0)
    lines = source.split("\n")
    if lines and lines[-1] == "" and source.endswith("\n"):
        lines = lines[:-1]
    if not lines:
        return (0, 0, 0, 0, 0.0)
    widths = [len(line) for line in lines]
    non_blank = sum(1 for line in lines if line.strip())
    mean_width = round(sum(widths) / len(widths), 2)
    return (
        len(source),
        len(lines),
        non_blank,
        max(widths),
        mean_width,
    )


def _stats_to_text(stats: SourceStats) -> str:
    """Render a :data:`SourceStats` as a 5-line fixed ``key: value`` block.

    Parallel to :func:`whisperpaw._screen.describe_to_text` —
    fixed format, predictable layout, easy to grep / awk. Each
    field is on its own line so a downstream consumer can pick
    any one of them with a one-line selector. ``mean_line_width``
    is rendered with a stable 2-decimal format (``1.50``, not
    ``1.5``) so the layout is predictable for a downstream
    ``awk`` / ``cut`` / ``column`` pipeline; ``int`` fields use
    the obvious ``%d`` so a ``stats.txt`` file sorted by line
    number is human-readable.
    """
    chars, lines, non_blank, max_w, mean_w = stats
    return (
        f"chars: {chars}\n"
        f"lines: {lines}\n"
        f"non_blank_lines: {non_blank}\n"
        f"max_line_width: {max_w}\n"
        f"mean_line_width: {mean_w:.2f}"
    )


def _stats_to_json(stats: SourceStats) -> str:
    """Render a :data:`SourceStats` as a single-line parseable JSON object.

    Parallel to :func:`whisperpaw._screen.describe_to_json` and
    :func:`_size_to_json` — the shape is fixed so ``jq '.chars'``
    and friends work without further coercion. Single-line,
    sorted keys, ``ensure_ascii=False``. ``mean_line_width`` is
    emitted as a JSON number (not a string) so the field can be
    compared numerically.
    """
    chars, lines, non_blank, max_w, mean_w = stats
    return json.dumps(
        {
            "chars": chars,
            "lines": lines,
            "max_line_width": max_w,
            "mean_line_width": mean_w,
            "non_blank_lines": non_blank,
        },
        sort_keys=True,
        ensure_ascii=False,
    )


#: Algorithm used by ``paw-zoom --sha``. Centralised so the help
#: text, the JSON key, and the underlying :mod:`hashlib` call all
#: agree (changing it would be a one-line edit instead of a
#: three-line edit in three different sections of the file).
DEFAULT_SHA_ALGORITHM = "sha256"


def _source_sha(source: str, *, algorithm: str = DEFAULT_SHA_ALGORITHM) -> str:
    """Return a stable hex digest of ``source``.

    The "did the source change?" discovery primitive. Useful in
    scripts that want to skip work when the input hasn't moved:
    capture the digest once, compare against a re-computed digest
    later, branch on the result. The default algorithm is
    SHA-256 (64 hex chars, collision-resistant enough that two
    distinct log files will always produce distinct digests in
    practice). Other algorithms accepted by :mod:`hashlib`
    (``"sha1"``, ``"sha512"``, ``"md5"``) are reachable via
    ``--sha-algo`` so a user who already has a digest-comparison
    pipeline keyed on a specific algorithm can plug in.

    Conventions:

    * The source is encoded as UTF-8 before being hashed, so a
      multi-byte CJK source produces the same digest the
      renderer would see. (Python ``str`` would fail outright
      on :func:`hashlib.sha256`.update` — the encoding is not
      a stylistic choice, it's a hard requirement.)
    * The hex digest is lowercase (matches the
      :func:`hashlib` default and most other tooling).
    * No length limit on the source — :mod:`hashlib` streams
      arbitrarily long input, so a 1 GB log file hashes in
      constant memory.

    Raises :class:`ValueError` for an unsupported ``algorithm``
    name so the caller gets a clear error instead of the
    generic :class:`hashlib`'s ``ValueError: unsupported hash
    type`` traceback.
    """
    try:
        digest = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(
            f"unsupported hash algorithm {algorithm!r} "
            f"(supported: sha1, sha256, sha512, md5, ...)"
        ) from exc
    digest.update(source.encode("utf-8"))
    return digest.hexdigest()


def _sha_to_text(digest: str) -> str:
    """Render a hex digest as a single line.

    Parallel to :func:`_size_to_text` and :func:`_stats_to_text`
    — single line, no surrounding JSON object, easy to grep
    and diff. The line ends with ``\\n`` on stdout because
    :func:`print` always adds one, so a downstream ``diff``
    sees exactly the bytes ``echo "$digest"`` would produce.
    """
    return digest


def _sha_to_json(digest: str, *, algorithm: str = DEFAULT_SHA_ALGORITHM) -> str:
    """Render a hex digest as a single-line parseable JSON object.

    Parallel to :func:`_size_to_json` and :func:`_stats_to_json`
    — single line, sorted keys, ``ensure_ascii=False``. The
    ``algorithm`` key is included so a downstream tool knows
    *which* digest family to expect; without it a SHA-512
    digest could be silently mistaken for a truncated
    SHA-256 and matched against the wrong comparison value.
    """
    return json.dumps(
        {algorithm: digest, "algorithm": algorithm},
        sort_keys=True,
        ensure_ascii=False,
    )


def _tail_and_render(
    path: str,
    cfg: ZoomConfig,
    *,
    interval: float,
    follow: bool = False,
    max_frames: int = 0,
    max_seconds: float = 0.0,
    stop_predicate=None,
    clock=None,
    time_fn=None,
    sink=None,
    line_numbers: bool = False,
    col_ruler: bool = False,
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

    When ``follow`` is true, each frame's ``row_offset`` is recomputed
    so the *last* ``cfg.rows`` lines of the source are in view
    (``tail -f`` semantics) instead of the first ``cfg.rows`` lines.
    The config itself is left untouched; the per-frame offset is a
    derived value that lives only inside this function. Useful for
    log magnifiers: a 5-row viewport on a 1000-line log file
    follows the latest lines as they arrive.

    When ``max_frames`` is positive, the loop runs at most that many
    *iterations* (not emitted frames) before exiting. So a static
    source under ``max_frames=2`` emits 1 frame (the initial state)
    and then exits on the second iteration, even though no second
    frame was emitted. This makes the flag actually terminate a
    ``--live`` session — without it, an idle source would loop
    forever waiting for a change that never arrives. ``0`` (the
    default) means no cap. Composes with everything else (``--follow``,
    ``--snapshot``, ``--interval``) and is useful for scripting:
    "render the first N polls, then exit" — a bounded, deterministic
    window onto a live log.

    When ``max_seconds`` is positive, the loop also runs for at
    most that many wall-clock seconds before exiting. The check
    fires at the *top* of the loop using ``time_fn()`` (defaults
    to :func:`time.monotonic`; tests pass a fake clock that
    advances manually), so the wall-clock window is "the time the
    loop actually spent", not "the time the loop took excluding
    the final sleep". ``0`` (the default) means no cap. Composes
    with ``--max-frames`` — whichever cap fires first wins. Useful
    for "render the live log for 30s, then exit" without having
    to estimate the iteration count in advance.

    Returns ``0`` on a clean exit. Errors are surfaced as a single
    stderr line and the loop continues — a transient ENOENT during
    log rotation shouldn't kill the magnifier.

    Note: this is the **text-source** tail (like ``tail -f`` for a log
    file). It is not the real screen-capture tail that v0.2 will
    need. But the loop shape is identical, so this can be reused.
    """
    sleep = clock if clock is not None else time.sleep
    # ``time_fn`` is the wall-clock source for ``--max-seconds``.
    # Default is ``time.monotonic`` (immune to NTP slews; the right
    # clock for measuring elapsed time). Tests inject a fake clock
    # that returns a controlled sequence of timestamps so the cap
    # is exercised deterministically without sleeping.
    actual_time = time_fn if time_fn is not None else time.monotonic
    last_text: str | None = None
    last_mtime: float | None = None
    # When ``max_frames`` is set, the loop runs at most that many
    # iterations — not just that many emitted frames. A static file
    # under ``max_frames=2`` therefore emits 1 frame (the initial
    # state) and then exits on the second iteration, even though no
    # second frame was emitted. This makes the flag actually
    # terminate a ``--live`` session; without it, an idle source
    # would loop forever waiting for a change that never arrives.
    max_iters = max_frames if max_frames > 0 else None
    iter_count = 0
    # ``--max-seconds``: sample the wall clock once before the
    # loop starts. Each iteration re-samples (via ``actual_time()``)
    # at the top of the loop, so a slow source that spends most of
    # its time in the change detector or the renderer still
    # respects the cap. ``max_seconds <= 0`` means "no cap" and the
    # helper short-circuits to ``False``, so the per-iteration
    # call is essentially free.
    loop_start = actual_time() if max_seconds > 0 else 0.0
    while stop_predicate is None or not stop_predicate():
        if max_iters is not None and iter_count >= max_iters:
            break
        if _max_seconds_exceeded(loop_start, max_seconds, actual_time()):
            break
        iter_count += 1
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
        # In --follow mode the row_offset is derived from the source's
        # current size on every frame, so the viewport tracks the
        # tail of the file as it grows. We rebuild a per-frame cfg
        # instead of mutating the caller's (frozen) config.
        frame_cfg = cfg
        if follow:
            frame_cfg = dataclasses.replace(
                cfg, row_offset=_tail_offset(text, cfg.rows)
            )
        rendered = render_viewport(text, frame_cfg)
        if line_numbers:
            # --line-numbers: per-frame, the first source row in
            # view is ``frame_cfg.row_offset + 1`` (1-based, same
            # as the one-shot path). ``frame_cfg`` already
            # accounts for ``--follow``'s per-frame
            # ``row_offset`` rewrite, so a tail-tracking live
            # view shows the right numbers as the source grows.
            # Applied BEFORE --col-ruler so the gutter is added
            # to the viewport lines only, not to the ruler line
            # above them — the ruler then sits cleanly above
            # the guttered lines.
            rendered = _format_line_numbers(
                rendered,
                zoom=frame_cfg.zoom,
                first_source_row=max(1, frame_cfg.row_offset + 1),
            )
        if col_ruler:
            # --col-ruler: per-frame, the first source column in
            # view is ``frame_cfg.col_offset + 1`` (1-based, same
            # as the one-shot path). The col_offset does not
            # change between frames (only row_offset does, via
            # ``--follow``), so the ruler is stable across frames
            # of a tail-tracking live view. Applied AFTER
            # --line-numbers so the ruler sits above the
            # guttered lines.
            rendered = _format_col_ruler(
                rendered,
                zoom=frame_cfg.zoom,
                cols=frame_cfg.cols,
                first_source_col=max(1, frame_cfg.col_offset + 1),
            )
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


def _tail_screen_and_render(
    cap: _screen.ScreenCapture,
    cfg: ZoomConfig,
    *,
    region: str | None = None,
    interval: float,
    follow: bool = False,
    max_frames: int = 0,
    max_seconds: float = 0.0,
    stop_predicate=None,
    clock=None,
    time_fn=None,
    sink=None,
    capture_fn=None,
    has_changed=None,
    line_numbers: bool = False,
    col_ruler: bool = False,
) -> int:
    """Follow a screen-capture adapter and re-render the magnified
    viewport on every change.

    The screen-tail mirror of :func:`_tail_and_render`: instead of
    re-reading a file, this re-runs :func:`whisperpaw._screen.
    capture_screen_to_source` against ``cap`` on every poll. The
    captured ``str`` is then handed to the same
    :func:`render_viewport` the text-source path uses, so the
    magnification math, the padding, the ``--follow`` offset
    derivation, and the ``--snapshot`` / sink plumbing are shared
    with the text-source path with zero duplication.

    Loops until ``stop_predicate()`` returns truthy (or forever, if
    not given). On every iteration:

    1. Re-capture the screen (``capture_fn()`` or, by default,
       :func:`_screen.capture_screen_to_source` over ``region``).
    2. If the captured text differs from the previous iteration
       (``has_changed(prev, current)`` — defaults to ``!=``), call
       :func:`render_viewport` and push the result through
       ``sink`` (defaults to stdout with a blank-line separator,
       same as the text-source tail).
    3. Sleep ``interval`` seconds using ``clock()`` (defaults to
       :func:`time.sleep`; tests pass a no-op clock).

    The ``capture_fn`` and ``has_changed`` injection points exist
    so the tests can run the entire pipeline against a
    :class:`whisperpaw._screen.FakeScreen` whose grid is mutated
    between iterations, without touching the real screen. Real
    OS adapters (X11 / Win32 / Quartz) use the defaults — the
    string-diff change detector naturally fires whenever the
    captured text grid changes (e.g. a new console prompt
    appears), and a totally static screen yields exactly one
    frame, the same way a totally static file does in the
    text-source path.

    When ``follow`` is true, each frame's ``row_offset`` is
    recomputed from the captured source so the *last*
    ``cfg.rows`` lines are in view (tail semantics). The
    frozen config is left untouched; the per-frame offset is a
    derived value that lives only inside this function.

    When ``max_frames`` is positive, the loop runs at most that many
    *iterations* (not emitted frames) before exiting — same
    semantics as :func:`_tail_and_render`. ``0`` (the default)
    means no cap.

    When ``max_seconds`` is positive, the loop also runs for at
    most that many wall-clock seconds before exiting — same
    semantics as :func:`_tail_and_render`. ``0`` (the default)
    means no cap. Composes with ``--max-frames`` — whichever cap
    fires first wins.

    Returns ``0`` on a clean exit. Capture errors
    (``RuntimeError`` / ``ValueError`` from the adapter or the
    region parser) are surfaced as a single stderr line and the
    loop continues — a transient adapter failure shouldn't kill
    a screen magnifier that's been running for hours.

    Note: this is the **screen-capture** tail. It is not the
    same as :func:`_tail_and_render` (which tails a file). The
    two functions are deliberately separate so each one stays
    small, single-purpose, and independently testable; the only
    thing they share is the loop shape.
    """
    sleep = clock if clock is not None else time.sleep
    # ``time_fn`` is the wall-clock source for ``--max-seconds``.
    # Same default (``time.monotonic``) and same injection pattern
    # as ``_tail_and_render``, so the two live-tail implementations
    # are symmetric: tests can drive either with a fake clock.
    actual_time = time_fn if time_fn is not None else time.monotonic
    # Default capture_fn: re-capture the screen adapter each poll.
    # Named ``_do_capture`` so we don't shadow the kwarg name.
    if capture_fn is None:

        def _do_capture() -> str:
            return _screen.capture_screen_to_source(cap, region=region)

        actual_capture = _do_capture
    else:
        actual_capture = capture_fn

    # Default change detector: string inequality. ``prev`` may be
    # ``None`` on the very first iteration (no previous frame
    # exists), in which case we *always* render the first frame.
    if has_changed is None:

        def _default_changed(prev: str | None, cur: str) -> bool:
            return prev != cur

        actual_changed = _default_changed
    else:
        actual_changed = has_changed

    last_text: str | None = None
    max_iters = max_frames if max_frames > 0 else None
    iter_count = 0
    # ``--max-seconds`` cap. Sample the wall clock once before the
    # loop starts, re-sample at the top of every iteration. Same
    # shape as ``_tail_and_render``'s cap so a future refactor can
    # lift the two into a shared helper if more caps land.
    loop_start = actual_time() if max_seconds > 0 else 0.0
    while stop_predicate is None or not stop_predicate():
        if max_iters is not None and iter_count >= max_iters:
            break
        if _max_seconds_exceeded(loop_start, max_seconds, actual_time()):
            break
        iter_count += 1
        try:
            text = actual_capture()
        except ValueError as exc:
            # Region / grid shape errors from the adapter.
            print(
                f"paw-zoom: --screen capture failed: {exc}",
                file=sys.stderr,
            )
            sleep(interval)
            continue
        except RuntimeError as exc:
            # Adapter-level failures (e.g. region parser, capture
            # pipeline). Treat as transient and keep looping.
            print(
                f"paw-zoom: --screen capture failed: {exc}",
                file=sys.stderr,
            )
            sleep(interval)
            continue
        # Change detection. The default ``!=`` is exact for a
        # text-grid capture (FakeScreen, a logged console, a
        # piped view of a text mode). Real OS adapters
        # (X11 / Win32 / Quartz) feed this same string through
        # their own pixel-to-text grid, and any visible change
        # will surface as a string diff. A perfectly static
        # screen yields exactly one frame — the same way a
        # perfectly static file does in the text-source path.
        if not actual_changed(last_text, text):
            sleep(interval)
            continue
        last_text = text
        # In --follow mode the row_offset is derived from the
        # captured source's current size on every frame, so the
        # viewport tracks the tail of the screen as it grows.
        # We rebuild a per-frame cfg instead of mutating the
        # caller's (frozen) config.
        frame_cfg = cfg
        if follow:
            frame_cfg = dataclasses.replace(
                cfg, row_offset=_tail_offset(text, cfg.rows)
            )
        rendered = render_viewport(text, frame_cfg)
        if line_numbers:
            # --line-numbers: per-frame, the first source row in
            # view is ``frame_cfg.row_offset + 1`` (1-based, same
            # as the text-source tail and the one-shot path).
            # ``frame_cfg`` already accounts for ``--follow``'s
            # per-frame rewrite, so a tail-tracking live screen
            # view shows the right numbers as the captured
            # region grows. Applied BEFORE --col-ruler so the
            # gutter is added to the viewport lines only, not to
            # the ruler line above them — the ruler then sits
            # cleanly above the guttered lines.
            rendered = _format_line_numbers(
                rendered,
                zoom=frame_cfg.zoom,
                first_source_row=max(1, frame_cfg.row_offset + 1),
            )
        if col_ruler:
            # --col-ruler: per-frame, the first source column in
            # view is ``frame_cfg.col_offset + 1`` (1-based, same
            # as the text-source tail and the one-shot path). The
            # col_offset does not change between frames (only
            # row_offset does, via ``--follow``), so the ruler is
            # stable across frames of a tail-tracking live screen
            # view. Applied AFTER --line-numbers so the ruler sits
            # above the guttered lines.
            rendered = _format_col_ruler(
                rendered,
                zoom=frame_cfg.zoom,
                cols=frame_cfg.cols,
                first_source_col=max(1, frame_cfg.col_offset + 1),
            )
        if sink is not None:
            sink(rendered)
        else:
            sys.stdout.write(rendered)
            # Frame separator. A blank line keeps successive
            # frames visually distinct in scrollback, same as
            # the text-source tail.
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
        "--line-numbers",
        action="store_true",
        dest="line_numbers",
        help=(
            "Prefix each magnified source row with its 1-based row "
            "number (right-aligned in a 4-char column, then a ' | ' "
            "gutter — same shape as 'cat -n' / 'nl' / most editors). "
            "The first number is the 1-based index of the first "
            "source row in view, so a 'paw-zoom --offset 12 ...' "
            "viewport starts at '13'. Useful for correlating the "
            "magnified block with the source: 'which source line "
            "did this magnified row come from?'. Composes with every "
            "render-driving flag (--zoom / --rows / --cols / "
            "--offset / --col-offset / --charset / --live / "
            "--follow / --max-frames / --max-seconds / --snapshot) "
            "and is ignored by the discovery flags (--list-backends, "
            "--info, --size, --stats, --sha) and --raw, which never "
            "produce magnified output to annotate. The gutter does "
            "not count against --cols: the magnified viewport's "
            "horizontal extent is unchanged."
        ),
    )
    parser.add_argument(
        "--col-ruler",
        action="store_true",
        dest="col_ruler",
        help=(
            "Prepend a column-number ruler line above the magnified "
            "viewport (same shape as the ruler in 'vim' / 'nano' / "
            "'less'). The ruler is exactly 'cols * zoom' code points "
            "wide, matching the magnified viewport below; for each "
            "source column N (1-based, starting at 'col_offset + 1') "
            "it shows the last digit of N repeated 'zoom' times, so "
            "'paw-zoom --col-ruler --cols 10 --zoom 2 ...' emits "
            "'1122334455667788990\\n' above the magnified block. "
            "Useful for correlating a magnified column with its "
            "source position: 'which source column is this "
            "magnified cell from?'. Composes with --line-numbers "
            "(ruler on top, gutter on the left) and every "
            "render-driving flag (--zoom / --rows / --cols / "
            "--offset / --col-offset / --charset / --live / "
            "--follow / --max-frames / --max-seconds / --snapshot); "
            "ignored by the discovery flags (--list-backends, "
            "--info, --size, --stats, --sha) and --raw, which never "
            "produce magnified output to annotate. The ruler does "
            "not count against --rows: the magnified viewport's "
            "vertical extent is unchanged."
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
            "Follow the source like 'tail -f': re-render the magnified "
            "viewport every time it changes. With --file PATH this is a "
            "text-source tail (mtime-tracked). With --screen this is a "
            "screen-capture tail (re-captures on every poll). Press "
            "Ctrl-C to stop. The poll interval is --interval "
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
    parser.add_argument(
        "--follow",
        action="store_true",
        help=(
            "Track the tail of --file PATH (like 'tail -f') instead of "
            "showing the start. Each rendered frame shows the last "
            "--rows lines of the source. Composes with --live (so the "
            "magnifier follows new lines as they arrive) and --snapshot. "
            "Without --live, --follow renders the tail exactly once and "
            "exits (equivalent to piping the file through 'tail -n N' "
            "before magnifying)."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Cap --live at N loop iterations (default: 0 = unlimited, "
            "the current behaviour). The loop stops after the N-th "
            "poll regardless of whether a frame was emitted, which "
            "is what makes --live actually exit on a quiet source. "
            "Useful for scripting: 'render the first 3 changes, then "
            "exit'. Composes with --follow, --snapshot, --max-seconds, "
            "and the standard --interval poll cadence. Has no effect "
            "without --live (the one-shot render always emits exactly "
            "one frame)."
        ),
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        metavar="SECS",
        help=(
            "Cap --live at SECS wall-clock seconds (default: 0 = "
            "unlimited, the current behaviour). The loop stops once "
            "SECS have elapsed since --live started, regardless of "
            "how many frames were emitted. Useful for scripting: "
            "'watch the live log for 30s, then exit' — without "
            "having to estimate the iteration count in advance. "
            "Composes with --follow, --snapshot, --max-frames "
            "(whichever cap fires first wins), and the standard "
            "--interval poll cadence. Has no effect without --live "
            "(the one-shot render always emits exactly one frame). "
            "Measured with time.monotonic() so an NTP slew can't "
            "extend the wall-clock window."
        ),
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help=(
            "Dump the source (text or screen) to stdout as plain "
            "text, with no magnification, no padding, and no "
            "announcement line unless --quiet is *not* set. With "
            "--screen, prints the captured screen grid (after "
            "--region clamping); with a text source, prints the "
            "resolved text verbatim. Useful for debugging a screen "
            "capture adapter ('what did the adapter actually see?') "
            "and for piping the raw source into a downstream tool. "
            "Mutually exclusive with --live, --follow, --max-frames, "
            "and --snapshot (none of them make sense for a raw "
            "dump). Has no effect with --list-backends (which "
            "already short-circuits)."
        ),
    )
    parser.add_argument(
        "--size",
        action="store_true",
        dest="size",
        help=(
            "Print the source dimensions (rows x cols, in code "
            "points) and exit 0 without rendering or capturing "
            "anything. The source is resolved exactly the way it "
            "would be for a render (positional -> --file -> "
            "stdin -> --screen with --backend/--region/--fake-grid "
            "for screen capture), then counted. Combine with "
            "--json for a single-line parseable object "
            "({'rows': N, 'cols': M}). The text-side analog of "
            "--info: --info answers 'what screen-capture setup "
            "would I get?'; --size answers 'how big is the "
            "source?'. Mutually exclusive with --info, --stats, "
            "--live, --follow, --max-frames, --snapshot, --raw, "
            "and --list-backends (all of them exist to drive a "
            "render or a different discovery; --size is the "
            "smallest discovery there is)."
        ),
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        dest="stats",
        help=(
            "Print per-source statistics (chars, lines, "
            "non-blank lines, max line width, mean line width) "
            "and exit 0 without rendering or capturing anything. "
            "The source is resolved exactly the way it would be "
            "for a render (positional -> --file -> stdin -> "
            "--screen with --backend/--region/--fake-grid for "
            "screen capture), then counted. The text-side "
            "counterpart of --size: --size answers 'how big is "
            "the source as a rectangle?'; --stats answers 'what "
            "is in the source?'. Combine with --json for a "
            "single-line parseable object ({'chars': N, 'lines': "
            "M, 'non_blank_lines': K, 'max_line_width': W, "
            "'mean_line_width': X.XX}). Mutually exclusive with "
            "--size, --info, --live, --follow, --max-frames, "
            "--snapshot, --raw, and --list-backends (all of "
            "them exist to drive a render or a different "
            "discovery; --stats is the 'counting' discovery)."
        ),
    )
    parser.add_argument(
        "--sha",
        action="store_true",
        dest="sha",
        help=(
            "Print a stable hex digest of the source (default: "
            "SHA-256, 64 lowercase hex chars) and exit 0 without "
            "rendering or capturing anything. The source is "
            "resolved exactly the way it would be for a render "
            "(positional -> --file -> stdin -> --screen with "
            "--backend/--region/--fake-grid for screen capture), "
            "then hashed as UTF-8. Useful in scripts that want to "
            "skip work when the input hasn't moved: capture the "
            "digest once, compare against a re-computed digest "
            "later, branch on the result. Combine with --json for "
            "a single-line parseable object ({'algorithm': "
            "'sha256', 'sha256': '...'}). Pick the algorithm with "
            "--sha-algo (sha1 / sha256 / sha512 / md5). Mutually "
            "exclusive with --size, --stats, --info, --live, "
            "--follow, --max-frames, --snapshot, --raw, and "
            "--list-backends (all of them exist to drive a "
            "render or a different discovery; --sha is the "
            "'fingerprint' discovery)."
        ),
    )
    parser.add_argument(
        "--sha-algo",
        default=DEFAULT_SHA_ALGORITHM,
        dest="sha_algo",
        metavar="NAME",
        help=(
            f"Hash algorithm for --sha (default: {DEFAULT_SHA_ALGORITHM}). "
            "Any algorithm accepted by Python's hashlib is supported "
            "(sha1, sha256, sha512, md5, ...). Has no effect without "
            "--sha. An unknown name is a usage error at parse time."
        ),
    )
    # v0.2: screen-capture flags. The group lives behind
    # ``add_screen_args`` so the parser stays readable.
    _screen.add_screen_args(parser)
    parser.add_argument(
        "--info",
        action="store_true",
        dest="info",
        help=(
            "Print a short description of the screen-capture setup "
            "this invocation would use (selected backend, whether "
            "an adapter is available, the screen size, and the "
            "resolved region) and exit 0 without capturing or "
            "rendering anything. Requires --screen. Combine with "
            "--json for a single-line parseable object. Mutually "
            "exclusive with --live, --follow, --max-frames, "
            "--snapshot, and --raw (all of them exist to drive a "
            "render; --info answers the 'is this set up right?' "
            "question instead)."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help=(
            "Combine with --list-backends, --info, --size, or "
            "--stats to emit a single-line JSON object instead of "
            "the default text output. --list-backends --json -> "
            "{'backends': [...]}; --info --json -> {'backend': "
            "..., 'available': ..., 'adapter': ..., "
            "'screen_size': [w, h], 'region': [x, y, w, h]}; "
            "--size --json -> {'rows': N, 'cols': M}; --stats "
            "--json -> {'chars': N, 'lines': M, "
            "'non_blank_lines': K, 'max_line_width': W, "
            "'mean_line_width': X.XX}. Using --json without "
            "--list-backends, --info, --size, or --stats is a "
            "usage error (exit 2)."
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
    # --size is the text-source-side analog of --info: a
    # metadata-only mode that exits 0 before any render. It
    # contradicts the same render-driving flags --info does,
    # plus the other discovery flags (--info, --stats,
    # --list-backends) because each one is a different kind of
    # discovery and we don't want to emit more than one of them
    # per invocation. We run this check *before* the
    # value-of-flag checks below (e.g. ``--live requires --file
    # or --screen``) so the mutual-exclusion message wins when
    # both would fire — the contradiction between the two flags
    # is the more useful diagnostic.
    if args.size:
        for flag, value in (
            ("--info", args.info),
            ("--stats", args.stats),
            ("--live", args.live),
            ("--follow", args.follow),
            ("--max-frames", args.max_frames),
            ("--snapshot", args.snapshot),
            ("--raw", args.raw),
            ("--list-backends", args.list_backends),
        ):
            if value:
                print(
                    f"paw-zoom: --size cannot be combined with {flag} "
                    f"(--size is a metadata-only mode that exits "
                    f"before any render or other discovery)",
                    file=sys.stderr,
                )
                raise SystemExit(2)
    # --stats is the *counting* discovery — the text-side
    # counterpart of --size. It runs through the same
    # source-resolution block --size does (so the source can be a
    # text arg, --file, stdin, or a --screen capture) and
    # short-circuits BEFORE the render / --raw block, so it
    # contradicts the same render-driving flags --size does,
    # plus the other discovery flags (--size, --info,
    # --list-backends) for the same reason --size's
    # mutual-exclusion list does: each one is a different kind
    # of discovery, and we don't want to emit more than one of
    # them per invocation. We run this check *after* the --size
    # block above so --size's mutual-exclusion message wins
    # when both would fire (the contradiction between the two
    # discovery flags is the more useful diagnostic).
    if args.stats:
        for flag, value in (
            ("--size", args.size),
            ("--info", args.info),
            ("--live", args.live),
            ("--follow", args.follow),
            ("--max-frames", args.max_frames),
            ("--snapshot", args.snapshot),
            ("--raw", args.raw),
            ("--list-backends", args.list_backends),
        ):
            if value:
                print(
                    f"paw-zoom: --stats cannot be combined with {flag} "
                    f"(--stats is a metadata-only mode that exits "
                    f"before any render or other discovery)",
                    file=sys.stderr,
                )
                raise SystemExit(2)
    # --sha is the *fingerprint* discovery — the stable-hash
    # companion of --size / --stats. It runs through the same
    # source-resolution block they do (so the source can be a
    # text arg, --file, stdin, or a --screen capture) and
    # short-circuits BEFORE the render / --raw block, so it
    # contradicts the same render-driving flags --size and
    # --stats do, plus the other discovery flags (--size,
    # --stats, --info, --list-backends) for the same reason
    # they do: each one is a different kind of discovery, and
    # we don't want to emit more than one of them per
    # invocation. We run this check *after* the --size and
    # --stats blocks above so their mutual-exclusion messages
    # win when more than one discovery flag would fire (the
    # contradiction between two discovery flags is the more
    # useful diagnostic). --sha-algo alone is fine (it just
    # sets the algorithm for the next --sha invocation the
    # user might add); we only reject the combination when
    # --sha is set.
    if args.sha:
        for flag, value in (
            ("--size", args.size),
            ("--stats", args.stats),
            ("--info", args.info),
            ("--live", args.live),
            ("--follow", args.follow),
            ("--max-frames", args.max_frames),
            ("--snapshot", args.snapshot),
            ("--raw", args.raw),
            ("--list-backends", args.list_backends),
        ):
            if value:
                print(
                    f"paw-zoom: --sha cannot be combined with {flag} "
                    f"(--sha is a metadata-only mode that exits "
                    f"before any render or other discovery)",
                    file=sys.stderr,
                )
                raise SystemExit(2)
        # Validate the algorithm eagerly so a typo (e.g.
        # ``--sha-algo SHA-256`` with the upper-case name) is
        # caught at parse time with a friendly exit-2 message
        # instead of crashing the render path with a generic
        # ``hashlib`` ValueError. We use the same helper the
        # runtime path uses so the two cannot drift.
        try:
            _source_sha("", algorithm=args.sha_algo)
        except ValueError as exc:
            print(f"paw-zoom: --sha-algo: {exc}", file=sys.stderr)
            raise SystemExit(2)
    if args.live and not args.file and not args.screen:
        print(
            "paw-zoom: --live requires --file PATH or --screen "
            "(there is nothing else to tail)",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if args.max_frames < 0:
        print(
            "paw-zoom: --max-frames must be >= 0 (0 means unlimited)",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if args.max_seconds < 0:
        print(
            "paw-zoom: --max-seconds must be >= 0 (0 means unlimited)",
            file=sys.stderr,
        )
        raise SystemExit(2)
    # --raw is the "no magnification" mode: dump the source as-is
    # and exit. It contradicts the live / follow / max-frames /
    # snapshot machinery, all of which exist to feed the magnified
    # pipeline. We reject the combinations explicitly (exit 2) so
    # the user gets a clear message instead of a silently
    # contradictory behaviour. The viewport-modifying flags
    # (--zoom / --rows / --cols / --offset / --col-offset /
    # --charset) are *not* rejected: the output shape of a raw
    # dump does not depend on them, so silently ignoring them
    # matches the spirit of "dump it as-is" without forcing the
    # user to drop flags from a shell alias or wrapper.
    if args.raw:
        for flag, value in (
            ("--live", args.live),
            ("--follow", args.follow),
            ("--max-frames", args.max_frames),
            ("--snapshot", args.snapshot),
        ):
            if value:
                print(
                    f"paw-zoom: --raw cannot be combined with {flag} "
                    f"(a raw dump is a single non-magnified output)",
                    file=sys.stderr,
                )
                raise SystemExit(2)
    # v0.2: --fake-grid needs --backend fake (otherwise it has
    # nothing to do, and silently ignoring the grid would be a
    # confusing footgun). --screen without a usable backend
    # (including 'auto' on a system with no OS adapter yet)
    # becomes a runtime error in main(), not here, because
    # 'auto' is genuinely "try what you can".
    if args.fake_grid is not None and args.backend != "fake":
        print(
            "paw-zoom: --fake-grid requires --backend fake",
            file=sys.stderr,
        )
        raise SystemExit(2)
    # --region needs --screen; we don't want a typo like
    # ``--region 0,0,10,5`` (forgetting --screen) to silently
    # parse and then do nothing.
    if args.region is not None and not args.screen:
        print(
            "paw-zoom: --region requires --screen",
            file=sys.stderr,
        )
        raise SystemExit(2)
    # --info is the screen-capture "what would the pipeline do?"
    # debug flag. It requires --screen (no point describing a
    # capture setup when we're in text-source mode) and
    # contradicts every other render-driving flag for the same
    # reason --raw does: those flags exist to feed a render,
    # --info exists to skip the render. We reject the
    # combinations explicitly (exit 2) so a typo never
    # silently no-ops.
    if args.info:
        if not args.screen:
            print(
                "paw-zoom: --info requires --screen "
                "(it describes the screen-capture setup)",
                file=sys.stderr,
            )
            raise SystemExit(2)
        for flag, value in (
            ("--live", args.live),
            ("--follow", args.follow),
            ("--max-frames", args.max_frames),
            ("--snapshot", args.snapshot),
            ("--raw", args.raw),
        ):
            if value:
                print(
                    f"paw-zoom: --info cannot be combined with {flag} "
                    f"(--info is a metadata-only mode that exits "
                    f"before any render)",
                    file=sys.stderr,
                )
                raise SystemExit(2)
    # --json only pairs with --list-backends, --info, --size,
    # --stats, or --sha. Anything else is ambiguous — an empty
    # JSON object would be a worse failure mode than a clear
    # stderr message.
    if args.as_json and not (
        args.list_backends or args.info or args.size or args.stats or args.sha
    ):
        print(
            "paw-zoom: --json requires --list-backends, --info, "
            "--size, --stats, or --sha",
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
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2

    # v0.2: discovery short-circuit. ``--list-backends`` and
    # ``--list-backends --json`` exit 0 without doing any
    # source resolution or capture — they are pure metadata
    # about what backends the binary knows about, useful for
    # shell completion and for ``jq``-driven tooling.
    if args.list_backends:
        if args.as_json:
            print(_screen.to_json("backends"))
        else:
            for name in _screen.list_backends():
                print(name)
        return 0

    # v0.2.1: --info is the screen-capture diagnostic
    # short-circuit. It runs *before* the source-resolution
    # block (so a missing OS adapter is reported as
    # ``available: false`` rather than a fatal exit-1), uses
    # the same ``describe_capture`` helper the library callers
    # can use, and respects --json for the same single-line
    # parseable shape the rest of the discovery flags use.
    if args.info:
        # --fake-grid is optional in --info mode: ``describe_capture``
        # handles the "fake backend without a grid" case by
        # reporting ``available: false`` (the same answer the
        # user would get from the real --screen path), so a
        # diagnostic run never lies about the setup. We still
        # try to parse it if it's there so the dict's
        # ``fake_grid`` shape matches what the real path would
        # see — a malformed grid shows up as a usage error
        # rather than a misleading "available: false" line.
        fake_grid: list[str] | None = None
        if args.fake_grid is not None:
            try:
                fake_grid = _screen._read_fake_grid(args.fake_grid)
            except ValueError as exc:
                print(f"paw-zoom: {exc}", file=sys.stderr)
                return 2
        info = _screen.describe_capture(
            args.backend,
            region=args.region,
            fake_grid=fake_grid,
        )
        if args.as_json:
            print(_screen.describe_to_json(info))
        else:
            print(_screen.describe_to_text(info))
        return 0

    # v0.2: --screen takes over source resolution. We bypass
    # ``_resolve_source`` entirely and capture from a
    # ``ScreenCapture`` adapter instead. The captured grid
    # becomes the source string the rest of the pipeline
    # (ZoomConfig, render_viewport, --snapshot, --live /
    # --follow / --max-frames) already understands.
    # ``cap`` is initialised to None up-front so the live
    # dispatch below (which also uses it when ``--screen`` is
    # set) doesn't trip a "possibly unbound" diagnostic.
    cap: _screen.ScreenCapture | None = None
    if args.screen:
        fake_grid: list[str] | None = None
        if args.fake_grid is not None:
            try:
                fake_grid = _screen._read_fake_grid(args.fake_grid)
            except ValueError as exc:
                print(f"paw-zoom: {exc}", file=sys.stderr)
                return 2
        cap = _screen.get_capture(args.backend, fake_grid=fake_grid)
        if cap is None:
            print(_screen.backend_unsupported_message(args.backend), file=sys.stderr)
            return 1
        try:
            source = _screen.capture_screen_to_source(cap, region=args.region)
        except ValueError as exc:
            print(f"paw-zoom: {exc}", file=sys.stderr)
            return 2
    else:
        try:
            source = _resolve_source(
                text=args.text,
                file=args.file,
                stdin_text=_read_stdin(),
            )
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    # v0.2.2: --size is the text-side analog of --info. It runs
    # *after* source resolution (so the source can be a text arg,
    # --file, stdin, or a --screen capture) and *before* the
    # render / --raw block, so it short-circuits without ever
    # building a ZoomConfig or writing a magnified viewport.
    # The source string we just resolved is exactly the string
    # the renderer would see (same _resolve_source stripping,
    # same capture_screen_to_source joining), so the size we
    # report is the size the user would magnify.
    if args.size:
        size = _source_size(source)
        if args.as_json:
            print(_size_to_json(size))
        else:
            print(_size_to_text(size))
        return 0

    # v0.2.3: --stats is the *counting* companion of --size —
    # --size answers "how big is the source as a rectangle?"
    # (rows × cols); --stats answers "what is in the source?"
    # (chars / lines / non-blank lines / max line width / mean
    # line width). It runs at the same point in the pipeline as
    # --size (after source resolution, before the render /
    # --raw block) so the source it counts is the source the
    # renderer would see, and a user can pipe the same input
    # through both flags without surprises. Mutual-exclusion
    # is enforced in parse_args() — reaching this branch with
    # --stats means the user asked for exactly one thing.
    if args.stats:
        stats = _source_stats(source)
        if args.as_json:
            print(_stats_to_json(stats))
        else:
            print(_stats_to_text(stats))
        return 0

    # v0.2.4: --sha is the *fingerprint* discovery — the
    # stable-hash companion of --size / --stats. --size answers
    # "how big is the source as a rectangle?"; --stats answers
    # "what is in the source?"; --sha answers "is this the
    # same source I saw last time?". It runs at the same point
    # in the pipeline as --size and --stats (after source
    # resolution, before the render / --raw block) so the
    # source it hashes is the source the renderer would see,
    # and a user can pipe the same input through any of the
    # three flags without surprises. Mutual-exclusion is
    # enforced in parse_args() — reaching this branch with
    # --sha means the user asked for exactly one thing. The
    # algorithm is validated at parse time too, so a typo
    # (e.g. ``--sha-algo SHA-256`` with the upper-case name)
    # never reaches this line.
    if args.sha:
        digest = _source_sha(source, algorithm=args.sha_algo)
        if args.as_json:
            print(_sha_to_json(digest, algorithm=args.sha_algo))
        else:
            print(_sha_to_text(digest))
        return 0

    if args.raw:
        # --raw: dump the source to stdout verbatim, no magnification,
        # no padding, no ZoomConfig. The source is already in the
        # ``str`` shape we want: text sources have been resolved
        # (and the trailing newline stripped by ``_resolve_source``);
        # screen sources have been captured by
        # ``_screen.capture_screen_to_source`` (rows joined with
        # ``\n``, padded with spaces for out-of-range cells). We
        # respect --quiet: by default the announcement line is
        # suppressed in --raw mode (the whole point is "give me the
        # data and nothing else"), and --quiet on top of that would
        # be a no-op. The source string is written as-is followed
        # by a single trailing newline so the cursor lands on a
        # new line for the shell prompt.
        if not args.quiet:
            # An opt-in one-liner identifying what was dumped, so
            # the user can tell a raw screen capture from a raw
            # text source in scrollback.
            kind = "screen" if args.screen else "text"
            print(
                f"🐾 paw-zoom: --raw {kind} dump "
                f"({len(source)} chars, "
                f"{source.count(chr(10)) + 1 if source else 0} lines)",
                file=sys.stderr,
            )
        # Use sys.stdout.write + newline so the dump ends on a
        # fresh line regardless of the source's own trailing
        # newline (text sources have their trailing \n stripped
        # by ``_resolve_source``; screen captures end with a
        # ``\n`` from the ``"\n".join`` in
        # ``capture_screen_to_source``).
        sys.stdout.write(source)
        if not source.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()
        return 0

    cfg = ZoomConfig(
        rows=args.rows,
        cols=args.cols,
        zoom=args.zoom,
        fill=args.fill,
        row_offset=args.offset,
        col_offset=args.col_offset,
    )
    if args.follow:
        # --follow is a row_offset modifier: re-derive it from the
        # source's current size so the last ``cfg.rows`` lines are
        # in view. We need a source for that, so the resolver has
        # already run above. If the source is empty, _tail_offset
        # returns 0 and we fall back to the default offset.
        cfg = dataclasses.replace(cfg, row_offset=_tail_offset(source, cfg.rows))
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
            if args.screen:
                # Screen-capture tail. The adapter was already
                # constructed in the --screen branch above; we
                # re-capture from it on every poll. ``cap`` is
                # always set in this branch because
                # ``parse_args()`` rejects ``--live`` without
                # either ``--file`` or ``--screen``, and the
                # source-resolution block early-returns when
                # the screen backend is unavailable. The
                # ``cap is not None`` guard exists as
                # defence-in-depth (and to keep the type
                # checker happy without an ``assert`` that
                # would be stripped under ``python -O``).
                if cap is None:
                    print(
                        "paw-zoom: --live --screen: no capture backend "
                        "available; this should have been caught at "
                        "source resolution",
                        file=sys.stderr,
                    )
                    return 1
                return _tail_screen_and_render(
                    cap,
                    cfg,
                    region=args.region,
                    interval=args.interval,
                    follow=args.follow,
                    max_frames=args.max_frames,
                    max_seconds=args.max_seconds,
                    sink=_sink,
                    line_numbers=args.line_numbers,
                    col_ruler=args.col_ruler,
                )
                return _tail_and_render(
                    args.file,
                    cfg,
                    interval=args.interval,
                    follow=args.follow,
                    max_frames=args.max_frames,
                    max_seconds=args.max_seconds,
                    sink=_sink,
                    line_numbers=args.line_numbers,
                    col_ruler=args.col_ruler,
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
    if args.line_numbers:
        # --line-numbers: prepend the 1-based source-row index to
        # each magnified line, so the user can correlate the
        # rendered block with the source. The first source row
        # in view is ``cfg.row_offset + 1`` (1-based, matching
        # what `cat -n` would print); the gutter is added on
        # top of the rendered viewport so the magnified width
        # is unchanged. Applied BEFORE --col-ruler so the gutter
        # is added to the viewport lines only, not to the ruler
        # line above them — the ruler then sits cleanly above
        # the guttered lines (top → ruler; below → guttered
        # viewport). If --col-ruler ran first, the ruler would
        # itself get a "1 │ " gutter prefix, which would break
        # the visual alignment.
        first_row = max(1, cfg.row_offset + 1)
        rendered = _format_line_numbers(
            rendered, zoom=cfg.zoom, first_source_row=first_row
        )
    if args.col_ruler:
        # --col-ruler: prepend a 1-line column-number ruler above the
        # (now guttered) magnified viewport. The ruler mirrors the
        # magnified width (cols * zoom code points) and shows the
        # last digit of each source column index, repeated ``zoom``
        # times. The first source column in view is ``cfg.col_offset
        # + 1`` (1-based, matching the --line-numbers convention for
        # rows). Applied AFTER --line-numbers so the ruler sits
        # cleanly above the guttered lines (top → ruler; below →
        # guttered viewport).
        first_col = max(1, cfg.col_offset + 1)
        rendered = _format_col_ruler(
            rendered,
            zoom=cfg.zoom,
            cols=cfg.cols,
            first_source_col=first_col,
        )
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
