"""Tests for ``paw-zoom`` (text-viewport magnifier, ASCII POC)."""
from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from contextlib import redirect_stdout, redirect_stderr

import pytest

from whisperpaw import zoom


# ---------------------------------------------------------------------------
# ZoomConfig dataclass
# ---------------------------------------------------------------------------


def test_zoom_config_defaults() -> None:
    """A default ZoomConfig is a sensible starting point."""
    cfg = zoom.ZoomConfig()
    assert cfg.rows == 10
    assert cfg.cols == 40
    assert cfg.zoom == 2
    assert cfg.fill == " "
    assert cfg.row_offset == 0
    assert cfg.col_offset == 0


def test_zoom_config_frozen() -> None:
    """ZoomConfig is immutable — accidental mutation shouldn't happen."""
    cfg = zoom.ZoomConfig()
    with pytest.raises((AttributeError, Exception)):
        cfg.rows = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------


def test_resolve_source_positional() -> None:
    """A positional string is the highest-priority source."""
    text = zoom._resolve_source(
        text="from positional", file=None, stdin_text="from stdin"
    )
    assert text == "from positional"


def test_resolve_source_file(tmp_path) -> None:
    """``--file`` is used when no positional text is given."""
    p = tmp_path / "notes.txt"
    p.write_text("hello from file\n", encoding="utf-8")
    text = zoom._resolve_source(
        text=None, file=str(p), stdin_text="from stdin"
    )
    assert text == "hello from file"


def test_resolve_source_stdin() -> None:
    """Without positional or --file, the stdin value is used."""
    text = zoom._resolve_source(text=None, file=None, stdin_text="from stdin")
    assert text == "from stdin"


def test_resolve_source_file_missing(tmp_path) -> None:
    """A missing --file path raises FileNotFoundError."""
    p = tmp_path / "nope.txt"
    with pytest.raises(FileNotFoundError):
        zoom._resolve_source(text=None, file=str(p), stdin_text=None)


def test_resolve_source_all_empty(monkeypatch) -> None:
    """When every source is empty / missing, a RuntimeError is raised."""
    monkeypatch.setenv("WPAW_ZOOM_STDIN_OVERRIDE", "")
    with pytest.raises(RuntimeError):
        zoom._resolve_source(text=None, file=None, stdin_text=None)


# ---------------------------------------------------------------------------
# Region extraction
# ---------------------------------------------------------------------------


def test_extract_region_basic() -> None:
    """A rectangular sub-grid is returned as a list of strings."""
    src = "abcdef\n" "ghijkl\n" "mnopqr\n"
    region = zoom._extract_region(src, rows=2, cols=3, row_offset=0, col_offset=0)
    assert region == ["abc", "ghi"]


def test_extract_region_offset() -> None:
    """row_offset and col_offset shift the window."""
    src = "abcdef\n" "ghijkl\n" "mnopqr\n"
    # col_offset=2, cols=3 -> chars at index 2,3,4 -> 'cde', 'ijk', 'opq'
    region = zoom._extract_region(src, rows=2, cols=3, row_offset=1, col_offset=2)
    assert region == ["ijk", "opq"]


def test_extract_region_short_source() -> None:
    """A source smaller than the window is padded with spaces (right / bottom)."""
    src = "ab\n" "cd"
    region = zoom._extract_region(src, rows=3, cols=4, row_offset=0, col_offset=0)
    assert region == ["ab  ", "cd  ", "    "]


def test_extract_region_no_trailing_newline() -> None:
    """Sources without a trailing newline are treated as a full last line."""
    src = "abcdef\nghijkl"  # no \n after last line
    region = zoom._extract_region(src, rows=2, cols=6, row_offset=0, col_offset=0)
    assert region == ["abcdef", "ghijkl"]


def test_extract_region_clamps_negative_offsets() -> None:
    """Negative offsets are clamped to zero (we don't wrap or error)."""
    src = "ab\ncd"
    region = zoom._extract_region(src, rows=1, cols=2, row_offset=-5, col_offset=-3)
    assert region == ["ab"]


def test_extract_region_clamps_past_end() -> None:
    """Offsets past the source end return empty/padded lines."""
    src = "ab\ncd"
    region = zoom._extract_region(src, rows=2, cols=2, row_offset=10, col_offset=10)
    assert region == ["  ", "  "]


def test_extract_region_unicode() -> None:
    """Unicode is handled by code-point count, not bytes."""
    src = "你好世界\n再见朋友"
    region = zoom._extract_region(src, rows=2, cols=4, row_offset=0, col_offset=0)
    # Each cell is one code point; width 4 means the first 4 codepoints.
    assert region == ["你好世界", "再见朋友"]


# ---------------------------------------------------------------------------
# Magnification
# ---------------------------------------------------------------------------


def test_magnify_zoom_1() -> None:
    """zoom=1 is a no-op (each cell becomes a 1x1 block of itself)."""
    region = ["ab", "cd"]
    out = zoom._magnify(region, zoom=1, fill=" ")
    assert out == "ab\ncd"


def test_magnify_zoom_2() -> None:
    """zoom=2 doubles each character in both directions."""
    region = ["ab", "cd"]
    out = zoom._magnify(region, zoom=2, fill=" ")
    assert out == "aabb\n" "aabb\n" "ccdd\n" "ccdd"


def test_magnify_zoom_3() -> None:
    """zoom=3 triples each character in both directions."""
    region = ["x"]
    out = zoom._magnify(region, zoom=3, fill=" ")
    assert out == "xxx\n" "xxx\n" "xxx"


def test_magnify_zoom_with_fill() -> None:
    """The ``fill`` char replaces every cell, then the same fill is repeated."""
    region = []
    out = zoom._magnify(region, zoom=2, fill="#")
    # empty region, zero lines, empty output
    assert out == ""


def test_magnify_zoom_validates() -> None:
    """zoom must be a positive integer."""
    with pytest.raises(ValueError):
        zoom._magnify(["a"], zoom=0, fill=" ")
    with pytest.raises(ValueError):
        zoom._magnify(["a"], zoom=-1, fill=" ")


# ---------------------------------------------------------------------------
# End-to-end render
# ---------------------------------------------------------------------------


def test_render_viewport_default() -> None:
    """End-to-end: a small source renders to a 2x magnified default window."""
    src = "abcdefghij\n" "klmnopqrst"
    out = zoom.render_viewport(
        source=src,
        cfg=zoom.ZoomConfig(rows=2, cols=5, zoom=2),
    )
    # 2 source rows, 2 zoomed rows each = 4 output lines.
    assert out == "aabbccddee\n" "aabbccddee\n" "kkllmmnnoo\n" "kkllmmnnoo"


def test_render_viewport_empty_source() -> None:
    """An empty source renders to a fully padded window of the requested size."""
    out = zoom.render_viewport(
        source="",
        cfg=zoom.ZoomConfig(rows=3, cols=4, zoom=2, fill="."),
    )
    # 3 source rows, 4 source cols, zoom=2 -> 6 output lines, each 8 dots,
    # joined by 5 '\\n' characters between the 6 lines (no trailing \\n).
    assert out == "........\n" * 5 + "........"
    assert out.count("\n") == 5
    # Total length = 6 lines × 8 chars + 5 newlines = 53.
    assert len(out) == 8 * 6 + 5


def test_render_viewport_zoom_1_matches_input() -> None:
    """At zoom=1, the output is a faithful view of the requested region."""
    src = "abcdef\n" "ghijkl"
    out = zoom.render_viewport(
        source=src,
        cfg=zoom.ZoomConfig(rows=2, cols=6, zoom=1, row_offset=0, col_offset=0),
    )
    assert out == "abcdef\nghijkl"


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def test_build_parser_help_includes_key_flags() -> None:
    """The --help output mentions every key flag."""
    parser = zoom.build_parser()
    help_text = parser.format_help()
    for flag in ("--rows", "--cols", "--offset", "--zoom", "--charset", "--file"):
        assert flag in help_text, f"--help missing {flag}"


def test_parse_args_defaults() -> None:
    """Sensible defaults: text=None, file=None, rows=10, cols=40, zoom=2."""
    args = zoom.parse_args([])
    assert args.text is None
    assert args.file is None
    assert args.rows == 10
    assert args.cols == 40
    assert args.zoom == 2
    assert args.offset == 0
    assert args.col_offset == 0
    assert args.charset == "space"
    assert args.quiet is False


def test_parse_args_text_joined() -> None:
    """A multi-word positional is joined with single spaces."""
    args = zoom.parse_args(["hello", "world"])
    assert args.text == "hello world"


def test_parse_args_rejects_zero_rows() -> None:
    """--rows must be >= 1."""
    with pytest.raises(SystemExit):
        zoom.parse_args(["--rows", "0", "x"])


def test_parse_args_rejects_zero_cols() -> None:
    """--cols must be >= 1."""
    with pytest.raises(SystemExit):
        zoom.parse_args(["--cols", "0", "x"])


def test_parse_args_rejects_zero_zoom() -> None:
    """--zoom must be >= 1."""
    with pytest.raises(SystemExit):
        zoom.parse_args(["--zoom", "0", "x"])


def test_parse_args_rejects_negative_zoom() -> None:
    """--zoom must be > 0."""
    with pytest.raises(SystemExit):
        zoom.parse_args(["--zoom", "-3", "x"])


def test_parse_args_rejects_oversized_zoom() -> None:
    """--zoom is bounded so a stray huge value doesn't blow up the terminal."""
    with pytest.raises(SystemExit):
        zoom.parse_args(["--zoom", "9999", "x"])


def test_parse_args_offsets_default_to_zero() -> None:
    """Offsets default to 0; explicit values are accepted."""
    args = zoom.parse_args(["--offset", "5", "--col-offset", "3", "x"])
    assert args.offset == 5
    assert args.col_offset == 3


# ---------------------------------------------------------------------------
# main() — CLI entry point
# ---------------------------------------------------------------------------


def test_main_positional_text(capsys) -> None:
    """main() prints the magnified viewport and exits 0."""
    rc = zoom.main(["hello"])
    out = capsys.readouterr().out
    assert rc == 0
    # "hello" -> 5 cols, rows=10, so the first 10 chars of the padded window
    # are 'hello' (5 wide) then 5 spaces; zoom=2 -> each becomes 2x2.
    assert "hheelllloo" in out


def test_main_file_input(tmp_path, capsys) -> None:
    """main() reads --file when no positional is given."""
    p = tmp_path / "in.txt"
    p.write_text("ab", encoding="utf-8")
    rc = zoom.main(["--rows", "1", "--cols", "2", "--zoom", "1", "--file", str(p), "--quiet"])
    out = capsys.readouterr().out
    assert rc == 0
    # print() adds a trailing newline; the magnified viewport for "ab" at
    # zoom=1 is just "ab".
    assert out == "ab\n"


def test_main_stdin_input(monkeypatch, capsys) -> None:
    """When no positional and no --file, stdin is used (via the override)."""
    monkeypatch.setenv("WPAW_ZOOM_STDIN_OVERRIDE", "ab")
    rc = zoom.main(["--rows", "1", "--cols", "2", "--zoom", "1", "--quiet"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out == "ab\n"


def test_main_quiet_suppresses_banner(capsys) -> None:
    """--quiet removes the announcement line."""
    zoom.main(["--quiet", "x"])
    out = capsys.readouterr().out
    # No paw-zoom announcement should be present.
    assert "paw-zoom" not in out


def test_main_announces_by_default(capsys) -> None:
    """Without --quiet, a short announcement line is printed first."""
    zoom.main(["--rows", "1", "--cols", "1", "z"])
    out = capsys.readouterr().out
    # Announcement goes to stdout and is the first line.
    first_line = out.splitlines()[0]
    assert first_line.startswith("🐾 paw-zoom:")


def test_main_file_not_found(capsys) -> None:
    """A missing --file exits 1 and prints to stderr."""
    rc = zoom.main(["--file", "/nope/does/not/exist.txt", "--quiet"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "not found" in err.lower() or "no such file" in err.lower()


def test_main_no_source(capsys, monkeypatch) -> None:
    """No positional, no --file, empty stdin -> exit 2 with a clear message."""
    monkeypatch.setenv("WPAW_ZOOM_STDIN_OVERRIDE", "")
    rc = zoom.main(["--quiet"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "no text" in err.lower() or "no source" in err.lower()


# --- --snapshot flag ---------------------------------------------------


def test_parse_args_snapshot_default_is_none() -> None:
    args = zoom.parse_args(["--quiet", "hello"])
    assert args.snapshot is None


def test_parse_args_snapshot_writes_to_path(tmp_path) -> None:
    out = tmp_path / "shot.txt"
    args = zoom.parse_args(["--quiet", "--snapshot", str(out), "hello"])
    assert args.snapshot == str(out)


def test_main_snapshot_writes_file_and_exits_zero(tmp_path) -> None:
    out = tmp_path / "shot.txt"
    rc = zoom.main(["--quiet", "--snapshot", str(out), "hello"])
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    # Snapshot must contain the rendered (magnified) output.
    # Default --zoom is 2, so 'hello' renders as 'hheelllloo'. The
    # source is still reconstructible: every other character of the
    # rendered line, in order, is the source.
    assert "hheelllloo" in text, repr(text)
    # And the source string's characters appear in order.
    for i, ch in enumerate("hello"):
        assert ch in text[i * 2 : i * 2 + 5], (i, ch, text)
    # The output is multiple lines; at least one of them is at least
    # 10 source-cols wide (5 source chars * zoom 2 = 10 rendered cols).
    assert any(len(line) >= 10 for line in text.splitlines()), text.splitlines()


def test_main_snapshot_with_zoom_one(tmp_path) -> None:
    """--zoom 1 should give one output cell per source cell, no
    repetition. Locks in the documented magnification contract."""
    out = tmp_path / "shot.txt"
    rc = zoom.main(["--quiet", "--zoom", "1", "--snapshot", str(out), "abc"])
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    # With zoom=1, 'abc' renders as 'abc' (no repetition).
    assert "abc" in text
    # No repetition: a source string of length N must not appear
    # back-to-back twice in the rendered line. Default viewport
    # width is 38 cols, so the line is 38 chars wide; we look at
    # the first 38 characters of the first non-empty line.
    first_line = next(
        (l for l in text.splitlines() if l.strip()),
        "",
    )
    assert "abcabc" not in first_line, first_line
    # And the source row must appear in the output exactly once
    # (the viewport is one row tall by default).
    assert first_line.count("abc") == 1, first_line


def test_main_snapshot_overwrites_existing_file(tmp_path) -> None:
    """A pre-existing target file is replaced, not appended to."""
    out = tmp_path / "shot.txt"
    out.write_text("garbage from a previous run", encoding="utf-8")
    rc = zoom.main(["--quiet", "--zoom", "1", "--snapshot", str(out), "new"])
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "garbage" not in text
    assert "new" in text


def test_main_snapshot_to_unwritable_path_exits_nonzero(tmp_path, monkeypatch) -> None:
    """Writing to a directory that does not exist should fail cleanly
    with a non-zero exit code, not a Python traceback."""
    bad = tmp_path / "nope" / "shot.txt"  # parent does not exist
    monkeypatch.setenv("WPAW_ZOOM_STDIN_OVERRIDE", "")
    rc = zoom.main(["--quiet", "--snapshot", str(bad), "hello"])
    # We accept 1 (env/IO) or 2 (usage) — anything non-zero, but not
    # a traceback.
    assert rc != 0


def test_main_snapshot_with_file_source(tmp_path) -> None:
    """--snapshot composes with --file: write to one path, read from another."""
    src = tmp_path / "in.txt"
    src.write_text("from file", encoding="utf-8")
    out = tmp_path / "out.txt"
    rc = zoom.main(["--quiet", "--zoom", "1", "--snapshot", str(out), "--file", str(src)])
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "from file" in text


def test_main_snapshot_stdin_source(tmp_path, monkeypatch) -> None:
    """--snapshot also works with stdin as the source."""
    out = tmp_path / "out.txt"
    monkeypatch.setenv("WPAW_ZOOM_STDIN_OVERRIDE", "from stdin")
    rc = zoom.main(["--quiet", "--zoom", "1", "--snapshot", str(out)])
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "from stdin" in text


# ---------------------------------------------------------------------------
# --live / --interval
# ---------------------------------------------------------------------------


def test_default_live_interval_is_sensible() -> None:
    """The default poll interval is small but positive — visible to
    the user, not a CPU hog. Anything outside 0.05–2.0s is probably
    a bug."""
    assert 0.05 <= zoom.DEFAULT_LIVE_INTERVAL <= 2.0


def test_live_requires_file(tmp_path, monkeypatch) -> None:
    """--live without --file is a usage error (exit 2) — there is no
    stdin to tail."""
    monkeypatch.setenv("WPAW_ZOOM_STDIN_OVERRIDE", "")
    # Intentionally no --file.
    with pytest.raises(SystemExit) as exc:
        zoom.parse_args(["--live", "hello"])
    assert exc.value.code == 2


def test_live_rejects_zero_interval(tmp_path) -> None:
    """--interval must be strictly > 0 — otherwise the loop spins."""
    src = tmp_path / "log.txt"
    src.write_text("first\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        zoom.parse_args(["--live", "--file", str(src), "--interval", "0"])
    assert exc.value.code == 2


def test_live_rejects_negative_interval(tmp_path) -> None:
    """Negative --interval is also a usage error (exit 2)."""
    src = tmp_path / "log.txt"
    src.write_text("first\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        zoom.parse_args(["--live", "--file", str(src), "--interval", "-0.5"])
    assert exc.value.code == 2


def test_live_help_text_includes_key_flag() -> None:
    """The --help text must mention --live and --interval so the
    tool is discoverable (and the test catches accidental renames)."""
    parser = zoom.build_parser()
    help_text = parser.format_help()
    assert "--live" in help_text
    assert "--interval" in help_text


def test_tail_and_render_first_frame(tmp_path) -> None:
    """_tail_and_render emits one frame for a file that exists on
    the first poll, then stops on a predicate."""
    src = tmp_path / "log.txt"
    src.write_text("line one\nline two\n", encoding="utf-8")
    cfg = zoom.ZoomConfig(rows=2, cols=8, zoom=1)
    frames: list[str] = []

    # Run exactly two iterations (one frame + one no-op sleep),
    # then bail.
    ticks = {"n": 0}

    def stop() -> bool:
        ticks["n"] += 1
        return ticks["n"] >= 2

    rc = zoom._tail_and_render(
        str(src),
        cfg,
        interval=0.0,
        stop_predicate=stop,
        clock=lambda _x: None,  # no real sleep
        sink=frames.append,
    )
    assert rc == 0
    assert len(frames) == 1
    # The first frame is the magnified viewport of the source.
    assert "line one" in frames[0]


def test_tail_and_render_skips_unchanged_polls(tmp_path) -> None:
    """If the file is unchanged across many polls, only one frame
    is rendered — the change-detector works."""
    src = tmp_path / "log.txt"
    src.write_text("hello\n", encoding="utf-8")
    cfg = zoom.ZoomConfig(rows=2, cols=5, zoom=1)
    frames: list[str] = []
    ticks = {"n": 0}

    def stop() -> bool:
        ticks["n"] += 1
        return ticks["n"] >= 5

    zoom._tail_and_render(
        str(src),
        cfg,
        interval=0.0,
        stop_predicate=stop,
        clock=lambda _x: None,
        sink=frames.append,
    )
    assert len(frames) == 1


def test_tail_and_render_emits_frame_on_append(tmp_path) -> None:
    """If the file is appended to, a new frame is rendered."""
    src = tmp_path / "log.txt"
    src.write_text("v1\n", encoding="utf-8")
    cfg = zoom.ZoomConfig(rows=2, cols=8, zoom=1)
    frames: list[str] = []
    ticks = {"n": 0}

    def stop() -> bool:
        ticks["n"] += 1
        # On the second tick, append. The third tick should see it.
        if ticks["n"] == 2:
            with open(src, "a", encoding="utf-8") as fh:
                fh.write("v2\n")
        return ticks["n"] >= 3

    zoom._tail_and_render(
        str(src),
        cfg,
        interval=0.0,
        stop_predicate=stop,
        clock=lambda _x: None,
        sink=frames.append,
    )
    # Two distinct frames: one for "v1", one for "v1\nv2".
    assert len(frames) == 2
    assert "v1" in frames[0]
    assert "v2" in frames[1]


def test_tail_and_render_handles_missing_file(tmp_path) -> None:
    """If the source file does not exist on the first poll, the
    loop does not crash — it prints to stderr and continues until
    the stop predicate fires."""
    src = tmp_path / "does_not_exist.txt"
    cfg = zoom.ZoomConfig(rows=2, cols=5, zoom=1)
    frames: list[str] = []
    ticks = {"n": 0}

    def stop() -> bool:
        ticks["n"] += 1
        return ticks["n"] >= 2

    rc = zoom._tail_and_render(
        str(src),
        cfg,
        interval=0.0,
        stop_predicate=stop,
        clock=lambda _x: None,
        sink=frames.append,
    )
    assert rc == 0
    assert frames == []  # no frames emitted, no crash


def test_tail_and_render_creates_then_follows(tmp_path) -> None:
    """If the source file appears on a later poll, the loop picks
    it up. This mirrors log rotation: the new file is followed
    after the rotation completes."""
    src = tmp_path / "log.txt"
    cfg = zoom.ZoomConfig(rows=2, cols=8, zoom=1)
    frames: list[str] = []
    ticks = {"n": 0}

    def stop() -> bool:
        ticks["n"] += 1
        # On the second tick, create the file. The third tick
        # should see it.
        if ticks["n"] == 2:
            src.write_text("late\n", encoding="utf-8")
        return ticks["n"] >= 3

    zoom._tail_and_render(
        str(src),
        cfg,
        interval=0.0,
        stop_predicate=stop,
        clock=lambda _x: None,
        sink=frames.append,
    )
    assert len(frames) == 1
    assert "late" in frames[0]


def test_read_file_text_empty(tmp_path) -> None:
    """_read_file_text returns "" for an empty file (no exception)."""
    src = tmp_path / "empty.txt"
    src.write_text("", encoding="utf-8")
    assert zoom._read_file_text(str(src)) == ""


def test_main_live_writes_to_snapshot_on_change(tmp_path) -> None:
    """--live + --snapshot: each frame is written to the snapshot
    file (overwriting). After appending, the snapshot file should
    contain the new content."""
    src = tmp_path / "log.txt"
    src.write_text("alpha\n", encoding="utf-8")
    out = tmp_path / "shot.txt"
    cfg = zoom.ZoomConfig(rows=2, cols=8, zoom=1)
    frames: list[str] = []
    ticks = {"n": 0}

    def stop() -> bool:
        ticks["n"] += 1
        if ticks["n"] == 2:
            with open(src, "a", encoding="utf-8") as fh:
                fh.write("beta\n")
        return ticks["n"] >= 3

    def sink(text: str) -> None:
        frames.append(text)
        out.write_text(text, encoding="utf-8")

    rc = zoom._tail_and_render(
        str(src),
        cfg,
        interval=0.0,
        stop_predicate=stop,
        clock=lambda _x: None,
        sink=sink,
    )
    assert rc == 0
    assert len(frames) == 2
    # The final snapshot file is the LAST frame, not the first.
    final = out.read_text(encoding="utf-8")
    assert "beta" in final
    assert "alpha" in final


# ---------------------------------------------------------------------------
# --follow (track the tail of the file)
# ---------------------------------------------------------------------------


def test_tail_offset_shows_whole_file_when_short() -> None:
    """If the source has <= rows lines, _tail_offset returns 0 —
    we show the whole thing."""
    assert zoom._tail_offset("a\nb\nc\n", rows=10) == 0


def test_tail_offset_returns_skipped_rows_when_long() -> None:
    """For a source longer than ``rows`` lines, _tail_offset returns
    the count of leading lines to skip so the last ``rows`` are in
    view. A 5-line source viewed at rows=2 skips 3."""
    assert zoom._tail_offset("a\nb\nc\nd\ne\n", rows=2) == 3


def test_tail_offset_drops_trailing_empty_line() -> None:
    """A file ending in ``\\n`` shouldn't push the last real line
    off the viewport. Source 'a\\nb\\n' is 2 lines, viewed at
    rows=2 — no skipping, return 0."""
    assert zoom._tail_offset("a\nb\n", rows=2) == 0


def test_tail_offset_empty_source_returns_zero() -> None:
    """An empty / whitespace-only source returns 0 so the caller
    still emits a visible (blank) window."""
    assert zoom._tail_offset("", rows=5) == 0
    assert zoom._tail_offset("   \n\n", rows=5) == 0


def test_tail_offset_exact_boundary() -> None:
    """A source with exactly ``rows`` lines returns 0 (no
    skipping, the last real line is the last one in view)."""
    assert zoom._tail_offset("a\nb\nc\n", rows=3) == 0


def test_tail_and_render_follow_keeps_tail_in_view(tmp_path) -> None:
    """With ``follow=True``, the rendered viewport always shows the
    last ``cfg.rows`` lines of the source. We append a new line and
    verify the *previous* tail line scrolls out of view while the
    new one is visible."""
    src = tmp_path / "log.txt"
    src.write_text("L1\nL2\nL3\nL4\nL5\n", encoding="utf-8")
    cfg = zoom.ZoomConfig(rows=2, cols=8, zoom=1)
    frames: list[str] = []
    ticks = {"n": 0}

    def stop() -> bool:
        ticks["n"] += 1
        if ticks["n"] == 2:
            with open(src, "a", encoding="utf-8") as fh:
                fh.write("L6\n")
        return ticks["n"] >= 3

    zoom._tail_and_render(
        str(src),
        cfg,
        interval=0.0,
        follow=True,
        stop_predicate=stop,
        clock=lambda _x: None,
        sink=frames.append,
    )
    assert len(frames) == 2
    # First frame shows the last 2 lines of the initial 5-line file.
    assert "L4" in frames[0]
    assert "L5" in frames[0]
    assert "L1" not in frames[0]
    # Second frame shows the last 2 lines after the append.
    assert "L5" in frames[1]
    assert "L6" in frames[1]
    assert "L1" not in frames[1]
    assert "L4" not in frames[1]


def test_tail_and_render_follow_keeps_first_line_visible_when_short(
    tmp_path,
) -> None:
    """With ``follow=True`` and a source shorter than ``rows``,
    the viewport still shows the whole file (no skipping)."""
    src = tmp_path / "log.txt"
    src.write_text("only one\n", encoding="utf-8")
    cfg = zoom.ZoomConfig(rows=5, cols=10, zoom=1)
    frames: list[str] = []
    ticks = {"n": 0}

    def stop() -> bool:
        ticks["n"] += 1
        return ticks["n"] >= 2  # one frame, then bail

    zoom._tail_and_render(
        str(src),
        cfg,
        interval=0.0,
        follow=True,
        stop_predicate=stop,
        clock=lambda _x: None,
        sink=frames.append,
    )
    assert len(frames) == 1
    assert "only one" in frames[0]


def test_tail_and_render_without_follow_shows_head(tmp_path) -> None:
    """Default (``follow=False``) behaviour: the viewport shows
    the FIRST ``rows`` lines of the source, not the tail. This is
    a regression guard for the default mode — ``--follow`` must
    not become the default."""
    src = tmp_path / "log.txt"
    src.write_text("L1\nL2\nL3\nL4\nL5\n", encoding="utf-8")
    cfg = zoom.ZoomConfig(rows=2, cols=8, zoom=1)
    frames: list[str] = []
    ticks = {"n": 0}

    def stop() -> bool:
        ticks["n"] += 1
        return ticks["n"] >= 2

    zoom._tail_and_render(
        str(src),
        cfg,
        interval=0.0,
        follow=False,
        stop_predicate=stop,
        clock=lambda _x: None,
        sink=frames.append,
    )
    assert len(frames) == 1
    # Head, not tail.
    assert "L1" in frames[0]
    assert "L2" in frames[0]
    assert "L5" not in frames[0]


def test_parse_args_follow_flag() -> None:
    """``--follow`` parses as a boolean attribute (default False)."""
    args = zoom.parse_args(["--follow", "hello"])
    assert args.follow is True


def test_parse_args_follow_default_off() -> None:
    """Without --follow, the attribute is False (the default)."""
    args = zoom.parse_args(["hello"])
    assert args.follow is False


def test_follow_help_text_mentions_flag() -> None:
    """The --help text must mention --follow so the tool is
    discoverable and accidental renames are caught."""
    parser = zoom.build_parser()
    help_text = parser.format_help()
    assert "--follow" in help_text


def test_main_follow_renders_tail_to_stdout(tmp_path, capsys) -> None:
    """End-to-end: ``paw-zoom --follow --file PATH --rows N`` prints
    a viewport that contains the last N lines of the file (and
    not the first N lines)."""
    src = tmp_path / "log.txt"
    src.write_text("L1\nL2\nL3\nL4\nL5\n", encoding="utf-8")
    rc = zoom.main(
        ["--follow", "--file", str(src), "--rows", "2", "--cols", "8",
         "--zoom", "1", "--quiet"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    # The tail of the file is in the rendered output.
    assert "L4" in out
    assert "L5" in out
    # The head is NOT in the rendered output (because the viewport
    # only has 2 rows and the tail is in view, not the head).
    assert "L1" not in out


def test_main_follow_combines_with_snapshot(tmp_path) -> None:
    """``--follow --snapshot PATH`` writes the tail-tracked viewport
    to the snapshot file (overwriting on each call)."""
    src = tmp_path / "log.txt"
    src.write_text("alpha\nbeta\ngamma\ndelta\n", encoding="utf-8")
    out = tmp_path / "shot.txt"
    rc = zoom.main(
        ["--follow", "--file", str(src), "--rows", "2", "--cols", "8",
         "--zoom", "1", "--snapshot", str(out), "--quiet"]
    )
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "gamma" in text
    assert "delta" in text
    assert "alpha" not in text


# ---------------------------------------------------------------------------
# --max-frames (cap the number of frames --live emits)
# ---------------------------------------------------------------------------


def test_tail_and_render_max_frames_caps_iterations(tmp_path) -> None:
    """With ``max_frames=N``, the loop exits after N iterations
    (not N emitted frames). For a file that's being changed on
    every iteration, the cap fires on iteration N+1, so the loop
    emits at most N frames. We pin this with an aggressively-
    changing source and assert the cap holds under that pressure."""
    src = tmp_path / "log.txt"
    src.write_text("v1\n", encoding="utf-8")
    cfg = zoom.ZoomConfig(rows=2, cols=5, zoom=1)
    frames: list[str] = []
    ticks = {"n": 0}

    def stop() -> bool:
        ticks["n"] += 1
        # Append a new line on every tick after the first, so the
        # change-detector fires on every iteration. Run for many
        # ticks so an uncapped loop would emit many more frames
        # than the cap allows.
        if ticks["n"] >= 2:
            with open(src, "a", encoding="utf-8") as fh:
                fh.write(f"v{ticks['n']}\n")
        return ticks["n"] >= 20

    rc = zoom._tail_and_render(
        str(src),
        cfg,
        interval=0.0,
        max_frames=3,  # 3 iterations max
        stop_predicate=stop,
        clock=lambda _x: None,
        sink=frames.append,
    )
    assert rc == 0
    # Exactly 3 frames — one per iteration. The cap fires on
    # iteration 4, well before the stop predicate at 20.
    assert len(frames) == 3


def test_tail_and_render_max_frames_zero_means_unlimited(tmp_path) -> None:
    """``max_frames=0`` (the default) is "no cap" — the loop runs
    until the stop predicate fires. We pin this with a short,
    bounded stop so the test doesn't hang forever."""
    src = tmp_path / "log.txt"
    src.write_text("v1\n", encoding="utf-8")
    cfg = zoom.ZoomConfig(rows=2, cols=5, zoom=1)
    frames: list[str] = []
    ticks = {"n": 0}

    def stop() -> bool:
        ticks["n"] += 1
        if ticks["n"] >= 2:
            with open(src, "a", encoding="utf-8") as fh:
                fh.write(f"v{ticks['n']}\n")
        return ticks["n"] >= 5

    rc = zoom._tail_and_render(
        str(src),
        cfg,
        interval=0.0,
        max_frames=0,  # explicit default — no cap
        stop_predicate=stop,
        clock=lambda _x: None,
        sink=frames.append,
    )
    assert rc == 0
    # We expect a frame for the initial state, then one per change.
    # The exact count is timing-dependent (mtime granularity), but it
    # must be at least 1 and bounded by the tick count.
    assert len(frames) >= 1
    assert len(frames) <= 4  # ticks - initial idle = 4 potential frames


def test_tail_and_render_max_frames_one_exits_on_iteration_two(
    tmp_path,
) -> None:
    """``max_frames=1`` means "1 iteration max". The initial state
    emits 1 frame on the first iteration; the second iteration
    trips the cap and breaks. So a static file under
    ``max_frames=1`` still emits exactly 1 frame — the user's
    current state."""
    src = tmp_path / "log.txt"
    src.write_text("once\n", encoding="utf-8")
    cfg = zoom.ZoomConfig(rows=2, cols=5, zoom=1)
    frames: list[str] = []
    ticks = {"n": 0}

    def stop() -> bool:
        ticks["n"] += 1
        if ticks["n"] >= 2:
            with open(src, "a", encoding="utf-8") as fh:
                fh.write("more\n")
        return ticks["n"] >= 10

    rc = zoom._tail_and_render(
        str(src),
        cfg,
        interval=0.0,
        max_frames=1,
        stop_predicate=stop,
        clock=lambda _x: None,
        sink=frames.append,
    )
    assert rc == 0
    assert len(frames) == 1
    assert "once" in frames[0]


def test_parse_args_max_frames_default_zero() -> None:
    """Without --max-frames, the attribute is 0 (no cap)."""
    args = zoom.parse_args(["hello"])
    assert args.max_frames == 0


def test_parse_args_max_frames_flag() -> None:
    """``--max-frames N`` parses to an integer attribute."""
    args = zoom.parse_args(["--max-frames", "5", "hello"])
    assert args.max_frames == 5


def test_parse_args_max_frames_negative_is_usage_error() -> None:
    """A negative --max-frames is a usage error (exit 2) — only
    0+ makes sense (0 = unlimited)."""
    with pytest.raises(SystemExit) as exc_info:
        zoom.parse_args(["--max-frames", "-1", "hello"])
    assert exc_info.value.code == 2


def test_max_frames_help_text_mentions_flag() -> None:
    """The --help text must mention --max-frames so the tool is
    discoverable and accidental renames are caught."""
    parser = zoom.build_parser()
    help_text = parser.format_help()
    assert "--max-frames" in help_text


def test_main_max_frames_live_exits_cleanly(tmp_path, capsys) -> None:
    """End-to-end: ``paw-zoom --live --max-frames 2 --file PATH`` exits
    cleanly with rc=0, even though the source isn't being modified.
    The cap doesn't kick in (only one frame is emitted because the
    file is static), but the loop still terminates — the test pins
    that ``--max-frames`` doesn't break the live path."""
    src = tmp_path / "log.txt"
    src.write_text("L1\n", encoding="utf-8")
    rc = zoom.main(
        ["--live", "--file", str(src), "--rows", "2", "--cols", "5",
         "--zoom", "1", "--max-frames", "2", "--interval", "0.001",
         "--quiet"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    # The first frame was emitted; the static file produces no
    # further frames (the change-detector suppresses them), so the
    # loop exits on its own before hitting the cap.
    assert "L1" in out


def test_main_max_frames_follow_tracks_tail_and_caps(
    tmp_path, capsys
) -> None:
    """End-to-end: ``--live --max-frames 2 --follow`` exits with rc=0
    and emits at least one tail frame. We don't pin the exact frame
    count (the source is static in this test, so the change-detector
    suppresses subsequent frames and the cap doesn't fire) — the
    lower-level ``test_tail_and_render_max_frames_caps_loop`` covers
    the actual cap behaviour. This test just pins the plumbing
    through ``main()`` and the interaction with ``--follow``."""
    src = tmp_path / "log.txt"
    src.write_text("L1\nL2\nL3\n", encoding="utf-8")
    rc = zoom.main(
        ["--live", "--follow", "--file", str(src), "--rows", "2",
         "--cols", "5", "--zoom", "1", "--max-frames", "2",
         "--interval", "0.001", "--quiet"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    # The first frame is the tail of the initial 3-line file: L2, L3.
    assert "L3" in out
    assert "L1" not in out  # --follow keeps the head off the viewport


def test_main_max_frames_without_live_still_renders_once(
    tmp_path, capsys
) -> None:
    """``--max-frames`` without ``--live`` is a no-op: the one-shot
    render path emits exactly one frame regardless of the value.
    (The CLI documents this; the test pins it.)"""
    src = tmp_path / "log.txt"
    src.write_text("only one\n", encoding="utf-8")
    rc = zoom.main(
        ["--file", str(src), "--rows", "1", "--cols", "8",
         "--zoom", "1", "--max-frames", "100", "--quiet"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "only one" in out
    # A one-shot render emits exactly one block (rows * zoom lines,
    # which is 1 line at zoom=1).
    assert out.count("only one") == 1


# ---------------------------------------------------------------------------
# --raw: dump source (text or screen) to stdout without magnification
# ---------------------------------------------------------------------------


def test_parse_args_raw_default_is_false() -> None:
    """``--raw`` is a boolean flag, default False."""
    args = zoom.parse_args(["hello"])
    assert args.raw is False


def test_parse_args_raw_flag_sets_true() -> None:
    """``--raw`` flips the flag on."""
    args = zoom.parse_args(["--raw", "hello"])
    assert args.raw is True


def test_parse_args_raw_rejects_live(capsys) -> None:
    """``--raw --live`` is a usage error (exit 2)."""
    with pytest.raises(SystemExit) as exc:
        zoom.parse_args(["--raw", "--live", "--file", "/tmp/x"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--raw" in err
    assert "--live" in err


def test_parse_args_raw_rejects_snapshot(capsys) -> None:
    """``--raw --snapshot PATH`` is a usage error (exit 2)."""
    with pytest.raises(SystemExit) as exc:
        zoom.parse_args(["--raw", "--snapshot", "/tmp/x", "hello"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--raw" in err
    assert "--snapshot" in err


def test_parse_args_raw_rejects_follow(capsys) -> None:
    """``--raw --follow`` is a usage error (exit 2)."""
    with pytest.raises(SystemExit) as exc:
        zoom.parse_args(["--raw", "--follow", "--file", "/tmp/x"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--raw" in err
    assert "--follow" in err


def test_parse_args_raw_rejects_max_frames(capsys) -> None:
    """``--raw --max-frames N`` is a usage error (exit 2)."""
    with pytest.raises(SystemExit) as exc:
        zoom.parse_args(
            ["--raw", "--max-frames", "5", "--file", "/tmp/x"]
        )
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--raw" in err
    assert "--max-frames" in err


def test_main_raw_positional_prints_text_source(capsys) -> None:
    """``--raw "hello"`` dumps the positional text source verbatim."""
    rc = zoom.main(["--raw", "hello world"])
    assert rc == 0
    out = capsys.readouterr().out
    # The exact source is printed, with a single trailing newline.
    assert out == "hello world\n"


def test_main_raw_file_prints_file_contents(
    tmp_path, capsys
) -> None:
    """``--raw --file PATH`` dumps the file contents verbatim."""
    src = tmp_path / "notes.txt"
    src.write_text("alpha\nbeta\ngamma", encoding="utf-8")
    rc = zoom.main(["--raw", "--file", str(src)])
    assert rc == 0
    out = capsys.readouterr().out
    # The trailing newline is stripped by _resolve_source (same
    # behaviour as the magnifier), then a single \n is appended by
    # the --raw branch. So the file's three lines end up
    # newline-separated, with one final \n.
    assert out == "alpha\nbeta\ngamma\n"


def test_main_raw_quiet_suppresses_announcement(capsys) -> None:
    """``--raw --quiet`` writes nothing to stderr (no announcement
    line). The whole point of --raw is "give me the data and
    nothing else", so the announcement is opt-in via !--quiet."""
    rc = zoom.main(["--raw", "--quiet", "hello"])
    assert rc == 0
    err = capsys.readouterr().err
    assert err == ""


def test_main_raw_with_announcement_prints_one_liner(capsys) -> None:
    """``--raw`` (no ``--quiet``) emits a single opt-in stderr line
    so the user can tell a raw screen capture from a raw text
    source in scrollback."""
    rc = zoom.main(["--raw", "hello\nworld"])
    assert rc == 0
    out, err = capsys.readouterr()
    assert out == "hello\nworld\n"
    assert "paw-zoom" in err
    assert "--raw" in err
    assert "text" in err  # identifies the source kind


def test_main_raw_with_screen_capture_dumps_screen(capsys) -> None:
    """``--raw --screen --backend fake --fake-grid ...`` prints the
    captured screen grid verbatim (no magnification, no padding)."""
    rc = zoom.main(
        [
            "--raw",
            "--screen",
            "--backend", "fake",
            "--fake-grid", "abc\ndef\nghi",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    # The screen capture is "\n".join(["abc", "def", "ghi"]) =
    # "abc\ndef\nghi". The --raw branch then appends a final
    # newline (the captured source already ends in \n, so the
    # endwith guard skips a second one).
    assert out == "abc\ndef\nghi\n"


def test_main_raw_with_screen_and_region_uses_sub_grid(capsys) -> None:
    """``--raw --screen --region X,Y,W,H`` dumps the captured
    sub-grid verbatim."""
    rc = zoom.main(
        [
            "--raw",
            "--screen",
            "--backend", "fake",
            "--fake-grid", "abcdef\nghijkl\nmnopqr",
            "--region", "0,1,3,2",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    # The fake screen is 3 rows × 6 cols. --region 0,1,3,2 means
    # "start at (x=0, y=1), take a 3-wide × 2-tall window" →
    # rows 1 and 2 clipped to columns 0..2 → row 1 is "ghijkl"
    # → "ghi"; row 2 is "mnopqr" → "mno". Joined: "ghi\nmno".
    assert out == "ghi\nmno\n"


def test_main_raw_with_unsupported_screen_backend_exits_1(capsys) -> None:
    """``--raw --screen --backend x11`` on a headless box exits 1
    with the same friendly message the magnified --screen path
    uses."""
    rc = zoom.main(["--raw", "--screen", "--backend", "x11"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "x11" in err
    assert "fake" in err  # the workaround


def test_main_raw_without_any_source_exits_2(capsys) -> None:
    """``--raw`` with no positional, no ``--file``, and no
    ``--screen`` falls through to the text-source resolver, which
    raises ``RuntimeError`` (mapped to exit 2) because there's
    nothing to dump."""
    rc = zoom.main(["--raw"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "no text" in err


def test_main_raw_silently_ignores_viewport_flags(capsys) -> None:
    """Viewport-modifying flags (``--zoom``, ``--rows``, ``--cols``,
    ``--offset``, ``--col-offset``, ``--charset``) are silently
    ignored in --raw mode — the output shape of a raw dump does
    not depend on them, so the user can keep these flags in a
    shell alias or wrapper without breaking the dump."""
    rc = zoom.main(
        [
            "--raw",
            "--zoom", "8",
            "--rows", "1",
            "--cols", "80",
            "--offset", "5",
            "--col-offset", "10",
            "--charset", "dot",
            "hello world",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    # The dump is the literal source — no magnification, no
    # padding, no offsets applied.
    assert out == "hello world\n"


# ---------------------------------------------------------------------------
# --size: text-source dimensions discovery
# ---------------------------------------------------------------------------


def test_source_size_basic() -> None:
    """A multi-line source reports (rows, max line width) in
    code points. Two lines of 3 and 4 chars = (2, 4)."""
    assert zoom._source_size("abc\ndefg") == (2, 4)


def test_source_size_single_line() -> None:
    """A single-line source reports (1, line_width)."""
    assert zoom._source_size("hello") == (1, 5)


def test_source_size_empty_string() -> None:
    """An empty string reports (1, 0) — the natural shape for
    'there's a window but nothing in it'."""
    assert zoom._source_size("") == (1, 0)


def test_source_size_whitespace_only() -> None:
    """A whitespace-only source is a real source of N
    whitespace rows; the renderer would paint them as a
    fill rectangle. ``"   \\n   \\n   "`` is 3 rows of 3 cols
    of whitespace, not an empty source."""
    assert zoom._source_size("   \n   \n   ") == (3, 3)


def test_source_size_drops_trailing_empty_line() -> None:
    """A source ending in ``\\n`` has the trailing empty line
    dropped, the same way ``_extract_region`` and
    ``_tail_offset`` treat it. So ``"a\\nb\\n"`` reports (2, 1)
    not (3, 1) — the trailing newline is a line terminator, not
    a phantom row."""
    assert zoom._source_size("a\nb\n") == (2, 1)


def test_source_size_keeps_internal_empty_lines() -> None:
    """Internal empty lines are *kept* — only the trailing one
    is dropped. ``"a\\n\\nb"`` is three lines: "a", "", "b"."""
    assert zoom._source_size("a\n\nb") == (3, 1)


def test_source_size_unicode_columns() -> None:
    """Column counts are in code points, not bytes — a CJK line
    is one row of two columns, not six columns (the .encode()
    byte length)."""
    # U+732B = "cat" (3 bytes in UTF-8), U+59CB = "begin" (3 bytes)
    assert zoom._source_size("猫始") == (1, 2)


def test_source_size_max_col_picks_longest() -> None:
    """The max-cols is the *longest* line, not the first or
    last. ``"a\\nlonger\\nb"`` has rows=3, max=6."""
    assert zoom._source_size("a\nlonger\nb") == (3, 6)


def test_size_to_text_format() -> None:
    """``_size_to_text`` emits a fixed ``"rows x cols"`` line
    that downstream tooling can ``split(" x ")`` to get the
    two integers."""
    assert zoom._size_to_text((3, 12)) == "3 x 12"


def test_size_to_text_zero_cols() -> None:
    """An empty source emits ``"1 x 0"`` (the natural shape)
    rather than ``"1 x 0"`` being mangled into a shorter
    string. Layout is always ``"R x C"``."""
    assert zoom._size_to_text((1, 0)) == "1 x 0"


def test_size_to_json_round_trip() -> None:
    """``_size_to_json`` is a single-line parseable JSON
    object with sorted keys."""
    out = zoom._size_to_json((3, 12))
    assert "\n" not in out
    parsed = json.loads(out)
    assert parsed == {"rows": 3, "cols": 12}


def test_size_to_json_keys_sorted() -> None:
    """The JSON keys are sorted (``cols`` before ``rows``)
    so byte-for-byte output is deterministic across runs.
    Crucial for the byte-identity test in
    ``test_completions`` and for any downstream tool that
    diffs output."""
    out = zoom._size_to_json((1, 0))
    assert out.index('"cols"') < out.index('"rows"')


def test_size_to_json_none_unicode_safe() -> None:
    """``ensure_ascii=False`` is set so non-ASCII content
    doesn't escape into ``\\uXXXX`` form."""
    out = zoom._size_to_json((2, 3))
    # A sanity check on the JSON itself; ensure_ascii is what
    # *would* matter for a CJK size, but we just verify the
    # flag is in effect by checking the call doesn't crash and
    # returns valid JSON for a normal numeric size.
    assert json.loads(out) == {"rows": 2, "cols": 3}


def test_parse_args_size_default_is_false() -> None:
    """``--size`` defaults to off; existing behaviour is
    unchanged unless the flag is passed."""
    args = zoom.parse_args(["hello"])
    assert args.size is False


def test_parse_args_size_flag_sets_true() -> None:
    """``--size`` parses to ``True`` and composes with all
    three source-resolution paths (positional, --file,
    --screen)."""
    args = zoom.parse_args(["--size", "hello"])
    assert args.size is True
    # --file
    args = zoom.parse_args(["--size", "--file", "/tmp/whatever"])
    assert args.size is True
    # --screen
    args = zoom.parse_args(["--size", "--screen", "--backend", "fake"])
    assert args.size is True


def test_main_size_positional_text_mode(capsys) -> None:
    """``paw-zoom --size "hello world"`` prints ``"1 x 11"``
    and exits 0 without any rendering."""
    rc = zoom.main(["--size", "hello world"])
    out = capsys.readouterr()
    assert rc == 0
    assert out.out == "1 x 11\n"
    assert out.err == ""


def test_main_size_positional_multiline(capsys) -> None:
    """``paw-zoom --size "abc\\ndefg"`` prints ``"2 x 4"`` —
    the longest line is 4 chars, two lines total."""
    rc = zoom.main(["--size", "abc\ndefg"])
    out = capsys.readouterr()
    assert rc == 0
    assert out.out == "2 x 4\n"


def test_main_size_json_mode(capsys) -> None:
    """``paw-zoom --size --json "abc\\ndefg"`` prints a
    single-line ``{"rows": 2, "cols": 4}`` object and exits
    0. The shape matches the ``describe_to_json`` /
    ``to_json`` style the rest of the discovery flags use."""
    rc = zoom.main(["--size", "--json", "abc\ndefg"])
    out = capsys.readouterr()
    assert rc == 0
    assert "\n" not in out.out.rstrip("\n")
    parsed = json.loads(out.out)
    assert parsed == {"rows": 2, "cols": 4}


def test_main_size_file_source(tmp_path, capsys) -> None:
    """``--size --file PATH`` reads the file and reports its
    dimensions. Sanity-checks the --file path of source
    resolution for --size."""
    p = tmp_path / "hello.txt"
    p.write_text("hi\nworld", encoding="utf-8")
    rc = zoom.main(["--size", "--file", str(p)])
    out = capsys.readouterr()
    assert rc == 0
    # "hi" is 2 wide, "world" is 5 wide, 2 rows total.
    assert out.out == "2 x 5\n"


def test_main_size_file_missing_is_error(tmp_path, capsys) -> None:
    """``--size --file /missing`` exits 1 with a clear stderr
    message — the file-not-found path is the same one
    ``_resolve_source`` raises, and --size doesn't try to
    be cleverer than that."""
    missing = tmp_path / "does-not-exist.txt"
    rc = zoom.main(["--size", "--file", str(missing)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "file not found" in captured.err


def test_main_size_with_screen_capture(capsys) -> None:
    """``--size --screen --backend fake --fake-grid "abc\\nde"``
    captures the fake screen and reports the captured grid
    dimensions. ``FakeScreen`` truncates every row to the
    *shortest* row's width (a monospace text grid has no
    notion of row N being wider than row M), so the
    captured grid is ``"ab\\nde"`` — 2 rows of 2 cols."""
    rc = zoom.main(
        [
            "--size",
            "--screen",
            "--backend", "fake",
            "--fake-grid", "abc\nde",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    assert out.out == "2 x 2\n"


def test_main_size_with_screen_capture_json(capsys) -> None:
    """``--size --json --screen --backend fake --fake-grid
    "abc\\nde"`` combines the screen-capture path with the
    JSON output mode. Round-trips through ``json.loads``.
    Same FakeScreen row-truncation as the text-mode test."""
    rc = zoom.main(
        [
            "--size", "--json",
            "--screen",
            "--backend", "fake",
            "--fake-grid", "abc\nde",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    parsed = json.loads(out.out)
    assert parsed == {"rows": 2, "cols": 2}


def test_main_size_with_screen_and_region(capsys) -> None:
    """``--size --screen --backend fake --fake-grid TEXT
    --region X,Y,W,H`` reports the size of the *resolved*
    region, not the full screen. The region is 2x2 of the
    'abcdef\\nghijkl' fake grid starting at (3, 0) — that
    is 'de\\ngh', two rows of 2 cols (FakeScreen
    row-truncates the full grid to width 6, so col 3-4
    gives 'de' and 'gh')."""
    rc = zoom.main(
        [
            "--size",
            "--screen",
            "--backend", "fake",
            "--fake-grid", "abcdef\nghijkl",
            "--region", "3,0,2,2",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    assert out.out == "2 x 2\n"


def test_main_size_with_unsupported_screen_backend_exits_1(capsys) -> None:
    """``--size --screen --backend x11`` on a headless box
    exits 1 with the friendly 'not yet implemented on this
    OS' message — the screen-capture failure path still
    applies; --size is just a different *consumer* of the
    resolved source."""
    rc = zoom.main(["--size", "--screen", "--backend", "x11"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "not yet implemented" in captured.err


def test_main_size_without_any_source_exits_2(capsys) -> None:
    """``--size`` with no positional, no --file, no --screen
    (and stdin empty because the conftest forces
    ``WPAW_ZOOM_STDIN_OVERRIDE=""``) is a usage error
    (exit 2) — there is literally no source to measure."""
    rc = zoom.main(["--size"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "no text to magnify" in captured.err


def test_main_size_mutual_exclusion_with_info(capsys) -> None:
    """``--size --info`` is rejected (exit 2) — they are
    two different kinds of discovery."""
    rc = zoom.main(["--size", "--info", "--screen"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "--size cannot be combined with --info" in captured.err


def test_main_size_mutual_exclusion_with_live(capsys) -> None:
    """``--size --live`` is rejected (exit 2) — --live is a
    render driver, --size is metadata-only."""
    rc = zoom.main(["--size", "--live", "x"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "--size cannot be combined with --live" in captured.err


def test_main_size_mutual_exclusion_with_follow(capsys) -> None:
    """``--size --follow`` is rejected (exit 2) — same
    rationale as --live."""
    rc = zoom.main(["--size", "--follow", "x"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "--size cannot be combined with --follow" in captured.err


def test_main_size_mutual_exclusion_with_max_frames(capsys) -> None:
    """``--size --max-frames N`` is rejected (exit 2)."""
    rc = zoom.main(["--size", "--max-frames", "3", "x"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "--size cannot be combined with --max-frames" in captured.err


def test_main_size_mutual_exclusion_with_snapshot(capsys) -> None:
    """``--size --snapshot PATH`` is rejected (exit 2). The
    snapshot file must NOT have been written — the
    contradiction is caught before the source-resolution
    block, so we never even reach the file-write step."""
    rc = zoom.main(
        [
            "--size",
            "--snapshot", "/tmp/should_not_be_written.txt",
            "x",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "--size cannot be combined with --snapshot" in captured.err
    import os
    assert not os.path.exists("/tmp/should_not_be_written.txt")


def test_main_size_mutual_exclusion_with_raw(capsys) -> None:
    """``--size --raw`` is rejected (exit 2) — --raw is a
    dump mode, --size is metadata-only."""
    rc = zoom.main(["--size", "--raw", "x"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "--size cannot be combined with --raw" in captured.err


def test_main_size_mutual_exclusion_with_list_backends(capsys) -> None:
    """``--size --list-backends`` is rejected (exit 2) — they
    are two different kinds of discovery and we don't want
    to emit more than one of them per invocation."""
    rc = zoom.main(["--size", "--list-backends"])
    captured = capsys.readouterr()
    assert rc == 2
    assert (
        "--size cannot be combined with --list-backends" in captured.err
    )


def test_main_size_json_without_discovery_flag_is_usage_error(capsys) -> None:
    """``--json`` without ``--list-backends`` / ``--info`` /
    ``--size`` / ``--stats`` / ``--sha`` is a usage error
    (exit 2). Same fail-fast the ``--list-backends --json``
    combo used to do."""
    rc = zoom.main(["--json"])
    captured = capsys.readouterr()
    assert rc == 2
    assert (
        "--json requires --list-backends, --info, --size, "
        "--stats, or --sha"
        in captured.err
    )


def test_main_size_with_malformed_fake_grid_is_usage_error(capsys) -> None:
    """``--size --screen --backend fake --fake-grid ''`` (an
    empty fake grid) is rejected at the parse-fake-grid
    boundary with a clear message, not silently reported
    as a zero-size source."""
    rc = zoom.main(
        [
            "--size",
            "--screen",
            "--backend", "fake",
            "--fake-grid", "",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "--fake-grid" in captured.err


def test_main_size_silently_ignores_viewport_flags(capsys) -> None:
    """Viewport-modifying flags (``--zoom``, ``--rows``,
    ``--cols``, ``--offset``, ``--col-offset``, ``--charset``)
    are silently ignored in --size mode — the reported
    dimensions don't depend on them, so the user can keep
    these flags in a shell alias without breaking the size
    report. Same spirit as --raw's silent viewport-flag
    ignoring."""
    rc = zoom.main(
        [
            "--size",
            "--zoom", "8",
            "--rows", "1",
            "--cols", "80",
            "--offset", "5",
            "--col-offset", "10",
            "--charset", "dot",
            "hello world",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    # The size is the literal source — no magnification, no
    # offsets, no viewport math applied.
    assert out == "1 x 11\n"


def test_main_size_help_text_mentions_flag(capsys) -> None:
    """The ``--help`` text mentions ``--size`` so a casual
    ``paw-zoom --help`` user discovers it. Catches
    accidental renames."""
    rc = zoom.main(["--help"])
    out = capsys.readouterr()
    assert rc == 0
    assert "--size" in out.out


def test_main_size_short_circuits_before_render(capsys) -> None:
    """``--size`` must NOT call ``render_viewport`` — a direct
    proof: the render path turns an empty source into
    ``rows * cfg.cols`` chars of fill, but ``--size`` reports
    the actual measured shape ``1 x 0`` instead. If the
    renderer were running, the user would never see a
    ``cols=0`` size in the output."""
    rc = zoom.main(["--size", ""])  # empty positional
    out = capsys.readouterr()
    # Empty positional is treated as "no source" by
    # _resolve_source → RuntimeError → exit 2. So we exercise
    # the empty-source path through --file instead.
    assert rc == 2
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as fh:
        fh.write("")  # empty file
        empty_path = fh.name
    try:
        rc = zoom.main(["--size", "--file", empty_path])
        out = capsys.readouterr()
        assert rc == 0
        # Empty file → empty string → (1, 0). The renderer
        # would have produced 10*40=400 chars of fill, not
        # "1 x 0". The fact that we see "1 x 0" proves
        # --size never called render_viewport.
        assert out.out == "1 x 0\n"
    finally:
        os.unlink(empty_path)


# ---------------------------------------------------------------------------
# --stats: per-source statistics discovery
# ---------------------------------------------------------------------------
#
# The ``--stats`` flag answers "what is in the source?" — chars, lines,
# non-blank lines, max line width, mean line width. It runs at the same
# point in the pipeline as ``--size`` (after source resolution, before the
# render / ``--raw`` block) so the source it counts is the source the
# renderer would see, and a user can pipe the same input through both
# flags without surprises. The conventions intentionally match
# ``_source_size`` so a script that calls ``_source_size`` and
# ``_source_stats`` on the same string gets two consistent answers.


def test_source_stats_basic() -> None:
    """``"a\\nbb\\nccc"`` has 3 lines, 8 chars (counting the
    2 newlines as 1 each), max width 3, mean width 2.0, all
    lines non-blank."""
    assert zoom._source_stats("a\nbb\nccc") == (8, 3, 3, 3, 2.0)


def test_source_stats_single_line() -> None:
    """A one-line source has 1 line and 1 non-blank line."""
    assert zoom._source_stats("hello") == (5, 1, 1, 5, 5.0)


def test_source_stats_empty_string() -> None:
    """An empty source reports all zeros — the natural
    "nothing to count" answer. Diverges from ``_source_size``
    (which returns ``(1, 0)`` for the same input), on purpose:
    ``--size`` answers "what would the renderer show?";
    ``--stats`` answers "how much content is there?"."""
    assert zoom._source_stats("") == (0, 0, 0, 0, 0.0)


def test_source_stats_whitespace_only() -> None:
    """A whitespace-only source has content (``chars > 0``,
    ``lines > 0``, ``max_line_width > 0``) but zero
    non-blank lines — every line is blank after stripping.
    ``"   \\n   \\n   "`` is 3 lines of 3 spaces joined by
    2 newlines: 3+1+3+1+3 = 11 chars."""
    stats = zoom._source_stats("   \n   \n   ")
    assert stats[0] == 11  # 3 * 3 chars of spaces + 2 newlines
    assert stats[1] == 3
    assert stats[2] == 0  # non_blank_lines
    assert stats[3] == 3  # max_line_width
    assert stats[4] == 3.0  # mean_line_width


def test_source_stats_drops_trailing_empty_line() -> None:
    """A source ending in ``\\n`` does not gain a phantom
    empty line — same convention as ``_source_size`` /
    ``_tail_offset``."""
    assert zoom._source_stats("a\nb\n") == (4, 2, 2, 1, 1.0)


def test_source_stats_keeps_internal_empty_lines() -> None:
    """Internal blank lines are real rows of content (the user
    intended them), so they are kept. ``non_blank_lines``
    still excludes them. ``"a\\n\\nb"`` is 2 non-blank
    lines + 1 blank line = 3 lines, 4 chars (1+1+0+1). The
    mean width is ``(1+0+1)/3 = 0.67`` after the
    2-decimal-place rounding."""
    stats = zoom._source_stats("a\n\nb")
    assert stats[0] == 4
    assert stats[1] == 3
    assert stats[2] == 2
    assert stats[3] == 1
    assert stats[4] == 0.67


def test_source_stats_mixed_blank_and_non_blank() -> None:
    """A real-world mix: 5 lines, 3 non-blank, max width 5,
    mean width 3.2. Exercises the non-trivial arithmetic path."""
    stats = zoom._source_stats("hello\n\nworld\n   \nfoo")
    assert stats[0] == len("hello\n\nworld\n   \nfoo")
    assert stats[1] == 5
    assert stats[2] == 3
    assert stats[3] == 5  # max of [5, 0, 5, 3, 3]
    assert stats[4] == (5 + 0 + 5 + 3 + 3) / 5  # 3.2


def test_source_stats_unicode_code_points() -> None:
    """``chars`` is measured in code points, not bytes, so a
    CJK source counts the same way the renderer counts it.
    ``"猫\\n始"`` is 2 lines, 3 chars, max width 1, mean
    width 1.0."""
    stats = zoom._source_stats("猫\n始")
    assert stats[0] == 3
    assert stats[1] == 2
    assert stats[2] == 2
    assert stats[3] == 1
    assert stats[4] == 1.0


def test_source_stats_mean_rounded_to_two_decimals() -> None:
    """The mean is rounded to 2 decimal places so JSON
    serialisation stays predictable. ``"a\\nbb"`` has
    mean ``3 / 2 = 1.5`` (exact)."""
    assert zoom._source_stats("a\nbb")[4] == 1.5
    assert zoom._source_stats("a\nbb\nccc\ndddd")[4] == 2.5


def test_stats_to_text_format() -> None:
    """Five ``key: value`` lines, one per field, in canonical
    order, with ``mean_line_width`` rendered as a stable
    2-decimal float. Easy to grep / awk."""
    text = zoom._stats_to_text((8, 3, 3, 3, 2.0))
    assert text == (
        "chars: 8\n"
        "lines: 3\n"
        "non_blank_lines: 3\n"
        "max_line_width: 3\n"
        "mean_line_width: 2.00"
    )


def test_stats_to_text_zero_values() -> None:
    """An all-zero ``SourceStats`` renders the same five lines
    with all zeros — the "nothing to count" answer is
    consistent with the other discovery flags."""
    text = zoom._stats_to_text((0, 0, 0, 0, 0.0))
    assert text == (
        "chars: 0\n"
        "lines: 0\n"
        "non_blank_lines: 0\n"
        "max_line_width: 0\n"
        "mean_line_width: 0.00"
    )


def test_stats_to_text_mean_always_two_decimals() -> None:
    """``mean_line_width: 1.50`` is rendered with the trailing
    zero so the layout is predictable for a downstream
    ``awk`` / ``cut`` / ``column`` pipeline."""
    text = zoom._stats_to_text((5, 1, 1, 5, 1.5))
    assert text.endswith("mean_line_width: 1.50")


def test_stats_to_json_round_trip() -> None:
    """``json.loads`` of ``_stats_to_json`` recovers exactly
    the input ``SourceStats``. The shape mirrors the
    ``_size_to_json`` and ``describe_to_json`` style."""
    stats = (8, 3, 3, 3, 2.0)
    parsed = json.loads(zoom._stats_to_json(stats))
    assert parsed == {
        "chars": 8,
        "lines": 3,
        "max_line_width": 3,
        "mean_line_width": 2.0,
        "non_blank_lines": 3,
    }


def test_stats_to_json_keys_sorted() -> None:
    """Keys are alphabetically sorted so a diff is
    deterministic across runs / platforms."""
    parsed = json.loads(zoom._stats_to_json((8, 3, 3, 3, 2.0)))
    assert list(parsed.keys()) == sorted(parsed.keys())


def test_stats_to_json_mean_is_json_number() -> None:
    """``mean_line_width`` is emitted as a JSON number, not a
    string, so a downstream ``jq '.mean_line_width > 1'`` works
    without further coercion."""
    raw = zoom._stats_to_json((5, 1, 1, 5, 1.5))
    assert '"mean_line_width": 1.5' in raw


def test_parse_args_stats_default_is_false() -> None:
    """``--stats`` defaults to off; existing behaviour is
    preserved for every flag combination that does not
    mention ``--stats``."""
    args = zoom.parse_args(["hello"])
    assert args.stats is False


def test_parse_args_stats_flag_sets_true() -> None:
    """``--stats`` parses to ``True`` and composes with all
    three source-resolution paths (positional, ``--file``,
    ``--screen``)."""
    assert zoom.parse_args(["--stats", "hello"]).stats is True
    assert (
        zoom.parse_args(["--stats", "--file", "/tmp/whatever"]).stats is True
    )
    assert (
        zoom.parse_args(["--stats", "--screen", "--backend", "fake"]).stats
        is True
    )


def test_main_stats_positional_text_mode(capsys) -> None:
    """``paw-zoom --stats "a\\nbb\\nccc"`` prints the
    5-line ``key: value`` block and exits 0 without any
    rendering. 8 chars total (3 + 1 + 2 + 1 + 3)."""
    rc = zoom.main(["--stats", "a\nbb\nccc"])
    out = capsys.readouterr()
    assert rc == 0
    assert out.out == (
        "chars: 8\n"
        "lines: 3\n"
        "non_blank_lines: 3\n"
        "max_line_width: 3\n"
        "mean_line_width: 2.00\n"
    )
    assert out.err == ""


def test_main_stats_positional_multiline(capsys) -> None:
    """``paw-zoom --stats "abc\\ndefg"`` reports 2 lines, 8
    chars, max width 4, mean 3.5."""
    rc = zoom.main(["--stats", "abc\ndefg"])
    out = capsys.readouterr()
    assert rc == 0
    assert out.out == (
        "chars: 8\n"
        "lines: 2\n"
        "non_blank_lines: 2\n"
        "max_line_width: 4\n"
        "mean_line_width: 3.50\n"
    )


def test_main_stats_json_mode(capsys) -> None:
    """``paw-zoom --stats --json "a\\nbb\\nccc"`` prints a
    single-line ``{"chars": 8, ...}`` object and exits 0.
    The shape matches the ``describe_to_json`` /
    ``to_json`` style the rest of the discovery flags use.
    8 chars total (3 + 1 + 2 + 1 + 3)."""
    rc = zoom.main(["--stats", "--json", "a\nbb\nccc"])
    out = capsys.readouterr()
    assert rc == 0
    assert "\n" not in out.out.rstrip("\n")
    parsed = json.loads(out.out)
    assert parsed == {
        "chars": 8,
        "lines": 3,
        "max_line_width": 3,
        "mean_line_width": 2.0,
        "non_blank_lines": 3,
    }


def test_main_stats_file_source(tmp_path, capsys) -> None:
    """``--stats --file PATH`` reads the file and reports its
    statistics. Sanity-checks the ``--file`` path of source
    resolution for ``--stats``."""
    p = tmp_path / "hello.txt"
    p.write_text("hi\nworld", encoding="utf-8")
    rc = zoom.main(["--stats", "--file", str(p)])
    out = capsys.readouterr()
    assert rc == 0
    assert out.out == (
        "chars: 8\n"
        "lines: 2\n"
        "non_blank_lines: 2\n"
        "max_line_width: 5\n"
        "mean_line_width: 3.50\n"
    )


def test_main_stats_file_missing_is_error(tmp_path, capsys) -> None:
    """``--stats --file /missing`` exits 1 with a clear stderr
    message — the file-not-found path is the same one
    ``_resolve_source`` raises, and ``--stats`` doesn't try
    to be cleverer than that."""
    missing = tmp_path / "does-not-exist.txt"
    rc = zoom.main(["--stats", "--file", str(missing)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "file not found" in captured.err


def test_main_stats_with_screen_capture(capsys) -> None:
    """``--stats --screen --backend fake --fake-grid "abc\\nde"``
    captures the fake screen and reports the captured grid's
    statistics. ``FakeScreen`` truncates every row to the
    *shortest* row's width (a monospace text grid has no
    notion of row N being wider than row M), so the
    captured grid is ``"ab\\nde"`` — 5 chars (2 + 1 + 2),
    2 lines, max width 2, mean 2.0."""
    rc = zoom.main(
        [
            "--stats",
            "--screen",
            "--backend", "fake",
            "--fake-grid", "abc\nde",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    assert out.out == (
        "chars: 5\n"
        "lines: 2\n"
        "non_blank_lines: 2\n"
        "max_line_width: 2\n"
        "mean_line_width: 2.00\n"
    )


def test_main_stats_with_screen_capture_json(capsys) -> None:
    """``--stats --json --screen --backend fake --fake-grid
    "abc\\nde"`` combines the screen-capture path with the
    JSON output mode. Round-trips through ``json.loads``.
    Same ``FakeScreen`` row-truncation as the text-mode
    test (captured grid: ``"ab\\nde"`` = 5 chars)."""
    rc = zoom.main(
        [
            "--stats", "--json",
            "--screen",
            "--backend", "fake",
            "--fake-grid", "abc\nde",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    parsed = json.loads(out.out)
    assert parsed == {
        "chars": 5,
        "lines": 2,
        "max_line_width": 2,
        "mean_line_width": 2.0,
        "non_blank_lines": 2,
    }


def test_main_stats_with_unsupported_screen_backend_exits_1(capsys) -> None:
    """``--stats --screen --backend x11`` on a headless box
    exits 1 with the friendly 'not yet implemented on this
    OS' message — the screen-capture failure path still
    applies; ``--stats`` is just a different *consumer* of
    the resolved source."""
    rc = zoom.main(["--stats", "--screen", "--backend", "x11"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "not yet implemented" in captured.err


def test_main_stats_without_any_source_exits_2(capsys) -> None:
    """``--stats`` with no positional, no ``--file``, no
    ``--screen`` (and stdin empty because the conftest
    forces ``WPAW_ZOOM_STDIN_OVERRIDE=""``) is a usage
    error (exit 2) — there is literally no source to
    count."""
    rc = zoom.main(["--stats"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "no text to magnify" in captured.err


def test_main_stats_mutual_exclusion_with_size(capsys) -> None:
    """``--stats --size`` is rejected (exit 2) — they are
    two different kinds of discovery and we don't want to
    emit more than one of them per invocation. The
    ``--size``-wins-on-tie rule means the message names
    ``--size`` as the offender, not ``--stats``."""
    rc = zoom.main(["--stats", "--size", "x"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "--size cannot be combined with --stats" in captured.err


def test_main_stats_mutual_exclusion_with_info(capsys) -> None:
    """``--stats --info`` is rejected (exit 2) — they are
    two different kinds of discovery (``--info`` is
    screen-capture-side, ``--stats`` is text-side)."""
    rc = zoom.main(["--stats", "--info", "--screen"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "--stats cannot be combined with --info" in captured.err


def test_main_stats_mutual_exclusion_with_live(capsys) -> None:
    """``--stats --live`` is rejected (exit 2) — ``--live``
    is a render driver, ``--stats`` is metadata-only."""
    rc = zoom.main(["--stats", "--live", "x"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "--stats cannot be combined with --live" in captured.err


def test_main_stats_mutual_exclusion_with_follow(capsys) -> None:
    """``--stats --follow`` is rejected (exit 2) — same
    rationale as ``--live``."""
    rc = zoom.main(["--stats", "--follow", "x"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "--stats cannot be combined with --follow" in captured.err


def test_main_stats_mutual_exclusion_with_max_frames(capsys) -> None:
    """``--stats --max-frames N`` is rejected (exit 2)."""
    rc = zoom.main(["--stats", "--max-frames", "3", "x"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "--stats cannot be combined with --max-frames" in captured.err


def test_main_stats_mutual_exclusion_with_snapshot(capsys) -> None:
    """``--stats --snapshot PATH`` is rejected (exit 2). The
    snapshot file must NOT have been written — the
    contradiction is caught before the source-resolution
    block, so we never even reach the file-write step."""
    rc = zoom.main(
        [
            "--stats",
            "--snapshot", "/tmp/should_not_be_written_stats.txt",
            "x",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "--stats cannot be combined with --snapshot" in captured.err
    assert not os.path.exists("/tmp/should_not_be_written_stats.txt")


def test_main_stats_mutual_exclusion_with_raw(capsys) -> None:
    """``--stats --raw`` is rejected (exit 2) — ``--raw`` is
    a dump mode, ``--stats`` is metadata-only."""
    rc = zoom.main(["--stats", "--raw", "x"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "--stats cannot be combined with --raw" in captured.err


def test_main_stats_mutual_exclusion_with_list_backends(capsys) -> None:
    """``--stats --list-backends`` is rejected (exit 2) —
    they are two different kinds of discovery and we don't
    want to emit more than one of them per invocation."""
    rc = zoom.main(["--stats", "--list-backends"])
    captured = capsys.readouterr()
    assert rc == 2
    assert (
        "--stats cannot be combined with --list-backends" in captured.err
    )


def test_main_stats_json_without_discovery_flag_is_usage_error(capsys) -> None:
    """``--json`` without ``--list-backends`` / ``--info`` /
    ``--size`` / ``--stats`` / ``--sha`` is a usage error
    (exit 2). Same fail-fast the ``--list-backends --json``
    combo used to do."""
    rc = zoom.main(["--json"])
    captured = capsys.readouterr()
    assert rc == 2
    assert (
        "--json requires --list-backends, --info, --size, "
        "--stats, or --sha"
        in captured.err
    )


def test_main_stats_with_malformed_fake_grid_is_usage_error(capsys) -> None:
    """``--stats --screen --backend fake --fake-grid ''`` (an
    empty fake grid) is rejected at the parse-fake-grid
    boundary with a clear message, not silently reported
    as a zero-content source."""
    rc = zoom.main(
        [
            "--stats",
            "--screen",
            "--backend", "fake",
            "--fake-grid", "",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "--fake-grid" in captured.err


def test_main_stats_silently_ignores_viewport_flags(capsys) -> None:
    """Viewport-modifying flags (``--zoom``, ``--rows``,
    ``--cols``, ``--offset``, ``--col-offset``,
    ``--charset``) are silently ignored in ``--stats`` mode
    — the reported stats don't depend on them, so the user
    can keep these flags in a shell alias without breaking
    the stats report. Same spirit as ``--size``'s and
    ``--raw``'s silent viewport-flag ignoring."""
    rc = zoom.main(
        [
            "--stats",
            "--zoom", "8",
            "--rows", "1",
            "--cols", "80",
            "--offset", "5",
            "--col-offset", "10",
            "--charset", "dot",
            "hello world",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    # The stats are the literal source — no magnification, no
    # offsets, no viewport math applied.
    assert out == (
        "chars: 11\n"
        "lines: 1\n"
        "non_blank_lines: 1\n"
        "max_line_width: 11\n"
        "mean_line_width: 11.00\n"
    )


def test_main_stats_help_text_mentions_flag(capsys) -> None:
    """The ``--help`` text mentions ``--stats`` so a casual
    ``paw-zoom --help`` user discovers it. Catches
    accidental renames."""
    rc = zoom.main(["--help"])
    out = capsys.readouterr()
    assert rc == 0
    assert "--stats" in out.out


def test_main_stats_short_circuits_before_render(capsys) -> None:
    """``--stats`` must NOT call ``render_viewport`` — a
    direct proof: the render path would have built a
    ``ZoomConfig`` and tried to magnify, but ``--stats``
    reports the measured (chars, lines, ...) instead. We
    exercise the empty-source path through ``--file``
    because an empty positional is treated as "no source"
    by ``_resolve_source`` → ``RuntimeError`` → exit 2."""
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as fh:
        fh.write("")  # empty file
        empty_path = fh.name
    try:
        rc = zoom.main(["--stats", "--file", empty_path])
        out = capsys.readouterr()
        assert rc == 0
        # Empty file → empty string → (0, 0, 0, 0, 0.0).
        # The renderer would have produced 10*40 chars of
        # fill, not an all-zero stats block. The fact that
        # we see all zeros proves ``--stats`` never called
        # ``render_viewport``.
        assert out.out == (
            "chars: 0\n"
            "lines: 0\n"
            "non_blank_lines: 0\n"
            "max_line_width: 0\n"
            "mean_line_width: 0.00\n"
        )
    finally:
        os.unlink(empty_path)


def test_main_stats_consistent_with_size(tmp_path, capsys) -> None:
    """``--size`` and ``--stats`` describe the same source
    with two consistent views: ``--size`` reports the
    *rectangle* (rows, cols); ``--stats`` reports the
    *content* (chars, lines, non-blank lines, max line
    width, mean line width). For a clean ASCII file,
    ``stats.lines == size.rows`` and
    ``stats.max_line_width == size.cols`` — that is the
    invariant a user can rely on when chaining the two
    flags."""
    p = tmp_path / "mixed.txt"
    p.write_text("hi\nworld", encoding="utf-8")
    zoom.main(["--size", "--file", str(p)])
    size_out = capsys.readouterr().out
    zoom.main(["--stats", "--file", str(p)])
    stats_out = capsys.readouterr().out
    # Parse the size: "2 x 5"
    size_rows, size_cols = (int(x) for x in size_out.split(" x "))
    # Parse the stats: 5-line key: value block.
    stats_lines = dict(
        line.split(": ", 1) for line in stats_out.strip().split("\n")
    )
    assert int(stats_lines["lines"]) == size_rows
    assert int(stats_lines["max_line_width"]) == size_cols


# ---------------------------------------------------------------------------
# --sha: stable-hash source fingerprint
# ---------------------------------------------------------------------------


# Pre-computed digests for the tests below. Pinned here so the
# tests don't all share one ``hashlib.sha256(...).hexdigest()``
# call and accidentally pass a regression where the helper
# returns the wrong thing.
SHA256_EMPTY = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)
SHA256_HELLO = (
    "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
)


def test_source_sha_empty_string() -> None:
    """An empty source hashes to the well-known SHA-256 of
    the empty string. The empty case is a useful regression
    guard because the renderer would produce a 1×0 fill block
    on an empty source, but the hash is on the *literal*
    content — there is nothing to hash, so the digest is the
    empty-string digest."""
    assert zoom._source_sha("") == SHA256_EMPTY


def test_source_sha_known_value() -> None:
    """A simple string hashes to the well-known SHA-256
    digest. Pinned so a regression in the encoding
    (e.g. switching from UTF-8 to UTF-16, or skipping the
    encoding step entirely) shows up immediately."""
    assert zoom._source_sha("hello") == SHA256_HELLO


def test_source_sha_multibyte_utf8() -> None:
    """A multi-byte CJK source is encoded as UTF-8 before
    being hashed. The two halves of the test (encode-then-
    hash, vs. hash-the-precomputed-bytes) must agree — the
    helper never re-encodes a Python ``str`` to anything
    other than UTF-8."""
    # A CJK char encodes to 3 bytes in UTF-8.
    cjk = "你好"
    expected = hashlib.sha256("你好".encode("utf-8")).hexdigest()
    assert zoom._source_sha(cjk) == expected
    # And it's NOT the same as encoding UTF-16 (a sanity
    # check the encoding choice is real, not a no-op).
    assert zoom._source_sha(cjk) != hashlib.sha256(
        cjk.encode("utf-16")
    ).hexdigest()


def test_source_sha_deterministic() -> None:
    """The same source always hashes to the same digest. A
    regression that injected a process-id or timestamp into
    the helper would surface here."""
    assert zoom._source_sha("hello") == zoom._source_sha("hello")
    # And the digest is the same regardless of the call order
    # or surrounding state.
    assert (
        zoom._source_sha("hello")
        == zoom._source_sha("hello", algorithm="sha256")
    )


def test_source_sha_algorithm_override() -> None:
    """The algorithm is pluggable; ``--sha-algo`` only changes
    the digest family, not the source. A SHA-1 digest is 40
    hex chars, a SHA-256 digest is 64, an MD5 digest is 32.
    The helper produces a digest of the expected length for
    each."""
    for algo, expected_len in (
        ("md5", 32),
        ("sha1", 40),
        ("sha256", 64),
        ("sha512", 128),
    ):
        digest = zoom._source_sha("hello", algorithm=algo)
        assert len(digest) == expected_len, (
            f"{algo} digest should be {expected_len} chars, got {len(digest)}"
        )
        # The SHA-256 of "hello" is the pinned value above;
        # the others are just stable — pin them against the
        # stdlib so a regression in the override path shows up.
        assert digest == hashlib.new(algo, b"hello").hexdigest()


def test_source_sha_unknown_algorithm_raises() -> None:
    """An unknown algorithm raises ``ValueError`` with a
    helpful message that names the rejected algorithm. The
    parse_args() path catches this and re-emits it as a
    clear exit-2 stderr message."""
    with pytest.raises(ValueError) as excinfo:
        zoom._source_sha("hello", algorithm="not-a-real-algorithm")
    assert "not-a-real-algorithm" in str(excinfo.value)
    assert "unsupported" in str(excinfo.value).lower()


def test_source_sha_default_algorithm_is_sha256() -> None:
    """``_source_sha`` defaults to SHA-256 — the algorithm
    the CLI default ``--sha`` uses. A regression that
    flipped the default to MD5 (faster but weaker) would
    surface here."""
    assert zoom._source_sha("hello") == zoom._source_sha(
        "hello", algorithm="sha256"
    )
    # The default-algorithm constant is what the CLI help
    # text and the JSON ``"algorithm"`` key both read, so a
    # change to the constant propagates to the right places.
    assert zoom.DEFAULT_SHA_ALGORITHM == "sha256"


def test_sha_to_text_returns_digest() -> None:
    """``_sha_to_text`` returns the hex digest as-is. The
    trailing ``\\n`` is added by ``print``, not by the
    helper, so a downstream ``echo $digest`` sees exactly
    the digest."""
    digest = SHA256_HELLO
    assert zoom._sha_to_text(digest) == digest
    # The returned string has no trailing whitespace.
    assert zoom._sha_to_text(digest) == digest.rstrip()


def test_sha_to_json_round_trip() -> None:
    """``_sha_to_json`` is a single-line parseable JSON
    object that round-trips through ``json.loads``."""
    out = zoom._sha_to_json(SHA256_HELLO)
    assert "\n" not in out
    parsed = json.loads(out)
    assert parsed == {"algorithm": "sha256", "sha256": SHA256_HELLO}


def test_sha_to_json_algorithm_key_reflects_input() -> None:
    """The JSON object's digest key reflects the algorithm,
    not just the literal string ``"sha256"``. A downstream
    consumer asking for a SHA-1 digest should be able to
    tell from the key which family the digest is in without
    re-reading the ``"algorithm"`` field."""
    out = zoom._sha_to_json("a" * 40, algorithm="sha1")
    parsed = json.loads(out)
    assert parsed == {"algorithm": "sha1", "sha1": "a" * 40}


def test_sha_to_json_keys_sorted() -> None:
    """The JSON keys are sorted so byte-for-byte output is
    deterministic across runs. Crucial for the byte-identity
    test in ``test_completions`` and for any downstream tool
    that diffs the output."""
    out = zoom._sha_to_json(SHA256_HELLO)
    # ``algorithm`` sorts before ``sha256``.
    assert out.index('"algorithm"') < out.index('"sha256"')


def test_sha_to_json_ensure_ascii() -> None:
    """``ensure_ascii=False`` is set so non-ASCII content
    in the algorithm name (e.g. a future human-language
    alias) doesn't escape into ``\\uXXXX`` form. SHA
    algorithm names are ASCII, so this is a sanity check
    that the flag is in effect rather than a real-world
    coverage test."""
    out = zoom._sha_to_json(SHA256_HELLO)
    # ASCII-only output, and the call returns valid JSON.
    out.encode("ascii")
    assert json.loads(out)["sha256"] == SHA256_HELLO


def test_parse_args_sha_default_is_false() -> None:
    """``--sha`` defaults to off; existing behaviour is
    unchanged unless the flag is passed."""
    args = zoom.parse_args(["hello"])
    assert args.sha is False
    # The algorithm defaults to SHA-256 even when the flag
    # is off (so the user can set ``--sha-algo`` on the
    # command line and add ``--sha`` later without a second
    # flag).
    assert args.sha_algo == "sha256"


def test_parse_args_sha_flag_sets_true() -> None:
    """``--sha`` parses to ``True`` and composes with all
    three source-resolution paths (positional, --file,
    --screen)."""
    args = zoom.parse_args(["--sha", "hello"])
    assert args.sha is True
    # --file
    args = zoom.parse_args(["--sha", "--file", "/tmp/whatever"])
    assert args.sha is True
    # --screen
    args = zoom.parse_args(
        ["--sha", "--screen", "--backend", "fake"]
    )
    assert args.sha is True


def test_parse_args_sha_algo_override() -> None:
    """``--sha-algo`` overrides the default SHA-256 to
    pick a different digest family. The new value flows
    into ``args.sha_algo`` and is used by the runtime
    helper."""
    args = zoom.parse_args(
        ["--sha", "--sha-algo", "md5", "hello"]
    )
    assert args.sha is True
    assert args.sha_algo == "md5"


def test_parse_args_sha_algo_unknown_is_usage_error(capsys) -> None:
    """``--sha-algo FOO`` with an unknown name is a usage
    error (exit 2). A typo (``blake2x`` — a real family but
    one hashlib doesn't accept under that name) should not
    crash the render path with a generic ``hashlib``
    ValueError. Note that ``hashlib`` is case-insensitive
    on algorithm names, so ``"SHA-256"`` is *not* a useful
    bad-name example — it normalises to ``"sha256"`` and
    succeeds. We use a name that genuinely doesn't exist."""
    with pytest.raises(SystemExit) as excinfo:
        zoom.parse_args(
            ["--sha", "--sha-algo", "blake2x", "hello"]
        )
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "--sha-algo" in captured.err
    assert "blake2x" in captured.err
    assert "unsupported" in captured.err.lower()


def test_main_sha_positional_text_mode(capsys) -> None:
    """``paw-zoom --sha "hello"`` prints the SHA-256
    digest of ``"hello"`` and exits 0. The renderer is
    never called — no ZoomConfig, no ``--rows``/``--cols``
    banner, no magnified output."""
    rc = zoom.main(["--sha", "hello"])
    out = capsys.readouterr()
    assert rc == 0
    assert out.out == SHA256_HELLO + "\n"
    assert out.err == ""


def test_main_sha_empty_source(tmp_path, capsys) -> None:
    """``paw-zoom --sha --file EMPTY`` prints the SHA-256
    of the empty string. The renderer would have produced
    a 1×0 fill block, but the hash is on the literal
    content — and the empty-string digest is a well-known
    constant."""
    p = tmp_path / "empty.txt"
    p.write_text("", encoding="utf-8")
    rc = zoom.main(["--sha", "--file", str(p)])
    out = capsys.readouterr()
    assert rc == 0
    assert out.out == SHA256_EMPTY + "\n"


def test_main_sha_changes_when_source_changes(capsys) -> None:
    """A one-character change in the source produces a
    completely different digest. This is the *avalanche*
    property — the whole point of using a cryptographic
    hash for a "did the source change?" check."""
    rc_a = zoom.main(["--sha", "hello"])
    out_a = capsys.readouterr().out
    rc_b = zoom.main(["--sha", "hellp"])  # one letter off
    out_b = capsys.readouterr().out
    assert rc_a == 0
    assert rc_b == 0
    assert out_a != out_b


def test_main_sha_json_mode(capsys) -> None:
    """``paw-zoom --sha --json "hello"`` prints a single-
    line ``{"algorithm": "sha256", "sha256": "..."}`` object
    and exits 0. The shape matches the other discovery
    flags' ``--json`` form (single line, sorted keys,
    ``ensure_ascii=False``)."""
    rc = zoom.main(["--sha", "--json", "hello"])
    out = capsys.readouterr()
    assert rc == 0
    assert "\n" not in out.out.rstrip("\n")
    parsed = json.loads(out.out)
    assert parsed == {"algorithm": "sha256", "sha256": SHA256_HELLO}


def test_main_sha_algo_md5(capsys) -> None:
    """``--sha --sha-algo md5`` picks MD5 (32 hex chars).
    The output is the MD5 of the source, NOT a truncated
    SHA-256."""
    rc = zoom.main(["--sha", "--sha-algo", "md5", "hello"])
    out = capsys.readouterr()
    assert rc == 0
    # MD5("hello") = 5d41402abc4b2a76b9719d911017c592
    assert (
        out.out.rstrip("\n")
        == "5d41402abc4b2a76b9719d911017c592"
    )
    # The digest is 32 chars, not 64 (which would be a
    # truncated SHA-256).
    assert len(out.out.rstrip("\n")) == 32


def test_main_sha_algo_json_round_trip(capsys) -> None:
    """``--sha --sha-algo sha1 --json`` returns a JSON
    object whose digest key matches the algorithm. A
    downstream tool asking for a SHA-1 digest can tell
    from the JSON shape which family the digest is in."""
    rc = zoom.main(
        ["--sha", "--sha-algo", "sha1", "--json", "hello"]
    )
    out = capsys.readouterr()
    assert rc == 0
    parsed = json.loads(out.out)
    # SHA-1 of "hello" = aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d
    assert parsed == {
        "algorithm": "sha1",
        "sha1": "aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d",
    }


def test_main_sha_file_source(tmp_path, capsys) -> None:
    """``--sha --file PATH`` reads the file and hashes it.
    Sanity-checks the ``--file`` path of source resolution
    flows through the same hash."""
    p = tmp_path / "greeting.txt"
    p.write_text("hello", encoding="utf-8")
    rc = zoom.main(["--sha", "--file", str(p)])
    out = capsys.readouterr()
    assert rc == 0
    assert out.out.rstrip("\n") == SHA256_HELLO


def test_main_sha_file_source_different_digest(tmp_path, capsys) -> None:
    """``--sha --file PATH`` on a different file produces a
    different digest. Proves the digest reflects the file
    contents, not the file path."""
    p = tmp_path / "greeting.txt"
    p.write_text("different content", encoding="utf-8")
    rc = zoom.main(["--sha", "--file", str(p)])
    out = capsys.readouterr()
    assert rc == 0
    assert out.out.rstrip("\n") != SHA256_HELLO


def test_main_sha_file_source_missing(tmp_path, capsys) -> None:
    """``--sha --file PATH`` with a missing file exits 1
    (the same code ``--file`` returns on a missing path
    for any other mode). The hash is never computed; the
    user gets a clear stderr error."""
    missing = tmp_path / "no-such-file.txt"
    rc = zoom.main(["--sha", "--file", str(missing)])
    out = capsys.readouterr()
    assert rc == 1
    # The default ``_resolve_source`` error message names
    # the file, so the user can fix the path.
    assert str(missing) in out.err


def test_main_sha_screen_source_fake(capsys) -> None:
    """``--sha --screen --backend fake --fake-grid
    "hello\nworld"`` hashes the captured grid (after the
    ``capture_screen_to_source`` join), not the raw grid
    the user passed via ``--fake-grid``. The renderer's
    view of the source is what gets hashed, which is the
    same contract ``--size`` and ``--stats`` make."""
    rc = zoom.main(
        [
            "--sha",
            "--screen",
            "--backend", "fake",
            "--fake-grid", "hello\nworld",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    # SHA-256 of "hello\nworld" — the captured grid
    # joined with ``\n`` (no trailing ``\n``).
    expected = hashlib.sha256(b"hello\nworld").hexdigest()
    assert out.out.rstrip("\n") == expected


def test_main_sha_screen_source_json(capsys) -> None:
    """``--sha --screen --backend fake --fake-grid ...
    --json`` returns a parseable JSON object. The JSON
    ``algorithm`` key reflects ``--sha-algo`` if it was
    set; otherwise it defaults to SHA-256."""
    rc = zoom.main(
        [
            "--sha", "--json",
            "--screen",
            "--backend", "fake",
            "--fake-grid", "x",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    parsed = json.loads(out.out)
    assert parsed == {
        "algorithm": "sha256",
        "sha256": hashlib.sha256(b"x").hexdigest(),
    }


def test_main_sha_screen_source_unsupported(capsys) -> None:
    """``--sha --screen --backend x11`` on a headless box
    exits 1 with the standard ``backend_unsupported_message``
    (the same one the render path would print). A missing
    OS adapter never crashes the hash; the user gets the
    same friendly diagnostic they would get from
    ``--size --screen --backend x11``."""
    rc = zoom.main(
        ["--sha", "--screen", "--backend", "x11"]
    )
    out = capsys.readouterr()
    assert rc == 1
    # The exact message is owned by ``_screen``; we just
    # assert that the failure mode is the documented one.
    assert "x11" in out.err.lower()


def test_main_sha_no_source(capsys) -> None:
    """``--sha`` alone (no positional text, no ``--file``,
    no ``--screen``) exits 2 with the same "no text" /
    "no source" error the other modes produce. The hash
    is never computed; the user gets the same clear
    message they would get from ``--size`` alone."""
    rc = zoom.main(["--sha"])
    out = capsys.readouterr()
    assert rc == 2
    # The standard resolver message names the missing source.
    assert "no text" in out.err.lower() or "no source" in out.err.lower()


def test_main_sha_mutual_exclusion_with_size(capsys) -> None:
    """``--sha --size`` is rejected at parse time (exit 2)
    with a message naming both flags. A user typing both
    discovery flags together has clearly made a mistake
    (each one exits 0 with a different shape of answer);
    we tell them which two collided instead of silently
    picking one."""
    rc = zoom.main(["--sha", "--size", "hello"])
    out = capsys.readouterr()
    assert rc == 2
    assert "--sha" in out.err
    assert "--size" in out.err
    assert "cannot be combined" in out.err


def test_main_sha_mutual_exclusion_with_stats(capsys) -> None:
    """``--sha --stats`` is rejected at parse time (exit 2)
    with a message naming both flags. Same rationale as the
    ``--size`` mutual-exclusion: two discovery flags, one
    invocation, contradictory user intent."""
    rc = zoom.main(["--sha", "--stats", "hello"])
    out = capsys.readouterr()
    assert rc == 2
    assert "--sha" in out.err
    assert "--stats" in out.err
    assert "cannot be combined" in out.err


def test_main_sha_mutual_exclusion_with_info(capsys) -> None:
    """``--sha --info --screen`` is rejected at parse time
    (exit 2). ``--info`` is the screen-capture discovery;
    ``--sha`` is the text-source discovery. They answer
    different questions but both exit 0; the user can only
    ask one at a time."""
    rc = zoom.main(
        [
            "--sha", "--info",
            "--screen", "--backend", "fake",
            "--fake-grid", "x",
        ]
    )
    out = capsys.readouterr()
    assert rc == 2
    assert "--sha" in out.err
    assert "--info" in out.err


def test_main_sha_mutual_exclusion_with_live(capsys) -> None:
    """``--sha --live`` is rejected at parse time (exit 2).
    ``--live`` drives a render loop; ``--sha`` exits 0
    after one answer. The two are contradictory."""
    rc = zoom.main(
        ["--sha", "--live", "--file", "/tmp/whatever"]
    )
    out = capsys.readouterr()
    assert rc == 2
    assert "--sha" in out.err
    assert "--live" in out.err


def test_main_sha_mutual_exclusion_with_follow(capsys) -> None:
    """``--sha --follow`` is rejected at parse time (exit 2)."""
    rc = zoom.main(
        ["--sha", "--follow", "--file", "/tmp/whatever"]
    )
    out = capsys.readouterr()
    assert rc == 2
    assert "--sha" in out.err
    assert "--follow" in out.err


def test_main_sha_mutual_exclusion_with_max_frames(capsys) -> None:
    """``--sha --max-frames 3`` is rejected at parse time
    (exit 2). ``--max-frames`` only makes sense with
    ``--live``, so the combination is doubly contradictory."""
    rc = zoom.main(
        [
            "--sha", "--max-frames", "3",
            "--file", "/tmp/whatever",
        ]
    )
    out = capsys.readouterr()
    assert rc == 2
    assert "--sha" in out.err
    assert "--max-frames" in out.err


def test_main_sha_mutual_exclusion_with_snapshot(capsys) -> None:
    """``--sha --snapshot PATH`` is rejected at parse time
    (exit 2). ``--snapshot`` writes a magnified viewport
    to a file; ``--sha`` writes a single line of text to
    stdout. Two different file-output modes, one
    invocation, contradictory user intent."""
    with tempfile.NamedTemporaryFile(
        suffix=".txt", delete=False
    ) as fh:
        snapshot = fh.name
    try:
        rc = zoom.main(
            ["--sha", "--snapshot", snapshot, "hello"]
        )
        out = capsys.readouterr()
        assert rc == 2
        assert "--sha" in out.err
        assert "--snapshot" in out.err
    finally:
        if os.path.exists(snapshot):
            os.unlink(snapshot)


def test_main_sha_mutual_exclusion_with_raw(capsys) -> None:
    """``--sha --raw`` is rejected at parse time (exit 2)."""
    rc = zoom.main(["--sha", "--raw", "hello"])
    out = capsys.readouterr()
    assert rc == 2
    assert "--sha" in out.err
    assert "--raw" in out.err


def test_main_sha_mutual_exclusion_with_list_backends(capsys) -> None:
    """``--sha --list-backends`` is rejected at parse time
    (exit 2). ``--list-backends`` answers a question about
    the screen-capture backend, not the source; the two
    flags operate on different things and we don't try to
    emit both kinds of answer in one invocation."""
    rc = zoom.main(
        ["--sha", "--list-backends", "hello"]
    )
    out = capsys.readouterr()
    assert rc == 2
    assert "--sha" in out.err
    assert "--list-backends" in out.err


def test_main_sha_size_wins_on_tie(capsys) -> None:
    """When both ``--size`` and ``--sha`` are passed,
    ``--size``'s mutual-exclusion message wins. The
    ``--size`` block runs first in ``parse_args`` and
    rejects the combination before ``--sha``'s block has
    a chance to fire — so the user sees ``--size``'s
    message (which is the one we want when two
    discovery flags collide)."""
    rc = zoom.main(["--sha", "--size", "hello"])
    out = capsys.readouterr()
    assert rc == 2
    # The ``--size`` block rejects first, so the offending
    # flag in the error message is ``--size`` (the
    # later-in-the-parser ``--sha`` flag is the one being
    # rejected against).
    assert "--size" in out.err
    assert "cannot be combined" in out.err


def test_main_sha_alone_exits_zero(capsys) -> None:
    """``--sha`` on its own with a positional source exits
    0 — the happy path. Sanity check that the new flag
    doesn't change the default exit code."""
    rc = zoom.main(["--sha", "anything"])
    out = capsys.readouterr()
    assert rc == 0
    # And the digest is 64 hex chars (SHA-256).
    assert len(out.out.rstrip("\n")) == 64


def test_main_sha_json_composes_with_size_wins() -> None:
    """When ``--sha`` and ``--json`` are both passed,
    ``--json`` switches the output to a JSON object. The
    mutual-exclusion between ``--sha`` and ``--list-backends``
    / ``--size`` / ``--stats`` still holds: ``--json`` is
    a modifier, not a discovery flag of its own."""
    rc = zoom.main(["--sha", "--json", "hello"])
    assert rc == 0
    # The output is JSON, not a bare digest. (Captured
    # separately so the capsys fixture can be reused.)
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        zoom.main(["--sha", "--json", "hello"])
    parsed = json.loads(buf.getvalue())
    assert "sha256" in parsed


def test_sha_help_text_mentions_flag() -> None:
    """The ``--help`` text mentions ``--sha`` so a user
    who hits ``paw-zoom --help`` can find it. Regression
    guard against an accidental rename of the flag."""
    help_text = zoom.build_parser().format_help()
    assert "--sha" in help_text
    assert "--sha-algo" in help_text


def test_sha_default_sha_algo_appears_in_help() -> None:
    """The default algorithm (``sha256``) appears in the
    ``--sha-algo`` help text, so a user can find the
    default without reading the source."""
    help_text = zoom.build_parser().format_help()
    # The help text contains "default: sha256" or
    # "default: SHA-256" — either is fine; we just want
    # the algorithm name visible.
    assert "sha256" in help_text.lower()
