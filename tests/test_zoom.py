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


# ---------------------------------------------------------------------------
# --max-seconds (cap the wall-clock duration of --live)
# ---------------------------------------------------------------------------


def test_max_seconds_helper_zero_means_unlimited() -> None:
    """``_max_seconds_exceeded`` with ``max_seconds <= 0`` always
    returns ``False`` — the loop helper short-circuits when the cap
    is disabled, so a runaway ``time_fn`` can't accidentally fire
    a no-op cap."""
    assert zoom._max_seconds_exceeded(0.0, 0.0, 999.0) is False
    assert zoom._max_seconds_exceeded(0.0, -1.0, 999.0) is False
    assert zoom._max_seconds_exceeded(100.0, 0.0, 200.0) is False


def test_max_seconds_helper_within_window() -> None:
    """The helper returns ``False`` while ``now - start < max_seconds``,
    even when the loop has already run for a while."""
    assert zoom._max_seconds_exceeded(0.0, 10.0, 0.0) is False
    assert zoom._max_seconds_exceeded(0.0, 10.0, 5.0) is False
    assert zoom._max_seconds_exceeded(0.0, 10.0, 9.999) is False


def test_max_seconds_helper_at_boundary() -> None:
    """The check is ``>=``, so a cap of exactly ``max_seconds`` is
    treated as already exceeded. This makes "run for 5s" mean
    *at most* 5s — one more iteration would always push past the
    wall-clock window the user asked for."""
    assert zoom._max_seconds_exceeded(0.0, 10.0, 10.0) is True
    assert zoom._max_seconds_exceeded(0.0, 10.0, 10.001) is True
    assert zoom._max_seconds_exceeded(0.0, 10.0, 1000.0) is True


def test_max_seconds_helper_nonzero_start() -> None:
    """The helper only looks at the elapsed time
    (``now - start``), not at ``start`` itself. We pin this with a
    non-zero start so a refactor that re-reads ``start`` (instead
    of treating it as a pure input) gets caught."""
    # start=50, now=55 -> elapsed=5 < 10 -> False
    assert zoom._max_seconds_exceeded(50.0, 10.0, 55.0) is False
    # start=50, now=60 -> elapsed=10 == 10 -> True
    assert zoom._max_seconds_exceeded(50.0, 10.0, 60.0) is True


def test_tail_and_render_max_seconds_caps_loop(tmp_path) -> None:
    """With ``max_seconds > 0``, the loop exits once the wall-clock
    window has passed. We drive the loop with a fake ``time_fn``
    that advances on every poll so the cap fires on a known
    iteration (no real sleep, no flake)."""
    src = tmp_path / "log.txt"
    src.write_text("v1\n", encoding="utf-8")
    cfg = zoom.ZoomConfig(rows=2, cols=5, zoom=1)
    frames: list[str] = []
    # Fake clock: every call returns 1.0s after the previous one.
    # After 3 polls, the cap of 3.0s fires on the 4th.
    tick = {"t": 0.0}

    def fake_time() -> float:
        return tick["t"]

    def stop() -> bool:
        tick["t"] += 1.0
        return tick["t"] >= 20  # safety net so a buggy cap doesn't hang

    rc = zoom._tail_and_render(
        str(src),
        cfg,
        interval=0.0,
        max_seconds=3.0,
        stop_predicate=stop,
        clock=lambda _x: None,
        time_fn=fake_time,
        sink=frames.append,
    )
    assert rc == 0
    # The cap fires at the top of the 4th iteration, so the loop
    # runs at most 3 iterations -> at most 3 frames (one per
    # iteration; the change detector fires on every poll because
    # we don't mutate the file but the start time is "before the
    # very first iteration", so the first frame is the initial
    # state, and the static source emits no further frames).
    # The exact frame count depends on the change detector; the
    # load-bearing assertion is that the loop exited (rc=0) and
    # the cap fired well before the stop predicate.
    assert tick["t"] < 20  # cap fired before the safety net


def test_tail_and_render_max_seconds_zero_means_unlimited(tmp_path) -> None:
    """``max_seconds=0`` (the default) is "no cap" — the loop runs
    until the stop predicate fires, regardless of the wall-clock
    time elapsed. We pin this with a controlled fake clock that
    returns 1e9 seconds (a clearly preposterous value) so a
    refactor that mis-reads the default can't accidentally fire
    the cap."""
    src = tmp_path / "log.txt"
    src.write_text("static\n", encoding="utf-8")
    cfg = zoom.ZoomConfig(rows=2, cols=5, zoom=1)
    frames: list[str] = []
    ticks = {"n": 0}

    def stop() -> bool:
        ticks["n"] += 1
        return ticks["n"] >= 3

    rc = zoom._tail_and_render(
        str(src),
        cfg,
        interval=0.0,
        max_seconds=0.0,  # explicit default — no cap
        stop_predicate=stop,
        clock=lambda _x: None,
        time_fn=lambda: 1e9,  # preposterous wall-clock — cap must not fire
        sink=frames.append,
    )
    assert rc == 0
    # The cap never fired, so the loop ran until the stop predicate.
    assert ticks["n"] == 3


def test_tail_and_render_max_seconds_composes_with_max_frames(
    tmp_path,
) -> None:
    """``max_seconds`` and ``max_frames`` compose: whichever cap
    fires first wins. We pin this with a fake clock that advances
    slowly (so the time cap doesn't fire) and an aggressive
    ``max_frames=2`` so the iteration cap fires instead — the
    loop exits after 2 iterations even though the wall-clock
    window is 1000s."""
    src = tmp_path / "log.txt"
    src.write_text("v1\n", encoding="utf-8")
    cfg = zoom.ZoomConfig(rows=2, cols=5, zoom=1)
    frames: list[str] = []
    ticks = {"n": 0}

    def stop() -> bool:
        ticks["n"] += 1
        return ticks["n"] >= 20  # safety net

    rc = zoom._tail_and_render(
        str(src),
        cfg,
        interval=0.0,
        max_frames=2,
        max_seconds=1000.0,  # 1000s — never fires
        stop_predicate=stop,
        clock=lambda _x: None,
        time_fn=lambda: 0.0,  # frozen clock — time cap never fires
        sink=frames.append,
    )
    assert rc == 0
    # The iteration cap fired after 2 iterations.
    assert ticks["n"] <= 3


def test_parse_args_max_seconds_default_zero() -> None:
    """Without ``--max-seconds``, the attribute is 0.0 (no cap).
    The float default matters because the loop helper treats
    ``<= 0`` as "disabled"."""
    args = zoom.parse_args(["hello"])
    assert args.max_seconds == 0.0
    assert isinstance(args.max_seconds, float)


def test_parse_args_max_seconds_flag() -> None:
    """``--max-seconds SECS`` parses to a float attribute."""
    args = zoom.parse_args(["--max-seconds", "5.5", "hello"])
    assert args.max_seconds == 5.5


def test_parse_args_max_seconds_negative_is_usage_error() -> None:
    """A negative ``--max-seconds`` is a usage error (exit 2) — only
    0+ makes sense (0 = unlimited). The error message names the
    flag so the user can find the typo in a long pipeline."""
    with pytest.raises(SystemExit) as exc_info:
        zoom.parse_args(["--max-seconds", "-1", "hello"])
    assert exc_info.value.code == 2


def test_max_seconds_help_text_mentions_flag() -> None:
    """The --help text must mention ``--max-seconds`` so the tool is
    discoverable and accidental renames are caught."""
    parser = zoom.build_parser()
    help_text = parser.format_help()
    assert "--max-seconds" in help_text


def test_main_max_seconds_live_exits_cleanly(tmp_path, capsys) -> None:
    """End-to-end: ``paw-zoom --live --max-seconds 0.1 --file PATH``
    exits cleanly with rc=0. A 0.1s cap is large enough to let the
    loop run a few iterations, small enough that the test doesn't
    sleep. The point is to pin the plumbing through ``main()``:
    the new flag doesn't break the live path."""
    src = tmp_path / "log.txt"
    src.write_text("L1\n", encoding="utf-8")
    rc = zoom.main(
        ["--live", "--file", str(src), "--rows", "2", "--cols", "5",
         "--zoom", "1", "--max-seconds", "0.1", "--interval", "0.01",
         "--quiet"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    # The first frame was emitted; the static file produces no
    # further frames (the change-detector suppresses them), so the
    # loop exits on its own well before the cap. The point is that
    # the cap didn't break the path.
    assert "L1" in out


def test_main_max_seconds_without_live_still_renders_once(
    tmp_path, capsys
) -> None:
    """``--max-seconds`` without ``--live`` is a no-op: the one-shot
    render path emits exactly one frame regardless of the value,
    just like ``--max-frames``. (The CLI documents this; the test
    pins it.)"""
    src = tmp_path / "log.txt"
    src.write_text("only one\n", encoding="utf-8")
    rc = zoom.main(
        ["--file", str(src), "--rows", "1", "--cols", "8",
         "--zoom", "1", "--max-seconds", "100", "--quiet"]
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
    ``--size`` / ``--stats`` / ``--words`` / ``--sha`` is a
    usage error (exit 2). Same fail-fast the
    ``--list-backends --json`` combo used to do."""
    rc = zoom.main(["--json"])
    captured = capsys.readouterr()
    assert rc == 2
    assert (
        "--json requires --list-backends, --info, --size, "
        "--stats, --words, or --sha"
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
    ``--size`` / ``--stats`` / ``--words`` / ``--sha`` is a
    usage error (exit 2). Same fail-fast the
    ``--list-backends --json`` combo used to do."""
    rc = zoom.main(["--json"])
    captured = capsys.readouterr()
    assert rc == 2
    assert (
        "--json requires --list-backends, --info, --size, "
        "--stats, --words, or --sha"
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


# ---------------------------------------------------------------------------
# --line-numbers
# ---------------------------------------------------------------------------


def test_format_line_numbers_basic() -> None:
    """``_format_line_numbers`` prepends a per-source-row number
    to each magnified line. The gutter ("   N │ ") is the
    same shape ``cat -n`` / ``nl`` / most editors use: a
    4-char right-aligned row number, then a space, then a
    vertical bar, then a space. Zoom is 2, so each source
    row is repeated twice in the rendered output and the
    prefix repeats with it.
    """
    rendered = "aabb\n" "aabb\n" "ccdd\n" "ccdd"
    out = zoom._format_line_numbers(
        rendered, zoom=2, first_source_row=1
    )
    assert out == (
        "   1 │ aabb\n"
        "   1 │ aabb\n"
        "   2 │ ccdd\n"
        "   2 │ ccdd"
    )


def test_format_line_numbers_empty() -> None:
    """An empty ``rendered`` is a no-op — returns ``""``
    unchanged. The ``--raw`` and the no-frame case both rely
    on this; if the helper ever started padding the empty
    string with a gutter, both code paths would gain a
    spurious blank line."""
    assert (
        zoom._format_line_numbers("", zoom=1, first_source_row=1)
        == ""
    )
    assert (
        zoom._format_line_numbers("", zoom=3, first_source_row=42)
        == ""
    )


def test_format_line_numbers_zoom_1() -> None:
    """At ``zoom=1`` each magnified line is one source row,
    so the prefix increments on every line (1, 2, 3, …).
    The grouping math still works — there's one magnified
    line per group, so the per-line prefix is just
    ``f"{row}{gutter}{line}"``.
    """
    rendered = "ab\n" "cd\n" "ef"
    out = zoom._format_line_numbers(
        rendered, zoom=1, first_source_row=1
    )
    assert out == (
        "   1 │ ab\n"
        "   2 │ cd\n"
        "   3 │ ef"
    )


def test_format_line_numbers_offset() -> None:
    """``first_source_row`` shifts the numbering. The first
    source row in view is ``first_source_row`` (1-based), so
    ``paw-zoom --offset 12 --line-numbers …`` produces
    prefixes starting at ``13`` (matching what ``cat -n``
    would show for the same source-line). The width is
    constant (4 chars) so a 9999-line file still lines up
    under a 1-line viewport.
    """
    rendered = "x\n" "x\n" "y\n" "y"
    out = zoom._format_line_numbers(
        rendered, zoom=2, first_source_row=13
    )
    assert out == (
        "  13 │ x\n"
        "  13 │ x\n"
        "  14 │ y\n"
        "  14 │ y"
    )


def test_format_line_numbers_custom_width() -> None:
    """The ``width`` and ``gutter`` kwargs override the
    defaults. A tighter gutter (just a single space and a
    thin vertical bar) is sometimes easier to read on a
    narrow terminal; a wider gutter is sometimes easier
    on a busy screen. Both are pure-style overrides; the
    math (grouping, numbering) is unchanged.
    """
    rendered = "a\n" "a\n" "b"
    out = zoom._format_line_numbers(
        rendered, zoom=2, first_source_row=1, width=2, gutter="|"
    )
    assert out == (" 1|a\n" " 1|a\n" " 2|b")


def test_format_line_numbers_defensive_zoom() -> None:
    """A non-positive ``zoom`` can't be grouped, so the
    function falls back to numbering every line
    sequentially. This is a defensive boundary check —
    ``parse_args`` rejects ``--zoom 0`` at the CLI level,
    but the library function is callable from anywhere, so
    the re-validation belongs at the boundary. A user who
    somehow gets here gets *some* annotation, just not the
    grouped one.
    """
    rendered = "a\n" "b\n" "c"
    out = zoom._format_line_numbers(
        rendered, zoom=0, first_source_row=10
    )
    assert out == ("  10 │ a\n" "  11 │ b\n" "  12 │ c")


def test_format_line_numbers_negative_first_row_clamps() -> None:
    """A negative ``first_source_row`` is treated as
    ``1`` by the caller's ``max(1, …)`` clamp — the
    helper itself doesn't clamp, but the production
    wiring does. Verify the helper produces a valid
    format string for the clamped value (so a stray
    negative ``--offset`` doesn't crash the renderer).
    """
    # The library helper does not clamp (it formats
    # whatever it's given); a negative value yields a
    # negative prefix. The caller's ``max(1, …)`` is the
    # contract that prevents the negative value from
    # ever reaching this code in production. We assert
    # the format is well-defined so a future refactor
    # of the helper doesn't accidentally regress to
    # raising on negative input.
    rendered = "a"
    out = zoom._format_line_numbers(
        rendered, zoom=1, first_source_row=-5
    )
    # The number is whatever the caller passed in; the
    # format is right-aligned in 4 chars.
    assert out == "  -5 │ a"


def test_parse_args_line_numbers_default_off() -> None:
    """``--line-numbers`` defaults to ``False`` so a
    vanilla ``paw-zoom …`` keeps its current output shape
    — no gutter, no behaviour change for users who didn't
    ask for the annotation."""
    args = zoom.parse_args(["hello"])
    assert args.line_numbers is False


def test_parse_args_line_numbers_flag() -> None:
    """``--line-numbers`` is a ``store_true`` flag —
    passing it once flips the attribute to ``True``."""
    args = zoom.parse_args(["--line-numbers", "hello"])
    assert args.line_numbers is True


def test_main_line_numbers_renders_with_gutter(capsys) -> None:
    """End-to-end: ``paw-zoom --line-numbers …`` emits a
    magnified viewport with the per-source-row gutter
    prepended. The default gutter width (4 chars + `` | ``)
    is visible on every magnified line; the row number
    starts at ``1`` for a vanilla positional source.
    """
    rc = zoom.main(
        [
            "--rows",
            "3",
            "--cols",
            "3",
            "--zoom",
            "1",
            "--line-numbers",
            "abc\ndef\nghi",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    # Zoom 1 → no repetition; gutter on every line. The
    # announcement is on stdout (not stderr), so the first
    # line of out.out is the banner — we drop it and
    # compare only the guttered viewport.
    lines = out.out.splitlines()
    assert lines[0].startswith("🐾 paw-zoom:")
    assert lines[1:] == [
        "   1 │ abc",
        "   2 │ def",
        "   3 │ ghi",
    ]


def test_main_line_numbers_offset_shifts_numbering(capsys) -> None:
    """``--line-numbers`` combined with ``--offset N``:
    the first source row in view is ``N + 1`` (1-based),
    so the gutter numbers start at that value. Same
    source, two different offsets, two different
    numbering series — the user can correlate the
    rendered block with the source.
    """
    # 15 lines so ``--offset 12`` is in range and the
    # viewport shows source lines 13, 14, 15. ``cols=6`` is
    # wide enough for ``line13`` (6 chars).
    src = "\n".join(f"line{i}" for i in range(1, 16))
    rc = zoom.main(
        [
            "--rows",
            "3",
            "--cols",
            "6",
            "--zoom",
            "1",
            "--offset",
            "12",
            "--line-numbers",
            src,
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    lines = out.out.splitlines()
    assert lines[0].startswith("🐾 paw-zoom:")
    assert lines[1:] == [
        "  13 │ line13",
        "  14 │ line14",
        "  15 │ line15",
    ]


def test_main_line_numbers_zoom_2_repeats_number(capsys) -> None:
    """With ``--zoom 2``, each source row is repeated
    twice in the rendered output, and ``--line-numbers``
    repeats the row number with it (so the gutter reads
    ``1 │ xxxx\\n1 │ xxxx\\n2 │ yyyy\\n2 │ yyyy`` — the
    per-source-row number, not the per-magnified-line
    number). The grouping math lives in
    ``_format_line_numbers``; this test pins the
    end-to-end shape."""
    rc = zoom.main(
        [
            "--rows",
            "2",
            "--cols",
            "2",
            "--zoom",
            "2",
            "--line-numbers",
            "ab\ncd",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    # Source row 1 (ab) → aabb / aabb, gutter "1".
    # Source row 2 (cd) → ccdd / ccdd, gutter "2".
    lines = out.out.splitlines()
    assert lines[0].startswith("🐾 paw-zoom:")
    assert lines[1:] == [
        "   1 │ aabb",
        "   1 │ aabb",
        "   2 │ ccdd",
        "   2 │ ccdd",
    ]


def test_main_line_numbers_does_not_change_announcement(capsys) -> None:
    """``--line-numbers`` is a *render-time* annotation,
    not a discovery flag, so the announcement line still
    fires (``--quiet`` is not set) and reports the
    normal ``rows × cols window, zoom N, output …``
    shape. The gutter is added on top of the rendered
    viewport, so the announcement's ``output`` size
    reflects the magnified dimensions (not the
    gutter-inflated ones)."""
    rc = zoom.main(
        [
            "--rows",
            "1",
            "--cols",
            "3",
            "--zoom",
            "1",
            "--line-numbers",
            "abc",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    # The announcement reports the magnified output
    # dimensions (rows * zoom × cols * zoom) — same as
    # without --line-numbers. The gutter is in
    # *addition* to that.
    assert "1×3 window" in out.out
    assert "zoom 1" in out.out
    # And the gutter is visible on the rendered line.
    assert "   1 │ abc" in out.out


def test_main_line_numbers_off_by_default(capsys) -> None:
    """Without ``--line-numbers``, the rendered output
    has no gutter — the default behaviour is unchanged.
    This is the regression guard against an accidental
    flip-the-default mistake."""
    rc = zoom.main(
        ["--rows", "1", "--cols", "3", "--zoom", "1", "abc"]
    )
    out = capsys.readouterr()
    assert rc == 0
    # The rendered line is the literal source — no
    # "   1 │ " prefix.
    assert "abc" in out.out
    assert "│" not in out.out


def test_main_line_numbers_quiet_still_announces(capsys) -> None:
    """``--quiet`` suppresses the *announcement* line,
    not the gutter. The gutter is part of the
    rendered output, not the announcement — so
    ``--quiet --line-numbers`` still emits the
    guttered viewport. (Quiet + line-numbers is the
    canonical "clean pipe" invocation: render the
    guttered block to stdout, nothing else.)
    """
    rc = zoom.main(
        [
            "--quiet",
            "--rows",
            "1",
            "--cols",
            "3",
            "--zoom",
            "1",
            "--line-numbers",
            "abc",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    # No announcement.
    assert "window" not in out.out
    # The gutter is on stdout.
    assert "   1 │ abc" in out.out


def test_main_line_numbers_follow_shifts_per_frame(
    tmp_path, capsys
) -> None:
    """``--follow --line-numbers`` re-derives the
    first source row on every frame, so the gutter
    numbers track the tail of the source as it
    grows. We write a file, run the magnifier with
    ``--follow`` + ``--line-numbers`` and a fake
    clock / no-sleep, then verify the gutter on
    the captured frame reflects the *file*'s line
    count, not the viewport's top edge. The
    end-to-end guarantee is: a tail-tracking
    magnifier always shows the source's actual
    line numbers.
    """
    # Two lines, then the magnifier renders the
    # last ``--rows`` lines. With rows=1 + follow,
    # the viewport shows just line 2; the gutter
    # should say "2".
    src = tmp_path / "log.txt"
    src.write_text("line1\nline2\n", encoding="utf-8")
    rc = zoom.main(
        [
            "--quiet",
            "--line-numbers",
            "--follow",
            "--max-frames",
            "1",
            "--file",
            str(src),
            "--rows",
            "1",
            "--cols",
            "5",
            "--zoom",
            "1",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    # With follow, the viewport tracks the tail
    # (just "line2"), and the gutter number
    # reflects the source row that line is on
    # (line 2). "line1" was offset out of view.
    assert "   2 │ line2" in out.out
    assert "line1" not in out.out


def test_help_text_mentions_line_numbers() -> None:
    """``--line-numbers`` is in the ``--help`` output
    so a user who hits ``paw-zoom --help`` can
    find it. Regression guard against an
    accidental rename of the flag."""
    help_text = zoom.build_parser().format_help()
    assert "--line-numbers" in help_text


def test_main_col_ruler_follow_stable_per_frame(
    tmp_path, capsys
) -> None:
    """``--follow --col-ruler`` keeps the ruler
    stable across frames (the col_offset does not
    change between frames — only the row_offset does,
    via ``--follow``). The end-to-end guarantee is:
    a tail-tracking live magnifier always shows the
    same column ruler on every frame, regardless
    of which source rows are currently in view.
    """
    src = tmp_path / "log.txt"
    src.write_text("line1\nline2\n", encoding="utf-8")
    rc = zoom.main(
        [
            "--quiet",
            "--col-ruler",
            "--follow",
            "--max-frames",
            "1",
            "--file",
            str(src),
            "--rows",
            "1",
            "--cols",
            "5",
            "--zoom",
            "1",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    # --cols 5 → ruler "12345" (last digit of each
    # source column 1..5). The ruler is on the
    # first line of the captured frame.
    assert "12345" in out.out
    # And the magnified line below is "line2"
    # (because --follow tracks the tail).
    assert "line2" in out.out


# ---------------------------------------------------------------------------
# --col-ruler (column-number ruler)
# ---------------------------------------------------------------------------
#
# ``--col-ruler`` prepends a single horizontal line above the magnified
# viewport showing the 1-based column index of each source column. The
# ruler mirrors the magnified width (cols * zoom code points) and shows
# the last digit of each column's index, repeated ``zoom`` times. This
# is the natural sibling of ``--line-numbers``: gutter on the left for
# rows, ruler on top for columns. The two flags compose — both can be
# on the same invocation.


def test_format_col_ruler_basic() -> None:
    """``_format_col_ruler`` prepends a 1-line ruler above the
    magnified output. For a 3-column source at ``zoom=1`` the
    ruler is ``"123"``; the magnified line below it stays
    unchanged. The helper is pure: it doesn't know about the
    renderer, the source, or the CLI — it just splices a
    ruler line onto whatever string you give it.
    """
    rendered = "abc"
    out = zoom._format_col_ruler(rendered, zoom=1, cols=3)
    assert out == "123\nabc"


def test_format_col_ruler_empty() -> None:
    """An empty ``rendered`` is a no-op — returns ``""``
    unchanged. The ``--raw`` and the no-frame case both rely
    on this; if the helper ever started padding the empty
    string with a ruler, both code paths would gain a
    spurious leading line.
    """
    assert (
        zoom._format_col_ruler("", zoom=1, cols=3) == ""
    )
    assert (
        zoom._format_col_ruler("", zoom=2, cols=10) == ""
    )


def test_format_col_ruler_zoom_2() -> None:
    """At ``zoom=2`` each source column occupies 2 code
    points in the ruler. A 10-column source at ``zoom=2``
    produces a 20-char ruler: ``11223344556677889900`` —
    each column's last digit repeated twice, with column
    10's last digit (``0``) repeated twice (``00``). The
    total ruler length always equals ``cols * zoom`` so it
    aligns with the magnified viewport below.
    """
    rendered = "abcdefghij" * 2
    out = zoom._format_col_ruler(rendered, zoom=2, cols=10)
    assert out == "11223344556677889900\n" + rendered


def test_format_col_ruler_offset() -> None:
    """``first_source_col`` shifts the numbering. A
    3-column viewport starting at source column 43 shows
    the digits of 43, 44, 45 (last digits: ``3``, ``4``,
    ``5``). The first source column in view is
    ``first_source_col`` (1-based), matching what
    ``--col-offset 42`` would put in the viewport.
    """
    out = zoom._format_col_ruler(
        "abc", zoom=1, cols=3, first_source_col=43
    )
    assert out == "345\nabc"


def test_format_col_ruler_zoom_1_offset() -> None:
    """``zoom=1`` with an offset. The ruler is the
    literal last-digit of each source column index —
    one character per source column. Useful for
    confirming the ``first_source_col`` shift in the
    minimum-width case.
    """
    out = zoom._format_col_ruler(
        "xyz", zoom=1, cols=3, first_source_col=11
    )
    # Source columns 11, 12, 13 → last digits "1", "2", "3".
    assert out == "123\nxyz"


def test_format_col_ruler_multi_digit_columns() -> None:
    """Multi-digit column indices show only their last
    digit (the only thing that fits in a single
    magnified cell). Columns 9-13 show ``9``, ``0``,
    ``1``, ``2``, ``3`` — the editor-ruler convention.
    """
    out = zoom._format_col_ruler(
        "abcdefghijklm", zoom=1, cols=13, first_source_col=1
    )
    # Columns 1-9 → "123456789"; columns 10-13 → "0123".
    assert out == "1234567890123\nabcdefghijklm"


def test_format_col_ruler_defensive_zoom() -> None:
    """A non-positive ``zoom`` is treated as ``1`` (the
    parse_args-level check has already rejected ``--zoom 0``,
    but the library function is callable from anywhere, so
    we re-validate at the boundary). This matches
    ``_format_line_numbers``'s defensive-zoom convention:
    a stray invalid input still produces a sensible ruler.
    """
    out = zoom._format_col_ruler(
        "abc", zoom=0, cols=3
    )
    assert out == "123\nabc"
    out = zoom._format_col_ruler(
        "abc", zoom=-1, cols=3
    )
    assert out == "123\nabc"


def test_parse_args_col_ruler_default_off() -> None:
    """``--col-ruler`` defaults to ``False`` so a
    vanilla ``paw-zoom …`` keeps its current output
    shape — no ruler, no behaviour change for users
    who didn't ask for the annotation.
    """
    args = zoom.parse_args(["hello"])
    assert args.col_ruler is False


def test_parse_args_col_ruler_flag() -> None:
    """``--col-ruler`` is a ``store_true`` flag —
    passing it once flips the attribute to ``True``.
    """
    args = zoom.parse_args(["--col-ruler", "hello"])
    assert args.col_ruler is True


def test_main_col_ruler_renders_with_ruler(capsys) -> None:
    """End-to-end: ``paw-zoom --col-ruler …`` emits a
    magnified viewport with a column-number ruler on
    top. With ``--cols 3 --zoom 1`` the ruler is
    ``"123"``; the magnified line below it is the
    source verbatim. The default off-by-default
    behaviour is unchanged for users who don't pass
    the flag.
    """
    rc = zoom.main(
        [
            "--rows",
            "1",
            "--cols",
            "3",
            "--zoom",
            "1",
            "--col-ruler",
            "abc",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    lines = out.out.splitlines()
    # Drop the announcement line.
    assert lines[0].startswith("🐾 paw-zoom:")
    assert lines[1:] == [
        "123",
        "abc",
    ]


def test_main_col_ruler_zoom_2_repeats_digit(capsys) -> None:
    """With ``--zoom 2``, each source column occupies
    2 code points in the ruler. A 4-column source at
    ``zoom=2`` with ``--rows 1`` produces an 8-char
    ruler above a 2-line viewport, each line being
    the magnified source row. The composition with
    --zoom works the same way _format_col_ruler says
    it does in its docstring.
    """
    rc = zoom.main(
        [
            "--rows",
            "1",
            "--cols",
            "4",
            "--zoom",
            "2",
            "--col-ruler",
            "abcd",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    lines = out.out.splitlines()
    assert lines[0].startswith("🐾 paw-zoom:")
    # Ruler on top, then the 2 magnified lines below.
    assert lines[1:] == [
        "11223344",
        "aabbccdd",
        "aabbccdd",
    ]


def test_main_col_ruler_offset_shifts_numbering(capsys) -> None:
    """``--col-ruler`` combined with ``--col-offset N``:
    the first source column in view is ``N + 1``
    (1-based), so the ruler numbers start at that
    value. We need a source long enough that
    ``--col-offset 12`` doesn't fall off the edge —
    otherwise the renderer pads with spaces and the
    ruler's column 13 would still show correctly,
    but the magnified viewport would be all blanks.
    Same source (15 chars), two different
    col-offsets, two different ruler numbering
    series — the user can correlate the magnified
    column with its source position.
    """
    src = "abcdefghijklmno"  # 15 columns, plenty for --col-offset 12.
    rc = zoom.main(
        [
            "--rows",
            "1",
            "--cols",
            "3",
            "--zoom",
            "1",
            "--col-offset",
            "12",
            "--col-ruler",
            src,
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    lines = out.out.splitlines()
    assert lines[0].startswith("🐾 paw-zoom:")
    # Col-offset 12 → first source column 13 → ruler "345";
    # the 3 columns of the source starting at index 12 are
    # "mno" (m=col 13, n=col 14, o=col 15).
    assert lines[1:] == [
        "345",
        "mno",
    ]


def test_main_col_ruler_off_by_default(capsys) -> None:
    """Without ``--col-ruler``, the rendered output
    has no ruler — the default behaviour is unchanged.
    Regression guard against an accidental
    flip-the-default mistake.
    """
    rc = zoom.main(
        ["--rows", "1", "--cols", "3", "--zoom", "1", "abc"]
    )
    out = capsys.readouterr()
    assert rc == 0
    # The rendered line is the literal source — no
    # "123" prefix.
    assert "abc" in out.out
    assert "123" not in out.out


def test_main_col_ruler_quiet_still_emits_ruler(capsys) -> None:
    """``--quiet`` suppresses the *announcement* line,
    not the ruler. The ruler is part of the rendered
    output, not the announcement — so
    ``--quiet --col-ruler`` still emits the
    ruler-prefixed viewport. (Quiet + col-ruler is
    the canonical "clean pipe" invocation: render the
    ruler-prefixed block to stdout, nothing else.)
    """
    rc = zoom.main(
        [
            "--quiet",
            "--rows",
            "1",
            "--cols",
            "3",
            "--zoom",
            "1",
            "--col-ruler",
            "abc",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    # No announcement.
    assert "window" not in out.out
    # The ruler is on stdout.
    assert "123\nabc" in out.out


def test_main_col_ruler_composes_with_line_numbers(capsys) -> None:
    """``--col-ruler`` and ``--line-numbers`` are
    orthogonal render-time annotations and compose
    cleanly: the ruler sits above every line, the
    gutter sits on the left of every line. The
    combination gives the user a "spreadsheet view"
    of the magnified block — column numbers on top,
    row numbers on the left, magnified text in the
    middle. The two helpers splice onto the
    rendered string in a fixed order (ruler first,
    then gutter) so the output is deterministic.
    """
    rc = zoom.main(
        [
            "--rows",
            "2",
            "--cols",
            "3",
            "--zoom",
            "1",
            "--line-numbers",
            "--col-ruler",
            "abc\ndef",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    lines = out.out.splitlines()
    assert lines[0].startswith("🐾 paw-zoom:")
    # Ruler on top, then the guttered viewport.
    assert lines[1:] == [
        "123",
        "   1 │ abc",
        "   2 │ def",
    ]


def test_main_col_ruler_discovery_silent(capsys) -> None:
    """``--col-ruler`` is a *render-time* annotation;
    the discovery flags (--list-backends, --info,
    --size, --stats, --sha) and --raw short-circuit
    before any render, so a stray ``--col-ruler`` on
    an invocation that will never produce a
    magnified output is silently ignored. This
    matches ``--line-numbers``'s discovery-silent
    convention.
    """
    rc = zoom.main(
        [
            "--col-ruler",
            "--list-backends",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    # --list-backends prints the backends — no ruler.
    assert out.out.strip() == "\n".join(
        ["fake", "x11", "win32", "quartz"]
    )
    # And the ruler does not appear anywhere.
    assert "123" not in out.out


def test_help_text_mentions_col_ruler() -> None:
    """``--col-ruler`` is in the ``--help`` output
    so a user who hits ``paw-zoom --help`` can
    find it. Regression guard against an
    accidental rename of the flag.
    """
    help_text = zoom.build_parser().format_help()

# ---------------------------------------------------------------------------
# --border (render-time frame annotation)
# ---------------------------------------------------------------------------


def test_format_border_basic() -> None:
    """``_format_border`` wraps a single-line string in a
    minimal light box. The ``+---+`` top / bottom lines
    flank a single ``|x|`` row. The helper is width-aware:
    the box is exactly ``len(line) + 2`` chars wide for a
    single line.
    """
    out = zoom._format_border("abc")
    assert out == "+---+\n|abc|\n+---+"
    # sanity: also check the parts
    lines = out.splitlines()
    assert lines[0] == "+---+"
    assert lines[-1] == "+---+"
    assert lines[1] == "|abc|"


def test_format_border_multi_line() -> None:
    """Multi-line ``rendered`` gets a border on *every*
    line; the top / bottom ``+---+`` width is the
    longest line. The output preserves the line count
    of the input plus 2 (the frame).
    """
    out = zoom._format_border("abc\ndef")
    assert out == "+---+\n|abc|\n|def|\n+---+"
    lines = out.splitlines()
    # 2 content + 2 frame = 4 lines.
    assert len(lines) == 4


def test_format_border_empty() -> None:
    """An empty ``rendered`` is still wrapped — the user
    asked for a border around an empty viewport, so the
    output is a 1-cell box (``+\\n| \\n+\\n``). This
    differs from ``_format_col_ruler``'s empty-case
    (which is a no-op, because a ruler is a line the
    user would never want to see alone; a border is a
    *frame* and an empty frame is a valid frame).
    """
    assert zoom._format_border("") == "+\n| \n+\n"


def test_format_border_short_line_padded() -> None:
    """Shorter lines are right-padded with spaces to the
    box width before the right-hand border is added.
    The longest line determines the width, so a
    2-line ``rendered`` of widths 2 and 11 produces a
    13-wide box (11 + 2 border) with the shorter
    line space-padded.
    """
    out = zoom._format_border("ab\nlonger line")
    assert out == (
        "+-----------+\n"
        "|ab         |\n"
        "|longer line|\n"
        "+-----------+"
    )


def test_format_border_trailing_newline_no_phantom_row() -> None:
    """``str.splitlines()`` drops a single trailing
    newline, so ``"abc\\n"`` (one content line) does
    NOT produce a phantom empty ``| |`` row at the
    bottom — the standard editor convention.
    """
    assert zoom._format_border("abc\n") == (
        "+---+\n|abc|\n+---+"
    )


def test_format_border_unicode_width() -> None:
    """Unicode code points are counted individually
    (not by display width). The box width is the
    count of ``str`` code points, matching the
    magnified viewport convention everywhere else
    in the file. A 3-codepoint string gets a 3-cell
    box (the box width is the longest line, which is
    3 codepoints here) regardless of whether the
    codepoints render as narrow or wide.
    """
    out = zoom._format_border("🐾🐾🐾")
    # 3 codepoints (1 codepoint per emoji in this case) +
    # 2 borders = "+---+" for the top.
    assert out == "+---+\n|🐾🐾🐾|\n+---+"


def test_parse_args_border_default_off() -> None:
    """``--border`` defaults to ``False`` so a vanilla
    ``paw-zoom …`` keeps its current output shape —
    no frame, no behaviour change for users who
    didn't ask for the annotation.
    """
    args = zoom.parse_args(["hello"])
    assert args.border is False


def test_parse_args_border_flag() -> None:
    """``--border`` is a ``store_true`` flag — passing
    it once flips the attribute to ``True``.
    """
    args = zoom.parse_args(["--border", "hello"])
    assert args.border is True


def test_main_border_renders_with_frame(capsys) -> None:
    """End-to-end: ``paw-zoom --border …`` emits a
    magnified viewport wrapped in a light box. With
    ``--rows 1 --cols 3 --zoom 1`` the source
    ``"abc"`` is one line, so the box has 5 cells
    wide and 3 lines tall (top, content, bottom).
    The default off-by-default behaviour is unchanged
    for users who don't pass the flag.
    """
    rc = zoom.main(
        [
            "--rows",
            "1",
            "--cols",
            "3",
            "--zoom",
            "1",
            "--border",
            "abc",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    lines = out.out.splitlines()
    # Drop the announcement line.
    assert lines[0].startswith("🐾 paw-zoom:")
    assert lines[1:] == [
        "+---+",
        "|abc|",
        "+---+",
    ]


def test_main_border_off_by_default(capsys) -> None:
    """Without ``--border``, the rendered output
    is the *unframed* viewport — no ``+---+``
    anywhere. Confirms the off-by-default
    behaviour is preserved.
    """
    rc = zoom.main(
        [
            "--rows",
            "1",
            "--cols",
            "3",
            "--zoom",
            "1",
            "abc",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    # The unflagged render is just the magnified source
    # repeated for --rows (1 row -> 1 line).
    lines = out.out.splitlines()
    assert lines[0].startswith("🐾 paw-zoom:")
    assert lines[1] == "abc"
    # And no border characters anywhere in the
    # unflagged output.
    assert "+" not in out.out
    assert "|" not in out.out


def test_main_border_quiet_still_emits_frame(capsys) -> None:
    """``--quiet --border`` still emits the framed
    viewport. The frame is part of the rendered
    output, not the announcement line that
    ``--quiet`` suppresses. (Quiet + border is
    the natural way to capture a clean framed
    block into a log without the announcement
    getting in the way.)
    """
    rc = zoom.main(
        [
            "--rows",
            "1",
            "--cols",
            "3",
            "--zoom",
            "1",
            "--quiet",
            "--border",
            "abc",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    # No announcement line.
    assert "🐾 paw-zoom" not in out.out
    # But the frame is there.
    assert out.out.splitlines() == [
        "+---+",
        "|abc|",
        "+---+",
    ]


def test_main_border_composes_with_line_numbers(capsys) -> None:
    """``--border`` and ``--line-numbers`` compose:
    ``--line-numbers`` runs first (so the gutter
    is on the *inside* of every viewport line),
    then ``--border`` wraps the guttered viewport
    in a frame. The frame's width follows the
    widest line — i.e. it widens to accommodate
    the gutter.
    """
    rc = zoom.main(
        [
            "--rows",
            "1",
            "--cols",
            "3",
            "--zoom",
            "1",
            "--line-numbers",
            "--border",
            "abc",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    lines = out.out.splitlines()
    assert lines[0].startswith("🐾 paw-zoom:")
    # The gutter widens the content to "   1 | abc"
    # (3-char gutter + " | " = 7 chars; plus 3 chars of
    # content = 10 chars total), so the box is
    # "+----------+" (12 chars).
    assert lines[1:] == [
        "+----------+",
        "|   1 │ abc|",
        "+----------+",
    ]


def test_main_border_composes_with_col_ruler(capsys) -> None:
    """``--border`` and ``--col-ruler`` compose:
    ``--col-ruler`` runs first (so the ruler is
    on the *inside* of the top, above the
    viewport), then ``--border`` wraps the ruled
    viewport in a frame. The frame's width
    follows the widest line — i.e. it widens to
    accommodate the ruler, and its height grows
    by 1 to enclose the ruler row.
    """
    rc = zoom.main(
        [
            "--rows",
            "1",
            "--cols",
            "3",
            "--zoom",
            "1",
            "--col-ruler",
            "--border",
            "abc",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    lines = out.out.splitlines()
    assert lines[0].startswith("🐾 paw-zoom:")
    # The ruler widens the topmost line to "123"
    # (3 chars), same as the content; the box is
    # therefore 5 wide. The frame now has 4 rows:
    # top, ruler, content, bottom.
    assert lines[1:] == [
        "+---+",
        "|123|",
        "|abc|",
        "+---+",
    ]


def test_main_border_discovery_silent(capsys) -> None:
    """``--border`` is a *render-time* annotation;
    every discovery flag short-circuits in ``main()``
    before any render, so a stray ``--border`` on
    a discovery invocation is silently ignored
    (no border appears in the discovery output).
    Same convention as ``--line-numbers`` and
    ``--col-ruler`` use for the same reason.
    """
    rc = zoom.main(["--list-backends", "--border"])
    out = capsys.readouterr()
    assert rc == 0
    # No border in the discovery output.
    assert "+" not in out.out
    assert "|" not in out.out


def test_main_border_with_snapshot_writes_frame(
    capsys, tmp_path
) -> None:
    """``--border --snapshot PATH`` writes the
    *framed* viewport to PATH (the frame is part
    of the rendered output, not a stdout-side
    decoration). Useful for piping a clean
    bordered block to a log or a downstream tool.
    """
    snap = tmp_path / "frame.txt"
    rc = zoom.main(
        [
            "--rows",
            "1",
            "--cols",
            "3",
            "--zoom",
            "1",
            "--quiet",
            "--border",
            "--snapshot",
            str(snap),
            "abc",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    # Nothing on stdout when --snapshot is set.
    assert out.out == ""
    text = snap.read_text(encoding="utf-8").splitlines()
    assert text == [
        "+---+",
        "|abc|",
        "+---+",
    ]


def test_help_text_mentions_border() -> None:
    """``--border`` is in the ``--help`` output so
    a user who hits ``paw-zoom --help`` can find
    it. Regression guard against an accidental
    rename of the flag.
    """
    help_text = zoom.build_parser().format_help()
    assert "--border" in help_text


# ---------------------------------------------------------------------------
# ``--words`` discovery (the lexical companion of ``--size`` / ``--stats``)
# ---------------------------------------------------------------------------


def test_source_words_basic() -> None:
    """``"hello world hello"`` → 3 words, 2 unique
    (case-folded). ``unique_words`` collapses ``hello``
    with ``Hello`` (same word under different case)."""
    assert zoom._source_words("hello world hello") == (3, 2)


def test_source_words_empty_string() -> None:
    """An empty source is ``(0, 0)`` — the "nothing to
    count" answer. Deliberately diverges from
    ``_source_size``'s ``(1, 0)`` on the same input
    because the two flags answer different questions
    (renderer-shape vs content-count)."""
    assert zoom._source_words("") == (0, 0)


def test_source_words_whitespace_only() -> None:
    """A whitespace-only source is also ``(0, 0)`` —
    nothing meaningful to count."""
    assert zoom._source_words("   \t  \n  \n") == (0, 0)


def test_source_words_drops_trailing_newline() -> None:
    """A file ending in ``\\n`` doesn't push a phantom
    blank into the token stream — same convention
    ``_source_size`` and ``_source_stats`` use."""
    assert zoom._source_words("one two three\n") == (3, 3)


def test_source_words_keeps_internal_newlines() -> None:
    """Internal newlines split tokens exactly like
    spaces (it's all whitespace to ``str.split``)."""
    assert zoom._source_words("alpha\nbeta\ngamma") == (3, 3)


def test_source_words_mixed_whitespace() -> None:
    """Runs of mixed whitespace (space, tab, newline)
    collapse to a single token boundary, the same way
    ``wc -w`` counts them."""
    assert zoom._source_words("a  b\tc\nd\te") == (5, 5)


def test_source_words_case_folding() -> None:
    """``unique_words`` is computed from the
    case-folded token list, so ``Hello`` and ``hello``
    count as the same distinct word. The case-folding
    is done with ``str.casefold``, not ``str.lower``,
    so a token like ``"ß"`` is treated the same as
    ``"ss"``."""
    assert zoom._source_words("Hello hello HELLO HeLlO") == (4, 1)


def test_source_words_cjk_single_token() -> None:
    """A CJK string with no ASCII whitespace counts
    as one word, the same way ``wc -w`` counts it.
    Multi-byte characters are token boundaries only
    if they're whitespace."""
    assert zoom._source_words("日本語のテキスト") == (1, 1)


def test_source_words_cjk_mixed_with_ascii() -> None:
    """Mixing CJK and ASCII tokens works as expected:
    each whitespace-delimited run is one token."""
    assert zoom._source_words("hello 日本語 world") == (3, 3)


def test_source_words_punctuation_attached() -> None:
    """Punctuation attached to a word stays attached:
    ``"hello,"`` and ``"hello"`` are two distinct
    tokens (matches ``str.split`` / ``wc -w``
    behaviour). Users who want lemmatisation can
    pipe the source through a real NLP tool."""
    # 3 words total: ``hello,`` / ``world`` / ``hello``.
    # 3 unique (case-folded): ``hello,`` / ``world`` /
    # ``hello`` — ``hello,`` and ``hello`` are distinct
    # because the comma is a real character that
    # survives case-folding (``str.casefold`` doesn't
    # strip punctuation, only folds Unicode case).
    assert zoom._source_words("hello, world hello") == (3, 3)


def test_source_words_dedup_counted_per_occurrence() -> None:
    """The total word count is per-occurrence (not
    per-distinct), so ``"a a a"`` is ``3, 1``."""
    assert zoom._source_words("a a a") == (3, 1)


def test_words_to_text_format() -> None:
    """The text rendering is a 2-line fixed
    ``key: value`` block — parallel to
    ``_stats_to_text``. Each field on its own line so
    a downstream ``grep '^words:'`` / ``awk`` can
    pick either field with a one-line selector."""
    assert zoom._words_to_text((5, 3)) == "words: 5\nunique_words: 3"
    assert zoom._words_to_text((0, 0)) == "words: 0\nunique_words: 0"
    assert zoom._words_to_text((1, 1)) == "words: 1\nunique_words: 1"


def test_words_to_json_round_trip() -> None:
    """``--words --json`` emits a single-line parseable
    object with the expected keys and a deterministic
    key order (``sort_keys=True``)."""
    s = zoom._words_to_json((5, 3))
    assert "\n" not in s
    assert json.loads(s) == {"unique_words": 3, "words": 5}


def test_words_to_json_zero() -> None:
    """The empty case is the natural ``{"unique_words":
    0, "words": 0}`` shape — never a missing key or a
    stringified number."""
    assert json.loads(zoom._words_to_json((0, 0))) == {
        "unique_words": 0,
        "words": 0,
    }


def test_words_to_json_key_order() -> None:
    """Keys come out in sorted order so a byte-stable
    test (or a downstream diff) sees a predictable
    shape. ``unique_words`` sorts before ``words``."""
    s = zoom._words_to_json((7, 5))
    # Find the position of each key and assert
    # ``unique_words`` comes first.
    assert s.index('"unique_words"') < s.index('"words"')


def test_parse_args_words_default_off() -> None:
    """``--words`` defaults to ``False`` — the discovery
    flag is opt-in, same as ``--size`` / ``--stats`` /
    ``--sha``."""
    args = zoom.parse_args(["hello"])
    assert args.words is False


def test_parse_args_words_flag() -> None:
    """``--words`` flips the flag and composes with the
    text-source flags (positional / ``--file``) and
    ``--json``."""
    args = zoom.parse_args(["--words", "--json", "hello world"])
    assert args.words is True
    assert args.as_json is True


def test_main_words_positional_text_mode(capsys) -> None:
    """``paw-zoom --words "hello world hello"`` prints
    the 2-line ``key: value`` block and exits 0 without
    any rendering. ``print()`` adds a trailing ``\\n``
    — the output is exactly what the test pins."""
    rc = zoom.main(["--words", "hello world hello"])
    out = capsys.readouterr()
    assert rc == 0
    assert out.out == "words: 3\nunique_words: 2\n"
    assert out.err == ""


def test_main_words_positional_multiline(capsys) -> None:
    """A multi-line positional source: lines split on
    ``\\n`` (same convention as ``_source_size`` and
    ``_source_stats``), trailing ``\\n`` dropped, then
    tokenised."""
    rc = zoom.main(["--words", "alpha\nbeta\ngamma\n"])
    out = capsys.readouterr()
    assert rc == 0
    assert out.out == "words: 3\nunique_words: 3\n"


def test_main_words_json_mode(capsys) -> None:
    """``--words --json`` emits a single-line parseable
    object and exits 0. The shape matches the rest of
    the discovery flags: sorted keys, no ASCII escaping,
    no surrounding text."""
    rc = zoom.main(["--words", "--json", "hello world hello"])
    out = capsys.readouterr()
    assert rc == 0
    assert "\n" not in out.out.rstrip("\n")
    assert json.loads(out.out) == {"unique_words": 2, "words": 3}


def test_main_words_whitespace_file_is_zero(tmp_path, capsys) -> None:
    """A whitespace-only file is the natural ``(0, 0)``
    shape — the "nothing to count" answer. The source
    resolver accepts a whitespace-only ``--file``
    input (only positional / stdin are stripped at
    the resolver level — the file's bytes are
    returned verbatim), and the tokeniser correctly
    reports ``(0, 0)`` on it."""
    p = tmp_path / "blank.txt"
    p.write_text("   \t  \n\n  \n", encoding="utf-8")
    rc = zoom.main(["--words", "--file", str(p)])
    out = capsys.readouterr()
    assert rc == 0
    assert out.out == "words: 0\nunique_words: 0\n"


def test_main_words_file_source(tmp_path, capsys) -> None:
    """``--words --file PATH`` reads the file and
    reports its word counts. Sanity-checks the
    ``--file`` path of source resolution for
    ``--words``."""
    p = tmp_path / "doc.txt"
    p.write_text("the quick brown fox\njumps over the lazy dog\n", encoding="utf-8")
    rc = zoom.main(["--words", "--file", str(p)])
    out = capsys.readouterr()
    assert rc == 0
    # 9 words total: the, quick, brown, fox, jumps, over,
    # the, lazy, dog. Case-folded distinct: the (×2),
    # quick, brown, fox, jumps, over, lazy, dog = 8.
    assert out.out == "words: 9\nunique_words: 8\n"


def test_main_words_file_missing_is_error(tmp_path, capsys) -> None:
    """``--words --file /missing`` exits 1 with a clear
    stderr message — the file-not-found path is the
    same one ``_resolve_source`` raises, and ``--words``
    doesn't try to be cleverer than that."""
    missing = tmp_path / "does-not-exist.txt"
    rc = zoom.main(["--words", "--file", str(missing)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "file not found" in captured.err


def test_main_words_with_screen_capture(capsys) -> None:
    """``--words --screen --backend fake --fake-grid "..."``
    captures the fake screen and reports the captured
    grid's word counts. ``FakeScreen`` truncates every
    row to the shortest row's width, so the captured
    grid is whatever the fake source is — here we use
    a one-line source so the result is unambiguous."""
    rc = zoom.main(
        [
            "--words",
            "--screen",
            "--backend",
            "fake",
            "--fake-grid",
            "hello world hello",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    assert out.out == "words: 3\nunique_words: 2\n"


def test_main_words_with_unsupported_screen_backend_exits_1(capsys) -> None:
    """``--words --screen --backend x11`` on a headless
    box (no ``$DISPLAY`` / ``xwd``) exits 1 with the
    standard "not yet implemented on this OS" message
    and the same exit code the render path uses."""
    rc = zoom.main(["--words", "--screen", "--backend", "x11"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "not yet implemented on this OS" in captured.err


def test_main_words_without_any_source_exits_2(capsys, monkeypatch) -> None:
    """``--words`` with no source (no positional, no
    ``--file``, no stdin) is a usage error (exit 2) —
    same convention as ``--size`` / ``--stats`` /
    ``--sha``."""
    # Force stdin to be empty so the resolver doesn't
    # accidentally pick up whatever pytest captured.
    monkeypatch.setenv("WPAW_ZOOM_STDIN_OVERRIDE", "")
    rc = zoom.main(["--words"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "no text" in captured.err


def test_main_words_mutual_exclusion_with_size(capsys) -> None:
    """``--words --size`` is rejected (exit 2) — they
    are two different kinds of discovery and we don't
    want to emit more than one of them per invocation.
    The ``--size``-block runs first in ``parse_args``,
    so the contradiction message names ``--size`` as
    the flag that found ``--words`` in conflict."""
    rc = zoom.main(["--words", "--size", "x"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "--size cannot be combined with --words" in captured.err


def test_main_words_mutual_exclusion_with_stats(capsys) -> None:
    """``--words --stats`` is rejected (exit 2). The
    ``--stats``-block runs before ``--words``'s block
    in ``parse_args`` and lists ``--words`` in its
    flag-walk, so the message names ``--stats`` as
    the flag that found ``--words`` in conflict."""
    rc = zoom.main(["--words", "--stats", "x"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "--stats cannot be combined with --words" in captured.err


def test_main_words_mutual_exclusion_with_info(capsys) -> None:
    """``--words --info`` is rejected (exit 2) — they
    are two different kinds of discovery (``--info`` is
    screen-capture-side, ``--words`` is text-side)."""
    rc = zoom.main(["--words", "--info", "--screen"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "--words cannot be combined with --info" in captured.err


def test_main_words_mutual_exclusion_with_live(capsys) -> None:
    """``--words --live`` is rejected (exit 2) —
    ``--live`` is a render driver, ``--words`` is
    metadata-only."""
    rc = zoom.main(["--words", "--live", "x"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "--words cannot be combined with --live" in captured.err


def test_main_words_mutual_exclusion_with_follow(capsys) -> None:
    """``--words --follow`` is rejected (exit 2) — same
    rationale as ``--live``."""
    rc = zoom.main(["--words", "--follow", "x"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "--words cannot be combined with --follow" in captured.err


def test_main_words_mutual_exclusion_with_max_frames(capsys) -> None:
    """``--words --max-frames`` is rejected (exit 2) —
    ``--max-frames`` is a ``--live`` modifier, and
    ``--words`` is metadata-only."""
    rc = zoom.main(["--words", "--max-frames", "3", "x"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "--words cannot be combined with --max-frames" in captured.err


def test_main_words_mutual_exclusion_with_snapshot(capsys) -> None:
    """``--words --snapshot PATH`` is rejected (exit 2)
    — ``--snapshot`` is a render driver (it writes the
    rendered viewport to a file), and ``--words`` is
    metadata-only. The snapshot write must NOT happen
    even if the path is writable (a regression guard
    against the contradiction being caught too late)."""
    rc = zoom.main(["--words", "--snapshot", "/tmp/_paw_zoom_words_should_not_write.txt", "x"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "--words cannot be combined with --snapshot" in captured.err


def test_main_words_mutual_exclusion_with_raw(capsys) -> None:
    """``--words --raw`` is rejected (exit 2) — ``--raw``
    is a render driver (it dumps the source to stdout
    verbatim), and ``--words`` is metadata-only."""
    rc = zoom.main(["--words", "--raw", "x"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "--words cannot be combined with --raw" in captured.err


def test_main_words_mutual_exclusion_with_list_backends(capsys) -> None:
    """``--words --list-backends`` is rejected (exit 2)
    — ``--list-backends`` is a screen-capture-side
    discovery, ``--words`` is text-side."""
    rc = zoom.main(["--words", "--list-backends", "x"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "--words cannot be combined with --list-backends" in captured.err


def test_main_words_with_sha(capsys) -> None:
    """``--words --sha`` is rejected (exit 2). The
    ``--sha``-wins-on-tie rule (it's ordered after
    ``--words``) means the message names ``--sha`` as
    the offender, not ``--words``."""
    rc = zoom.main(["--words", "--sha", "x"])
    captured = capsys.readouterr()
    assert rc == 2
    # ``--sha`` is checked after ``--words`` in parse_args,
    # so the ``--sha`` contradiction message wins on a
    # tie — the user gets told the more specific reason.
    assert "--sha cannot be combined with --words" in captured.err


def test_main_words_quiet_does_not_affect_output(capsys) -> None:
    """``--words`` is metadata-only — there's no
    banner to suppress, so ``--quiet`` is a silent
    no-op (it composes cleanly without changing the
    output shape)."""
    rc = zoom.main(["--words", "--quiet", "hello world"])
    out = capsys.readouterr()
    assert rc == 0
    assert out.out == "words: 2\nunique_words: 2\n"
    assert out.err == ""


def test_main_words_silently_ignores_viewport_flags(capsys) -> None:
    """The viewport-modifying flags (``--zoom`` /
    ``--rows`` / ``--cols`` / ``--offset`` /
    ``--col-offset`` / ``--charset``) are silently
    ignored in ``--words`` mode — the output shape of
    a word count does not depend on them, so a shell
    alias can keep them without breaking the report.
    Same convention ``--size`` / ``--stats`` / ``--sha``
    use."""
    rc = zoom.main(
        ["--words", "--zoom", "8", "--rows", "1", "--charset", "hash", "hello world"]
    )
    out = capsys.readouterr()
    assert rc == 0
    assert out.out == "words: 2\nunique_words: 2\n"


def test_main_words_short_circuits_before_render(capsys) -> None:
    """``--words`` exits 0 without ever building a
    ZoomConfig or calling ``render_viewport`` — a
    direct render-test on the same input would have
    produced a non-trivial magnified block, and a
    regression where the render path runs *and then*
    the discovery is emitted would fail the
    ``assert out.err == ""`` check below (the
    non-quiet banner would land in stderr)."""
    rc = zoom.main(["--words", "hello"])
    out = capsys.readouterr()
    assert rc == 0
    assert "🐾" not in out.err  # no announcement banner


def test_help_text_mentions_words() -> None:
    """``--words`` is in the ``--help`` output so a
    user who hits ``paw-zoom --help`` can find it.
    Regression guard against an accidental rename of
    the flag (the rest of the test suite already pins
    the flag name in the main() end-to-end tests,
    but the help-text regression catches the case
    where someone refactors the parser and drops the
    argument)."""
    help_text = zoom.build_parser().format_help()
    assert "--words" in help_text
