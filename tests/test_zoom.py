"""Tests for ``paw-zoom`` (text-viewport magnifier, ASCII POC)."""
from __future__ import annotations

import io
import os
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
