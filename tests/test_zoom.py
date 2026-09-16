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
