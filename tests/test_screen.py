"""Tests for ``whisperpaw._screen`` (paw-zoom v0.2 screen-capture adapters)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout

import pytest

from whisperpaw import _screen, zoom


# ---------------------------------------------------------------------------
# FakeScreen
# ---------------------------------------------------------------------------


def test_fake_screen_screen_size_basic() -> None:
    """``screen_size`` returns (width, height) derived from the grid."""
    cap = _screen.FakeScreen(["abc", "def", "ghi"])
    assert cap.screen_size() == (3, 3)


def test_fake_screen_normalises_row_widths() -> None:
    """Rows of different lengths are trimmed to the shortest width
    (a real monospace text grid has no notion of "row N is wider")."""
    cap = _screen.FakeScreen(["abcde", "ab"])
    assert cap.screen_size() == (2, 2)
    # The internal grid is trimmed too — re-capture returns the
    # canonical width, not the original.
    assert cap.capture(x=0, y=0, w=2, h=2) == ["ab", "ab"]


def test_fake_screen_empty_grid_rejected() -> None:
    """An empty grid is a usage error (no rows = no "screen")."""
    with pytest.raises(ValueError):
        _screen.FakeScreen([])


def test_fake_screen_capture_basic() -> None:
    """A rectangular capture returns the requested sub-grid."""
    cap = _screen.FakeScreen(
        [
            "abcdef",
            "ghijkl",
            "mnopqr",
        ]
    )
    assert cap.capture(x=0, y=0, w=3, h=2) == ["abc", "ghi"]


def test_fake_screen_capture_clamps_past_right_edge() -> None:
    """Capturing past the row's right edge pads with spaces on the right."""
    cap = _screen.FakeScreen(["abc", "def"])
    # x=2, w=5 on a 3-wide screen: chars at index 2 only, then 4 spaces.
    assert cap.capture(x=2, y=0, w=5, h=1) == ["c    "]


def test_fake_screen_capture_clamps_past_bottom_edge() -> None:
    """Capturing past the grid's bottom edge returns empty lines."""
    cap = _screen.FakeScreen(["abc"])
    # y=2, h=3 on a 3-row screen: row 2 is past the end, so 3 empty lines.
    assert cap.capture(x=0, y=2, w=3, h=3) == ["   ", "   ", "   "]


def test_fake_screen_capture_negative_x_pads_on_left() -> None:
    """A negative x is treated as a window that starts left of the screen;
    the missing left columns come back as spaces."""
    cap = _screen.FakeScreen(["abcdef"])
    # x=-2, w=5: 2 spaces of left padding, then "abcde" from the row.
    assert cap.capture(x=-2, y=0, w=5, h=1) == ["  abc"]


def test_fake_screen_capture_negative_y_pads_with_empty_rows() -> None:
    """A negative y yields empty rows at the top (no wraparound)."""
    cap = _screen.FakeScreen(["abc"])
    # y=-1, h=2: first row is empty, second row is "abc".
    assert cap.capture(x=0, y=-1, w=3, h=2) == ["   ", "abc"]


def test_fake_screen_capture_validates_dimensions() -> None:
    """Negative w/h is a programmer error, not a render error."""
    cap = _screen.FakeScreen(["abc"])
    with pytest.raises(ValueError):
        cap.capture(x=0, y=0, w=-1, h=1)
    with pytest.raises(ValueError):
        cap.capture(x=0, y=0, w=1, h=-1)


# ---------------------------------------------------------------------------
# _parse_region
# ---------------------------------------------------------------------------


def test_parse_region_none_means_full_screen() -> None:
    """``None`` defaults to the entire screen extent."""
    assert _screen._parse_region(None, screen_size=(80, 24)) == (0, 0, 80, 24)


def test_parse_region_full_string_means_full_screen() -> None:
    """``'full'`` (any case, surrounding whitespace ok) is the whole screen."""
    assert _screen._parse_region("full", screen_size=(80, 24)) == (0, 0, 80, 24)
    assert _screen._parse_region(" FULL ", screen_size=(80, 24)) == (0, 0, 80, 24)


def test_parse_region_xywh_parsed() -> None:
    """A 'X,Y,W,H' string is parsed into the four components."""
    assert _screen._parse_region("10,5,20,8", screen_size=(100, 50)) == (10, 5, 20, 8)


def test_parse_region_clamps_w_to_screen() -> None:
    """A rectangle wider than the screen is clipped on the right."""
    # 80-wide screen, region starts at x=70 with w=20 → w is clipped to 10.
    assert _screen._parse_region("70,0,20,24", screen_size=(80, 24)) == (70, 0, 10, 24)


def test_parse_region_clamps_h_to_screen() -> None:
    """A rectangle taller than the screen is clipped at the bottom."""
    # 24-tall screen, region starts at y=20 with h=10 → h is clipped to 4
    # (y + h was 30, screen is only 24 tall).
    assert _screen._parse_region("0,20,80,10", screen_size=(80, 24)) == (0, 20, 80, 4)


def test_parse_region_off_screen_returns_zero_area() -> None:
    """A region entirely past the right / bottom edge returns (0,0,0,0)."""
    assert _screen._parse_region("100,0,10,10", screen_size=(80, 24)) == (0, 0, 0, 0)
    assert _screen._parse_region("0,50,10,10", screen_size=(80, 24)) == (0, 0, 0, 0)


def test_parse_region_wrong_component_count_is_error() -> None:
    """A region string with != 4 components raises ValueError."""
    with pytest.raises(ValueError):
        _screen._parse_region("1,2,3", screen_size=(80, 24))
    with pytest.raises(ValueError):
        _screen._parse_region("1,2,3,4,5", screen_size=(80, 24))


def test_parse_region_non_integer_is_error() -> None:
    """Non-integer components raise ValueError."""
    with pytest.raises(ValueError):
        _screen._parse_region("a,b,c,d", screen_size=(80, 24))
    with pytest.raises(ValueError):
        _screen._parse_region("0,0,3.5,4", screen_size=(80, 24))


def test_parse_region_negative_component_is_error() -> None:
    """Negative components are almost always a typo, so we error rather
    than silently clamp (the caller can fix it)."""
    with pytest.raises(ValueError):
        _screen._parse_region("-1,0,10,10", screen_size=(80, 24))
    with pytest.raises(ValueError):
        _screen._parse_region("0,0,-10,10", screen_size=(80, 24))


def test_parse_region_zero_area_is_not_an_error() -> None:
    """A 0×0 region is degenerate but legal — the renderer produces
    an empty string and the loop continues. We don't second-guess."""
    assert _screen._parse_region("0,0,0,0", screen_size=(80, 24)) == (0, 0, 0, 0)
    assert _screen._parse_region("0,0,10,0", screen_size=(80, 24)) == (0, 0, 10, 0)


# ---------------------------------------------------------------------------
# capture_screen_to_source
# ---------------------------------------------------------------------------


def test_capture_screen_to_source_full_screen() -> None:
    """A full-screen capture produces a string with the right number of lines."""
    cap = _screen.FakeScreen(["abc", "def"])
    out = _screen.capture_screen_to_source(cap)
    assert out == "abc\ndef"


def test_capture_screen_to_source_with_region() -> None:
    """A region is applied and the result is just that sub-grid."""
    cap = _screen.FakeScreen(
        [
            "abcdef",
            "ghijkl",
            "mnopqr",
        ]
    )
    out = _screen.capture_screen_to_source(cap, region="0,1,3,2")
    assert out == "ghi\nmno"


def test_capture_screen_to_source_zero_region_returns_empty() -> None:
    """A region with w=0 or h=0 returns an empty string (no work to do)."""
    cap = _screen.FakeScreen(["abc"])
    assert _screen.capture_screen_to_source(cap, region="0,0,0,5") == ""
    assert _screen.capture_screen_to_source(cap, region="0,0,5,0") == ""


def test_capture_screen_to_source_renders_through_zoom() -> None:
    """End-to-end: capture → render_viewport produces the expected magnified output.

    This is the proof that the v0.2 pipeline composes with v0.1
    unchanged — the only thing that changed is the *source* of the
    text. The magnification math, the padding, the offset/follow
    behaviour all work the same way.
    """
    cap = _screen.FakeScreen(
        [
            "abc",
            "def",
        ]
    )
    src = _screen.capture_screen_to_source(cap, region="0,0,3,2")
    rendered = zoom.render_viewport(
        source=src,
        cfg=zoom.ZoomConfig(rows=2, cols=3, zoom=2),
    )
    # 2 source rows × zoom 2 → 4 output lines, each 6 wide.
    assert rendered == "aabbcc\n" "aabbcc\n" "ddeeff\n" "ddeeff"


# ---------------------------------------------------------------------------
# get_capture
# ---------------------------------------------------------------------------


def test_get_capture_fake_requires_grid() -> None:
    """``--backend fake`` without a grid is a usage error — we don't
    know what screen to pretend to have."""
    with pytest.raises(ValueError):
        _screen.get_capture("fake")


def test_get_capture_fake_returns_instance() -> None:
    """``--backend fake`` with a grid returns a FakeScreen over it."""
    cap = _screen.get_capture("fake", fake_grid=["abc"])
    assert isinstance(cap, _screen.FakeScreen)
    assert cap.screen_size() == (3, 1)


def test_get_capture_auto_with_grid_returns_fake() -> None:
    """``auto`` plus a grid is shorthand for the fake adapter — handy for
    scripts that want to test the rest of the pipeline without a display."""
    cap = _screen.get_capture("auto", fake_grid=["abc", "def"])
    assert isinstance(cap, _screen.FakeScreen)


def test_get_capture_auto_without_grid_returns_none_on_headless() -> None:
    """On a headless box (no X11 / Win32 / Quartz), ``auto`` returns ``None``."""
    # This box has no DISPLAY, no X11 libs, no mss, no PIL — so all
    # three OS factories return None and ``auto`` propagates that.
    cap = _screen.get_capture("auto")
    assert cap is None


def test_get_capture_unknown_backend_is_error() -> None:
    """An unknown backend name is a programmer error, not a render error."""
    with pytest.raises(ValueError):
        _screen.get_capture("wayland")


def test_get_capture_os_backend_returns_none_today() -> None:
    """All three OS adapters are stubs in v0.2; they return None and the
    CLI prints ``backend_unsupported_message`` instead. This test pins
    that contract so the next tick can flip them to "real" without
    breaking callers that already handle ``None``."""
    assert _screen.get_capture("x11") is None
    assert _screen.get_capture("win32") is None
    assert _screen.get_capture("quartz") is None


# ---------------------------------------------------------------------------
# Discovery helpers (list_backends, to_json)
# ---------------------------------------------------------------------------


def test_list_backends_returns_canonical_order() -> None:
    """``list_backends`` returns KNOWN_BACKENDS, fresh copy."""
    assert _screen.list_backends() == ["fake", "x11", "win32", "quartz"]


def test_list_backends_returns_fresh_list() -> None:
    """Mutating the returned list doesn't affect future calls (or
    KNOWN_BACKENDS)."""
    a = _screen.list_backends()
    a.append("rogue")
    b = _screen.list_backends()
    assert "rogue" not in b
    assert _screen.KNOWN_BACKENDS == ("fake", "x11", "win32", "quartz")


def test_to_json_backends_shape() -> None:
    """``to_json('backends')`` is a single-line, parseable JSON object."""
    out = _screen.to_json("backends")
    assert "\n" not in out
    parsed = json.loads(out)
    assert parsed == {"backends": ["fake", "x11", "win32", "quartz"]}


def test_to_json_unknown_kind_is_error() -> None:
    """Only ``'backends'`` is supported today; other kinds are programmer errors."""
    with pytest.raises(ValueError):
        _screen.to_json("captures")


def test_backend_unsupported_message_mentions_backend_name() -> None:
    """The stderr message names the backend so the user knows which one failed."""
    msg = _screen.backend_unsupported_message("x11")
    assert "x11" in msg
    assert "fake" in msg  # points the user at the workaround


# ---------------------------------------------------------------------------
# read_fake_grid (the CLI helper that turns --fake-grid TEXT into list[str])
# ---------------------------------------------------------------------------


def test_read_fake_grid_basic() -> None:
    """A simple multi-line string becomes a list of rows, with the
    trailing newline dropped (so ``'a\\nb\\n'`` is 2 rows, not 3)."""
    assert _screen._read_fake_grid("abc\ndef\n") == ["abc", "def"]


def test_read_fake_grid_no_trailing_newline() -> None:
    """A string without a trailing newline still produces its rows."""
    assert _screen._read_fake_grid("abc\ndef") == ["abc", "def"]


def test_read_fake_grid_empty_is_error() -> None:
    """Empty input is a usage error — a FakeScreen with no rows is meaningless."""
    with pytest.raises(ValueError):
        _screen._read_fake_grid("")
    with pytest.raises(ValueError):
        _screen._read_fake_grid("\n\n\n")


# ---------------------------------------------------------------------------
# parse_args + main() end-to-end (via the public zoom CLI)
# ---------------------------------------------------------------------------


def test_parse_args_screen_flag_default_off() -> None:
    """``--screen`` defaults to off (text-source mode is the v0.1 default)."""
    args = zoom.parse_args(["hello"])
    assert args.screen is False


def test_parse_args_screen_flag_parses() -> None:
    """``--screen`` flips the flag on."""
    args = zoom.parse_args(
        [
            "--screen",
            "--backend",
            "fake",
            "--fake-grid",
            "abc",
        ]
    )
    assert args.screen is True
    assert args.backend == "fake"
    assert args.fake_grid == "abc"


def test_parse_args_fake_grid_without_fake_backend_is_usage_error(
    capsys: pytest.CaptureFixture,
) -> None:
    """``--fake-grid`` without ``--backend fake`` is rejected at parse time."""
    with pytest.raises(SystemExit) as exc_info:
        zoom.parse_args(["--fake-grid", "abc"])
    assert exc_info.value.code == 2
    assert "--fake-grid requires --backend fake" in capsys.readouterr().err


def test_parse_args_region_without_screen_is_usage_error(
    capsys: pytest.CaptureFixture,
) -> None:
    """``--region`` without ``--screen`` is rejected at parse time."""
    with pytest.raises(SystemExit) as exc_info:
        zoom.parse_args(["--region", "0,0,10,5"])
    assert exc_info.value.code == 2
    assert "--region requires --screen" in capsys.readouterr().err


def test_parse_args_json_without_list_backends_is_usage_error(
    capsys: pytest.CaptureFixture,
) -> None:
    """``--json`` without ``--list-backends`` is rejected at parse time."""
    with pytest.raises(SystemExit) as exc_info:
        zoom.parse_args(["--json"])
    assert exc_info.value.code == 2
    assert "--json requires --list-backends" in capsys.readouterr().err


def test_main_list_backends_prints_one_per_line(
    capsys: pytest.CaptureFixture,
) -> None:
    """``paw-zoom --list-backends`` prints the canonical list, one per line."""
    rc = zoom.main(["--list-backends"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.strip().splitlines() == ["fake", "x11", "win32", "quartz"]


def test_main_list_backends_json_emits_parseable_object(
    capsys: pytest.CaptureFixture,
) -> None:
    """``paw-zoom --list-backends --json`` emits a parseable JSON object
    on its own line. The ``to_json()`` helper itself is single-line;
    the trailing newline comes from ``print()`` (POSIX convention)."""
    rc = zoom.main(["--list-backends", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    # Strip the single trailing newline that print() adds, then
    # assert the body is single-line and parseable.
    body = out.rstrip("\n")
    assert "\n" not in body
    assert json.loads(body) == {"backends": ["fake", "x11", "win32", "quartz"]}


def test_main_screen_fake_backend_renders_magnified_grid(
    capsys: pytest.CaptureFixture,
) -> None:
    """End-to-end: ``--screen --backend fake --fake-grid ...`` produces
    the magnified viewport of the fake screen."""
    rc = zoom.main(
        [
            "--screen",
            "--backend",
            "fake",
            "--fake-grid",
            "abc\ndef",
            "--rows",
            "2",
            "--cols",
            "3",
            "--zoom",
            "1",
            "--quiet",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    # zoom=1, so the output is the captured grid verbatim.
    assert "abc" in out
    assert "def" in out


def test_main_screen_with_region_uses_sub_grid(
    capsys: pytest.CaptureFixture,
) -> None:
    """``--region X,Y,W,H`` clips the captured sub-grid before magnification."""
    rc = zoom.main(
        [
            "--screen",
            "--backend",
            "fake",
            "--fake-grid",
            "abcdef\nghijkl\nmnopqr",
            "--region",
            "0,1,3,2",
            "--rows",
            "2",
            "--cols",
            "3",
            "--zoom",
            "1",
            "--quiet",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    # Row 0 of the fake screen is "abcdef"; row 1 is "ghijkl" → "ghi".
    assert "ghi" in out
    assert "abc" not in out  # row 0 was clipped out by --region
    assert "mno" in out  # row 2 of the fake screen is "mnopqr" → "mno"


def test_main_screen_unsupported_backend_exits_1(
    capsys: pytest.CaptureFixture,
) -> None:
    """``--screen --backend x11`` on a headless box exits 1 with a
    helpful stderr message."""
    rc = zoom.main(["--screen", "--backend", "x11"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "x11" in err
    assert "fake" in err  # the workaround


def test_main_screen_unsupported_backend_auto_also_exits_1(
    capsys: pytest.CaptureFixture,
) -> None:
    """``--screen`` (backend defaults to 'auto') on a headless box also exits 1."""
    rc = zoom.main(["--screen"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "not yet implemented" in err


def test_main_screen_composes_with_snapshot(tmp_path) -> None:
    """``--screen --snapshot PATH`` writes the captured-then-magnified
    viewport to PATH (no stdout rendering)."""
    snap = tmp_path / "out.txt"
    rc = zoom.main(
        [
            "--screen",
            "--backend",
            "fake",
            "--fake-grid",
            "abc\ndef",
            "--rows",
            "2",
            "--cols",
            "3",
            "--zoom",
            "1",
            "--quiet",
            "--snapshot",
            str(snap),
        ]
    )
    assert rc == 0
    text = snap.read_text(encoding="utf-8")
    assert "abc" in text
    assert "def" in text


def test_main_screen_without_fake_grid_uses_caller_supplied(
    capsys: pytest.CaptureFixture,
) -> None:
    """When ``--backend auto`` is combined with a ``fake_grid`` kwarg
    (only possible via the Python API, not the CLI), the fake adapter
    is picked. The CLI doesn't expose this — but the test pins the
    factory's behaviour so the next tick can build on it."""
    cap = _screen.get_capture("auto", fake_grid=["hi\nmom"])
    assert isinstance(cap, _screen.FakeScreen)
    # Round-trip through the public helper.
    assert _screen.capture_screen_to_source(cap) == "hi\nmom"


# ---------------------------------------------------------------------------
# CLI smoke: the entry point works with the real shell-installed shim
# ---------------------------------------------------------------------------


def test_cli_shim_list_backends() -> None:
    """The installed ``paw-zoom`` console-script shim handles the new
    discovery flag the same way as ``python -m whisperpaw.zoom`` does.
    This catches any divergence between the entry-point shim and the
    module's ``main()``."""
    venv_bin = os.path.dirname(sys.executable)
    shim = os.path.join(venv_bin, "paw-zoom")
    if not os.path.isfile(shim):
        pytest.skip("paw-zoom shim not installed in this venv")
    result = subprocess.run(
        [shim, "--list-backends"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.strip().splitlines() == [
        "fake",
        "x11",
        "win32",
        "quartz",
    ]


def test_cli_shim_screen_fake_renders() -> None:
    """End-to-end via the installed shim: ``paw-zoom --screen --backend
    fake --fake-grid ...`` produces the expected magnified output."""
    venv_bin = os.path.dirname(sys.executable)
    shim = os.path.join(venv_bin, "paw-zoom")
    if not os.path.isfile(shim):
        pytest.skip("paw-zoom shim not installed in this venv")
    result = subprocess.run(
        [
            shim,
            "--screen",
            "--backend",
            "fake",
            "--fake-grid",
            "abc\ndef",
            "--rows",
            "2",
            "--cols",
            "3",
            "--zoom",
            "1",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "abc" in result.stdout
    assert "def" in result.stdout


# ---------------------------------------------------------------------------
# --screen --live (the screen-capture tail)
# ---------------------------------------------------------------------------


def test_main_screen_live_emits_first_frame(capsys: pytest.CaptureFixture) -> None:
    """``--screen --live --backend fake --fake-grid ...`` emits the
    first magnified frame on the first poll and exits cleanly when
    --max-frames caps the loop.

    This is the FakeScreen end-to-end test the v0.2 Quartz tick's
    bookkeeping promised but didn't land: the captured screen is
    re-rendered through the full live-tail pipeline, with
    --max-frames bounding the run so the test doesn't loop forever.
    """
    rc = zoom.main(
        [
            "--screen",
            "--backend",
            "fake",
            "--fake-grid",
            "abc\ndef",
            "--live",
            "--max-frames",
            "1",
            "--rows",
            "2",
            "--cols",
            "3",
            "--zoom",
            "1",
            "--quiet",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    # The first frame is the captured grid at zoom 1.
    assert "abc" in out
    assert "def" in out


def test_main_screen_live_without_max_frames_does_not_loop_forever(
    capsys: pytest.CaptureFixture,
) -> None:
    """Without --max-frames, the ``--screen --live`` loop only runs
    for one tick in this test because we inject a clock that
    always returns and a stop_predicate that fires after one
    iteration. The real CLI requires Ctrl-C; this test pins the
    plumbing path that the CLI dispatches to, by going through
    the helper directly.

    This is the end-to-end "loop shape" guarantee: the
    :func:`whisperpaw.zoom._tail_screen_and_render` helper, when
    called with a one-iteration stop_predicate, exits cleanly
    with exactly one frame in the sink.
    """
    cap = _screen.FakeScreen(["abc", "def"])
    cfg = zoom.ZoomConfig(rows=2, cols=3, zoom=1)
    frames: list[str] = []
    ticks = {"n": 0}

    def stop() -> bool:
        ticks["n"] += 1
        return ticks["n"] >= 2

    rc = zoom._tail_screen_and_render(
        cap,
        cfg,
        interval=0.0,
        stop_predicate=stop,
        clock=lambda _x: None,
        sink=frames.append,
    )
    assert rc == 0
    assert len(frames) == 1
    assert "abc" in frames[0]
    assert "def" in frames[0]


def test_tail_screen_and_render_emits_frame_on_grid_change() -> None:
    """If the FakeScreen's grid changes between polls, the
    helper re-renders. This is the proof that the screen-tail
    composition is real: a fresh ``_grid`` between iterations
    surfaces a string-diff in the change detector and yields a
    second frame.
    """
    cap = _screen.FakeScreen(["aaa", "bbb"])
    cfg = zoom.ZoomConfig(rows=2, cols=3, zoom=1)
    frames: list[str] = []
    ticks = {"n": 0}

    def stop() -> bool:
        ticks["n"] += 1
        if ticks["n"] == 2:
            # Mutate the captured grid between polls.
            cap._grid = ["xxx", "yyy"]  # type: ignore[attr-defined]
        return ticks["n"] >= 3

    zoom._tail_screen_and_render(
        cap,
        cfg,
        interval=0.0,
        stop_predicate=stop,
        clock=lambda _x: None,
        sink=frames.append,
    )
    # Two distinct frames: the original "aaa/bbb" and the mutated
    # "xxx/yyy".
    assert len(frames) == 2
    assert "aaa" in frames[0]
    assert "xxx" in frames[1]
    assert "aaa" not in frames[1]


def test_tail_screen_and_render_skips_unchanged_polls() -> None:
    """If the captured grid is the same across many polls, only
    one frame is rendered. Mirrors the text-source path's
    change-detector contract: a static screen yields exactly
    one frame (the initial state).
    """
    cap = _screen.FakeScreen(["aaa", "bbb"])
    cfg = zoom.ZoomConfig(rows=2, cols=3, zoom=1)
    frames: list[str] = []
    ticks = {"n": 0}

    def stop() -> bool:
        ticks["n"] += 1
        return ticks["n"] >= 5

    zoom._tail_screen_and_render(
        cap,
        cfg,
        interval=0.0,
        stop_predicate=stop,
        clock=lambda _x: None,
        sink=frames.append,
    )
    assert len(frames) == 1


def test_tail_screen_and_render_handles_capture_errors() -> None:
    """If the injected ``capture_fn`` raises ``ValueError`` (region
    shape error) or ``RuntimeError`` (adapter-level failure),
    the helper prints a stderr line and keeps looping. The
    loop must not crash on transient adapter failures — a
    screen magnifier that has been running for hours should
    survive a momentary blip.
    """
    cfg = zoom.ZoomConfig(rows=2, cols=3, zoom=1)
    frames: list[str] = []
    ticks = {"n": 0}
    call_log: list[int] = []

    def flaky_capture() -> str:
        call_log.append(1)
        n = sum(call_log)
        if n == 1:
            raise ValueError("region too small")
        if n == 2:
            raise RuntimeError("adapter unavailable")
        return "abc\ndef"  # third call succeeds

    def stop() -> bool:
        ticks["n"] += 1
        # stop_predicate is consulted at the top of each iteration.
        # Three calls means we need the predicate to fire on the
        # *fourth* tick: 1=ValueError, 2=RuntimeError, 3=success,
        # 4=stop.
        return ticks["n"] >= 4

    rc = zoom._tail_screen_and_render(
        _screen.FakeScreen(["placeholder"]),  # adapter is unused (we inject)
        cfg,
        interval=0.0,
        stop_predicate=stop,
        clock=lambda _x: None,
        sink=frames.append,
        capture_fn=flaky_capture,
    )
    assert rc == 0
    # Only the third call (the successful one) yields a frame.
    assert len(frames) == 1
    assert "abc" in frames[0]


def test_tail_screen_and_render_follow_keeps_tail_in_view() -> None:
    """``--follow`` on the screen-capture path keeps the *last*
    ``cfg.rows`` lines of the captured source in view, the
    same way the text-source ``--follow`` does. The frozen
    config is left untouched; the per-frame offset is a
    derived value.
    """
    cap = _screen.FakeScreen(["L1", "L2", "L3", "L4", "L5"])
    cfg = zoom.ZoomConfig(rows=2, cols=2, zoom=1)
    frames: list[str] = []
    ticks = {"n": 0}

    def stop() -> bool:
        ticks["n"] += 1
        return ticks["n"] >= 2

    zoom._tail_screen_and_render(
        cap,
        cfg,
        interval=0.0,
        follow=True,
        stop_predicate=stop,
        clock=lambda _x: None,
        sink=frames.append,
    )
    # cfg.rows=2, the source has 5 lines → tail is lines 3-5, but
    # cols=2 so we only see the first 2 chars of each: "L4" and "L5".
    assert len(frames) == 1
    assert "L4" in frames[0]
    assert "L5" in frames[0]
    assert "L1" not in frames[0]  # --follow keeps the head off the viewport
    # The original (frozen) config is untouched.
    assert cfg.row_offset == 0


def test_tail_screen_and_render_max_frames_caps_iterations() -> None:
    """``--max-frames`` on the screen-capture path caps the loop
    at the requested number of *iterations*, not emitted
    frames. Same semantics as the text-source path.
    """
    cap = _screen.FakeScreen(["aaa", "bbb"])
    cfg = zoom.ZoomConfig(rows=2, cols=3, zoom=1)
    frames: list[str] = []
    ticks = {"n": 0}

    def stop() -> bool:
        ticks["n"] += 1
        # Change the grid every poll so the change detector
        # would keep emitting.
        cap._grid = ["xxx", "yyy"] if ticks["n"] >= 2 else ["aaa", "bbb"]  # type: ignore[attr-defined]
        return ticks["n"] >= 5

    rc = zoom._tail_screen_and_render(
        cap,
        cfg,
        interval=0.0,
        max_frames=2,
        stop_predicate=stop,
        clock=lambda _x: None,
        sink=frames.append,
    )
    assert rc == 0
    # Two iterations, two distinct frames (we mutated on tick 2).
    assert len(frames) == 2
    assert "aaa" in frames[0]
    assert "xxx" in frames[1]


def test_tail_screen_and_render_uses_capture_fn_when_provided() -> None:
    """When ``capture_fn`` is injected, the adapter (``cap``) is
    never touched. The injection point exists so tests can run
    the screen-tail loop with a fake source without a real
    adapter — and so callers can pre-derive a string from
    their own pipeline (e.g. a logged tty) without going
    through the ScreenCapture protocol.
    """
    cfg = zoom.ZoomConfig(rows=2, cols=3, zoom=1)
    frames: list[str] = []
    ticks = {"n": 0}
    adapter_used: list[bool] = []

    class TrackingAdapter(_screen.ScreenCapture):
        """Adapter that records whether it was ever called."""

        def screen_size(self) -> tuple[int, int]:
            adapter_used.append(True)
            return (3, 2)

        def capture(self, *, x: int, y: int, w: int, h: int) -> list[str]:
            adapter_used.append(True)
            return ["should not be called"]

    def stop() -> bool:
        ticks["n"] += 1
        return ticks["n"] >= 2

    zoom._tail_screen_and_render(
        TrackingAdapter(),
        cfg,
        interval=0.0,
        stop_predicate=stop,
        clock=lambda _x: None,
        sink=frames.append,
        capture_fn=lambda: "abc\ndef",
    )
    assert adapter_used == []  # the adapter was never queried
    assert len(frames) == 1
    assert "abc" in frames[0]


def test_tail_screen_and_render_has_changed_injection() -> None:
    """When ``has_changed`` is injected, it overrides the default
    ``!=`` check. This is the knob real OS adapters could use
    to say "always re-render" (e.g. if they can't cheaply diff
    their pixel grid) or "never re-render after the first"
    (e.g. a one-shot magnifier).
    """
    cap = _screen.FakeScreen(["aaa", "bbb"])
    cfg = zoom.ZoomConfig(rows=2, cols=3, zoom=1)
    frames: list[str] = []
    ticks = {"n": 0}

    def stop() -> bool:
        ticks["n"] += 1
        return ticks["n"] >= 4

    # Always re-render: even though the grid is static, the
    # override forces a frame on every poll.
    zoom._tail_screen_and_render(
        cap,
        cfg,
        interval=0.0,
        stop_predicate=stop,
        clock=lambda _x: None,
        sink=frames.append,
        has_changed=lambda _prev, _cur: True,
    )
    assert len(frames) == 3  # three polls, all forced to render


def test_main_screen_live_with_snapshot_writes_each_frame(
    tmp_path, capsys
) -> None:
    """``--screen --live --snapshot PATH``: each frame is written
    to PATH (overwriting). On a FakeScreen whose grid is
    static, only the first frame is written; the test pins
    that the snapshot file actually contains the magnified
    viewport.
    """
    snap = tmp_path / "out.txt"
    rc = zoom.main(
        [
            "--screen",
            "--backend",
            "fake",
            "--fake-grid",
            "abc\ndef",
            "--live",
            "--max-frames",
            "1",
            "--rows",
            "2",
            "--cols",
            "3",
            "--zoom",
            "1",
            "--snapshot",
            str(snap),
            "--quiet",
        ]
    )
    assert rc == 0
    text = snap.read_text(encoding="utf-8")
    assert "abc" in text
    assert "def" in text


def test_main_screen_live_unsupported_backend_exits_1(
    capsys: pytest.CaptureFixture,
) -> None:
    """``--screen --live --backend x11`` on a headless box exits 1
    with a helpful stderr message. ``--live`` doesn't change
    the backend-resolution path; the ``--screen`` source
    resolution still early-returns when no OS adapter is
    available.
    """
    rc = zoom.main(["--screen", "--live", "--backend", "x11"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "x11" in err
    assert "fake" in err  # the workaround


def test_main_screen_live_follow_tracks_screen_tail(
    capsys: pytest.CaptureFixture,
) -> None:
    """``--screen --live --follow --backend fake --fake-grid ...``
    shows the *last* ``--rows`` lines of the captured grid,
    not the first. The FakeScreen's grid is fixed in this
    test, so the tail doesn't change — but the viewport
    composition must still apply.
    """
    rc = zoom.main(
        [
            "--screen",
            "--backend",
            "fake",
            "--fake-grid",
            "L1\nL2\nL3\nL4",
            "--live",
            "--follow",
            "--max-frames",
            "1",
            "--rows",
            "2",
            "--cols",
            "2",
            "--zoom",
            "1",
            "--quiet",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    # tail of the 4-line fake screen with rows=2: L3, L4 → "L3", "L4"
    assert "L3" in out
    assert "L4" in out
    assert "L1" not in out  # --follow keeps the head off the viewport


def test_main_live_without_file_or_screen_is_usage_error(
    capsys: pytest.CaptureFixture,
) -> None:
    """``--live`` without ``--file`` *or* ``--screen`` is a usage
    error (exit 2). The previous behaviour was
    ``--live --file`` only; the screen path is now also
    accepted, but neither is a hard error."""
    with pytest.raises(SystemExit) as exc_info:
        zoom.parse_args(["--live"])
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "--live requires --file PATH or --screen" in err


def test_parse_args_live_screen_no_file_parses() -> None:
    """``--live --screen`` (no ``--file``) parses cleanly: the
    screen-capture tail doesn't need a file to tail. This is
    the new behaviour the tick just unlocked."""
    args = zoom.parse_args(
        [
            "--screen",
            "--backend",
            "fake",
            "--fake-grid",
            "abc\ndef",
            "--live",
        ]
    )
    assert args.live is True
    assert args.screen is True
    assert args.file is None
    assert args.backend == "fake"


def test_main_screen_live_max_frames_bounds_run(
    capsys: pytest.CaptureFixture,
) -> None:
    """``--screen --live --max-frames N`` actually exits after N
    iterations, even when the FakeScreen's grid is static.
    End-to-end through ``main()`` — proves the
    :func:`_tail_screen_and_render` dispatch wired by the
    tick is reachable from the CLI and that ``--max-frames``
    flows through.
    """
    rc = zoom.main(
        [
            "--screen",
            "--backend",
            "fake",
            "--fake-grid",
            "abc\ndef",
            "--live",
            "--max-frames",
            "3",
            "--rows",
            "2",
            "--cols",
            "3",
            "--zoom",
            "1",
            "--interval",
            "0.001",
            "--quiet",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    # Only one frame is actually emitted (static FakeScreen
    # grid → change detector fires once, then idles), but the
    # loop must still have terminated cleanly with rc=0.
    assert "abc" in out


# ---------------------------------------------------------------------------
# describe_capture / describe_to_text / describe_to_json
# ---------------------------------------------------------------------------


def _fake_capture_factory(grid: list[str] | None):
    """Build a ``get_capture_fn``-shaped callable that returns
    a ``FakeScreen(grid)`` for ``backend='fake'`` and ``None``
    for any other backend. Mirrors the ``fake_grid`` semantics
    of the real factory.
    """
    def _factory(backend: str, *, fake_grid: list[str] | None = None):
        if backend == "fake":
            if fake_grid is None:
                return None
            return _screen.FakeScreen(fake_grid)
        return None
    return _factory


def test_describe_capture_fake_with_grid_available() -> None:
    """``describe_capture('fake', fake_grid=['abc', 'def'])``
    reports the adapter as available, names ``FakeScreen``,
    reports the screen size as (3, 2), and computes a full-
    screen region as (0, 0, 3, 2).
    """
    info = _screen.describe_capture(
        "fake",
        fake_grid=["abc", "def"],
        get_capture_fn=_fake_capture_factory(["abc", "def"]),
    )
    assert info == {
        "backend": "fake",
        "available": True,
        "adapter": "FakeScreen",
        "screen_size": [3, 2],
        "region": [0, 0, 3, 2],
    }


def test_describe_capture_fake_without_grid_unavailable() -> None:
    """A ``fake`` backend with no grid is reported as
    unavailable: no adapter could be constructed, so the dict
    stays empty on the size / region fields.
    """
    info = _screen.describe_capture(
        "fake",
        get_capture_fn=_fake_capture_factory(None),
    )
    assert info == {
        "backend": "fake",
        "available": False,
        "adapter": None,
        "screen_size": None,
        "region": None,
    }


def test_describe_capture_unknown_backend_via_factory_error() -> None:
    """A factory that raises ``ValueError`` (e.g. an unknown
    backend or a missing fake_grid for ``fake``) is reported as
    unavailable rather than propagated — ``describe_capture``
    is a diagnostic, not a validator.
    """
    def _raising_factory(backend, *, fake_grid=None):
        raise ValueError("nope")

    info = _screen.describe_capture("x11", get_capture_fn=_raising_factory)
    assert info["available"] is False
    assert info["adapter"] is None
    assert info["screen_size"] is None
    assert info["region"] is None
    assert info["backend"] == "x11"


def test_describe_capture_os_backend_returns_none() -> None:
    """``describe_capture('x11', ...)`` on a headless box (the
    factory returns ``None``) reports the backend as unavailable
    and leaves the size / region fields empty.
    """
    info = _screen.describe_capture(
        "x11",
        get_capture_fn=lambda backend, *, fake_grid=None: None,
    )
    assert info["backend"] == "x11"
    assert info["available"] is False
    assert info["adapter"] is None
    assert info["screen_size"] is None
    assert info["region"] is None


def test_describe_capture_with_explicit_region() -> None:
    """An explicit ``--region`` is parsed and clamped to the
    reported screen size.
    """
    info = _screen.describe_capture(
        "fake",
        region="0,0,2,1",
        fake_grid=["abc", "def"],
        get_capture_fn=_fake_capture_factory(["abc", "def"]),
    )
    assert info["available"] is True
    assert info["screen_size"] == [3, 2]
    assert info["region"] == [0, 0, 2, 1]


def test_describe_capture_with_region_string_full() -> None:
    """``--region full`` (case-insensitive) means the whole screen.
    """
    info = _screen.describe_capture(
        "fake",
        region="FULL",
        fake_grid=["abcdef", "ghijkl"],
        get_capture_fn=_fake_capture_factory(["abcdef", "ghijkl"]),
    )
    assert info["region"] == [0, 0, 6, 2]


def test_describe_capture_with_region_string_out_of_range_clamps() -> None:
    """A region past the bottom-right edge is clamped to the screen.
    """
    info = _screen.describe_capture(
        "fake",
        region="2,1,99,99",
        fake_grid=["abc", "def"],
        get_capture_fn=_fake_capture_factory(["abc", "def"]),
    )
    assert info["screen_size"] == [3, 2]
    assert info["region"] == [2, 1, 1, 1]


def test_describe_capture_adapter_screen_size_failure_tolerated() -> None:
    """An adapter that constructs but blows up in
    ``screen_size()`` is still reported as available — the
    exception is swallowed so the diagnostic still tells the
    user *which* adapter was picked, and only the size field
    is left empty.
    """
    class _BrokenAdapter(_screen.ScreenCapture):
        def screen_size(self):
            raise RuntimeError("nope")

        def capture(self, *, x, y, w, h):
            return []

    def _factory(backend, *, fake_grid=None):
        if backend == "fake":
            return _BrokenAdapter()
        return None

    info = _screen.describe_capture("fake", get_capture_fn=_factory)
    assert info["available"] is True
    assert info["adapter"] == "_BrokenAdapter"
    assert info["screen_size"] is None
    assert info["region"] is None


def test_describe_capture_region_parsing_error_keeps_dict_alive() -> None:
    """A region the parser would normally reject (e.g. a
    negative component) does not crash the diagnostic — the
    region field is left as ``None`` and the rest of the
    dict still answers the user's question.
    """
    info = _screen.describe_capture(
        "fake",
        region="0,0,-1,1",
        fake_grid=["abc", "def"],
        get_capture_fn=_fake_capture_factory(["abc", "def"]),
    )
    assert info["available"] is True
    assert info["screen_size"] == [3, 2]
    assert info["region"] is None


def test_describe_to_text_format() -> None:
    """The text output is five ``key: value`` lines in a fixed
    order, with ``screen_size`` formatted as ``WxH`` and
    ``region`` as ``X,Y,W,H``.
    """
    info = {
        "backend": "fake",
        "available": True,
        "adapter": "FakeScreen",
        "screen_size": [80, 24],
        "region": [0, 0, 80, 24],
    }
    text = _screen.describe_to_text(info)
    assert text == "\n".join(
        [
            "backend: fake",
            "available: yes",
            "adapter: FakeScreen",
            "screen_size: 80x24",
            "region: 0,0,80,24",
        ]
    )


def test_describe_to_text_unavailable_uses_dashes() -> None:
    """When a field is missing (e.g. an unavailable OS adapter
    that never reports a size), the text output uses ``-`` so
    the line layout is still predictable for a downstream
    grep / awk.
    """
    info = {
        "backend": "x11",
        "available": False,
        "adapter": None,
        "screen_size": None,
        "region": None,
    }
    text = _screen.describe_to_text(info)
    assert text == "\n".join(
        [
            "backend: x11",
            "available: no",
            "adapter: -",
            "screen_size: -",
            "region: -",
        ]
    )


def test_describe_to_json_round_trip() -> None:
    """The JSON output round-trips through ``json.loads`` and
    exposes the screen_size / region as JSON arrays.
    """
    info = {
        "backend": "fake",
        "available": True,
        "adapter": "FakeScreen",
        "screen_size": [3, 2],
        "region": [0, 0, 3, 2],
    }
    raw = _screen.describe_to_json(info)
    assert "\n" not in raw
    parsed = json.loads(raw)
    assert parsed == info


def test_describe_to_json_with_none_fields() -> None:
    """The JSON output preserves ``None`` fields as ``null``
    so a downstream consumer can tell the adapter was queried
    but didn't return a value.
    """
    info = {
        "backend": "x11",
        "available": False,
        "adapter": None,
        "screen_size": None,
        "region": None,
    }
    parsed = json.loads(_screen.describe_to_json(info))
    assert parsed["screen_size"] is None
    assert parsed["region"] is None
    assert parsed["available"] is False


def test_describe_capture_default_get_capture_is_real(monkeypatch) -> None:
    """With no ``get_capture_fn`` override, ``describe_capture``
    uses the real module-level :func:`get_capture`. This
    pins down the contract: on a headless box the real
    factory returns ``None`` for OS backends, and the dict
    reflects that.
    """
    info = _screen.describe_capture("x11")
    assert info["backend"] == "x11"
    # No $DISPLAY on this CI box → no X11 adapter → unavailable.
    assert info["available"] is False


# ---------------------------------------------------------------------------
# paw-zoom --info (CLI integration)
# ---------------------------------------------------------------------------


def test_main_info_requires_screen(capsys) -> None:
    """``paw-zoom --info`` without ``--screen`` is a usage
    error (exit 2 with a clear stderr message).
    """
    rc = zoom.main(["--info"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "--info requires --screen" in captured.err


def test_main_info_text_mode_with_fake_backend(capsys) -> None:
    """``paw-zoom --info --screen --backend fake --fake-grid
    'abc\\ndef'`` prints the five-line text description and
    exits 0 without any screen capture or rendering.
    """
    rc = zoom.main(
        [
            "--info",
            "--screen",
            "--backend",
            "fake",
            "--fake-grid",
            "abc\ndef",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    assert "backend: fake" in out.out
    assert "available: yes" in out.out
    assert "adapter: FakeScreen" in out.out
    assert "screen_size: 3x2" in out.out
    assert "region: 0,0,3,2" in out.out


def test_main_info_json_mode(capsys) -> None:
    """``paw-zoom --info --json --screen --backend fake
    --fake-grid 'abc\\ndef'`` prints a single-line parseable
    JSON object with the same five fields.
    """
    rc = zoom.main(
        [
            "--info",
            "--json",
            "--screen",
            "--backend",
            "fake",
            "--fake-grid",
            "abc\ndef",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    assert "\n" not in out.out.rstrip("\n")
    parsed = json.loads(out.out)
    assert parsed == {
        "backend": "fake",
        "available": True,
        "adapter": "FakeScreen",
        "screen_size": [3, 2],
        "region": [0, 0, 3, 2],
    }


def test_main_info_unavailable_backend_does_not_exit_1(capsys) -> None:
    """``paw-zoom --info --screen --backend x11`` on a headless
    box reports ``available: no`` and exits 0 — the whole
    point of ``--info`` is "what would happen?", not "do
    the thing".
    """
    rc = zoom.main(["--info", "--screen", "--backend", "x11"])
    out = capsys.readouterr()
    assert rc == 0
    assert "backend: x11" in out.out
    assert "available: no" in out.out
    assert "adapter: -" in out.out
    assert "screen_size: -" in out.out


def test_main_info_with_explicit_region(capsys) -> None:
    """``--info --region X,Y,W,H`` resolves the region and
    shows it in the output.
    """
    rc = zoom.main(
        [
            "--info",
            "--screen",
            "--backend",
            "fake",
            "--fake-grid",
            "abc\ndef",
            "--region",
            "1,0,2,1",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    assert "region: 1,0,2,1" in out.out


def test_main_info_mutual_exclusion_with_live(capsys) -> None:
    """``--info --live`` is rejected (exit 2): --info is a
    metadata-only mode, --live is a render driver.
    """
    rc = zoom.main(
        [
            "--info",
            "--screen",
            "--live",
            "--max-frames",
            "1",
            "--interval",
            "0.001",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "--info cannot be combined with --live" in captured.err


def test_main_info_mutual_exclusion_with_raw(capsys) -> None:
    """``--info --raw`` is rejected (exit 2). Same rationale.
    """
    rc = zoom.main(["--info", "--screen", "--raw"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "--info cannot be combined with --raw" in captured.err


def test_main_info_mutual_exclusion_with_snapshot(capsys) -> None:
    """``--info --snapshot`` is rejected (exit 2).
    """
    rc = zoom.main(
        ["--info", "--screen", "--snapshot", "/tmp/should_not_be_written.txt"]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "--info cannot be combined with --snapshot" in captured.err
    # Sanity: the snapshot file must NOT have been written.
    import os
    assert not os.path.exists("/tmp/should_not_be_written.txt")


def test_main_info_json_without_discovery_flag_is_usage_error(capsys) -> None:
    """``--json`` without ``--list-backends`` / ``--info`` / ``--size``
    / ``--stats`` / ``--sha`` is a usage error (exit 2) — the same
    fail-fast the ``--list-backends --json`` combo used to do.
    """
    rc = zoom.main(["--json"])
    captured = capsys.readouterr()
    assert rc == 2
    assert (
        "--json requires --list-backends, --info, --size, "
        "--stats, or --sha" in captured.err
    )


def test_main_info_with_malformed_fake_grid_is_usage_error(capsys) -> None:
    """``--info --fake-grid ''`` (an empty fake grid) is
    rejected at the parse-fake-grid boundary with a clear
    message, not silently reported as ``available: no``.
    """
    rc = zoom.main(
        [
            "--info",
            "--screen",
            "--backend",
            "fake",
            "--fake-grid",
            "",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "--fake-grid" in captured.err


def test_main_info_help_text_mentions_flag(capsys) -> None:
    """The ``--help`` text mentions ``--info`` so a casual
    ``paw-zoom --help`` user discovers it. Catches accidental
    renames.
    """
    rc = zoom.main(["--help"])
    out = capsys.readouterr()
    assert rc == 0
    assert "--info" in out.out


def test_parse_args_info_default_is_false() -> None:
    """``--info`` defaults to off (the existing behaviour is
    unchanged unless the flag is passed).
    """
    args = zoom.parse_args(["hello"])
    assert args.info is False


def test_parse_args_info_flag() -> None:
    """``--info`` parses to ``True`` and composes with
    ``--screen`` / ``--backend`` / ``--region``.
    """
    args = zoom.parse_args(
        ["--info", "--screen", "--backend", "fake", "--region", "0,0,10,5"]
    )
    assert args.info is True
    assert args.screen is True
    assert args.backend == "fake"
    assert args.region == "0,0,10,5"
