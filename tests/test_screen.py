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
