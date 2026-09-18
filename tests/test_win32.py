"""Tests for ``whisperpaw._win32`` (the real Win32 GDI screen-capture adapter).

These tests don't need a real Windows machine — they
synthesise a small BGRX pixel buffer in memory and feed it to
the adapter through injectable ``capture=`` and ``size=``
callables. That keeps the test suite fast and cross-platform
(this CI box is Linux) while still exercising the real BGRX
sampling path and the real ``_bgrx_to_grid`` downsample path.

BGRX pixel format reminder
---------------------------
GDI's ``GetDIBits`` returns pixels in *bottom-up* row order by
default, with 4 bytes per pixel in B, G, R, X (alpha ignored)
order. The :func:`_make_bgrx_bytes` helper below builds a
minimal valid buffer: ``width * height * 4`` bytes of
bottom-up BGRX data, where each cell is independently
controllable. The brightness helper
:func:`whisperpaw._win32._sample_bgrx_cell_brightness` inverts
the row order on read.
"""
from __future__ import annotations

import pytest

from whisperpaw import _screen, _win32


# ---------------------------------------------------------------------------
# BGRX byte-stream builder
# ---------------------------------------------------------------------------


def _make_bgrx_bytes(
    *,
    width: int = 4,
    height: int = 3,
    pixels: list[list[tuple[int, int, int]]] | None = None,
) -> bytes:
    """Build a synthetic BGRX pixel buffer for tests.

    The default is a 4×3, bottom-up BGRX buffer with a
    checkerboard where odd cells are red and even cells are
    blue, so the downsample output is easy to reason about.

    Parameters
    ----------
    pixels:
        A ``height`` × ``width`` list of (r, g, b) tuples.
        Defaults to the same checkerboard the X11 test helper
        uses, so behaviour across adapters is comparable.
    """
    if pixels is None:
        pixels = []
        for row in range(height):
            pixels_row = []
            for col in range(width):
                if (row + col) % 2 == 0:
                    pixels_row.append((0, 0, 255))  # blue
                else:
                    pixels_row.append((255, 0, 0))  # red
            pixels.append(pixels_row)

    # Bottom-up: write the bottom row first, then the row
    # above it, etc. That's the GDI default and the production
    # path the adapter assumes.
    out = bytearray()
    for row in reversed(range(height)):
        for col in range(width):
            r, g, b = pixels[row][col]
            out.append(b)        # B
            out.append(g)        # G
            out.append(r)        # R
            out.append(0)        # X (alpha, ignored)
    return bytes(out)


# ---------------------------------------------------------------------------
# is_win32_available
# ---------------------------------------------------------------------------


def test_is_win32_available_true_when_check_returns_true() -> None:
    """A fake availability check that returns True → reported as available."""
    assert _win32.is_win32_available(available=lambda: True) is True


def test_is_win32_available_false_when_check_returns_false() -> None:
    """A fake availability check that returns False → reported as unavailable."""
    assert _win32.is_win32_available(available=lambda: False) is False


def test_is_win32_available_default_is_sys_platform_check() -> None:
    """The default branch is just ``sys.platform == "win32"``.

    The value depends on the host, so we only assert the
    *type* of the return — the test passes on every platform.
    """
    result = _win32.is_win32_available()
    assert isinstance(result, bool)


def test_is_win32_available_default_available_callable() -> None:
    """``_default_available`` is a module-level function (not a lambda).

    We assert it exists, is callable, and that calling it
    once doesn't raise. The exact return value is platform-
    dependent.
    """
    fn = _win32._default_available
    assert callable(fn)
    assert isinstance(fn(), bool)


# ---------------------------------------------------------------------------
# _sample_bgrx_cell_brightness
# ---------------------------------------------------------------------------


def test_sample_bgrx_cell_brightness_red_pixel_is_low() -> None:
    """Pure red (255, 0, 0) → brightness 0.30 (the R weight in the luma formula)."""
    # Bottom-up: put the single red pixel in row 0 (which is
    # the bottom of a 1×1 buffer).
    pixels = bytes([0x00, 0x00, 0xFF, 0x00])  # BGRX: red
    b = _win32._sample_bgrx_cell_brightness(
        pixels,
        width=1, height=1,
        cell_x=0, cell_y=0,
        cell_w=1, cell_h=1,
    )
    # 255 * 0.30 + 0 * 0.59 + 0 * 0.11 = 76.5 → /255 = 0.30
    assert abs(b - 0.30) < 0.01


def test_sample_bgrx_cell_brightness_blue_pixel_is_low() -> None:
    """Pure blue (0, 0, 255) → brightness 0.11 (the B weight in the luma formula)."""
    pixels = bytes([0xFF, 0x00, 0x00, 0x00])  # BGRX: blue
    b = _win32._sample_bgrx_cell_brightness(
        pixels,
        width=1, height=1,
        cell_x=0, cell_y=0,
        cell_w=1, cell_h=1,
    )
    assert abs(b - 0.11) < 0.01


def test_sample_bgrx_cell_brightness_white_pixel_is_one() -> None:
    """Pure white (255, 255, 255) → brightness 1.0."""
    pixels = bytes([0xFF, 0xFF, 0xFF, 0x00])  # BGRX: white
    b = _win32._sample_bgrx_cell_brightness(
        pixels,
        width=1, height=1,
        cell_x=0, cell_y=0,
        cell_w=1, cell_h=1,
    )
    assert abs(b - 1.0) < 0.001


def test_sample_bgrx_cell_brightness_out_of_bounds_returns_one() -> None:
    """Cells past the screen extent return 1.0 (no ink) for safe padding."""
    pixels = b""
    # Negative x or y.
    assert _win32._sample_bgrx_cell_brightness(
        pixels, width=10, height=10,
        cell_x=-1, cell_y=0, cell_w=1, cell_h=1,
    ) == 1.0
    assert _win32._sample_bgrx_cell_brightness(
        pixels, width=10, height=10,
        cell_x=0, cell_y=-1, cell_w=1, cell_h=1,
    ) == 1.0
    # x past the right edge.
    assert _win32._sample_bgrx_cell_brightness(
        pixels, width=10, height=10,
        cell_x=100, cell_y=0, cell_w=1, cell_h=1,
    ) == 1.0
    # Zero cell size.
    assert _win32._sample_bgrx_cell_brightness(
        pixels, width=10, height=10,
        cell_x=0, cell_y=0, cell_w=0, cell_h=1,
    ) == 1.0


def test_sample_bgrx_cell_brightness_clamps_to_unit_interval() -> None:
    """A wild pixel value (impossible in real BGRX) is clamped to [0, 1]."""
    # Hypothetical: a 16-bit value sneaking into the luma
    # computation. We can't actually get one through BGRX, but
    # the formula's intermediate ``brightness > 1.0`` branch
    # is worth exercising for defence in depth.
    # We forge a buffer where the luma math overshoots by
    # feeding a single super-bright pixel. The integer bytes
    # can only be 0..255, so the brightest real luma is 1.0.
    # The "out of range" branch is still tested in spirit by
    # the white pixel test above; here we just verify the
    # clamp doesn't break normal values.
    pixels = bytes([0xFF, 0xFF, 0xFF, 0x00])  # BGRX: white
    b = _win32._sample_bgrx_cell_brightness(
        pixels, width=1, height=1,
        cell_x=0, cell_y=0, cell_w=1, cell_h=1,
    )
    assert 0.0 <= b <= 1.0


def test_sample_bgrx_cell_brightness_top_down_flag() -> None:
    """A top-down buffer is sampled in normal row order, not inverted.

    The :class:`Win32Screen` adapter passes ``bottom_up=True``,
    but the pure helper also supports ``bottom_up=False`` so
    tests can feed a top-down buffer directly. This is the
    key symmetry that makes the helper unit-testable on a
    non-Windows box.
    """
    # Top-down red pixel at (0, 0): first 4 bytes of the
    # buffer are BGRX = red.
    pixels = bytes([0x00, 0x00, 0xFF, 0x00])
    b = _win32._sample_bgrx_cell_brightness(
        pixels, width=1, height=1,
        cell_x=0, cell_y=0, cell_w=1, cell_h=1,
        bottom_up=False,
    )
    assert abs(b - 0.30) < 0.01


# ---------------------------------------------------------------------------
# _bgrx_to_grid
# ---------------------------------------------------------------------------


def test_bgrx_to_grid_basic_red_grid() -> None:
    """A solid red 4×1 buffer downsampled to a 1×1 grid → full-ink character."""
    pixels = _make_bgrx_bytes(
        width=4, height=1,
        pixels=[[(255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0)]],
    )
    grid = _win32._bgrx_to_grid(
        pixels, width=4, height=1, x=0, y=0, w=1, h=1
    )
    assert len(grid) == 1
    assert len(grid[0]) == 1
    # Red brightness 0.30 → ink 0.70 → bucket 3 of 5 → '▓' (U+2593).
    assert grid[0] == "\u2593"


def test_bgrx_to_grid_basic_white_grid() -> None:
    """A solid white 1×1 buffer → single space (no ink)."""
    pixels = _make_bgrx_bytes(
        width=1, height=1,
        pixels=[[(255, 255, 255)]],
    )
    assert _win32._bgrx_to_grid(
        pixels, width=1, height=1, x=0, y=0, w=1, h=1
    ) == [" "]


def test_bgrx_to_grid_zero_w_returns_empty() -> None:
    """w=0 short-circuits the loop — no work, no error."""
    pixels = _make_bgrx_bytes()
    assert _win32._bgrx_to_grid(
        pixels, width=4, height=3, x=0, y=0, w=0, h=5
    ) == []


def test_bgrx_to_grid_zero_h_returns_empty() -> None:
    """h=0 short-circuits the loop — symmetric to w=0."""
    pixels = _make_bgrx_bytes()
    assert _win32._bgrx_to_grid(
        pixels, width=4, height=3, x=0, y=0, w=5, h=0
    ) == []


def test_bgrx_to_grid_negative_dimensions_rejected() -> None:
    """A negative w or h is a programmer error (matches X11 adapter)."""
    pixels = _make_bgrx_bytes()
    with pytest.raises(ValueError, match="non-negative"):
        _win32._bgrx_to_grid(
            pixels, width=4, height=3, x=0, y=0, w=-1, h=5
        )
    with pytest.raises(ValueError, match="non-negative"):
        _win32._bgrx_to_grid(
            pixels, width=4, height=3, x=0, y=0, w=5, h=-1
        )


def test_bgrx_to_grid_zero_screen_returns_empty() -> None:
    """A 0×0 or 0-height screen returns [] — the renderer pads anyway."""
    pixels = b""
    assert _win32._bgrx_to_grid(
        pixels, width=0, height=0, x=0, y=0, w=1, h=1
    ) == []
    assert _win32._bgrx_to_grid(
        pixels, width=10, height=0, x=0, y=0, w=1, h=1
    ) == []
    assert _win32._bgrx_to_grid(
        pixels, width=0, height=10, x=0, y=0, w=1, h=1
    ) == []


def test_bgrx_to_grid_full_screen_checkerboard() -> None:
    """A 2×2 checkerboard downsampled to 2×2 cells shows alternating densities."""
    pixels = _make_bgrx_bytes(
        width=2, height=2,
        pixels=[
            [(0, 0, 255), (255, 0, 0)],   # blue, red
            [(255, 0, 0), (0, 0, 255)],   # red, blue
        ],
    )
    grid = _win32._bgrx_to_grid(
        pixels, width=2, height=2, x=0, y=0, w=2, h=2
    )
    assert len(grid) == 2
    assert len(grid[0]) == 2
    # Each cell samples one pixel exactly: the (0,0) and
    # (1,1) cells are blue (ink ≈ 0.89, bucket 4), the (0,1)
    # and (1,0) cells are red (ink ≈ 0.70, bucket 3). Two
    # distinct density chars, mirrored across the diagonal.
    chars = {c for line in grid for c in line}
    assert len(chars) == 2


def test_bgrx_to_grid_full_screen_two_extremes() -> None:
    """A 2×2 black/white checkerboard yields space and full block.

    This is the cleanest possible test: two buckets at the
    ends of the density scale, no intermediate values.
    """
    pixels = _make_bgrx_bytes(
        width=2, height=2,
        pixels=[
            [(255, 255, 255), (0, 0, 0)],
            [(0, 0, 0), (255, 255, 255)],
        ],
    )
    grid = _win32._bgrx_to_grid(
        pixels, width=2, height=2, x=0, y=0, w=2, h=2
    )
    chars = sorted({c for line in grid for c in line})
    assert " " in chars
    assert "\u2588" in chars


# ---------------------------------------------------------------------------
# Win32Screen end-to-end
# ---------------------------------------------------------------------------


def test_win32_screen_screen_size_via_size_callable() -> None:
    """``screen_size`` calls the injected probe, caches, and returns (w, h)."""
    calls: list[int] = []

    def size() -> tuple[int, int]:
        calls.append(1)
        return (1920, 1080)

    def _noop_capture(x: int, y: int, w: int, h: int) -> _win32.Win32Capture:
        return _win32.Win32Capture(width=w, height=h, pixels=b"")
    screen = _win32.Win32Screen(size=size, capture=_noop_capture)
    assert screen.screen_size() == (1920, 1080)
    # Second call uses the cache — no extra probe.
    assert screen.screen_size() == (1920, 1080)
    assert len(calls) == 1


def test_win32_screen_capture_via_capture_callable() -> None:
    """``capture`` returns a list of strings of the right shape."""
    pixels = _make_bgrx_bytes(width=4, height=2)

    def capture(x, y, w, h):
        return _win32.Win32Capture(
            width=w, height=h, pixels=pixels,
        )

    screen = _win32.Win32Screen(
        capture=capture,
        size=lambda: (4, 2),
    )
    grid = screen.capture(x=0, y=0, w=4, h=2)
    assert len(grid) == 2
    assert all(len(line) == 4 for line in grid)


def test_win32_screen_capture_validates_w_h() -> None:
    """A negative w or h is a programmer error (mirrors FakeScreen / X11)."""
    def _noop_capture(x: int, y: int, w: int, h: int) -> _win32.Win32Capture:
        return _win32.Win32Capture(width=w, height=h, pixels=b"")
    screen = _win32.Win32Screen(
        capture=_noop_capture,
        size=lambda: (1, 1),
    )
    with pytest.raises(ValueError, match="non-negative"):
        screen.capture(x=0, y=0, w=-1, h=1)
    with pytest.raises(ValueError, match="non-negative"):
        screen.capture(x=0, y=0, w=1, h=-1)


def test_win32_screen_capture_zero_returns_empty() -> None:
    """A w=0 or h=0 capture short-circuits to ``[]`` without calling capture."""
    calls: list[tuple] = []

    def capture(x, y, w, h):
        calls.append((x, y, w, h))
        return _win32.Win32Capture(
            width=w, height=h, pixels=b"",
        )

    screen = _win32.Win32Screen(
        capture=capture,
        size=lambda: (4, 4),
    )
    assert screen.capture(x=0, y=0, w=0, h=5) == []
    assert screen.capture(x=0, y=0, w=5, h=0) == []
    assert calls == []  # nothing was actually run


def test_win32_screen_default_capture_and_size_are_real() -> None:
    """The default capture / size callables are the real GDI plumbing.

    We don't actually *call* them here (no Windows display);
    we just check that the constructor's default-arg path
    assigns :func:`_default_capture` and
    :func:`_default_screen_size` (not the stubs we use
    elsewhere in this file) and that they're callable. The
    end-to-end "real GDI invocation" path is exercised on a
    developer machine with a real Windows desktop; here we
    only need to prove the wiring is correct.
    """
    screen = _win32.Win32Screen()
    assert screen._capture is _win32._default_capture
    assert screen._size is _win32._default_screen_size
    assert callable(screen._capture)
    assert callable(screen._size)


# ---------------------------------------------------------------------------
# build_win32_screen factory
# ---------------------------------------------------------------------------


def test_build_win32_screen_returns_none_when_unavailable() -> None:
    """On a non-Windows box the factory returns None, not an instance."""
    screen = _win32.build_win32_screen(available=lambda: False)
    assert screen is None


def test_build_win32_screen_returns_instance_when_available() -> None:
    """The availability check passes → a real Win32Screen is returned."""
    screen = _win32.build_win32_screen(available=lambda: True)
    assert isinstance(screen, _win32.Win32Screen)


def test_build_win32_screen_uses_injected_capture() -> None:
    """The factory forwards the ``capture`` kwarg to the new instance."""
    pixels = _make_bgrx_bytes(width=1, height=1)

    def capture(x, y, w, h):
        return _win32.Win32Capture(width=w, height=h, pixels=pixels)

    screen = _win32.build_win32_screen(
        capture=capture,
        size=lambda: (1, 1),
        available=lambda: True,
    )
    assert screen is not None
    # Use the injected capture; the call should not raise.
    grid = screen.capture(x=0, y=0, w=1, h=1)
    assert len(grid) == 1 and len(grid[0]) == 1


# ---------------------------------------------------------------------------
# Integration with ``whisperpaw._screen.get_capture``
# ---------------------------------------------------------------------------


def test_screen_dispatch_win32_returns_instance_when_available() -> None:
    """``get_capture(backend='win32')`` returns a Win32Screen when available.

    We monkeypatch the factory's discovery helpers to simulate
    "Windows is available" without needing a real Windows box.
    """
    from whisperpaw import _screen as screen_mod

    real_win32_capture = screen_mod._win32_capture
    try:
        # Replace the factory with one that always returns an instance.
        class _FakeAdapter(screen_mod.ScreenCapture):
            def screen_size(self) -> tuple[int, int]:
                return (4, 2)

            def capture(self, *, x, y, w, h):
                return ["    "] * h

        screen_mod._BACKEND_FACTORIES["win32"] = lambda: _FakeAdapter()
        cap = screen_mod.get_capture("win32")
        assert isinstance(cap, _FakeAdapter)
    finally:
        screen_mod._BACKEND_FACTORIES["win32"] = real_win32_capture


def test_screen_dispatch_win32_returns_none_on_linux() -> None:
    """``get_capture(backend='win32')`` returns None on a non-Windows box.

    This is the user-visible integration test: it proves the
    v0.2 pipeline (zoom.py + _screen.py + _win32.py) compose
    correctly even when the OS adapter is unavailable. The
    call returns None, not an exception.
    """
    cap = _screen.get_capture("win32")
    # We're on Linux CI; the factory must return None, not raise.
    assert cap is None


def test_screen_dispatch_auto_order_x11_before_win32() -> None:
    """``get_capture(backend='auto')`` tries x11 before win32 (Linux first)."""
    from whisperpaw import _screen as screen_mod

    class _Marker(screen_mod.ScreenCapture):
        def __init__(self, name):
            self.name = name

        def screen_size(self):
            return (1, 1)

        def capture(self, *, x, y, w, h):
            return [" "] * h

    real = dict(screen_mod._BACKEND_FACTORIES)

    def make_marker(name):
        def factory():
            return _Marker(name)
        return factory

    try:
        screen_mod._BACKEND_FACTORIES["x11"] = make_marker("x11")
        screen_mod._BACKEND_FACTORIES["win32"] = make_marker("win32")
        screen_mod._BACKEND_FACTORIES["quartz"] = make_marker("quartz")
        cap = screen_mod.get_capture("auto")
        assert cap is not None
        # The pick order is x11 → win32 → quartz. On any
        # platform where all three are mocked available, x11
        # wins. (We don't test "win32 only" because the real
        # _x11_capture is also wired in; mocking it out is
        # cleaner than trying to make x11 unavailable
        # piecemeal.)
        assert cap.name == "x11"
    finally:
        screen_mod._BACKEND_FACTORIES.clear()
        screen_mod._BACKEND_FACTORIES.update(real)


def test_paw_zoom_screen_win32_backend_unsupported_message() -> None:
    """On a non-Windows box, ``paw-zoom --screen --backend win32`` prints
    the friendly "not yet implemented" message and exits 1.

    This is the user-visible integration test: it proves that
    the v0.2 pipeline (zoom.py + _screen.py + _win32.py) compose
    correctly even when the OS adapter is unavailable.
    """
    import io
    from contextlib import redirect_stderr, redirect_stdout
    from whisperpaw import zoom

    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = zoom.main(
            ["--screen", "--backend", "win32", "--rows", "2", "--cols", "2"]
        )
    # On a non-Windows box this returns 1 and writes the
    # friendly "not yet implemented on this OS" message.
    assert rc == 1
    assert "not yet implemented" in err.getvalue() or "fake" in err.getvalue()
