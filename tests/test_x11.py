"""Tests for ``whisperpaw._x11`` (the real X11 screen-capture adapter).

These tests don't need a real X server — they synthesise a
small XWD byte stream in memory and feed it to the adapter
through an injectable :class:`Runner`. That keeps the test
suite fast and headless-friendly (this CI box has no
``$DISPLAY``) while still exercising the real XWD-header
parsing path and the real ``_pixels_to_grid`` downsample
path.

XWD file format reminder
------------------------
The XWD header is 100 bytes on the wire, but the first 56
bytes (14 × uint32) are the only fields we care about. The
:func:`_make_xwd_bytes` helper below builds a minimal valid
dump: a 56-byte header (``header_size = 100`` for the full
100-byte on-wire header, the rest zeroed), a zero-colour-map
section (24 bytes of padding so ``header_size + 0 ==
pixel_offset``), then a tightly-packed pixel grid in BGRX /
BGR order.
"""
from __future__ import annotations

import struct

import pytest

from whisperpaw import _screen, _x11


# ---------------------------------------------------------------------------
# XWD byte-stream builder
# ---------------------------------------------------------------------------


def _make_xwd_bytes(
    *,
    width: int = 4,
    height: int = 3,
    pixels: list[list[tuple[int, int, int]]] | None = None,
    bits_per_pixel: int = 24,
    byte_order: int = 0,  # 0 = LSBFirst (little-endian on the wire)
    ncolors: int = 0,
    version: int = 7,
    header_size: int = 100,
    bytes_per_line: int | None = None,
) -> bytes:
    """Build a synthetic XWD byte stream for tests.

    The default is a 4×3, 24bpp, little-endian dump with
    ``ncolors = 0`` (no colour-map section), which is the
    shape a modern Linux X server actually produces.

    Parameters
    ----------
    pixels:
        A ``height`` × ``width`` list of (r, g, b) tuples.
        Defaults to a checkerboard where odd cells are red
        and even cells are blue, so the downsample output is
        easy to reason about.
    bits_per_pixel:
        24 (BGR, no alpha) or 32 (BGRX, alpha ignored). The
        16-bit case is rejected by the downsample path; we
        don't test it here.
    byte_order:
        0 = little-endian (the X server on x86 / x86_64
        Linux), 1 = big-endian (rare, but we test it).
    ncolors:
        0 by default. If > 0, the colour-map section is
        ``ncolors * 12`` bytes of zeros inserted between the
        header and the pixel data.
    """
    if pixels is None:
        # Checkerboard: (0, 0) is blue, (1, 0) is red, etc.
        pixels = []
        for row in range(height):
            pixels_row = []
            for col in range(width):
                if (row + col) % 2 == 0:
                    pixels_row.append((0, 0, 255))  # blue
                else:
                    pixels_row.append((255, 0, 0))  # red
            pixels.append(pixels_row)

    bpp = bits_per_pixel // 8  # 3 or 4
    if bytes_per_line is None:
        bytes_per_line = width * bpp
    bo = "<" if byte_order == 0 else ">"

    # 56-byte XWD header (the part we actually parse).
    header = struct.pack(
        bo + "IIIIIIIIIIIIII",
        header_size,   # header_size
        version,       # version (7 = the only XWD version)
        0,             # pixmap_format (ZPixmap = 2; ignored)
        24,            # depth
        width,         # width
        height,        # height
        0,             # xoffset
        byte_order,    # byte_order
        32,            # bitmap_unit
        0,             # bitmap_bit_order
        32,            # bitmap_pad
        bits_per_pixel,  # bits_per_pixel
        bytes_per_line,  # bytes_per_line
        0,             # visual_class
    )
    assert len(header) == 56
    # Pad up to the full 100-byte on-wire header. The
    # colour-map count (``ncolors``) lives at offset 64 — the
    # 17th uint32 — followed by another 32 bytes of fields
    # we don't parse. We put the ncolors count there, then
    # pad with zeros up to header_size.
    extra = b"\x00" * 8 + struct.pack(bo + "I", ncolors) + b"\x00" * 32
    full_header = (header + extra)[:header_size]
    assert len(full_header) == header_size

    # Colour-map section (skipped by _color_map_size).
    color_map = b"\x00" * (ncolors * 12)

    # Pixel data: every row is bytes_per_line bytes, in BGR or
    # BGRX order. The X server on little-endian writes bytes
    # in memory order, so for 24bpp it's b, g, r, b, g, r, ...
    row_bytes: list[bytes] = []
    for row in range(height):
        cells: list[bytes] = []
        for col in range(width):
            r, g, b = pixels[row][col]
            if bpp == 3:
                cells.append(struct.pack("BBB", b, g, r))
            else:  # bpp == 4
                cells.append(struct.pack("BBBB", b, g, r, 0))
        line = b"".join(cells)
        # Pad up to bytes_per_line (we never set a pad bigger
        # than the natural row, so the test just uses the
        # natural size — but the pad logic is still exercised
        # for any future call that sets bytes_per_line > width*bpp).
        if len(line) < bytes_per_line:
            line = line + b"\x00" * (bytes_per_line - len(line))
        row_bytes.append(line)
    pixel_data = b"".join(row_bytes)
    return full_header + color_map + pixel_data


# ---------------------------------------------------------------------------
# is_x11_available
# ---------------------------------------------------------------------------


def test_is_x11_available_true_when_display_and_xwd_present() -> None:
    """Both discovery checks pass: we report the OS adapter is available."""
    env = {"DISPLAY": ":0"}
    # Fake `which` that always reports xwd as present.
    def fake_which(name: str) -> str | None:
        return f"/usr/bin/{name}" if name == "xwd" else None
    assert _x11.is_x11_available(env=env, which=fake_which) is True


def test_is_x11_available_false_when_display_unset() -> None:
    """No $DISPLAY → no X server → unavailable, even if xwd is on PATH."""
    env: dict[str, str] = {}
    def fake_which(name: str) -> str | None:
        return f"/usr/bin/{name}" if name == "xwd" else None
    assert _x11.is_x11_available(env=env, which=fake_which) is False


def test_is_x11_available_false_when_xwd_missing() -> None:
    """$DISPLAY set but no xwd binary → still unavailable."""
    env = {"DISPLAY": ":0"}
    def fake_which(name: str) -> str | None:
        return None  # nothing on PATH
    assert _x11.is_x11_available(env=env, which=fake_which) is False


def test_is_x11_available_uses_real_which_when_not_injected() -> None:
    """The default branches (os.environ, shutil.which) are exercised.

    On a headless box this returns False; on a Linux desktop
    with xwd it returns True. We only assert the *type* of
    the return value here, not the specific value, so the
    test passes in both environments.
    """
    result = _x11.is_x11_available()
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# _parse_xwd_header
# ---------------------------------------------------------------------------


def test_parse_xwd_header_happy_path() -> None:
    """A well-formed 56-byte header parses into a populated NamedTuple."""
    data = _make_xwd_bytes(width=1920, height=1080, header_size=100)
    header = _x11._parse_xwd_header(data)
    assert header.width == 1920
    assert header.height == 1080
    assert header.bits_per_pixel == 24
    assert header.bytes_per_line == 1920 * 3
    assert header.byte_order == 0
    assert header.header_size == 100
    assert header.version == 7


def test_parse_xwd_header_rejects_short_input() -> None:
    """Less than 56 bytes is a hard error — we can't even unpack the struct."""
    with pytest.raises(ValueError, match="too short"):
        _x11._parse_xwd_header(b"\x00" * 40)


def test_parse_xwd_header_rejects_wrong_version() -> None:
    """A version other than 7 is not an XWD file (only version 7 exists)."""
    data = bytearray(_make_xwd_bytes())
    # The version is the second uint32 in the header (offset 4).
    struct.pack_into("<I", data, 4, 99)
    with pytest.raises(ValueError, match="version"):
        _x11._parse_xwd_header(bytes(data))


def test_parse_xwd_header_rejects_tiny_header_size() -> None:
    """A header_size smaller than 56 means the file is truncated/corrupt."""
    data = bytearray(_make_xwd_bytes())
    struct.pack_into("<I", data, 0, 16)  # header_size = 16
    with pytest.raises(ValueError, match="header_size"):
        _x11._parse_xwd_header(bytes(data))


def test_parse_xwd_header_handles_big_endian() -> None:
    """A big-endian X server (byte_order = 1) still parses correctly."""
    data = _make_xwd_bytes(width=10, height=5, byte_order=1)
    header = _x11._parse_xwd_header(data)
    assert header.width == 10
    assert header.height == 5
    assert header.byte_order == 1


# ---------------------------------------------------------------------------
# _color_map_size
# ---------------------------------------------------------------------------


def test_color_map_size_zero_when_ncolors_is_zero() -> None:
    """The common case: a 24/32bpp display has no colour map."""
    data = _make_xwd_bytes(ncolors=0)
    assert _x11._color_map_size(data) == 0


def test_color_map_size_counts_entries_in_little_endian() -> None:
    """A 5-entry colour map adds 60 bytes to skip."""
    data = _make_xwd_bytes(ncolors=5, byte_order=0)
    assert _x11._color_map_size(data) == 5 * 12


def test_color_map_size_counts_entries_in_big_endian() -> None:
    """The byte-order field controls how ncolors is unpacked."""
    data = _make_xwd_bytes(ncolors=7, byte_order=1)
    assert _x11._color_map_size(data) == 7 * 12


def test_color_map_size_zero_when_input_too_short() -> None:
    """Less than 68 bytes means we can't even read ncolors — return 0."""
    assert _x11._color_map_size(b"\x00" * 30) == 0


# ---------------------------------------------------------------------------
# _sample_cell_brightness
# ---------------------------------------------------------------------------


def test_sample_cell_brightness_red_pixel_is_low() -> None:
    """Pure red (255, 0, 0) → brightness 0.30 (the R weight in the luma formula)."""
    pixels = b"\x00\x00\xff"  # B, G, R in BGR order
    header = _x11.XwdHeader(
        header_size=100, version=7, pixmap_format=0, depth=24,
        width=1, height=1, xoffset=0, byte_order=0,
        bitmap_unit=32, bitmap_bit_order=0, bitmap_pad=32,
        bits_per_pixel=24, bytes_per_line=3, visual_class=0,
    )
    b = _x11._sample_cell_brightness(
        pixels, header, cell_x=0, cell_y=0, cell_w=1, cell_h=1
    )
    # 255 * 0.30 + 0 * 0.59 + 0 * 0.11 = 76.5 → /255 = 0.30
    assert abs(b - 0.30) < 0.01


def test_sample_cell_brightness_blue_pixel_is_low() -> None:
    """Pure blue (0, 0, 255) → brightness 0.11 (the B weight in the luma formula)."""
    pixels = b"\xff\x00\x00"  # BGR: B=255
    header = _x11.XwdHeader(
        header_size=100, version=7, pixmap_format=0, depth=24,
        width=1, height=1, xoffset=0, byte_order=0,
        bitmap_unit=32, bitmap_bit_order=0, bitmap_pad=32,
        bits_per_pixel=24, bytes_per_line=3, visual_class=0,
    )
    b = _x11._sample_cell_brightness(
        pixels, header, cell_x=0, cell_y=0, cell_w=1, cell_h=1
    )
    assert abs(b - 0.11) < 0.01


def test_sample_cell_brightness_white_pixel_is_one() -> None:
    """Pure white (255, 255, 255) → brightness 1.0."""
    pixels = b"\xff\xff\xff"  # BGR: B=G=R=255
    header = _x11.XwdHeader(
        header_size=100, version=7, pixmap_format=0, depth=24,
        width=1, height=1, xoffset=0, byte_order=0,
        bitmap_unit=32, bitmap_bit_order=0, bitmap_pad=32,
        bits_per_pixel=24, bytes_per_line=3, visual_class=0,
    )
    b = _x11._sample_cell_brightness(
        pixels, header, cell_x=0, cell_y=0, cell_w=1, cell_h=1
    )
    assert abs(b - 1.0) < 0.001


def test_sample_cell_brightness_out_of_bounds_returns_one() -> None:
    """Cells past the screen extent return 1.0 (no ink) for safe padding."""
    pixels = b""
    header = _x11.XwdHeader(
        header_size=100, version=7, pixmap_format=0, depth=24,
        width=10, height=10, xoffset=0, byte_order=0,
        bitmap_unit=32, bitmap_bit_order=0, bitmap_pad=32,
        bits_per_pixel=24, bytes_per_line=30, visual_class=0,
    )
    assert _x11._sample_cell_brightness(
        pixels, header, cell_x=-1, cell_y=0, cell_w=1, cell_h=1
    ) == 1.0
    assert _x11._sample_cell_brightness(
        pixels, header, cell_x=0, cell_y=-1, cell_w=1, cell_h=1
    ) == 1.0
    assert _x11._sample_cell_brightness(
        pixels, header, cell_x=100, cell_y=0, cell_w=1, cell_h=1
    ) == 1.0


def test_sample_cell_brightness_monochrome_falls_back_to_white() -> None:
    """16-bit / 1-bit displays return 1.0 — we don't try to decode them."""
    pixels = b"\x00\x00"
    header = _x11.XwdHeader(
        header_size=100, version=7, pixmap_format=0, depth=16,
        width=1, height=1, xoffset=0, byte_order=0,
        bitmap_unit=32, bitmap_bit_order=0, bitmap_pad=32,
        bits_per_pixel=16, bytes_per_line=2, visual_class=0,
    )
    assert _x11._sample_cell_brightness(
        pixels, header, cell_x=0, cell_y=0, cell_w=1, cell_h=1
    ) == 1.0


# ---------------------------------------------------------------------------
# _pixels_to_grid
# ---------------------------------------------------------------------------


def test_pixels_to_grid_basic_red_grid() -> None:
    """A solid red 4×1 image downsampled to a 1×1 grid → full-ink character."""
    data = _make_xwd_bytes(
        width=4, height=1,
        pixels=[[(255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0)]],
    )
    header = _x11._parse_xwd_header(data)
    pixels = data[header.header_size:]  # ncolors=0 → direct
    grid = _x11._pixels_to_grid(pixels, header, x=0, y=0, w=1, h=1)
    assert len(grid) == 1
    assert len(grid[0]) == 1
    # Red brightness 0.30 → ink 0.70 → bucket 3 of 5 → '▓' (U+2593).
    assert grid[0] == "▓"


def test_pixels_to_grid_basic_white_grid() -> None:
    """A solid white 1×1 image → single space (no ink)."""
    data = _make_xwd_bytes(
        width=1, height=1,
        pixels=[[(255, 255, 255)]],
    )
    header = _x11._parse_xwd_header(data)
    pixels = data[header.header_size:]
    grid = _x11._pixels_to_grid(pixels, header, x=0, y=0, w=1, h=1)
    assert grid == [" "]


def test_pixels_to_grid_zero_w_returns_empty() -> None:
    """w=0 short-circuits the loop — no work, no error."""
    data = _make_xwd_bytes()
    header = _x11._parse_xwd_header(data)
    pixels = data[header.header_size:]
    assert _x11._pixels_to_grid(pixels, header, x=0, y=0, w=0, h=5) == []


def test_pixels_to_grid_zero_h_returns_empty() -> None:
    """h=0 short-circuits the loop — symmetric to w=0."""
    data = _make_xwd_bytes()
    header = _x11._parse_xwd_header(data)
    pixels = data[header.header_size:]
    assert _x11._pixels_to_grid(pixels, header, x=0, y=0, w=5, h=0) == []


def test_pixels_to_grid_negative_dimensions_rejected() -> None:
    """A negative w or h is a programmer error (matches FakeScreen)."""
    data = _make_xwd_bytes()
    header = _x11._parse_xwd_header(data)
    pixels = data[header.header_size:]
    with pytest.raises(ValueError, match="non-negative"):
        _x11._pixels_to_grid(pixels, header, x=0, y=0, w=-1, h=5)
    with pytest.raises(ValueError, match="non-negative"):
        _x11._pixels_to_grid(pixels, header, x=0, y=0, w=5, h=-1)


def test_pixels_to_grid_full_screen_checkerboard() -> None:
    """A 2×2 checkerboard downsampled to 2×2 cells shows alternating densities."""
    data = _make_xwd_bytes(
        width=2, height=2,
        pixels=[
            [(0, 0, 255), (255, 0, 0)],   # blue, red
            [(255, 0, 0), (0, 0, 255)],   # red, blue
        ],
    )
    header = _x11._parse_xwd_header(data)
    # Skip the colour map.
    pixel_offset = header.header_size + _x11._color_map_size(data)
    pixels = data[pixel_offset:]
    grid = _x11._pixels_to_grid(pixels, header, x=0, y=0, w=2, h=2)
    assert len(grid) == 2
    assert len(grid[0]) == 2
    # The (0,0) cell samples (0,0)=blue (ink ≈ 0.89, bucket 4) and
    # the (1,1) cell samples (1,1)=blue (same), the (0,1) and (1,0)
    # cells each sample a red pixel (ink ≈ 0.70, bucket 3). The
    # checkerboard therefore produces two distinct density chars.
    chars = {c for line in grid for c in line}
    assert len(chars) == 2


def test_pixels_to_grid_with_color_map_section() -> None:
    """A non-zero ncolors shifts the pixel offset; the grid still renders."""
    data = _make_xwd_bytes(
        width=2, height=2,
        ncolors=2,  # adds 24 bytes between header and pixels
        pixels=[
            [(255, 255, 255), (0, 0, 0)],
            [(0, 0, 0), (255, 255, 255)],
        ],
    )
    header = _x11._parse_xwd_header(data)
    pixel_offset = header.header_size + _x11._color_map_size(data)
    pixels = data[pixel_offset:]
    # Asking for 2×2 cells on a 2×2 image: each cell samples one
    # pixel exactly, so we get exactly the checkerboard brightness.
    grid = _x11._pixels_to_grid(pixels, header, x=0, y=0, w=2, h=2)
    assert len(grid) == 2 and all(len(line) == 2 for line in grid)
    # Two black and two white cells → space and full block.
    chars = sorted({c for line in grid for c in line})
    assert " " in chars
    assert "█" in chars


def test_pixels_to_grid_32bpp_ignores_alpha() -> None:
    """A 32bpp image's alpha byte doesn't affect the brightness reading."""
    data_24 = _make_xwd_bytes(
        width=1, height=1, bits_per_pixel=24,
        pixels=[[(255, 0, 0)]],  # red
    )
    data_32 = _make_xwd_bytes(
        width=1, height=1, bits_per_pixel=32,
        pixels=[[(255, 0, 0)]],  # red, alpha byte is 0
    )
    h24 = _x11._parse_xwd_header(data_24)
    h32 = _x11._parse_xwd_header(data_32)
    p24 = data_24[h24.header_size:]
    p32 = data_32[h32.header_size:]
    g24 = _x11._pixels_to_grid(p24, h24, x=0, y=0, w=1, h=1)
    g32 = _x11._pixels_to_grid(p32, h32, x=0, y=0, w=1, h=1)
    # The alpha byte in 32bpp is the *fourth* byte; the first
    # three (B, G, R) are identical to the 24bpp case, so the
    # output is identical.
    assert g24 == g32


# ---------------------------------------------------------------------------
# X11Screen end-to-end
# ---------------------------------------------------------------------------


def test_x11_screen_screen_size_via_runner() -> None:
    """``screen_size`` runs xwd once, caches, and returns (w, h)."""
    data = _make_xwd_bytes(width=1920, height=1080)
    calls: list[tuple] = []

    def runner(argv):
        calls.append(tuple(argv))
        return (0, data, b"")

    screen = _x11.X11Screen(runner=runner)
    assert screen.screen_size() == (1920, 1080)
    # Second call uses the cache — no extra subprocess.
    assert screen.screen_size() == (1920, 1080)
    assert len(calls) == 1


def test_x11_screen_capture_via_runner() -> None:
    """``capture`` returns a list of strings of the right shape."""
    data = _make_xwd_bytes(width=4, height=2)

    def runner(argv):
        return (0, data, b"")

    screen = _x11.X11Screen(runner=runner)
    grid = screen.capture(x=0, y=0, w=4, h=2)
    assert len(grid) == 2
    assert all(len(line) == 4 for line in grid)


def test_x11_screen_capture_invalidates_size_cache() -> None:
    """Every ``capture`` refreshes the size cache, so a monitor resize
    between two captures doesn't leave a stale cached value behind."""
    data_small = _make_xwd_bytes(width=4, height=2)
    data_big = _make_xwd_bytes(width=1920, height=1080)
    outputs = [data_small, data_big, data_small, data_big]
    idx = {"n": 0}

    def runner(argv):
        n = idx["n"]
        idx["n"] += 1
        return (0, outputs[n % len(outputs)], b"")

    screen = _x11.X11Screen(runner=runner)
    assert screen.screen_size() == (4, 2)
    # First capture: server now reports 1920×1080.
    screen.capture(x=0, y=0, w=4, h=2)
    assert screen.screen_size() == (1920, 1080)
    # Second capture: server back to 4×2.
    screen.capture(x=0, y=0, w=4, h=2)
    assert screen.screen_size() == (4, 2)


def test_x11_screen_capture_validates_w_h() -> None:
    """A negative w or h is a programmer error (mirrors FakeScreen)."""
    screen = _x11.X11Screen(runner=lambda argv: (0, b"", b""))
    with pytest.raises(ValueError, match="non-negative"):
        screen.capture(x=0, y=0, w=-1, h=1)
    with pytest.raises(ValueError, match="non-negative"):
        screen.capture(x=0, y=0, w=1, h=-1)


def test_x11_screen_capture_zero_returns_empty() -> None:
    """A w=0 or h=0 capture short-circuits to ``[]`` without subprocess."""
    calls: list[tuple] = []

    def runner(argv):
        calls.append(tuple(argv))
        return (0, _make_xwd_bytes(), b"")

    screen = _x11.X11Screen(runner=runner)
    assert screen.capture(x=0, y=0, w=0, h=5) == []
    assert screen.capture(x=0, y=0, w=5, h=0) == []
    assert calls == []  # nothing was actually run


def test_x11_screen_runner_failure_raises_runtime_error() -> None:
    """A non-zero xwd exit code is wrapped in a friendly RuntimeError."""

    def runner(argv):
        return (1, b"", b"Cannot connect to X server")

    screen = _x11.X11Screen(runner=runner)
    with pytest.raises(RuntimeError, match="xwd failed"):
        screen.screen_size()


def test_x11_screen_runner_short_output_raises() -> None:
    """A truncated xwd dump (< 56 bytes) is a RuntimeError, not a crash."""
    def runner(argv):
        return (0, b"\x00" * 20, b"")
    screen = _x11.X11Screen(runner=runner)
    with pytest.raises(RuntimeError, match="56-byte header"):
        screen.screen_size()


def test_x11_screen_runner_short_pixels_padded() -> None:
    """A dump that's too short in the pixel area is padded with zeros,
    so the downsample loop doesn't IndexError."""
    data = _make_xwd_bytes(width=10, height=10)
    # Lop off half the pixel data.
    truncated = data[:len(data) - 50]

    def runner(argv):
        return (0, truncated, b"")
    screen = _x11.X11Screen(runner=runner)
    # Should not raise; result is just a 10×1 string.
    grid = screen.capture(x=0, y=0, w=10, h=1)
    assert len(grid) == 1 and len(grid[0]) == 10


def test_x11_screen_default_runner_uses_subprocess() -> None:
    """The default runner is the real subprocess wrapper, lazily imported.

    We don't actually *run* xwd here (no $DISPLAY); we just
    check that the constructor's default-arg path assigns
    the real :func:`_default_runner` (not the stub we use
    elsewhere in this file) and that it's callable. The
    end-to-end "real xwd invocation" path is exercised on a
    developer machine with a real X server; here we only
    need to prove the wiring is correct.
    """
    screen = _x11.X11Screen()
    # The default runner is _default_runner; we don't invoke
    # it because the dev box (and this CI box) have no
    # $DISPLAY. We just confirm it's the real wrapper.
    assert screen._runner is _x11._default_runner
    # And confirm it's callable with a tuple of strings.
    assert callable(screen._runner)


# ---------------------------------------------------------------------------
# build_x11_screen factory
# ---------------------------------------------------------------------------


def test_build_x11_screen_returns_none_on_headless() -> None:
    """On a box with no $DISPLAY the factory returns None, not an instance."""
    screen = _x11.build_x11_screen(
        env={},
        which=lambda name: None,
    )
    assert screen is None


def test_build_x11_screen_returns_none_when_xwd_missing() -> None:
    """$DISPLAY set but no xwd → still None."""
    screen = _x11.build_x11_screen(
        env={"DISPLAY": ":0"},
        which=lambda name: None,
    )
    assert screen is None


def test_build_x11_screen_returns_instance_when_available() -> None:
    """Both discovery checks pass → a real X11Screen is returned."""
    env = {"DISPLAY": ":0"}
    def fake_which(name: str) -> str | None:
        return f"/usr/bin/{name}" if name == "xwd" else None
    screen = _x11.build_x11_screen(env=env, which=fake_which)
    assert isinstance(screen, _x11.X11Screen)


def test_build_x11_screen_uses_injected_runner() -> None:
    """The factory forwards the ``runner`` kwarg to the new instance."""
    def runner(argv):
        return (0, _make_xwd_bytes(width=1, height=1), b"")
    env = {"DISPLAY": ":0"}
    def fake_which(name: str) -> str | None:
        return f"/usr/bin/{name}" if name == "xwd" else None
    screen = _x11.build_x11_screen(runner=runner, env=env, which=fake_which)
    assert screen is not None
    # Use the injected runner; the call should not raise.
    assert screen.screen_size() == (1, 1)


# ---------------------------------------------------------------------------
# Integration with ``whisperpaw._screen.get_capture``
# ---------------------------------------------------------------------------


def test_screen_dispatch_x11_returns_instance_when_available() -> None:
    """``get_capture(backend='x11')`` returns an X11Screen when on a real X desktop.

    We monkeypatch the factory's discovery helpers to simulate
    "X is available" without needing a real $DISPLAY.
    """
    from whisperpaw import _screen as screen_mod

    real_x11_capture = screen_mod._x11_capture
    try:
        # Replace the factory with one that always returns an instance.
        class _FakeAdapter(screen_mod.ScreenCapture):
            def screen_size(self) -> tuple[int, int]:
                return (4, 2)
            def capture(self, *, x, y, w, h):
                return ["    "] * h
        screen_mod._BACKEND_FACTORIES["x11"] = lambda: _FakeAdapter()
        cap = screen_mod.get_capture("x11")
        assert isinstance(cap, _FakeAdapter)
    finally:
        screen_mod._BACKEND_FACTORIES["x11"] = real_x11_capture


def test_screen_dispatch_x11_returns_none_on_headless() -> None:
    """``get_capture(backend='x11')`` returns None on a headless box."""
    # This box has no $DISPLAY and no xwd; the real factory
    # should return None without crashing.
    cap = _screen.get_capture("x11")
    assert cap is None


def test_screen_dispatch_auto_picks_x11_first() -> None:
    """``get_capture(backend='auto')`` tries x11 before win32 / quartz.

    We monkeypatch all three to return markers, then assert
    the chosen one is the x11 marker. This is the proof that
    the dispatch order in :func:`get_capture` honours the
    "Linux desktop first" rule in the docstring.
    """
    from whisperpaw import _screen as screen_mod

    class _Marker(screen_mod.ScreenCapture):
        def __init__(self, name): self.name = name
        def screen_size(self): return (1, 1)
        def capture(self, *, x, y, w, h): return [" "] * h

    # Remember the real factories so we can restore them.
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
        assert cap.name == "x11"
    finally:
        screen_mod._BACKEND_FACTORIES.clear()
        screen_mod._BACKEND_FACTORIES.update(real)


def test_paw_zoom_screen_x11_backend_unsupported_message() -> None:
    """On a headless box, ``paw-zoom --screen --backend x11`` prints
    the friendly "not yet implemented" message and exits 1.

    This is the user-visible integration test: it proves that
    the v0.2 pipeline (zoom.py + _screen.py + _x11.py) compose
    correctly even when the OS adapter is unavailable.
    """
    import io
    from contextlib import redirect_stderr, redirect_stdout
    from whisperpaw import zoom

    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = zoom.main(["--screen", "--backend", "x11", "--rows", "2", "--cols", "2"])
    # On a headless box this returns 1 and writes the friendly
    # "not yet implemented on this OS" message to stderr.
    assert rc == 1
    assert "not yet implemented" in err.getvalue() or "fake" in err.getvalue()
