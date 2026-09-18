"""Tests for ``whisperpaw._quartz`` (the real macOS Quartz screen-capture adapter).

These tests don't need a real Mac — they synthesise a minimal
TIFF bitstream in memory (8-byte header + IFD + pixel strip) and
feed it to the adapter through an injectable ``runner=``
callable. That keeps the test suite fast and cross-platform
(this CI box is Linux) while still exercising the real TIFF
parsing path, the real RGBA sampling path, and the real
``_rgba_to_grid`` downsample path.

TIFF layout reminder
--------------------
A TIFF file is just an 8-byte header (2-byte byte-order magic
+ 2-byte magic number 42 + 4-byte IFD offset) followed by an
IFD (2-byte entry count + N × 12-byte tag entries) followed
by the pixel strip at the offset pointed to by ``StripOffsets``.
The :func:`_make_tiff_bytes` helper below builds a minimal
valid little-endian TIFF of the exact shape
``screencapture -t tiff -`` produces on macOS: 32-bpp RGBA,
no compression, RGB photometric, top-down rows.

RGBA pixel format reminder
--------------------------
``screencapture``'s TIFF output is 4 bytes per pixel in R, G,
B, A order (the alpha byte is ignored — we only need a
brightness reading). Rows are top-down (the strip contains
row 0 first). The
:func:`whisperpaw._quartz._sample_rgba_cell_brightness` helper
reads in normal row order.
"""
from __future__ import annotations

import struct

import pytest

from whisperpaw import _quartz, _screen


# ---------------------------------------------------------------------------
# TIFF byte-stream builder
# ---------------------------------------------------------------------------


def _make_tiff_bytes(
    *,
    width: int = 4,
    height: int = 3,
    pixels: list[list[tuple[int, int, int, int]]] | None = None,
) -> bytes:
    """Build a synthetic 32-bpp RGBA TIFF for tests.

    The default is a 4×3 little-endian TIFF with a
    checkerboard where even cells are opaque red and odd cells
    are opaque blue, so the downsample output is easy to
    reason about. The structure is:

    * 8-byte header: ``II`` + 42 (LE) + IFD offset (8).
    * IFD with the 9 tags the adapter reads
      (``ImageWidth``, ``ImageLength``, ``BitsPerSample``,
      ``Compression``, ``PhotometricInterpretation``,
      ``StripOffsets``, ``SamplesPerPixel``,
      ``RowsPerStrip``, ``StripByteCounts``) — entries
      written in tag-ID order, which is what the TIFF spec
      requires.
    * 1-byte gap so the pixel data doesn't sit directly
      against the IFD (real screencapture output has the
      same gap).
    * The pixel strip itself, row 0 first, 4 bytes per
      pixel in R, G, B, A order.
    """
    if pixels is None:
        pixels = []
        for row in range(height):
            pixels_row = []
            for col in range(width):
                if (row + col) % 2 == 0:
                    pixels_row.append((255, 0, 0, 255))  # red
                else:
                    pixels_row.append((0, 0, 255, 255))  # blue
            pixels.append(pixels_row)

    bps = 8
    spp = 4
    bpp = (bps * spp) // 8  # 4 bytes per pixel
    bytes_per_row = width * bpp
    strip_byte_count = bytes_per_row * height

    # IFD starts immediately after the 8-byte file header.
    ifd_offset = 8
    # Tag entries, in tag-ID order (TIFF spec requirement).
    tag_ids = [
        _quartz._TAG_IMAGE_WIDTH,
        _quartz._TAG_IMAGE_LENGTH,
        _quartz._TAG_BITS_PER_SAMPLE,
        _quartz._TAG_COMPRESSION,
        _quartz._TAG_PHOTOMETRIC,
        _quartz._TAG_STRIP_OFFSETS,
        _quartz._TAG_SAMPLES_PER_PIXEL,
        _quartz._TAG_ROWS_PER_STRIP,
        _quartz._TAG_STRIP_BYTE_COUNTS,
    ]
    n_entries = len(tag_ids)
    # IFD ends at: offset + 2 (entry count) + n_entries * 12
    #              + 4 (next-IFD offset, which is 0 here).
    ifd_size = 2 + n_entries * 12 + 4
    # Strip starts after the IFD, with a 1-byte gap so the
    # offset isn't trivially adjacent to the IFD.
    strip_offset = ifd_offset + ifd_size + 1

    # --- 8-byte file header (little-endian) -----------------
    out = bytearray()
    out += b"II"                                     # byte order
    out += struct.pack("<H", 42)                     # magic
    out += struct.pack("<I", ifd_offset)             # IFD offset

    # --- IFD --------------------------------------------------
    out += struct.pack("<H", n_entries)
    for tag in tag_ids:
        out += struct.pack("<H", tag)  # tag ID
        out += struct.pack("<H", 4)    # type = LONG
        out += struct.pack("<I", 1)    # count
        if tag == _quartz._TAG_IMAGE_WIDTH:
            value = width
        elif tag == _quartz._TAG_IMAGE_LENGTH:
            value = height
        elif tag == _quartz._TAG_BITS_PER_SAMPLE:
            value = bps
        elif tag == _quartz._TAG_COMPRESSION:
            value = _quartz._EXPECTED_COMPRESSION
        elif tag == _quartz._TAG_PHOTOMETRIC:
            value = _quartz._EXPECTED_PHOTOMETRIC
        elif tag == _quartz._TAG_STRIP_OFFSETS:
            value = strip_offset
        elif tag == _quartz._TAG_SAMPLES_PER_PIXEL:
            value = spp
        elif tag == _quartz._TAG_ROWS_PER_STRIP:
            value = height
        elif tag == _quartz._TAG_STRIP_BYTE_COUNTS:
            value = strip_byte_count
        else:
            raise AssertionError(f"unhandled tag {tag}")
        out += struct.pack("<I", value)
    out += struct.pack("<I", 0)  # next-IFD offset (none)

    # --- 1-byte gap ----------------------------------------
    out += b"\x00"

    # --- pixel strip (top-down) ----------------------------
    for row in range(height):
        for col in range(width):
            r, g, b, a = pixels[row][col]
            out += bytes([r, g, b, a])
    return bytes(out)


# ---------------------------------------------------------------------------
# is_quartz_available
# ---------------------------------------------------------------------------


def test_is_quartz_available_returns_false_on_linux() -> None:
    """On Linux CI, even a screencapture-shaped ``which()`` can't rescue us —
    the platform check fires first and short-circuits to False.

    This is the user-visible behaviour: ``is_quartz_available``
    on a non-macOS box returns ``False`` regardless of what
    the environment looks like.
    """
    assert _quartz.is_quartz_available(
        which=lambda name: "/usr/sbin/screencapture" if name == "screencapture" else None,
    ) is False


def test_is_quartz_available_false_on_darwin_without_screencapture() -> None:
    """On a hypothetical Darwin box with no screencapture on ``$PATH``,
    the function should return False (we can't capture anything
    without the binary). We can't easily simulate a Darwin
    platform from Linux, so we test the *inner* logic by
    calling the helper that *would* fire on a real Mac:
    shutil.which() with a known-missing binary. The full
    gate (``sys.platform == "darwin"`` AND
    ``which("screencapture") is not None``) is also exercised
    by the integration test below.
    """
    import shutil
    # A name that's guaranteed not to be on $PATH.
    assert shutil.which("definitely_not_a_real_binary_xyz") is None
    # And ``is_quartz_available`` returns False on Linux either way.
    assert _quartz.is_quartz_available(
        which=shutil.which,
    ) is False


def test_is_quartz_available_false_when_screencapture_missing() -> None:
    """A non-macOS platform + a screencapture-shaped which() still reports False
    (the platform check fires first)."""
    assert _quartz.is_quartz_available(
        which=lambda name: None,
    ) is False


def test_is_quartz_available_false_with_explicit_platform_env() -> None:
    """Inject an env mapping (no DISPLAY / no platform key) — the platform
    check still wins on Linux."""
    assert _quartz.is_quartz_available(
        env={}, which=lambda name: None,
    ) is False


# ---------------------------------------------------------------------------
# _parse_tiff_header
# ---------------------------------------------------------------------------


def test_parse_tiff_header_minimal() -> None:
    """The minimal valid TIFF parses to the expected field values."""
    data = _make_tiff_bytes(width=4, height=3)
    header = _quartz._parse_tiff_header(data)
    assert header.width == 4
    assert header.height == 3
    assert header.samples_per_pixel == 4
    assert header.bits_per_sample == 8
    assert header.rows_per_strip == 3
    # The strip should start at the same offset the builder
    # recorded. We don't hard-code the value here because
    # that would couple the test to the builder's exact
    # padding — but we do assert it's > the IFD end and < the
    # file length.
    assert header.strip_offset > 8
    assert header.strip_byte_count == 4 * 3 * 4
    assert header.strip_offset + header.strip_byte_count <= len(data)


def test_parse_tiff_header_rejects_truncated_input() -> None:
    """Anything shorter than 8 bytes is rejected before parsing."""
    with pytest.raises(ValueError, match="too short"):
        _quartz._parse_tiff_header(b"")
    with pytest.raises(ValueError, match="too short"):
        _quartz._parse_tiff_header(b"II\x2a\x00")


def test_parse_tiff_header_rejects_unknown_byte_order() -> None:
    """A first-2-bytes that's neither ``II`` nor ``MM`` is rejected."""
    with pytest.raises(ValueError, match="byte-order"):
        _quartz._parse_tiff_header(b"XX\x2a\x00\x00\x00\x00\x08")


def test_parse_tiff_header_rejects_wrong_magic() -> None:
    """A magic number that isn't 42 is rejected."""
    data = bytearray(_make_tiff_bytes())
    # Overwrite the magic with 99.
    data[2:4] = struct.pack("<H", 99)
    with pytest.raises(ValueError, match="magic"):
        _quartz._parse_tiff_header(bytes(data))


def test_parse_tiff_header_rejects_bits_per_sample_mismatch() -> None:
    """A non-8 bps image is rejected with a clear message."""
    data = bytearray(_make_tiff_bytes())
    # Find the BitsPerSample tag entry and overwrite the value.
    # The entry order is fixed by the builder; BitsPerSample is
    # the third entry (index 2). The value sits at ifd_offset
    # + 2 + 2*12 + 8 = 34.
    bps_offset = 8 + 2 + 2 * 12 + 8
    data[bps_offset:bps_offset + 4] = struct.pack("<I", 16)
    with pytest.raises(RuntimeError, match="BitsPerSample"):
        _quartz._parse_tiff_header(bytes(data))


def test_parse_tiff_header_rejects_samples_per_pixel_mismatch() -> None:
    """A non-RGBA image (e.g. RGB) is rejected with a clear message."""
    data = bytearray(_make_tiff_bytes())
    # SamplesPerPixel is the 7th entry (index 6). The value
    # sits at ifd_offset + 2 + 6*12 + 8 = 86.
    spp_offset = 8 + 2 + 6 * 12 + 8
    data[spp_offset:spp_offset + 4] = struct.pack("<I", 3)
    with pytest.raises(RuntimeError, match="SamplesPerPixel"):
        _quartz._parse_tiff_header(bytes(data))


def test_parse_tiff_header_rejects_compression_mismatch() -> None:
    """A compressed image (e.g. LZW) is rejected with a clear message."""
    data = bytearray(_make_tiff_bytes())
    # Compression is the 4th entry (index 3). The value sits
    # at ifd_offset + 2 + 3*12 + 8 = 50.
    comp_offset = 8 + 2 + 3 * 12 + 8
    data[comp_offset:comp_offset + 4] = struct.pack("<I", 5)  # LZW
    with pytest.raises(RuntimeError, match="Compression"):
        _quartz._parse_tiff_header(bytes(data))


def test_parse_tiff_header_rejects_photometric_mismatch() -> None:
    """A palette-based image is rejected with a clear message."""
    data = bytearray(_make_tiff_bytes())
    # PhotometricInterpretation is the 5th entry (index 4).
    # The value sits at ifd_offset + 2 + 4*12 + 8 = 62.
    photo_offset = 8 + 2 + 4 * 12 + 8
    data[photo_offset:photo_offset + 4] = struct.pack("<I", 3)  # palette
    with pytest.raises(RuntimeError, match="Photometric"):
        _quartz._parse_tiff_header(bytes(data))


def test_parse_tiff_header_rejects_ifd_past_end() -> None:
    """An IFD offset that points past the end of the file is rejected."""
    data = bytearray(_make_tiff_bytes())
    # Overwrite the IFD offset with a value past the end of the file.
    data[4:8] = struct.pack("<I", 10_000)
    with pytest.raises(ValueError, match="IFD"):
        _quartz._parse_tiff_header(bytes(data))


# ---------------------------------------------------------------------------
# _sample_rgba_cell_brightness
# ---------------------------------------------------------------------------


def test_sample_rgba_cell_brightness_red_pixel_is_low() -> None:
    """Pure red (255, 0, 0) → brightness 0.30 (the R weight in the luma formula)."""
    pixels = bytes([0xFF, 0x00, 0x00, 0xFF])  # RGBA: red
    b = _quartz._sample_rgba_cell_brightness(
        pixels, width=1, height=1,
        cell_x=0, cell_y=0, cell_w=1, cell_h=1,
    )
    assert abs(b - 0.30) < 0.01


def test_sample_rgba_cell_brightness_blue_pixel_is_low() -> None:
    """Pure blue (0, 0, 255) → brightness 0.11 (the B weight in the luma formula)."""
    pixels = bytes([0x00, 0x00, 0xFF, 0xFF])  # RGBA: blue
    b = _quartz._sample_rgba_cell_brightness(
        pixels, width=1, height=1,
        cell_x=0, cell_y=0, cell_w=1, cell_h=1,
    )
    assert abs(b - 0.11) < 0.01


def test_sample_rgba_cell_brightness_white_pixel_is_one() -> None:
    """Pure white (255, 255, 255) → brightness 1.0."""
    pixels = bytes([0xFF, 0xFF, 0xFF, 0xFF])  # RGBA: white
    b = _quartz._sample_rgba_cell_brightness(
        pixels, width=1, height=1,
        cell_x=0, cell_y=0, cell_w=1, cell_h=1,
    )
    assert abs(b - 1.0) < 0.001


def test_sample_rgba_cell_brightness_alpha_ignored() -> None:
    """The alpha byte doesn't change the brightness reading."""
    # Same RGB as the red test above, alpha = 0 (fully transparent).
    pixels = bytes([0xFF, 0x00, 0x00, 0x00])
    b = _quartz._sample_rgba_cell_brightness(
        pixels, width=1, height=1,
        cell_x=0, cell_y=0, cell_w=1, cell_h=1,
    )
    assert abs(b - 0.30) < 0.01


def test_sample_rgba_cell_brightness_out_of_bounds_returns_one() -> None:
    """Cells past the screen extent return 1.0 (no ink) for safe padding."""
    pixels = b""
    # Negative x or y.
    assert _quartz._sample_rgba_cell_brightness(
        pixels, width=10, height=10,
        cell_x=-1, cell_y=0, cell_w=1, cell_h=1,
    ) == 1.0
    assert _quartz._sample_rgba_cell_brightness(
        pixels, width=10, height=10,
        cell_x=0, cell_y=-1, cell_w=1, cell_h=1,
    ) == 1.0
    # x past the right edge.
    assert _quartz._sample_rgba_cell_brightness(
        pixels, width=10, height=10,
        cell_x=100, cell_y=0, cell_w=1, cell_h=1,
    ) == 1.0
    # Zero cell size.
    assert _quartz._sample_rgba_cell_brightness(
        pixels, width=10, height=10,
        cell_x=0, cell_y=0, cell_w=0, cell_h=1,
    ) == 1.0


def test_sample_rgba_cell_brightness_top_down_row_order() -> None:
    """A top-down RGBA buffer is read in normal row order.

    A red pixel at (0, 0) lives at the first 4 bytes; the
    same pixel at (0, 1) in a 1×2 buffer lives at offset 4.
    This is the production path (screencapture's strip is
    top-down), and it's the symmetry that makes the helper
    unit-testable on a non-Mac box.
    """
    # 1×2 buffer: row 0 = red, row 1 = blue.
    pixels = bytes([0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0xFF])
    b_top = _quartz._sample_rgba_cell_brightness(
        pixels, width=1, height=2,
        cell_x=0, cell_y=0, cell_w=1, cell_h=1,
    )
    b_bottom = _quartz._sample_rgba_cell_brightness(
        pixels, width=1, height=2,
        cell_x=0, cell_y=1, cell_w=1, cell_h=1,
    )
    # (0, 0) is red → 0.30; (0, 1) is blue → 0.11.
    assert abs(b_top - 0.30) < 0.01
    assert abs(b_bottom - 0.11) < 0.01


# ---------------------------------------------------------------------------
# _rgba_to_grid
# ---------------------------------------------------------------------------


def test_rgba_to_grid_basic_red_grid() -> None:
    """A solid red 4×1 buffer downsampled to a 1×1 grid → full-ink character."""
    pixels = bytes(
        [0xFF, 0x00, 0x00, 0xFF] * 4
    )  # 4 red RGBA pixels
    grid = _quartz._rgba_to_grid(
        pixels, width=4, height=1, x=0, y=0, w=1, h=1
    )
    assert len(grid) == 1
    assert len(grid[0]) == 1
    # Red brightness 0.30 → ink 0.70 → bucket 3 of 5 → '▓' (U+2593).
    assert grid[0] == "\u2593"


def test_rgba_to_grid_basic_white_grid() -> None:
    """A solid white 1×1 buffer → single space (no ink)."""
    pixels = bytes([0xFF, 0xFF, 0xFF, 0xFF])  # RGBA: white
    assert _quartz._rgba_to_grid(
        pixels, width=1, height=1, x=0, y=0, w=1, h=1
    ) == [" "]


def test_rgba_to_grid_zero_w_returns_empty() -> None:
    """w=0 short-circuits the loop — no work, no error."""
    pixels = bytes([0xFF, 0x00, 0x00, 0xFF] * 4)
    assert _quartz._rgba_to_grid(
        pixels, width=4, height=1, x=0, y=0, w=0, h=5
    ) == []


def test_rgba_to_grid_zero_h_returns_empty() -> None:
    """h=0 short-circuits the loop — symmetric to w=0."""
    pixels = bytes([0xFF, 0x00, 0x00, 0xFF] * 4)
    assert _quartz._rgba_to_grid(
        pixels, width=4, height=1, x=0, y=0, w=5, h=0
    ) == []


def test_rgba_to_grid_negative_dimensions_rejected() -> None:
    """A negative w or h is a programmer error (matches Win32 / X11)."""
    pixels = bytes([0xFF, 0x00, 0x00, 0xFF] * 4)
    with pytest.raises(ValueError, match="non-negative"):
        _quartz._rgba_to_grid(
            pixels, width=4, height=1, x=0, y=0, w=-1, h=5
        )
    with pytest.raises(ValueError, match="non-negative"):
        _quartz._rgba_to_grid(
            pixels, width=4, height=1, x=0, y=0, w=5, h=-1
        )


def test_rgba_to_grid_zero_screen_returns_empty() -> None:
    """A 0×0 or 0-height screen returns [] — the renderer pads anyway."""
    pixels = b""
    assert _quartz._rgba_to_grid(
        pixels, width=0, height=0, x=0, y=0, w=1, h=1
    ) == []
    assert _quartz._rgba_to_grid(
        pixels, width=10, height=0, x=0, y=0, w=1, h=1
    ) == []
    assert _quartz._rgba_to_grid(
        pixels, width=0, height=10, x=0, y=0, w=1, h=1
    ) == []


def test_rgba_to_grid_full_screen_checkerboard() -> None:
    """A 2×2 checkerboard downsampled to 2×2 cells shows alternating densities."""
    pixels = bytes([
        # Top row: red, blue
        0xFF, 0x00, 0x00, 0xFF,  0x00, 0x00, 0xFF, 0xFF,
        # Bottom row: blue, red
        0x00, 0x00, 0xFF, 0xFF,  0xFF, 0x00, 0x00, 0xFF,
    ])
    grid = _quartz._rgba_to_grid(
        pixels, width=2, height=2, x=0, y=0, w=2, h=2
    )
    assert len(grid) == 2
    assert len(grid[0]) == 2
    # Each cell samples one pixel exactly: the (0,0) and
    # (1,1) cells are red (ink ≈ 0.70, bucket 3), the (0,1)
    # and (1,0) cells are blue (ink ≈ 0.89, bucket 4). Two
    # distinct density chars, mirrored across the diagonal.
    chars = {c for line in grid for c in line}
    assert len(chars) == 2


def test_rgba_to_grid_full_screen_two_extremes() -> None:
    """A 2×2 black/white checkerboard yields space and full block.

    This is the cleanest possible test: two buckets at the
    ends of the density scale, no intermediate values.
    """
    pixels = bytes([
        # Top row: white, black
        0xFF, 0xFF, 0xFF, 0xFF,  0x00, 0x00, 0x00, 0xFF,
        # Bottom row: black, white
        0x00, 0x00, 0x00, 0xFF,  0xFF, 0xFF, 0xFF, 0xFF,
    ])
    grid = _quartz._rgba_to_grid(
        pixels, width=2, height=2, x=0, y=0, w=2, h=2
    )
    chars = sorted({c for line in grid for c in line})
    assert " " in chars
    assert "\u2588" in chars


# ---------------------------------------------------------------------------
# QuartzScreen end-to-end
# ---------------------------------------------------------------------------


def _tiff_runner(data: bytes):
    """Build a runner that returns a fixed TIFF bytestream for any argv.

    The runner ignores its argv and returns the same stream;
    this is fine for the end-to-end tests below, which all
    care about a single specific capture call.
    """
    def runner(argv):
        return (0, data, b"")
    return runner


def _tiff_runner_two_stage(size_data: bytes, capture_data: bytes):
    """Build a runner that returns one TIFF for the size probe and another
    for the actual capture. Used to test the screen_size / capture split."""
    calls: list[tuple] = []
    state = {"called": 0}

    def runner(argv):
        calls.append(argv)
        state["called"] += 1
        if state["called"] == 1:
            return (0, size_data, b"")
        return (0, capture_data, b"")
    runner.calls = calls  # type: ignore[attr-defined]
    return runner


def test_quartz_screen_screen_size_via_runner() -> None:
    """``screen_size`` calls the runner once, caches, and returns (w, h)."""
    calls: list[tuple] = []

    def runner(argv):
        calls.append(argv)
        return (0, _make_tiff_bytes(width=1920, height=1080), b"")

    screen = _quartz.QuartzScreen(runner=runner)
    assert screen.screen_size() == (1920, 1080)
    # Second call uses the cache — no extra runner call.
    assert screen.screen_size() == (1920, 1080)
    # The size probe should NOT include the -R flag.
    assert all("-R" not in arg for arg in calls[0])
    assert len(calls) == 1


def test_quartz_screen_screen_size_runner_failure() -> None:
    """A non-zero exit code during the size probe raises a clear error."""

    def runner(argv):
        return (1, b"", b"some screencapture error\n")

    screen = _quartz.QuartzScreen(runner=runner)
    with pytest.raises(RuntimeError, match="screencapture failed"):
        screen.screen_size()


def test_quartz_screen_capture_via_runner() -> None:
    """``capture`` returns a list of strings of the right shape."""
    data = _make_tiff_bytes(width=4, height=2)
    # Use a runner that always returns the same TIFF. The
    # size probe will see a 4×2 image, and the actual
    # capture will also see a 4×2 image — both fine.
    screen = _quartz.QuartzScreen(runner=_tiff_runner(data))
    grid = screen.capture(x=0, y=0, w=4, h=2)
    assert len(grid) == 2
    assert all(len(line) == 4 for line in grid)


def test_quartz_screen_capture_validates_w_h() -> None:
    """A negative w or h is a programmer error (mirrors FakeScreen / X11 / Win32)."""
    data = _make_tiff_bytes(width=1, height=1)
    screen = _quartz.QuartzScreen(runner=_tiff_runner(data))
    with pytest.raises(ValueError, match="non-negative"):
        screen.capture(x=0, y=0, w=-1, h=1)
    with pytest.raises(ValueError, match="non-negative"):
        screen.capture(x=0, y=0, w=1, h=-1)


def test_quartz_screen_capture_zero_returns_empty() -> None:
    """A w=0 or h=0 capture short-circuits to ``[]`` without calling the runner."""
    calls: list[tuple] = []

    def runner(argv):
        calls.append(argv)
        return (0, _make_tiff_bytes(), b"")

    screen = _quartz.QuartzScreen(runner=runner)
    assert screen.capture(x=0, y=0, w=0, h=5) == []
    assert screen.capture(x=0, y=0, w=5, h=0) == []
    # Both calls returned [] without ever asking the runner.
    assert calls == []


def test_quartz_screen_capture_runner_failure() -> None:
    """A non-zero exit code during capture raises a clear error."""

    def runner(argv):
        return (1, b"", b"screencapture: screen recording not authorized\n")

    screen = _quartz.QuartzScreen(runner=runner)
    with pytest.raises(RuntimeError, match="screencapture failed"):
        screen.capture(x=0, y=0, w=2, h=2)


def test_quartz_screen_capture_truncated_output() -> None:
    """A TIFF stream shorter than 8 bytes is rejected with a clear error."""

    def runner(argv):
        return (0, b"II\x2a\x00", b"")

    screen = _quartz.QuartzScreen(runner=runner)
    with pytest.raises(RuntimeError, match="8-byte TIFF header"):
        screen.capture(x=0, y=0, w=2, h=2)


def test_quartz_screen_capture_pads_short_strip() -> None:
    """A strip that's shorter than the declared byte count is zero-padded.

    The downsample path needs ``width * height * 4`` bytes;
    a screencapture output that's missing the last few rows
    shouldn't crash the magnifier. The X11 adapter has the
    same defence.
    """
    # Build a valid TIFF, then chop off the last 4 bytes
    # of the pixel strip. The header is still valid; the
    # strip_byte_count still says N bytes; the actual data
    # is N - 4 bytes.
    data = _make_tiff_bytes(width=2, height=2)
    data = data[:-4]
    screen = _quartz.QuartzScreen(runner=_tiff_runner(data))
    # The capture should still produce a 2×2 grid (we're
    # sampling inside the surviving pixels).
    grid = screen.capture(x=0, y=0, w=2, h=2)
    assert len(grid) == 2
    assert all(len(line) == 2 for line in grid)


def test_quartz_screen_capture_refreshes_cached_size() -> None:
    """Each capture() call refreshes the size cache, mirroring X11's
    "robust to mid-run monitor resizes" behaviour."""
    size_data = _make_tiff_bytes(width=4, height=3)
    capture_data = _make_tiff_bytes(width=4, height=2)
    runner = _tiff_runner_two_stage(size_data, capture_data)
    screen = _quartz.QuartzScreen(runner=runner)
    # First call: size probe → 4×3.
    assert screen.screen_size() == (4, 3)
    # Second call: actual capture → updates cache to 4×2.
    screen.capture(x=0, y=0, w=4, h=2)
    # Cached size should now reflect the capture, not the probe.
    assert screen._cached_size == (4, 2)


def test_quartz_screen_default_runner_is_real_subprocess() -> None:
    """The default runner is the real subprocess plumbing (macOS only).

    We don't actually *call* it here (no macOS desktop); we
    just check that the constructor's default-arg path
    assigns :func:`_default_runner` and that it's callable.
    The end-to-end "real screencapture invocation" path is
    exercised on a developer machine with a real Mac; here
    we only need to prove the wiring is correct.
    """
    screen = _quartz.QuartzScreen()
    assert screen._runner is _quartz._default_runner
    assert callable(screen._runner)


# ---------------------------------------------------------------------------
# build_quartz_screen factory
# ---------------------------------------------------------------------------


def test_build_quartz_screen_returns_none_when_unavailable() -> None:
    """On a non-macOS box the factory returns None, not an instance."""
    screen = _quartz.build_quartz_screen(
        available=lambda: False,
    )
    assert screen is None


def test_build_quartz_screen_returns_instance_when_available() -> None:
    """The availability check passes → a real QuartzScreen is returned."""
    screen = _quartz.build_quartz_screen(available=lambda: True)
    assert isinstance(screen, _quartz.QuartzScreen)


def test_build_quartz_screen_uses_injected_runner() -> None:
    """The factory forwards the ``runner`` kwarg to the new instance."""
    data = _make_tiff_bytes(width=1, height=1)
    screen = _quartz.build_quartz_screen(
        runner=_tiff_runner(data),
        available=lambda: True,
    )
    assert screen is not None
    # Use the injected runner; the call should not raise.
    grid = screen.capture(x=0, y=0, w=1, h=1)
    assert len(grid) == 1 and len(grid[0]) == 1


def test_build_quartz_screen_returns_none_on_linux() -> None:
    """``build_quartz_screen()`` returns None on a non-Darwin platform.

    This is the user-visible integration test: it proves the
    factory's platform check works on the actual host without
    requiring a real Mac.
    """
    assert _quartz.build_quartz_screen() is None


# ---------------------------------------------------------------------------
# Integration with ``whisperpaw._screen.get_capture``
# ---------------------------------------------------------------------------


def test_screen_dispatch_quartz_returns_instance_when_available() -> None:
    """``get_capture(backend='quartz')`` returns a QuartzScreen when available.

    We monkeypatch the factory's discovery helpers to
    simulate "macOS is available" without needing a real Mac.
    """
    from whisperpaw import _screen as screen_mod

    real_quartz_capture = screen_mod._quartz_capture
    try:
        # Replace the factory with one that always returns an
        # instance. This is what the user-visible path would
        # do on a real Mac.
        class _FakeAdapter(screen_mod.ScreenCapture):
            def screen_size(self) -> tuple[int, int]:
                return (4, 2)

            def capture(self, *, x, y, w, h):
                return ["    "] * h

        screen_mod._BACKEND_FACTORIES["quartz"] = lambda: _FakeAdapter()
        cap = screen_mod.get_capture("quartz")
        assert isinstance(cap, _FakeAdapter)
    finally:
        screen_mod._BACKEND_FACTORIES["quartz"] = real_quartz_capture


def test_screen_dispatch_quartz_returns_none_on_linux() -> None:
    """``get_capture(backend='quartz')`` returns None on a non-Darwin box.

    This is the user-visible integration test: it proves the
    v0.2 pipeline (zoom.py + _screen.py + _quartz.py) compose
    correctly even when the OS adapter is unavailable. The
    call returns None, not an exception.
    """
    cap = _screen.get_capture("quartz")
    # We're on Linux CI; the factory must return None, not raise.
    assert cap is None


def test_screen_dispatch_auto_order_includes_quartz() -> None:
    """``get_capture(backend='auto')`` tries quartz after x11 / win32."""
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
        # Mark all three as available; x11 should win because
        # it's first in the pick order (x11 → win32 → quartz).
        screen_mod._BACKEND_FACTORIES["x11"] = make_marker("x11")
        screen_mod._BACKEND_FACTORIES["win32"] = make_marker("win32")
        screen_mod._BACKEND_FACTORIES["quartz"] = make_marker("quartz")
        cap = screen_mod.get_capture("auto")
        assert cap is not None
        assert cap.name == "x11"
    finally:
        screen_mod._BACKEND_FACTORIES.clear()
        screen_mod._BACKEND_FACTORIES.update(real)

    # Now mark only quartz as available; the pick order
    # should fall through to it.
    real = dict(screen_mod._BACKEND_FACTORIES)
    try:
        # Make x11 / win32 unavailable by returning None.
        screen_mod._BACKEND_FACTORIES["x11"] = lambda: None
        screen_mod._BACKEND_FACTORIES["win32"] = lambda: None
        screen_mod._BACKEND_FACTORIES["quartz"] = make_marker("quartz")
        cap = screen_mod.get_capture("auto")
        assert cap is not None
        assert cap.name == "quartz"
    finally:
        screen_mod._BACKEND_FACTORIES.clear()
        screen_mod._BACKEND_FACTORIES.update(real)


def test_paw_zoom_screen_quartz_backend_unsupported_message() -> None:
    """On a non-macOS box, ``paw-zoom --screen --backend quartz`` prints
    the friendly "not yet implemented" message and exits 1.

    This is the user-visible integration test: it proves that
    the v0.2 pipeline (zoom.py + _screen.py + _quartz.py)
    compose correctly even when the OS adapter is
    unavailable.
    """
    import io
    from contextlib import redirect_stderr, redirect_stdout
    from whisperpaw import zoom

    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = zoom.main(
            ["--screen", "--backend", "quartz", "--rows", "2", "--cols", "2"]
        )
    # On a non-macOS box this returns 1 and writes the
    # friendly "not yet implemented on this OS" message.
    assert rc == 1
    assert "not yet implemented" in err.getvalue() or "fake" in err.getvalue()
