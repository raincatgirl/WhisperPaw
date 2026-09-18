"""macOS Quartz screen-capture adapter for ``paw-zoom`` v0.2.

This module ships the third *real* OS-specific
:class:`whisperpaw._screen.ScreenCapture` adapter: one that talks
to the macOS screen via the built-in ``screencapture`` CLI and
turns the resulting TIFF into the same five-character density
grid the X11 / Win32 adapters emit.

Why a CLI instead of CoreGraphics via ``ctypes``
------------------------------------------------
The "real" way to capture a screen region on macOS is to call
the CoreGraphics / ImageIO C APIs directly — ``CGDirectDisplayID``
+ ``CGDisplayBounds`` for size, ``CGDisplayCreateImage`` for
the bitmap, ``CGImageGetDataProvider`` + ``CGDataProviderCopyData``
for the pixel bytes, ``CGImageGetBytesPerRow`` /
``CGImageGetWidth`` / ``CGImageGetHeight`` for the metadata.
That is roughly 200 lines of ctypes plumbing, the kind of which
we explicitly avoided in the Win32 adapter (GDI was already
intrusive enough). More importantly, it locks the module to a
single OS — we couldn't even *import* it on a Linux CI box.

The ``screencapture`` CLI ships with every macOS install since
10.4, lives in ``/usr/sbin``, and emits a region-cropped TIFF
to stdout when invoked as ``screencapture -R x,y,w,h -t tiff -``.
That gives us:

* a one-line subprocess call (with stdout piped, the same
  shape as the X11 ``xwd -root -silent -out -`` invocation);
* a real, no-deps TIFF bitstream to parse — TIFF is a tiny
  fixed-format container (8-byte header + IFD + strip), and
  the only fields we care about are ``ImageWidth``,
  ``ImageLength``, ``BitsPerSample``, ``SamplesPerPixel``,
  ``RowsPerStrip``, ``StripOffsets``, ``StripByteCounts``, and
  ``PhotometricInterpretation``;
* the same lazy-import / injectable-runner architecture the
  X11 adapter uses, so the module imports on every platform
  and the tests synthesise pixel buffers in memory.

What we capture and how it's laid out
-------------------------------------
``screencapture -R`` captures a region of the *primary* display
in display coordinates (origin top-left of the screen, units =
points, not pixels — but for our magnifier purposes the
"density" output is unit-agnostic: we downsample a screen
rectangle to a fixed number of character cells regardless of
the source pixel density). The output TIFF is 32-bpp RGBA
(``SamplesPerPixel = 4``, ``BitsPerSample = 8``,
``PhotometricInterpretation = 2`` = RGB) by default; we
restrict ourselves to that configuration and raise a
:class:`RuntimeError` for anything else (a 16-bit or paletted
screencapture output is unusual enough that we want a loud
failure rather than silent mis-rendering).

Pixel format reminder
---------------------
Each pixel is 4 bytes: R, G, B, A (the alpha byte is ignored —
we only need a brightness reading). Rows are top-down
(``screencapture``'s default — the strip contains row 0 first
when the IFD's ``RowsPerStrip`` is the full image height).

Why not just decode the pixels into RGB
---------------------------------------
We don't try to be a real screen-reader. ``paw-zoom``'s job is
to *magnify a region* — the user already knows what the screen
looks like, they just can't see the detail. So we downsample
each cell of the target character grid to a single "ink
density" character picked from :data:`_DENSITY_CHARS`. The
character is a function of the average brightness of the few
pixels we sample in that cell, not a function of the actual
glyphs on the screen. This keeps the adapter small (~250 lines
incl. comments) and gives the magnifier a meaningful preview.

Public API
----------
* :class:`QuartzScreen` — the :class:`ScreenCapture` subclass.
* :class:`TiffHeader` — a NamedTuple with the seven TIFF
  fields the adapter actually uses.
* :func:`is_quartz_available` — ``True`` if the current process
  is on Darwin (``sys.platform == "darwin"``) and the
  ``screencapture`` binary is on ``$PATH``. The subprocess
  plumbing is lazy: an attempt to construct the adapter on
  Linux or Windows just returns ``None`` from
  :func:`build_quartz_screen`.
* :func:`build_quartz_screen` — construct a :class:`QuartzScreen`,
  with injectable ``runner=`` / ``available=`` / ``which=`` hooks
  for tests.
* :func:`_rgba_to_grid` — pure function: raw RGBA pixel bytes +
  image size + ``(x, y, w, h)`` → list[str] of the same shape
  :class:`whisperpaw._screen.FakeScreen.capture` returns.
* :func:`_sample_rgba_cell_brightness` — the per-cell sampling
  helper.

Pure stdlib, no third-party deps, no telemetry, no network.
"""
from __future__ import annotations

import os
import shutil
import struct
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple

from whisperpaw import _screen


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------


#: The five characters we downsample a cell to, sorted from
#: "no ink" to "full ink". Same shape as the X11 / Win32 adapters
#: so the rendered output is consistent across all three OS adapters.
_DENSITY_CHARS: str = " \u2591\u2592\u2593\u2588"  # space, ░, ▒, ▓, █


#: TIFF "byte-order" magic for little-endian files
#: (``screencapture`` defaults to LE on every macOS we tested).
_TIFF_MAGIC_LE: bytes = b"II"


#: ``screencapture`` invocation for a region capture, TIFF to
#: stdout. ``-R`` takes ``x,y,w,h`` in display coordinates
#: (top-left origin), ``-t tiff`` selects the container,
#: ``-`` writes the dump to stdout. ``-x`` suppresses the
#: default shutter sound, which is a nicety for the
#: accessibility use case but not strictly necessary.
_SCREENCAPTURE_INVOCATION: tuple[str, ...] = (
    "screencapture",
    "-x",
    "-R", "0,0,0,0",  # placeholder; the adapter substitutes the real region
    "-t", "tiff",
    "-",
)


#: The TIFF tag IDs we read from the IFD. Defined as module
#: constants so the parser doesn't have a magic-number salad.
_TAG_IMAGE_WIDTH: int = 256
_TAG_IMAGE_LENGTH: int = 257
_TAG_BITS_PER_SAMPLE: int = 258
_TAG_COMPRESSION: int = 259
_TAG_PHOTOMETRIC: int = 262
_TAG_STRIP_OFFSETS: int = 273
_TAG_SAMPLES_PER_PIXEL: int = 277
_TAG_ROWS_PER_STRIP: int = 278
_TAG_STRIP_BYTE_COUNTS: int = 279


#: What we expect: 32-bpp RGBA, no compression, RGB photometric.
#: Any other combination raises a clear RuntimeError so the
#: adapter doesn't silently mis-render a 16-bit or paletted image.
_EXPECTED_BITS_PER_SAMPLE: int = 8
_EXPECTED_SAMPLES_PER_PIXEL: int = 4
_EXPECTED_COMPRESSION: int = 1  # no compression
_EXPECTED_PHOTOMETRIC: int = 2  # RGB


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def is_quartz_available(
    *,
    env: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] | None = None,
) -> bool:
    """Return ``True`` if a Quartz capture is likely to work here.

    Two cheap checks: ``sys.platform == "darwin"`` (Quartz is
    the macOS display server; the name comes from the
    ``Quartz Compositor`` introduced in 10.4) and the
    ``screencapture`` binary is on ``$PATH``. Neither is
    *sufficient* on its own — a headless macOS user with no
    window server will still fail at capture time — but
    together they catch the common "Linux CI" / "Windows
    desktop" cases before we even try to fork ``screencapture``.

    The ``env`` and ``which`` parameters exist so tests can
    simulate a different platform without monkeypatching
    ``sys.platform`` (a process-wide global and a bad idea to
    mutate).
    """
    if sys.platform != "darwin":
        return False
    if env is None:
        env = os.environ
    if which is None:
        which = shutil.which
    return which("screencapture") is not None


# ---------------------------------------------------------------------------
# TIFF parsing
# ---------------------------------------------------------------------------


class TiffHeader(NamedTuple):
    """The slice of a TIFF header ``paw-zoom`` actually uses.

    The TIFF spec defines dozens of tags; we only read the
    seven the downsample path needs. ``strip_offset`` is the
    byte offset from the start of the file to the pixel data;
    ``strip_byte_count`` is the number of bytes of pixel data.
    Both can be larger than 4 bytes if the image is > 4 GiB —
    we don't support that, since ``screencapture`` won't
    produce one either.
    """

    width: int
    height: int
    samples_per_pixel: int
    bits_per_sample: int
    rows_per_strip: int
    strip_offset: int
    strip_byte_count: int


def _parse_tiff_header(data: bytes) -> TiffHeader:
    """Parse a minimal TIFF header and return the seven fields
    the downsample path needs.

    TIFF begins with either ``II`` (little-endian) or ``MM``
    (big-endian), then a 2-byte magic number ``42``, then a
    4-byte offset to the first IFD. The IFD is a 2-byte entry
    count followed by that many 12-byte tag entries; we scan
    the IFD linearly for the seven tags we care about, and
    ignore the rest.

    The format the adapter accepts is narrow by design: only
    32-bpp RGBA, no compression, RGB photometric. Anything
    else raises :class:`RuntimeError` so we fail loudly rather
    than silently mis-rendering.
    """
    if len(data) < 8:
        raise ValueError(
            f"tiff: header is too short (got {len(data)} bytes, need >= 8)"
        )
    bo = data[:2]
    if bo == _TIFF_MAGIC_LE:
        endian = "<"
    elif bo == b"MM":
        endian = ">"
    else:
        raise ValueError(
            f"tiff: unknown byte-order marker {bo!r} (expected II or MM)"
        )
    magic, = struct.unpack(endian + "H", data[2:4])
    if magic != 42:
        raise ValueError(
            f"tiff: magic number is {magic}, expected 42 (the only TIFF magic)"
        )
    ifd_offset, = struct.unpack(endian + "I", data[4:8])
    if ifd_offset + 2 > len(data):
        raise ValueError(
            f"tiff: IFD offset {ifd_offset} is past end of file "
            f"({len(data)} bytes)"
        )
    n_entries, = struct.unpack(endian + "H", data[ifd_offset:ifd_offset + 2])
    ifd_end = ifd_offset + 2 + n_entries * 12
    if ifd_end > len(data):
        raise ValueError(
            f"tiff: IFD ends at byte {ifd_end}, past end of file "
            f"({len(data)} bytes)"
        )

    # Pre-fill with values that will trigger a clear error if
    # the IFD doesn't contain the tag we need.
    found: dict[int, int] = {}

    def read_value(tag: int, type_id: int) -> int:
        """Read a single value of the given TIFF type.

        We only ever read SHORT (type 3, 2 bytes) or LONG
        (type 4, 4 bytes); TIFF also defines BYTE / ASCII /
        RATIONAL / etc., but none of the seven tags we care
        about are those types. A mismatched type is a corrupt
        image — we raise, the caller surfaces the error.
        """
        if type_id == 3:  # SHORT
            return struct.unpack(endian + "H", raw)[0]
        if type_id == 4:  # LONG
            return struct.unpack(endian + "I", raw)[0]
        raise ValueError(
            f"tiff: tag {tag} has unexpected type {type_id} "
            f"(only SHORT / LONG supported)"
        )

    for i in range(n_entries):
        entry_offset = ifd_offset + 2 + i * 12
        entry = data[entry_offset:entry_offset + 12]
        tag, type_id, count = struct.unpack(endian + "HHI", entry[:8])
        # ``count`` is the number of values, not bytes. SHORT
        # is 2 bytes, LONG is 4 bytes; we only ever read a
        # single value, so the relevant bytes are the first
        # ``sizeof(type)`` bytes of the value/offset field.
        if type_id == 3:
            value_size = 2
        elif type_id == 4:
            value_size = 4
        else:
            # The ``raw`` slice is only used for SHORT / LONG,
            # so we point it at the value/offset field and let
            # the read_value() error path handle the unknown
            # type. This keeps the parser simple — a single
            # branch handles "unknown type" for any tag.
            value_size = 4
        raw = entry[8:8 + value_size]
        if count == 1:
            # The value fits in the value/offset field directly.
            value = read_value(tag, type_id)
        else:
            # The value/offset field is an offset to the actual
            # data. We don't follow it — none of the tags we
            # care about use multi-value storage in the
            # screencapture output. A multi-value tag in the
            # tag list is therefore a "we don't support this"
            # signal.
            raise ValueError(
                f"tiff: tag {tag} has count={count} (>1), "
                f"but the adapter only reads single-value tags"
            )
        found[tag] = value

    try:
        width = found[_TAG_IMAGE_WIDTH]
        height = found[_TAG_IMAGE_LENGTH]
        bps = found[_TAG_BITS_PER_SAMPLE]
        compression = found[_TAG_COMPRESSION]
        photometric = found[_TAG_PHOTOMETRIC]
        spp = found[_TAG_SAMPLES_PER_PIXEL]
        rps = found[_TAG_ROWS_PER_STRIP]
        strip_offset = found[_TAG_STRIP_OFFSETS]
        strip_byte_count = found[_TAG_STRIP_BYTE_COUNTS]
    except KeyError as exc:
        raise ValueError(
            f"tiff: required tag {exc.args[0]} missing from IFD"
        ) from exc

    if bps != _EXPECTED_BITS_PER_SAMPLE:
        raise RuntimeError(
            f"quartz capture: BitsPerSample={bps}, expected "
            f"{_EXPECTED_BITS_PER_SAMPLE} "
            f"(screencapture defaults to 8; pass a different "
            f"format and the adapter will refuse rather than "
            f"mis-render)"
        )
    if spp != _EXPECTED_SAMPLES_PER_PIXEL:
        raise RuntimeError(
            f"quartz capture: SamplesPerPixel={spp}, expected "
            f"{_EXPECTED_SAMPLES_PER_PIXEL} (RGBA only)"
        )
    if compression != _EXPECTED_COMPRESSION:
        raise RuntimeError(
            f"quartz capture: Compression={compression}, expected "
            f"{_EXPECTED_COMPRESSION} (no compression only)"
        )
    if photometric != _EXPECTED_PHOTOMETRIC:
        raise RuntimeError(
            f"quartz capture: PhotometricInterpretation="
            f"{photometric}, expected {_EXPECTED_PHOTOMETRIC} (RGB)"
        )

    return TiffHeader(
        width=width,
        height=height,
        samples_per_pixel=spp,
        bits_per_sample=bps,
        rows_per_strip=rps,
        strip_offset=strip_offset,
        strip_byte_count=strip_byte_count,
    )


# ---------------------------------------------------------------------------
# Pixel decoding (pure functions — testable on any platform)
# ---------------------------------------------------------------------------


def _sample_rgba_cell_brightness(
    pixels: bytes,
    *,
    width: int,
    height: int,
    cell_x: int,
    cell_y: int,
    cell_w: int,
    cell_h: int,
) -> float:
    """Return the average brightness (0.0 = black, 1.0 = white) of
    one cell in an RGBA pixel grid.

    ``cell_w`` and ``cell_h`` are the *target* character cell
    dimensions; the actual sample point is a single pixel in
    the middle of the cell. We don't average multiple pixels
    per cell because that would slow the magnifier down
    significantly for a feature whose output is five discrete
    characters — one sample is more than enough resolution.

    TIFF ``StripOffsets`` data for ``screencapture`` is
    *top-down* by default (the strip contains row 0 first
    when ``RowsPerStrip`` is the full image height), so this
    helper reads in normal row order — no bottom-up inversion
    needed (unlike the GDI adapter).

    Out-of-bounds cells return 1.0 (white / "no ink") so the
    renderer can pad without special-casing the adapter.
    """
    if cell_x < 0 or cell_y < 0 or cell_w < 1 or cell_h < 1:
        return 1.0
    px = cell_x + cell_w // 2
    py = cell_y + cell_h // 2
    if px < 0 or py < 0 or px >= width or py >= height:
        return 1.0
    # 4 bytes per pixel: R, G, B, A (alpha ignored).
    offset = (py * width + px) * 4
    if offset + 3 >= len(pixels):
        return 1.0
    r = pixels[offset]
    g = pixels[offset + 1]
    b = pixels[offset + 2]
    # Rec.601 luma weights. Same formula the X11 / Win32
    # adapters use, so a magnifier user sees the same density
    # for the same pixel regardless of which OS adapter is
    # active.
    brightness = (r * 0.30 + g * 0.59 + b * 0.11) / 255.0
    if brightness < 0.0:
        return 0.0
    if brightness > 1.0:
        return 1.0
    return brightness


def _rgba_to_grid(
    pixels: bytes,
    *,
    width: int,
    height: int,
    x: int,
    y: int,
    w: int,
    h: int,
) -> list[str]:
    """Convert a sub-rectangle of an RGBA pixel buffer into a
    ``h``×``w`` list of density strings, the same shape
    :class:`whisperpaw._screen.FakeScreen.capture` returns.

    Each cell of the output is the *density character* whose
    bucket contains the cell's average brightness, so a
    magnifier user sees a coarse "where the ink is" view of
    the screen — the actual glyphs are not recovered.
    """
    if w < 0 or h < 0:
        raise ValueError(
            f"_rgba_to_grid: w/h must be non-negative (got w={w}, h={h})"
        )
    if w == 0 or h == 0:
        return []
    if width <= 0 or height <= 0:
        return []
    # Each target character cell covers (width / w) pixels
    # horizontally and (height / h) pixels vertically. We use
    # the target grid dimensions as the divisor so a request
    # for a 5×2 character grid on a 1920×1080 screen gives
    # each cell ~384 pixels horizontally and ~540 pixels
    # vertically — plenty of resolution for a single sample.
    cell_w = max(1, width // w)
    cell_h = max(1, height // h)
    out: list[str] = []
    n_chars = len(_DENSITY_CHARS)
    for row in range(h):
        line_chars: list[str] = []
        for col in range(w):
            cell_x = x * cell_w + col * cell_w
            cell_y = y * cell_h + row * cell_h
            brightness = _sample_rgba_cell_brightness(
                pixels,
                width=width,
                height=height,
                cell_x=cell_x,
                cell_y=cell_y,
                cell_w=cell_w,
                cell_h=cell_h,
            )
            # Map [0.0, 1.0] to [0, n_chars - 1] inclusive.
            # 0.0 is full ink (black, block char), 1.0 is no
            # ink (white, space) — the opposite of the
            # _DENSITY_CHARS string order, so we invert.
            ink = 1.0 - brightness
            bucket = int(ink * n_chars)
            if bucket >= n_chars:
                bucket = n_chars - 1
            if bucket < 0:
                bucket = 0
            line_chars.append(_DENSITY_CHARS[bucket])
        out.append("".join(line_chars))
    return out


# ---------------------------------------------------------------------------
# Subprocess runner (injectable for tests)
# ---------------------------------------------------------------------------


#: Signature of the subprocess runner. Returns
#: ``(returncode, stdout, stderr)`` — same shape as the X11
#: adapter's runner, so test fakes are interchangeable.
Runner = Callable[[Sequence[str]], tuple[int, bytes, bytes]]


def _default_runner(argv: Sequence[str]) -> tuple[int, bytes, bytes]:
    """The real subprocess runner — used in production.

    We capture both stdout (the TIFF bytes) and stderr (any
    error from ``screencapture``) so the caller can surface
    them. ``check=False`` because we want to convert the
    non-zero exit code into a friendly :class:`RuntimeError`,
    not a ``CalledProcessError`.
    """
    import subprocess
    proc = subprocess.run(
        list(argv),
        capture_output=True,
        check=False,
    )
    return (proc.returncode, proc.stdout, proc.stderr)


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


class QuartzScreen(_screen.ScreenCapture):
    """A :class:`ScreenCapture` backed by ``screencapture`` on macOS.

    Construction is lazy: we only shell out to ``screencapture``
    on the first :meth:`screen_size` or :meth:`capture`. That
    way, an attempt to instantiate the adapter on a system
    without macOS doesn't immediately fail — the failure is
    deferred to the first capture call, which is also when the
    CLI's friendly "not yet implemented" message is most
    useful.

    The class is intentionally simple: one ``_cached_size``
    slot, one ``_runner`` slot, no state machine. The
    pipeline that consumes the capture doesn't care that
    we're a Quartz adapter — it just sees a
    :class:`ScreenCapture` whose ``screen_size()`` returns
    ``(w, h)`` and whose ``capture()`` returns a list of
    strings.
    """

    __slots__ = ("_runner", "_cached_size")

    def __init__(
        self,
        *,
        runner: Runner | None = None,
    ) -> None:
        if runner is None:
            runner = _default_runner
        self._runner: Runner = runner
        self._cached_size: tuple[int, int] | None = None

    def screen_size(self) -> tuple[int, int]:
        """Return the primary display's pixel size.

        We don't have a separate "size-only" CLI on macOS
        (the equivalent would be a CoreGraphics call that we
        can't make without a 200-line ctypes shim), so the
        implementation captures the *whole* primary display
        via ``screencapture`` (no ``-R`` flag) and reads
        ``ImageWidth`` / ``ImageLength`` out of the resulting
        TIFF header. That's a single subprocess call on the
        first ``screen_size()`` — slow, but cached.
        Subsequent calls return the cached value (the screen
        size doesn't change at runtime on a typical setup;
        if the user resizes monitors while ``paw-zoom`` is
        running, they'll need to restart the loop).
        """
        if self._cached_size is not None:
            return self._cached_size
        # ``screencapture -t tiff -`` (no ``-R``) dumps the
        # full primary display. The resulting TIFF header's
        # width / height IS the primary display extent.
        argv = ("screencapture", "-x", "-t", "tiff", "-")
        rc, stdout, stderr = self._runner(argv)
        if rc != 0:
            msg = stderr.decode("utf-8", errors="replace").strip() or (
                f"screencapture exited with code {rc}"
            )
            raise RuntimeError(
                f"quartz capture: screencapture failed during "
                f"screen-size probe: {msg}"
            )
        if len(stdout) < 8:
            raise RuntimeError(
                f"quartz capture: screencapture produced only "
                f"{len(stdout)} bytes during screen-size probe; "
                f"expected at least the 8-byte TIFF header"
            )
        header = _parse_tiff_header(stdout)
        self._cached_size = (header.width, header.height)
        return self._cached_size

    def capture(
        self, *, x: int, y: int, w: int, h: int
    ) -> list[str]:
        """Return a ``h``×``w`` sub-grid of the screen as a list of strings.

        Out-of-bounds regions return padded space strings so
        the downstream renderer doesn't have to special-case
        them — see :class:`whisperpaw._screen.FakeScreen` for
        the exact same convention.
        """
        if w < 0 or h < 0:
            raise ValueError(
                f"capture: w/h must be non-negative (got w={w}, h={h})"
            )
        if w == 0 or h == 0:
            return []
        header, pixels = self._run_screencapture(x, y, w, h)
        # Refresh the cached size every capture — it's free
        # (we already have the header) and it makes the
        # adapter robust to mid-run monitor resizes.
        self._cached_size = (header.width, header.height)
        return _rgba_to_grid(
            pixels,
            width=header.width,
            height=header.height,
            x=0, y=0, w=w, h=h,
        )

    def _run_screencapture(
        self, x: int, y: int, w: int, h: int
    ) -> tuple[TiffHeader, bytes]:
        """Invoke ``screencapture`` and return ``(header, pixel_bytes)``.

        The pixel bytes are exactly the RGBA strip — no
        container overhead — so the downsample loop can
        consume them directly.
        """
        argv = (
            "screencapture",
            "-x",
            "-R", f"{x},{y},{w},{h}",
            "-t", "tiff",
            "-",
        )
        rc, stdout, stderr = self._runner(argv)
        if rc != 0:
            msg = stderr.decode("utf-8", errors="replace").strip() or (
                f"screencapture exited with code {rc}"
            )
            raise RuntimeError(
                f"quartz capture: screencapture failed: {msg} "
                f"(region={x,y,w,h})"
            )
        if len(stdout) < 8:
            raise RuntimeError(
                f"quartz capture: screencapture produced only "
                f"{len(stdout)} bytes; expected at least the 8-byte TIFF header"
            )
        header = _parse_tiff_header(stdout)
        # The pixel data starts at strip_offset and runs for
        # strip_byte_count bytes. Pad with zeros if the
        # stream is short, the same way the X11 adapter does.
        pixel_offset = header.strip_offset
        pixel_len = header.strip_byte_count
        pixels = stdout[pixel_offset:pixel_offset + pixel_len]
        if len(pixels) < pixel_len:
            pixels = pixels + b"\x00" * (pixel_len - len(pixels))
        return header, pixels


# ---------------------------------------------------------------------------
# Factory hook for ``whisperpaw._screen``
# ---------------------------------------------------------------------------


def build_quartz_screen(
    *,
    runner: Runner | None = None,
    available: Callable[[], bool] | None = None,
    env: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] | None = None,
) -> QuartzScreen | None:
    """Construct a :class:`QuartzScreen` if one can be built here.

    Returns ``None`` when the discovery check fails (not on
    Darwin, or no ``screencapture`` on ``$PATH``). The factory
    is the single function :func:`whisperpaw._screen._quartz_capture`
    delegates to, so the dispatch path is one line.

    The ``available`` and ``env`` / ``which`` parameters are
    forwarded to :func:`is_quartz_available` so tests can
    simulate "macOS is available" without touching
    ``sys.platform``. The ``runner`` parameter is forwarded to
    the new instance; it defaults to the real subprocess
    plumbing (macOS-only) so the factory can be called on any
    platform without crashing — the platform check in
    :func:`is_quartz_available` short-circuits first.
    """
    if available is not None:
        if not available():
            return None
    elif not is_quartz_available(env=env, which=which):
        return None
    try:
        return QuartzScreen(runner=runner)
    except (OSError, RuntimeError):
        # QuartzScreen's constructor itself doesn't do I/O
        # (deferred to capture()), but if a future refactor
        # moves discovery into __init__, we still want the
        # factory to degrade gracefully.
        return None


__all__ = [
    "QuartzScreen",
    "TiffHeader",
    "is_quartz_available",
    "build_quartz_screen",
]
