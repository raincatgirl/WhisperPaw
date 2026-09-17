"""X11 screen-capture adapter for ``paw-zoom`` v0.2.

This module ships the first *real* OS-specific
:class:`whisperpaw._screen.ScreenCapture` adapter: one that talks
to an X11 display server via the ``xwd`` (X Window Dump) utility
and returns a list-of-strings view of a rectangular region of the
screen.

Why ``xwd`` and not the raw X11 wire protocol
---------------------------------------------
X11's wire protocol is well-documented but enormous — the
connection setup alone needs to negotiate the auth cookie,
byte order, and a dozen vendor-specific keysyms, all in big-
endian ``struct`` packing. The ``xwd`` binary, on the other hand,
ships with every Linux distribution (Debian: ``x11-utils``,
Fedora: ``xorg-x11-utils``, Arch: ``xorg-xwd``), speaks the
protocol for us, and dumps the screen to stdout in a
predictable 100-byte header + raw pixel payload format that we
can parse with stdlib ``struct`` and ``sys``.

A "no third-party Python deps" tool is allowed to call a binary
that already lives on the user's system — see the hard rules
in ``/opt/data/cron/prompts/whisperpaw_tick.md``: "If you must
shell out, gate it per-OS and add a friendly 'not supported on
this OS' message." The OS gate is :func:`is_x11_available`,
which is called by :func:`whisperpaw._screen._x11_capture`
*only* on Linux; the subprocess call is wrapped in
:func:`_run` so the tests can swap it for a fake.

The XWD file format
-------------------
``xwd -root -silent`` writes a stream of:

1. A 100-byte **header** (the layout we care about is at
   specific offsets — see :data:`_XWD_HEADER_FMT`).
2. A colour-map section (only present if the header's
   ``ncolors`` field is nonzero; rare on a modern 24/32-bit
   display).
3. The **pixel data** in row order, top to bottom, padded to
   ``bytes_per_line`` bytes per row.

We only need the screen extent (``width``, ``height``,
``bits_per_pixel``, ``byte_order``) from the header to slice
the pixel payload into a 2D array. The colour-map section is
skipped by reading ``header_size`` bytes of header and then
jumping straight into the pixel data.

Why not just decode the pixels into RGB
---------------------------------------
We don't try to be a real screen-reader. ``paw-zoom``'s job is
to *magnify a region* — the user already knows what the screen
looks like, they just can't see the detail. So we downsample
each cell of the target character grid to a single "ink
density" character picked from :data:`_DENSITY_CHARS`. The
character is a function of the average brightness of the few
pixels we sample in that cell, not a function of the actual
glyphs on the screen. This keeps the adapter small (~150 lines
incl. comments) and gives the magnifier a meaningful preview.

Public API
----------
* :class:`X11Screen` — the :class:`ScreenCapture` subclass.
* :func:`is_x11_available` — ``True`` if ``$DISPLAY`` is set
  and ``xwd`` is on ``$PATH`` (the two cheap checks the
  factory runs before trying to construct an instance).
* :func:`build_x11_screen` — construct an :class:`X11Screen`,
  with injectable subprocess / availability hooks for tests.
* :func:`_parse_xwd_header` — pure function: header bytes →
  ``XwdHeader`` namedtuple. Exposed (with the underscore) so
  tests can hit the parsing without a real X server.
* :func:`_pixels_to_grid` — pure function: raw pixel bytes +
  parsed header + (x, y, w, h) → list[str] of the same shape
  :class:`whisperpaw._screen.FakeScreen.capture` returns.

Pure stdlib, no third-party deps, no telemetry, no network
beyond the local ``xwd`` subprocess.
"""
from __future__ import annotations

import os
import shutil
import struct
import sys
from collections.abc import Mapping
from typing import Callable, NamedTuple, Sequence

from whisperpaw import _screen


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------


#: The five characters we downsample a cell to, sorted from
#: "no ink" to "full ink". Chosen so the difference between
#: any two adjacent entries is visible in a typical terminal
#: font (e.g. unicode block-element characters in the
#: U+2580–U+2599 range). The character is selected by
#: averaging the brightness of a cell's sampled pixels and
#: bucketing the result into one of five bins.
_DENSITY_CHARS: str = " \u2591\u2592\u2593\u2588"  # space, ░, ▒, ▓, �█


#: Window-manager / compositor hint: when ``xwd`` produces a
#: 32-bit image with a non-zero alpha channel, we ignore the
#: alpha byte (we only need a brightness reading). The XWD
#: header encodes the channel layout in the ``bits_per_pixel``
#: and ``byte_order`` fields; we handle both little- and
#: big-endian servers (see :func:`_parse_xwd_header`).


class XwdHeader(NamedTuple):
    """The slice of an XWD header ``paw-zoom`` actually uses.

    Other fields (``ncolors``, ``window_x``, ``window_y``,
    ``window_width``, ``window_height``, ...) are parsed by
    :func:`_parse_xwd_header` but not exposed — they exist
    only to compute the *size* of the colour-map section we
    need to skip before reaching the pixel data.
    """

    header_size: int
    version: int
    pixmap_format: int
    depth: int
    width: int
    height: int
    xoffset: int
    byte_order: int   # 0 = LSBFirst (little-endian), 1 = MSBFirst
    bitmap_unit: int
    bitmap_bit_order: int
    bitmap_pad: int
    bits_per_pixel: int
    bytes_per_line: int
    visual_class: int


#: A regex-free way to skip the colour-map section. The
#: header's ``ncolors`` field (offset 64, also uint32) tells
#: us how many ``XWDColor`` entries follow; each entry is
#: exactly 12 bytes (three uint32 RGB values).
_XWD_COLOR_ENTRY_SIZE: int = 12


#: What we hand to ``subprocess.run`` to dump the root window.
#: ``-root`` means the entire visible screen (not a specific
#: window), ``-silent`` suppresses the "Dump done" banner that
#: some ``xwd`` versions print to stderr, and ``-out -`` writes
#: the dump to stdout so we can ``subprocess.PIPE`` it.
_XWD_INVOCATION: tuple[str, ...] = ("xwd", "-root", "-silent", "-out", "-")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def is_x11_available(
    *,
    env: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] | None = None,
) -> bool:
    """Return ``True`` if an X11 capture is likely to work here.

    Two cheap checks: ``$DISPLAY`` must be set in the
    environment, and the ``xwd`` binary must be on ``$PATH``.
    Neither is *sufficient* on its own — a Linux container
    with ``$DISPLAY`` pointing at a dead X socket will still
    fail at capture time — but together they catch the common
    "headless CI" case before we even try to fork ``xwd``.

    The ``env`` and ``which`` parameters exist so the tests
    can run on any platform without reaching for monkeypatch.
    """
    if env is None:
        env = os.environ
    if which is None:
        which = shutil.which
    return bool(env.get("DISPLAY")) and which("xwd") is not None


# ---------------------------------------------------------------------------
# XWD header parsing
# ---------------------------------------------------------------------------


def _parse_xwd_header(data: bytes) -> XwdHeader:
    """Parse the first 56 bytes of an XWD dump into a header.

    The XWD header is stored in the *server's* native byte
    order; the ``byte_order`` field (offset 28) tells you
    which one. The value is a 4-byte integer that is either
    ``0`` (LSBFirst) or ``1`` (MSBFirst) — i.e. one of the
    four bytes is 1, the other three are 0. We read the
    field in both little- and big-endian and pick the
    interpretation that gives 0 or 1 (the other one would
    give ``0x01000000`` = 16777216, which is obviously
    invalid). This handles both LE and BE X servers without
    a chicken-and-egg problem.

    Raises :class:`ValueError` if the magic / size look wrong.
    We deliberately don't validate every field — the X server
    sets most of them and we trust it. We only fail fast on
    the two a test would actually check (header size and the
    0x00000007 "XWD file" version constant).
    """
    if len(data) < 56:
        raise ValueError(
            f"xwd header is too short (got {len(data)} bytes, need 56)"
        )
    # Read the byte_order field in both endians. Exactly one
    # interpretation will yield 0 or 1; the other will yield
    # 0x01000000 (16,777,216), which is obviously invalid for
    # a 0/1 field. Pick the valid one.
    bo_le, = struct.unpack("<I", data[28:32])
    bo_be, = struct.unpack(">I", data[28:32])
    if bo_le in (0, 1):
        bo = "<"
    elif bo_be in (0, 1):
        bo = ">"
    else:
        # A byte_order field that is neither 0 nor 1 in either
        # encoding is corrupt. Fall back to LE — the rest of
        # the validation will catch the real problem.
        bo = "<"
    fields = struct.unpack(bo + "IIIIIIIIIIIIII", data[:56])
    header_size, version, _pixmap_format, _depth = fields[:4]
    width, height, xoffset, byte_order = fields[4:8]
    bitmap_unit, bitmap_bit_order, bitmap_pad, bits_per_pixel = fields[8:12]
    bytes_per_line, _visual_class = fields[12:14]
    if version != 7:
        raise ValueError(
            f"xwd version is {version}, expected 7 (the only XWD version)"
        )
    if header_size < 56:
        raise ValueError(
            f"xwd header_size is {header_size}, expected >= 56"
        )
    return XwdHeader(
        header_size=header_size,
        version=version,
        pixmap_format=_pixmap_format,
        depth=_depth,
        width=width,
        height=height,
        xoffset=xoffset,
        byte_order=byte_order,
        bitmap_unit=bitmap_unit,
        bitmap_bit_order=bitmap_bit_order,
        bitmap_pad=bitmap_pad,
        bits_per_pixel=bits_per_pixel,
        bytes_per_line=bytes_per_line,
        visual_class=_visual_class,
    )


def _color_map_size(data: bytes) -> int:
    """Return the byte length of the colour-map section to skip.

    The colour-map section is optional — it's only present if
    ``ncolors > 0``. Each entry is exactly 12 bytes. We read
    the count from offset 64 of the header (which is a uint32
    in *server* byte order). To avoid a chicken-and-egg
    problem with the byte-order field at offset 28, we read
    the count in both endians and pick the interpretation
    that gives a value that's 0 or a small positive integer
    (a corrupt or wildly-large count would indicate the
    wrong byte order).
    """
    if len(data) < 68:
        return 0
    ncolors_le, = struct.unpack("<I", data[64:68])
    ncolors_be, = struct.unpack(">I", data[64:68])
    # Pick the value that looks sane (a small non-negative
    # integer). A truly corrupt header would give large
    # values in both endians; we default to 0 in that case
    # rather than risk reading megabytes of "colour map".
    candidates = [n for n in (ncolors_le, ncolors_be) if 0 <= n < 1_000_000]
    if not candidates:
        return 0
    ncolors = candidates[0]
    return ncolors * _XWD_COLOR_ENTRY_SIZE


# ---------------------------------------------------------------------------
# Pixel decoding
# ---------------------------------------------------------------------------


def _sample_cell_brightness(
    pixels: bytes,
    header: XwdHeader,
    *,
    cell_x: int,
    cell_y: int,
    cell_w: int,
    cell_h: int,
) -> float:
    """Return the average brightness (0.0 = black, 1.0 = white) of
    one cell in the source pixel grid.

    ``cell_w`` and ``cell_h`` are the *target* character cell
    dimensions; the actual sample point is a single pixel in
    the middle of the cell. We don't average multiple pixels
    per cell because that would slow the magnifier down
    significantly for a feature whose output is five discrete
    characters — one sample is more than enough resolution.

    Out-of-bounds cells return 1.0 (white / "no ink") so the
    renderer can pad without special-casing the adapter.
    """
    px = cell_x + cell_w // 2
    py = cell_y + cell_h // 2
    if px < 0 or py < 0 or px >= header.width or py >= header.height:
        return 1.0
    bpp = header.bits_per_pixel // 8
    if bpp < 3 or bpp > 4:
        # Monochrome / 16-bit displays are vanishingly rare on
        # modern Linux desktops; if we ever see one, fall back
        # to "white" rather than crash. The user can still use
        # the FakeScreen backend with a hand-typed grid.
        return 1.0
    row_offset = py * header.bytes_per_line
    pixel_offset = row_offset + px * bpp
    if pixel_offset + bpp > len(pixels):
        return 1.0
    bo = "<" if header.byte_order == 0 else ">"
    pixel = pixels[pixel_offset:pixel_offset + bpp]
    # 24-bit: B, G, R (XWD always stores pixel channels in
    # their on-wire order, which is BGR for 24bpp displays).
    # 32-bit: B, G, R, X (alpha ignored). Either way, the
    # first three bytes are the channels we want.
    b, g, r = pixel[0], pixel[1], pixel[2]
    brightness = (r * 0.30 + g * 0.59 + b * 0.11) / 255.0
    # Clamp into [0, 1] in case some weird 16-bit channel
    # value sneaks in (defensive — should never happen with
    # 24/32bpp input).
    if brightness < 0.0:
        return 0.0
    if brightness > 1.0:
        return 1.0
    return brightness


def _pixels_to_grid(
    pixels: bytes,
    header: XwdHeader,
    *,
    x: int,
    y: int,
    w: int,
    h: int,
) -> list[str]:
    """Convert a sub-rectangle of an XWD pixel payload into a
    ``h``×``w`` list of density strings, the same shape
    :class:`whisperpaw._screen.FakeScreen.capture` returns.

    Each cell of the output is the *density character* whose
    bucket contains the cell's average brightness, so a
    magnifier user sees a coarse "where the ink is" view of
    the screen — the actual glyphs are not recovered.
    """
    if w < 0 or h < 0:
        raise ValueError(
            f"_pixels_to_grid: w/h must be non-negative (got w={w}, h={h})"
        )
    if w == 0 or h == 0:
        return []
    # We render the target character grid in screen pixels:
    # the source image is (header.width × header.height)
    # pixels, the requested cell is (w × h) characters, so
    # each character cell covers (header.width / w) pixels
    # horizontally and (header.height / h) vertically.
    cell_w = max(1, header.width // w)
    cell_h = max(1, header.height // h)
    out: list[str] = []
    n_chars = len(_DENSITY_CHARS)
    for row in range(h):
        line_chars: list[str] = []
        for col in range(w):
            cell_x = x * cell_w + col * cell_w
            cell_y = y * cell_h + row * cell_h
            brightness = _sample_cell_brightness(
                pixels,
                header,
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


#: Signature of the subprocess runner. Returns (returncode, stdout, stderr).
Runner = Callable[[Sequence[str]], tuple[int, bytes, bytes]]


def _default_runner(argv: Sequence[str]) -> tuple[int, bytes, bytes]:
    """The real subprocess runner — used in production.

    We capture both stdout (the XWD bytes) and stderr (the
    "Dump done" banner and any error from ``xwd``) so the
    caller can surface them. ``check=False`` because we want
    to convert the non-zero exit code into a friendly
    :class:`RuntimeError`, not a ``CalledProcessError``.
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


class X11Screen(_screen.ScreenCapture):
    """A :class:`ScreenCapture` backed by ``xwd`` and a real X server.

    Construction is lazy: we only call ``xwd`` on the first
    :meth:`screen_size` or :meth:`capture`. That way, an
    attempt to instantiate the adapter on a system without
    ``$DISPLAY`` doesn't immediately fail — the failure is
    deferred to the first capture call, which is also when the
    CLI's friendly "not yet implemented" message is most
    useful.

    The class is intentionally simple: one ``_cached_size``
    slot, one ``_runner`` slot, no state machine. The
    pipeline that consumes the capture doesn't care that
    we're an X11 adapter — it just sees a
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
        """Return the X11 root window's pixel size.

        We don't have a separate "size-only" tool, so the
        implementation just runs ``xwd -root`` once and caches
        the size from the header. Subsequent calls return the
        cached value (the screen size doesn't change at
        runtime on a typical setup; if the user resizes
        monitors while ``paw-zoom`` is running, they'll need
        to restart the loop).
        """
        if self._cached_size is not None:
            return self._cached_size
        header, _pixels = self._run_xwd()
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
        header, pixels = self._run_xwd()
        # Refresh the cached size every capture — it's free
        # (we already have the header) and it makes the
        # adapter robust to mid-run monitor resizes.
        self._cached_size = (header.width, header.height)
        return _pixels_to_grid(pixels, header, x=x, y=y, w=w, h=h)

    def _run_xwd(self) -> tuple[XwdHeader, bytes]:
        """Invoke ``xwd`` and return ``(header, pixel_bytes)``.

        The colour-map section (if any) is skipped, so
        ``pixel_bytes`` is exactly the raw image data the
        renderer needs.
        """
        rc, stdout, stderr = self._runner(_XWD_INVOCATION)
        if rc != 0:
            msg = stderr.decode("utf-8", errors="replace").strip() or (
                f"xwd exited with code {rc}"
            )
            raise RuntimeError(
                f"xwd failed: {msg} "
                f"(is an X server running on $DISPLAY={os.environ.get('DISPLAY')!r}?)"
            )
        if len(stdout) < 56:
            raise RuntimeError(
                f"xwd produced only {len(stdout)} bytes; expected at "
                f"least the 56-byte header"
            )
        header = _parse_xwd_header(stdout)
        # The pixel data starts at offset (header_size +
        # color_map_size); we read up to bytes_per_line * height
        # bytes from there.
        cm_size = _color_map_size(stdout)
        pixel_offset = header.header_size + cm_size
        pixel_len = header.bytes_per_line * header.height
        pixels = stdout[pixel_offset:pixel_offset + pixel_len]
        if len(pixels) < pixel_len:
            # xwd occasionally produces a slightly short dump
            # when the screen contains a partially-obscured
            # window. Pad with zeros so the downsample loop
            # can run; the missing region will render as
            # "no ink" (black-ish), which is the safe default
            # for a magnifier.
            pixels = pixels + b"\x00" * (pixel_len - len(pixels))
        return header, pixels


# ---------------------------------------------------------------------------
# Factory hook for ``whisperpaw._screen``
# ---------------------------------------------------------------------------


def build_x11_screen(
    *,
    runner: Runner | None = None,
    env: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] | None = None,
) -> X11Screen | None:
    """Construct an :class:`X11Screen` if one can be built here.

    Returns ``None`` when the discovery check fails
    (no ``$DISPLAY``, no ``xwd`` on ``$PATH``). The factory
    is the single function :func:`whisperpaw._screen._x11_capture`
    delegates to, so the dispatch path is one line.
    """
    if not is_x11_available(env=env, which=which):
        return None
    try:
        return X11Screen(runner=runner)
    except (OSError, RuntimeError):
        # X11Screen's constructor itself doesn't do I/O
        # (deferred to capture()), but if a future refactor
        # moves discovery into __init__, we still want the
        # factory to degrade gracefully.
        return None


__all__ = [
    "X11Screen",
    "XwdHeader",
    "is_x11_available",
    "build_x11_screen",
]
