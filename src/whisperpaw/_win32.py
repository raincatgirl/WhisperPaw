"""Win32 GDI screen-capture adapter for ``paw-zoom`` v0.2.

This module ships the second *real* OS-specific
:class:`whisperpaw._screen.ScreenCapture` adapter: one that talks
to the Windows GDI subsystem via the ``gdi32`` / ``user32`` DLLs
and returns a list-of-strings view of a rectangular region of
the screen.

Why GDI and not DXGI / WGC / PrintWindow
-----------------------------------------
Modern Windows has several screen-capture APIs. The newer ones
(Windows.Graphics.Capture, DXGI Output Duplication) are faster
and avoid the GDI mouse-cursor / window-decoration quirks, but
they are also significantly more code: they need a ``RoGetActivationFactory``
shim, a ``IInitializeWithWindow::Initialize`` call, and either a
COM apartment or a packaged-app identity. ``ctypes`` is enough to
speak *GDI* with no shims, and GDI works on every Windows version
since 95 — which is the accessibility-first guarantee the
project makes.

The shape of a GDI screen capture
---------------------------------
The "GDI way" to dump the desktop is:

1. ``GetDC(NULL)`` — get a handle to the screen device context.
2. ``CreateCompatibleDC(hdc_screen)`` — create a memory DC that
   matches the screen's colour depth.
3. ``CreateCompatibleBitmap(hdc_screen, w, h)`` — create a
   bitmap in the memory DC of the size we want.
4. ``SelectObject(hdc_mem, hbitmap)`` — bind the bitmap to the
   memory DC.
5. ``BitBlt(hdc_mem, 0, 0, w, h, hdc_screen, x, y, SRCCOPY)`` —
   copy the screen region into our bitmap.
6. ``GetDIBits(hdc_mem, hbitmap, ...)`` — pull the raw pixel
   bytes out of the bitmap (bottom-up, BGRX order, 32bpp).
7. Clean up: ``DeleteObject``, ``DeleteDC``, ``ReleaseDC``.

We don't need the full pipeline in production code — we just
need to *call* it. So this module hides the ctypes plumbing in
a single :class:`Win32Screen` class and exposes the pixel-decoding
math in a pure function so the tests don't have to reach into
the GDI subsystem at all.

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
* :class:`Win32Screen` — the :class:`ScreenCapture` subclass.
* :func:`is_win32_available` — ``True`` if the current process
  is on Windows. The actual GDI plumbing is lazy: a Win32
  process on a headless server with no display will still
  report available (the ``GetDC`` call is what fails in that
  case, and we surface that as a friendly ``RuntimeError``).
* :func:`build_win32_screen` — construct a :class:`Win32Screen`,
  with injectable capture / availability hooks for tests.
* :func:`_bgrx_to_grid` — pure function: raw BGRX pixel bytes +
  screen size + (x, y, w, h) → list[str] of the same shape
  :class:`whisperpaw._screen.FakeScreen.capture` returns.
* :func:`_sample_bgrx_cell_brightness` — the per-cell sampling
  helper.

Pure stdlib, no third-party deps, no telemetry, no network.
"""
from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Callable, NamedTuple, Sequence

from whisperpaw import _screen


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------


#: The five characters we downsample a cell to, sorted from
#: "no ink" to "full ink". Same shape as ``whisperpaw._x11`` so
#: the rendered output is consistent across OS adapters.
_DENSITY_CHARS: str = " \u2591\u2592\u2593\u2588"  # space, ░, ▒, ▓, █


#: GDI's ``SRCCOPY`` raster-op code for ``BitBlt``. We hard-code
#: it as a module constant so the runner doesn't have to import
#: ``ctypes`` just to look it up.
_SRCCOPY: int = 0x00CC0020


#: The expected pixel format of a ``GetDIBits`` call with a
#: 32-bpp DIB. ``biCompression = BI_RGB`` (no compression) is
#: the value 0; ``biBitCount = 32`` is the value 32. We
#: hard-code them so a test can verify the right fields are set
#: without ``ctypes`` plumbing.
_BI_RGB: int = 0
_BIT_COUNT_32: int = 32


class Win32Capture(NamedTuple):
    """The result of a single GDI capture call.

    A 2-tuple of ``(width, height)`` and ``pixels`` — the BGRX
    raw pixel bytes, ``width * height * 4`` long, bottom-up
    (GDI convention — the bottom row comes first in memory).
    Exposed as a NamedTuple so the test runner can return it
    directly without a wrapper class.
    """

    width: int
    height: int
    pixels: bytes


#: Signature of the GDI capture runner. Takes a region
#: ``(x, y, w, h)`` and returns a :class:`Win32Capture`. The
#: default implementation does the full GetDC / BitBlt /
#: GetDIBits dance; tests substitute a fake that returns a
#: synthetic BGRX buffer.
CaptureFn = Callable[[int, int, int, int], Win32Capture]


#: Signature of the screen-size probe. Returns ``(width, height)``
#: of the primary display, in pixels. The default calls
#: ``GetSystemMetrics(SM_CXSCREEN)`` / ``GetSystemMetrics(SM_CYSCREEN)``;
#: tests substitute a fake.
SizeFn = Callable[[], tuple[int, int]]


#: Signature of the availability check. The default is a
#: ``sys.platform == "win32"`` test; tests substitute whatever
#: they like. Kept injectable so a test can verify the
#: ``is_win32_available`` path on a non-Windows box.
AvailableFn = Callable[[], bool]


def _default_available() -> bool:
    """The default availability check: ``sys.platform == "win32"``.

    Lives at module scope so it can be patched by tests with
    ``monkeypatch.setattr`` (a module-level function is easier
    to patch than a ``lambda``-as-default-arg).
    """
    return sys.platform == "win32"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def is_win32_available(
    *,
    available: AvailableFn | None = None,
) -> bool:
    """Return ``True`` if a Win32 capture is likely to work here.

    The default check is just ``sys.platform == "win32"``: there
    is no other cheap pre-flight test (we can't ask the kernel
    for the desktop resolution without going through GDI, and
    we want to avoid the cost of loading ``gdi32`` until we
    actually capture). The ``available`` parameter exists so
    tests can simulate a different platform without monkey-patching
    ``sys.platform`` (which is a process-wide global and a bad
    idea to mutate).
    """
    if available is None:
        available = _default_available
    return bool(available())


# ---------------------------------------------------------------------------
# Pixel decoding (pure functions — testable on any platform)
# ---------------------------------------------------------------------------


def _sample_bgrx_cell_brightness(
    pixels: bytes,
    *,
    width: int,
    height: int,
    cell_x: int,
    cell_y: int,
    cell_w: int,
    cell_h: int,
    bottom_up: bool = True,
) -> float:
    """Return the average brightness (0.0 = black, 1.0 = white) of
    one cell in a BGRX pixel grid.

    ``cell_w`` and ``cell_h`` are the *target* character cell
    dimensions; the actual sample point is a single pixel in
    the middle of the cell. We don't average multiple pixels
    per cell because that would slow the magnifier down
    significantly for a feature whose output is five discrete
    characters — one sample is more than enough resolution.

    GDI's ``GetDIBits`` returns pixels in *bottom-up* row order
    (the bottom row first in memory) by default. The
    ``bottom_up`` flag inverts the row index when set, which is
    the production behaviour. Tests can pass ``False`` to feed
    a top-down buffer directly.

    Out-of-bounds cells return 1.0 (white / "no ink") so the
    renderer can pad without special-casing the adapter.
    """
    if cell_x < 0 or cell_y < 0 or cell_w < 1 or cell_h < 1:
        return 1.0
    px = cell_x + cell_w // 2
    py = cell_y + cell_h // 2
    if px < 0 or py < 0 or px >= width or py >= height:
        return 1.0
    # 4 bytes per pixel: B, G, R, X (alpha ignored).
    if bottom_up:
        row_in_buffer = (height - 1) - py
    else:
        row_in_buffer = py
    offset = row_in_buffer * width * 4 + px * 4
    if offset + 3 >= len(pixels):
        return 1.0
    b = pixels[offset]
    g = pixels[offset + 1]
    r = pixels[offset + 2]
    # Rec.601 luma weights. Same formula the X11 adapter uses,
    # so a magnifier user sees the same density for the same
    # pixel regardless of which OS adapter is active.
    brightness = (r * 0.30 + g * 0.59 + b * 0.11) / 255.0
    if brightness < 0.0:
        return 0.0
    if brightness > 1.0:
        return 1.0
    return brightness


def _bgrx_to_grid(
    pixels: bytes,
    *,
    width: int,
    height: int,
    x: int,
    y: int,
    w: int,
    h: int,
    bottom_up: bool = True,
) -> list[str]:
    """Convert a sub-rectangle of a BGRX pixel buffer into a
    ``h``×``w`` list of density strings, the same shape
    :class:`whisperpaw._screen.FakeScreen.capture` returns.

    Each cell of the output is the *density character* whose
    bucket contains the cell's average brightness, so a
    magnifier user sees a coarse "where the ink is" view of
    the screen — the actual glyphs are not recovered.
    """
    if w < 0 or h < 0:
        raise ValueError(
            f"_bgrx_to_grid: w/h must be non-negative (got w={w}, h={h})"
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
            brightness = _sample_bgrx_cell_brightness(
                pixels,
                width=width,
                height=height,
                cell_x=cell_x,
                cell_y=cell_y,
                cell_w=cell_w,
                cell_h=cell_h,
                bottom_up=bottom_up,
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
# The GDI capture runner (real ctypes plumbing — Windows only)
# ---------------------------------------------------------------------------


def _default_capture(x: int, y: int, w: int, h: int) -> Win32Capture:
    """The real GDI capture.

    Does the full GetDC / CreateCompatibleDC / CreateCompatibleBitmap
    / BitBlt / GetDIBits dance via ``ctypes``, then returns the
    raw 32-bpp BGRX pixel buffer. Lazy-imports ``ctypes`` and
    the two Windows DLLs so the module itself stays importable
    on every platform (the GDI plumbing is only touched when
    capture is actually called).

    The output buffer is bottom-up (GDI convention), and the
    width / height reported are the *requested* w / h, not the
    primary display's extent — so the caller can ask for a
    sub-rectangle and the pixel data matches.

    Raises :class:`RuntimeError` if any GDI call returns an
    error. The caller surfaces that to the user.
    """
    import ctypes
    from ctypes import wintypes

    # gdi32 is always present on Windows; user32 too. We let
    # ctypes raise its own ``OSError`` if either is missing
    # (which would mean the system is so broken nothing else
    # would work either).
    gdi32 = ctypes.WinDLL("gdi32")  # type: ignore[attr-defined]
    user32 = ctypes.WinDLL("user32")  # type: ignore[attr-defined]

    # --- 1. Get the screen DC ---------------------------------------
    # ``GetDC(NULL)`` returns a DC for the entire primary
    # display. We release it at the end with ``ReleaseDC``.
    GetDC = user32.GetDC
    GetDC.argtypes = [wintypes.HWND]
    GetDC.restype = wintypes.HDC
    hdc_screen = GetDC(None)
    if not hdc_screen:
        raise RuntimeError(
            "win32 capture: GetDC(NULL) returned NULL — no display?"
        )

    try:
        # --- 2. Create a memory DC --------------------------------
        CreateCompatibleDC = gdi32.CreateCompatibleDC
        CreateCompatibleDC.argtypes = [wintypes.HDC]
        CreateCompatibleDC.restype = wintypes.HDC
        hdc_mem = CreateCompatibleDC(hdc_screen)
        if not hdc_mem:
            raise RuntimeError(
                "win32 capture: CreateCompatibleDC failed"
            )

        try:
            # --- 3. Create a compatible bitmap -----------------
            CreateCompatibleBitmap = gdi32.CreateCompatibleBitmap
            CreateCompatibleBitmap.argtypes = [
                wintypes.HDC, ctypes.c_int, ctypes.c_int,
            ]
            CreateCompatibleBitmap.restype = wintypes.HBITMAP
            hbitmap = CreateCompatibleBitmap(hdc_screen, w, h)
            if not hbitmap:
                raise RuntimeError(
                    "win32 capture: CreateCompatibleBitmap failed"
                )

            try:
                # --- 4. Select the bitmap into the memory DC ----
                SelectObject = gdi32.SelectObject
                SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
                SelectObject.restype = wintypes.HGDIOBJ
                SelectObject(hdc_mem, hbitmap)

                # --- 5. BitBlt the screen region into the bitmap
                BitBlt = gdi32.BitBlt
                BitBlt.argtypes = [
                    wintypes.HDC, ctypes.c_int, ctypes.c_int,
                    ctypes.c_int, ctypes.c_int,
                    wintypes.HDC, ctypes.c_int, ctypes.c_int,
                    wintypes.DWORD,
                ]
                BitBlt.restype = wintypes.BOOL
                ok = BitBlt(
                    hdc_mem, 0, 0, w, h,
                    hdc_screen, x, y,
                    _SRCCOPY,
                )
                if not ok:
                    raise RuntimeError(
                        f"win32 capture: BitBlt failed (region={x,y,w,h})"
                    )

                # --- 6. GetDIBits --------------------------------
                # Build a BITMAPINFOHEADER that asks for 32-bpp
                # BGRX output, top-down (we set biHeight negative
                # to override GDI's default bottom-up). The
                # pixel buffer comes back bottom-up if biHeight
                # is positive, so we accept the GDI default and
                # invert in the sampling helper.
                class BITMAPINFOHEADER(ctypes.Structure):
                    _fields_ = [
                        ("biSize", wintypes.DWORD),
                        ("biWidth", wintypes.LONG),
                        ("biHeight", wintypes.LONG),
                        ("biPlanes", wintypes.WORD),
                        ("biBitCount", wintypes.WORD),
                        ("biCompression", wintypes.DWORD),
                        ("biSizeImage", wintypes.DWORD),
                        ("biXPelsPerMeter", wintypes.LONG),
                        ("biYPelsPerMeter", wintypes.LONG),
                        ("biClrUsed", wintypes.DWORD),
                        ("biClrImportant", wintypes.DWORD),
                    ]

                class BITMAPINFO(ctypes.Structure):
                    _fields_ = [
                        ("bmiHeader", BITMAPINFOHEADER),
                        ("bmiColors", wintypes.DWORD * 3),
                    ]

                bmi = BITMAPINFO()
                bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
                bmi.bmiHeader.biWidth = w
                bmi.bmiHeader.biHeight = h
                bmi.bmiHeader.biPlanes = 1
                bmi.bmiHeader.biBitCount = _BIT_COUNT_32
                bmi.bmiHeader.biCompression = _BI_RGB

                buf = (ctypes.c_ubyte * (w * h * 4))()
                GetDIBits = gdi32.GetDIBits
                GetDIBits.argtypes = [
                    wintypes.HDC, wintypes.HBITMAP, wintypes.UINT,
                    wintypes.UINT, ctypes.c_void_p,
                    ctypes.c_void_p, wintypes.UINT,
                ]
                GetDIBits.restype = ctypes.c_int
                # DIB_RGB_COLORS = 0; we use the literal so the
                # ctypes import block stays small.
                got = GetDIBits(
                    hdc_mem, hbitmap, 0, h,
                    ctypes.byref(buf), ctypes.byref(bmi), 0,
                )
                if got == 0:
                    raise RuntimeError(
                        "win32 capture: GetDIBits returned 0 rows"
                    )
                return Win32Capture(
                    width=w, height=h,
                    pixels=bytes(buf),
                )
            finally:
                # --- 7a. Clean up the bitmap --------------------
                DeleteObject = gdi32.DeleteObject
                DeleteObject.argtypes = [wintypes.HGDIOBJ]
                DeleteObject.restype = wintypes.BOOL
                DeleteObject(hbitmap)
        finally:
            # --- 7b. Clean up the memory DC ---------------------
            DeleteDC = gdi32.DeleteDC
            DeleteDC.argtypes = [wintypes.HDC]
            DeleteDC.restype = wintypes.BOOL
            DeleteDC(hdc_mem)
    finally:
        # --- 7c. Release the screen DC -----------------------
        ReleaseDC = user32.ReleaseDC
        ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
        ReleaseDC.restype = ctypes.c_int
        ReleaseDC(None, hdc_screen)


def _default_screen_size() -> tuple[int, int]:
    """The real ``GetSystemMetrics`` probe — Windows only.

    Returns the primary display's pixel extent as ``(width, height)``.
    ``SM_CXSCREEN = 0`` and ``SM_CYSCREEN = 1`` are the
    canonical index constants; we hard-code the ints so the
    call doesn't have to import ``ctypes`` just to look them
    up.
    """
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32")  # type: ignore[attr-defined]
    GetSystemMetrics = user32.GetSystemMetrics
    GetSystemMetrics.argtypes = [ctypes.c_int]
    GetSystemMetrics.restype = ctypes.c_int
    w = GetSystemMetrics(0)  # SM_CXSCREEN
    h = GetSystemMetrics(1)  # SM_CYSCREEN
    if w <= 0 or h <= 0:
        raise RuntimeError(
            f"win32 capture: GetSystemMetrics returned ({w}, {h}) — "
            "no primary display?"
        )
    return (w, h)


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


class Win32Screen(_screen.ScreenCapture):
    """A :class:`ScreenCapture` backed by Win32 GDI.

    Construction is lazy: we only call GDI on the first
    :meth:`screen_size` or :meth:`capture`. The :attr:`_capture`
    and :attr:`_size` callables are injectable so tests don't
    need a real Windows display; in production they're the
    ``_default_capture`` / ``_default_screen_size`` defined
    above, which lazy-import ``ctypes`` and the GDI / user32
    DLLs on first use.

    The class is intentionally simple: one cached size, two
    injected callables, no state machine. The pipeline that
    consumes the capture doesn't care that we're a Win32
    adapter — it just sees a :class:`ScreenCapture` whose
    ``screen_size()`` returns ``(w, h)`` and whose
    ``capture()`` returns a list of strings.
    """

    __slots__ = ("_capture", "_size", "_cached_size")

    def __init__(
        self,
        *,
        capture: CaptureFn | None = None,
        size: SizeFn | None = None,
    ) -> None:
        if capture is None:
            capture = _default_capture
        if size is None:
            size = _default_screen_size
        self._capture: CaptureFn = capture
        self._size: SizeFn = size
        self._cached_size: tuple[int, int] | None = None

    def screen_size(self) -> tuple[int, int]:
        """Return the primary display's pixel size.

        The size is cached after the first call, mirroring the
        X11 adapter's behaviour (a hot magnifier loop calls
        ``screen_size()`` on every poll). Callers that resize
        the monitor mid-session can construct a fresh
        :class:`Win32Screen` to refresh the cache.
        """
        if self._cached_size is not None:
            return self._cached_size
        self._cached_size = self._size()
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
        # GDI gives us exactly the rectangle we asked for, not
        # the whole primary display. That saves us a slice
        # step. We do, however, ask the screen size first so
        # the user can pass ``--region`` against a known
        # extent without us having to do extra math.
        self.screen_size()  # refresh the cache
        result = self._capture(x, y, w, h)
        # GetDIBits returns bottom-up by default; the helper
        # inverts the row order in that case. ``_default_capture``
        # asks for bottom-up, so we leave the default here.
        return _bgrx_to_grid(
            result.pixels,
            width=result.width,
            height=result.height,
            x=0, y=0, w=w, h=h,
            bottom_up=True,
        )


# ---------------------------------------------------------------------------
# Factory hook for ``whisperpaw._screen``
# ---------------------------------------------------------------------------


def build_win32_screen(
    *,
    capture: CaptureFn | None = None,
    size: SizeFn | None = None,
    available: AvailableFn | None = None,
) -> Win32Screen | None:
    """Construct a :class:`Win32Screen` if one can be built here.

    Returns ``None`` when the availability check fails (i.e.
    we're not on Windows, or the test injects a fake
    availability). The factory is the single function
    :func:`whisperpaw._screen._win32_capture` delegates to, so
    the dispatch path is one line.

    The ``capture`` and ``size`` callables are forwarded to
    the new instance. They default to the real GDI plumbing
    (Windows-only) so the factory can be called on any
    platform without crashing — the platform check in
    :func:`is_win32_available` short-circuits first.
    """
    if not is_win32_available(available=available):
        return None
    try:
        return Win32Screen(capture=capture, size=size)
    except (OSError, RuntimeError):
        # Win32Screen's constructor itself doesn't do I/O
        # (deferred to capture()), but if a future refactor
        # moves discovery into __init__, we still want the
        # factory to degrade gracefully.
        return None


__all__ = [
    "Win32Screen",
    "Win32Capture",
    "is_win32_available",
    "build_win32_screen",
]
