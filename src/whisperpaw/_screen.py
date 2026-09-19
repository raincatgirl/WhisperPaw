"""Screen-capture adapters for ``paw-zoom`` v0.2.

What this is
------------
v0.1 of ``paw-zoom`` is the ASCII proof-of-concept — it magnifies a
rectangular sub-grid of *text* and writes the result to stdout. The
real magnifier will instead capture a rectangular region of the
*visual* screen (pixels) and turn that into text before feeding the
existing renderer.

This module is the seam between the two: a tiny ``ScreenCapture``
protocol that any per-OS adapter (X11, Wayland, Win32 GDI, macOS
Quartz, etc.) can implement, plus one ready-to-use reference
implementation (:class:`FakeScreen`) that the tests run against.

Why an ABC instead of a free function
-------------------------------------
The real adapters need to open a per-process connection to the
display server (an X11 ``Display`` handle, a Win32 ``HWND`` for the
desktop window, a ``CGDirectDisplayID`` on macOS, etc.), and they
need to do that lazily so a system without a display (a headless
container, a CI box, a build worker) doesn't crash on import. A
factory function (``get_capture()``) gives us that lazy behaviour
for free and lets the live loop's ``stop_predicate`` stay in
charge of when to call the adapter again.

The contract every adapter honours
-----------------------------------
An adapter is anything with two methods:

* ``screen_size() -> tuple[int, int]`` — the current screen
  extent in *display coordinates* (the same coordinate system the
  OS uses for the top-left of each monitor: typically ``(width,
  height)`` of the primary display).
* ``capture(*, x: int, y: int, w: int, h: int) -> list[str]`` —
  a rectangular region of the screen, returned as a list of
  ``h`` strings each ``w`` code points long, in row order (top
  row first). Cells outside the screen or that the adapter can't
  read return ``" "`` (a single space) so the renderer doesn't
  have to special-case them.

That's it. The renderer already knows how to take a ``list[str]``
of the right shape, pad short lines, and magnify it. Keeping the
adapter contract this small is what lets the next two ticks
("add an X11 adapter" and "add a Win32 adapter") be small,
self-contained, and independently testable.

Reference adapter
-----------------
:class:`FakeScreen` is the only adapter that ships today. It
keeps an in-memory grid of strings and serves ``capture()`` by
slicing that grid — so the entire ``paw-zoom`` v0.2 code path
(data shape, change detection, magnification, follow / live /
max-frames) can be tested end-to-end on any machine, including
this one. The real OS adapters slot in by implementing the same
two methods.

Public API
----------
* :class:`ScreenCapture` — the ABC every adapter implements.
* :class:`FakeScreen` — the in-memory reference adapter.
* :func:`get_capture` — factory that returns a ``ScreenCapture`` or
  ``None`` if no backend is available on the current platform.
  Right now it only ever returns a :class:`FakeScreen` *or* ``None``
  — the OS adapters ship as stubs that return ``None`` with a
  friendly stderr message, so the user can see why the feature is
  unavailable.
* :func:`capture_screen_to_source` — turn a captured ``list[str]``
  into the ``str`` shape :func:`whisperpaw.zoom.render_viewport`
  already accepts.
* :func:`list_backends` / :func:`to_json` — discovery helpers that
  the CLI's ``--list-backends`` / ``--json`` flags surface.
* :data:`KNOWN_BACKENDS` — canonical list of backend names
  (``["fake", "x11", "win32", "quartz"]``) used by both
  ``--list-backends`` and shell completion.

Pure stdlib, no third-party deps, no telemetry, no network.
"""
from __future__ import annotations

import abc
import json
import os
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical list of adapter names, in the order the CLI / shell
#: completion presents them. The "fake" backend always exists; the
#: OS-specific ones are placeholders that ship as stubs (they
#: return ``None`` from :func:`get_capture` and a friendly "not yet
#: implemented on this OS" message when the user picks them).
KNOWN_BACKENDS: tuple[str, ...] = ("fake", "x11", "win32", "quartz")

#: Region coordinate string used to mean "the whole screen". The
#: CLI recognises this on the ``--region`` flag.
FULL_SCREEN_REGION: str = "full"


# ---------------------------------------------------------------------------
# Adapter contract
# ---------------------------------------------------------------------------


class ScreenCapture(abc.ABC):
    """Abstract base class every screen-capture adapter implements.

    Two methods are enough because the renderer downstream of the
    capture already knows how to slice / pad / magnify a list of
    strings. The adapter is just a window onto the screen expressed
    as a 2D character grid.

    Implementations are free to do whatever opening / closing
    dance they need (e.g. an X11 adapter would open a single
    ``Display`` connection in ``__init__`` and close it in
    ``close()``); the protocol doesn't care.
    """

    @abc.abstractmethod
    def screen_size(self) -> tuple[int, int]:
        """Return the current screen extent in display coordinates.

        For a single-monitor setup this is ``(width, height)`` of
        the primary display. Multi-monitor adapters should return
        the bounding box of all attached displays.
        """

    @abc.abstractmethod
    def capture(
        self, *, x: int, y: int, w: int, h: int
    ) -> list[str]:
        """Return a ``h``×``w`` sub-grid of the screen as a list of strings.

        Coordinates are in the same system as :meth:`screen_size`,
        with the origin at the top-left of the display. Cells
        outside the screen, or that the adapter can't read for
        any reason, return ``" "`` (a single space). Empty rows
        are still returned as ``h`` strings (the renderer pads
        short lines with ``" "``), so the caller's slice math
        doesn't have to special-case anything.
        """


# ---------------------------------------------------------------------------
# Reference adapter (the only one that actually captures anything today)
# ---------------------------------------------------------------------------


class FakeScreen(ScreenCapture):
    """In-memory screen stand-in used by the tests and by ``--backend fake``.

    The grid is a list of strings, one per row, exactly as
    :meth:`capture` will return them. ``screen_size()`` is derived
    from the grid: ``(max_line_length, len(grid))``. ``capture()``
    slices the grid; rows that don't exist or columns that go past
    the end of a row come back as a single space.

    Using a string-list grid (instead of a 2D pixel array) means
    the captured "screen" is already in the shape the renderer
    wants — no transcoding needed. The OS adapters will need to
    do that transcoding themselves (X11 → Pillow / numpy → string
    grid, Win32 GDI → bit-blit → string grid, etc.); the FakeScreen
    just shows what the final output should look like.

    Parameters
    ----------
    grid:
        A list of strings, one per row. Lines longer than the
        shortest row are accepted but only the first
        ``min_len = min(len(line) for line in grid)`` characters
        of each row are exposed (the rest is dropped — there's
        no real "screen" that has rows of different widths).
    """

    __slots__ = ("_grid", "_width", "_height")

    def __init__(self, grid: list[str]) -> None:
        if not grid:
            raise ValueError("FakeScreen grid must be non-empty")
        self._grid = list(grid)
        self._height = len(self._grid)
        # All rows are the same width. If the caller passes rows
        # of different lengths, the shortest wins (and the rest
        # is dropped on capture). This matches the real screen
        # behaviour: a monospace text grid has no notion of
        # "row N is 3 chars wider than row M".
        self._width = min(len(line) for line in self._grid)
        # Trim every row to the canonical width so capture() can
        # slice without re-checking each line.
        self._grid = [line[: self._width] for line in self._grid]

    def screen_size(self) -> tuple[int, int]:
        return (self._width, self._height)

    def capture(
        self, *, x: int, y: int, w: int, h: int
    ) -> list[str]:
        if w < 0 or h < 0:
            raise ValueError(
                f"capture: w/h must be non-negative (got w={w}, h={h})"
            )
        out: list[str] = []
        for r in range(h):
            src_row = y + r
            if src_row < 0 or src_row >= self._height:
                # Above or below the screen — pad with spaces.
                out.append(" " * w)
                continue
            line = self._grid[src_row]
            # Slice [x, x+w), padding on either side if x is out
            # of range or the row is shorter than the window.
            if x >= self._width:
                out.append(" " * w)
                continue
            start = max(0, x)
            end = min(self._width, x + w)
            slice_ = line[start:end]
            # Pad on the left if x was negative.
            if x < 0:
                slice_ = " " * (-x) + slice_
            # Pad on the right if the window extends past the row.
            if len(slice_) < w:
                slice_ = slice_ + " " * (w - len(slice_))
            out.append(slice_)
        return out


# ---------------------------------------------------------------------------
# Region parsing
# ---------------------------------------------------------------------------


def _parse_region(
    region: str | None,
    *,
    screen_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Parse the ``--region`` string into ``(x, y, w, h)``.

    The accepted shapes are:

    * ``None`` or ``"full"`` — the whole screen.
    * ``"X,Y,W,H"`` — explicit rectangle, four non-negative
      integers separated by commas.

    Returns the parsed rectangle clamped to the screen's
    ``screen_size`` so out-of-range regions still render a
    padded window instead of raising. Negative inputs are a
    usage error (``ValueError``) because they almost always
    signal a typo.
    """
    sw, sh = screen_size
    if region is None or region.strip().lower() == FULL_SCREEN_REGION:
        return (0, 0, sw, sh)
    parts = region.split(",")
    if len(parts) != 4:
        raise ValueError(
            f"--region must be 'X,Y,W,H' or 'full' (got {region!r})"
        )
    try:
        x, y, w, h = (int(p.strip()) for p in parts)
    except ValueError as exc:
        raise ValueError(
            f"--region components must be integers (got {region!r})"
        ) from exc
    if x < 0 or y < 0 or w < 0 or h < 0:
        raise ValueError(
            f"--region components must be non-negative (got {region!r})"
        )
    if w == 0 or h == 0:
        # Zero-area rectangle is a no-op render: the renderer
        # produces an empty string. A user passing
        # ``--region 0,0,0,0`` probably meant "full screen" and
        # made a typo, but a 0-area crop is also a legitimate
        # script-level guard, so we don't second-guess. We keep
        # the *original* w/h so a caller can still tell which
        # axis was the zero one (e.g. a 0-width row scan vs a
        # 0-height column scan).
        return (x, y, w, h)
    # Clamp to screen extent. x/y get pushed to the screen
    # boundary; w/h get capped so the rectangle never extends
    # past the right / bottom edge.
    if x >= sw or y >= sh:
        # Entirely off-screen — return a zero-area rectangle.
        return (0, 0, 0, 0)
    if x + w > sw:
        w = sw - x
    if y + h > sh:
        h = sh - y
    return (x, y, w, h)


def capture_screen_to_source(
    cap: ScreenCapture,
    *,
    region: str | None = None,
) -> str:
    """Capture ``cap``'s screen into the ``str`` shape the renderer wants.

    The string is the captured sub-grid joined with ``"\\n"``, in
    row order (top row first). Pass it directly to
    :func:`whisperpaw.zoom.render_viewport` to get the magnified
    viewport.

    The function takes care of:

    * calling :meth:`ScreenCapture.screen_size` to know how big
      the screen is,
    * parsing the ``--region`` flag (defaulting to "full screen"),
    * clamping the region to the screen's extent,
    * calling :meth:`ScreenCapture.capture` once and joining the
      result with newlines.

    No retries, no diffing, no caching. The live loop in
    :mod:`whisperpaw.zoom` is what decides when to call this
    function again.
    """
    sw, sh = cap.screen_size()
    x, y, w, h = _parse_region(region, screen_size=(sw, sh))
    if w == 0 or h == 0:
        return ""
    grid = cap.capture(x=x, y=y, w=w, h=h)
    return "\n".join(grid)


# ---------------------------------------------------------------------------
# OS-specific adapter stubs (return None from get_capture today)
# ---------------------------------------------------------------------------


def _x11_capture() -> ScreenCapture | None:
    """Return a real X11 adapter if one can be constructed.

    Linux-only: imports the adapter module lazily so the
    ``whisperpaw._screen`` module itself stays importable on
    Windows / macOS (where ``whisperpaw._x11`` would also be
    importable but the underlying ``xwd`` binary isn't
    present). The :func:`whisperpaw._x11.build_x11_screen`
    factory returns ``None`` when ``$DISPLAY`` is unset or
    ``xwd`` isn't on ``$PATH`` — both of which are the
    common case on a headless box.
    """
    from whisperpaw import _x11
    return _x11.build_x11_screen()


def _win32_capture() -> ScreenCapture | None:
    """Return a real Win32 GDI adapter if one can be constructed.

    Lazy-imports the adapter module so ``whisperpaw._screen``
    itself stays importable on platforms that don't have the
    ``gdi32`` / ``user32`` DLLs available. The
    :func:`whisperpaw._win32.build_win32_screen` factory
    returns ``None`` on non-Windows platforms — the common
    case on a Linux / macOS dev box.
    """
    from whisperpaw import _win32
    return _win32.build_win32_screen()


def _quartz_capture() -> ScreenCapture | None:
    """Return a real macOS Quartz adapter if one can be constructed.

    Lazy-imports the adapter module so ``whisperpaw._screen``
    itself stays importable on platforms that don't have
    macOS's ``screencapture`` binary available. The
    :func:`whisperpaw._quartz.build_quartz_screen` factory
    returns ``None`` on non-Darwin platforms (and on Darwin
    without ``screencapture``) — the common case on a Linux
    / Windows dev box.
    """
    from whisperpaw import _quartz
    return _quartz.build_quartz_screen()


# ---------------------------------------------------------------------------
# Factory + discovery
# ---------------------------------------------------------------------------


#: Map of backend name → factory function. ``fake`` is special-cased
#: in :func:`get_capture` because it needs a grid argument.
_BACKEND_FACTORIES: dict[str, Any] = {
    "x11": _x11_capture,
    "win32": _win32_capture,
    "quartz": _quartz_capture,
}


def get_capture(
    backend: str = "auto",
    *,
    fake_grid: list[str] | None = None,
) -> ScreenCapture | None:
    """Return a :class:`ScreenCapture` for ``backend`` or ``None``.

    The ``backend`` argument is one of :data:`KNOWN_BACKENDS`, or
    ``"auto"`` (the default) which picks the first adapter whose
    factory returns a non-``None`` instance. The pick order is
    ``x11`` → ``win32`` → ``quartz`` (Linux desktop, then
    Windows, then macOS); ``fake`` is only ever picked when the
    caller asks for it explicitly or passes a ``fake_grid``.

    ``fake_grid`` is required when ``backend == "fake"`` (otherwise
    we don't know what screen to pretend to have). When the
    caller passes a grid but ``backend == "auto"``, we still
    return a :class:`FakeScreen` over that grid — handy for
    scripts that want to test the rest of the pipeline without a
    real display.
    """
    if backend == "auto":
        if fake_grid is not None:
            return FakeScreen(fake_grid)
        for name in KNOWN_BACKENDS:
            if name == "fake":
                continue  # only picked when fake_grid is given
            factory = _BACKEND_FACTORIES.get(name)
            if factory is None:
                continue
            instance = factory()
            if instance is not None:
                return instance
        return None
    if backend == "fake":
        if fake_grid is None:
            raise ValueError(
                "get_capture(backend='fake') requires a fake_grid"
            )
        return FakeScreen(fake_grid)
    if backend in _BACKEND_FACTORIES:
        return _BACKEND_FACTORIES[backend]()
    raise ValueError(
        f"unknown screen-capture backend: {backend!r} "
        f"(known: {', '.join(KNOWN_BACKENDS)})"
    )


def list_backends() -> list[str]:
    """Return the list of known backend names in canonical order.

    The same data the CLI's ``--list-backends`` flag prints,
    exposed as a plain function so ``paw-complete`` and tests
    don't have to reach into the parser.
    """
    return list(KNOWN_BACKENDS)


def to_json(kind: str) -> str:
    """Return a single-line JSON object describing the discovery data.

    Mirrors the convention established by :mod:`whisperpaw.sound`
    and :mod:`whisperpaw.read`: ``--list-backends --json`` emits
    ``{"backends": [...]}`` on a single line, parseable by
    ``json.loads`` / ``jq`` without further work.
    """
    if kind == "backends":
        payload = {"backends": list_backends()}
    else:
        raise ValueError(
            f"to_json: unknown kind {kind!r} (only 'backends' is supported)"
        )
    return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI helpers (the small bits ``whisperpaw/zoom.py`` re-uses)
# ---------------------------------------------------------------------------


def backend_unsupported_message(backend: str) -> str:
    """The stderr message :func:`get_capture` callers print on failure.

    Kept in one place so the wording stays consistent between
    ``--screen`` (which fails if the OS adapter is missing) and
    future flags that might also try to capture.
    """
    return (
        f"paw-zoom: screen capture backend {backend!r} is not yet "
        f"implemented on this OS; pass --backend fake with a "
        f"--fake-grid argument, or pipe text via --file / stdin"
    )


def describe_capture(
    backend: str = "auto",
    *,
    region: str | None = None,
    fake_grid: list[str] | None = None,
    get_capture_fn: Any = None,
) -> dict[str, Any]:
    """Return a small metadata dict describing the screen-capture setup.

    The output is what ``paw-zoom --info`` (and ``--info --json``)
    surface. It has the following keys, all present:

    * ``backend`` — the requested backend name (``"auto"``, ``"fake"``,
      ``"x11"``, ``"win32"``, ``"quartz"``).
    * ``available`` — whether a :class:`ScreenCapture` adapter could
      actually be constructed on this OS. ``True`` for ``"fake"`` with
      a grid; ``True`` on a real desktop for the matching OS adapter;
      ``False`` otherwise.
    * ``adapter`` — the resolved adapter class name (e.g.
      ``"FakeScreen"``, ``"X11Screen"``) or ``None`` when unavailable.
    * ``screen_size`` — ``[width, height]`` if the adapter reported a
      size; ``None`` otherwise.
    * ``region`` — the resolved ``[x, y, w, h]`` after
      :func:`capture_screen_to_source` clamping, or ``None`` when the
      adapter is unavailable.

    The function is intentionally pure: it does not raise on a
    missing backend. The CLI's ``--info`` is "what would the
    pipeline do?" — failing hard here would defeat the point of the
    flag (debugging "why doesn't --screen work on this box?").

    The ``get_capture_fn`` parameter is an injection point so tests
    don't have to monkey-patch the module-level
    :func:`get_capture` global. It defaults to the real
    :func:`get_capture`.
    """
    if get_capture_fn is None:
        get_capture_fn = get_capture
    info: dict[str, Any] = {
        "backend": backend,
        "available": False,
        "adapter": None,
        "screen_size": None,
        "region": None,
    }
    try:
        cap = get_capture_fn(backend, fake_grid=fake_grid)
    except ValueError:
        # An unknown backend or a missing fake_grid. The CLI has
        # already validated --backend and --fake-grid at parse time,
        # so this only fires if a caller wires ``--info`` into a
        # state parse_args would have rejected. We surface the
        # unavailable status and let the caller decide.
        return info
    if cap is None:
        return info
    info["available"] = True
    info["adapter"] = type(cap).__name__
    try:
        w, h = cap.screen_size()
    except Exception:
        # An adapter that advertises availability but whose
        # screen_size() call fails at runtime is still
        # "available" — the dict just can't fill in a size. This
        # is the right shape for diagnostics: "the adapter is
        # there, but something went wrong probing it".
        return info
    info["screen_size"] = [int(w), int(h)]
    if region is None:
        info["region"] = [0, 0, int(w), int(h)]
        return info
    try:
        x, y, rw, rh = _parse_region(region, screen_size=(w, h))
    except ValueError:
        # A region the parser would have accepted but that
        # _parse_region still rejects (e.g. negative w/h) — leave
        # the region as None so the user sees "we couldn't
        # compute it" rather than a wrong answer.
        return info
    info["region"] = [int(x), int(y), int(rw), int(rh)]
    return info


def describe_to_text(info: dict[str, Any]) -> str:
    """Render :func:`describe_capture`'s dict as a human-readable string.

    One ``key: value`` per line, in a fixed order, so the output is
    diffable and easy to grep. The ``screen_size`` and ``region``
    tuples are formatted as ``"WIDTHxHEIGHT"`` and
    ``"X,Y,W,H"`` respectively to match the conventions the rest
    of the project uses for the same data.
    """
    lines: list[str] = []
    lines.append(f"backend: {info.get('backend')!s}")
    lines.append(f"available: {'yes' if info.get('available') else 'no'}")
    lines.append(f"adapter: {info.get('adapter') or '-'}")
    size = info.get("screen_size")
    if size is not None:
        lines.append(f"screen_size: {size[0]}x{size[1]}")
    else:
        lines.append("screen_size: -")
    region = info.get("region")
    if region is not None:
        lines.append(f"region: {region[0]},{region[1]},{region[2]},{region[3]}")
    else:
        lines.append("region: -")
    return "\n".join(lines)


def describe_to_json(info: dict[str, Any]) -> str:
    """Return :func:`describe_capture`'s dict as a single-line JSON string.

    Same shape as the dict (no flattening), single line, parseable
    by ``json.loads`` / ``jq``. ``screen_size`` and ``region``
    are emitted as JSON arrays (``[w, h]`` / ``[x, y, w, h]``) so
    downstream tooling can index them positionally.
    """
    return json.dumps(info, ensure_ascii=False)


def _read_fake_grid(text: str) -> list[str]:
    """Turn a ``--fake-grid`` CLI string into the list-of-strings
    shape :class:`FakeScreen` wants.

    Splits on ``"\\n"``; trailing newlines are dropped (so
    ``--fake-grid "abc\\ndef\\n"`` becomes ``["abc", "def"]``,
    not ``["abc", "def", ""]``). Empty input raises
    ``ValueError`` because a FakeScreen with an empty grid is
    meaningless.
    """
    if not text:
        raise ValueError("--fake-grid must be non-empty")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if not lines or all(line == "" for line in lines):
        raise ValueError("--fake-grid must contain at least one non-empty row")
    return lines


# Re-export the read-fake-grid helper for the CLI; kept private
# to this module's namespace so the public surface stays focused
# on the adapter contract.
__all__ = [
    "KNOWN_BACKENDS",
    "FULL_SCREEN_REGION",
    "ScreenCapture",
    "FakeScreen",
    "get_capture",
    "list_backends",
    "to_json",
    "describe_capture",
    "describe_to_text",
    "describe_to_json",
    "capture_screen_to_source",
    "backend_unsupported_message",
]


# ---------------------------------------------------------------------------
# Argument parsing (a small parser fragment the zoom CLI composes in)
# ---------------------------------------------------------------------------


def add_screen_args(
    parser: Any,
    *,
    default_backend: str = "auto",
) -> None:
    """Add the ``--screen`` / ``--region`` / ``--backend`` flags to ``parser``.

    A separate function so the parser in
    :func:`whisperpaw.zoom.build_parser` doesn't need to import
    every adapter module by name. The default backend is ``auto``
    which tries each OS in order.
    """
    group = parser.add_argument_group("screen capture (v0.2)")
    group.add_argument(
        "--screen",
        action="store_true",
        help=(
            "Capture a rectangular region of the visual screen "
            "and magnify it, instead of magnifying a text source. "
            "On platforms without a screen-capture adapter yet "
            "(Linux X11, Windows, macOS), combine with "
            "--backend fake --fake-grid '...' to test the "
            "pipeline. The rectangle is --region X,Y,W,H "
            "(default: full screen)."
        ),
    )
    group.add_argument(
        "--region",
        default=None,
        metavar="X,Y,W,H",
        help=(
            "Screen rectangle to capture when --screen is set. "
            "Either 'X,Y,W,H' (four non-negative integers) or "
            "'full' (the whole screen, the default). The region "
            "is clamped to the screen's extent."
        ),
    )
    group.add_argument(
        "--backend",
        default=default_backend,
        choices=("auto",) + KNOWN_BACKENDS,
        help=(
            "Screen-capture backend to use when --screen is set "
            "(default: auto — picks the first available OS "
            "adapter). 'fake' is always available if you also "
            "pass --fake-grid."
        ),
    )
    group.add_argument(
        "--fake-grid",
        default=None,
        metavar="TEXT",
        help=(
            "With --backend fake, the text used as the 'screen' "
            "(rows separated by '\\n'). Required when "
            "--backend fake is set; ignored otherwise. Useful "
            "for testing the v0.2 pipeline on a headless box."
        ),
    )
    group.add_argument(
        "--list-backends",
        action="store_true",
        dest="list_backends",
        help=(
            "Print the names of every supported screen-capture "
            "backend, one per line, and exit. Nothing is "
            "captured. Useful for discovery and for shell "
            "completion."
        ),
    )


def parse_screen_args(args: Any) -> dict[str, Any]:
    """Pull the screen-capture fields out of an argparse ``Namespace``.

    Returns a dict with the keys the zoom CLI needs (``region``,
    ``fake_grid``, ``backend``, ``list_backends``). The caller is
    responsible for passing ``fake_grid`` to
    :func:`_read_fake_grid` when ``backend == "fake"``.

    The dict shape is intentional: it mirrors the namespace layout
    so the caller can do ``namespace.update(parse_screen_args(...))``
    if it wants to.
    """
    return {
        "region": getattr(args, "region", None),
        "fake_grid": getattr(args, "fake_grid", None),
        "backend": getattr(args, "backend", "auto"),
        "list_backends": bool(getattr(args, "list_backends", False)),
    }
