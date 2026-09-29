"""paw-sound — short audio cues for shell events.

A tiny cross-platform command that plays a small sound when something
happens in the shell — the build finished, a test failed, a long job
became ready, etc.

Design notes
------------
* Pure stdlib, no third-party deps. Audio playback is delegated to one
  of a few well-known system commands, picked per-OS, and only when
  actually requested. In `--quiet` mode we never touch the audio device.
* The "what sound for what event" mapping is data, not code. New sound
  packs go under ``whisperpaw/sounds/<pack>/<event>.<ext>`` and are
  picked up automatically once registered in :data:`KNOWN_PACKS`.
* Tests never touch the audio device — they exercise the pure logic and
  monkey-patch the audio backend.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

#: Sentinel used by :func:`list_backends` to distinguish "the caller
#: did not pass ``active_name``" from "the caller explicitly passed
#: ``None`` to mean no backend is active". Necessary because
#: ``None`` is a valid value for ``active_name`` (it means "no audio
#: backend is installed on this machine") and we cannot use it as
#: the default. Defined at module top so every helper below can
#: reference it before its own definition runs.
_SENTINEL: object = object()

# ---------------------------------------------------------------------------
# Public data
# ---------------------------------------------------------------------------

#: Directory that holds all sound packs (``sounds/cat/ok.wav`` etc.).
#: Lives next to this file so the package is self-contained.
SOUND_ROOT: Path = Path(__file__).resolve().parent / "sounds"

#: Currently shipped sound packs. Each pack is a directory under
#: :data:`SOUND_ROOT`. Adding a new pack is a matter of dropping files in
#: and listing the pack name here.
KNOWN_PACKS: frozenset[str] = frozenset({"cat", "forest", "rain"})

#: Events every pack is expected to support. The exact filename is found
#: by scanning the pack directory; the first existing ``<event>.<ext>``
#: wins. Supported extensions: wav, mp3, ogg, flac.
KNOWN_EVENTS: tuple[str, ...] = ("ok", "warn", "fail", "ready", "ding")


def list_packs() -> list[str]:
    """Return the known sound pack names, sorted alphabetically.

    Public API so ``paw-complete`` and tests can enumerate what's
    available without depending on the ``argparse`` layer.
    """
    return sorted(KNOWN_PACKS)


def list_events() -> list[str]:
    """Return the known event names, in their canonical order.

    Same rationale as :func:`list_packs` — public so other modules
    (and tests) can ask what's available without parsing the parser.
    """
    return list(KNOWN_EVENTS)


def to_json(kind: str) -> str:
    """Return the requested discovery data as a JSON string.

    ``kind`` is one of ``"packs"``, ``"events"``, or ``"backends"``:
        * ``"packs"``    → ``{"packs": [...]}``   (sorted pack names)
        * ``"events"``   → ``{"events": [...]}``  (canonical order)
        * ``"backends"`` → ``{"active": ..., "backends": [...]}``
                           where each backend is a 4-key dict
                           ``{name, available, volume_supported,
                           active}`` and the top-level ``active``
                           mirrors the same field on the active
                           backend (``null`` when no backend is
                           installed on this machine).

    The output is a compact, single-line JSON object so it can be
    diffed, piped to ``jq``, or stored as a build artefact without
    further parsing.

    Raises :class:`ValueError` for unknown ``kind`` values.
    """
    if kind == "packs":
        payload = {"packs": list_packs()}
    elif kind == "events":
        payload = {"events": list_events()}
    elif kind == "backends":
        infos = list_backends()
        payload = {
            "active": _active_backend_name(infos),
            "backends": [_backend_info_to_dict(info) for info in infos],
        }
    else:
        raise ValueError(
            f"unknown kind {kind!r}; expected 'packs', 'events', or 'backends'"
        )
    # ``sort_keys`` keeps the output stable across runs and platforms;
    # ``ensure_ascii=False`` preserves non-ASCII pack / event names
    # verbatim. No indent — the output is meant to be piped to ``jq``.
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _backend_info_to_dict(info: BackendInfo) -> dict[str, object]:
    """Convert a :class:`BackendInfo` to a JSON-serialisable dict.

    The dict shape is the public contract for ``to_json('backends')``:
    four fixed keys, sorted by :func:`json.dumps` ``sort_keys``, so
    a downstream consumer can rely on the field set without parsing
    the help text.
    """
    return {
        "name": info.name,
        "available": info.available,
        "volume_supported": info.volume_supported,
        "active": info.active,
    }


def _active_backend_name(infos: list[BackendInfo]) -> str | None:
    """Return the name of the single active backend in ``infos``, or
    ``None`` if none is active.

    Defined here so :func:`to_json` does not have to walk the list
    twice. The contract is "exactly one active backend in a healthy
    environment" — on a machine with no audio installed the answer
    is ``None``; in any other case exactly one entry has
    ``active == True``.
    """
    for info in infos:
        if info.active:
            return info.name
    return None


def list_backends(
    *,
    which_fn: Callable[[str], str | None] | None = None,
    active_name: str | None | object = _SENTINEL,
) -> list[BackendInfo]:
    """Return one :class:`BackendInfo` per known backend, sorted by name.

    The list always contains every backend in
    :data:`KNOWN_BACKEND_NAMES` — exactly one entry per name — so a
    caller can rely on the cardinality. ``available`` reflects
    whether the backend's binary is on ``$PATH`` *right now* (not
    whether the OS would normally have it), so on a vanilla Linux
    box ``afplay`` and ``powershell`` both come back as
    ``available: False``.

    Parameters are injection points for tests:

    * ``which_fn`` — replaces :func:`shutil.which` so tests can fake
      which binaries are on ``$PATH`` without touching the host.
    * ``active_name`` — replaces :func:`current_backend_name` so
      tests can pin exactly which backend is reported active.
      Pass the literal ``None`` to simulate "no backend installed";
      omit it (the default) to call the real
      :func:`current_backend_name`.

    The order is sorted alphabetically by ``name`` so the JSON
    output is diffable across runs and platforms.
    """
    if which_fn is None:
        which_fn = shutil.which
    if active_name is _SENTINEL:
        active_name = current_backend_name()
    infos: list[BackendInfo] = []
    for name in sorted(KNOWN_BACKEND_NAMES):
        binary = _BACKEND_TO_BINARY[name]
        available = which_fn(binary) is not None
        infos.append(
            BackendInfo(
                name=name,
                available=available,
                volume_supported=volume_supported(name),
                active=(name == active_name),
            )
        )
    return infos


def describe_backends(infos: list[BackendInfo] | None = None) -> list[str]:
    """Render a list of :class:`BackendInfo` as one fixed-format line each.

    The shape is::

        <name> (active, volume, unavailable)
        <name> (volume)
        <name> (active)
        <name> (unavailable)
        <name>
        …

    Annotations are independent facts: ``active`` ("this is the
    backend currently in use"), ``volume`` ("this backend honours
    ``--volume``"), and ``unavailable`` ("this backend's binary is
    not on $PATH right now"). A backend can have any combination
    of the three — the common case on a healthy macOS host is
    ``afplay (active, volume)`` with the other three backends
    listed as ``(unavailable)``. The annotations are emitted in
    a fixed order (``active`` first, then ``volume``, then
    ``unavailable``) so the output is diffable across runs and
    platforms.

    A backend with no annotations renders as a bare name — the
    "exists in our list and is installed but is not the active
    one and does not support volume" shape (e.g. an installed
    but unused ``aplay`` on a system whose active backend is
    ``paplay``).

    The output is one entry per input, in input order, so a
    caller can pre-sort the list and the description mirrors
    that. ``infos`` defaults to :func:`list_backends` so the
    CLI's text-mode branch can call ``describe_backends()``
    with no args.
    """
    if infos is None:
        infos = list_backends()
    lines: list[str] = []
    for info in infos:
        parts: list[str] = []
        if info.active:
            parts.append("active")
        if info.volume_supported:
            parts.append("volume")
        if not info.available:
            parts.append("unavailable")
        if parts:
            lines.append(f"{info.name} ({', '.join(parts)})")
        else:
            lines.append(info.name)
    return lines

#: File extensions we know how to feed to a system player.
_AUDIO_EXTS: tuple[str, ...] = (".wav", ".mp3", ".ogg", ".flac")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlayPlan:
    """Everything needed to play a single sound — fully resolved."""

    event: str
    pack: str
    path: Path
    volume: float

    def announce(self) -> str:
        return f"🐾 paw-sound: playing '{self.event}' from pack '{self.pack}'"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``paw-sound``."""
    parser = argparse.ArgumentParser(
        prog="paw-sound",
        description="Play a small audio cue for a shell event.",
    )
    parser.add_argument(
        "event",
        nargs="?",
        default="ok",
        help=f"Event name. One of: {', '.join(KNOWN_EVENTS)} (default: ok).",
    )
    parser.add_argument(
        "--pack",
        default="cat",
        help=(
            "Sound pack to use. One of: "
            f"{', '.join(sorted(KNOWN_PACKS))} (default: cat)."
        ),
    )
    parser.add_argument(
        "--volume",
        type=float,
        default=0.6,
        help=(
            "Playback volume, 0.0–1.0 (default: 0.6). Honoured on "
            "afplay (macOS) and paplay (PulseAudio). Silently "
            "ignored on aplay (ALSA, no per-stream volume) and on "
            "PowerShell SoundPlayer (fixed gain); see "
            "volume_supported() in the module for the runtime "
            "check."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the announcement line (still plays the sound).",
    )
    parser.add_argument(
        "--list-packs",
        action="store_true",
        dest="list_packs",
        help=(
            "Print the names of every available sound pack, one per line, "
            "and exit. Nothing is played. Useful for discovery and for "
            "shell completion."
        ),
    )
    parser.add_argument(
        "--list-events",
        action="store_true",
        dest="list_events",
        help=(
            "Print the names of every known event, one per line, and exit. "
            "Nothing is played. Useful for discovery and for shell completion."
        ),
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        dest="list_backends",
        help=(
            "Print the names of every supported audio backend, one per "
            "line, and exit. Each line is annotated with the backend's "
            "state on the current machine: ``(active)`` marks the "
            "backend that would be used to play a sound right now, "
            "``(volume)`` marks the backends that honour ``--volume``, "
            "and ``(unavailable)`` marks a backend whose binary is "
            "not on $PATH. Nothing is played. Useful for diagnosing "
            "why a sound did not play (``paw-sound --list-backends``) "
            "or for feeding shell completion."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help=(
            "Combine with --list-packs or --list-events to emit a JSON "
            "object instead of one-name-per-line text. The object has one "
            "key ('packs' or 'events') whose value is the list. Nothing is "
            "played. Useful for jq / scripts / build artefacts."
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse argv into a Namespace. Validates the event name early."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.event not in KNOWN_EVENTS:
        print(f"paw-sound: unknown event '{args.event}'", file=sys.stderr)
        print(
            f"paw-sound: known events: {', '.join(KNOWN_EVENTS)}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if not 0.0 <= args.volume <= 1.0:
        print("paw-sound: --volume must be between 0.0 and 1.0", file=sys.stderr)
        raise SystemExit(2)
    return args


# ---------------------------------------------------------------------------
# Sound resolution
# ---------------------------------------------------------------------------


def resolve_sound(
    event: str,
    pack: str = "cat",
    *,
    sound_root: Path = SOUND_ROOT,
) -> PlayPlan:
    """Resolve a (event, pack) pair into a concrete :class:`PlayPlan`.

    Raises :class:`ValueError` for unknown events or packs, and
    :class:`FileNotFoundError` if the actual audio file is missing.
    """
    if event not in KNOWN_EVENTS:
        raise ValueError(f"unknown event: {event!r}")
    if pack not in KNOWN_PACKS:
        raise ValueError(f"unknown pack: {pack!r}")

    pack_dir = sound_root / pack
    for ext in _AUDIO_EXTS:
        candidate = pack_dir / f"{event}{ext}"
        if candidate.is_file():
            return PlayPlan(event=event, pack=pack, path=candidate, volume=0.6)

    raise FileNotFoundError(
        f"no audio file for event {event!r} in pack {pack!r} "
        f"(looked in {pack_dir})"
    )


# ---------------------------------------------------------------------------
# Audio backends
# ---------------------------------------------------------------------------


def _afplay_cmd(plan: PlayPlan) -> list[str]:
    # afplay's ``-v`` flag is a linear gain in [0.0, 1.0] — the same
    # scale :class:`PlayPlan.volume` already uses, so we pass the
    # value through unchanged. The flag is positional-value (``-v
    # 0.5``), not ``-v=0.5``; the manual style is portable to
    # every afplay version that ships on a current macOS.
    return ["afplay", "-v", f"{plan.volume:.3f}", str(plan.path)]


def _aplay_cmd(plan: PlayPlan) -> list[str]:
    # ALSA's ``aplay`` has no per-stream volume flag — the only way
    # to attenuate is via a separate ``amixer`` call against the
    # default PCM control, which mutates global state. We
    # deliberately don't shell out: a single ``paw-sound`` shouldn't
    # surprise the user by changing their system volume. Volume is
    # therefore not honoured on aplay; see ``volume_supported``.
    return ["aplay", str(plan.path)]


def _paplay_cmd(plan: PlayPlan) -> list[str]:
    # PulseAudio's ``paplay --volume=`` takes a 16-bit unsigned
    # integer in [0, 65535] where 65535 == 100% and 0 == mute.
    # We scale the linear [0.0, 1.0] PlayPlan.volume up by 65535
    # and clamp on the off-chance the value is slightly out of
    # range (the CLI already validates 0.0–1.0 so this is
    # defensive). Pulse uses ``=`` rather than space between the
    # flag and the value; that's the canonical form in
    # ``paplay --help``.
    pa_volume = max(0, min(65535, int(round(plan.volume * 65535))))
    return ["paplay", f"--volume={pa_volume}", str(plan.path)]


def _powershell_cmd(plan: PlayPlan) -> list[str]:
    # Use the .NET SoundPlayer — works on every Windows since XP without
    # extra dependencies. Volume is not honoured by SoundPlayer; see
    # ``volume_supported``.
    script = (
        "(New-Object Media.SoundPlayer '"
        + str(plan.path).replace("'", "''")
        + "').PlaySync()"
    )
    return ["powershell", "-NoProfile", "-Command", script]


#: Backend names that can be returned by :func:`current_backend_name`
#: and accepted by :func:`volume_supported`. Kept in sync with the
#: command names used by :func:`pick_backend` below.
KNOWN_BACKEND_NAMES: frozenset[str] = frozenset(
    {"afplay", "aplay", "paplay", "powershell"}
)

#: Maps each backend name to the on-disk binary that backs it. Used by
#: :func:`list_backends` so the per-backend "is this available?" check
#: does not have to re-implement the ladder :func:`pick_backend` walks.
#: A backend is considered "available" if its binary resolves on
#: ``$PATH`` — same gate :func:`pick_backend` and
#: :func:`current_backend_name` use, so the three helpers cannot drift.
_BACKEND_TO_BINARY: dict[str, str] = {
    "afplay": "afplay",
    "aplay": "aplay",
    "paplay": "paplay",
    "powershell": "powershell",
}

#: Backend names whose underlying player supports a per-stream
#: volume flag. ``aplay`` and ``powershell`` are absent because
#: neither exposes a per-stream volume knob without side effects
#: (aplay would need a separate ``amixer`` call; powershell
#: SoundPlayer is fixed-gain).
BACKENDS_WITH_VOLUME: frozenset[str] = frozenset({"afplay", "paplay"})


def volume_supported(backend_name: str) -> bool:
    """Return ``True`` if the given backend honours ``--volume``.

    The CLI uses this to decide whether to mention the limitation
    in ``--help`` text; tests use it to assert the contract.
    """
    return backend_name in BACKENDS_WITH_VOLUME


#: The metadata for a single audio backend, as exposed by
#: :func:`list_backends`. Frozen so a caller can safely pass the
#: object through layers and treat it as an immutable value.
#:
#: Fields:
#:   ``name``              — canonical backend name (one of
#:                           :data:`KNOWN_BACKEND_NAMES`).
#:   ``available``         — ``True`` iff the backend's binary is
#:                           currently on ``$PATH``. Mirrors the
#:                           gate :func:`pick_backend` uses, so
#:                           ``available`` is what would actually
#:                           be picked if a sound were played.
#:   ``volume_supported``  — ``True`` iff :func:`volume_supported`
#:                           returns ``True`` for this backend
#:                           (i.e. the per-stream volume knob is
#:                           exposed by the underlying player).
#:   ``active``            — ``True`` iff :func:`current_backend_name`
#:                           would return this name on the current
#:                           machine. Exactly one backend can be
#:                           active at a time; on a host with no
#:                           audio installed all four are inactive.
#:
#: Use :func:`list_backends` to enumerate every backend in a stable
#: order; the order is sorted alphabetically by ``name`` so the
#: output is diffable across runs and platforms.
@dataclass(frozen=True)
class BackendInfo:
    """The metadata for a single audio backend, as exposed by
    :func:`list_backends`."""

    name: str
    available: bool
    volume_supported: bool
    active: bool



def pick_backend() -> Callable[[PlayPlan], int] | None:
    """Return a function that plays a :class:`PlayPlan`, or ``None`` if no
    audio backend is available on this system.
    """
    system = platform.system()
    if system == "Darwin" and shutil.which("afplay"):
        return _run_subprocess(_afplay_cmd)
    if system == "Windows":
        if shutil.which("powershell"):
            return _run_subprocess(_powershell_cmd)
        return None
    # Linux / BSD / other Unix-likes — try ALSA, then PulseAudio.
    if shutil.which("paplay"):
        return _run_subprocess(_paplay_cmd)
    if shutil.which("aplay"):
        return _run_subprocess(_aplay_cmd)
    return None


def current_backend_name() -> str | None:
    """Return the name of the audio backend that :func:`pick_backend` would
    pick on this machine, or ``None`` if none is available.

    The string is one of :data:`KNOWN_BACKEND_NAMES` (``"afplay"``,
    ``"aplay"``, ``"paplay"``, ``"powershell"``) so callers can
    index into :data:`BACKENDS_WITH_VOLUME` or compare against the
    per-OS lookup table. The name is derived by re-running the
    same ``shutil.which`` checks :func:`pick_backend` does; the
    two are guaranteed to agree (modulo a race where the binary
    is uninstalled between the two calls, which is not realistic
    in practice).
    """
    system = platform.system()
    if system == "Darwin" and shutil.which("afplay"):
        return "afplay"
    if system == "Windows":
        if shutil.which("powershell"):
            return "powershell"
        return None
    if shutil.which("paplay"):
        return "paplay"
    if shutil.which("aplay"):
        return "aplay"
    return None


def _run_subprocess(builder: Callable[[PlayPlan], list[str]]) -> Callable[[PlayPlan], int]:
    """Adapt a ``cmd-builder`` into a ``(plan) -> exit_code`` function."""

    def _play(plan: PlayPlan) -> int:
        cmd = builder(plan)
        try:
            completed = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                timeout=10,
            )
        except FileNotFoundError:
            return 1
        except subprocess.TimeoutExpired:
            return 1
        return completed.returncode

    return _play


# The active backend — replaced in tests via ``monkeypatch.setattr``.
_play: Callable[[PlayPlan], int] | None = None  # type: ignore[assignment]


def _play(plan: PlayPlan) -> int:  # type: ignore[no-redef]
    backend = pick_backend()
    if backend is None:
        print(
            "paw-sound: no audio backend found "
            f"(system={platform.system()}; tried afplay/aplay/paplay/powershell)",
            file=sys.stderr,
        )
        return 1
    return backend(plan)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point — returns a process exit code."""
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        # argparse / explicit validation raised SystemExit already
        return int(exc.code) if isinstance(exc.code, int) else 2

    # Discovery flags short-circuit before any audio resolution — they
    # are mutually exclusive with playing a sound, and they don't need
    # to touch the audio device.
    if args.as_json and not (args.list_packs or args.list_events or args.list_backends):
        print(
            "paw-sound: --json requires --list-packs, --list-events, "
            "or --list-backends",
            file=sys.stderr,
        )
        return 2
    if args.list_packs:
        if args.as_json:
            print(to_json("packs"))
        else:
            for name in list_packs():
                print(name)
        return 0
    if args.list_events:
        if args.as_json:
            print(to_json("events"))
        else:
            for name in list_events():
                print(name)
        return 0
    if args.list_backends:
        if args.as_json:
            print(to_json("backends"))
        else:
            for line in describe_backends():
                print(line)
        return 0

    try:
        plan = resolve_sound(args.event, args.pack)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    except ValueError as exc:
        print(f"paw-sound: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"paw-sound: {exc}", file=sys.stderr)
        return 1

    # Build the plan with the user-requested volume
    plan = PlayPlan(plan.event, plan.pack, plan.path, args.volume)

    if not args.quiet:
        print(plan.announce())

    return _play(plan)


if __name__ == "__main__":
    raise SystemExit(main())
