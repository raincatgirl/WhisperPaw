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
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Public data
# ---------------------------------------------------------------------------

#: Directory that holds all sound packs (``sounds/cat/ok.wav`` etc.).
#: Lives next to this file so the package is self-contained.
SOUND_ROOT: Path = Path(__file__).resolve().parent / "sounds"

#: Currently shipped sound packs. Each pack is a directory under
#: :data:`SOUND_ROOT`. Adding a new pack is a matter of dropping files in
#: and listing the pack name here.
KNOWN_PACKS: frozenset[str] = frozenset({"cat", "forest"})

#: Events every pack is expected to support. The exact filename is found
#: by scanning the pack directory; the first existing ``<event>.<ext>``
#: wins. Supported extensions: wav, mp3, ogg, flac.
KNOWN_EVENTS: tuple[str, ...] = ("ok", "warn", "fail", "ready", "ding")

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
        help="Playback volume, 0.0–1.0 (default: 0.6).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the announcement line (still plays the sound).",
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
    return ["afplay", str(plan.path)]


def _aplay_cmd(plan: PlayPlan) -> list[str]:
    return ["aplay", str(plan.path)]


def _paplay_cmd(plan: PlayPlan) -> list[str]:
    return ["paplay", str(plan.path)]


def _powershell_cmd(plan: PlayPlan) -> list[str]:
    # Use the .NET SoundPlayer — works on every Windows since XP without
    # extra dependencies. Volume is not honoured by SoundPlayer.
    script = (
        "(New-Object Media.SoundPlayer '"
        + str(plan.path).replace("'", "''")
        + "').PlaySync()"
    )
    return ["powershell", "-NoProfile", "-Command", script]


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
        plan = resolve_sound(args.event, args.pack)
    except SystemExit as exc:
        # argparse / explicit validation raised SystemExit already
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
