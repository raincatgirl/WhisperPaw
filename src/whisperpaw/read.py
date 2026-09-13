"""paw-read — read text aloud via the operating system's speech engine.

A small cross-platform TTS wrapper. Accepts text from:

* a positional argument        ``paw-read "hello there"``
* ``--file PATH``              a UTF-8 text file
* ``--clipboard``              whatever the platform clipboard has
* stdin (default)              ``echo "hi" | paw-read``

The shape mirrors :mod:`whisperpaw.sound`: arg parser, source resolver,
chunking for long text, OS-aware backend picker, CLI entry point with
sensible exit codes (0 ok, 1 backend/env, 2 usage).

Backends tried, in order, on each platform:

* **macOS**     ``say``        (system, always present)
* **Linux**     ``spd-say``    (speech-dispatcher — most desktops) → ``espeak``
* **Windows**   PowerShell    ``System.Speech.Synthesis`` (SAPI)

If nothing is available, ``paw-read`` prints a clear message and
exits 1 — it does NOT install packages for you.
"""
from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

#: Lowest acceptable speech rate in words-per-minute.
MIN_RATE: float = 80.0
#: Highest acceptable speech rate in words-per-minute.
MAX_RATE: float = 600.0
#: Default chunk size for long text — most engines can swallow ~200 chars.
DEFAULT_MAX_CHARS: int = 200


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paw-read",
        description="Read text aloud using the OS speech engine.",
    )
    parser.add_argument(
        "text",
        nargs=argparse.REMAINDER,
        help=(
            "Text to read. If omitted, read from --file / --clipboard / stdin "
            "(in that priority order)."
        ),
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Read text from this UTF-8 file instead of positional args.",
    )
    parser.add_argument(
        "--clipboard",
        action="store_true",
        dest="use_clipboard",
        help="Read text from the system clipboard.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=200.0,
        help=f"Speech rate, words per minute ({int(MIN_RATE)}–{int(MAX_RATE)}, default: 200).",
    )
    parser.add_argument(
        "--volume",
        type=float,
        default=1.0,
        help="Playback volume, 0.0–1.0 (default: 1.0; ignored on Windows).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help=(
            f"Soft max chars per spoken chunk (default: {DEFAULT_MAX_CHARS}). "
            "Long text is split on sentence boundaries."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the announcement line (still speaks).",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    # argparse.REMAINDER returns a list; join into a single string for the
    # "text" attribute, but keep it None if empty.
    if args.text:
        args.text = " ".join(args.text)
    else:
        args.text = None
    if not (MIN_RATE <= args.rate <= MAX_RATE):
        print(
            f"paw-read: --rate must be between {int(MIN_RATE)} and {int(MAX_RATE)} wpm",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if not 0.0 <= args.volume <= 1.0:
        print("paw-read: --volume must be between 0.0 and 1.0", file=sys.stderr)
        raise SystemExit(2)
    if args.max_chars < 20:
        print("paw-read: --max-chars must be at least 20", file=sys.stderr)
        raise SystemExit(2)
    return args


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------


def read_clipboard() -> str | None:
    """Read the system clipboard. Returns ``None`` if no backend works.

    Uses ``pbcopy/pbpaste`` on macOS, ``wl-paste`` / ``xclip`` / ``xsel`` on
    Linux, and the Win32 clipboard via PowerShell on Windows.
    """
    system = platform.system()
    try:
        if system == "Darwin" and shutil.which("pbpaste"):
            out = subprocess.run(
                ["pbpaste"], check=False, capture_output=True, text=True, timeout=3
            )
            if out.returncode == 0:
                return out.stdout
        elif system == "Windows" and shutil.which("powershell"):
            script = "Get-Clipboard"
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                check=False, capture_output=True, text=True, timeout=3,
            )
            if out.returncode == 0:
                return out.stdout
        else:
            # Linux / BSD: try Wayland then X11 tools in order.
            for cmd in (["wl-paste"], ["xclip", "-selection", "clipboard", "-o"],
                        ["xsel", "--clipboard", "--output"]):
                if shutil.which(cmd[0]):
                    out = subprocess.run(
                        cmd, check=False, capture_output=True, text=True, timeout=3
                    )
                    if out.returncode == 0:
                        return out.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    return None


def _read_stdin() -> str:
    """Read all of stdin as text. Returns ``""`` if stdin is a TTY.

    Test-friendly: returns ``""`` when ``WPAW_READ_STDIN_OVERRIDE`` is set
    in the environment, so tests don't trip pytest's stdin capture.
    """
    import os
    if os.environ.get("WPAW_READ_STDIN_OVERRIDE") is not None:
        return os.environ["WPAW_READ_STDIN_OVERRIDE"]
    if sys.stdin.isatty():
        return ""
    return sys.stdin.read()


def resolve_source(
    *,
    text: str | None,
    file: str | None,
    use_clipboard: bool,
    stdin_text: str,
) -> str:
    """Pick the first non-empty source in priority order.

    Priority: positional ``text`` → ``--clipboard`` → ``--file`` → ``stdin``.
    Empty / whitespace-only results from any source are skipped.
    """
    def _clean(s: str) -> str:
        return s.strip()

    if text is not None and _clean(text):
        return _clean(text)
    if use_clipboard:
        clip = read_clipboard()
        if clip is None:
            raise RuntimeError(
                "paw-read: --clipboard requested but no clipboard backend is "
                "available (tried pbpaste / powershell / wl-paste / xclip / xsel)"
            )
        if _clean(clip):
            return _clean(clip)
    if file is not None:
        path = Path(file)
        if not path.is_file():
            raise FileNotFoundError(f"paw-read: file not found: {file}")
        return _clean(path.read_text(encoding="utf-8"))
    if _clean(stdin_text):
        return _clean(stdin_text)
    raise RuntimeError(
        "paw-read: no text to read — pass a positional argument, "
        "--file, --clipboard, or pipe via stdin"
    )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Split ``text`` into chunks no longer than ``max_chars`` chars.

    Prefers sentence boundaries (``.``, ``!``, ``?`` followed by space);
    falls back to whitespace; never breaks a single word.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining.strip())
            break
        # Look for a sentence boundary inside the window
        window = remaining[:max_chars]
        cut_at = -1
        for sep in (". ", "! ", "? ", "。", "！", "？", "\n\n"):
            idx = window.rfind(sep)
            if idx > cut_at:
                cut_at = idx + len(sep)
        if cut_at <= 0:
            # Fall back to whitespace
            sp = window.rfind(" ")
            if sp > 0:
                cut_at = sp + 1
            else:
                # Pathological: one giant word — emit as-is
                cut_at = max_chars
        chunks.append(remaining[:cut_at].strip())
        remaining = remaining[cut_at:].lstrip()
    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Audio backends
# ---------------------------------------------------------------------------


def _say_cmd(text: str, rate: float, volume: float) -> list[str]:
    return ["say", "-r", str(int(rate)), text]


def _spd_say_cmd(text: str, rate: float, volume: float) -> list[str]:
    # spd-say has no rate argument, but supports -i / -r for speed (-r is WPM).
    return ["spd-say", "-r", str(int(rate)), "-i", str(int(volume * 100)), text]


def _espeak_cmd(text: str, rate: float, volume: float) -> list[str]:
    # espeak: -s = speed in words per minute, -a = amplitude 0..200
    return [
        "espeak",
        "-s", str(int(rate)),
        "-a", str(int(volume * 200)),
        text,
    ]


def _sapi_cmd(text: str, rate: float, volume: float) -> list[str]:
    # PowerShell SAPI. Rate: -10..10 (we map from WPM); volume: 0..100.
    sapi_rate = max(-10, min(10, int((rate - 200) / 40)))
    sapi_vol = int(volume * 100)
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.Rate = {sapi_rate}; "
        f"$s.Volume = {sapi_vol}; "
        f"$s.Speak({text!r})"
    )
    return ["powershell", "-NoProfile", "-Command", script]


def pick_backend() -> Callable[[str, float, float], int] | None:
    """Return a function ``(text, rate, volume) -> exit_code`` or ``None``."""
    system = platform.system()
    if system == "Darwin" and shutil.which("say"):
        return _run_subprocess(_say_cmd)
    if system == "Windows" and shutil.which("powershell"):
        return _run_subprocess(_sapi_cmd)
    # Linux / BSD / other Unix-likes
    if shutil.which("spd-say"):
        return _run_subprocess(_spd_say_cmd)
    if shutil.which("espeak"):
        return _run_subprocess(_espeak_cmd)
    return None


def _run_subprocess(builder: Callable[[str, float, float], list[str]]) -> Callable[[str, float, float], int]:
    def _speak(text: str, rate: float, volume: float) -> int:
        cmd = builder(text, rate, volume)
        try:
            completed = subprocess.run(
                cmd, check=False, capture_output=True, timeout=60
            )
        except FileNotFoundError:
            return 1
        except subprocess.TimeoutExpired:
            return 1
        return completed.returncode
    return _speak


def _speak(text: str, rate: float, volume: float) -> int:
    backend = pick_backend()
    if backend is None:
        print(
            "paw-read: no TTS backend found "
            f"(system={platform.system()}; tried say/spd-say/espeak/powershell)",
            file=sys.stderr,
        )
        return 1
    return backend(text, rate, volume)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        text = resolve_source(
            text=args.text,
            file=args.file,
            use_clipboard=args.use_clipboard,
            stdin_text=_read_stdin(),
        )
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    chunks = chunk_text(text, max_chars=args.max_chars)
    if not chunks:
        print("paw-read: nothing to read (text is empty after normalization)", file=sys.stderr)
        return 2

    if not args.quiet:
        preview = chunks[0] if len(chunks) == 1 else f"{chunks[0]}… ({len(chunks)} chunks)"
        print(f"🐾 paw-read: {preview}")

    last_code = 0
    for chunk in chunks:
        code = _speak(chunk, args.rate, args.volume)
        if code != 0:
            last_code = code
    return last_code


if __name__ == "__main__":
    raise SystemExit(main())
