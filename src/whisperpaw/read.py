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
import glob
import json
import os
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

#: Backend names accepted by ``--backend``. The set is also exposed
#: publicly via :func:`list_backends` so discovery and the CLI stay in
#: lock-step with argparse's ``choices=`` list.
KNOWN_BACKENDS: tuple[str, ...] = ("auto", "piper", "system")


# ---------------------------------------------------------------------------
# Public discovery API
# ---------------------------------------------------------------------------

#: Well-known locations where Home Assistant / standalone installs place
#: Piper voice models. Used both by :func:`list_voices` (discovery) and
#: :func:`_resolve_piper_voice` (the actual TTS path). Keeping a single
#: source of truth means adding a search path is a one-line change.
_PIPER_AUTO_PATHS: tuple[str, ...] = (
    "~/.local/share/piper/voices",
    "~/.config/piper/voices",
    "/usr/share/piper/voices",
    "/usr/local/share/piper/voices",
    "./voices",
)


def list_backends() -> list[str]:
    """Return the supported TTS backend names, in canonical order.

    Mirrors :data:`KNOWN_BACKENDS`. Public so ``paw-complete`` and tests
    can enumerate what's available without depending on the ``argparse``
    layer. The output of :func:`to_json` is derived from this list.
    """
    return list(KNOWN_BACKENDS)


def list_voices(*, search_paths: tuple[str, ...] | None = None) -> list[str]:
    """Return the absolute paths of every Piper ``*.onnx`` voice found.

    Searches the well-known Piper install locations (see
    :data:`_PIPER_AUTO_PATHS`) by default and returns each ``*.onnx`` as
    an absolute path, sorted. The list is empty if Piper is not
    installed anywhere reachable.

    The ``search_paths`` override exists for tests so the discovery path
    is exercisable without touching the real filesystem.
    """
    paths = search_paths if search_paths is not None else _PIPER_AUTO_PATHS
    seen: set[str] = set()
    out: list[str] = []
    for d in paths:
        expanded = os.path.expanduser(d)
        if not os.path.isdir(expanded):
            continue
        for cand in sorted(glob.glob(os.path.join(expanded, "*.onnx"))):
            abs_path = os.path.abspath(cand)
            if abs_path in seen:
                continue
            seen.add(abs_path)
            out.append(abs_path)
    return out


def to_json(kind: str) -> str:
    """Return the requested discovery data as a JSON string.

    ``kind`` is either ``"backends"`` or ``"voices"``. The output is a
    compact, single-line JSON object with one key, so it can be
    diffed, piped to ``jq``, or stored as a build artefact without
    further parsing.

    Raises :class:`ValueError` for unknown ``kind`` values.
    """
    if kind == "backends":
        payload = {"backends": list_backends()}
    elif kind == "voices":
        payload = {"voices": list_voices()}
    else:
        raise ValueError(
            f"unknown kind {kind!r}; expected 'backends' or 'voices'"
        )
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


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
        "--backend",
        default="auto",
        choices=KNOWN_BACKENDS,
        help=(
            "TTS backend to use. 'auto' picks the first available system engine "
            "(say / spd-say / espeak / SAPI). 'piper' uses local Piper TTS "
            "(requires --piper-voice). 'system' is the same as 'auto' but "
            "skips Piper even if installed."
        ),
    )
    parser.add_argument(
        "--piper-voice",
        default="auto",
        help=(
            "Path to a Piper .onnx voice model. Only used with --backend piper. "
            "Use 'auto' to try a few well-known locations (~/.local/share/piper/"
            "voices, /usr/share/piper/voices, ./voices). Default: auto."
        ),
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        dest="list_backends",
        help=(
            "Print the names of every supported TTS backend, one per line, "
            "and exit. Nothing is spoken. Useful for discovery and for "
            "shell completion."
        ),
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        dest="list_voices",
        help=(
            "Print the absolute path of every Piper .onnx voice model found "
            "in the well-known search locations, one per line, and exit. "
            "Nothing is spoken. Useful for discovery and for shell completion."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help=(
            "Combine with --list-backends or --list-voices to emit a JSON "
            "object instead of one-name-per-line text. The object has one "
            "key ('backends' or 'voices') whose value is the list. Nothing "
            "is spoken. Useful for jq / scripts / build artefacts."
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


#: Well-known locations where Home Assistant / standalone installs place
#: Piper voice models. Used when the user passes ``--piper-voice auto``.
#: (The full definition lives in the discovery section above; this
#: comment exists only so the chunk in :func:`_resolve_piper_voice`
#: still reads naturally.)


def _resolve_piper_voice(spec: str | None) -> str | None:
    """Return an absolute path to a Piper .onnx voice, or ``None``.

    ``spec`` is either a literal path, or ``"auto"`` / ``None``, in which
    case the well-known search locations are tried and the first
    ``*.onnx`` file found is returned.
    """
    if spec and spec != "auto":
        return spec
    import glob
    import os
    for d in _PIPER_AUTO_PATHS:
        expanded = os.path.expanduser(d)
        if not os.path.isdir(expanded):
            continue
        candidates = sorted(glob.glob(os.path.join(expanded, "*.onnx")))
        if candidates:
            return candidates[0]
    return None


def _piper_cmd(text: str, rate: float, volume: float, voice: str) -> list[str]:
    """Build a ``piper`` invocation.

    Piper reads text from stdin (one chunk per stdin write) and writes
    raw 16-bit PCM to stdout (``--output-raw``). Playback is done out of
    band by ``_piper_speak``, which pipes the PCM into the platform's
    audio output (``aplay`` / ``afplay`` / powershell ``SoundPlayer``).

    Rate mapping: Piper's ``--length_scale`` is inverse to speed.
    At 200 WPM the baseline is 1.0; faster (higher WPM) -> smaller value.
    Volume maps to ``--volume`` (Piper's 0..1 range maps cleanly to ours).
    """
    length_scale = max(0.5, min(2.0, 200.0 / max(80.0, rate)))
    return [
        "piper",
        "--model", voice,
        "--length_scale", f"{length_scale:.2f}",
        "--volume", f"{max(0.0, min(1.0, volume)):.2f}",
        "--output-raw",
    ]


def _piper_speak(text: str, rate: float, volume: float, voice: str) -> int:
    """Run Piper on ``text`` and stream the raw PCM to the audio backend.

    We need a two-stage pipe (piper -> aplay) that survives Piper's
    ~1.5s first-token latency. The implementation:
      1. Spawn Piper with stdout=PIPE
      2. Spawn aplay (or platform equivalent) with stdin=PIPE
      3. Copy Piper's stdout to aplay's stdin in a background thread
      4. Wait for both; return aplay's exit code (the audible one)
    """
    piper_cmd = _piper_cmd(text, rate, volume, voice)
    play_cmd = _pcm_playback_cmd()
    if play_cmd is None:
        print(
            "paw-read: piper selected but no PCM playback tool found "
            "(tried aplay / afplay / powershell)",
            file=sys.stderr,
        )
        return 1
    try:
        piper_proc = subprocess.Popen(
            piper_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    except FileNotFoundError:
        print(
            "paw-read: --backend piper but 'piper' binary not found on PATH",
            file=sys.stderr,
        )
        return 1
    try:
        play_proc = subprocess.Popen(
            play_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except FileNotFoundError:
        piper_proc.kill()
        print(
            f"paw-read: piper needs a PCM player but '{play_cmd[0]}' not found",
            file=sys.stderr,
        )
        return 1
    # We need to write text to Piper's stdin AND read its PCM out.
    # Threading is the simplest correct way without going full async.
    import threading
    piper_stdin = piper_proc.stdin
    if piper_stdin is not None:
        try:
            piper_stdin.write(text)
            piper_stdin.close()
        except (BrokenPipeError, OSError):
            pass
    pump_err: list[bytes] = []

    def _pump() -> None:
        try:
            if piper_proc.stdout is not None and play_proc.stdin is not None:
                while True:
                    chunk = piper_proc.stdout.read(4096)
                    if not chunk:
                        break
                    play_proc.stdin.write(chunk)
                play_proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
    t = threading.Thread(target=_pump, daemon=True)
    t.start()
    piper_rc = piper_proc.wait(timeout=120)
    t.join(timeout=5)
    play_rc = play_proc.wait(timeout=120)
    return play_rc if play_rc != 0 else piper_rc


def _pcm_playback_cmd() -> list[str] | None:
    """Return a command line that plays raw 16-bit little-endian 22050 Hz
    mono PCM from stdin, or ``None`` if no player is available.
    """
    system = platform.system()
    if system == "Linux":
        # Piper's --output-raw defaults to 22050 Hz mono s16le. Tell aplay.
        for tool, args in (
            (["aplay", "-q", "-f", "S16_LE", "-r", "22050", "-c", "1"], None),
        ):
            if shutil.which(tool[0]):
                return tool
    if system == "Darwin" and shutil.which("afplay"):
        # afplay reads PCM from stdin (`-`); the _pump thread in
        # _piper_speak streams Piper's stdout directly into it.
        return ["afplay", "-"]
    if system == "Windows" and shutil.which("powershell"):
        # PowerShell's $input reads PCM from the pipeline; the _pump
        # thread in _piper_speak streams Piper's stdout into it.
        return ["powershell", "-NoProfile", "-Command", "$input"]
    return None


def pick_backend(
    backend: str = "auto",
    piper_voice: str | None = "auto",
) -> Callable[..., int] | None:
    """Return a function ``(text, rate, volume) -> exit_code`` or ``None``.

    The ``backend`` argument selects between the local-Piper path and
    the OS-system-engine chain:

    * ``"auto"``  — try Piper first if installed + a voice is found,
      else fall back to the system chain.
    * ``"piper"`` — require Piper + a voice; return ``None`` if not.
    * ``"system"`` — skip Piper, use the system chain (say / spd-say /
      espeak / SAPI).
    """
    if backend in ("auto", "piper"):
        if shutil.which("piper"):
            voice_path = _resolve_piper_voice(piper_voice)
            if voice_path is None:
                if backend == "piper":
                    return None  # caller will report the missing voice
            else:
                # Bind voice into a closure so _speak can pass only (text, rate, volume).
                bound_voice = voice_path
                def _piper_bound(text: str, rate: float, volume: float) -> int:
                    return _piper_speak(text, rate, volume, bound_voice)
                return _piper_bound
    if backend == "piper":
        return None  # explicit Piper request but nothing usable
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


def _run_subprocess(builder: Callable[..., list[str]]) -> Callable[..., int]:
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


def _speak(text: str, rate: float, volume: float, backend: str = "auto",
          piper_voice: str | None = "auto") -> int:
    chosen = pick_backend(backend, piper_voice)
    if chosen is None:
        if backend == "piper":
            if not shutil.which("piper"):
                print(
                    "paw-read: --backend piper but 'piper' is not installed",
                    file=sys.stderr,
                )
            else:
                print(
                    "paw-read: --backend piper but no .onnx voice model was "
                    "found. Pass --piper-voice PATH or place a model in one of: "
                    + ", ".join(_PIPER_AUTO_PATHS),
                    file=sys.stderr,
                )
            return 2
        print(
            "paw-read: no TTS backend found "
            f"(system={platform.system()}; tried piper/say/spd-say/espeak/powershell)",
            file=sys.stderr,
        )
        return 1
    return chosen(text, rate, volume)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        # argparse / explicit validation raised SystemExit already
        return int(exc.code) if isinstance(exc.code, int) else 2

    # Discovery flags short-circuit before any source resolution or
    # TTS playback — they are mutually exclusive with reading, and they
    # do not need a text source or a working TTS engine.
    if args.as_json and not (args.list_backends or args.list_voices):
        print(
            "paw-read: --json requires --list-backends or --list-voices",
            file=sys.stderr,
        )
        return 2
    if args.list_backends:
        if args.as_json:
            print(to_json("backends"))
        else:
            for name in list_backends():
                print(name)
        return 0
    if args.list_voices:
        if args.as_json:
            print(to_json("voices"))
        else:
            for path in list_voices():
                print(path)
        return 0

    try:
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
        code = _speak(
            chunk, args.rate, args.volume,
            backend=args.backend, piper_voice=args.piper_voice,
        )
        if code != 0:
            last_code = code
    return last_code


if __name__ == "__main__":
    raise SystemExit(main())
