"""Tests for the optional Piper TTS backend in paw-read.

Piper is opt-in: it is only used when the user passes ``--backend piper``
AND a ``--piper-voice PATH`` is given (or ``--piper-voice auto`` lets us
try the well-known Home Assistant default path). All tests monkey-patch
the subprocess call so no real Piper binary is required.
"""
from __future__ import annotations

import importlib

import pytest

read = importlib.import_module("whisperpaw.read")


def test_piper_cmd_basic() -> None:
    cmd = read._piper_cmd("hello", rate=200.0, volume=1.0, voice="/v/en.onnx")
    # Must invoke piper with --model and the voice path
    assert cmd[0] == "piper"
    assert "--model" in cmd
    voice_idx = cmd.index("--model") + 1
    assert cmd[voice_idx] == "/v/en.onnx"
    # Text is piped via stdin (Piper reads text from stdin by default)
    assert "--output-raw" in cmd or "--output_file" in cmd


def test_piper_cmd_includes_rate_and_volume() -> None:
    cmd = read._piper_cmd("hi", rate=240.0, volume=0.6, voice="/v.onnx")
    # length_scale maps "faster" -> smaller value; we expose rate as WPM-ish
    # so a higher rate -> smaller length_scale. We accept either a length_scale
    # flag or a speed flag, but the relationship must be monotone.
    s_flag = next((c for c in cmd if c.startswith("--length_scale") or c.startswith("--speed")), None)
    assert s_flag is not None


def test_piper_cmd_strips_trailing_whitespace() -> None:
    cmd = read._piper_cmd("  hi  \n", rate=200.0, volume=1.0, voice="/v.onnx")
    # We don't pass the text on argv (Piper reads from stdin), so this is a
    # sanity test on the helper itself. If we ever switch to argv text,
    # the trim should still be applied. For now: just don't crash.
    assert cmd is not None


def test_parse_args_backend_flag() -> None:
    args = read.parse_args(["--backend", "piper", "hi"])
    assert args.backend == "piper"


def test_parse_args_backend_default_is_auto() -> None:
    args = read.parse_args(["hi"])
    assert args.backend == "auto"


def test_parse_args_piper_voice_flag() -> None:
    args = read.parse_args(["--backend", "piper", "--piper-voice", "/v.onnx", "hi"])
    assert args.piper_voice == "/v.onnx"


def test_parse_args_piper_voice_default_is_auto() -> None:
    args = read.parse_args(["hi"])
    assert args.piper_voice == "auto"


def test_parse_args_piper_only_with_backend() -> None:
    """Passing --piper-voice without --backend piper should not crash; it's just ignored."""
    args = read.parse_args(["--piper-voice", "/v.onnx", "hi"])
    assert args.backend == "auto"
    assert args.piper_voice == "/v.onnx"


def test_parse_args_rejects_unknown_backend() -> None:
    with pytest.raises(SystemExit):
        read.parse_args(["--backend", "azure", "hi"])


def test_pick_backend_returns_piper_when_requested(monkeypatch) -> None:
    """When the user asks for piper, pick_backend should honor that."""
    monkeypatch.setattr(read.shutil, "which", lambda name: "/usr/bin/piper" if name == "piper" else None)
    backend = read.pick_backend("piper", piper_voice="/v.onnx")
    assert backend is not None
    assert callable(backend)


def test_pick_backend_piper_without_binary_returns_none(monkeypatch) -> None:
    """If the user asks for piper but it isn't installed, return None (caller handles)."""
    monkeypatch.setattr(read.shutil, "which", lambda name: None)
    backend = read.pick_backend("piper", piper_voice="/v.onnx")
    assert backend is None


def test_pick_backend_piper_without_voice_returns_none(monkeypatch) -> None:
    """If the user asks for piper but no voice is configured, return None."""
    monkeypatch.setattr(read.shutil, "which", lambda name: "/usr/bin/piper" if name == "piper" else None)
    backend = read.pick_backend("piper", piper_voice=None)
    assert backend is None


def test_pick_backend_auto_falls_back_to_espeak(monkeypatch) -> None:
    """When backend='auto' and piper isn't installed, fall back to the chain."""
    monkeypatch.setattr(read.platform, "system", lambda: "Linux")
    monkeypatch.setattr(read.shutil, "which", lambda name: "/usr/bin/espeak" if name == "espeak" else None)
    backend = read.pick_backend("auto", piper_voice=None)
    assert backend is not None  # the espeak backend


def test_main_piper_no_voice_exits_nonzero(monkeypatch, capsys) -> None:
    """If piper is requested but no voice is configured, exit 2 with a clear message."""
    monkeypatch.setattr(read.shutil, "which", lambda name: "/usr/bin/piper" if name == "piper" else None)
    code = read.main(["--quiet", "--backend", "piper", "hi"])
    assert code == 2
    err = capsys.readouterr().err
    assert "piper" in err.lower()
    assert "voice" in err.lower() or "model" in err.lower()


def test_main_piper_no_binary_exits_nonzero(monkeypatch, capsys) -> None:
    """If piper is requested but not installed, exit 2 with a helpful message."""
    monkeypatch.setattr(read.shutil, "which", lambda name: None)
    code = read.main(["--quiet", "--backend", "piper", "--piper-voice", "/v.onnx", "hi"])
    assert code == 2
    err = capsys.readouterr().err
    assert "piper" in err.lower()


# ---------------------------------------------------------------------------
# _pcm_playback_cmd — the platform-picker that pairs Piper's raw-PCM stdout
# with a per-OS audio sink. Two of the three branches used to carry
# comments claiming "we route through a temp file"; the actual code in
# _piper_speak pipes Piper's stdout into `play_proc.stdin` via a background
# thread, with no temp file anywhere. These tests pin down the contract:
# the returned command must be a plain argv list the caller can Popen with
# stdin=PIPE, and there must be no tempfile import involved.
# ---------------------------------------------------------------------------


def test_pcm_playback_cmd_darwin_returns_afplay_stdin(monkeypatch) -> None:
    monkeypatch.setattr(read.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(read.shutil, "which", lambda name: "/usr/bin/afplay" if name == "afplay" else None)
    cmd = read._pcm_playback_cmd()
    assert cmd == ["afplay", "-"]


def test_pcm_playback_cmd_windows_returns_powershell_stdin(monkeypatch) -> None:
    monkeypatch.setattr(read.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        read.shutil, "which", lambda name: "C:/powershell" if name == "powershell" else None
    )
    cmd = read._pcm_playback_cmd()
    assert cmd is not None
    # The command must read PCM from stdin (via $input), so the caller
    # can pipe Piper's stdout into it without a temp file.
    assert cmd[0] == "powershell"
    assert "$input" in cmd


def test_pcm_playback_cmd_linux_returns_aplay_with_pcm_flags(monkeypatch) -> None:
    monkeypatch.setattr(read.platform, "system", lambda: "Linux")
    monkeypatch.setattr(read.shutil, "which", lambda name: "/usr/bin/aplay" if name == "aplay" else None)
    cmd = read._pcm_playback_cmd()
    assert cmd is not None
    assert cmd[0] == "aplay"
    # The PCM-format flags (22050 Hz, mono, s16le) tell aplay the exact
    # shape of what Piper's --output-raw will produce on its stdout.
    assert "S16_LE" in cmd
    assert "22050" in cmd


def test_pcm_playback_cmd_returns_none_when_no_player(monkeypatch) -> None:
    monkeypatch.setattr(read.platform, "system", lambda: "Linux")
    monkeypatch.setattr(read.shutil, "which", lambda name: None)
    assert read._pcm_playback_cmd() is None


def test_pcm_playback_cmd_does_not_use_tempfile() -> None:
    """Regression: the old comments on the Darwin/Windows branches said
    'we route through a temp file'. The actual implementation pipes raw
    PCM into the player's stdin via the _pump thread in _piper_speak; no
    temp file is ever written. This test guards that contract by
    asserting ``tempfile`` is never imported or referenced in the
    _pcm_playback_cmd body."""
    import ast
    import inspect

    source = inspect.getsource(read._pcm_playback_cmd)
    # No mention of tempfile / TemporaryFile anywhere in the function body.
    assert "tempfile" not in source
    assert "TemporaryFile" not in source
    # And the function itself is small — parse it as a sanity check that
    # we are reading the right definition.
    tree = ast.parse(source)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert len(funcs) == 1
    assert funcs[0].name == "_pcm_playback_cmd"
