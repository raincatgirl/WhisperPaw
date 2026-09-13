"""Tests for paw-sound — short audio cues for shell events.

These tests cover the small pure-logic API of paw-sound: argument parsing,
exit-code mapping, and the resolved play plan. They do NOT touch the
audio device, so they run anywhere without speakers.
"""
from __future__ import annotations

import importlib

import pytest

sound = importlib.import_module("whisperpaw.sound")


# --- arg parsing ---------------------------------------------------------


def test_parse_args_defaults_to_ok_when_no_args() -> None:
    args = sound.parse_args([])
    assert args.event == "ok"
    assert args.pack == "cat"
    assert args.volume == pytest.approx(0.6)
    assert args.quiet is False


def test_parse_args_explicit_event() -> None:
    args = sound.parse_args(["fail"])
    assert args.event == "fail"


def test_parse_args_all_known_events() -> None:
    for ev in ("ok", "warn", "fail", "ready", "ding"):
        args = sound.parse_args([ev])
        assert args.event == ev


def test_parse_args_rejects_unknown_event() -> None:
    with pytest.raises(SystemExit):
        sound.parse_args(["meow"])  # not a known event


def test_parse_args_volume_flag() -> None:
    args = sound.parse_args(["--volume", "0.25", "ok"])
    assert args.volume == pytest.approx(0.25)


def test_parse_args_pack_flag() -> None:
    args = sound.parse_args(["--pack", "forest", "ok"])
    assert args.pack == "forest"


def test_parse_args_quiet_flag() -> None:
    args = sound.parse_args(["--quiet", "fail"])
    assert args.quiet is True


# --- event -> sound file mapping ----------------------------------------


def test_resolve_sound_returns_existing_file_for_known_event() -> None:
    # Every known event must resolve to a file inside the sound pack.
    plan = sound.resolve_sound("ok", pack="cat", sound_root=sound.SOUND_ROOT)
    assert plan.event == "ok"
    assert plan.pack == "cat"
    assert plan.path.is_file(), f"missing sound file: {plan.path}"
    assert plan.path.suffix in {".wav", ".mp3", ".ogg", ".flac"}


def test_resolve_sound_unknown_event_raises_value_error() -> None:
    with pytest.raises(ValueError):
        sound.resolve_sound("zzz", pack="cat", sound_root=sound.SOUND_ROOT)


# --- main() exit codes ---------------------------------------------------


def test_main_unknown_event_exits_nonzero(capsys) -> None:
    code = sound.main(["--quiet", "totally-bogus-event"])
    assert code == 2
    err = capsys.readouterr().err
    assert "unknown event" in err.lower()


def test_main_quiet_does_not_print_banner(capsys, monkeypatch) -> None:
    # Quiet mode + dry run (no audio backend) should not print "playing".
    monkeypatch.setattr(sound, "_play", lambda plan: 0)
    code = sound.main(["--quiet", "ok"])
    assert code == 0
    out = capsys.readouterr().out
    assert out == ""


def test_main_default_prints_announcement(capsys, monkeypatch) -> None:
    monkeypatch.setattr(sound, "_play", lambda plan: 0)
    code = sound.main(["ok"])
    assert code == 0
    out = capsys.readouterr().out
    assert "ok" in out.lower() or "🐾" in out


def test_main_propagates_audio_backend_exit_code(monkeypatch) -> None:
    # If the audio backend fails, main() should reflect that (1 = user env issue).
    monkeypatch.setattr(sound, "_play", lambda plan: 1)
    code = sound.main(["--quiet", "ok"])
    assert code == 1


# --- pack listing --------------------------------------------------------


def test_known_packs_includes_cat() -> None:
    assert "cat" in sound.KNOWN_PACKS


def test_resolve_sound_unknown_pack_raises_value_error() -> None:
    with pytest.raises(ValueError):
        sound.resolve_sound("ok", pack="atlantis", sound_root=sound.SOUND_ROOT)
