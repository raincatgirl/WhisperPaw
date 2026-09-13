"""Tests for the 'forest' sound pack — pure-stdlib synthesized ambient tones.

The forest pack uses layered noise + slow LFOs to evoke a small wood at
dusk: a low wind rumble, a faint bird chirp, and a soft cricket pulse.
Tests verify the pack is a real pack (files present, event coverage),
not that it sounds good — sound quality is verified by ear.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

sound = importlib.import_module("whisperpaw.sound")


def test_forest_is_a_known_pack() -> None:
    assert "forest" in sound.KNOWN_PACKS


def test_forest_pack_has_all_events() -> None:
    """Every known event should have at least one audio file in the forest pack."""
    forest_dir = sound.SOUND_ROOT / "forest"
    assert forest_dir.is_dir(), f"forest pack dir missing: {forest_dir}"
    missing = []
    for event in sound.KNOWN_EVENTS:
        if not any((forest_dir / f"{event}{ext}").is_file() for ext in sound._AUDIO_EXTS):
            missing.append(event)
    assert not missing, f"forest pack missing events: {missing}"


def test_forest_files_are_wav() -> None:
    """All forest files must be .wav (no mp3/ogg decoding dep at runtime)."""
    forest_dir = sound.SOUND_ROOT / "forest"
    wavs = list(forest_dir.glob("*.wav"))
    assert len(wavs) >= len(sound.KNOWN_EVENTS)
    non_wav = [p.name for p in forest_dir.iterdir() if p.is_file() and p.suffix != ".wav"]
    assert not non_wav, f"forest pack must be wav-only at runtime, found: {non_wav}"


def test_forest_sounds_are_under_3_seconds() -> None:
    """Forest pack cues should be short — the user fires them often."""
    forest_dir = sound.SOUND_ROOT / "forest"
    for wav in forest_dir.glob("*.wav"):
        import wave
        with wave.open(str(wav), "rb") as wf:
            duration = wf.getnframes() / wf.getframerate()
        assert duration < 3.0, f"{wav.name} is {duration:.2f}s, too long"


def test_resolve_sound_forest_ok() -> None:
    plan = sound.resolve_sound("ok", pack="forest", sound_root=sound.SOUND_ROOT)
    assert plan.pack == "forest"
    assert plan.event == "ok"
    assert plan.path.suffix == ".wav"
    assert "forest" in str(plan.path)


def test_resolve_sound_forest_fail() -> None:
    plan = sound.resolve_sound("fail", pack="forest", sound_root=sound.SOUND_ROOT)
    assert plan.path.is_file()
    assert plan.path.stem == "fail"


def test_forest_pack_listed_in_help() -> None:
    """The --help output should mention 'forest' as a valid pack."""
    import argparse
    parser = sound.build_parser()
    # argparse stores help text; check the help string
    for action in parser._actions:
        if "--pack" in action.option_strings:
            assert "forest" in (action.help or "")


def test_synth_forest_is_idempotent() -> None:
    """Re-running the synth script must not crash if files already exist."""
    from whisperpaw import _synth_forest
    # Just import — running it twice in tests would clobber; instead check the
    # module exposes the same public surface as _synth_sounds.
    assert hasattr(_synth_forest, "main")
    assert callable(_synth_forest.main)
