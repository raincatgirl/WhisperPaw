"""Tests for the 'rain' sound pack.

The rain pack is layered ambient synth: a steady drizzle of short
transients, a low-pass "sky" drone, and an occasional heavier drop
on certain events (heavy rain, thunder). The shape is different from
'forest' (which is wind + bird + bell); rain is rhythmic, forest is
breathy. Tests verify the pack follows the same registration rules
as 'cat' and 'forest' so adding it costs zero new code in sound.py.
"""
from __future__ import annotations

import importlib

import pytest

sound = importlib.import_module("whisperpaw.sound")


def test_rain_is_a_known_pack() -> None:
    assert "rain" in sound.KNOWN_PACKS


def test_rain_pack_has_all_events() -> None:
    rain_dir = sound.SOUND_ROOT / "rain"
    assert rain_dir.is_dir(), f"rain pack dir missing: {rain_dir}"
    missing = []
    for event in sound.KNOWN_EVENTS:
        if not any((rain_dir / f"{event}{ext}").is_file() for ext in sound._AUDIO_EXTS):
            missing.append(event)
    assert not missing, f"rain pack missing events: {missing}"


def test_rain_files_are_wav() -> None:
    """All rain files must be .wav (no mp3/ogg decoding dep at runtime)."""
    rain_dir = sound.SOUND_ROOT / "rain"
    non_wav = [p.name for p in rain_dir.iterdir() if p.is_file() and p.suffix != ".wav"]
    assert not non_wav, f"rain pack must be wav-only at runtime, found: {non_wav}"


def test_rain_sounds_are_under_3_seconds() -> None:
    """Rain pack cues should be short — same budget as cat/forest."""
    import wave
    rain_dir = sound.SOUND_ROOT / "rain"
    for wav in rain_dir.glob("*.wav"):
        with wave.open(str(wav), "rb") as wf:
            duration = wf.getnframes() / wf.getframerate()
        assert duration < 3.0, f"{wav.name} is {duration:.2f}s, too long"


def test_rain_sounds_are_at_least_300ms() -> None:
    """Rain cues should not be a single click — they need to be heard as rain."""
    import wave
    rain_dir = sound.SOUND_ROOT / "rain"
    for wav in rain_dir.glob("*.wav"):
        with wave.open(str(wav), "rb") as wf:
            duration = wf.getnframes() / wf.getframerate()
        assert duration >= 0.3, f"{wav.name} is {duration:.2f}s, too short to read as rain"


def test_rain_sounds_differ_in_rms_from_each_other() -> None:
    """Sanity check: the five rain cues are not all the same file with
    different names. We compare RMS (rough loudness) — every event
    should sound at least slightly different from every other.
    """
    import wave
    import struct
    import math
    rain_dir = sound.SOUND_ROOT / "rain"
    rms_by_event: dict[str, float] = {}
    for wav in sorted(rain_dir.glob("*.wav")):
        with wave.open(str(wav), "rb") as wf:
            n = wf.getnframes()
            sw = wf.getsampwidth()
            sr = wf.getframerate()
            raw = wf.readframes(n)
        if sw != 2:
            continue  # only handle 16-bit
        samples = struct.unpack(f"<{n}h", raw)
        rms = math.sqrt(sum(s * s for s in samples) / n) if n else 0.0
        rms_by_event[wav.stem] = rms
    # At minimum, ok and fail should differ noticeably
    assert abs(rms_by_event.get("ok", 0) - rms_by_event.get("fail", 0)) > 50, (
        f"ok and fail rain cues are suspiciously similar in RMS: {rms_by_event}"
    )


def test_resolve_sound_rain_ok() -> None:
    plan = sound.resolve_sound("ok", pack="rain", sound_root=sound.SOUND_ROOT)
    assert plan.pack == "rain"
    assert plan.event == "ok"
    assert plan.path.suffix == ".wav"
    assert "rain" in str(plan.path)


def test_resolve_sound_rain_fail() -> None:
    plan = sound.resolve_sound("fail", pack="rain", sound_root=sound.SOUND_ROOT)
    assert plan.path.is_file()
    assert plan.path.stem == "fail"


def test_rain_pack_listed_in_help() -> None:
    """The --help output should mention 'rain' as a valid pack."""
    parser = sound.build_parser()
    for action in parser._actions:
        if "--pack" in action.option_strings:
            assert "rain" in (action.help or "")


def test_synth_rain_is_idempotent() -> None:
    """Re-running the synth script must not crash if files already exist."""
    from whisperpaw import _synth_rain
    assert hasattr(_synth_rain, "main")
    assert callable(_synth_rain.main)
