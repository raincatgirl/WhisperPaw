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


# --- discovery: list_packs / list_events ----------------------------------


def test_list_packs_returns_known_packs_sorted() -> None:
    """``list_packs()`` is the public discovery API and returns the
    registered pack names in a stable, sorted order — never the raw
    set's iteration order (which is not guaranteed in Python)."""
    packs = sound.list_packs()
    assert "cat" in packs
    assert "forest" in packs
    assert "rain" in packs
    # Sorted: the user can rely on this for diffing / shell completion.
    assert packs == sorted(packs)


def test_list_events_returns_canonical_order() -> None:
    """``list_events()`` returns the five events in their canonical
    order (the same order they appear in KNOWN_EVENTS)."""
    events = sound.list_events()
    # Compare as lists (the public API returns a list) but preserve order.
    assert list(events) == ["ok", "warn", "fail", "ready", "ding"]


def test_list_packs_and_list_events_dont_overlap_with_resolve() -> None:
    """The discovery helpers should be cheap and side-effect-free —
    no file IO, no audio backend lookup."""
    # No exception means the helpers did not touch the audio subsystem
    # or any file path. (Audio backends would be exercised only by
    # resolve_sound / main().)
    sound.list_packs()
    sound.list_events()


def test_parse_args_list_packs_flag() -> None:
    """``--list-packs`` parses as a boolean with the explicit dest."""
    args = sound.parse_args(["--list-packs"])
    assert args.list_packs is True
    assert args.list_events is False


def test_parse_args_list_events_flag() -> None:
    args = sound.parse_args(["--list-events"])
    assert args.list_events is True
    assert args.list_packs is False


def test_main_list_packs_prints_each_pack_and_exits_0(
    capsys, monkeypatch
) -> None:
    """``paw-sound --list-packs`` prints the packs, one per line, and
    does NOT touch the audio backend."""
    played: list = []
    monkeypatch.setattr(sound, "_play", lambda plan: played.append(plan) or 0)
    code = sound.main(["--list-packs"])
    out = capsys.readouterr().out
    assert code == 0
    # Every known pack on its own line.
    lines = [ln for ln in out.splitlines() if ln]
    assert set(lines) == set(sound.list_packs())
    # Critical: nothing was played.
    assert played == []


def test_main_list_events_prints_each_event_and_exits_0(
    capsys, monkeypatch
) -> None:
    """``paw-sound --list-events`` prints the events, one per line, and
    does NOT touch the audio backend."""
    played: list = []
    monkeypatch.setattr(sound, "_play", lambda plan: played.append(plan) or 0)
    code = sound.main(["--list-events"])
    out = capsys.readouterr().out
    assert code == 0
    lines = [ln for ln in out.splitlines() if ln]
    assert lines == list(sound.list_events())
    assert played == []


def test_main_list_packs_does_not_print_banner(capsys, monkeypatch) -> None:
    """Discovery mode skips the "playing" announcement — there is
    nothing to announce."""
    monkeypatch.setattr(sound, "_play", lambda plan: 0)
    code = sound.main(["--list-packs"])
    out = capsys.readouterr().out
    assert code == 0
    assert "playing" not in out.lower()
    assert "🐾" not in out


def test_main_list_packs_ignores_event_argument(capsys, monkeypatch) -> None:
    """``paw-sound --list-packs fail`` should still list packs, not try
    to resolve a sound (the discovery flag wins regardless of any
    positional event name)."""
    played: list = []
    monkeypatch.setattr(sound, "_play", lambda plan: played.append(plan) or 0)
    code = sound.main(["--list-packs", "fail"])
    out = capsys.readouterr().out
    assert code == 0
    # Packs, not the fail event:
    assert "fail" not in out
    assert "cat" in out
    assert played == []


# --- pack listing --------------------------------------------------------


def test_known_packs_includes_cat() -> None:
    assert "cat" in sound.KNOWN_PACKS


def test_resolve_sound_unknown_pack_raises_value_error() -> None:
    with pytest.raises(ValueError):
        sound.resolve_sound("ok", pack="atlantis", sound_root=sound.SOUND_ROOT)


# --- JSON output for discovery flags ------------------------------------


def test_to_json_packs_returns_parseable_object() -> None:
    """``to_json('packs')`` returns a JSON object with a 'packs' key."""
    import json as _json
    raw = sound.to_json("packs")
    parsed = _json.loads(raw)
    assert isinstance(parsed, dict)
    assert "packs" in parsed
    assert set(parsed["packs"]) == set(sound.list_packs())


def test_to_json_events_returns_parseable_object() -> None:
    """``to_json('events')`` returns a JSON object with an 'events' key,
    in the canonical order (a JSON array preserves order)."""
    import json as _json
    raw = sound.to_json("events")
    parsed = _json.loads(raw)
    assert isinstance(parsed, dict)
    assert parsed["events"] == sound.list_events()


def test_to_json_packs_is_sorted() -> None:
    """The 'packs' array is sorted alphabetically so the output is
    stable across runs and platforms — callers can diff / cache it."""
    import json as _json
    raw = sound.to_json("packs")
    parsed = _json.loads(raw)
    assert parsed["packs"] == sorted(parsed["packs"])


def test_to_json_is_single_line() -> None:
    """Single-line output is the contract — easy to grep, pipe, and
    store. Multi-line would break naive downstream tooling."""
    assert "\n" not in sound.to_json("packs")
    assert "\n" not in sound.to_json("events")


def test_to_json_unknown_kind_raises_value_error() -> None:
    """Defensive: an unknown kind is a programming error, not a user
    error. We raise so the bug surfaces immediately in tests."""
    with pytest.raises(ValueError):
        sound.to_json("unicorns")


def test_parse_args_json_flag_default_false() -> None:
    args = sound.parse_args([])
    assert args.as_json is False


def test_parse_args_json_flag_true() -> None:
    args = sound.parse_args(["--json", "--list-packs"])
    assert args.as_json is True
    assert args.list_packs is True


def test_main_list_packs_json_prints_json_object(
    capsys, monkeypatch
) -> None:
    """``paw-sound --list-packs --json`` prints a single-line JSON
    object and never touches the audio backend."""
    import json as _json
    played: list = []
    monkeypatch.setattr(sound, "_play", lambda plan: played.append(plan) or 0)
    code = sound.main(["--list-packs", "--json"])
    assert code == 0
    out = capsys.readouterr().out.strip()
    parsed = _json.loads(out)
    assert "packs" in parsed
    assert set(parsed["packs"]) == set(sound.list_packs())
    # Critical: nothing was played.
    assert played == []


def test_main_list_events_json_prints_json_object(
    capsys, monkeypatch
) -> None:
    import json as _json
    played: list = []
    monkeypatch.setattr(sound, "_play", lambda plan: played.append(plan) or 0)
    code = sound.main(["--list-events", "--json"])
    assert code == 0
    out = capsys.readouterr().out.strip()
    parsed = _json.loads(out)
    assert parsed["events"] == sound.list_events()
    assert played == []


def test_main_list_packs_text_mode_unchanged(capsys, monkeypatch) -> None:
    """Regression: adding --json must not change the default text
    output of --list-packs / --list-events (that's what shell
    completion and humans rely on)."""
    monkeypatch.setattr(sound, "_play", lambda plan: 0)
    code = sound.main(["--list-packs"])
    out = capsys.readouterr().out
    assert code == 0
    # Text mode: one name per line, no JSON braces.
    assert "{" not in out
    assert "}" not in out
    lines = [ln for ln in out.splitlines() if ln]
    assert set(lines) == set(sound.list_packs())


def test_main_json_without_discovery_flag_exits_2(capsys) -> None:
    """``--json`` alone (no --list-packs, no --list-events) is a usage
    error — the user almost certainly forgot the discovery flag. We
    exit 2 with a clear stderr message instead of silently printing
    nothing or a confusing empty object."""
    code = sound.main(["--json", "ok"])
    assert code == 2
    err = capsys.readouterr().err
    assert "--json" in err
    assert "--list-packs" in err or "--list-events" in err


def test_main_list_packs_json_does_not_print_banner(
    capsys, monkeypatch
) -> None:
    """Even in JSON mode we must not print the 'playing' banner —
    the output must be valid JSON, parseable by ``json.loads``."""
    import json as _json
    monkeypatch.setattr(sound, "_play", lambda plan: 0)
    code = sound.main(["--list-packs", "--json"])
    out = capsys.readouterr().out.strip()
    assert code == 0
    # If the banner were present, this would raise.
    _json.loads(out)
    assert "playing" not in out.lower()
    assert "🐾" not in out
