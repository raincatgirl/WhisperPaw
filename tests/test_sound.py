"""Tests for paw-sound — short audio cues for shell events.

These tests cover the small pure-logic API of paw-sound: argument parsing,
exit-code mapping, and the resolved play plan. They do NOT touch the
audio device, so they run anywhere without speakers.
"""
from __future__ import annotations

import dataclasses
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


# --- discovery: list_backends / BackendInfo / describe_backends ----------
#
# Mirrors the structure of the list_packs / list_events section
# above: low-level helper tests, parse-args coverage, and main()
# end-to-end tests for both text and JSON modes. The "active" and
# "available" fields are environment-sensitive, so all the tests
# inject ``which_fn`` / ``active_name`` via the new helper
# signatures — the test never depends on which audio binaries
# happen to be on this host's $PATH.


def test_list_backends_returns_one_entry_per_known_backend() -> None:
    """``list_backends()`` always returns exactly one entry per
    backend in :data:`sound.KNOWN_BACKEND_NAMES`, no more and no
    less — a caller can rely on the cardinality."""
    infos = sound.list_backends(
        which_fn=lambda name: None, active_name=None
    )
    assert len(infos) == len(sound.KNOWN_BACKEND_NAMES)
    names = {info.name for info in infos}
    assert names == set(sound.KNOWN_BACKEND_NAMES)


def test_list_backends_is_sorted_by_name() -> None:
    """The list is sorted alphabetically by ``name`` so the JSON
    output is diffable across runs and platforms."""
    infos = sound.list_backends(
        which_fn=lambda name: None, active_name=None
    )
    assert [info.name for info in infos] == sorted(sound.KNOWN_BACKEND_NAMES)


def test_list_backends_available_uses_which_fn() -> None:
    """``available`` is true iff the injected ``which_fn`` returns a
    non-None path for the backend's binary."""
    infos = sound.list_backends(
        which_fn=lambda name: f"/usr/bin/{name}" if name == "paplay" else None,
        active_name=None,
    )
    by_name = {info.name: info for info in infos}
    assert by_name["paplay"].available is True
    assert by_name["afplay"].available is False
    assert by_name["aplay"].available is False
    assert by_name["powershell"].available is False


def test_list_backends_active_is_set_exactly_once() -> None:
    """Exactly one backend has ``active == True`` when
    ``active_name`` is set; none do when ``active_name`` is
    ``None``. The contract is "one active backend on a healthy
    host, zero on a host with no audio installed"."""
    infos = sound.list_backends(
        which_fn=lambda name: None, active_name="afplay"
    )
    actives = [info for info in infos if info.active]
    assert len(actives) == 1
    assert actives[0].name == "afplay"
    # Other fields are unaffected by the active flag.
    assert all(info.volume_supported == (info.name in sound.BACKENDS_WITH_VOLUME)
               for info in infos)


def test_list_backends_active_none_means_no_active() -> None:
    """An explicit ``active_name=None`` means "no backend is
    active" — the sentinel-vs-None distinction matters here so
    a real ``current_backend_name()`` result is not overwritten."""
    infos = sound.list_backends(
        which_fn=lambda name: f"/usr/bin/{name}", active_name=None
    )
    assert all(info.active is False for info in infos)


def test_list_backends_active_default_uses_current_backend_name(
    monkeypatch,
) -> None:
    """Omitting ``active_name`` calls :func:`current_backend_name`
    on the live machine — and the active field matches its
    return value."""
    monkeypatch.setattr(
        sound, "current_backend_name", lambda: "paplay"
    )
    infos = sound.list_backends(
        which_fn=lambda name: f"/usr/bin/{name}"
    )
    actives = [info for info in infos if info.active]
    assert len(actives) == 1
    assert actives[0].name == "paplay"


def test_list_backends_volume_supported_matches_helper() -> None:
    """``volume_supported`` on a BackendInfo agrees with the
    :func:`volume_supported` module-level helper — single source
    of truth, no drift."""
    infos = sound.list_backends(
        which_fn=lambda name: None, active_name=None
    )
    for info in infos:
        assert info.volume_supported == sound.volume_supported(info.name)


def test_backend_info_is_frozen() -> None:
    """BackendInfo is frozen so a caller can pass it through
    layers without worrying about mutation."""
    info = sound.BackendInfo(
        name="afplay", available=True,
        volume_supported=True, active=False,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.name = "paplay"  # type: ignore[misc]


def test_describe_backends_active_only() -> None:
    """A backend that is active AND volume-capable renders as
    ``<name> (active, volume)`` — annotations in a fixed order
    so the output is diffable."""
    info = sound.BackendInfo(
        name="paplay", available=True,
        volume_supported=True, active=True,
    )
    assert sound.describe_backends([info]) == ["paplay (active, volume)"]


def test_describe_backends_unavailable_only() -> None:
    """``unavailable`` and ``volume`` are independent annotations,
    so a backend that is not on PATH but WOULD support volume
    once installed renders as ``<name> (volume, unavailable)``
    — both facts are reported so the user can decide whether
    to install the binary."""
    info = sound.BackendInfo(
        name="afplay", available=False,
        volume_supported=True, active=False,
    )
    assert sound.describe_backends([info]) == ["afplay (volume, unavailable)"]


def test_describe_backends_volume_only() -> None:
    """A backend that is available + volume-capable but NOT the
    active one renders as ``<name> (volume)`` — no ``active``
    annotation."""
    info = sound.BackendInfo(
        name="aplay", available=True,
        volume_supported=False, active=False,
    )
    assert sound.describe_backends([info]) == ["aplay"]


def test_describe_backends_preserves_input_order() -> None:
    """``describe_backends`` does not re-sort — a caller that
    pre-sorted the list sees that order in the output, so the
    helper composes cleanly with the JSON-mode sorted order."""
    info_a = sound.BackendInfo(
        name="afplay", available=False,
        volume_supported=True, active=False,
    )
    info_b = sound.BackendInfo(
        name="paplay", available=True,
        volume_supported=True, active=True,
    )
    # Deliberately reverse the alphabetical order.
    out = sound.describe_backends([info_b, info_a])
    assert out == [
        "paplay (active, volume)",
        "afplay (volume, unavailable)",
    ]


def test_describe_backends_empty_list_is_empty() -> None:
    """Defensive: an empty input list yields an empty output list,
    not a phantom blank line."""
    assert sound.describe_backends([]) == []


def test_to_json_backends_round_trip() -> None:
    """``to_json('backends')`` returns a parseable JSON object
    with an ``active`` field and a ``backends`` array."""
    import json as _json
    raw = sound.to_json("backends")
    parsed = _json.loads(raw)
    assert set(parsed) == {"active", "backends"}
    assert isinstance(parsed["backends"], list)
    for entry in parsed["backends"]:
        assert set(entry) == {
            "name", "available", "volume_supported", "active"
        }


def test_to_json_backends_active_is_null_when_none() -> None:
    """When no backend is active, the top-level ``active`` field
    is JSON null — not the string ``"None"``, not omitted, not
    an empty string."""
    import json as _json
    raw = sound.to_json("backends")
    parsed = _json.loads(raw)
    # We don't know whether this host has audio, so we can only
    # assert the *type* of the field, not its specific value.
    assert parsed["active"] is None or isinstance(parsed["active"], str)


def test_to_json_backends_is_single_line() -> None:
    """Single-line is the contract for all ``to_json`` outputs,
    including the new ``backends`` kind — easy to grep, pipe,
    and diff."""
    assert "\n" not in sound.to_json("backends")


def test_to_json_backends_keys_are_sorted() -> None:
    """The per-backend dict keys are emitted in sorted order, so
    the output is diffable across runs and platforms."""
    import json as _json
    raw = sound.to_json("backends")
    parsed = _json.loads(raw)
    if parsed["backends"]:
        first = parsed["backends"][0]
        assert list(first) == sorted(first)


def test_to_json_unknown_kind_message_lists_backends() -> None:
    """The error message for an unknown kind now mentions all
    three valid kinds — a regression guard for the help text
    that ships with the new ``backends`` kind."""
    with pytest.raises(ValueError) as excinfo:
        sound.to_json("unicorns")
    msg = str(excinfo.value)
    assert "packs" in msg
    assert "events" in msg
    assert "backends" in msg


def test_parse_args_list_backends_flag() -> None:
    """``--list-backends`` parses as a boolean with the explicit
    dest, same convention as the existing ``--list-packs`` /
    ``--list-events`` flags."""
    args = sound.parse_args(["--list-backends"])
    assert args.list_backends is True
    assert args.list_packs is False
    assert args.list_events is False


def test_parse_args_list_backends_default_false() -> None:
    args = sound.parse_args([])
    assert args.list_backends is False


def test_main_list_backends_prints_each_backend_and_exits_0(
    capsys, monkeypatch
) -> None:
    """``paw-sound --list-backends`` prints one backend per line
    and exits 0 — mirroring ``--list-packs`` / ``--list-events``."""
    played: list = []
    monkeypatch.setattr(sound, "_play", lambda plan: played.append(plan) or 0)
    code = sound.main(["--list-backends"])
    assert code == 0
    out_lines = [
        line for line in capsys.readouterr().out.splitlines() if line
    ]
    # One line per known backend.
    assert len(out_lines) == len(sound.KNOWN_BACKEND_NAMES)
    # Every known backend name appears somewhere in the output.
    for name in sound.KNOWN_BACKEND_NAMES:
        assert any(name in line for line in out_lines)
    # Nothing was played.
    assert played == []


def test_main_list_backends_does_not_print_banner(
    capsys, monkeypatch
) -> None:
    """The 'playing' banner must not appear in the discovery
    output — the output is the discovery report, period."""
    monkeypatch.setattr(sound, "_play", lambda plan: 0)
    code = sound.main(["--list-backends"])
    out = capsys.readouterr().out
    assert code == 0
    assert "playing" not in out.lower()
    assert "🐾" not in out


def test_main_list_backends_ignores_event_argument(
    capsys, monkeypatch
) -> None:
    """``paw-sound --list-backends fail`` should still list
    backends, not try to play ``fail``. Discovery is a
    short-circuit, so the event argument is silently dropped —
    same convention ``--list-packs`` already follows."""
    played: list = []
    monkeypatch.setattr(sound, "_play", lambda plan: played.append(plan) or 0)
    code = sound.main(["--list-backends", "fail"])
    assert code == 0
    out_lines = [
        line for line in capsys.readouterr().out.splitlines() if line
    ]
    assert len(out_lines) == len(sound.KNOWN_BACKEND_NAMES)
    assert played == []


def test_main_list_backends_json_prints_json_object(
    capsys, monkeypatch
) -> None:
    """``paw-sound --list-backends --json`` prints a single-line
    JSON object and never touches the audio backend."""
    import json as _json
    played: list = []
    monkeypatch.setattr(sound, "_play", lambda plan: played.append(plan) or 0)
    code = sound.main(["--list-backends", "--json"])
    assert code == 0
    out = capsys.readouterr().out.strip()
    parsed = _json.loads(out)
    assert set(parsed) == {"active", "backends"}
    assert len(parsed["backends"]) == len(sound.KNOWN_BACKEND_NAMES)
    # Critical: nothing was played.
    assert played == []


def test_main_list_backends_json_does_not_print_banner(
    capsys, monkeypatch
) -> None:
    """Even in JSON mode the 'playing' banner must not appear —
    the output must be valid JSON, parseable by ``json.loads``."""
    import json as _json
    monkeypatch.setattr(sound, "_play", lambda plan: 0)
    code = sound.main(["--list-backends", "--json"])
    out = capsys.readouterr().out.strip()
    assert code == 0
    # If the banner were present, this would raise.
    _json.loads(out)
    assert "playing" not in out.lower()
    assert "🐾" not in out


def test_main_json_without_discovery_flag_lists_all_three(
    capsys, monkeypatch
) -> None:
    """``--json`` alone (no --list-packs, no --list-events, no
    --list-backends) is a usage error and the stderr message
    now names all three valid discovery flags."""
    monkeypatch.setattr(sound, "_play", lambda plan: 0)
    code = sound.main(["--json", "ok"])
    assert code == 2
    err = capsys.readouterr().err
    assert "--json" in err
    assert "--list-packs" in err
    assert "--list-events" in err
    assert "--list-backends" in err


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


# --- per-backend volume wiring -----------------------------------------
#
# These tests assert the cmd-builder for each backend actually threads
# ``plan.volume`` through to the right flag. Without them the
# ``--volume`` flag would still parse, but every audio backend would
# silently ignore the user's value.


def _plan(event: str = "ok", pack: str = "cat", volume: float = 0.6):
    """Build a real :class:`PlayPlan` for the cmd-builder tests.

    The path is read from the on-disk pack so we don't need to
    stub a file; the cmd-builder only ever uses ``plan.path`` as
    a string, never opens it. ``PlayPlan`` is a frozen dataclass
    so we use :func:`dataclasses.replace` rather than the
    stdlib's ``_replace`` (which only exists on the unfrozen
    variant).
    """
    base = sound.resolve_sound(event, pack)
    return dataclasses.replace(base, volume=volume)


def test_afplay_cmd_passes_volume_flag() -> None:
    """afplay takes ``-v VALUE`` in the same [0.0, 1.0] scale the CLI uses."""
    plan = _plan(volume=0.5)
    cmd = sound._afplay_cmd(plan)
    assert cmd[0] == "afplay"
    # The flag + value are the two middle elements; the path is last.
    assert cmd[1:3] == ["-v", "0.500"]
    assert cmd[-1] == str(plan.path)


def test_afplay_cmd_volume_zero_is_passed_verbatim() -> None:
    """A volume of 0.0 is a real value (mute), not a default-fallback."""
    plan = _plan(volume=0.0)
    cmd = sound._afplay_cmd(plan)
    assert "-v" in cmd
    # The value immediately after -v is the volume.
    v_idx = cmd.index("-v")
    assert cmd[v_idx + 1] == "0.000"


def test_afplay_cmd_volume_one_is_passed_verbatim() -> None:
    """A volume of 1.0 (full) is also passed through unchanged."""
    plan = _plan(volume=1.0)
    cmd = sound._afplay_cmd(plan)
    v_idx = cmd.index("-v")
    assert cmd[v_idx + 1] == "1.000"


def test_paplay_cmd_scales_volume_to_pulse_range() -> None:
    """paplay takes ``--volume=`` as a 16-bit integer in [0, 65535].

    0.0 → 0 (mute), 1.0 → 65535 (full), 0.5 → 32768 (half).
    """
    # The paplay helper clamps the int conversion; 0.5 * 65535 = 32767.5
    # which rounds to 32768.
    assert sound._paplay_cmd(_plan(volume=0.0)) == [
        "paplay", "--volume=0", str(_plan().path),
    ]
    assert sound._paplay_cmd(_plan(volume=1.0)) == [
        "paplay", "--volume=65535", str(_plan().path),
    ]
    assert sound._paplay_cmd(_plan(volume=0.5)) == [
        "paplay", "--volume=32768", str(_plan().path),
    ]


def test_paplay_cmd_clamps_out_of_range_values() -> None:
    """paplay's volume is integer-clamped even if the input is a hair off.

    The CLI already rejects 0.0 > volume > 1.0 so this is a
    defensive belt — without it a downstream call could pass
    ``-0.01`` and Pulse would reject the negative volume.
    """
    assert sound._paplay_cmd(_plan(volume=-0.1)) == [
        "paplay", "--volume=0", str(_plan().path),
    ]
    assert sound._paplay_cmd(_plan(volume=1.5)) == [
        "paplay", "--volume=65535", str(_plan().path),
    ]


def test_aplay_cmd_does_not_emit_volume_flag() -> None:
    """ALSA's aplay has no per-stream volume flag; the cmd has no extra args."""
    plan = _plan(volume=0.42)
    assert sound._aplay_cmd(plan) == ["aplay", str(plan.path)]


def test_powershell_cmd_does_not_emit_volume() -> None:
    """The .NET SoundPlayer used by PowerShell is fixed-gain.

    The script only contains the SoundPlayer invocation, not any
    ``Volume`` property setter. We assert that the script is
    unchanged regardless of the requested volume.
    """
    plan_low = _plan(volume=0.1)
    plan_high = _plan(volume=0.9)
    assert sound._powershell_cmd(plan_low) == sound._powershell_cmd(plan_high)


# --- volume_supported() --------------------------------------------------


def test_volume_supported_true_for_afplay_and_paplay() -> None:
    """afplay (macOS) and paplay (PulseAudio) honour --volume."""
    assert sound.volume_supported("afplay") is True
    assert sound.volume_supported("paplay") is True


def test_volume_supported_false_for_aplay_and_powershell() -> None:
    """aplay (ALSA) and powershell SoundPlayer cannot honour --volume."""
    assert sound.volume_supported("aplay") is False
    assert sound.volume_supported("powershell") is False


def test_volume_supported_false_for_unknown_backend() -> None:
    """An unknown backend name returns False (not a KeyError)."""
    assert sound.volume_supported("nosuchplayer") is False
    assert sound.volume_supported("") is False


def test_known_backend_names_matches_the_set_we_pick() -> None:
    """The :data:`KNOWN_BACKEND_NAMES` set covers every backend the
    ``pick_backend`` ladder can return, and nothing else.

    This is the safety net for :func:`volume_supported` — if a
    new backend lands in ``pick_backend`` and isn't added here,
    ``current_backend_name()`` can return a name that
    ``volume_supported()`` doesn't know about. (We don't make
    that an error, but the test will at least flag the drift.)
    """
    from whisperpaw import sound as s
    # The intersection: every name pick_backend / current_backend_name
    # can return must be in KNOWN_BACKEND_NAMES.
    expected = {"afplay", "aplay", "paplay", "powershell"}
    assert s.KNOWN_BACKEND_NAMES == frozenset(expected)


# --- current_backend_name() ---------------------------------------------


def test_current_backend_name_agrees_with_pick_backend(
    monkeypatch,
) -> None:
    """``current_backend_name()`` and ``pick_backend()`` must agree on
    the active backend — they share the same ``shutil.which``
    ladder. We force both branches to be testable by stubbing
    ``shutil.which`` to claim a single binary exists.
    """
    import shutil as _shutil

    def fake_which(name: str) -> str | None:
        return f"/usr/bin/{name}" if name == "afplay" else None

    monkeypatch.setattr(_shutil, "which", fake_which)
    monkeypatch.setattr(sound.platform, "system", lambda: "Darwin")
    assert sound.current_backend_name() == "afplay"
    assert sound.pick_backend() is not None
    # And the actually-picked function should be the afplay builder.
    plan = _plan()
    assert sound._afplay_cmd(plan)[0] == "afplay"


def test_current_backend_name_returns_none_when_no_backend(
    monkeypatch,
) -> None:
    """If no audio binary is on PATH, both helpers return ``None``."""
    import shutil as _shutil
    monkeypatch.setattr(_shutil, "which", lambda name: None)
    monkeypatch.setattr(sound.platform, "system", lambda: "Linux")
    assert sound.current_backend_name() is None
    assert sound.pick_backend() is None


# --- end-to-end: --volume flows from CLI to backend ---------------------


def test_main_volume_flag_reaches_afplay_backend(
    capsys, monkeypatch
) -> None:
    """The user's ``--volume`` must reach the audio backend.

    We monkey-patch ``pick_backend`` to return a recorder that
    captures the :class:`PlayPlan`, then assert the volume
    field made it through ``main()`` unchanged.
    """
    captured: list = []

    def fake_pick():
        def _play(plan):
            captured.append(plan)
            return 0
        return _play

    monkeypatch.setattr(sound, "pick_backend", fake_pick)
    code = sound.main(["--volume", "0.25", "ok"])
    assert code == 0
    assert len(captured) == 1
    assert captured[0].volume == pytest.approx(0.25)


def test_main_default_volume_reaches_backend(capsys, monkeypatch) -> None:
    """Without ``--volume`` the default (0.6) is what reaches the backend."""
    captured: list = []

    def fake_pick():
        def _play(plan):
            captured.append(plan)
            return 0
        return _play

    monkeypatch.setattr(sound, "pick_backend", fake_pick)
    code = sound.main(["ok"])
    assert code == 0
    assert captured[0].volume == pytest.approx(0.6)
