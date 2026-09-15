"""Tests for paw-read — read text aloud via TTS.

Tests cover the pure logic: argument parsing, source resolution (stdin
/ file / arg / clipboard), chunking of long text, and exit-code
semantics. Audio playback is monkey-patched so no real TTS engine is
needed in CI.
"""
from __future__ import annotations

import importlib
import os

import pytest

read = importlib.import_module("whisperpaw.read")


# --- arg parsing ---------------------------------------------------------


def test_parse_args_defaults_to_stdin_quiet() -> None:
    args = read.parse_args([])
    assert args.text is None
    assert args.file is None
    assert args.use_clipboard is False
    assert args.rate == pytest.approx(200.0)
    assert args.quiet is False


def test_parse_args_text_positional() -> None:
    args = read.parse_args(["hello world"])
    assert args.text == "hello world"


def test_parse_args_multiple_text_words_joined() -> None:
    args = read.parse_args(["hello", "cruel", "world"])
    assert args.text == "hello cruel world"


def test_parse_args_file_flag() -> None:
    args = read.parse_args(["--file", "/tmp/x.txt"])
    assert args.file == "/tmp/x.txt"
    assert args.text is None


def test_parse_args_clipboard_flag() -> None:
    args = read.parse_args(["--clipboard"])
    assert args.use_clipboard is True


def test_parse_args_rate_flag() -> None:
    args = read.parse_args(["--rate", "320", "hi"])
    assert args.rate == pytest.approx(320.0)


def test_parse_args_volume_flag() -> None:
    args = read.parse_args(["--volume", "0.4", "hi"])
    assert args.volume == pytest.approx(0.4)


def test_parse_args_quiet_flag() -> None:
    args = read.parse_args(["--quiet", "hi"])
    assert args.quiet is True


def test_parse_args_rejects_rate_out_of_range() -> None:
    with pytest.raises(SystemExit):
        read.parse_args(["--rate", "5"])  # too slow


def test_parse_args_rejects_rate_too_high() -> None:
    with pytest.raises(SystemExit):
        read.parse_args(["--rate", "9999"])


def test_parse_args_rejects_volume_out_of_range() -> None:
    with pytest.raises(SystemExit):
        read.parse_args(["--volume", "1.5"])


# --- source resolution --------------------------------------------------


def test_resolve_text_prefers_explicit_text() -> None:
    out = read.resolve_source(
        text="hello",
        file=None,
        use_clipboard=False,
        stdin_text="from-stdin",
    )
    assert out == "hello"


def test_resolve_text_from_file(tmp_path) -> None:
    p = tmp_path / "hi.txt"
    p.write_text("file content", encoding="utf-8")
    out = read.resolve_source(
        text=None,
        file=str(p),
        use_clipboard=False,
        stdin_text="from-stdin",
    )
    assert out == "file content"


def test_resolve_text_from_stdin_when_empty_args() -> None:
    out = read.resolve_source(
        text=None,
        file=None,
        use_clipboard=False,
        stdin_text="hello stdin",
    )
    assert out == "hello stdin"


def test_resolve_text_from_stdin_strips_trailing_newline() -> None:
    out = read.resolve_source(
        text=None, file=None, use_clipboard=False, stdin_text="hi\n"
    )
    assert out == "hi"


def test_resolve_text_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        read.resolve_source(
            text=None,
            file="/nonexistent/path.txt",
            use_clipboard=False,
            stdin_text="x",
        )


def test_resolve_text_clipboard_backend_none_raises(monkeypatch) -> None:
    monkeypatch.setattr(read, "read_clipboard", lambda: None)
    with pytest.raises(RuntimeError):
        read.resolve_source(
            text=None, file=None, use_clipboard=True, stdin_text="x"
        )


def test_resolve_text_clipboard_backend_returns_value(monkeypatch) -> None:
    monkeypatch.setattr(read, "read_clipboard", lambda: "from clipboard")
    out = read.resolve_source(
        text=None, file=None, use_clipboard=True, stdin_text="ignored"
    )
    assert out == "from clipboard"


# --- chunking -----------------------------------------------------------


def test_chunk_short_text_unchanged() -> None:
    assert read.chunk_text("short", max_chars=100) == ["short"]


def test_chunk_splits_on_sentence_boundary() -> None:
    text = "First sentence. Second sentence. Third."
    chunks = read.chunk_text(text, max_chars=20)
    # Should keep sentences together when possible
    assert " ".join(chunks).replace(" ", "") == text.replace(" ", "")
    assert all(len(c) <= 25 for c in chunks), chunks  # some slack


def test_chunk_huge_single_word_is_split_at_boundary() -> None:
    """A single word longer than max_chars is split at the boundary, not lost."""
    word = "a" * 500
    chunks = read.chunk_text(word, max_chars=100)
    # Reassembling must give back the original — no characters dropped.
    assert "".join(chunks) == word
    # And the first chunk is no longer than max_chars (it's a pure cut).
    assert len(chunks[0]) == 100
    assert len(chunks) > 1


def test_chunk_word_just_under_max_kept_intact() -> None:
    """A single word at exactly the limit stays as one chunk."""
    word = "a" * 100
    chunks = read.chunk_text(word, max_chars=100)
    assert chunks == [word]


def test_chunk_empty_text() -> None:
    assert read.chunk_text("", max_chars=100) == []


def test_chunk_whitespace_only() -> None:
    assert read.chunk_text("   \n\n  ", max_chars=100) == []


# --- main() exit codes -------------------------------------------------


def test_main_with_text_succeeds(monkeypatch, capsys) -> None:
    spoken = []
    monkeypatch.setattr(read, "_speak", lambda text, rate, volume, **_: spoken.append(text) or 0)
    code = read.main(["--quiet", "hello world"])
    assert code == 0
    assert spoken == ["hello world"]
    out = capsys.readouterr().out
    assert out == ""  # --quiet


def test_main_default_prints_announcement(monkeypatch, capsys) -> None:
    monkeypatch.setattr(read, "_speak", lambda text, rate, volume, **_: 0)
    code = read.main(["hi"])
    assert code == 0
    out = capsys.readouterr().out
    assert "hi" in out.lower() or "🐾" in out


def test_main_propagates_speak_exit_code(monkeypatch) -> None:
    monkeypatch.setattr(read, "_speak", lambda text, rate, volume, **_: 1)
    code = read.main(["--quiet", "x"])
    assert code == 1


def test_main_empty_text_exits_with_usage_code(monkeypatch, capsys) -> None:
    code = read.main(["--quiet"])  # no text, no stdin (tty), no file
    assert code == 2
    err = capsys.readouterr().err
    assert "no text" in err.lower() or "nothing" in err.lower()


def test_main_long_text_chunked(monkeypatch) -> None:
    spoken: list[str] = []
    monkeypatch.setattr(read, "_speak", lambda text, rate, volume, **_: spoken.append(text) or 0)
    long = ". ".join(["sentence"] * 30) + "."  # ~300 chars
    code = read.main(["--quiet", "--max-chars", "50", long])
    assert code == 0
    assert len(spoken) >= 2
    assert " ".join(spoken).replace(" ", "") == long.replace(" ", "")


# --- backend selection -------------------------------------------------


def test_pick_backend_returns_callable_for_darwin(monkeypatch) -> None:
    monkeypatch.setattr(read.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(read.shutil, "which", lambda name: "/usr/bin/say" if name == "say" else None)
    backend = read.pick_backend("system", None)
    assert backend is not None
    assert callable(backend)


def test_pick_backend_returns_callable_for_linux_with_espeak(monkeypatch) -> None:
    monkeypatch.setattr(read.platform, "system", lambda: "Linux")
    monkeypatch.setattr(read.shutil, "which", lambda name: "/usr/bin/espeak" if name == "espeak" else None)
    backend = read.pick_backend("system", None)
    assert backend is not None


def test_pick_backend_returns_callable_for_linux_with_spd_say(monkeypatch) -> None:
    monkeypatch.setattr(read.platform, "system", lambda: "Linux")
    monkeypatch.setattr(read.shutil, "which", lambda name: "/usr/bin/spd-say" if name == "spd-say" else None)
    backend = read.pick_backend("system", None)
    assert backend is not None


def test_pick_backend_returns_callable_for_windows(monkeypatch) -> None:
    monkeypatch.setattr(read.platform, "system", lambda: "Windows")
    monkeypatch.setattr(read.shutil, "which", lambda name: "powershell" if name == "powershell" else None)
    backend = read.pick_backend("system", None)
    assert backend is not None


def test_pick_backend_returns_none_when_nothing_available(monkeypatch) -> None:
    monkeypatch.setattr(read.platform, "system", lambda: "Linux")
    monkeypatch.setattr(read.shutil, "which", lambda name: None)
    assert read.pick_backend("system", None) is None


# --- discovery: list_backends / list_voices / to_json --------------------


def test_list_backends_canonical_order() -> None:
    """list_backends() returns the three known backends in canonical order."""
    assert read.list_backends() == ["auto", "piper", "system"]


def test_list_backends_matches_known_constant() -> None:
    """The runtime list and KNOWN_BACKENDS must stay in lock-step."""
    assert tuple(read.list_backends()) == read.KNOWN_BACKENDS


def test_list_voices_returns_absolute_paths(tmp_path) -> None:
    """list_voices() finds .onnx files under a custom search dir, as absolute paths."""
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "amy.onnx").write_bytes(b"")
    (voices / "bob.onnx").write_bytes(b"")
    # Add a non-onnx file that must be ignored.
    (voices / "notes.txt").write_text("ignore me", encoding="utf-8")
    found = read.list_voices(search_paths=(str(voices),))
    assert len(found) == 2
    for path in found:
        assert os.path.isabs(path)
        assert path.endswith(".onnx")
    # Sorted alphabetically.
    assert found == sorted(found)


def test_list_voices_empty_when_no_search_dirs_match(tmp_path) -> None:
    """A search path that doesn't exist yields an empty list (no error)."""
    assert read.list_voices(search_paths=(str(tmp_path / "missing"),)) == []


def test_list_voices_dedupes_when_paths_overlap(tmp_path) -> None:
    """The same voice file referenced by two paths shows up once."""
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "amy.onnx").write_bytes(b"")
    # Pass the same path twice under different spellings (one absolute,
    # one not) — the dedup must collapse them into a single entry.
    found = read.list_voices(
        search_paths=(str(voices), str(voices.resolve()))
    )
    assert len(found) == 1


def test_list_voices_ignores_non_onnx_files(tmp_path) -> None:
    """Only files ending in .onnx are returned; .json / .txt are skipped."""
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "amy.onnx").write_bytes(b"")
    (voices / "amy.onnx.json").write_text("{}", encoding="utf-8")
    (voices / "config.txt").write_text("nope", encoding="utf-8")
    found = read.list_voices(search_paths=(str(voices),))
    assert found == [os.path.abspath(str(voices / "amy.onnx"))]


def test_to_json_backends_round_trip() -> None:
    """to_json('backends') produces a parseable JSON object with the right key."""
    import json as _json
    payload = _json.loads(read.to_json("backends"))
    assert payload == {"backends": ["auto", "piper", "system"]}


def test_to_json_voices_round_trip(tmp_path, monkeypatch) -> None:
    """to_json('voices') produces a parseable JSON object whose voices
    list is whatever list_voices() returned."""
    import json as _json
    voices = tmp_path / "v"
    voices.mkdir()
    (voices / "amy.onnx").write_bytes(b"")
    monkeypatch.setattr(read, "_PIPER_AUTO_PATHS", (str(voices),))
    payload = _json.loads(read.to_json("voices"))
    assert "voices" in payload
    assert isinstance(payload["voices"], list)
    assert any(p.endswith("amy.onnx") for p in payload["voices"])


def test_to_json_voices_empty_when_no_voices_installed(monkeypatch) -> None:
    """to_json('voices') is a valid JSON object with an empty list when
    no .onnx files are reachable."""
    import json as _json
    monkeypatch.setattr(read, "_PIPER_AUTO_PATHS", ("/nonexistent/voices",))
    payload = _json.loads(read.to_json("voices"))
    assert payload == {"voices": []}


def test_to_json_single_line() -> None:
    """The JSON output must be single-line so it pipes cleanly to jq."""
    assert "\n" not in read.to_json("backends")
    assert "\n" not in read.to_json("voices")


def test_to_json_unknown_kind_raises() -> None:
    """Unknown kind values raise ValueError (a programming error, not user)."""
    with pytest.raises(ValueError):
        read.to_json("not-a-kind")


# --- discovery: CLI flags -------------------------------------------------


def test_parse_args_list_backends_default_false() -> None:
    args = read.parse_args([])
    assert args.list_backends is False


def test_parse_args_list_voices_default_false() -> None:
    args = read.parse_args([])
    assert args.list_voices is False


def test_parse_args_json_default_false() -> None:
    args = read.parse_args([])
    assert args.as_json is False


def test_main_list_backends_text_mode(capsys) -> None:
    """--list-backends prints the three backends, one per line, exits 0."""
    code = read.main(["--list-backends"])
    out = capsys.readouterr().out
    assert code == 0
    assert out.strip().splitlines() == ["auto", "piper", "system"]


def test_main_list_voices_text_mode(capsys, monkeypatch) -> None:
    """--list-voices prints discovered voice paths, exits 0."""
    monkeypatch.setattr(read, "_PIPER_AUTO_PATHS", ("/nonexistent",))
    code = read.main(["--list-voices"])
    out = capsys.readouterr().out
    assert code == 0
    # No voices on a system without Piper -> no lines printed, output is empty.
    assert out == ""


def test_main_list_backends_with_voice_arg_ignores_voice(capsys) -> None:
    """--list-backends wins over a positional text argument."""
    code = read.main(["--list-backends", "this would be the text"])
    out = capsys.readouterr().out
    assert code == 0
    assert "auto" in out and "piper" in out and "system" in out
    # The text is never spoken — no "🐾 paw-read:" banner should appear.
    assert "🐾 paw-read:" not in out


def test_main_list_backends_json(capsys) -> None:
    """--list-backends --json emits a parseable JSON object."""
    import json as _json
    code = read.main(["--list-backends", "--json"])
    out = capsys.readouterr().out
    assert code == 0
    assert _json.loads(out) == {"backends": ["auto", "piper", "system"]}


def test_main_list_voices_json(capsys, monkeypatch) -> None:
    """--list-voices --json emits a parseable JSON object with a voices key."""
    import json as _json
    monkeypatch.setattr(read, "_PIPER_AUTO_PATHS", ("/nonexistent",))
    code = read.main(["--list-voices", "--json"])
    out = capsys.readouterr().out
    assert code == 0
    assert _json.loads(out) == {"voices": []}


def test_main_list_backends_does_not_call_speak(monkeypatch, capsys) -> None:
    """Discovery mode must never invoke the TTS backend."""
    called = {"n": 0}

    def _fake_speak(*args, **kwargs):
        called["n"] += 1
        return 0

    monkeypatch.setattr(read, "_speak", _fake_speak)
    code = read.main(["--list-backends"])
    assert code == 0
    assert called["n"] == 0


def test_main_json_without_discovery_flag_exits_2(capsys) -> None:
    """--json alone is a usage error (the user forgot --list-*)."""
    code = read.main(["--json"])
    err = capsys.readouterr().err
    assert code == 2
    assert "--json requires" in err
    assert "--list-backends" in err
    assert "--list-voices" in err


def test_main_list_backends_no_text_source_needed(monkeypatch, capsys) -> None:
    """--list-backends works on a system with no TTS backend AND no text."""
    # Force the auto backend picker to find nothing — discovery should
    # still succeed because it short-circuits before any TTS call.
    monkeypatch.setattr(read, "pick_backend", lambda *a, **k: None)
    monkeypatch.setenv("WPAW_READ_STDIN_OVERRIDE", "")
    code = read.main(["--list-backends"])
    out = capsys.readouterr().out
    assert code == 0
    assert "auto" in out and "system" in out


def test_main_text_mode_unchanged_by_discovery_flags(capsys, monkeypatch) -> None:
    """Regression: the default text output for --list-backends is one
    name per line, never a JSON object."""
    code = read.main(["--list-backends"])
    out = capsys.readouterr().out
    assert code == 0
    # The text path is one-name-per-line, so the output must NOT look
    # like a JSON object (no leading '{', no closing '}').
    assert "{" not in out
    assert "}" not in out
