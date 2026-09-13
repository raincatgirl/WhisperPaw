"""Tests for paw-read — read text aloud via TTS.

Tests cover the pure logic: argument parsing, source resolution (stdin
/ file / arg / clipboard), chunking of long text, and exit-code
semantics. Audio playback is monkey-patched so no real TTS engine is
needed in CI.
"""
from __future__ import annotations

import importlib

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
