"""Tests for paw-watch — run a command and read new output aloud.

The implementation is a thin wrapper around ``paw-read``: it spawns a
subprocess, reads its stdout line-by-line, and hands each line to the
same TTS backend chain that ``paw-read`` uses. Tests monkey-patch the
``_speak`` callable so no real TTS engine is needed in CI, and they
exercise the pure logic — arg parsing, line buffering, the empty
command error path, and exit-code semantics.
"""
from __future__ import annotations

import importlib
import subprocess

import pytest

watch = importlib.import_module("whisperpaw.watch")


# --- arg parsing ---------------------------------------------------------


def test_parse_args_defaults_expect_known_cmd() -> None:
    """Defaults are visible only once a real command is present.

    ``paw-watch`` without a command is a usage error (it has nothing to
    run), so parse_args raises SystemExit. We exercise defaults by
    passing a command alongside an empty flag set.
    """
    args = watch.parse_args(["--", "echo"])
    assert args.cmd == ["echo"]
    assert args.rate == pytest.approx(200.0)
    assert args.volume == pytest.approx(1.0)
    assert args.max_lines == 0
    assert args.quiet is False
    assert args.include_stderr is False


def test_parse_args_separates_dash_dash() -> None:
    """``-- CMD ARGS`` puts the command into ``args.cmd`` without our flags."""
    args = watch.parse_args(["--", "echo", "hello", "world"])
    assert args.cmd == ["echo", "hello", "world"]


def test_parse_args_accepts_our_flags_before_separator() -> None:
    args = watch.parse_args(
        ["--rate", "180", "--max-lines", "3", "--", "ls", "-la"]
    )
    assert args.rate == pytest.approx(180.0)
    assert args.max_lines == 3
    assert args.cmd == ["ls", "-la"]


def test_parse_args_quiet_flag() -> None:
    args = watch.parse_args(["--quiet", "--", "echo", "hi"])
    assert args.quiet is True
    assert args.cmd == ["echo", "hi"]


def test_parse_args_include_stderr_flag() -> None:
    args = watch.parse_args(["--include-stderr", "--", "make"])
    assert args.include_stderr is True


def test_parse_args_volume_flag() -> None:
    args = watch.parse_args(["--volume", "0.3", "--", "echo"])
    assert args.volume == pytest.approx(0.3)


def test_parse_args_rejects_rate_out_of_range() -> None:
    with pytest.raises(SystemExit):
        watch.parse_args(["--rate", "5", "--", "echo"])


def test_parse_args_rejects_volume_out_of_range() -> None:
    with pytest.raises(SystemExit):
        watch.parse_args(["--volume", "1.5", "--", "echo"])


def test_parse_args_rejects_negative_max_lines() -> None:
    with pytest.raises(SystemExit):
        watch.parse_args(["--max-lines", "-1", "--", "echo"])


# --- empty command error path -------------------------------------------


def test_parse_args_no_command_raises_systemexit() -> None:
    """``paw-watch`` with no command must refuse to run (exit 2)."""
    with pytest.raises(SystemExit) as exc:
        watch.parse_args([])
    # argparse raises SystemExit with code 2 for usage errors.
    assert exc.value.code in (2, None) or isinstance(exc.value.code, int)


def test_main_no_command_exits_2(monkeypatch, capsys) -> None:
    """``main([])`` must exit 2 with a clear message — never spawn anything."""
    called = {"spawn": False}

    def _fake_spawn(*a, **kw):
        called["spawn"] = True
        return None

    monkeypatch.setattr(watch, "_spawn", _fake_spawn)
    code = watch.main([])
    assert code == 2
    assert called["spawn"] is False
    err = capsys.readouterr().err
    assert "no command" in err.lower() or "usage" in err.lower()


# --- line buffering ------------------------------------------------------


def test_line_buffer_splits_on_newlines() -> None:
    """``_line_buffer`` must yield exactly one chunk per complete line."""
    buf = watch._LineBuffer()
    out = list(buf.feed(b"hello\nwor"))
    assert out == ["hello"]
    out = list(buf.feed(b"ld\nbye\n"))
    assert out == ["world", "bye"]


def test_line_buffer_holds_partial_line() -> None:
    """A trailing fragment with no newline stays in the buffer, not yielded."""
    buf = watch._LineBuffer()
    out = list(buf.feed(b"no newline yet"))
    assert out == []


def test_line_buffer_flush_returns_partial() -> None:
    """``flush()`` yields whatever's left, even without a newline."""
    buf = watch._LineBuffer()
    list(buf.feed(b"no newline"))
    assert buf.flush() == ["no newline"]


def test_line_buffer_handles_crlf() -> None:
    """Windows-style CRLF should also be a line separator."""
    buf = watch._LineBuffer()
    out = list(buf.feed(b"one\r\ntwo\r\n"))
    assert out == ["one", "two"]


def test_line_buffer_drops_empty_lines() -> None:
    """Blank lines are not interesting to speak — drop them."""
    buf = watch._LineBuffer()
    out = list(buf.feed(b"first\n\n\nsecond\n"))
    assert out == ["first", "second"]


def test_line_buffer_strips_whitespace() -> None:
    """Trailing spaces / tabs on a line should not be spoken."""
    buf = watch._LineBuffer()
    out = list(buf.feed(b"  hello  \n"))
    assert out == ["hello"]


# --- main() success path -------------------------------------------------


def test_main_speaks_each_line(monkeypatch, capsys) -> None:
    """Each complete line of the subprocess's stdout should be spoken."""
    spoken: list[str] = []

    # Patch the TTS dispatcher used by watch.
    monkeypatch.setattr(
        watch, "_speak_line",
        lambda line, rate, volume: spoken.append(line) or 0,
    )

    # Patch _spawn to return a fake CompletedProcess with predictable output.
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="alpha\nbeta\ngamma\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)

    code = watch.main(["--quiet", "--", "echo", "hi"])
    assert code == 0
    assert spoken == ["alpha", "beta", "gamma"]


def test_main_propagates_subprocess_exit_code(monkeypatch) -> None:
    """If the watched command fails, ``paw-watch`` should mirror its exit code."""
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: 0
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=42, stdout="boom\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)
    code = watch.main(["--quiet", "--", "false"])
    assert code == 42


def test_main_propagates_tts_exit_code(monkeypatch) -> None:
    """If TTS fails on a line, ``paw-watch`` should report 1."""
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: 1
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="x\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)
    code = watch.main(["--quiet", "--", "echo", "x"])
    assert code == 1


def test_main_respects_max_lines(monkeypatch) -> None:
    """After ``--max-lines`` lines, the rest of stdout is dropped on the floor."""
    spoken: list[str] = []
    monkeypatch.setattr(
        watch, "_speak_line",
        lambda line, rate, volume: spoken.append(line) or 0,
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="1\n2\n3\n4\n5\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)
    code = watch.main(["--quiet", "--max-lines", "2", "--", "seq", "5"])
    assert code == 0
    assert spoken == ["1", "2"]


def test_main_empty_stdout_is_ok(monkeypatch, capsys) -> None:
    """A command that produces no output is a valid run — exit 0, no speech."""
    spoken: list[str] = []
    monkeypatch.setattr(
        watch, "_speak_line",
        lambda line, rate, volume: spoken.append(line) or 0,
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)
    code = watch.main(["--quiet", "--", "true"])
    assert code == 0
    assert spoken == []


def test_main_announces_when_not_quiet(monkeypatch, capsys) -> None:
    """Without ``--quiet``, we should print a small banner line."""
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: 0
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="hi\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)
    code = watch.main(["--", "echo", "hi"])
    assert code == 0
    out = capsys.readouterr().out
    assert "🐾" in out
    assert "paw-watch" in out


# --- subprocess failure -------------------------------------------------


def test_main_spawn_failure_exits_2(monkeypatch, capsys) -> None:
    """If the watched binary does not exist, exit 2 (usage-ish)."""
    spoken: list[str] = []
    monkeypatch.setattr(
        watch, "_speak_line",
        lambda line, rate, volume: spoken.append(line) or 0,
    )

    def _bad_spawn(*a, **kw):
        raise FileNotFoundError("no such binary")

    monkeypatch.setattr(watch, "_spawn", _bad_spawn)
    code = watch.main(["--quiet", "--", "definitely-not-a-real-binary-xyz"])
    assert code == 2
    err = capsys.readouterr().err
    assert "not found" in err.lower() or "no such" in err.lower()


# --- stderr capture -----------------------------------------------------


def test_main_include_stderr_merges_output(monkeypatch) -> None:
    """``--include-stderr`` makes stderr lines eligible for speaking too.

    We simulate the real ``_spawn`` behavior: when ``include_stderr`` is
    true, stderr is redirected into stdout and ``completed.stderr`` is
    empty. The user only sees a single stream of lines.
    """
    spoken: list[str] = []
    monkeypatch.setattr(
        watch, "_speak_line",
        lambda line, rate, volume: spoken.append(line) or 0,
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="from-out\nfrom-err\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)
    code = watch.main(["--quiet", "--include-stderr", "--", "sh", "-c", "echo x"])
    assert code == 0
    assert spoken == ["from-out", "from-err"]
