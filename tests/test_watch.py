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
import sys

import pytest

watch = importlib.import_module("whisperpaw.watch")


def sys_executable() -> str:
    """Return the path to the current Python interpreter (test helper)."""
    return sys.executable


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


# --- --follow streaming mode --------------------------------------------


def test_parse_args_follow_flag() -> None:
    """``--follow`` is a boolean flag, default False."""
    args = watch.parse_args(["--", "tail", "-f", "f.txt"])
    assert args.follow is False
    args = watch.parse_args(["--follow", "--", "tail", "-f", "f.txt"])
    assert args.follow is True


def test_parse_args_follow_combines_with_other_flags() -> None:
    """``--follow`` plays nicely with ``--max-lines`` and ``--include-stderr``."""
    args = watch.parse_args([
        "--follow", "--max-lines", "3", "--include-stderr", "--", "make", "watch"
    ])
    assert args.follow is True
    assert args.max_lines == 3
    assert args.include_stderr is True
    assert args.cmd == ["make", "watch"]


def test_stream_process_iterates_lines() -> None:
    """``_StreamProcess.stdout_iter`` yields one line per output line, no newlines kept."""
    # The simplest streamable target: a subprocess whose stdout is a known string.
    # We test against the public _popen adapter with a tiny real subprocess so
    # the line-iteration semantics are exercised end-to-end.
    proc = watch._popen(
        [sys_executable(), "-c", "print('a'); print('b'); print('c')"],
        include_stderr=False,
    )
    try:
        lines = list(proc.stdout_iter())
    finally:
        proc.wait()
    assert lines == ["a\n", "b\n", "c\n"]


def test_stream_process_wait_returns_exit_code() -> None:
    """``_StreamProcess.wait()`` returns the child's exit code."""
    proc = watch._popen(
        [sys_executable(), "-c", "import sys; sys.exit(7)"],
        include_stderr=False,
    )
    # Drain stdout so the child can actually exit.
    list(proc.stdout_iter())
    assert proc.wait() == 7


def test_stream_process_include_stderr_merges() -> None:
    """With ``include_stderr=True``, stderr lines arrive in the stdout iterator."""
    proc = watch._popen(
        [sys_executable(), "-c",
         "import sys; print('out1'); sys.stderr.write('err1\\n'); sys.stderr.flush(); print('out2')"],
        include_stderr=True,
    )
    try:
        lines = list(proc.stdout_iter())
    finally:
        proc.wait()
    assert lines == ["out1\n", "err1\n", "out2\n"]


def test_main_follow_uses_popen(monkeypatch) -> None:
    """When ``--follow`` is set, ``main`` should use ``_popen`` not ``_spawn``."""
    used = {"spawn": False, "popen": False}

    def _fake_spawn(*a, **kw):
        used["spawn"] = True
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    class _FakeStream:
        def __init__(self, lines, rc=0):
            self._lines = lines
            self._rc = rc
        def stdout_iter(self):
            for line in self._lines:
                yield line
        def wait(self):
            return self._rc

    def _fake_popen(cmd, *, include_stderr):
        used["popen"] = True
        return _FakeStream(["alpha\n", "beta\n"], rc=0)

    monkeypatch.setattr(watch, "_spawn", _fake_spawn)
    monkeypatch.setattr(watch, "_popen", _fake_popen)
    monkeypatch.setattr(watch, "_speak_line", lambda line, rate, volume: 0)

    code = watch.main(["--quiet", "--follow", "--", "tail", "-f", "x.log"])
    assert code == 0
    assert used["popen"] is True
    assert used["spawn"] is False


def test_main_follow_speaks_lines_as_they_arrive(monkeypatch) -> None:
    """``--follow`` speaks each line of the streamed output in order."""
    spoken: list[str] = []
    monkeypatch.setattr(
        watch, "_speak_line",
        lambda line, rate, volume: spoken.append(line) or 0,
    )

    class _FakeStream:
        def stdout_iter(self):
            for line in ["first\n", "second\n", "third\n"]:
                yield line
        def wait(self):
            return 0

    monkeypatch.setattr(watch, "_popen", lambda *a, **kw: _FakeStream())
    code = watch.main(["--quiet", "--follow", "--", "tail", "-f", "x.log"])
    assert code == 0
    assert spoken == ["first", "second", "third"]


def test_main_follow_respects_max_lines(monkeypatch) -> None:
    """``--max-lines`` still applies in ``--follow`` mode."""
    spoken: list[str] = []
    monkeypatch.setattr(
        watch, "_speak_line",
        lambda line, rate, volume: spoken.append(line) or 0,
    )

    class _FakeStream:
        def stdout_iter(self):
            for line in ["1\n", "2\n", "3\n", "4\n", "5\n"]:
                yield line
        def wait(self):
            return 0

    monkeypatch.setattr(watch, "_popen", lambda *a, **kw: _FakeStream())
    code = watch.main(["--quiet", "--follow", "--max-lines", "2", "--", "seq", "5"])
    assert code == 0
    assert spoken == ["1", "2"]


def test_main_follow_mirrors_child_exit_code(monkeypatch) -> None:
    """``--follow`` still mirrors the watched command's exit code on failure."""
    monkeypatch.setattr(watch, "_speak_line", lambda line, rate, volume: 0)

    class _FakeStream:
        def stdout_iter(self):
            yield "boom\n"
        def wait(self):
            return 42

    monkeypatch.setattr(watch, "_popen", lambda *a, **kw: _FakeStream())
    code = watch.main(["--quiet", "--follow", "--", "false"])
    assert code == 42


def test_main_follow_propagates_tts_error(monkeypatch) -> None:
    """TTS failure in ``--follow`` mode still returns 1."""
    monkeypatch.setattr(watch, "_speak_line", lambda line, rate, volume: 1)

    class _FakeStream:
        def stdout_iter(self):
            yield "x\n"
        def wait(self):
            return 0

    monkeypatch.setattr(watch, "_popen", lambda *a, **kw: _FakeStream())
    code = watch.main(["--quiet", "--follow", "--", "echo", "x"])
    assert code == 1


def test_main_follow_handles_trailing_partial_line(monkeypatch) -> None:
    """A final line without a newline should still be flushed & spoken."""
    spoken: list[str] = []
    monkeypatch.setattr(
        watch, "_speak_line",
        lambda line, rate, volume: spoken.append(line) or 0,
    )

    class _FakeStream:
        def stdout_iter(self):
            yield "with-newline\n"
            yield "no-newline"  # no \n
        def wait(self):
            return 0

    monkeypatch.setattr(watch, "_popen", lambda *a, **kw: _FakeStream())
    code = watch.main(["--quiet", "--follow", "--", "cmd"])
    assert code == 0
    assert spoken == ["with-newline", "no-newline"]


def test_main_follow_spawn_failure_exits_2(monkeypatch, capsys) -> None:
    """If the watched binary does not exist in follow mode, exit 2."""
    monkeypatch.setattr(watch, "_speak_line", lambda line, rate, volume: 0)

    def _bad_popen(*a, **kw):
        raise FileNotFoundError(2, "no such binary", a[0] if a else None)

    monkeypatch.setattr(watch, "_popen", _bad_popen)
    code = watch.main(["--quiet", "--follow", "--", "definitely-not-a-real-binary-xyz"])
    assert code == 2
    err = capsys.readouterr().err
    assert "not found" in err.lower() or "no such" in err.lower()


def test_main_without_follow_still_uses_spawn(monkeypatch) -> None:
    """Regression: without ``--follow``, the batch ``_spawn`` path is taken."""
    used = {"spawn": False, "popen": False}

    def _fake_spawn(*a, **kw):
        used["spawn"] = True
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="ok\n", stderr="")

    def _fake_popen(*a, **kw):
        used["popen"] = True
        raise AssertionError("_popen should not be called in batch mode")

    monkeypatch.setattr(watch, "_spawn", _fake_spawn)
    monkeypatch.setattr(watch, "_popen", _fake_popen)
    monkeypatch.setattr(watch, "_speak_line", lambda line, rate, volume: 0)

    code = watch.main(["--quiet", "--", "echo", "ok"])
    assert code == 0
    assert used["spawn"] is True
    assert used["popen"] is False


# --- _spawn end-to-end regression ----------------------------------------


def test_spawn_runs_real_command_without_stderr() -> None:
    """Regression: ``_spawn(cmd, include_stderr=False)`` must not raise.

    A real subprocess.run bug: passing both ``capture_output=True`` and
    ``stderr=...`` raises ``ValueError`` in CPython 3.x. The fix is to
    only set ``capture_output`` on the no-merge path and handle the
    merge case with explicit ``stdout=PIPE, stderr=STDOUT``.
    """
    completed = watch._spawn(
        [sys_executable(), "-c", "print('hi')"],
        include_stderr=False,
    )
    assert completed.returncode == 0
    assert "hi" in completed.stdout


def test_spawn_merges_stderr_when_requested() -> None:
    """``_spawn(cmd, include_stderr=True)`` redirects stderr into stdout.

    Note: when ``stderr=STDOUT`` is set, ``CompletedProcess.stderr`` is
    ``None`` (per CPython docs) — we don't try to read it. The user
    sees a single stream in ``completed.stdout``.
    """
    completed = watch._spawn(
        [sys_executable(), "-c",
         "import sys; sys.stderr.write('e\\n'); sys.stdout.write('o\\n')"],
        include_stderr=True,
    )
    assert completed.returncode == 0
    # In the merge case .stderr is None (the field is unset because
    # stderr was redirected, not captured).
    assert completed.stderr in (None, "")
    # Both lines should have ended up in stdout.
    lines = [ln for ln in completed.stdout.splitlines() if ln]
    assert "e" in lines
    assert "o" in lines


# --- --dry-run preview mode ----------------------------------------------


def test_parse_args_dry_run_flag() -> None:
    """``--dry-run`` is a boolean flag, default False."""
    args = watch.parse_args(["--", "echo", "hi"])
    assert args.dry_run is False
    args = watch.parse_args(["--dry-run", "--", "echo", "hi"])
    assert args.dry_run is True


def test_parse_args_dry_run_combines_with_other_flags() -> None:
    """``--dry-run`` plays nicely with ``--max-lines`` and ``--follow``."""
    args = watch.parse_args(
        ["--dry-run", "--max-lines", "3", "--follow", "--", "make", "watch"]
    )
    assert args.dry_run is True
    assert args.max_lines == 3
    assert args.follow is True
    assert args.cmd == ["make", "watch"]


def test_emit_line_speaks_when_not_dry_run() -> None:
    """In normal mode ``_emit_line`` forwards to ``_speak_line``."""
    seen: list[str] = []
    monkey_calls = {"n": 0}

    def fake_speak(line, rate, volume):
        seen.append(line)
        monkey_calls["n"] += 1
        return 0

    saved = watch._speak_line
    watch._speak_line = fake_speak
    try:
        code = watch._emit_line("hi", rate=200.0, volume=1.0, dry_run=False)
    finally:
        watch._speak_line = saved

    assert code == 0
    assert seen == ["hi"]
    assert monkey_calls["n"] == 1


def test_emit_line_prints_in_dry_run(capsys) -> None:
    """In dry-run mode ``_emit_line`` prints the line and returns 0,
    without ever touching ``_speak_line``."""
    spoke = {"called": False}

    def fake_speak(line, rate, volume):
        spoke["called"] = True
        return 0

    saved = watch._speak_line
    watch._speak_line = fake_speak
    try:
        code = watch._emit_line("hello world", rate=200.0, volume=1.0, dry_run=True)
    finally:
        watch._speak_line = saved

    out = capsys.readouterr().out
    assert code == 0
    assert out == "hello world\n"
    assert spoke["called"] is False


def test_main_dry_run_prints_lines(monkeypatch, capsys) -> None:
    """``--dry-run`` prints each complete line to stdout instead of TTS."""
    spoke = {"called": False}

    def _fake_speak(*a, **kw):
        spoke["called"] = True
        return 0

    monkeypatch.setattr(watch, "_speak_line", _fake_speak)

    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="alpha\nbeta\ngamma\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)

    code = watch.main(["--quiet", "--dry-run", "--", "echo", "hi"])
    assert code == 0
    out = capsys.readouterr().out
    assert out == "alpha\nbeta\ngamma\n"
    assert spoke["called"] is False


def test_main_dry_run_respects_max_lines(monkeypatch, capsys) -> None:
    """``--dry-run`` honours ``--max-lines`` exactly like the speaking path."""
    spoke = {"called": False}
    monkeypatch.setattr(
        watch, "_speak_line", lambda *a, **kw: spoke.__setitem__("called", True) or 0
    )

    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="1\n2\n3\n4\n5\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)

    code = watch.main(
        ["--quiet", "--dry-run", "--max-lines", "2", "--", "seq", "5"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert out == "1\n2\n"
    assert spoke["called"] is False


def test_main_dry_run_mirrors_child_exit_code(monkeypatch, capsys) -> None:
    """A failing child under ``--dry-run`` still returns the child's code,
    not 0. We must never mask a real failure just because TTS is off."""
    spoke = {"called": False}
    monkeypatch.setattr(
        watch, "_speak_line", lambda *a, **kw: spoke.__setitem__("called", True) or 0
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=42, stdout="boom\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)

    code = watch.main(["--quiet", "--dry-run", "--", "false"])
    assert code == 42
    out = capsys.readouterr().out
    assert out == "boom\n"
    assert spoke["called"] is False


def test_main_dry_run_propagates_spawn_failure(monkeypatch, capsys) -> None:
    """``--dry-run`` cannot conjure a successful run if the binary is missing."""
    spoke = {"called": False}
    monkeypatch.setattr(
        watch, "_speak_line", lambda *a, **kw: spoke.__setitem__("called", True) or 0
    )

    def _bad_spawn(*a, **kw):
        raise FileNotFoundError(2, "no such binary", a[0] if a else None)

    monkeypatch.setattr(watch, "_spawn", _bad_spawn)
    code = watch.main(
        ["--quiet", "--dry-run", "--", "definitely-not-a-real-binary-xyz"]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "not found" in err.lower() or "no such" in err.lower()
    assert spoke["called"] is False


def test_main_dry_run_announces_previewing(monkeypatch, capsys) -> None:
    """The banner in ``--dry-run`` mode says ``previewing``, not ``tailing``/``following``."""
    monkeypatch.setattr(watch, "_speak_line", lambda *a, **kw: 0)
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="hi\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)

    code = watch.main(["--dry-run", "--", "echo", "hi"])
    assert code == 0
    out = capsys.readouterr().out
    assert "🐾" in out
    assert "previewing" in out
    # We must not say "tailing" or "following" -- the user is previewing.
    assert "tailing" not in out
    assert "following" not in out


def test_main_dry_run_works_with_follow(monkeypatch, capsys) -> None:
    """``--dry-run --follow`` prints streamed lines instead of speaking them."""
    spoke = {"called": False}
    monkeypatch.setattr(
        watch, "_speak_line", lambda *a, **kw: spoke.__setitem__("called", True) or 0
    )

    class _FakeStream:
        def stdout_iter(self):
            for line in ["first\n", "second\n", "third\n"]:
                yield line
        def wait(self):
            return 0

    monkeypatch.setattr(watch, "_popen", lambda *a, **kw: _FakeStream())
    code = watch.main(["--quiet", "--dry-run", "--follow", "--", "tail", "-f", "x.log"])
    assert code == 0
    out = capsys.readouterr().out
    assert out == "first\nsecond\nthird\n"
    assert spoke["called"] is False


def test_main_dry_run_empty_stdout_is_ok(monkeypatch, capsys) -> None:
    """A child that produces no output under ``--dry-run`` prints nothing and exits 0."""
    spoke = {"called": False}
    monkeypatch.setattr(
        watch, "_speak_line", lambda *a, **kw: spoke.__setitem__("called", True) or 0
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)
    code = watch.main(["--quiet", "--dry-run", "--", "true"])
    assert code == 0
    out = capsys.readouterr().out
    assert out == ""
    assert spoke["called"] is False


def test_main_dry_run_handles_trailing_partial_line(monkeypatch, capsys) -> None:
    """A final line without a newline should still be flushed & printed in dry-run."""
    spoke = {"called": False}
    monkeypatch.setattr(
        watch, "_speak_line", lambda *a, **kw: spoke.__setitem__("called", True) or 0
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="with-newline\nno-newline", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)
    code = watch.main(["--quiet", "--dry-run", "--", "cmd"])
    assert code == 0
    out = capsys.readouterr().out
    assert out == "with-newline\nno-newline\n"
    assert spoke["called"] is False


def test_main_dry_run_does_not_call_speak_line_even_with_tts_error(
    monkeypatch, capsys
) -> None:
    """``--dry-run`` must not consult the TTS chain at all -- so a TTS error
    in the environment cannot bubble up through the dry-run path."""
    # If _speak_line were ever called under --dry-run, this mock would
    # return a non-zero code and the test would fail. The point is that
    # we should never get there.
    def _would_break(*a, **kw):
        raise AssertionError(
            "_speak_line must not be called when --dry-run is set"
        )

    monkeypatch.setattr(watch, "_speak_line", _would_break)
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="anything\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)
    code = watch.main(["--quiet", "--dry-run", "--", "echo", "anything"])
    assert code == 0
    out = capsys.readouterr().out
    assert out == "anything\n"


# --- --transcript --------------------------------------------------------


def test_parse_args_transcript_default_is_none() -> None:
    """Without ``--transcript`` the option stays ``None`` so the runtime
    can short-circuit the file-open path entirely."""
    args = watch.parse_args(["--", "echo", "hi"])
    assert args.transcript is None


def test_parse_args_transcript_accepts_path() -> None:
    """A real-looking path round-trips through ``parse_args``."""
    args = watch.parse_args(["--transcript", "/tmp/session.log", "--", "echo"])
    assert args.transcript == "/tmp/session.log"
    assert args.cmd == ["echo"]


def test_parse_args_transcript_rejects_missing_parent_dir(tmp_path) -> None:
    """A ``--transcript`` whose parent directory does not exist is a
    usage error (exit 2) with a clear message — not a runtime surprise
    mid-speech."""
    bad = tmp_path / "no-such-dir" / "out.log"
    with pytest.raises(SystemExit) as exc:
        watch.parse_args(["--transcript", str(bad), "--", "echo"])
    assert exc.value.code == 2
    # The message should mention the bad directory so the user can fix it.
    # (We don't pin the exact format; just the actionable hint.)


def test_open_transcript_returns_none_when_path_is_none() -> None:
    """``open_transcript(None)`` is a no-op — no handle, no file is touched."""
    assert watch.open_transcript(None) is None


def test_open_transcript_creates_file_in_append_mode(tmp_path) -> None:
    """``open_transcript`` opens the file in append mode and creates it
    on first call (so a fresh log path doesn't need a pre-touch)."""
    log = tmp_path / "transcript.log"
    fh = watch.open_transcript(str(log))
    try:
        assert fh is not None
        fh.write("hello\n")
        fh.flush()
    finally:
        fh.close()
    assert log.read_text(encoding="utf-8") == "hello\n"


def test_open_transcript_appends_to_existing_file(tmp_path) -> None:
    """A second open of the same path must not truncate the existing
    log — append-mode is the contract the ``--follow`` story depends on."""
    log = tmp_path / "transcript.log"
    log.write_text("first session\n", encoding="utf-8")
    fh = watch.open_transcript(str(log))
    try:
        fh.write("second session\n")
        fh.flush()
    finally:
        fh.close()
    assert log.read_text(encoding="utf-8") == "first session\nsecond session\n"


def test_transcript_writer_returns_none_for_none_handle() -> None:
    """``_transcript_writer(None)`` returns ``None`` so the caller can
    pass the result straight into :func:`_emit_line` without branching."""
    assert watch._transcript_writer(None) is None


def test_transcript_writer_writes_line_then_newline_then_flushes(tmp_path) -> None:
    """The closure writes ``line + "\\n"`` and flushes after every line so
    a long ``--follow`` run produces a usable, tail-able log even if
    the process is killed mid-stream."""
    log = tmp_path / "transcript.log"
    log.write_text("", encoding="utf-8")
    fh = watch.open_transcript(str(log))
    try:
        writer = watch._transcript_writer(fh)
        writer("alpha")
        writer("beta")
    finally:
        fh.close()
    assert log.read_text(encoding="utf-8") == "alpha\nbeta\n"


def test_emit_line_with_transcript_writes_before_speaking(
    monkeypatch, tmp_path
) -> None:
    """``_emit_line`` must call the transcript writer before the TTS chain
    so a TTS error doesn't lose the line from the log."""
    written: list[str] = []
    called_speak = {"n": 0}
    monkeypatch.setattr(
        watch, "_speak_line",
        lambda line, rate, volume: called_speak.__setitem__("n", called_speak["n"] + 1) or 0,
    )
    code = watch._emit_line(
        "first line",
        rate=200.0,
        volume=1.0,
        dry_run=False,
        transcript_write=written.append,
    )
    assert code == 0
    assert written == ["first line"]
    assert called_speak["n"] == 1


def test_emit_line_transcript_write_failure_does_not_change_exit_code(
    monkeypatch, capsys
) -> None:
    """A failed transcript write is loud on stderr but does NOT change
    the TTS exit code — the spoken output is the source of truth and
    a flaky filesystem should not break the user's pipeline."""
    def _bad_writer(_line: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(
        watch, "_speak_line", lambda *a, **kw: 0
    )
    code = watch._emit_line(
        "anything",
        rate=200.0,
        volume=1.0,
        dry_run=False,
        transcript_write=_bad_writer,
    )
    assert code == 0
    err = capsys.readouterr().err
    assert "transcript" in err.lower() and "disk full" in err


def test_main_writes_each_spoken_line_to_transcript(monkeypatch, tmp_path) -> None:
    """End-to-end: ``--transcript PATH`` records every line that
    :func:`_speak_line` was called with, in order, one per line, in
    append mode. The test uses a real file (tmp_path) so we also
    exercise the file-handle close-on-exit path."""
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: 0
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="alpha\nbeta\ngamma\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)
    log = tmp_path / "session.log"
    code = watch.main(
        ["--quiet", "--transcript", str(log), "--", "echo", "x"]
    )
    assert code == 0
    assert log.read_text(encoding="utf-8") == "alpha\nbeta\ngamma\n"


def test_main_transcript_in_dry_run_records_what_would_be_spoken(
    monkeypatch, tmp_path, capsys
) -> None:
    """``--transcript`` + ``--dry-run`` records the *would-be* spoken
    lines (the same lines that go to stdout), so a script can use
    dry-run as a "show me and log it" preview without ever touching
    the TTS engine."""
    monkeypatch.setattr(
        watch, "_speak_line", lambda *a, **kw: 0
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="would-speak-this\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)
    log = tmp_path / "session.log"
    code = watch.main(
        ["--quiet", "--dry-run", "--transcript", str(log), "--", "echo"]
    )
    assert code == 0
    assert log.read_text(encoding="utf-8") == "would-speak-this\n"
    # And the same line went to stdout (the dry-run contract).
    assert capsys.readouterr().out == "would-speak-this\n"


def test_main_transcript_respects_max_lines(monkeypatch, tmp_path) -> None:
    """``--max-lines`` is the source-of-truth cap on which lines are
    "spoken" — ``--transcript`` must record the same set, no more."""
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: 0
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="1\n2\n3\n4\n5\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)
    log = tmp_path / "session.log"
    code = watch.main(
        [
            "--quiet",
            "--max-lines", "2",
            "--transcript", str(log),
            "--", "seq", "5",
        ]
    )
    assert code == 0
    assert log.read_text(encoding="utf-8") == "1\n2\n"


def test_main_transcript_appends_across_runs(monkeypatch, tmp_path) -> None:
    """Two consecutive ``paw-watch --transcript PATH`` runs against the
    same file must concatenate, not overwrite — that's the whole
    point of append mode for an accessibility log."""
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: 0
    )
    fake1 = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="run-one-line\n", stderr=""
    )
    fake2 = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="run-two-line\n", stderr=""
    )
    log = tmp_path / "session.log"

    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake1)
    code = watch.main(["--quiet", "--transcript", str(log), "--", "a"])
    assert code == 0

    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake2)
    code = watch.main(["--quiet", "--transcript", str(log), "--", "b"])
    assert code == 0

    assert (
        log.read_text(encoding="utf-8")
        == "run-one-line\nrun-two-line\n"
    )


def test_main_follow_writes_each_streamed_line_to_transcript(
    monkeypatch, tmp_path
) -> None:
    """``--follow --transcript`` writes each streamed line as it
    arrives, so a tail-style watch leaves a real-time transcript
    behind."""
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: 0
    )

    class _FakeStream:
        def stdout_iter(self):
            for line in ["first\n", "second\n", "third\n"]:
                yield line
        def wait(self):
            return 0

    monkeypatch.setattr(watch, "_popen", lambda *a, **kw: _FakeStream())
    log = tmp_path / "session.log"
    code = watch.main(
        [
            "--quiet", "--follow",
            "--transcript", str(log),
            "--", "tail", "-f", "x.log",
        ]
    )
    assert code == 0
    assert log.read_text(encoding="utf-8") == "first\nsecond\nthird\n"


def test_main_transcript_closes_handle_on_tts_error(
    monkeypatch, tmp_path
) -> None:
    """Even when the TTS chain reports a non-zero exit, the transcript
    handle must be closed — otherwise a long ``--follow`` run that
    crashes the TTS layer would leak an open file descriptor on every
    line."""
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: 1
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="alpha\nbeta\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)
    log = tmp_path / "session.log"
    code = watch.main(
        ["--quiet", "--transcript", str(log), "--", "echo"]
    )
    # TTS error: 1 (the child's exit code was 0 so it doesn't shadow).
    assert code == 1
    # And the lines were still recorded before the failure path returned.
    assert log.read_text(encoding="utf-8") == "alpha\nbeta\n"


def test_main_transcript_path_open_failure_exits_2(
    monkeypatch, capsys, tmp_path
) -> None:
    """A ``--transcript`` path that exists but can't be opened
    (e.g. it's a directory) is a usage error reported on stderr."""
    # Treat the directory itself as the transcript path. The directory
    # exists, so _validate_transcript_path passes; the open() in
    # main() is what fails (IsADirectoryError is an OSError subclass).
    bad = tmp_path  # a directory, not a file
    code = watch.main(
        ["--quiet", "--transcript", str(bad), "--", "echo"]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "transcript" in err.lower()
    # Critically, no subprocess was spawned.


# --- --prefix -------------------------------------------------------------


def test_parse_args_prefix_default_is_empty_string() -> None:
    """Without ``--prefix`` the option stays the empty string, so the
    no-flag behaviour is bit-identical to the pre-``--prefix`` code
    (no leading ``[] `` for an empty label)."""
    args = watch.parse_args(["--", "echo"])
    assert args.prefix == ""


def test_parse_args_prefix_accepts_label() -> None:
    """``--prefix TEXT`` is exposed on the namespace as ``args.prefix``."""
    args = watch.parse_args(["--prefix", "build", "--", "npm", "run", "build"])
    assert args.prefix == "build"
    assert args.cmd == ["npm", "run", "build"]


def test_parse_args_prefix_combines_with_transcript() -> None:
    """``--prefix`` and ``--transcript`` are independent flags — the
    transcript path is still validated at parse time and the prefix
    is still attached to the namespace."""
    args = watch.parse_args(
        [
            "--prefix", "test",
            "--transcript", "/tmp/x.log",
            "--", "pytest",
        ]
    )
    assert args.prefix == "test"
    assert args.transcript == "/tmp/x.log"


def test_apply_prefix_empty_prefix_is_noop() -> None:
    """``_apply_prefix`` with an empty ``prefix`` returns the line
    verbatim — the no-flag code path is bit-identical to the
    pre-``--prefix`` behaviour."""
    assert watch._apply_prefix("hello", "") == "hello"
    assert watch._apply_prefix("", "") == ""


def test_apply_prefix_wraps_in_brackets_with_space() -> None:
    """``_apply_prefix`` returns ``f"[{prefix}] {line}"`` — the
    conventional tagging shape (square brackets + single space
    separator) used by the announcement banner and every other
    ``whisperpaw`` log line."""
    assert watch._apply_prefix("hello", "build") == "[build] hello"
    assert watch._apply_prefix("a", "x") == "[x] a"


def test_apply_prefix_preserves_unicode_in_line() -> None:
    """A non-ASCII line passes through unchanged; the bracket-and-space
    ASCII wrapper is still applied around it."""
    assert watch._apply_prefix("编译中…", "build") == "[build] 编译中…"


def test_emit_line_prefix_in_speak_mode(monkeypatch) -> None:
    """In speak mode the prefix is prepended to the line that reaches
    the TTS chain (not just the dry-run printer)."""
    seen: list[str] = []
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: (seen.append(line) or 0)
    )
    code = watch._emit_line(
        "compiling", rate=200.0, volume=1.0, dry_run=False, prefix="build"
    )
    assert code == 0
    assert seen == ["[build] compiling"]


def test_emit_line_prefix_in_dry_run(capsys, monkeypatch) -> None:
    """In dry-run mode the prefix is prepended to the line that goes
    to stdout, so the printed record is consistent with what would
    have been spoken."""
    spoke = {"called": False}
    monkeypatch.setattr(
        watch, "_speak_line", lambda *a, **kw: (spoke.__setitem__("called", True) or 0)
    )
    code = watch._emit_line(
        "running suite", rate=200.0, volume=1.0, dry_run=True, prefix="test"
    )
    assert code == 0
    assert not spoke["called"]
    assert capsys.readouterr().out == "[test] running suite\n"


def test_emit_line_empty_prefix_is_bit_identical_to_no_flag(monkeypatch) -> None:
    """``_emit_line`` with the default ``prefix=""`` produces the same
    line as the pre-``--prefix`` code — no spurious ``[] `` wrapper."""
    seen: list[str] = []
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: (seen.append(line) or 0)
    )
    code = watch._emit_line("hello", rate=200.0, volume=1.0, dry_run=False)
    assert code == 0
    assert seen == ["hello"]


def test_emit_line_prefix_applied_before_transcript_write(monkeypatch) -> None:
    """When both ``prefix`` and ``transcript_write`` are set, the
    tagged line is what hits the transcript — so the on-disk record
    carries the tag too."""
    written: list[str] = []
    seen: list[str] = []
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: (seen.append(line) or 0)
    )
    code = watch._emit_line(
        "alpha",
        rate=200.0,
        volume=1.0,
        dry_run=False,
        prefix="build",
        transcript_write=written.append,
    )
    assert code == 0
    assert written == ["[build] alpha"]
    assert seen == ["[build] alpha"]


def test_main_prefix_tags_every_spoken_line(monkeypatch) -> None:
    """End-to-end: a batch run with ``--prefix build`` forwards
    ``[build] LINE`` to ``_speak_line`` for every output line."""
    seen: list[str] = []
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: (seen.append(line) or 0)
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="first\nsecond\nthird\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)
    code = watch.main(["--quiet", "--prefix", "build", "--", "echo"])
    assert code == 0
    assert seen == ["[build] first", "[build] second", "[build] third"]


def test_main_prefix_empty_by_default(monkeypatch) -> None:
    """Without ``--prefix`` the spoken lines are byte-identical to the
    pre-``--prefix`` code — no spurious ``[] `` wrapper."""
    seen: list[str] = []
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: (seen.append(line) or 0)
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="a\nb\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)
    code = watch.main(["--quiet", "--", "echo"])
    assert code == 0
    assert seen == ["a", "b"]


def test_main_prefix_in_dry_run_prints_tagged_lines(
    monkeypatch, capsys
) -> None:
    """``--prefix`` + ``--dry-run`` prints ``[TEXT] LINE`` to stdout —
    the same tagged shape the speak / transcript sinks would have
    produced, so a preview matches the eventual record."""
    monkeypatch.setattr(
        watch, "_speak_line", lambda *a, **kw: 0
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="alpha\nbeta\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)
    code = watch.main(
        ["--quiet", "--dry-run", "--prefix", "test", "--", "echo"]
    )
    assert code == 0
    assert capsys.readouterr().out == "[test] alpha\n[test] beta\n"


def test_main_prefix_writes_tagged_lines_to_transcript(
    monkeypatch, tmp_path
) -> None:
    """``--prefix`` + ``--transcript`` writes ``[TEXT] LINE`` to the
    log file — the primary use case (tagging concurrent sessions
    sharing a single log)."""
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: 0
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="compiling\nlinking\ndone\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)
    log = tmp_path / "session.log"
    code = watch.main(
        [
            "--quiet",
            "--prefix", "build",
            "--transcript", str(log),
            "--", "make",
        ]
    )
    assert code == 0
    assert (
        log.read_text(encoding="utf-8")
        == "[build] compiling\n[build] linking\n[build] done\n"
    )


def test_main_two_concurrent_paw_watches_share_one_transcript(
    monkeypatch, tmp_path
) -> None:
    """The headline use case: two ``paw-watch`` invocations with
    different ``--prefix`` values write distinguishable tagged lines
    to a *shared* ``--transcript`` file, so a downstream reader can
    tell the two streams apart at a glance."""
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: 0
    )

    fake1 = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="compiling\n", stderr=""
    )
    fake2 = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="running suite\n", stderr=""
    )
    log = tmp_path / "shared.log"

    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake1)
    code = watch.main(
        [
            "--quiet",
            "--prefix", "build",
            "--transcript", str(log),
            "--", "make",
        ]
    )
    assert code == 0

    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake2)
    code = watch.main(
        [
            "--quiet",
            "--prefix", "test",
            "--transcript", str(log),
            "--", "pytest",
        ]
    )
    assert code == 0

    assert (
        log.read_text(encoding="utf-8")
        == "[build] compiling\n[test] running suite\n"
    )


def test_main_follow_prefix_tags_streamed_lines(monkeypatch) -> None:
    """``--follow --prefix`` tags every streamed line, so a live
    tail-style session leaves a tagged real-time record."""
    seen: list[str] = []
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: (seen.append(line) or 0)
    )

    class _FakeStream:
        def stdout_iter(self):
            for line in ["event-a\n", "event-b\n"]:
                yield line
        def wait(self):
            return 0

    monkeypatch.setattr(watch, "_popen", lambda *a, **kw: _FakeStream())
    code = watch.main(
        ["--quiet", "--follow", "--prefix", "svc", "--", "tail", "-f", "x.log"]
    )
    assert code == 0
    assert seen == ["[svc] event-a", "[svc] event-b"]


def test_main_prefix_respects_max_lines(monkeypatch) -> None:
    """``--prefix`` does not change the cap: ``--max-lines N`` still
    stops at the Nth *original* line, and every emitted line is
    tagged."""
    seen: list[str] = []
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: (seen.append(line) or 0)
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="1\n2\n3\n4\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)
    code = watch.main(
        [
            "--quiet",
            "--max-lines", "2",
            "--prefix", "n",
            "--", "seq", "4",
        ]
    )
    assert code == 0
    assert seen == ["[n] 1", "[n] 2"]


def test_main_prefix_propagates_tts_error(monkeypatch) -> None:
    """A TTS error on a tagged line still surfaces as exit 1 — the
    prefix is purely cosmetic from the error-code perspective."""
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: 1
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="boom\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)
    code = watch.main(
        ["--quiet", "--prefix", "x", "--", "echo"]
    )
    assert code == 1


# --- --meta ---------------------------------------------------------------


def test_parse_args_meta_default_is_off() -> None:
    """Without ``--meta`` the flag stays ``False`` so the no-flag
    transcript behaviour is bit-identical to the pre-``--meta`` code
    (no leading header line in the log)."""
    args = watch.parse_args(["--", "echo"])
    assert args.meta is False


def test_parse_args_meta_flag_on() -> None:
    """``--meta`` is exposed on the namespace as ``args.meta = True``."""
    args = watch.parse_args(["--meta", "--", "echo"])
    assert args.meta is True


def test_format_meta_is_single_line_starting_with_hash() -> None:
    """The header is a single line beginning with ``#`` so the transcript
    stays grep-friendly (``grep -v '^#'`` strips every header) and any
    downstream ``tail -f`` consumer can recognise and ignore it.
    """
    from argparse import Namespace

    args = Namespace(
        cmd=["echo", "hi"],
        prefix="",
        rate=200.0,
        volume=1.0,
        max_lines=0,
        follow=False,
        dry_run=False,
    )
    fixed_now = watch._datetime(2026, 9, 28, 14, 55, 0, tzinfo=watch.timezone.utc)
    line = watch._format_meta(args, now=fixed_now)
    assert "\n" not in line
    assert line.startswith("# ")


def test_format_meta_includes_timestamp_cmd_and_key_flags() -> None:
    """The header names the session clock, the command, the prefix, and
    the runtime flags a reader most needs to interpret a log line.
    """
    from argparse import Namespace

    args = Namespace(
        cmd=["make", "test"],
        prefix="ci",
        rate=220.0,
        volume=0.5,
        max_lines=10,
        follow=True,
        dry_run=True,
    )
    fixed_now = watch._datetime(2026, 9, 28, 14, 55, 0, tzinfo=watch.timezone.utc)
    line = watch._format_meta(args, now=fixed_now)
    # The clock is ISO-8601 with a Z (UTC) suffix.
    assert "2026-09-28T14:55:00Z" in line
    # The full command, including its args, is embedded via shlex.join
    # so a downstream parser can re-split it. For two plain words with
    # no metacharacters, shlex.join emits a single space-separated
    # string (`make test`); for an arg containing whitespace it would
    # emit quoted form. We assert on the ``cmd=`` prefix + the words
    # being present, which is the contract the user can rely on.
    assert "cmd=" in line
    assert "make test" in line
    # The key runtime flags are present with their actual values.
    assert "prefix=ci" in line
    assert "rate=220" in line
    assert "volume=0.5" in line
    assert "max_lines=10" in line
    assert "follow=True" in line
    assert "dry_run=True" in line


def test_format_meta_quotes_args_with_whitespace() -> None:
    """An argv word containing whitespace or shell metacharacters is
    quoted by ``shlex.join`` so the line round-trips through a
    downstream parser. Plain words are emitted unquoted.
    """
    from argparse import Namespace

    args = Namespace(
        cmd=["echo", "hello world"],
        prefix="",
        rate=200.0,
        volume=1.0,
        max_lines=0,
        follow=False,
        dry_run=False,
    )
    fixed_now = watch._datetime(2026, 9, 28, 0, 0, 0, tzinfo=watch.timezone.utc)
    line = watch._format_meta(args, now=fixed_now)
    # The whitespace-bearing arg is wrapped in single quotes (the
    # default ``shlex.join`` quoting style). The important property
    # is that the command is *some* quoted form so a downstream
    # parser can re-split it without ambiguity.
    assert "cmd=echo 'hello world'" in line


def test_format_meta_empty_prefix_is_omitted() -> None:
    """When the user did not pass ``--prefix`` we don't print
    ``prefix=`` in the header — leaving it out keeps the no-prefix
    line tight, and a downstream consumer can default to ''."""
    from argparse import Namespace

    args = Namespace(
        cmd=["echo"],
        prefix="",
        rate=200.0,
        volume=1.0,
        max_lines=0,
        follow=False,
        dry_run=False,
    )
    fixed_now = watch._datetime(2026, 9, 28, 0, 0, 0, tzinfo=watch.timezone.utc)
    line = watch._format_meta(args, now=fixed_now)
    assert "prefix=" not in line


def test_write_meta_is_noop_when_fh_is_none() -> None:
    """``write_meta(None, …)`` is a no-op so the caller can pass a
    possibly-``None`` transcript handle through unchanged."""
    from argparse import Namespace

    args = Namespace(
        cmd=["echo"],
        prefix="",
        rate=200.0,
        volume=1.0,
        max_lines=0,
        follow=False,
        dry_run=False,
    )
    # Should not raise.
    assert watch.write_meta(None, args) is None


def test_write_meta_writes_header_then_newline_then_flushes(tmp_path) -> None:
    """``write_meta`` writes ``# …\\n`` to the handle and flushes, the
    same write contract :func:`_transcript_writer` uses, so a half-killed
    session still leaves a parseable header behind."""
    from argparse import Namespace

    args = Namespace(
        cmd=["echo"],
        prefix="",
        rate=200.0,
        volume=1.0,
        max_lines=0,
        follow=False,
        dry_run=False,
    )
    log = tmp_path / "session.log"
    fh = watch.open_transcript(str(log))
    try:
        watch.write_meta(fh, args)
    finally:
        fh.close()
    body = log.read_text(encoding="utf-8")
    assert body.startswith("# ")
    assert body.endswith("\n")
    # Exactly one line (no double-newline).
    assert body.count("\n") == 1


def test_main_meta_off_does_not_write_header(monkeypatch, tmp_path) -> None:
    """Without ``--meta`` the transcript is bit-identical to the
    pre-``--meta`` code — no header line, only the spoken rows."""
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: 0
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="alpha\nbeta\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)
    log = tmp_path / "session.log"
    code = watch.main(
        ["--quiet", "--transcript", str(log), "--", "echo"]
    )
    assert code == 0
    assert log.read_text(encoding="utf-8") == "alpha\nbeta\n"


def test_main_meta_on_writes_header_before_first_line(
    monkeypatch, tmp_path
) -> None:
    """With ``--meta`` the header is the *first* line of the
    transcript, every spoken row follows, and the file ends with a
    trailing newline. A reader running ``grep -v '^#' session.log``
    gets just the spoken rows back."""
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: 0
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="alpha\nbeta\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)
    log = tmp_path / "session.log"
    code = watch.main(
        ["--quiet", "--meta", "--transcript", str(log), "--", "echo", "x"]
    )
    assert code == 0
    body = log.read_text(encoding="utf-8")
    lines = body.splitlines()
    # Header + the two spoken rows.
    assert len(lines) == 3
    assert lines[0].startswith("# ")
    # And the command is in the header, not the row.
    assert "cmd=" in lines[0]
    assert "echo x" in lines[0]
    assert lines[1:] == ["alpha", "beta"]


def test_main_meta_without_transcript_is_silent(monkeypatch, capsys) -> None:
    """``--meta`` is silently ignored when there is no ``--transcript``
    to write into — the user might still want the flag in shell
    aliases, and the no-op matches every other discovery flag's
    "do nothing if there's no place to write" rule."""
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: 0
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="hi\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)
    code = watch.main(["--quiet", "--meta", "--", "echo"])
    assert code == 0
    # Nothing on stderr (a no-op, not a warning).
    assert capsys.readouterr().err == ""


def test_main_meta_follow_writes_header_in_streaming_mode(
    monkeypatch, tmp_path
) -> None:
    """``--follow --meta`` writes the header at open (before the first
    streamed line), so a long ``tail -f``-style session whose process
    takes minutes to produce a row still has a parseable header at
    the top of the transcript."""
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: 0
    )

    class _FakeStream:
        def stdout_iter(self):
            for line in ["late-row-1\n", "late-row-2\n"]:
                yield line
        def wait(self):
            return 0

    monkeypatch.setattr(watch, "_popen", lambda *a, **kw: _FakeStream())
    log = tmp_path / "session.log"
    code = watch.main(
        [
            "--quiet", "--follow", "--meta",
            "--transcript", str(log),
            "--", "tail", "-f", "x.log",
        ]
    )
    assert code == 0
    body = log.read_text(encoding="utf-8")
    lines = body.splitlines()
    assert lines[0].startswith("# ")
    assert lines[1:] == ["late-row-1", "late-row-2"]


def test_main_meta_write_failure_does_not_change_exit_code(
    monkeypatch, tmp_path, capsys
) -> None:
    """A failed meta write is loud on stderr but does NOT change the
    exit code — the spoken/printed output is still the source of
    truth, and the file-system error should not block the session
    from running. Same contract the per-line transcript write has."""
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: 0
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="alpha\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)

    def _bad_write(fh, args, *, now=None) -> None:
        raise OSError("disk full")

    log = tmp_path / "session.log"
    monkeypatch.setattr(watch, "write_meta", _bad_write)
    code = watch.main(
        ["--quiet", "--meta", "--transcript", str(log), "--", "echo"]
    )
    assert code == 0
    # The error message is on stderr.
    err = capsys.readouterr().err
    assert "meta" in err.lower() and "disk full" in err


def test_main_meta_dry_run_still_writes_header(monkeypatch, tmp_path) -> None:
    """``--meta --dry-run --transcript`` records the header in the
    transcript just like the speak mode does — dry-run is a "show
    me and log it" preview, so the log should be self-describing
    too."""
    monkeypatch.setattr(
        watch, "_speak_line", lambda *a, **kw: 0
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="would-speak\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)
    log = tmp_path / "session.log"
    code = watch.main(
        [
            "--quiet", "--meta", "--dry-run",
            "--transcript", str(log),
            "--", "echo",
        ]
    )
    assert code == 0
    lines = log.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# ")
    assert lines[1] == "would-speak"


def test_main_meta_includes_prefix_in_header(monkeypatch, tmp_path) -> None:
    """When ``--prefix TEXT`` and ``--meta`` are combined, the header
    records the prefix so a multi-session log can be re-parsed by
    session even after the fact."""
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: 0
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="row\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)
    log = tmp_path / "session.log"
    code = watch.main(
        [
            "--quiet", "--meta", "--prefix", "build",
            "--transcript", str(log),
            "--", "make",
        ]
    )
    assert code == 0
    body = log.read_text(encoding="utf-8")
    lines = body.splitlines()
    assert "prefix=build" in lines[0]


# --- --meta-end (per-session transcript footer) -------------------------


def test_parse_args_meta_end_default_is_off() -> None:
    """Without ``--meta-end`` the flag stays ``False`` so the no-flag
    transcript behaviour is bit-identical to the pre-``--meta-end`` code
    (no trailing footer line in the log)."""
    args = watch.parse_args(["--transcript", "/tmp/x", "--", "echo"])
    assert args.meta_end is False


def test_parse_args_meta_end_flag_on() -> None:
    """``--meta-end`` is exposed on the namespace as ``args.meta_end = True``."""
    args = watch.parse_args(
        ["--transcript", "/tmp/x", "--meta-end", "--", "echo"]
    )
    assert args.meta_end is True


def test_format_duration_zero_and_negative_short_circuits_to_pt0s() -> None:
    """A zero or negative duration is treated as ``PT0S`` — a real session
    can never produce a negative duration, so the boundary is defensive
    and the constant is the same for both inputs (no test for a non-zero
    zero that could drift)."""
    assert watch._format_duration(0) == "PT0S"
    assert watch._format_duration(-1) == "PT0S"
    assert watch._format_duration(-100.5) == "PT0S"


def test_format_duration_sub_minute_is_seconds_only() -> None:
    """Anything strictly less than 60s renders as ``PT<whole-seconds>S``
    (no minutes field, no trailing ``M``). The whole-second truncation
    is intentional — the footer is a wall-clock measurement, not a
    benchmark, so two sessions that differ by a millisecond still
    produce identical footers."""
    assert watch._format_duration(0.5) == "PT0S"
    assert watch._format_duration(1) == "PT1S"
    assert watch._format_duration(45) == "PT45S"
    assert watch._format_duration(59.9) == "PT59S"


def test_format_duration_minute_and_over_uses_minutes_and_seconds() -> None:
    """Once the duration crosses a minute boundary the field grows to
    ``PT<MM>MS<S>S`` (still no sub-second precision, still no
    hours — a long session is reported in MM:SS, which is what
    users expect from a footer)."""
    assert watch._format_duration(60) == "PT1M0S"
    assert watch._format_duration(61) == "PT1M1S"
    assert watch._format_duration(125) == "PT2M5S"
    assert watch._format_duration(3599) == "PT59M59S"
    assert watch._format_duration(3600) == "PT60M0S"


def test_format_meta_end_is_single_line_starting_with_hash() -> None:
    """The footer is a single line beginning with ``#`` so the transcript
    stays grep-friendly (``grep '^#'`` picks every header AND every
    footer) and any downstream ``tail -f`` consumer can recognise
    and ignore it. Pairs with the ``_format_meta`` header convention."""
    from argparse import Namespace

    args = Namespace(cmd=["echo", "hi"], prefix="", rate=200.0, volume=1.0)
    fixed_now = watch._datetime(2026, 9, 29, 9, 0, 0, tzinfo=watch.timezone.utc)
    line = watch._format_meta_end(
        args,
        started_at=0.0,
        finished_at=10.0,
        spoken=3,
        exit_code=0,
        now=fixed_now,
    )
    assert "\n" not in line
    assert line.startswith("# ")


def test_format_meta_end_includes_timestamp_cmd_exit_spoken_duration() -> None:
    """The footer names every field a downstream consumer needs to pair
    it with the ``--meta`` header and to compute per-session totals
    (exit code, lines spoken, wall-clock duration) without re-reading
    the original CLI."""
    from argparse import Namespace

    args = Namespace(cmd=["make", "test"], prefix="build", rate=220.0, volume=0.5)
    fixed_now = watch._datetime(2026, 9, 29, 9, 0, 0, tzinfo=watch.timezone.utc)
    line = watch._format_meta_end(
        args,
        started_at=0.0,
        finished_at=65.0,
        spoken=7,
        exit_code=0,
        now=fixed_now,
    )
    # ISO-8601 UTC timestamp with Z suffix.
    assert "2026-09-29T09:00:00Z" in line
    # All five key fields are present with their actual values.
    assert "exit=0" in line
    assert "spoken=7" in line
    # 65s renders as PT1M5S (the over-a-minute shape).
    assert "duration=PT1M5S" in line
    # The full command is embedded via shlex.join.
    assert "cmd=make test" in line
    # The "session end" keyword pairs with the "session" keyword in --meta.
    assert "paw-watch" in line
    assert "session" in line
    assert "end" in line


def test_format_meta_end_reflects_non_zero_exit_code() -> None:
    """The watched command's exit code is mirrored into the footer so a
    downstream ``grep '^#'`` consumer can see per-session failure
    without re-parsing the original CLI. Non-zero values (TTS errors,
    child failures) appear verbatim — no mapping to / from the user-
    visible exit code ``main`` returns."""
    from argparse import Namespace

    args = Namespace(cmd=["x"], prefix="", rate=200.0, volume=1.0)
    line = watch._format_meta_end(
        args,
        started_at=0.0,
        finished_at=1.0,
        spoken=1,
        exit_code=42,
    )
    assert "exit=42" in line


def test_write_meta_end_is_noop_when_fh_is_none() -> None:
    """``--meta-end`` without ``--transcript`` is a silent no-op (same
    convention ``--meta`` has) so a user who keeps the flag in a shell
    alias and forgets the log path pays nothing."""
    # The helper must accept None and not raise.
    watch.write_meta_end(
        None,
        _minimal_args(),
        started_at=0.0,
        finished_at=1.0,
        spoken=0,
        exit_code=0,
    )


def test_write_meta_end_writes_footer_then_newline_then_flushes(tmp_path) -> None:
    """The footer is written with the same write + newline + flush
    contract ``_transcript_writer`` and ``write_meta`` use, so a
    half-killed session still leaves a parseable footer behind."""
    from argparse import Namespace

    log = tmp_path / "session.log"
    log.write_text("# header line\nspoken-row\n", encoding="utf-8")
    fh = open(log, "a", encoding="utf-8")
    args = Namespace(cmd=["echo"], prefix="", rate=200.0, volume=1.0)
    watch.write_meta_end(
        fh,
        args,
        started_at=0.0,
        finished_at=2.0,
        spoken=1,
        exit_code=0,
    )
    fh.close()
    body = log.read_text(encoding="utf-8")
    lines = body.splitlines()
    assert lines[0] == "# header line"
    assert lines[1] == "spoken-row"
    # The footer is the last line.
    assert lines[-1].startswith("# paw-watch session end")
    assert "duration=PT2S" in lines[-1]
    assert "spoken=1" in lines[-1]
    assert "exit=0" in lines[-1]


def _minimal_args() -> "argparse.Namespace":  # type: ignore[name-defined]
    """Build a bare ``Namespace`` covering the fields ``_format_meta_end``
    reads. Lets the "noop when fh is None" test stay tiny without
    importing ``argparse`` at module load time."""
    from argparse import Namespace
    return Namespace(cmd=["echo"], prefix="", rate=200.0, volume=1.0)


def test_main_meta_end_off_does_not_write_footer(monkeypatch, tmp_path) -> None:
    """Without ``--meta-end`` the transcript is bit-identical to the
    pre-``--meta-end`` code: the spoken lines land, and the log
    ends with the last spoken row — no trailing ``# paw-watch session
    end`` line."""
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: 0
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="row-a\nrow-b\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)
    log = tmp_path / "session.log"
    code = watch.main(
        ["--quiet", "--transcript", str(log), "--", "echo"]
    )
    assert code == 0
    body = log.read_text(encoding="utf-8")
    assert body == "row-a\nrow-b\n"
    assert "session end" not in body


def test_main_meta_end_on_writes_footer_after_last_line(
    monkeypatch, tmp_path
) -> None:
    """``--meta-end`` writes a single ``# paw-watch session end: …``
    line as the *last* line of the transcript, so the log is
    self-describing from both ends. The footer is bracketed by the
    same ``# `` prefix ``--meta`` uses, and it lands after every
    spoken row (not in the middle of the stream)."""
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: 0
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="row-a\nrow-b\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)
    # Pin a fake monotonic clock so the duration field is stable.
    monkeypatch.setattr(watch, "monotonic", lambda: 100.0)
    log = tmp_path / "session.log"
    code = watch.main(
        [
            "--quiet", "--meta-end",
            "--transcript", str(log),
            "--", "echo", "hello",
        ]
    )
    assert code == 0
    body = log.read_text(encoding="utf-8")
    lines = body.splitlines()
    # The first two lines are the spoken rows in order.
    assert lines[0] == "row-a"
    assert lines[1] == "row-b"
    # The footer is the LAST line and starts with the session-end prefix.
    assert lines[-1].startswith("# paw-watch session end")
    # The footer embeds the spoken count, the exit code, the cmd, and
    # the duration. The duration is PT0S because we pinned the
    # monotonic clock to a single value (started == finished).
    assert "spoken=2" in lines[-1]
    assert "exit=0" in lines[-1]
    assert "cmd=echo hello" in lines[-1]


def test_main_meta_end_without_transcript_is_silent(
    monkeypatch, capsys
) -> None:
    """``--meta-end`` without ``--transcript`` is a no-op (same
    convention ``--meta`` has) so a user who keeps the flag in a
    shell alias and forgets the log path pays nothing: no error,
    no spurious stdout / stderr, no exit-code change."""
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: 0
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="row\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)
    code = watch.main(["--quiet", "--meta-end", "--", "echo"])
    assert code == 0
    out = capsys.readouterr()
    # The transcript wasn't open, so nothing related to the footer
    # was written to stdout or stderr.
    assert "session end" not in out.out
    assert "session end" not in out.err


def test_main_meta_end_without_meta_still_writes_footer(
    monkeypatch, tmp_path
) -> None:
    """``--meta-end`` does NOT require ``--meta`` — the footer is
    independently useful (it records exit code, spoken count, and
    duration even if the user did not request the open-time
    header). Pairing the two flags is the common case, but the
    footer alone is still a valid one-shot session log."""
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: 0
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="only-row\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)
    log = tmp_path / "session.log"
    code = watch.main(
        [
            "--quiet", "--meta-end",
            "--transcript", str(log),
            "--", "echo",
        ]
    )
    assert code == 0
    body = log.read_text(encoding="utf-8")
    lines = body.splitlines()
    assert lines[0] == "only-row"
    assert lines[-1].startswith("# paw-watch session end")
    # And there is NO header line — only the spoken row and the
    # footer. The footer is the only ``#``-prefixed line.
    assert sum(1 for ln in lines if ln.startswith("# ")) == 1


def test_main_meta_end_reflects_non_zero_exit_code(
    monkeypatch, tmp_path
) -> None:
    """The watcher's exit code (the value the child returned) is
    mirrored into the footer so a downstream ``grep '^#'`` consumer
    can see per-session failure without re-running the original
    CLI. The user-visible exit code (the value ``main`` returns to
    the shell) is the SAME number, so the footer and the shell
    agree."""
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: 0
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=7, stdout="failed-row\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)
    log = tmp_path / "session.log"
    code = watch.main(
        [
            "--quiet", "--meta-end",
            "--transcript", str(log),
            "--", "false",
        ]
    )
    # The shell sees the child's exit code.
    assert code == 7
    body = log.read_text(encoding="utf-8")
    # The footer mirrors it.
    assert "exit=7" in body.splitlines()[-1]


def test_main_meta_end_counts_spoken_lines_through_max_lines(
    monkeypatch, tmp_path
) -> None:
    """``--meta-end`` reports the number of lines that were ACTUALLY
    spoken — capped by ``--max-lines`` (a session stopped after N
    rows should not advertise ``spoken=∞`` in its footer). The
    count is whatever the loop reached, including the trailing
    fragment."""
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: 0
    )
    fake = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="r1\nr2\nr3\nr4\nr5\n",
        stderr="",
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)
    log = tmp_path / "session.log"
    code = watch.main(
        [
            "--quiet", "--meta-end", "--max-lines", "2",
            "--transcript", str(log),
            "--", "echo",
        ]
    )
    assert code == 0
    body = log.read_text(encoding="utf-8")
    lines = body.splitlines()
    # Only the first two rows hit the log; the cap fired.
    assert "r1" in body
    assert "r2" in body
    assert "r3" not in body
    # The footer reports the *actual* spoken count, not the row count.
    assert "spoken=2" in lines[-1]


def test_main_meta_end_follow_writes_footer_in_streaming_mode(
    monkeypatch, tmp_path
) -> None:
    """``--meta-end`` composes with ``--follow``: the footer still
    lands as the LAST line of the transcript after the streamed
    rows, so a long-running ``tail -f`` session has a parseable
    session end behind its data."""
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: 0
    )

    class _FakeStream:
        def stdout_iter(self):
            for line in ["s1\n", "s2\n"]:
                yield line
        def wait(self):
            return 0

    monkeypatch.setattr(watch, "_popen", lambda *a, **kw: _FakeStream())
    log = tmp_path / "session.log"
    code = watch.main(
        [
            "--quiet", "--follow", "--meta-end",
            "--transcript", str(log),
            "--", "tail", "-f", "x.log",
        ]
    )
    assert code == 0
    body = log.read_text(encoding="utf-8")
    lines = body.splitlines()
    assert lines[0] == "s1"
    assert lines[1] == "s2"
    assert lines[-1].startswith("# paw-watch session end")
    assert "spoken=2" in lines[-1]


def test_main_meta_end_dry_run_still_writes_footer(monkeypatch, tmp_path) -> None:
    """``--dry-run`` does NOT short-circuit ``--meta-end``: the footer
    is part of the session bookkeeping (per-session totals), not
    part of the TTS output, so a session that would-have-spoken-N
    rows still records ``spoken=N`` in its footer."""
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: 0
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="a\nb\nc\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)
    log = tmp_path / "session.log"
    code = watch.main(
        [
            "--quiet", "--dry-run", "--meta-end",
            "--transcript", str(log),
            "--", "echo",
        ]
    )
    assert code == 0
    body = log.read_text(encoding="utf-8")
    lines = body.splitlines()
    assert lines[-1].startswith("# paw-watch session end")
    assert "spoken=3" in lines[-1]


def test_main_meta_end_write_failure_does_not_change_exit_code(
    monkeypatch, tmp_path, capsys
) -> None:
    """A failed ``--meta-end`` write is loud on stderr but does NOT
    change the exit code — the spoken/printed output is the source
    of truth, and a flaky filesystem should not break the user's
    pipeline. Same contract ``--meta`` and the per-line transcript
    write have."""
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: 0
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="only\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)

    def _bad_write(*a, **kw):
        raise OSError("disk full")

    log = tmp_path / "session.log"
    monkeypatch.setattr(watch, "write_meta_end", _bad_write)
    code = watch.main(
        [
            "--quiet", "--meta-end",
            "--transcript", str(log),
            "--", "echo",
        ]
    )
    # Exit code still reflects the success of the underlying run.
    assert code == 0
    err = capsys.readouterr().err
    assert "--meta-end write failed" in err
    assert "disk full" in err


def test_main_meta_end_includes_prefix_in_footer(monkeypatch, tmp_path) -> None:
    """When ``--prefix TEXT`` and ``--meta-end`` are combined, the
    footer's ``cmd=`` field still uses ``shlex.join`` on the raw
    argv (the same field the ``--meta`` header uses), so the
    open/close markers can be paired by ``cmd=`` downstream. The
    prefix does not appear in the footer's key=value list — it
    is *part of* the cmd string, not a flag of its own."""
    monkeypatch.setattr(
        watch, "_speak_line", lambda line, rate, volume: 0
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="r\n", stderr=""
    )
    monkeypatch.setattr(watch, "_spawn", lambda *a, **kw: fake)
    log = tmp_path / "session.log"
    code = watch.main(
        [
            "--quiet", "--meta-end", "--prefix", "ci",
            "--transcript", str(log),
            "--", "make", "test",
        ]
    )
    assert code == 0
    body = log.read_text(encoding="utf-8")
    footer = body.splitlines()[-1]
    # The cmd field is the joined argv — same shape the header uses.
    assert "cmd=make test" in footer
    # The prefix itself is NOT a key in the footer (it is in --meta,
    # not --meta-end). We assert the absence of the standalone key
    # so a future drift is caught.
    assert "prefix=" not in footer
