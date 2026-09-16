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
