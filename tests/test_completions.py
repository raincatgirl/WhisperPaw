"""Tests for the shell-completion generator.

We cover:

* ``list_tools()`` returns the expected subcommand list
* ``render(shell)`` produces non-empty output for every supported shell
* The output of each renderer contains the tool name and its known
  flags (so the user actually gets something useful at the prompt)
* ``render(shell)`` is deterministic — same input, same output
* The shipped static files under ``completions/`` are byte-identical
  to the live generator output (this is the test that fails when a
  new flag is added and the static files are forgotten)
* ``paw-complete`` CLI entry point: unknown shell exits 2, known
  shell prints the same text as ``render()``
"""
from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import pytest

from whisperpaw import _completions
from whisperpaw.complete import main as complete_main


#: Path to the directory holding the shipped static completion files.
COMPLETIONS_DIR: Path = (
    Path(__file__).resolve().parent.parent / "src" / "whisperpaw" / "completions"
)


# ---------------------------------------------------------------------------
# list_tools / render
# ---------------------------------------------------------------------------


def test_list_tools_includes_every_shipped_cli() -> None:
    tools = set(_completions.list_tools())
    # Every console_scripts entry in pyproject.toml must have a
    # completion entry — except paw-complete itself, which is the
    # generator and doesn't need its own completion.
    expected = {"paw-read", "paw-sound", "paw-watch", "paw-zoom"}
    assert expected <= tools, f"missing: {expected - tools}"


def test_supported_shells_are_the_expected_four() -> None:
    # Bash, zsh, fish, nushell. Don't expand this set without a real
    # user-facing reason.
    assert _completions.SUPPORTED_SHELLS == ("bash", "zsh", "fish", "nu")


@pytest.mark.parametrize("shell", _completions.SUPPORTED_SHELLS)
def test_render_produces_nonempty_text_for_each_shell(shell: str) -> None:
    out = _completions.render(shell)
    assert out.strip(), f"render({shell!r}) returned empty text"
    # Every output mentions every tool we ship.
    for tool in _completions.list_tools():
        assert tool in out, f"render({shell!r}) missing tool {tool!r}"


@pytest.mark.parametrize("shell", _completions.SUPPORTED_SHELLS)
def test_render_is_deterministic(shell: str) -> None:
    # The header mentions a date by design? No — we don't; if we ever
    # do, this test will catch it.
    a = _completions.render(shell)
    b = _completions.render(shell)
    assert a == b


def test_render_unknown_shell_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unsupported shell"):
        _completions.render("powershell")  # type: ignore[arg-type]


def test_render_all_returns_one_entry_per_shell() -> None:
    out = _completions.render_all()
    assert set(out.keys()) == set(_completions.SUPPORTED_SHELLS)
    for shell, text in out.items():
        assert text == _completions.render(shell)


# ---------------------------------------------------------------------------
# Per-shell content checks
# ---------------------------------------------------------------------------


def test_bash_output_contains_every_paw_read_flag() -> None:
    text = _completions.render("bash")
    # The bash complete function is built from the same flag list, so
    # every long flag should appear literally in the output.
    for flag in ("--rate", "--volume", "--max-chars", "--backend",
                 "--piper-voice", "--file", "--clipboard", "--quiet"):
        assert flag in text, f"bash output missing {flag}"


def test_zsh_output_uses_compdef() -> None:
    text = _completions.render("zsh")
    assert "#compdef paw-read" in text
    assert "#compdef paw-sound" in text
    assert "#compdef paw-watch" in text
    # The zsh renderer uses _arguments with --flag[help] form.
    assert "--rate[" in text
    assert "--max-lines[" in text


def test_fish_output_uses_complete_c() -> None:
    text = _completions.render("fish")
    assert "complete -c paw-read -l rate" in text
    assert "complete -c paw-sound -l pack" in text
    assert "complete -c paw-watch -l max-lines" in text


def test_nu_output_is_external_completer_style() -> None:
    text = _completions.render("nu")
    # nushell declares external completers as `extern "name" [ --flag: type ]`.
    # These are not callable functions; they describe the binary's
    # flags to nushell's completion engine.
    assert 'extern "paw-read"' in text
    assert 'extern "paw-sound"' in text
    assert 'extern "paw-watch"' in text
    # Each flag is declared as `--<name>: string`.
    assert "--rate: string" in text
    assert "--max-lines: string" in text
    # The zoom stub still emits an extern declaration so sourcing the
    # file doesn't break.
    assert 'extern "paw-zoom"' in text


def test_paw_zoom_renders_a_real_entry() -> None:
    # paw-zoom now has a real build_parser(), so its completion entry
    # is generated from the parser — every long flag should appear in
    # at least one of the four shell renderings.
    for shell in _completions.SUPPORTED_SHELLS:
        text = _completions.render(shell)
        assert "paw-zoom" in text
        # A known flag from the parser should make it into the output.
        if shell == "nu":
            assert "--zoom" in text or "--rows" in text
        else:
            assert "--zoom" in text or "--rows" in text


# ---------------------------------------------------------------------------
# Sync with shipped static files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shell,filename", [
    ("bash", "whisperpaw.bash"),
    ("zsh", "whisperpaw.zsh"),
    ("fish", "whisperpaw.fish"),
    ("nu", "whisperpaw.nu"),
])
def test_shipped_static_file_matches_live_render(
    shell: str, filename: str
) -> None:
    """The static file under ``completions/`` must be byte-identical
    to the live generator output. If this fails, regenerate with::

        python -c "from whisperpaw._completions import render; \\
                   import sys; sys.stdout.write(render('<shell>'))" \\
            > completions/whisperpaw.<shell>
    """
    path = COMPLETIONS_DIR / filename
    assert path.is_file(), f"missing shipped file: {path}"
    on_disk = path.read_text(encoding="utf-8")
    generated = _completions.render(shell)
    assert on_disk == generated, (
        f"completions/{filename} is out of sync with the live generator. "
        f"Regenerate it."
    )


# ---------------------------------------------------------------------------
# paw-complete CLI
# ---------------------------------------------------------------------------


def test_paw_complete_unknown_shell_exits_2() -> None:
    # argparse's choices= handles this and raises SystemExit(2).
    code = complete_main(["powershell"])
    assert code == 2


def test_paw_complete_no_args_exits_2() -> None:
    code = complete_main([])
    assert code == 2


def test_paw_complete_known_shell_prints_to_stdout(capsys: pytest.CaptureFixture) -> None:
    code = complete_main(["bash"])
    out = capsys.readouterr()
    assert code == 0
    assert "paw-read" in out.out
    assert "paw-sound" in out.out
    assert "paw-watch" in out.out
    # Should be the same as render('bash').
    assert out.out == _completions.render("bash")


def test_paw_complete_list_tools_prints_names(capsys: pytest.CaptureFixture) -> None:
    code = complete_main(["--list-tools"])
    out = capsys.readouterr()
    assert code == 0
    names = out.out.strip().splitlines()
    assert "paw-read" in names
    assert "paw-sound" in names
    assert "paw-watch" in names
    assert "paw-zoom" in names
    # paw-complete is the generator; it does not need a completion entry.
    assert "paw-complete" not in names


# ---------------------------------------------------------------------------
# Shell syntax sanity (cheap, but catches obvious typos)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="bash not always available on Windows test runners",
)
def test_bash_completion_parses_with_bash_if_available() -> None:
    """If bash is on PATH, the generated file must be syntactically
    valid (no unbalanced braces, no unterminated strings). This is a
    cheap catch for renderer bugs that would only surface when the
    user tries to source the file."""
    bash = _which("bash")
    if bash is None:
        pytest.skip("bash not installed")
    text = _completions.render("bash")
    # ``bash -n`` parses but does not execute; exit 0 == no syntax error.
    result = subprocess.run(
        [bash, "-n"],
        input=text,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"bash -n rejected the generated completion:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_bash_output_has_balanced_braces() -> None:
    """A bare-bones structural check that doesn't need bash installed.
    Catches obvious ``{`` / ``}`` mismatches in the renderer."""
    text = _completions.render("bash")
    assert text.count("{") == text.count("}"), (
        "unbalanced braces in bash output: "
        f"{text.count('{')} open vs {text.count('}')} close"
    )


def _which(name: str) -> str | None:
    """Tiny ``shutil.which`` replacement — avoids the import at module
    top so the rest of the test module stays stdlib-light."""
    import shutil
    return shutil.which(name)
