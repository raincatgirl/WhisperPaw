"""Regenerate the four static shell-completion files from the live parsers.

Run after any tool's CLI surface changes. Idempotent: a no-op tick
exits 0 with no output. Bytes are written atomically (tmp + rename)
so a half-written file can't break a running Tab completion.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from whisperpaw import _completions


def main() -> int:
    out = Path(__file__).resolve().parent.parent / "src" / "whisperpaw" / "completions"
    out.mkdir(parents=True, exist_ok=True)
    for shell, filename in (
        ("bash", "whisperpaw.bash"),
        ("zsh",  "whisperpaw.zsh"),
        ("fish", "whisperpaw.fish"),
        ("nu",   "whisperpaw.nu"),
    ):
        text = _completions.render(shell)
        target = out / filename
        # Write to a sibling tmp file then rename so a half-written
        # file can never be observed by a Tab-completion read.
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=out, delete=False
        ) as tmp:
            tmp.write(text)
            tmp_path = Path(tmp.name)
        tmp_path.replace(target)
        print(f"regenerated {shell} -> {target} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
