# 📓 WhisperPaw Dev Log

> A short, dated journal of what was built in each tick. Keep entries tiny — one paragraph or a bullet list is plenty.

---

## 2026-09-13 — repo scaffold

- Created `raincatgirl/WhisperPaw` (public, MIT).
- Added `README.md`, `ROADMAP.md`, `LICENSE`, `pyproject.toml`, `.gitignore`, GitHub Actions.
- Stubbed four entry points (`paw-read`, `paw-sound`, `paw-zoom`, `paw-watch`) so `pip install -e .` works and CLI shims exist.
- Wrote two skeleton tests, fixed one bug found by them (`__init__.py` didn't import the submodules).
- Status: package installs, CLI prints a friendly "not implemented yet" line, pytest is green on 3.11–3.13.
- Next tick: pick the easiest concrete tool to ship — leaning towards `paw-sound` (no platform deps).
