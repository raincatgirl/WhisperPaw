# 🛣️ WhisperPaw Roadmap

This document is the **single source of truth** for what WhisperPaw will be and in what order things ship. Every cron tick should leave a tiny update here so progress is visible.

---

## 🎯 North star

> Make the terminal a place where everyone can hear, see, and feel what's happening — without staring at a wall of text.

---

## 🐾 Pillars

1. **Hear** — audio feedback (TTS + sound events)
2. **See** — magnification + visual clarity
3. **Touch** — gentle haptics where supported
4. **Compose** — each tool small enough to pipe

---

## 📦 Tools

| Tool | Status | Description |
| --- | :---: | --- |
| `paw-sound` | 🐣 planned | Play short audio cues for shell events. |
| `paw-read`  | 🐣 planned | Read text aloud (stdin / file / clipboard). |
| `paw-zoom`  | 🐣 planned | Magnify area around the cursor. |
| `paw-watch` | 🐣 planned | Tail a command's output and read new lines. |

Legend: 🐣 planned · 🛠 in progress · ✅ shipped · 🐛 buggy

---

## 📅 Tick log

A new entry is appended every time the cron job wakes up. This is the project's heartbeat.

<!-- TICK-LOG-START -->
- 2026-09-13 — scaffold tick: package + CLI stubs + 2 tests green on py3.11–3.13, GitHub Actions wired. See `DEVLOG.md`.
<!-- TICK-LOG-END -->
