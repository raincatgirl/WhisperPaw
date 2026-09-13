# 🐾 WhisperPaw

> *a terminal accessibility toolkit — read aloud, sound feedback, gentle zoom.*

WhisperPaw is a small collection of command-line tools that make the terminal kinder to use, especially for people who benefit from audio feedback, magnification, or hands-free operation.

```
       ∧＿∧
      ( • ω• )   meow?
     / >  ◇  \
    /  __∪__  \
   /  /    \  \
  (__/      \__)
```

## ✨ What's inside

| Command | What it does | Status |
| --- | --- | :---: |
| `paw-sound` | Audio feedback for shell events: `meow` on success, `mrrp` on warning, `hiss` on error. | ✅ shipped |
| `paw-read` | Reads selected text aloud via TTS (cross-platform). | 🐣 planned |
| `paw-zoom` | Magnifies a chosen area of the screen (high-DPI helper). | 🐣 planned |
| `paw-watch` | Watches a command's output and reads new lines aloud. | 🐣 planned |

## 🚀 Install

```bash
pip install whisperpaw
```

> WhisperPaw is a single Python package. Each subcommand (`paw-read`, `paw-sound`, `paw-zoom`, `paw-watch`) is also runnable on its own.

## 🐱 Quick start

```bash
# Read the clipboard aloud
paw-read

# Play an "ok" sound after a successful build
make && paw-sound ok || paw-sound fail

# Magnify the screen around your mouse cursor
paw-zoom
```

## 🧶 Design principles

- 🐾 **Tiny** — each tool fits in one file.
- 🐾 **Quiet** — no telemetry, no noise, no notifications you didn't ask for.
- 🐾 **Kind** — works for keyboard-only, screen-reader, and low-vision users.
- 🐾 **Portable** — pure Python 3.11+, runs on macOS / Linux / Windows.
- 🐾 **Composable** — pipe tools together with normal shell operators.

## 🛣️ Roadmap

- [x] Repo scaffold + README
- [ ] `paw-sound` — first concrete tool
- [ ] `paw-read` — TTS wrapper
- [ ] `paw-zoom` — screen magnifier
- [ ] `paw-watch` — tail-and-read
- [ ] Sound pack (cat / forest / rain / keyboard)
- [ ] Shell completions (bash / zsh / fish / nushell)
- [ ] Homebrew formula + pip release

See [`ROADMAP.md`](./ROADMAP.md) for the full plan and per-tick progress.

## 🤝 Contributing

Soft paws welcome. Open an issue first if you'd like to add a new tool.

## 📄 License

MIT — see [`LICENSE`](./LICENSE).
