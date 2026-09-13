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
| `paw-sound` | Audio feedback for shell events: `meow` on success, `mrrp` on warning, `hiss` on error. Ships with two packs: `cat` (synth tones) and `forest` (layered ambient). | ✅ shipped |
| `paw-read` | Reads text aloud via TTS. Accepts a positional arg, `--file`, `--clipboard`, or stdin. Backends: macOS `say` / Linux `spd-say`/`espeak` / Windows SAPI, plus optional **Piper** for high-quality local neural voices. | ✅ shipped |
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
paw-read --clipboard

# Read a file aloud at a faster rate
paw-read --rate 280 --file notes.md

# High-quality neural voice via Piper (auto-discover ~/.local/share/piper/voices)
paw-read --backend piper "this sounds way more natural"

# Or point Piper at a specific voice model
paw-read --backend piper --piper-voice ~/voices/en_US-amy-low.onnx "specific voice"

# Play an "ok" sound after a successful build
make && paw-sound ok || paw-sound fail

# Use a softer forest pack instead of the default cat pack
make && paw-sound --pack forest ok

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
- [x] `paw-sound` — first concrete tool (cat pack)
- [x] `paw-sound` forest pack — second sound pack, same API
- [x] `paw-read` — TTS via OS engine (say / spd-say / espeak / SAPI)
- [ ] `paw-zoom` — screen magnifier
- [ ] `paw-watch` — tail-and-read (will build on paw-read)
- [ ] Sound pack: `rain` / `keyboard` (only if forest feels good)
- [ ] Shell completions (bash / zsh / fish / nushell)
- [ ] Homebrew formula + pip release

See [`ROADMAP.md`](./ROADMAP.md) for the full plan and per-tick progress.

## 🤝 Contributing

Soft paws welcome. Open an issue first if you'd like to add a new tool.

## 📄 License

MIT — see [`LICENSE`](./LICENSE).
