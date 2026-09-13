"""Generate the 'forest' sound pack — a small ambient wood at dusk.

Unlike the 'cat' pack (which is short, percussive synth tones), the
'forest' pack layers noise + slow LFOs to evoke ambience: low wind,
a soft bird, a cricket pulse, a distant bell, a low warning hum.

Pure stdlib, no third-party deps. Run this once to (re)generate the
files; the runtime never depends on this script.
"""
from __future__ import annotations

import math
import random
import struct
import wave
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "sounds" / "forest"
SAMPLE_RATE = 22050

# Each event: (duration_s, base_amplitude, layers)
# layers are a list of (callable(t, total) -> float sample_contribution).
EVENTS: dict[str, tuple[float, float, list]] = {}


def _envelope(t: float, total: float, attack: float = 0.08, release: float = 0.18) -> float:
    if t < attack:
        return t / attack
    if t > total - release:
        return max(0.0, (total - t) / release)
    return 1.0


def _wind_layer(intensity: float, lfo_hz: float = 0.4):
    """A slow, low-pass-ish noise layer (cheap: smoothed white noise)."""

    def make(total: float) -> list[float]:
        rng = random.Random(0xC0FFEE ^ int(intensity * 1000))
        n = int(SAMPLE_RATE * total)
        out = [0.0] * n
        prev = 0.0
        # crude 1-pole low-pass: y = y + alpha * (x - y)
        alpha = 0.04 + 0.06 * intensity
        for i in range(n):
            x = rng.uniform(-1.0, 1.0)
            prev = prev + alpha * (x - prev)
            out[i] = prev
        return out

    return make


def _bird_chirp_layer(when: float, freq: float, dur: float = 0.12):
    """A short, bright chirp centered at `when` seconds."""

    def make(total: float) -> list[float]:
        out = [0.0] * int(SAMPLE_RATE * total)
        start = int(when * SAMPLE_RATE)
        end = min(len(out), start + int(dur * SAMPLE_RATE))
        for i in range(start, end):
            t = (i - start) / SAMPLE_RATE
            local = t / dur
            env = math.sin(math.pi * local)  # bell shape
            # Slight downward chirp — feels more birdlike than a pure tone.
            f = freq * (1.0 - 0.15 * local)
            out[i] = 0.7 * env * math.sin(2.0 * math.pi * f * t)
        return out

    return make


def _cricket_layer(pulse_hz: float = 6.0):
    """Soft cricket pulse — a slowly modulated high tone with gaps."""

    def make(total: float) -> list[float]:
        out = [0.0] * int(SAMPLE_RATE * total)
        for i in range(len(out)):
            t = i / SAMPLE_RATE
            # pulse envelope: sharp on, slow off, with a long silent gap
            phase = (t * pulse_hz) % 1.0
            pulse = math.exp(-6.0 * phase) if phase < 0.18 else 0.0
            # add a tiny silent gap most of the time
            gap = 1.0 if (int(t * pulse_hz) % 3 != 0) else 0.0
            out[i] = 0.18 * pulse * gap * math.sin(2.0 * math.pi * 4200.0 * t)
        return out

    return make


def _bell_layer(when: float, freq: float, dur: float = 0.9):
    """A soft bell-like tone with two harmonics and exponential decay."""

    def make(total: float) -> list[float]:
        out = [0.0] * int(SAMPLE_RATE * total)
        start = int(when * SAMPLE_RATE)
        end = min(len(out), start + int(dur * SAMPLE_RATE))
        for i in range(start, end):
            t = (i - start) / SAMPLE_RATE
            env = math.exp(-3.5 * t)
            out[i] = env * (
                0.6 * math.sin(2.0 * math.pi * freq * t)
                + 0.25 * math.sin(2.0 * math.pi * freq * 2.01 * t)
            )
        return out

    return make


def _hum_layer(freq: float):
    """A steady low hum — useful for 'warn' / 'fail' where you want unease."""

    def make(total: float) -> list[float]:
        out = [0.0] * int(SAMPLE_RATE * total)
        for i in range(len(out)):
            t = i / SAMPLE_RATE
            env = _envelope(t, total, attack=0.05, release=0.10)
            out[i] = 0.35 * env * (
                math.sin(2.0 * math.pi * freq * t)
                + 0.3 * math.sin(2.0 * math.pi * freq * 1.5 * t)
            )
        return out

    return make


# Populate EVENTS table -------------------------------------------------

# ok — soft wind + one distant bird + light cricket
EVENTS["ok"] = (
    0.9, 0.55,
    [
        _wind_layer(0.35),
        _bird_chirp_layer(0.20, 1800.0, dur=0.10),
        _bird_chirp_layer(0.55, 2200.0, dur=0.08),
        _cricket_layer(pulse_hz=4.0),
    ],
)

# warn — wind louder, a low hum starts underneath
EVENTS["warn"] = (
    1.2, 0.55,
    [
        _wind_layer(0.55),
        _hum_layer(110.0),
        _cricket_layer(pulse_hz=5.0),
    ],
)

# fail — wind + low rumble + descending hum
EVENTS["fail"] = (
    1.5, 0.6,
    [
        _wind_layer(0.7),
        _hum_layer(82.0),
    ],
)

# ready — bright bell + bird
EVENTS["ready"] = (
    1.2, 0.55,
    [
        _bell_layer(0.0, 880.0, dur=1.0),
        _bird_chirp_layer(0.7, 2400.0, dur=0.10),
        _wind_layer(0.2),
    ],
)

# ding — just the bell, no wind
EVENTS["ding"] = (
    0.9, 0.5,
    [
        _bell_layer(0.0, 1320.0, dur=0.8),
    ],
)


def _mix(layers: list[list[float]], total: float, peak: float) -> bytes:
    """Sum a list of per-sample layers, apply a master envelope, scale to int16."""
    n = int(SAMPLE_RATE * total)
    out = bytearray()
    # Sum samples
    mixed = [0.0] * n
    for layer_samples in layers:
        # Pad / trim in case a layer is the wrong length.
        for i in range(min(n, len(layer_samples))):
            mixed[i] += layer_samples[i]
    # Soft-clip by tanh to avoid harshness when many layers sum > 1
    for i in range(n):
        t = i / SAMPLE_RATE
        env = _envelope(t, total, attack=0.04, release=0.20)
        s = mixed[i] * env
        # gentle soft clip
        s = math.tanh(s * 1.2) / 1.2
        s = max(-1.0, min(1.0, s * peak))
        out += struct.pack("<h", int(s * 32767))
    return bytes(out)


def _write_wav(path: Path, duration: float, peak: float, layers: list) -> None:
    layer_buffers = [layer(duration) for layer in layers]
    pcm = _mix(layer_buffers, duration, peak)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for event, (duration, peak, layers) in EVENTS.items():
        out_path = OUT_DIR / f"{event}.wav"
        _write_wav(out_path, duration, peak, layers)
        print(f"wrote {out_path}  ({duration:.2f}s, {len(layers)} layers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
