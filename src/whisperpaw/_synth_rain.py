"""Generate the 'rain' sound pack — rhythmic ambient for restful focus.

Different from 'forest' (breathy wind + occasional bird) in that rain
is *rhythmic* — a steady drizzle of short transients. The five events
differ mostly in drop density and sky tone:

- ok: light steady drizzle, soft sky drone
- warn: medium rain, slightly darker drone
- fail: heavy rain, low rumble underneath
- ready: a single crack of thunder after a few seconds of silence
- ding: one single drop with a long tail (the bell of rain)

Pure stdlib: `wave`, `struct`, `math`, `random`. No samples, no third-party
deps. Run once after cloning or whenever the pack is missing.
"""
from __future__ import annotations

import math
import random
import struct
import wave
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "sounds" / "rain"
SAMPLE_RATE = 22050

# (event, duration_s, peak_amplitude, drops_per_second, base_drone_freq)
EVENTS: dict[str, tuple[float, float, float, float]] = {
    "ok":    (1.0, 0.45, 18.0, 180.0),   # light steady drizzle
    "warn":  (1.1, 0.50, 35.0, 140.0),   # medium rain
    "fail":  (1.4, 0.60, 80.0,  90.0),   # heavy rain + low rumble
    "ready": (1.5, 0.55, 12.0, 220.0),   # sparse drops, then one thunder crack
    "ding":  (0.8, 0.45,  4.0, 320.0),   # single high-pitched drop with long tail
}


def _envelope(t: float, total: float, attack: float = 0.005, release: float = 0.25) -> float:
    if t < attack:
        return t / attack
    if t > total - release:
        return max(0.0, (total - t) / release)
    return 1.0


def _drop_sample(t: float, t_hit: float, freq: float, decay: float = 0.04) -> float:
    """A single rain drop centered at t_hit. Short attack, exponential decay."""
    dt = t - t_hit
    if dt < 0:
        return 0.0
    # Envelope: very fast attack, exponential release
    if dt < 0.002:
        env = dt / 0.002
    else:
        env = math.exp(-(dt - 0.002) / decay)
    # Frequency sweeps down a bit during decay (a real drop has a slight
    # pitch glide as the resonance dies)
    f = freq * (1.0 - 0.4 * min(1.0, dt / (decay * 4)))
    return env * math.sin(2.0 * math.pi * f * dt)


def _sky_drone(total: float, freq: float, peak: float, rng: random.Random) -> list[float]:
    """A slow, low-pass-filtered noise floor — the 'air' between drops."""
    n = int(SAMPLE_RATE * total)
    out = [0.0] * n
    prev = 0.0
    alpha = 0.015  # strong low-pass — sounds like distant sky
    for i in range(n):
        x = rng.uniform(-1.0, 1.0)
        prev = prev + alpha * (x - prev)
        out[i] = prev * 0.5  # keep drone quiet so drops stay audible
    # Add a very slow wobble in the drone amplitude (atmospheric)
    for i in range(n):
        t = i / SAMPLE_RATE
        wobble = 0.7 + 0.3 * math.sin(2.0 * math.pi * 0.3 * t + rng.uniform(0, math.pi))
        out[i] *= wobble * peak
    return out


def _thunder_crack(t: float, total: float, peak: float) -> list[float]:
    """A low-frequency rumble + a brief crack at the start. Used for 'ready'."""
    n = int(SAMPLE_RATE * total)
    out = [0.0] * n
    crack_at = total * 0.55  # thunder lands after a moment of silence
    for i in range(n):
        ti = i / SAMPLE_RATE
        # The crack: short, loud, low-frequency burst
        if crack_at <= ti < crack_at + 0.4:
            dt = ti - crack_at
            env = math.exp(-dt / 0.15)  # 0.15s decay
            rumble = (
                0.5 * math.sin(2.0 * math.pi * 60.0 * ti) +
                0.3 * math.sin(2.0 * math.pi * 90.0 * ti) +
                0.2 * math.sin(2.0 * math.pi * 130.0 * ti + 0.7)
            )
            out[i] = env * rumble * peak
        # Pre-crack silence: a few soft drops
        elif ti < crack_at:
            for j in range(int(t * 4)):
                drop_t = j * 0.25 + (j * 0.07) % 0.15
                if abs(ti - drop_t) < 0.05:
                    out[i] += 0.15 * peak * math.sin(2.0 * math.pi * 300.0 * ti)
    return out


def _render_event(event: str, total: float, drops_per_sec: float, drone_freq: float, peak: float) -> list[float]:
    n = int(SAMPLE_RATE * total)
    rng = random.Random(hash(event) & 0xFFFFFFFF)

    if event == "ready":
        return _thunder_crack(0.0, total, peak)

    out = [0.0] * n

    # 1. Sky drone (the air)
    drone = _sky_drone(total, drone_freq, peak, rng)
    for i in range(n):
        out[i] += drone[i]

    # 2. Drops — Poisson-like spacing for natural rhythm
    expected_drops = max(1, int(total * drops_per_sec))
    drop_times: list[float] = []
    for _ in range(expected_drops):
        # Jitter around expected interval
        base = len(drop_times) * (total / expected_drops)
        drop_times.append(base + rng.uniform(-0.015, 0.015))
    # Add a couple of bigger "heavy" drops for warn/fail
    if event in ("warn", "fail"):
        drop_times.extend([rng.uniform(0.1, total - 0.1) for _ in range(3 if event == "warn" else 6)])

    # 3. Each drop
    drop_freq_base = 600.0 if event != "ding" else 1100.0  # ding is high-pitched
    for dt in drop_times:
        freq = drop_freq_base * rng.uniform(0.85, 1.15)
        # Larger drops have lower freq
        if event in ("warn", "fail") and rng.random() < 0.3:
            freq *= 0.6
        amp_scale = rng.uniform(0.5, 1.0)
        for i in range(n):
            ti = i / SAMPLE_RATE
            sample = _drop_sample(ti, dt, freq)
            out[i] += sample * amp_scale * peak * 0.7

    # 4. Apply master envelope (quick fade in, slower fade out)
    for i in range(n):
        ti = i / SAMPLE_RATE
        out[i] *= _envelope(ti, total, attack=0.02, release=0.30)

    # 5. Soft-clip via tanh
    for i in range(n):
        s = math.tanh(out[i] * 1.2) / 1.2
        out[i] = max(-1.0, min(1.0, s))

    return out


def _write_wav(path: Path, samples: list[float]) -> None:
    pcm = b"".join(struct.pack("<h", int(max(-1.0, min(1.0, s)) * 32767)) for s in samples)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for event, (duration, peak, drops, drone) in EVENTS.items():
        out_path = OUT_DIR / f"{event}.wav"
        samples = _render_event(event, duration, drops, drone, peak)
        _write_wav(out_path, samples)
        print(f"wrote {out_path}  ({duration:.2f}s, ~{int(duration * drops)} drops)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
