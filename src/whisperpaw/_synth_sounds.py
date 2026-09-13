"""Generate the tiny WAV files for the default 'cat' sound pack.

This is a one-off maintenance script — run it once when adding a new
event, or run it now if the sounds/ directory is empty.

It produces short, soft, single-frequency tones with a gentle envelope
so they are pleasant to hear even if a build script fires ``paw-sound``
several times in a row. All files are mono 16-bit PCM at 22050 Hz.
"""
from __future__ import annotations

import math
import struct
import sys
import wave
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "sounds" / "cat"

# (event, base_freq_hz, duration_seconds, peak_amplitude 0..1)
EVENTS: dict[str, tuple[float, float, float]] = {
    "ok":    (523.25, 0.18, 0.45),  # C5  — bright
    "warn":  (392.00, 0.22, 0.40),  # G4  — soft caution
    "fail":  (196.00, 0.30, 0.50),  # G3  — low
    "ready": (659.25, 0.16, 0.42),  # E5  — cheerful
    "ding":  (880.00, 0.20, 0.38),  # A5  — bell
}

SAMPLE_RATE = 22050


def _envelope(t: float, total: float) -> float:
    """Smooth attack + release so the tones don't click."""
    if t < 0.02:
        return t / 0.02
    if t > total - 0.04:
        return max(0.0, (total - t) / 0.04)
    return 1.0


def make_tone(freq: float, duration: float, peak: float) -> bytes:
    n_samples = int(SAMPLE_RATE * duration)
    frames = bytearray()
    two_pi_f = 2.0 * math.pi * freq
    for n in range(n_samples):
        t = n / SAMPLE_RATE
        env = _envelope(t, duration)
        sample = peak * env * math.sin(two_pi_f * t)
        # Add a faint second harmonic for a less synthetic feel.
        sample += 0.15 * peak * env * math.sin(2.0 * two_pi_f * t)
        sample = max(-1.0, min(1.0, sample))
        frames += struct.pack("<h", int(sample * 32767))
    return bytes(frames)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for event, (freq, dur, peak) in EVENTS.items():
        out_path = OUT_DIR / f"{event}.wav"
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(make_tone(freq, dur, peak))
        print(f"wrote {out_path}  ({freq:.1f} Hz, {dur:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
