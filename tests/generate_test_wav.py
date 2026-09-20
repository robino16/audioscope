"""Generate synthetic WAV files for testing.

Run:  python tests/generate_test_wav.py
Writes data/test_sweep.wav and data/test_tones.wav.
"""
from __future__ import annotations
import numpy as np
import soundfile as sf
from pathlib import Path


SR = 44100


def sine(freq: float, duration: float, amp: float = 0.5, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)


def chirp(f0: float, f1: float, duration: float, amp: float = 0.5, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    k = (f1 - f0) / duration
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t ** 2)
    return (np.sin(phase) * amp).astype(np.float32)


def noise(duration: float, amp: float = 0.2, sr: int = SR) -> np.ndarray:
    n = int(duration * sr)
    return (np.random.randn(n) * amp).astype(np.float32)


def fade(sig: np.ndarray, fade_secs: float = 0.01, sr: int = SR) -> np.ndarray:
    fade_n = int(fade_secs * sr)
    sig = sig.copy()
    sig[:fade_n] *= np.linspace(0, 1, fade_n)
    sig[-fade_n:] *= np.linspace(1, 0, fade_n)
    return sig


def main() -> None:
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    # -- Sweep (mono, 5 s) --------------------------------------------------
    duration = 5.0
    sweep_mono = fade(chirp(20, 20000, duration, amp=0.6))
    sweep_stereo = np.column_stack([sweep_mono, sweep_mono * 0.8])
    sf.write(data_dir / "test_sweep.wav", sweep_stereo, SR, subtype="PCM_24")
    print(f"Wrote {data_dir / 'test_sweep.wav'}")

    # -- Tones (stereo, 8 s) ------------------------------------------------
    dur = 8.0
    n = int(dur * SR)
    left = np.zeros(n, dtype=np.float32)
    right = np.zeros(n, dtype=np.float32)

    tones_l = [110, 220, 440, 880, 1760, 3520]
    tones_r = [165, 330, 660, 1320, 2640, 5280]

    for freq in tones_l:
        left += sine(freq, dur, amp=0.12)
    for freq in tones_r:
        right += sine(freq, dur, amp=0.12)

    left += noise(dur, amp=0.05)
    right += noise(dur, amp=0.05)

    # Normalize
    peak = max(np.max(np.abs(left)), np.max(np.abs(right)))
    if peak > 0:
        left /= peak * 1.05
        right /= peak * 1.05

    stereo = np.column_stack([left, right])
    sf.write(data_dir / "test_tones.wav", stereo, SR, subtype="PCM_24")
    print(f"Wrote {data_dir / 'test_tones.wav'}")

    # -- Kick-like transients + bassline (stereo, 6 s) ----------------------
    dur = 6.0
    n = int(dur * SR)
    left = np.zeros(n, dtype=np.float32)
    right = np.zeros(n, dtype=np.float32)
    bpm = 120
    beat_n = int(SR * 60 / bpm)

    for beat in range(int(dur * bpm / 60)):
        pos = beat * beat_n
        if pos >= n:
            break
        # Kick: exponential decay, pitch drop
        kick_len = int(SR * 0.4)
        t_k = np.arange(kick_len) / SR
        env = np.exp(-t_k * 20)
        freq_env = 60 + 200 * np.exp(-t_k * 40)
        kick = np.sin(2 * np.pi * freq_env * t_k) * env * 0.7
        end = min(pos + kick_len, n)
        left[pos:end] += kick[:end - pos]
        right[pos:end] += kick[:end - pos]

        # Hi-hat on off-beats (every half-beat after beat 1)
        hat_pos = pos + beat_n // 2
        if hat_pos < n:
            hat_len = int(SR * 0.05)
            hat = noise(hat_len / SR, amp=0.3) * np.exp(-np.arange(hat_len) / SR * 100)
            he = min(hat_pos + hat_len, n)
            left[hat_pos:he] += hat[:he - hat_pos] * 0.8
            right[hat_pos:he] += hat[:he - hat_pos] * 1.0

    peak = max(np.max(np.abs(left)), np.max(np.abs(right)), 1e-6)
    left /= peak * 1.1
    right /= peak * 1.1
    sf.write(data_dir / "test_beat.wav", np.column_stack([left, right]), SR, subtype="PCM_24")
    print(f"Wrote {data_dir / 'test_beat.wav'}")


if __name__ == "__main__":
    main()
