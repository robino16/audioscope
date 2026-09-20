"""DSP utilities for audioscope."""
from __future__ import annotations
import numpy as np
from typing import Tuple

WINDOW_FUNCTIONS = {
    "hann": np.hanning,
    "hamming": np.hamming,
    "blackman": np.blackman,
    "bartlett": np.bartlett,
    "flat": lambda n: np.ones(n),
}


def apply_window(samples: np.ndarray, func: str = "hann") -> np.ndarray:
    fn = WINDOW_FUNCTIONS.get(func, np.hanning)
    return samples * fn(len(samples))


def compute_fft(
    samples: np.ndarray,
    fft_size: int,
    sample_rate: int,
    window_func: str = "hann",
    db_floor: float = -96.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (frequencies_hz, magnitudes_db) for a real FFT."""
    if len(samples) >= fft_size:
        x = samples[-fft_size:].astype(np.float32)
    else:
        x = np.zeros(fft_size, dtype=np.float32)
        x[-len(samples):] = samples

    x = apply_window(x, window_func)
    spectrum = np.fft.rfft(x, n=fft_size)
    magnitudes = np.abs(spectrum) / fft_size

    # Compensate for one-sided spectrum (except DC and Nyquist)
    magnitudes[1:-1] *= 2.0

    with np.errstate(divide="ignore", invalid="ignore"):
        mag_db = 20.0 * np.log10(np.maximum(magnitudes, 1e-12))

    mag_db = np.maximum(mag_db, db_floor)
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    return freqs, mag_db


def log_freq_bins(
    freqs: np.ndarray,
    magnitudes: np.ndarray,
    n_bins: int,
    min_freq: float = 20.0,
    max_freq: float = 20000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample FFT bins onto logarithmically-spaced frequency bins."""
    fmin = max(min_freq, float(freqs[1]) if len(freqs) > 1 else 1.0)
    fmax = min(max_freq, float(freqs[-1]))
    target_freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_bins)
    target_mags = np.interp(target_freqs, freqs, magnitudes)
    return target_freqs, target_mags


def db_to_normalized(
    db: np.ndarray,
    db_min: float,
    db_max: float,
) -> np.ndarray:
    return np.clip((db - db_min) / (db_max - db_min), 0.0, 1.0)


def smooth_exp(
    current: np.ndarray,
    new: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Exponential moving average: alpha=1 → no smoothing, alpha=0 → frozen."""
    return alpha * current + (1.0 - alpha) * new


def resample_linear(arr: np.ndarray, target_len: int) -> np.ndarray:
    if len(arr) == target_len:
        return arr
    x_old = np.linspace(0.0, 1.0, len(arr))
    x_new = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_new, x_old, arr)


def peak_hold_update(
    peaks: np.ndarray,
    hold_countdown: np.ndarray,
    new_values: np.ndarray,
    hold_frames: int,
    decay_per_frame: float = 0.008,
) -> Tuple[np.ndarray, np.ndarray]:
    """Update peak-hold state. Returns updated (peaks, hold_countdown)."""
    hold_countdown -= 1
    expired = hold_countdown < 0
    peaks[expired] -= decay_per_frame
    np.clip(peaks, 0.0, 1.0, out=peaks)

    exceeded = new_values > peaks
    peaks[exceeded] = new_values[exceeded]
    hold_countdown[exceeded] = hold_frames
    return peaks, hold_countdown


def rms_db(samples: np.ndarray, db_floor: float = -90.0) -> float:
    rms = float(np.sqrt(np.mean(samples ** 2)))
    if rms < 1e-10:
        return db_floor
    return max(db_floor, 20.0 * np.log10(rms))


def peak_db(samples: np.ndarray, db_floor: float = -90.0) -> float:
    pk = float(np.max(np.abs(samples)))
    if pk < 1e-10:
        return db_floor
    return max(db_floor, 20.0 * np.log10(pk))


def stereo_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Pearson correlation between left and right channels (-1 to +1)."""
    if len(left) == 0:
        return 0.0
    left_z = left - left.mean()
    right_z = right - right.mean()
    norm = np.sqrt(np.sum(left_z ** 2) * np.sum(right_z ** 2))
    if norm < 1e-12:
        return 1.0
    return float(np.sum(left_z * right_z) / norm)


def draw_line_aa(
    canvas: np.ndarray,
    x0: int, y0: int,
    x1: int, y1: int,
    color: tuple,
    thickness: int = 1,
) -> None:
    """Bresenham-style line draw into a (H, W, 3) numpy array."""
    h, w = canvas.shape[:2]
    color_arr = np.array(color, dtype=np.uint8)

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x, y = x0, y0
    while True:
        for ty in range(-thickness // 2, thickness // 2 + 1):
            for tx in range(-thickness // 2, thickness // 2 + 1):
                px, py = x + tx, y + ty
                if 0 <= px < w and 0 <= py < h:
                    canvas[py, px] = color_arr
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
