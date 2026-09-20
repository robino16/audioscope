"""Audio file loading and sample access."""
from __future__ import annotations
from pathlib import Path
from typing import Tuple
import numpy as np
import soundfile as sf


class AudioFile:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data, self._sample_rate = sf.read(
            self.path, dtype="float32", always_2d=True
        )
        # shape: (n_frames, n_channels)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def n_frames(self) -> int:
        return self._data.shape[0]

    @property
    def n_channels(self) -> int:
        return self._data.shape[1]

    @property
    def duration(self) -> float:
        return self.n_frames / self._sample_rate

    @property
    def is_stereo(self) -> bool:
        return self._data.shape[1] >= 2

    # ------------------------------------------------------------------
    # Channel access
    # ------------------------------------------------------------------

    def get_mono(self) -> np.ndarray:
        return np.mean(self._data, axis=1)

    def get_left(self) -> np.ndarray:
        return self._data[:, 0]

    def get_right(self) -> np.ndarray:
        return self._data[:, min(1, self.n_channels - 1)]

    # ------------------------------------------------------------------
    # Per-frame sample access
    # ------------------------------------------------------------------

    def get_window(
        self,
        frame_index: int,
        fps: float,
        window_seconds: float,
        trailing: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (left, right) sample arrays for a trailing/centered window.

        The window always has exactly `int(window_seconds * sample_rate)` samples,
        zero-padded at the start if the audio hasn't started yet.
        """
        window_n = int(window_seconds * self._sample_rate)
        spf = self._sample_rate / fps

        if trailing:
            end = int((frame_index + 1) * spf)
        else:
            center = int(frame_index * spf)
            end = center + window_n // 2

        start = end - window_n

        # Clamp indices, build padded segment
        src_start = max(0, start)
        src_end = min(self.n_frames, end)

        segment = self._data[src_start:src_end]
        n_got = segment.shape[0]

        if n_got < window_n:
            pad_front = max(0, -start)
            pad_back = window_n - n_got - pad_front
            pad_front_arr = np.zeros((pad_front, self.n_channels), dtype=np.float32)
            pad_back_arr = np.zeros((pad_back, self.n_channels), dtype=np.float32)
            segment = np.concatenate([pad_front_arr, segment, pad_back_arr], axis=0)

        left = segment[:window_n, 0]
        right = segment[:window_n, min(1, self.n_channels - 1)]
        return left, right

    def get_frame_slice(
        self,
        frame_index: int,
        fps: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the exact per-frame audio slice (zero-padded at end)."""
        spf = int(self._sample_rate / fps)
        start = int(frame_index * (self._sample_rate / fps))
        end = min(start + spf, self.n_frames)
        segment = self._data[start:end]

        if len(segment) < spf:
            pad = np.zeros((spf - len(segment), self.n_channels), dtype=np.float32)
            segment = np.concatenate([segment, pad], axis=0)

        left = segment[:, 0]
        right = segment[:, min(1, self.n_channels - 1)]
        return left, right

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def apply_gain_db(self, db: float) -> None:
        if db != 0.0:
            self._data *= 10.0 ** (db / 20.0)

    def normalize(self) -> None:
        peak = np.max(np.abs(self._data))
        if peak > 1e-10:
            self._data /= peak

    def __repr__(self) -> str:
        return (
            f"AudioFile({self.path.name!r}, "
            f"{self.n_channels}ch, "
            f"{self._sample_rate}Hz, "
            f"{self.duration:.2f}s)"
        )
