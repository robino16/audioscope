"""Spectrum analyzer module - logarithmic FFT bar display."""
from __future__ import annotations
from typing import Optional
import numpy as np
from PIL import Image, ImageDraw

from audioscope.modules.base import BaseModule, get_fonts
from audioscope.utils import (
    compute_fft,
    log_freq_bins,
    db_to_normalized,
    smooth_exp,
    peak_hold_update,
)


class SpectrumModule(BaseModule):
    name = "spectrum"
    default_aspect = 2.0

    def _setup(self) -> None:
        cfg = self.config
        self._fft_size = int(cfg.get("fft_size", 4096))
        self._window_func = str(cfg.get("window_func", "hann"))
        self._min_freq = float(cfg.get("min_freq", 20.0))
        self._max_freq = float(cfg.get("max_freq", 20000.0))
        self._db_range = float(cfg.get("db_range", 80.0))
        self._db_max = float(cfg.get("db_max", 0.0))
        self._smoothing = float(cfg.get("smoothing", 0.72))
        self._peak_hold_secs = float(cfg.get("peak_hold_secs", 2.0))
        self._show_labels = bool(cfg.get("show_labels", True))
        self._show_peaks = bool(cfg.get("show_peaks", True))
        self._fill = bool(cfg.get("fill", True))

        self._pad_left = 38 if self._show_labels else 4
        self._pad_bottom = 18 if self._show_labels else 4
        self._plot_w = self.width - self._pad_left - 4
        self._plot_h = self.height - self._pad_bottom - 8

        # Number of bars: roughly 1 bar per 3 pixels, capped for readability
        self._n_bars = max(32, min(200, self._plot_w // 4))

        self._smoothed = np.zeros(self._n_bars, dtype=np.float32)
        self._peaks = np.zeros(self._n_bars, dtype=np.float32)
        self._peak_hold_countdown = np.zeros(self._n_bars, dtype=np.float32)

        self._sample_rate: int = 44100
        self._fps: float = 24.0
        self._db_min = self._db_max - self._db_range

    # ------------------------------------------------------------------

    def process_frame(
        self,
        left: np.ndarray,
        right: np.ndarray,
        sample_rate: int,
        frame_index: int,
        fps: float,
    ) -> None:
        self._sample_rate = sample_rate
        self._fps = fps

        mono = (left + right) * 0.5

        freqs, mag_db = compute_fft(
            mono, self._fft_size, sample_rate, self._window_func, self._db_min
        )
        _, bar_db = log_freq_bins(
            freqs, mag_db, self._n_bars, self._min_freq, self._max_freq
        )
        normalized = db_to_normalized(bar_db, self._db_min, self._db_max)

        self._smoothed = smooth_exp(self._smoothed, normalized, self._smoothing)

        hold_frames = int(self._peak_hold_secs * fps)
        self._peaks, self._peak_hold_countdown = peak_hold_update(
            self._peaks,
            self._peak_hold_countdown,
            self._smoothed,
            hold_frames,
            decay_per_frame=1.0 / (fps * 3.0),
        )

    # ------------------------------------------------------------------

    def render(self) -> Image.Image:
        self._clear()
        self._draw_grid()
        self._draw_bars()
        if self._show_peaks:
            self._draw_peak_markers()

        img = self._to_image()
        draw = ImageDraw.Draw(img)

        if self._show_labels:
            self._draw_freq_labels(draw)
            self._draw_db_labels(draw)

        self._draw_title(draw)
        return img

    # ------------------------------------------------------------------
    # Internal drawing helpers
    # ------------------------------------------------------------------

    def _db_to_y(self, db: float) -> int:
        t = (db - self._db_min) / (self._db_max - self._db_min)
        return int(self.height - self._pad_bottom - t * self._plot_h)

    def _draw_grid(self) -> None:
        step = 10 if self._db_range <= 80 else 20
        for db_val in range(int(self._db_min), int(self._db_max) + 1, step):
            y = self._db_to_y(db_val)
            if 0 < y < self.height - self._pad_bottom:
                self._draw_hline(y, self.theme.grid_color, self._pad_left)

        # Center separator line at 0 dB if in range
        if self._db_min < 0 <= self._db_max:
            y0 = self._db_to_y(0)
            col = tuple(min(255, c + 30) for c in self.theme.grid_color)
            self._draw_hline(y0, col, self._pad_left)

    def _draw_bars(self) -> None:
        n = self._n_bars
        total_gap = n  # 1px gap per bar
        bar_w = max(1, (self._plot_w - total_gap) // n)
        gap = 1 if bar_w > 2 else 0

        cmap = self.theme.spectrum_cmap
        plot_bottom = self.height - self._pad_bottom
        plot_top = plot_bottom - self._plot_h

        if self._fill:
            # Pre-compute gradient for the full plot height once
            y_all = np.arange(self.height, dtype=np.float32)
            t_all = np.clip((plot_bottom - y_all) / max(1, self._plot_h), 0.0, 1.0)
            grad_col = cmap[(t_all * 255).astype(np.int32)]  # (H, 3)

            for i, value in enumerate(self._smoothed):
                bar_h = int(value * self._plot_h)
                if bar_h < 1:
                    continue
                x0 = self._pad_left + i * (bar_w + gap)
                x1 = min(x0 + bar_w, self.width - 4)
                y_top = max(plot_top, plot_bottom - bar_h)
                if x0 >= x1:
                    continue
                # Slice assignment: broadcast single gradient column across bar width
                self._canvas[y_top:plot_bottom, x0:x1] = grad_col[y_top:plot_bottom, np.newaxis, :]
        else:
            for i, value in enumerate(self._smoothed):
                if value < 0.005:
                    continue
                x0 = self._pad_left + i * (bar_w + gap)
                x1 = min(x0 + bar_w, self.width - 4)
                y_top = max(plot_top, plot_bottom - int(value * self._plot_h))
                if x0 >= x1:
                    continue
                color = tuple(int(c) for c in cmap[int(value * 255)])
                self._canvas[y_top:y_top + 2, x0:x1] = color

    def _draw_peak_markers(self) -> None:
        n = self._n_bars
        bar_w = max(1, (self._plot_w - n) // n)
        gap = 1 if bar_w > 2 else 0
        plot_bottom = self.height - self._pad_bottom
        peak_color = np.array(self.theme.peak_color, dtype=np.uint8)

        for i, peak in enumerate(self._peaks):
            if peak < 0.01:
                continue
            x0 = self._pad_left + i * (bar_w + gap)
            x1 = min(x0 + bar_w, self.width - 4)
            y = int(plot_bottom - peak * self._plot_h)
            y = max(8, min(y, plot_bottom - 1))
            if x0 < x1:
                self._canvas[y:y + 2, x0:x1] = peak_color

    def _draw_freq_labels(self, draw: ImageDraw.ImageDraw) -> None:
        font_s, _ = get_fonts()
        n = self._n_bars
        bar_w = max(1, (self._plot_w - n) // n)
        gap = 1 if bar_w > 2 else 0
        plot_bottom = self.height - self._pad_bottom

        freq_marks = [31, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        for freq in freq_marks:
            if not (self._min_freq <= freq <= self._max_freq):
                continue
            t = (np.log10(freq) - np.log10(self._min_freq)) / (
                np.log10(self._max_freq) - np.log10(self._min_freq)
            )
            bar_i = int(t * (n - 1))
            x = self._pad_left + bar_i * (bar_w + gap) + bar_w // 2
            label = f"{freq // 1000}k" if freq >= 1000 else str(freq)
            draw.text(
                (x, plot_bottom + 3),
                label,
                fill=self.theme.text_color,
                font=font_s,
                anchor="mt",
            )
            # Tick mark
            self._draw_vline(x, self.theme.grid_color, plot_bottom, plot_bottom + 3)

    def _draw_db_labels(self, draw: ImageDraw.ImageDraw) -> None:
        font_s, _ = get_fonts()
        step = 10 if self._db_range <= 80 else 20
        for db_val in range(int(self._db_min), int(self._db_max) + 1, step):
            y = self._db_to_y(db_val)
            if 8 < y < self.height - self._pad_bottom:
                draw.text(
                    (self._pad_left - 3, y),
                    f"{db_val:+d}",
                    fill=self.theme.text_color,
                    font=font_s,
                    anchor="rm",
                )

    def _draw_title(self, draw: ImageDraw.ImageDraw) -> None:
        font_s, _ = get_fonts()
        draw.text(
            (self._pad_left + 4, 4),
            "SPECTRUM",
            fill=self.theme.accent_color,
            font=font_s,
        )
