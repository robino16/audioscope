"""Spectrogram module - rolling waterfall display."""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw

from audioscope.modules.base import BaseModule, get_fonts
from audioscope.utils import compute_fft, log_freq_bins, db_to_normalized


class SpectrogramModule(BaseModule):
    name = "spectrogram"
    default_aspect = 2.0

    def _setup(self) -> None:
        cfg = self.config
        self._fft_size = int(cfg.get("fft_size", 4096))
        self._window_func = str(cfg.get("window_func", "hann"))
        self._min_freq = float(cfg.get("min_freq", 20.0))
        self._max_freq = float(cfg.get("max_freq", 20000.0))
        self._db_range = float(cfg.get("db_range", 80.0))
        self._db_max = float(cfg.get("db_max", 0.0))
        self._show_labels = bool(cfg.get("show_labels", True))
        self._scroll_dir = str(cfg.get("scroll", "left"))  # "left" or "right"

        self._db_min = self._db_max - self._db_range

        self._pad_left = 38 if self._show_labels else 4
        self._pad_bottom = 18 if self._show_labels else 4
        self._plot_w = self.width - self._pad_left - 4
        self._plot_h = self.height - self._pad_bottom - 8

        # n_freq_bins = plot height; history = plot width
        self._n_freq = self._plot_h
        self._history = self._plot_w

        # Circular buffer: shape (n_freq, history)
        self._buffer = np.zeros((self._n_freq, self._history), dtype=np.float32)
        self._write_pos = 0  # next column to write

        self._sample_rate = 44100

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
        mono = (left + right) * 0.5

        freqs, mag_db = compute_fft(
            mono, self._fft_size, sample_rate, self._window_func, self._db_min
        )
        _, bin_db = log_freq_bins(
            freqs, mag_db, self._n_freq, self._min_freq, self._max_freq
        )
        # Normalize to 0-1
        normalized = db_to_normalized(bin_db, self._db_min, self._db_max)

        # Write into circular buffer; low freq at top index 0 → invert for display
        self._buffer[:, self._write_pos] = normalized[::-1]
        self._write_pos = (self._write_pos + 1) % self._history

    # ------------------------------------------------------------------

    def render(self) -> Image.Image:
        self._clear()

        # Reorder circular buffer so oldest on left, newest on right
        indices = [(self._write_pos + i) % self._history for i in range(self._history)]
        ordered = self._buffer[:, indices]  # (n_freq, history)

        # Map to colors using spectrogram colormap
        cmap = self.theme.spectrogram_cmap
        idx = np.clip((ordered * 255).astype(np.int32), 0, 255)
        # colors: (n_freq, history, 3)
        colors = cmap[idx]

        x0 = self._pad_left
        y0 = 8
        y1 = y0 + self._plot_h
        x1 = x0 + self._plot_w

        # Assign into canvas
        self._canvas[y0:y1, x0:x1] = colors

        img = self._to_image()
        draw = ImageDraw.Draw(img)

        if self._show_labels:
            self._draw_freq_labels(draw)
            self._draw_db_legend(draw)

        self._draw_title(draw)
        return img

    # ------------------------------------------------------------------

    def _draw_freq_labels(self, draw: ImageDraw.ImageDraw) -> None:
        font_s, _ = get_fonts()
        freq_marks = [31, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        y0 = 8
        y1 = y0 + self._plot_h

        for freq in freq_marks:
            if not (self._min_freq <= freq <= self._max_freq):
                continue
            t = (np.log10(freq) - np.log10(self._min_freq)) / (
                np.log10(self._max_freq) - np.log10(self._min_freq)
            )
            # Inverted: high freq at top
            y = int(y0 + (1.0 - t) * self._plot_h)
            label = f"{freq // 1000}k" if freq >= 1000 else str(freq)
            self._draw_hline(y, self.theme.grid_color, self._pad_left, self._pad_left + self._plot_w)
            draw.text(
                (self._pad_left - 3, y),
                label,
                fill=self.theme.text_color,
                font=font_s,
                anchor="rm",
            )

    def _draw_db_legend(self, draw: ImageDraw.ImageDraw) -> None:
        font_s, _ = get_fonts()
        bx = self.width - 50
        by = 8
        # Tiny gradient swatch
        swatch_h = 60
        for sy in range(swatch_h):
            t = 1.0 - sy / swatch_h
            idx = int(t * 255)
            color = tuple(int(c) for c in self.theme.spectrogram_cmap[idx])
            self._canvas[by + sy, bx:bx + 8] = color
        draw.text((bx + 10, by), f"{self._db_max:+.0f}dB", fill=self.theme.text_color, font=font_s)
        draw.text((bx + 10, by + swatch_h - 8), f"{self._db_min:+.0f}dB", fill=self.theme.text_color, font=font_s)

    def _draw_title(self, draw: ImageDraw.ImageDraw) -> None:
        font_s, _ = get_fonts()
        draw.text((self._pad_left + 4, 4), "SPECTROGRAM", fill=self.theme.accent_color, font=font_s)
