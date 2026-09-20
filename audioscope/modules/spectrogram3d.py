"""3D waterfall spectrogram - stacked perspective line plots via PIL drawing."""
from __future__ import annotations
from collections import deque
import numpy as np
from PIL import Image, ImageDraw

from audioscope.modules.base import BaseModule, get_fonts
from audioscope.utils import compute_fft, log_freq_bins, db_to_normalized


class Spectrogram3DModule(BaseModule):
    name = "spectrogram3d"
    default_aspect = 2.0

    def _setup(self) -> None:
        cfg = self.config
        self._fft_size = int(cfg.get("fft_size", 4096))
        self._window_func = str(cfg.get("window_func", "hann"))
        self._min_freq = float(cfg.get("min_freq", 20.0))
        self._max_freq = float(cfg.get("max_freq", 20000.0))
        self._db_range = float(cfg.get("db_range", 80.0))
        self._db_max = float(cfg.get("db_max", 0.0))
        self._db_min = self._db_max - self._db_range
        self._show_labels = bool(cfg.get("show_labels", True))
        self._n_slices = int(cfg.get("n_slices", 48))
        self._n_freq_bins = int(cfg.get("n_freq_bins", 128))

        # Oblique projection: how much each depth step shifts x (right) and y (up)
        self._slice_dx = float(cfg.get("slice_dx", 2.5))
        self._slice_dy = float(cfg.get("slice_dy", 4.0))
        # Maximum bar height as fraction of plot height
        self._amplitude_scale = float(cfg.get("amplitude_scale", 0.55))

        self._pad_left = 8
        self._pad_right = 8
        self._pad_top = 20
        self._pad_bottom = 18

        self._plot_w = self.width - self._pad_left - self._pad_right
        self._plot_h = self.height - self._pad_top - self._pad_bottom

        self._slices: deque = deque(maxlen=self._n_slices)

    # ------------------------------------------------------------------

    def process_frame(
        self,
        left: np.ndarray,
        right: np.ndarray,
        sample_rate: int,
        frame_index: int,
        fps: float,
    ) -> None:
        mono = (left + right) * 0.5
        freqs, mag_db = compute_fft(
            mono, self._fft_size, sample_rate, self._window_func, self._db_min
        )
        _, bin_db = log_freq_bins(
            freqs, mag_db, self._n_freq_bins, self._min_freq, self._max_freq
        )
        self._slices.append(db_to_normalized(bin_db, self._db_min, self._db_max))

    # ------------------------------------------------------------------

    def render(self) -> Image.Image:
        self._clear()
        img = self._to_image()
        draw = ImageDraw.Draw(img)

        slices = list(self._slices)
        n = len(slices)
        if n > 0:
            self._draw_slices(draw, slices)

        if self._show_labels:
            self._draw_labels(draw, n)

        self._draw_title(draw)
        return img

    # ------------------------------------------------------------------

    def _project(
        self, depth: int, n_slices: int, data: np.ndarray
    ):
        """Return (x_screen, y_screen, oy_base) arrays for one depth slice."""
        n_bins = self._n_freq_bins
        depth_frac = depth / max(1, n_slices - 1)

        # Slight perspective scale: front is full-size, back is slightly smaller
        persp = 1.0 - depth_frac * 0.20

        # Available width shrinks as depth offsets accumulate
        total_depth_x = (n_slices - 1) * self._slice_dx
        avail_w = (self._plot_w - total_depth_x) * persp
        bin_step = avail_w / max(1, n_bins - 1)

        ox = self._pad_left + depth * self._slice_dx
        oy_base = self.height - self._pad_bottom - int(depth * self._slice_dy)
        max_bar_h = int(self._plot_h * self._amplitude_scale * persp)

        xs = np.arange(n_bins)
        x_screen = np.clip((ox + xs * bin_step).astype(np.int32), 0, self.width - 1)
        y_screen = np.clip(
            (oy_base - data * max_bar_h).astype(np.int32), 0, self.height - 1
        )
        return x_screen, y_screen, int(oy_base)

    def _draw_slices(self, draw: ImageDraw.ImageDraw, slices: list) -> None:
        n = len(slices)
        cmap = self.theme.spectrum_cmap
        bg = self.theme.background

        # Painter's algorithm: draw from back (oldest) to front (newest)
        for slice_i in range(n):
            data = slices[slice_i]
            depth = n - 1 - slice_i   # 0 = most recent (front), n-1 = oldest (back)
            depth_frac = depth / max(1, n - 1)
            brightness = 1.0 - depth_frac * 0.72

            x_screen, y_screen, oy_base = self._project(depth, n, data)
            n_bins = len(x_screen)

            # --- Filled occlusion polygon (background colour under the curve) ---
            poly = [(int(x_screen[0]), oy_base)]
            for i in range(n_bins):
                poly.append((int(x_screen[i]), int(y_screen[i])))
            poly.append((int(x_screen[-1]), oy_base))
            draw.polygon(poly, fill=bg)

            # --- Coloured line segments across the profile ---
            # Group contiguous segments by colour bucket for efficiency
            for i in range(n_bins - 1):
                mag = float(data[i])
                ci = int(np.clip(mag * 255, 0, 255))
                base = cmap[ci].astype(np.float32)
                color = tuple(int(c) for c in np.clip(base * brightness, 0, 255))
                draw.line(
                    [(int(x_screen[i]), int(y_screen[i])),
                     (int(x_screen[i + 1]), int(y_screen[i + 1]))],
                    fill=color,
                    width=1,
                )

    def _draw_labels(self, draw: ImageDraw.ImageDraw, n_slices: int) -> None:
        font_s, _ = get_fonts()
        tc = self.theme.text_color

        if n_slices == 0:
            return

        draw.text(
            (self._pad_left, self.height - self._pad_bottom + 2),
            f"{n_slices} slices",
            fill=tc,
            font=font_s,
        )

        # Frequency marks along the front (depth=0) baseline
        freq_marks = [100, 500, 1000, 5000, 10000, 16000]
        dummy = np.zeros(self._n_freq_bins)
        x_screen, _, oy_base = self._project(0, n_slices, dummy)

        for freq in freq_marks:
            if not (self._min_freq <= freq <= self._max_freq):
                continue
            t = (np.log10(freq) - np.log10(self._min_freq)) / (
                np.log10(self._max_freq) - np.log10(self._min_freq)
            )
            xi = int(t * (self._n_freq_bins - 1))
            xi = max(0, min(xi, self._n_freq_bins - 1))
            x = int(x_screen[xi])
            label = f"{freq // 1000}k" if freq >= 1000 else str(freq)
            draw.line([(x, oy_base), (x, oy_base + 3)], fill=tc)
            draw.text((x, oy_base + 4), label, fill=tc, font=font_s, anchor="mt")

    def _draw_title(self, draw: ImageDraw.ImageDraw) -> None:
        font_s, _ = get_fonts()
        draw.text(
            (self._pad_left + 4, 4),
            "SPECTROGRAM 3D",
            fill=self.theme.accent_color,
            font=font_s,
        )
