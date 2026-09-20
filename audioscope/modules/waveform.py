"""Waveform module - scrolling time-domain display."""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw

from audioscope.modules.base import BaseModule, get_fonts
from audioscope.utils import resample_linear, rms_db, peak_db


class WaveformModule(BaseModule):
    name = "waveform"
    default_aspect = 2.5

    def _setup(self) -> None:
        cfg = self.config
        self._duration = float(cfg.get("duration_secs", 0.5))
        self._show_both_channels = bool(cfg.get("stereo", True))
        self._show_labels = bool(cfg.get("show_labels", True))
        self._fill = bool(cfg.get("fill", True))
        self._line_thickness = int(cfg.get("line_thickness", 1))

        self._pad_left = 8
        self._pad_right = 8
        self._pad_top = 16
        self._pad_bottom = 16
        self._plot_w = self.width - self._pad_left - self._pad_right

        # Per-frame state
        self._left_wave: np.ndarray = np.zeros(self._plot_w, dtype=np.float32)
        self._right_wave: np.ndarray = np.zeros(self._plot_w, dtype=np.float32)
        self._rms_left: float = -90.0
        self._rms_right: float = -90.0
        self._peak_left: float = -90.0
        self._peak_right: float = -90.0

    # ------------------------------------------------------------------

    def process_frame(
        self,
        left: np.ndarray,
        right: np.ndarray,
        sample_rate: int,
        frame_index: int,
        fps: float,
    ) -> None:
        n_samples = int(self._duration * sample_rate)

        if len(left) >= n_samples:
            left_win = left[-n_samples:]
            right_win = right[-n_samples:]
        else:
            left_win = np.zeros(n_samples, dtype=np.float32)
            right_win = np.zeros(n_samples, dtype=np.float32)
            left_win[-len(left):] = left
            right_win[-len(right):] = right

        self._left_wave = resample_linear(left_win, self._plot_w)
        self._right_wave = resample_linear(right_win, self._plot_w)
        self._rms_left = rms_db(left_win)
        self._rms_right = rms_db(right_win)
        self._peak_left = peak_db(left_win)
        self._peak_right = peak_db(right_win)

    # ------------------------------------------------------------------

    def render(self) -> Image.Image:
        self._clear()

        if self._show_both_channels:
            mid = self.height // 2
            self._draw_channel(
                self._left_wave,
                y_center=self._pad_top + (mid - self._pad_top - self._pad_bottom // 2) // 2,
                half_h=(mid - self._pad_top - self._pad_bottom // 2) // 2,
                cmap=self.theme.waveform_cmap,
            )
            self._draw_channel(
                self._right_wave,
                y_center=mid + (self.height - mid - self._pad_bottom) // 2,
                half_h=(self.height - mid - self._pad_bottom) // 2,
                cmap=self.theme.waveform_cmap,
            )
            # Divider
            self._draw_hline(mid, self.theme.grid_color)
        else:
            mono = (self._left_wave + self._right_wave) * 0.5
            center = self.height // 2
            half_h = center - self._pad_top - self._pad_bottom
            self._draw_channel(
                mono,
                y_center=center,
                half_h=half_h,
                cmap=self.theme.waveform_cmap,
            )

        img = self._to_image()
        draw = ImageDraw.Draw(img)

        if self._show_labels:
            self._draw_stats(draw)

        self._draw_title(draw)
        return img

    # ------------------------------------------------------------------

    def _draw_channel(
        self,
        wave: np.ndarray,
        y_center: int,
        half_h: int,
        cmap: np.ndarray,
    ) -> None:
        """Draw waveform centred at y_center with +/- half_h amplitude range."""
        if half_h <= 0:
            return

        # Center line
        self._draw_hline(y_center, self.theme.grid_color, self._pad_left, self.width - self._pad_right)

        x0 = self._pad_left
        x1 = self.width - self._pad_right
        w = x1 - x0

        clipped = np.clip(wave[:w] if len(wave) >= w else np.pad(wave, (0, w - len(wave))), -1.0, 1.0)
        amp_abs = (np.abs(clipped) * half_h).astype(np.float32)  # (W,)

        if self._fill:
            # Vectorised: build mask (H, W) where |y - y_center| <= amp_abs[x]
            y_range = np.arange(self.height, dtype=np.float32)[:, None]   # (H, 1)
            y_dist = np.abs(y_range - y_center)                            # (H, 1)
            mask = y_dist <= amp_abs[None, :]                              # (H, W)

            # Color based on normalized y-distance (bright at extremes)
            t_vals = np.clip(y_dist / max(1, half_h), 0.0, 1.0)          # (H, 1)
            color_idx = (t_vals * 255).astype(np.int32)[:, 0]             # (H,)
            colors = cmap[color_idx]                                       # (H, 3)

            # Apply: where mask is True, write color; else keep canvas value
            region = self._canvas[:, x0:x1, :]                            # (H, W, 3)
            region[:] = np.where(
                mask[:, :, np.newaxis],
                colors[:, np.newaxis, :],
                region,
            )
        else:
            # Outline-only: draw a thin line at the waveform edge using PIL
            img = self._to_image()
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            for xi in range(w - 1):
                a0 = float(clipped[xi]) * half_h
                a1 = float(clipped[xi + 1]) * half_h
                y0 = int(np.clip(y_center - a0, 0, self.height - 1))
                y1 = int(np.clip(y_center - a1, 0, self.height - 1))
                t = abs(a0) / max(1, half_h)
                ci = int(np.clip(t * 255, 0, 255))
                color = tuple(int(c) for c in cmap[ci])
                draw.line([(xi + x0, y0), (xi + 1 + x0, y1)], fill=color, width=1)
            self._canvas[:] = np.array(img)

    def _draw_stats(self, draw: ImageDraw.ImageDraw) -> None:
        font_s, _ = get_fonts()
        text_col = self.theme.text_color
        x = self._pad_left + 4
        if self._show_both_channels:
            draw.text((x, self._pad_top), f"L  RMS {self._rms_left:.1f}dB  PK {self._peak_left:.1f}dB",
                      fill=text_col, font=font_s)
            draw.text((x, self.height // 2 + 2),
                      f"R  RMS {self._rms_right:.1f}dB  PK {self._peak_right:.1f}dB",
                      fill=text_col, font=font_s)
        else:
            draw.text((x, self._pad_top),
                      f"RMS {self._rms_left:.1f}dB  PK {self._peak_left:.1f}dB",
                      fill=text_col, font=font_s)

    def _draw_title(self, draw: ImageDraw.ImageDraw) -> None:
        font_s, _ = get_fonts()
        draw.text((self._pad_left + 4, 2), "WAVEFORM", fill=self.theme.accent_color, font=font_s)
