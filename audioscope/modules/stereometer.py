"""Stereometer module - goniometer, correlation meter, and L/R level bars."""
from __future__ import annotations
from collections import deque
import numpy as np
from PIL import Image, ImageDraw

from audioscope.modules.base import BaseModule, get_fonts
from audioscope.utils import rms_db, peak_db, stereo_correlation, peak_hold_update


class StereometerModule(BaseModule):
    name = "stereometer"
    default_aspect = 1.5

    def _setup(self) -> None:
        cfg = self.config
        self._trail_length = int(cfg.get("trail_length_frames", 60))
        self._show_labels = bool(cfg.get("show_labels", True))
        self._db_range = float(cfg.get("db_range", 60.0))
        self._db_max = float(cfg.get("db_max", 0.0))
        self._db_min = self._db_max - self._db_range

        # Goniometer occupies a square on the left portion
        self._meter_w = 56  # width of level-meter panel on the right
        self._gonio_size = min(self.height - 30, self.width - self._meter_w - 20)
        self._gonio_cx = 10 + self._gonio_size // 2
        self._gonio_cy = self.height // 2
        self._gonio_r = (self._gonio_size // 2) - 6

        # Circular dot-trail buffer: each entry is (x_norm, y_norm) in [-1, 1]
        self._trail: deque = deque(maxlen=self._trail_length * 64)

        # Level state
        self._rms_l: float = self._db_min
        self._rms_r: float = self._db_min
        self._peak_l: float = self._db_min
        self._peak_r: float = self._db_min
        self._hold_l = np.zeros(1, dtype=np.float32)
        self._hold_r = np.zeros(1, dtype=np.float32)
        self._hold_cnt_l = np.zeros(1, dtype=np.float32)
        self._hold_cnt_r = np.zeros(1, dtype=np.float32)
        self._correlation: float = 0.0

        self._fps: float = 24.0

    # ------------------------------------------------------------------

    def process_frame(
        self,
        left: np.ndarray,
        right: np.ndarray,
        sample_rate: int,
        frame_index: int,
        fps: float,
    ) -> None:
        self._fps = fps

        # Subsample for goniometer trail (keep ~64 points per frame)
        step = max(1, len(left) // 64)
        l_sub = left[::step]
        r_sub = right[::step]

        for l_s, r_s in zip(l_sub, r_sub):
            # Classic goniometer: x = (R - L) / sqrt(2), y = (L + R) / sqrt(2)
            gx = (float(r_s) - float(l_s)) * 0.70711
            gy = (float(l_s) + float(r_s)) * 0.70711
            self._trail.append((gx, gy))

        self._rms_l = rms_db(left)
        self._rms_r = rms_db(right)
        self._peak_l = peak_db(left)
        self._peak_r = peak_db(right)
        self._correlation = stereo_correlation(left, right)

        norm_l = np.array([max(0.0, (self._rms_l - self._db_min) / self._db_range)])
        norm_r = np.array([max(0.0, (self._rms_r - self._db_min) / self._db_range)])
        hold_frames = int(2.0 * fps)
        self._hold_l, self._hold_cnt_l = peak_hold_update(
            self._hold_l, self._hold_cnt_l, norm_l, hold_frames, 1.0 / (fps * 3)
        )
        self._hold_r, self._hold_cnt_r = peak_hold_update(
            self._hold_r, self._hold_cnt_r, norm_r, hold_frames, 1.0 / (fps * 3)
        )

    # ------------------------------------------------------------------

    def render(self) -> Image.Image:
        self._clear()
        self._draw_goniometer_bg()
        self._draw_trail()
        self._draw_level_meters()
        self._draw_correlation_meter()

        img = self._to_image()
        draw = ImageDraw.Draw(img)
        if self._show_labels:
            self._draw_labels(draw)
        self._draw_title(draw)
        return img

    # ------------------------------------------------------------------

    def _draw_goniometer_bg(self) -> None:
        cx = self._gonio_cx
        cy = self._gonio_cy
        r = self._gonio_r

        # Outer circle
        for angle_deg in range(0, 360, 1):
            a = np.radians(angle_deg)
            x = int(cx + r * np.cos(a))
            y = int(cy - r * np.sin(a))
            if 0 <= x < self.width and 0 <= y < self.height:
                self._canvas[y, x] = self.theme.border_color

        # Inner ring at half radius
        r2 = r // 2
        for angle_deg in range(0, 360, 2):
            a = np.radians(angle_deg)
            x = int(cx + r2 * np.cos(a))
            y = int(cy - r2 * np.sin(a))
            if 0 <= x < self.width and 0 <= y < self.height:
                self._canvas[y, x] = self.theme.grid_color

        # L axis (135° — upper-left to lower-right)
        # R axis (45°  — upper-right to lower-left)
        for t in range(-r, r + 1):
            for angle in (135, 45, 90, 180):
                x = cx + int(t * np.cos(np.radians(angle)))
                y = cy - int(t * np.sin(np.radians(angle)))
                if 0 <= x < self.width and 0 <= y < self.height:
                    self._canvas[y, x] = self.theme.grid_color

    def _draw_trail(self) -> None:
        trail = list(self._trail)
        n = len(trail)
        if n == 0:
            return

        cx = self._gonio_cx
        cy = self._gonio_cy
        r = self._gonio_r

        gx_arr = np.array([p[0] for p in trail], dtype=np.float32)
        gy_arr = np.array([p[1] for p in trail], dtype=np.float32)

        # Pixel coordinates
        px = np.clip((cx + np.clip(gx_arr, -1.0, 1.0) * r).astype(np.int32), 0, self.width - 1)
        py = np.clip((cy - np.clip(gy_arr, -1.0, 1.0) * r).astype(np.int32), 0, self.height - 1)

        # Per-point brightness (older = dimmer)
        brightness = (np.arange(n, dtype=np.float32) / n) ** 1.4
        dot_c = np.array(self.theme.stereo_dot_color, dtype=np.float32)
        trail_c = np.array(self.theme.stereo_trail_color, dtype=np.float32)
        colors = trail_c + (dot_c - trail_c) * brightness[:, None]
        colors = np.clip(colors, 0, 255).astype(np.uint8)  # (N, 3)

        # Expand to 3×3 blocks
        offsets = np.array([-1, 0, 1], dtype=np.int32)
        dy_g, dx_g = np.meshgrid(offsets, offsets)
        all_px = (px[:, None] + dx_g.flatten()[None, :]).flatten()   # (N*9,)
        all_py = (py[:, None] + dy_g.flatten()[None, :]).flatten()   # (N*9,)
        all_c = np.repeat(colors, 9, axis=0)                          # (N*9, 3)

        # Bounds + circle mask
        in_bounds = (
            (all_px >= 0) & (all_px < self.width) &
            (all_py >= 0) & (all_py < self.height)
        )
        in_circle = (all_px - cx) ** 2 + (all_py - cy) ** 2 <= r * r
        valid = in_bounds & in_circle

        self._canvas[all_py[valid], all_px[valid]] = all_c[valid]

    def _draw_level_meters(self) -> None:
        bar_x0 = self._gonio_cx + self._gonio_r + 14
        bar_w = 20
        bar_h = self._gonio_r * 2
        bar_y0 = self._gonio_cy - self._gonio_r
        gap = 6

        for ch, (rms, hold) in enumerate([
            (self._rms_l, float(self._hold_l[0])),
            (self._rms_r, float(self._hold_r[0])),
        ]):
            bx = bar_x0 + ch * (bar_w + gap)
            if bx + bar_w > self.width:
                break

            # Background trough
            self._canvas[bar_y0:bar_y0 + bar_h, bx:bx + bar_w] = self.theme.grid_color

            # Filled bar (vectorised)
            t = max(0.0, (rms - self._db_min) / self._db_range)
            fill_h = int(t * bar_h)
            if fill_h > 0:
                fill_y0 = bar_y0 + (bar_h - fill_h)
                t_vals = np.arange(fill_h, dtype=np.float32) / max(1, fill_h)
                idx = np.clip((t_vals * 255).astype(np.int32), 0, 255)
                colors = self.theme.level_cmap[idx]                 # (fill_h, 3)
                row_idx = fill_y0 + fill_h - 1 - np.arange(fill_h) # bottom→top
                self._canvas[row_idx[:, None], np.arange(bx, bx + bar_w)[None, :]] = (
                    colors[:, np.newaxis, :]
                )

            # Peak hold tick
            ph = int(hold * bar_h)
            if ph > 0:
                py = bar_y0 + bar_h - ph
                if bar_y0 <= py < bar_y0 + bar_h:
                    self._canvas[py:py + 2, bx:bx + bar_w] = self.theme.peak_color

            # Border
            self._draw_rect_border(bx, bar_y0, bx + bar_w, bar_y0 + bar_h, self.theme.border_color)

    def _draw_correlation_meter(self) -> None:
        """Horizontal bar showing stereo correlation in -1..+1 range."""
        bar_x0 = 10
        bar_y = self.height - 14
        bar_w = self._gonio_size
        bar_h = 6

        if bar_y < 0 or bar_x0 + bar_w > self.width:
            return

        # Background trough
        self._canvas[bar_y:bar_y + bar_h, bar_x0:bar_x0 + bar_w] = self.theme.grid_color

        c = self._correlation
        mid = bar_x0 + bar_w // 2
        t = (c + 1.0) / 2.0
        px = int(bar_x0 + t * bar_w)

        if c >= 0:
            color = self.theme.correlation_pos_color
            x0 = min(mid, px)
            x1 = max(mid, px)
            self._canvas[bar_y:bar_y + bar_h, x0:x1] = color
        else:
            color = self.theme.correlation_neg_color
            x0 = min(mid, px)
            x1 = max(mid, px)
            self._canvas[bar_y:bar_y + bar_h, x0:x1] = color

        # Centre tick
        self._draw_vline(mid, self.theme.text_color, bar_y - 2, bar_y + bar_h + 2)

    def _draw_labels(self, draw: ImageDraw.ImageDraw) -> None:
        font_s, _ = get_fonts()
        cx = self._gonio_cx
        cy = self._gonio_cy
        r = self._gonio_r
        tc = self.theme.text_color

        draw.text((cx - 4, cy - r - 12), "M", fill=tc, font=font_s)
        draw.text((cx + r + 3, cy - 6), "R", fill=tc, font=font_s)
        draw.text((cx - r - 10, cy - 6), "L", fill=tc, font=font_s)
        draw.text((cx - 2, cy + r + 3), "S", fill=tc, font=font_s)

        # Correlation value
        bar_y = self.height - 14
        draw.text((10, bar_y - 11), f"COR {self._correlation:+.2f}", fill=tc, font=font_s)

        # Channel labels below meters
        bar_x0_m = self._gonio_cx + self._gonio_r + 14
        bar_w_m = 20
        gap = 6
        y_label = self._gonio_cy + self._gonio_r + 4
        if y_label < self.height:
            draw.text((bar_x0_m + bar_w_m // 2, y_label), "L", fill=tc, font=font_s, anchor="mt")
            draw.text((bar_x0_m + bar_w_m + gap + bar_w_m // 2, y_label), "R",
                      fill=tc, font=font_s, anchor="mt")

        # dB scale
        bar_y0 = self._gonio_cy - self._gonio_r
        bar_h = self._gonio_r * 2
        for db_val in range(int(self._db_min), int(self._db_max) + 1, 10):
            t = (db_val - self._db_min) / self._db_range
            y = int(bar_y0 + (1.0 - t) * bar_h)
            if bar_y0 <= y <= bar_y0 + bar_h:
                draw.text((bar_x0_m - 4, y), f"{db_val:+d}", fill=tc, font=font_s, anchor="rm")

    def _draw_title(self, draw: ImageDraw.ImageDraw) -> None:
        font_s, _ = get_fonts()
        draw.text((10, 3), "STEREOMETER", fill=self.theme.accent_color, font=font_s)
