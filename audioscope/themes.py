"""Color themes for audioscope visualizations."""
from __future__ import annotations
from typing import List, Tuple
import numpy as np

ColorRGB = Tuple[int, int, int]
GradientStop = Tuple[float, ColorRGB]


def interpolate_color(c1: ColorRGB, c2: ColorRGB, t: float) -> ColorRGB:
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def gradient_color(stops: List[GradientStop], t: float) -> ColorRGB:
    t = max(0.0, min(1.0, t))
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t0 <= t <= t1:
            local_t = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return interpolate_color(c0, c1, local_t)
    return stops[-1][1]


def build_colormap(stops: List[GradientStop], n: int = 256) -> np.ndarray:
    """Build a (n, 3) uint8 colormap from gradient stops."""
    cmap = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        t = i / (n - 1)
        c = gradient_color(stops, t)
        cmap[i] = c
    return cmap


class Theme:
    def __init__(
        self,
        name: str,
        background: ColorRGB,
        grid_color: ColorRGB,
        text_color: ColorRGB,
        border_color: ColorRGB,
        accent_color: ColorRGB,
        peak_color: ColorRGB,
        clip_color: ColorRGB,
        spectrum_stops: List[GradientStop],
        waveform_stops: List[GradientStop],
        spectrogram_stops: List[GradientStop],
        level_meter_stops: List[GradientStop],
        stereo_dot_color: ColorRGB,
        stereo_trail_color: ColorRGB,
        correlation_pos_color: ColorRGB,
        correlation_neg_color: ColorRGB,
    ):
        self.name = name
        self.background = background
        self.grid_color = grid_color
        self.text_color = text_color
        self.border_color = border_color
        self.accent_color = accent_color
        self.peak_color = peak_color
        self.clip_color = clip_color
        self.spectrum_stops = spectrum_stops
        self.waveform_stops = waveform_stops
        self.spectrogram_stops = spectrogram_stops
        self.level_meter_stops = level_meter_stops
        self.stereo_dot_color = stereo_dot_color
        self.stereo_trail_color = stereo_trail_color
        self.correlation_pos_color = correlation_pos_color
        self.correlation_neg_color = correlation_neg_color

        # Pre-built colormaps for fast per-pixel lookup
        self.spectrum_cmap = build_colormap(spectrum_stops)
        self.spectrogram_cmap = build_colormap(spectrogram_stops)
        self.level_cmap = build_colormap(level_meter_stops)
        self.waveform_cmap = build_colormap(waveform_stops)

    def spectrum_color_np(self, t_array: np.ndarray) -> np.ndarray:
        """Map 0-1 array to RGB array using spectrum colormap."""
        idx = np.clip((t_array * 255).astype(np.int32), 0, 255)
        return self.spectrum_cmap[idx]

    def spectrogram_color_np(self, t_array: np.ndarray) -> np.ndarray:
        idx = np.clip((t_array * 255).astype(np.int32), 0, 255)
        return self.spectrogram_cmap[idx]

    def level_color_np(self, t_array: np.ndarray) -> np.ndarray:
        idx = np.clip((t_array * 255).astype(np.int32), 0, 255)
        return self.level_cmap[idx]


# ---------------------------------------------------------------------------
# Theme definitions
# ---------------------------------------------------------------------------

THEME_NEON = Theme(
    name="neon",
    background=(8, 0, 18),
    grid_color=(28, 8, 50),
    text_color=(190, 140, 255),
    border_color=(60, 20, 100),
    accent_color=(0, 220, 180),
    peak_color=(255, 215, 0),
    clip_color=(255, 30, 30),
    spectrum_stops=[
        (0.00, (15, 0, 40)),
        (0.35, (80, 0, 170)),
        (0.60, (200, 0, 200)),
        (0.78, (255, 0, 120)),
        (0.92, (255, 90, 0)),
        (1.00, (255, 230, 20)),
    ],
    waveform_stops=[
        (0.00, (0, 50, 70)),
        (0.45, (0, 180, 150)),
        (1.00, (120, 255, 230)),
    ],
    spectrogram_stops=[
        (0.00, (8, 0, 18)),
        (0.10, (30, 0, 65)),
        (0.28, (85, 0, 165)),
        (0.48, (200, 0, 155)),
        (0.68, (255, 40, 0)),
        (0.84, (255, 175, 0)),
        (1.00, (255, 255, 200)),
    ],
    level_meter_stops=[
        (0.00, (0, 100, 70)),
        (0.65, (0, 210, 90)),
        (0.83, (210, 210, 0)),
        (0.93, (255, 110, 0)),
        (1.00, (255, 30, 30)),
    ],
    stereo_dot_color=(140, 60, 255),
    stereo_trail_color=(50, 0, 130),
    correlation_pos_color=(0, 200, 140),
    correlation_neg_color=(255, 60, 0),
)

THEME_PLASMA = Theme(
    name="plasma",
    background=(2, 0, 14),
    grid_color=(8, 12, 45),
    text_color=(140, 180, 255),
    border_color=(30, 40, 100),
    accent_color=(0, 200, 255),
    peak_color=(255, 255, 0),
    clip_color=(255, 0, 0),
    spectrum_stops=[
        (0.00, (2, 0, 30)),
        (0.30, (0, 20, 160)),
        (0.55, (110, 0, 220)),
        (0.73, (230, 0, 175)),
        (0.88, (255, 50, 50)),
        (1.00, (255, 235, 0)),
    ],
    waveform_stops=[
        (0.00, (0, 0, 100)),
        (0.50, (80, 0, 220)),
        (1.00, (200, 80, 255)),
    ],
    spectrogram_stops=[
        (0.00, (2, 0, 14)),
        (0.12, (8, 0, 60)),
        (0.30, (0, 18, 165)),
        (0.52, (130, 0, 200)),
        (0.72, (255, 0, 75)),
        (0.88, (255, 165, 0)),
        (1.00, (255, 255, 0)),
    ],
    level_meter_stops=[
        (0.00, (0, 70, 160)),
        (0.65, (0, 180, 255)),
        (0.83, (180, 0, 255)),
        (0.93, (255, 80, 0)),
        (1.00, (255, 0, 0)),
    ],
    stereo_dot_color=(0, 160, 255),
    stereo_trail_color=(0, 30, 120),
    correlation_pos_color=(0, 180, 255),
    correlation_neg_color=(255, 0, 100),
)

THEME_MATRIX = Theme(
    name="matrix",
    background=(10, 10, 10),
    grid_color=(25, 25, 25),
    text_color=(180, 180, 180),
    border_color=(50, 50, 50),
    accent_color=(255, 255, 255),
    peak_color=(255, 220, 0),
    clip_color=(255, 0, 0),
    spectrum_stops=[
        (0.00, (0, 40, 0)),
        (0.50, (0, 180, 0)),
        (0.75, (220, 220, 0)),
        (0.90, (255, 130, 0)),
        (1.00, (255, 0, 0)),
    ],
    waveform_stops=[
        (0.00, (0, 70, 0)),
        (0.50, (0, 210, 0)),
        (1.00, (120, 255, 120)),
    ],
    spectrogram_stops=[
        (0.00, (10, 10, 10)),
        (0.18, (0, 55, 0)),
        (0.45, (0, 185, 0)),
        (0.72, (220, 220, 0)),
        (0.88, (255, 110, 0)),
        (1.00, (255, 255, 255)),
    ],
    level_meter_stops=[
        (0.00, (0, 120, 0)),
        (0.65, (0, 230, 0)),
        (0.83, (230, 230, 0)),
        (0.93, (255, 100, 0)),
        (1.00, (255, 0, 0)),
    ],
    stereo_dot_color=(200, 200, 200),
    stereo_trail_color=(80, 80, 80),
    correlation_pos_color=(0, 220, 0),
    correlation_neg_color=(255, 50, 50),
)

THEME_EMBER = Theme(
    name="ember",
    background=(12, 4, 0),
    grid_color=(35, 12, 0),
    text_color=(255, 190, 120),
    border_color=(80, 30, 0),
    accent_color=(255, 200, 0),
    peak_color=(255, 255, 200),
    clip_color=(255, 255, 255),
    spectrum_stops=[
        (0.00, (30, 5, 0)),
        (0.35, (150, 30, 0)),
        (0.60, (220, 80, 0)),
        (0.78, (255, 150, 0)),
        (0.92, (255, 230, 0)),
        (1.00, (255, 255, 220)),
    ],
    waveform_stops=[
        (0.00, (80, 20, 0)),
        (0.50, (220, 80, 0)),
        (1.00, (255, 200, 60)),
    ],
    spectrogram_stops=[
        (0.00, (12, 4, 0)),
        (0.12, (50, 10, 0)),
        (0.30, (140, 30, 0)),
        (0.52, (220, 80, 0)),
        (0.72, (255, 160, 0)),
        (0.88, (255, 230, 0)),
        (1.00, (255, 255, 200)),
    ],
    level_meter_stops=[
        (0.00, (80, 15, 0)),
        (0.65, (200, 70, 0)),
        (0.83, (255, 160, 0)),
        (0.93, (255, 230, 0)),
        (1.00, (255, 255, 200)),
    ],
    stereo_dot_color=(255, 150, 0),
    stereo_trail_color=(100, 30, 0),
    correlation_pos_color=(255, 200, 0),
    correlation_neg_color=(255, 50, 0),
)

THEME_RED = Theme(
    name="red",
    background=(8, 0, 0),
    grid_color=(30, 6, 6),
    text_color=(255, 110, 110),
    border_color=(75, 12, 12),
    accent_color=(255, 40, 40),
    peak_color=(255, 255, 200),
    clip_color=(255, 255, 255),
    spectrum_stops=[
        (0.00, (18, 0, 0)),
        (0.30, (110, 0, 0)),
        (0.55, (210, 0, 0)),
        (0.72, (255, 50, 0)),
        (0.88, (255, 140, 60)),
        (1.00, (255, 245, 220)),
    ],
    waveform_stops=[
        (0.00, (55, 0, 0)),
        (0.50, (190, 15, 15)),
        (1.00, (255, 110, 90)),
    ],
    spectrogram_stops=[
        (0.00, (8, 0, 0)),
        (0.14, (45, 0, 0)),
        (0.32, (130, 0, 0)),
        (0.52, (220, 0, 0)),
        (0.72, (255, 70, 10)),
        (0.88, (255, 170, 60)),
        (1.00, (255, 250, 210)),
    ],
    level_meter_stops=[
        (0.00, (70, 8, 8)),
        (0.65, (200, 20, 20)),
        (0.83, (255, 90, 0)),
        (0.93, (255, 210, 0)),
        (1.00, (255, 255, 255)),
    ],
    stereo_dot_color=(255, 50, 50),
    stereo_trail_color=(90, 8, 8),
    correlation_pos_color=(255, 70, 70),
    correlation_neg_color=(180, 0, 0),
)

THEMES = {
    "neon": THEME_NEON,
    "plasma": THEME_PLASMA,
    "matrix": THEME_MATRIX,
    "ember": THEME_EMBER,
    "red": THEME_RED,
}
