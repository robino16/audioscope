"""Abstract base class for all visualization modules."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _try_load_font(size: int):
    """Return a PIL font, falling back to the default bitmap font."""
    try:
        import os
        font_paths = [
            "C:/Windows/Fonts/consola.ttf",
            "C:/Windows/Fonts/cour.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        ]
        for p in font_paths:
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    except Exception:
        pass
    return ImageFont.load_default()


FONT_SMALL = None
FONT_MED = None


def get_fonts():
    global FONT_SMALL, FONT_MED
    if FONT_SMALL is None:
        FONT_SMALL = _try_load_font(11)
        FONT_MED = _try_load_font(13)
    return FONT_SMALL, FONT_MED


class BaseModule(ABC):
    name: str = "base"
    default_aspect: float = 2.0

    def __init__(
        self,
        width: int,
        height: int,
        theme,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.width = width
        self.height = height
        self.theme = theme
        self.config = config or {}
        self._canvas = np.full((height, width, 3), theme.background, dtype=np.uint8)
        self._setup()

    def _setup(self) -> None:
        """One-time initialization. Override in subclasses."""
        pass

    @abstractmethod
    def process_frame(
        self,
        left: np.ndarray,
        right: np.ndarray,
        sample_rate: int,
        frame_index: int,
        fps: float,
    ) -> None:
        """Process one frame of audio data, updating internal state."""

    @abstractmethod
    def render(self) -> Image.Image:
        """Render internal state to a PIL Image of size (width, height)."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clear(self) -> None:
        self._canvas[:] = self.theme.background

    def _to_image(self) -> Image.Image:
        return Image.fromarray(self._canvas, mode="RGB")

    def _draw_hline(self, y: int, color: tuple, x0: int = 0, x1: Optional[int] = None) -> None:
        x1 = x1 if x1 is not None else self.width
        if 0 <= y < self.height:
            self._canvas[y, x0:x1] = color

    def _draw_vline(self, x: int, color: tuple, y0: int = 0, y1: Optional[int] = None) -> None:
        y1 = y1 if y1 is not None else self.height
        if 0 <= x < self.width:
            self._canvas[y0:y1, x] = color

    def _draw_rect_border(self, x0: int, y0: int, x1: int, y1: int, color: tuple) -> None:
        self._draw_hline(y0, color, x0, x1)
        self._draw_hline(y1 - 1, color, x0, x1)
        self._draw_vline(x0, color, y0, y1)
        self._draw_vline(x1 - 1, color, y0, y1)

    def _label_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        x: int,
        y: int,
        color: Optional[tuple] = None,
        anchor: str = "lt",
    ) -> None:
        font_s, _ = get_fonts()
        color = color or self.theme.text_color
        draw.text((x, y), text, fill=color, font=font_s, anchor=anchor)
