"""Frame rendering orchestrator."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image

from audioscope.audio import AudioFile
from audioscope.themes import THEMES, Theme
from audioscope.modules import REGISTRY
from audioscope.modules.base import BaseModule


def _compute_width(module_name: str, height: int, override: Optional[int] = None) -> int:
    if override:
        return override
    cls = REGISTRY.get(module_name)
    aspect = cls.default_aspect if cls else 2.0
    return int(height * aspect)


def build_modules(
    module_names: List[str],
    height: int,
    theme: Theme,
    width_override: Optional[int] = None,
    module_configs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[BaseModule]:
    modules = []
    for name in module_names:
        cls = REGISTRY.get(name.lower())
        if cls is None:
            raise ValueError(
                f"Unknown module '{name}'. Available: {', '.join(REGISTRY.keys())}"
            )
        w = _compute_width(name, height, width_override)
        cfg = (module_configs or {}).get(name.lower(), {})
        modules.append(cls(w, height, theme, cfg))
    return modules


def render_frame(
    modules: List[BaseModule],
    left: np.ndarray,
    right: np.ndarray,
    sample_rate: int,
    frame_index: int,
    fps: float,
    height: int,
    separator: int = 2,
    separator_color: Optional[tuple] = None,
) -> Image.Image:
    """Process one frame and return the composited PIL Image."""
    for mod in modules:
        mod.process_frame(left, right, sample_rate, frame_index, fps)

    rendered = [mod.render() for mod in modules]

    sep_color = separator_color or (20, 20, 30)
    total_w = sum(img.width for img in rendered) + separator * (len(rendered) - 1)
    composite = Image.new("RGB", (total_w, height), sep_color)

    x = 0
    for i, img in enumerate(rendered):
        composite.paste(img, (x, 0))
        x += img.width
        if i < len(rendered) - 1:
            x += separator  # leave a gap (already filled with sep_color)

    return composite


class Renderer:
    def __init__(
        self,
        audio: AudioFile,
        modules: List[BaseModule],
        fps: float,
        height: int,
        window_secs: float = 0.5,
        separator: int = 2,
    ):
        self.audio = audio
        self.modules = modules
        self.fps = fps
        self.height = height
        self.window_secs = window_secs
        self.separator = separator

        self.total_frames = int(audio.duration * fps)

    def render_all(
        self,
        output_dir: Path,
        fmt: str = "png",
        quality: int = 95,
        progress_callback=None,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        digits = len(str(self.total_frames))

        for frame_i in range(self.total_frames):
            left, right = self.audio.get_window(
                frame_i, self.fps, self.window_secs, trailing=True
            )
            img = render_frame(
                self.modules,
                left,
                right,
                self.audio.sample_rate,
                frame_i,
                self.fps,
                self.height,
                self.separator,
            )

            fname = f"frame_{frame_i:0{digits}d}.{fmt}"
            save_kwargs = {}
            if fmt.lower() in ("jpg", "jpeg"):
                save_kwargs["quality"] = quality
            img.save(output_dir / fname, **save_kwargs)

            if progress_callback:
                progress_callback(frame_i + 1, self.total_frames)
