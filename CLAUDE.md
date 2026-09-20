# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Install in editable mode (required before using the CLI):
```
pip install -e .
```

Run all tests:
```
pytest
```

Run a single test or class:
```
pytest tests/test_modules.py::TestSpectrumModule::test_render_returns_correct_size
pytest tests/test_audio.py
```

Run the CLI:
```
visualize -i data/input.wav -o output/frames
visualize -i data/input.wav -o output/frames -m spectrum,waveform,stereometer --theme plasma --fps 30
```

## Architecture

The pipeline is: load audio → build modules → render all frames → save PNGs.

**`audioscope/audio.py`** — `AudioFile` wraps a WAV file loaded via `soundfile`. The key method is `get_window(frame_index, fps, window_seconds)` which returns a `(left, right)` numpy float32 array pair representing the trailing audio window for a given frame. This is what each module receives per frame.

**`audioscope/modules/`** — Each visualization module inherits `BaseModule` and implements two methods:
- `process_frame(left, right, sample_rate, frame_index, fps)` — update internal state from audio data
- `render() -> Image.Image` — draw the current state onto `self._canvas` (a `(height, width, 3)` uint8 ndarray) and return it as a PIL Image via `self._to_image()`

The `REGISTRY` dict in `modules/__init__.py` maps CLI names (`"spectrum"`, `"waveform"`, etc.) to module classes. To add a new module: create the class, set `name` and `default_aspect` class attributes, register it in `REGISTRY`.

**`audioscope/renderer.py`** — `build_modules()` instantiates the requested modules with computed widths (from `default_aspect × height` unless `--width` overrides). `Renderer.render_all()` iterates frames, calls `get_window` for each, dispatches to all modules, then composites their outputs side-by-side with a pixel separator.

**`audioscope/themes.py`** — `Theme` objects hold color tuples and gradient stop lists; they also pre-build 256-entry colormaps (`spectrum_cmap`, `spectrogram_cmap`, etc.) as numpy arrays for fast per-pixel index lookups. Available themes: `neon`, `plasma`, `dark`, `ember`.

**`audioscope/utils.py`** — Shared DSP helpers used by multiple modules: `compute_fft`, `log_freq_bins`, `db_to_normalized`, `smooth_exp`, `rms_db`, `peak_db`, `stereo_correlation`, `resample_linear`.

## Module config

Each module receives a `config` dict. Shared DSP parameters (`fft_size`, `window_func`, `db_range`, `min_freq`, `max_freq`, `show_labels`) are passed to all frequency-domain modules. Module-specific keys (e.g. `smoothing`, `peak_hold_secs`, `n_slices`) are defined per module. The CLI wires these in `cli.py:main()` under `module_configs`.
