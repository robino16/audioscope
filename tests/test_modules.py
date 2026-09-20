"""Unit tests for visualization modules."""
from __future__ import annotations
import numpy as np
import pytest
from PIL import Image

from audioscope.themes import THEME_NEON, THEME_DARK
from audioscope.modules.spectrum import SpectrumModule
from audioscope.modules.waveform import WaveformModule
from audioscope.modules.spectrogram import SpectrogramModule
from audioscope.modules.stereometer import StereometerModule
from audioscope.modules.spectrogram3d import Spectrogram3DModule
from audioscope.utils import (
    compute_fft,
    log_freq_bins,
    db_to_normalized,
    smooth_exp,
    rms_db,
    peak_db,
    stereo_correlation,
    resample_linear,
)


SR = 44100
FPS = 24.0
HEIGHT = 200
FRAME_N = int(SR * 0.5)  # half-second window


def _sine(freq: float, n: int, amp: float = 0.5) -> np.ndarray:
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)


def _noise(n: int, amp: float = 0.2) -> np.ndarray:
    return (np.random.randn(n) * amp).astype(np.float32)


def _stereo(n: int):
    left = _sine(440, n)
    right = _sine(880, n, amp=0.4) + _noise(n, amp=0.05)
    return left, right


# ------------------------------------------------------------------ utils

class TestUtils:
    def test_compute_fft_shape(self):
        sig = _sine(440, SR)
        freqs, mags = compute_fft(sig, 4096, SR)
        assert len(freqs) == 4096 // 2 + 1
        assert len(mags) == len(freqs)

    def test_compute_fft_peak_near_440(self):
        sig = _sine(440, SR, amp=0.5)
        freqs, mags = compute_fft(sig, 4096, SR)
        peak_freq = freqs[np.argmax(mags)]
        assert abs(peak_freq - 440) < 20, f"Peak at {peak_freq:.0f} Hz, expected ~440 Hz"

    def test_log_freq_bins_length(self):
        sig = _sine(1000, SR)
        freqs, mags = compute_fft(sig, 4096, SR)
        _, bin_mags = log_freq_bins(freqs, mags, 128)
        assert len(bin_mags) == 128

    def test_db_to_normalized_clamps(self):
        db = np.array([-100.0, -80.0, -40.0, 0.0, 10.0])
        n = db_to_normalized(db, -80.0, 0.0)
        assert n[0] == 0.0  # below floor
        assert n[-1] == 1.0  # above ceiling
        assert 0.0 < n[2] < 1.0

    def test_smooth_exp_identity_at_zero(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = smooth_exp(a, b, alpha=0.0)
        np.testing.assert_array_almost_equal(result, b)

    def test_smooth_exp_frozen_at_one(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = smooth_exp(a, b, alpha=1.0)
        np.testing.assert_array_almost_equal(result, a)

    def test_rms_db_sine(self):
        sig = _sine(440, SR, amp=0.5)
        rms = rms_db(sig)
        # 0.5 amplitude sine → RMS = 0.5/sqrt(2) ≈ 0.354 → ~-9 dB
        assert abs(rms - (-9.03)) < 1.0

    def test_peak_db(self):
        sig = _sine(440, SR, amp=0.5)
        pk = peak_db(sig)
        assert abs(pk - (-6.02)) < 1.0  # 0.5 → -6 dB

    def test_stereo_correlation_identical(self):
        sig = _sine(440, SR)
        assert abs(stereo_correlation(sig, sig) - 1.0) < 0.001

    def test_stereo_correlation_inverted(self):
        sig = _sine(440, SR)
        corr = stereo_correlation(sig, -sig)
        assert abs(corr - (-1.0)) < 0.001

    def test_resample_linear_same_length(self):
        arr = np.arange(10, dtype=float)
        out = resample_linear(arr, 10)
        np.testing.assert_array_equal(out, arr)

    def test_resample_linear_upsample(self):
        arr = np.array([0.0, 1.0])
        out = resample_linear(arr, 5)
        assert len(out) == 5
        np.testing.assert_allclose(out[0], 0.0)
        np.testing.assert_allclose(out[-1], 1.0)


# ------------------------------------------------------------------ modules

class TestSpectrumModule:
    def _make(self, **kwargs):
        cfg = {"fft_size": 1024, **kwargs}
        return SpectrumModule(int(HEIGHT * 2), HEIGHT, THEME_NEON, cfg)

    def test_render_returns_correct_size(self):
        mod = self._make()
        left, right = _stereo(FRAME_N)
        mod.process_frame(left, right, SR, 0, FPS)
        img = mod.render()
        assert isinstance(img, Image.Image)
        assert img.size == (int(HEIGHT * 2), HEIGHT)

    def test_multiple_frames_no_crash(self):
        mod = self._make()
        for i in range(10):
            left, right = _stereo(FRAME_N)
            mod.process_frame(left, right, SR, i, FPS)
        img = mod.render()
        assert img is not None

    def test_silent_input(self):
        mod = self._make()
        silence = np.zeros(FRAME_N, dtype=np.float32)
        mod.process_frame(silence, silence, SR, 0, FPS)
        img = mod.render()
        arr = np.array(img)
        # Most pixels should be background color
        bg = THEME_NEON.background
        bg_count = np.sum(
            (arr[:, :, 0] == bg[0]) & (arr[:, :, 1] == bg[1]) & (arr[:, :, 2] == bg[2])
        )
        total = arr.shape[0] * arr.shape[1]
        assert bg_count / total > 0.5

    def test_themes(self):
        for theme in [THEME_NEON, THEME_DARK]:
            mod = SpectrumModule(int(HEIGHT * 2), HEIGHT, theme, {"fft_size": 1024})
            left, right = _stereo(FRAME_N)
            mod.process_frame(left, right, SR, 0, FPS)
            img = mod.render()
            assert isinstance(img, Image.Image)


class TestWaveformModule:
    def _make(self, **kwargs):
        cfg = {"duration_secs": 0.2, **kwargs}
        return WaveformModule(int(HEIGHT * 2.5), HEIGHT, THEME_NEON, cfg)

    def test_render_size(self):
        mod = self._make()
        left, right = _stereo(FRAME_N)
        mod.process_frame(left, right, SR, 0, FPS)
        img = mod.render()
        assert img.size == (int(HEIGHT * 2.5), HEIGHT)

    def test_mono_mode(self):
        mod = WaveformModule(int(HEIGHT * 2.5), HEIGHT, THEME_NEON, {"stereo": False})
        left, right = _stereo(FRAME_N)
        mod.process_frame(left, right, SR, 0, FPS)
        img = mod.render()
        assert img is not None

    def test_silent_no_crash(self):
        mod = self._make()
        silence = np.zeros(FRAME_N, dtype=np.float32)
        mod.process_frame(silence, silence, SR, 0, FPS)
        img = mod.render()
        assert img is not None


class TestSpectrogramModule:
    def _make(self, **kwargs):
        cfg = {"fft_size": 1024, **kwargs}
        return SpectrogramModule(int(HEIGHT * 2), HEIGHT, THEME_NEON, cfg)

    def test_render_size(self):
        mod = self._make()
        left, right = _stereo(FRAME_N)
        mod.process_frame(left, right, SR, 0, FPS)
        img = mod.render()
        assert img.size == (int(HEIGHT * 2), HEIGHT)

    def test_accumulation(self):
        mod = self._make()
        for i in range(20):
            left, right = _stereo(FRAME_N)
            mod.process_frame(left, right, SR, i, FPS)
        img = mod.render()
        arr = np.array(img)
        # After 20 frames with signal, image should not be all-background
        bg = THEME_NEON.background
        bg_count = np.sum(
            (arr[:, :, 0] == bg[0]) & (arr[:, :, 1] == bg[1]) & (arr[:, :, 2] == bg[2])
        )
        total = arr.shape[0] * arr.shape[1]
        assert bg_count / total < 0.95


class TestStereometerModule:
    def _make(self, **kwargs):
        return StereometerModule(int(HEIGHT * 1.5), HEIGHT, THEME_NEON, kwargs)

    def test_render_size(self):
        mod = self._make()
        left, right = _stereo(FRAME_N)
        mod.process_frame(left, right, SR, 0, FPS)
        img = mod.render()
        assert img.size == (int(HEIGHT * 1.5), HEIGHT)

    def test_render_mono_input(self):
        mod = self._make()
        sig = _sine(440, FRAME_N)
        mod.process_frame(sig, sig, SR, 0, FPS)
        img = mod.render()
        assert img is not None

    def test_multiple_frames(self):
        mod = self._make()
        for i in range(5):
            left, right = _stereo(FRAME_N)
            mod.process_frame(left, right, SR, i, FPS)
        img = mod.render()
        assert img is not None


class TestSpectrogram3DModule:
    def _make(self, **kwargs):
        cfg = {"fft_size": 1024, "n_slices": 12, "n_freq_bins": 64, **kwargs}
        return Spectrogram3DModule(int(HEIGHT * 2), HEIGHT, THEME_NEON, cfg)

    def test_render_size(self):
        mod = self._make()
        left, right = _stereo(FRAME_N)
        mod.process_frame(left, right, SR, 0, FPS)
        img = mod.render()
        assert img.size == (int(HEIGHT * 2), HEIGHT)

    def test_empty_first_render(self):
        mod = self._make()
        img = mod.render()
        assert img is not None

    def test_accumulates_slices(self):
        mod = self._make()
        for i in range(15):
            left, right = _stereo(FRAME_N)
            mod.process_frame(left, right, SR, i, FPS)
        img = mod.render()
        arr = np.array(img)
        bg = THEME_NEON.background
        bg_count = np.sum(
            (arr[:, :, 0] == bg[0]) & (arr[:, :, 1] == bg[1]) & (arr[:, :, 2] == bg[2])
        )
        total = arr.shape[0] * arr.shape[1]
        assert bg_count / total < 0.95


# ------------------------------------------------------------------ renderer integration

class TestRendererIntegration:
    def test_full_render_cycle(self, tmp_path):
        """Full end-to-end test without CLI."""
        import soundfile as sf
        from audioscope.audio import AudioFile
        from audioscope.renderer import build_modules, Renderer

        sr = 22050
        dur = 0.5
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        sig = np.column_stack([
            np.sin(2 * np.pi * 440 * t).astype(np.float32),
            np.sin(2 * np.pi * 880 * t).astype(np.float32),
        ])
        wav_path = tmp_path / "test.wav"
        sf.write(wav_path, sig, sr)

        audio = AudioFile(wav_path)
        modules = build_modules(
            ["spectrum", "waveform"],
            height=100,
            theme=THEME_NEON,
            module_configs={
                "spectrum": {"fft_size": 512},
                "waveform": {"duration_secs": 0.1},
            },
        )
        renderer = Renderer(audio, modules, fps=12.0, height=100, window_secs=0.1)
        out = tmp_path / "frames"
        renderer.render_all(out)

        frames = list(out.glob("*.png"))
        assert len(frames) == renderer.total_frames
        assert renderer.total_frames > 0

        # Verify output images are correct size
        img = Image.open(frames[0])
        assert img.height == 100
        assert img.width > 0
