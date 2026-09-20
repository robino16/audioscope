"""Unit tests for AudioFile."""
from __future__ import annotations
import io
import struct
import wave
import numpy as np
import pytest
import soundfile as sf
import tempfile
from pathlib import Path

from audioscope.audio import AudioFile


# ------------------------------------------------------------------ fixtures

def _make_wav(samples: np.ndarray, sr: int, path: Path) -> None:
    """Write a float32 stereo or mono WAV to path using soundfile."""
    if samples.ndim == 1:
        samples = samples[:, None]
    sf.write(path, samples, sr, subtype="PCM_16")


@pytest.fixture
def stereo_wav(tmp_path):
    sr = 44100
    dur = 1.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    left = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5
    right = np.sin(2 * np.pi * 880 * t).astype(np.float32) * 0.5
    samples = np.column_stack([left, right])
    path = tmp_path / "stereo.wav"
    sf.write(path, samples, sr)
    return path, sr, samples


@pytest.fixture
def mono_wav(tmp_path):
    sr = 22050
    dur = 0.5
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    sig = np.sin(2 * np.pi * 200 * t).astype(np.float32) * 0.3
    path = tmp_path / "mono.wav"
    sf.write(path, sig, sr)
    return path, sr, sig


# ------------------------------------------------------------------ tests

class TestAudioFile:
    def test_load_stereo(self, stereo_wav):
        path, sr, samples = stereo_wav
        af = AudioFile(path)
        assert af.sample_rate == sr
        assert af.n_channels == 2
        assert af.is_stereo
        assert abs(af.duration - 1.0) < 0.01

    def test_load_mono(self, mono_wav):
        path, sr, sig = mono_wav
        af = AudioFile(path)
        assert af.sample_rate == sr
        assert af.n_channels == 1
        assert not af.is_stereo
        assert abs(af.duration - 0.5) < 0.01

    def test_get_mono_stereo_input(self, stereo_wav):
        path, sr, samples = stereo_wav
        af = AudioFile(path)
        mono = af.get_mono()
        expected = np.mean(samples, axis=1)
        np.testing.assert_allclose(mono, expected, atol=1e-4)

    def test_get_left_right(self, stereo_wav):
        path, sr, samples = stereo_wav
        af = AudioFile(path)
        np.testing.assert_allclose(af.get_left(), samples[:, 0], atol=1e-4)
        np.testing.assert_allclose(af.get_right(), samples[:, 1], atol=1e-4)

    def test_get_window_returns_correct_length(self, stereo_wav):
        path, sr, _ = stereo_wav
        af = AudioFile(path)
        window_secs = 0.1
        for fi in [0, 5, 10]:
            l, r = af.get_window(fi, fps=24.0, window_seconds=window_secs)
            expected_n = int(window_secs * sr)
            assert len(l) == expected_n
            assert len(r) == expected_n

    def test_get_window_zero_pads_at_start(self, stereo_wav):
        path, sr, _ = stereo_wav
        af = AudioFile(path)
        # Frame 0 with a large window should be zero-padded at the front
        l, r = af.get_window(0, fps=24.0, window_seconds=1.0)
        assert len(l) == sr
        # The first sample of frame 0 at trailing=True means the window
        # ends at frame 1's boundary (~1842 samples); 1s window exceeds that,
        # so most of the left side should be silent.
        assert np.all(l[:sr - 2000] == 0.0)

    def test_get_frame_slice_length(self, stereo_wav):
        path, sr, _ = stereo_wav
        af = AudioFile(path)
        fps = 24.0
        spf = int(sr / fps)
        l, r = af.get_frame_slice(0, fps)
        assert len(l) == spf
        assert len(r) == spf

    def test_normalize(self, stereo_wav):
        path, sr, _ = stereo_wav
        af = AudioFile(path)
        af.normalize()
        assert np.max(np.abs(af.get_left())) <= 1.0 + 1e-6
        assert np.max(np.abs(af.get_right())) <= 1.0 + 1e-6

    def test_gain_positive(self, stereo_wav):
        path, sr, _ = stereo_wav
        af = AudioFile(path)
        before = float(np.max(np.abs(af.get_left())))
        af.apply_gain_db(6.0)
        after = float(np.max(np.abs(af.get_left())))
        assert abs(after / before - 2.0) < 0.01

    def test_gain_negative(self, stereo_wav):
        path, sr, _ = stereo_wav
        af = AudioFile(path)
        before = float(np.max(np.abs(af.get_left())))
        af.apply_gain_db(-6.0)
        after = float(np.max(np.abs(af.get_left())))
        assert abs(after / before - 0.5) < 0.01

    def test_repr(self, stereo_wav):
        path, sr, _ = stereo_wav
        af = AudioFile(path)
        r = repr(af)
        assert "stereo" in r.lower() or "2ch" in r
