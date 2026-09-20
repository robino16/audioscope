from audioscope.modules.spectrum import SpectrumModule
from audioscope.modules.waveform import WaveformModule
from audioscope.modules.spectrogram import SpectrogramModule
from audioscope.modules.stereometer import StereometerModule
from audioscope.modules.spectrogram3d import Spectrogram3DModule

REGISTRY = {
    "spectrum": SpectrumModule,
    "waveform": WaveformModule,
    "spectrogram": SpectrogramModule,
    "stereometer": StereometerModule,
    "spectrogram3d": Spectrogram3DModule,
}

__all__ = [
    "SpectrumModule",
    "WaveformModule",
    "SpectrogramModule",
    "StereometerModule",
    "Spectrogram3DModule",
    "REGISTRY",
]
