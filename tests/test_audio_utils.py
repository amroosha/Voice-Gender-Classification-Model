import pytest
import torch

from src import audio_utils
from src.audio_utils import AudioValidationError, prepare_audio_bytes, resample_audio, to_mono, validate_min_duration


def test_resample_audio_changes_rate() -> None:
    waveform = torch.randn(1, 8_000)
    resampled = resample_audio(waveform, sample_rate=8_000, target_sample_rate=16_000)
    assert resampled.shape[1] > waveform.shape[1]


def test_to_mono_averages_channels() -> None:
    stereo = torch.tensor([[1.0, -1.0], [3.0, 1.0]])
    mono = to_mono(stereo)
    assert mono.shape == (1, 2)
    assert torch.allclose(mono, torch.tensor([[2.0, 0.0]]))


def test_validate_min_duration_rejects_short_audio() -> None:
    short = torch.zeros(1, 1_000)
    with pytest.raises(AudioValidationError):
        validate_min_duration(short, sample_rate=16_000, min_duration_s=0.5)


def test_prepare_audio_bytes_returns_mono_16khz(monkeypatch: pytest.MonkeyPatch) -> None:
    stereo = torch.randn(2, 8_000)
    monkeypatch.setattr(audio_utils, "load_audio_from_bytes", lambda _: (stereo, 8_000))
    prepared = prepare_audio_bytes(b"wav-bytes", target_sample_rate=16_000, min_duration_s=0.2)
    assert prepared.shape[0] == 1
    assert prepared.shape[1] >= 16_000
