import io

import torch
import torchaudio

from src.config import MIN_AUDIO_DURATION_S, SAMPLE_RATE


class AudioValidationError(ValueError):
    pass


def load_audio_from_bytes(audio_bytes: bytes) -> tuple[torch.Tensor, int]:
    if not audio_bytes:
        raise AudioValidationError("Audio payload is empty")
    try:
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
    except Exception as exc:  # noqa: BLE001
        raise AudioValidationError("Unable to decode audio") from exc
    if waveform.numel() == 0:
        raise AudioValidationError("Decoded audio contains no samples")
    return waveform, sample_rate


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim != 2:
        raise AudioValidationError("Waveform must be 2D [channels, samples]")
    if waveform.shape[0] == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)


def resample_audio(waveform: torch.Tensor, sample_rate: int, target_sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    if sample_rate == target_sample_rate:
        return waveform
    return torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)


def validate_min_duration(
    waveform: torch.Tensor,
    sample_rate: int,
    min_duration_s: float = MIN_AUDIO_DURATION_S,
) -> None:
    duration_s = waveform.shape[1] / float(sample_rate)
    if duration_s < min_duration_s:
        raise AudioValidationError(
            f"Audio clip is too short ({duration_s:.3f}s). Minimum required is {min_duration_s:.3f}s"
        )


def prepare_audio_bytes(
    audio_bytes: bytes,
    target_sample_rate: int = SAMPLE_RATE,
    min_duration_s: float = MIN_AUDIO_DURATION_S,
) -> torch.Tensor:
    waveform, sample_rate = load_audio_from_bytes(audio_bytes)
    waveform = to_mono(waveform)
    waveform = resample_audio(waveform, sample_rate, target_sample_rate)
    validate_min_duration(waveform, target_sample_rate, min_duration_s)
    return waveform
