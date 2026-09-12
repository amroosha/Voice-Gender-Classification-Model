import io

import torch
import torchaudio

from src.config import MIN_AUDIO_DURATION_S, SAMPLE_RATE

# torchaudio logs a warning about torchaudio.load() some parameters will be ignored we don't pass any of these paramters anyway, so we can ignore this warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

class AudioValidationError(ValueError):
    pass

import numpy as np


def load_audio_from_bytes(audio_bytes: bytes) -> tuple[torch.Tensor, int]:
    """
    Loads audio from bytes using torchaudio with robust fallbacks (soundfile, standard wave).
    Raises AudioValidationError if the audio is invalid.
    """
    if not audio_bytes:
        raise AudioValidationError("Audio payload is empty")

    last_exc: Exception | None = None

    # 1. Primary decoder: torchaudio in-memory stream
    try:
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
        if waveform.numel() > 0:
            return waveform, sample_rate
    except Exception as exc:
        last_exc = exc

    # 2. Fallback: torchaudio with explicit format="wav" hint
    if audio_bytes[:4] == b"RIFF":
        try:
            waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes), format="wav")
            if waveform.numel() > 0:
                return waveform, sample_rate
        except Exception:
            pass

    # 3. Fallback: soundfile (handles 24-bit PCM, 32-bit float, and extensible WAV headers)
    try:
        import soundfile as sf

        data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if data.ndim == 1:
            waveform = torch.from_numpy(data).unsqueeze(0)
        else:
            waveform = torch.from_numpy(data.T)
        if waveform.numel() > 0:
            return waveform, sample_rate
    except Exception:
        pass

    # 4. Fallback: Python standard library wave module
    if audio_bytes[:4] == b"RIFF":
        try:
            import wave

            with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                raw_data = wf.readframes(n_frames)

                if sampwidth == 2:
                    dtype = np.int16
                    scale = 32768.0
                elif sampwidth == 1:
                    dtype = np.uint8
                    scale = 128.0
                elif sampwidth == 4:
                    dtype = np.int32
                    scale = 2147483648.0
                else:
                    raise ValueError(f"Unsupported bit depth: {sampwidth * 8}-bit")

                arr = np.frombuffer(raw_data, dtype=dtype).astype(np.float32)
                if sampwidth == 1:
                    arr = arr - 128.0
                arr = arr / scale
                arr = arr.reshape(-1, n_channels).T
                waveform = torch.from_numpy(arr)
                if waveform.numel() > 0:
                    return waveform, sample_rate
        except Exception:
            pass

    raise AudioValidationError(f"Unable to decode audio ({last_exc})")



def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """
    checks if audio is single channel.
    if not converts to mono by averaging channels.
    """

    if waveform.ndim != 2:
        raise AudioValidationError("Waveform must be 2D [channels, samples]")

    if waveform.shape[0] == 1:
        return waveform
    
    return waveform.mean(dim=0, keepdim=True)



def resample_audio(waveform: torch.Tensor, sample_rate: int, target_sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """
    resamples audio to the target sample rate if it is not already at that rate.
    """

    if sample_rate == target_sample_rate:
        return waveform
    return torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)


def validate_min_duration(waveform: torch.Tensor, sample_rate: int, min_duration_s: float = MIN_AUDIO_DURATION_S) -> None:
    """
    Validates that the audio clip is at least the minimum required duration.
    """
    duration_s = waveform.shape[1] / float(sample_rate)
    if duration_s < min_duration_s:
        raise AudioValidationError(
            f"Audio clip is too short ({duration_s:.3f}s). Minimum required is {min_duration_s:.3f}s"
        )


def prepare_audio_bytes( audio_bytes: bytes, target_sample_rate: int = SAMPLE_RATE, min_duration_s: float = MIN_AUDIO_DURATION_S ) -> torch.Tensor:
    """
    entire audio preprocessing pipeline for a single audio clip in bytes format.
    """
    waveform, sample_rate = load_audio_from_bytes(audio_bytes)
    waveform = to_mono(waveform)
    waveform = resample_audio(waveform, sample_rate, target_sample_rate)
    validate_min_duration(waveform, target_sample_rate, min_duration_s)
    return waveform
