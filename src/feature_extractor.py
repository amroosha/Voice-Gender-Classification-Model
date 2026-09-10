import numpy as np
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from src.audio_utils import prepare_audio_bytes
from src.config import DEVICE, SAMPLE_RATE, WAV2VEC2_LAYER, WAV2VEC2_MODEL_NAME, WAV2VEC2_REVISION


class Wav2Vec2FeatureExtractor:
    def __init__(
        self,
        model_name: str = WAV2VEC2_MODEL_NAME,
        revision: str = WAV2VEC2_REVISION,
        layer: int = WAV2VEC2_LAYER,
        device: str = DEVICE,
        model: Wav2Vec2Model | None = None,
        processor: Wav2Vec2Processor | None = None,
    ) -> None:
        self.layer = layer
        self.device = torch.device(device)
        self.processor = processor or Wav2Vec2Processor.from_pretrained(model_name, revision=revision)
        self.model = model or Wav2Vec2Model.from_pretrained(model_name, revision=revision)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.to(self.device)

    def extract_embedding(self, waveform: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        if waveform.ndim != 2 or waveform.shape[0] != 1:
            raise ValueError("Expected mono waveform shape [1, samples]")
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Expected sample rate {SAMPLE_RATE}, got {sample_rate}")

        inputs = self.processor(
            waveform.squeeze(0).cpu().numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model(
                input_values=inputs.input_values.to(self.device),
                attention_mask=inputs.get("attention_mask", None).to(self.device)
                if inputs.get("attention_mask", None) is not None
                else None,
                output_hidden_states=True,
            )
        hidden_state = outputs.hidden_states[self.layer]
        embedding = hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
        return embedding

    def extract_from_bytes(self, audio_bytes: bytes) -> np.ndarray:
        waveform = prepare_audio_bytes(audio_bytes)
        return self.extract_embedding(waveform, SAMPLE_RATE)
