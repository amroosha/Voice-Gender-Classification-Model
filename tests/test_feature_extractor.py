import types

import torch

from src.feature_extractor import Wav2Vec2FeatureExtractor


class DummyProcessor:
    def __call__(self, waveform, sampling_rate, return_tensors):
        return types.SimpleNamespace(input_values=torch.tensor(waveform).unsqueeze(0), get=lambda *_: None)


class DummyModel(torch.nn.Module):
    def forward(self, input_values, attention_mask=None, output_hidden_states=True):
        time_steps = input_values.shape[1] // 100 + 2
        hidden_states = []
        for layer_idx in range(13):
            hidden_states.append(torch.full((1, time_steps, 768), float(layer_idx)))
        return types.SimpleNamespace(hidden_states=tuple(hidden_states))


def test_embedding_shape_is_fixed() -> None:
    extractor = Wav2Vec2FeatureExtractor(model=DummyModel(), processor=DummyProcessor(), layer=6)
    waveform = torch.randn(1, 16_000)
    embedding = extractor.extract_embedding(waveform)
    assert embedding.shape == (768,)


def test_same_input_same_embedding() -> None:
    extractor = Wav2Vec2FeatureExtractor(model=DummyModel(), processor=DummyProcessor(), layer=6)
    waveform = torch.randn(1, 24_000)
    emb_1 = extractor.extract_embedding(waveform)
    emb_2 = extractor.extract_embedding(waveform)
    assert (emb_1 == emb_2).all()
