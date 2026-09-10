import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from src.audio_utils import AudioValidationError


class StubFeatureExtractor:
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail

    def extract_from_bytes(self, _: bytes):
        if self.should_fail:
            raise AudioValidationError("Unable to decode audio")
        return [0.1, 0.2, 0.3]


class StubPipeline:
    classes_ = ["female", "male"]

    def predict_proba(self, features):
        assert len(features) == 1
        return [[0.25, 0.75]]


def _wav_bytes() -> bytes:
    return b"synthetic-audio-bytes"


def test_predict_gender_endpoint() -> None:
    app = create_app(feature_extractor=StubFeatureExtractor(), classifier_pipeline=StubPipeline())
    client = TestClient(app)

    response = client.post(
        "/v1/predict-gender",
        files={"file": ("sample.wav", _wav_bytes(), "audio/wav")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_label"] == "male"
    assert payload["confidence_score"] == pytest.approx(0.75)
    assert payload["latency_ms"] >= 0


def test_predict_gender_decode_failure() -> None:
    app = create_app(feature_extractor=StubFeatureExtractor(should_fail=True), classifier_pipeline=StubPipeline())
    client = TestClient(app)

    response = client.post(
        "/v1/predict-gender",
        files={"file": ("broken.wav", b"not-audio", "audio/wav")},
    )

    assert response.status_code == 422
