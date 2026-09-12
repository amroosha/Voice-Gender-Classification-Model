import time
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from app.schemas import GenderPredictionResponse, HealthResponse
from src.audio_utils import AudioValidationError
from src.config import MODEL_ARTIFACT_PATH
from src.feature_extractor import Wav2Vec2FeatureExtractor


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "feature_extractor"):
        app.state.feature_extractor = Wav2Vec2FeatureExtractor()
    if not hasattr(app.state, "classifier_pipeline"):
        app.state.classifier_pipeline = joblib.load(MODEL_ARTIFACT_PATH)
    yield


def create_app(feature_extractor=None, classifier_pipeline=None) -> FastAPI:
    app = FastAPI(title="Voice Gender Classification Service", version="1.0.0", lifespan=lifespan)

    if feature_extractor is not None:
        app.state.feature_extractor = feature_extractor
    if classifier_pipeline is not None:
        app.state.classifier_pipeline = classifier_pipeline

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.post("/v1/predict-gender", response_model=GenderPredictionResponse)
    async def predict_gender(file: UploadFile = File(...)) -> GenderPredictionResponse:
        start = time.perf_counter()
        audio_bytes = await file.read()

        # Ensure models are available
        if not hasattr(app.state, "feature_extractor"):
            app.state.feature_extractor = Wav2Vec2FeatureExtractor()
        if not hasattr(app.state, "classifier_pipeline"):
            app.state.classifier_pipeline = joblib.load(MODEL_ARTIFACT_PATH)

        try:
            embedding = app.state.feature_extractor.extract_from_bytes(audio_bytes)
        except AudioValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Feature extraction failed: {type(exc).__name__} - {exc}") from exc

        try:
            probabilities = app.state.classifier_pipeline.predict_proba([embedding])[0]
            class_idx = int(np.argmax(probabilities))

            # Map class indices (0 -> female, 1 -> male)
            raw_label = app.state.classifier_pipeline.classes_[class_idx]
            id_to_label = {0: "female", 1: "male"}
            label_name = id_to_label.get(raw_label, str(raw_label))

            # Clamp confidence to [0.0, 1.0] to satisfy Pydantic validation
            confidence = float(np.clip(probabilities[class_idx], 0.0, 1.0))
            latency_ms = (time.perf_counter() - start) * 1000

            return GenderPredictionResponse(
                predicted_label=label_name,
                confidence_score=confidence,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Inference failed: {type(exc).__name__} - {exc}") from exc

    return app


app = create_app()
