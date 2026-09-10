from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class GenderPredictionResponse(BaseModel):
    predicted_label: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)
