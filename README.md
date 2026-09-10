# Voice Gender Classification — Production Microservice

## Setup (uv)

```bash
uv sync --frozen
```

## Train + MLflow tracking

```bash
uv run mlflow ui
uv run python -m src.train --experiment-name voice-gender-classification
```

## Run API locally

```bash
uv run uvicorn app.main:app --reload
```

## Docker

```bash
docker build -t voice-gender-classifier .
docker run --rm -p 8000:8000 voice-gender-classifier
```

## Predict

```bash
curl -X POST "http://localhost:8000/v1/predict-gender" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.wav"
```
