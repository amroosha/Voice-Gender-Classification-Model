FROM python:3.11-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends curl ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src ./src
COPY app ./app

ARG WAV2VEC2_MODEL_NAME=facebook/wav2vec2-base
ARG WAV2VEC2_REVISION=main
RUN uv run python -c "from transformers import Wav2Vec2Model, Wav2Vec2Processor; Wav2Vec2Model.from_pretrained('${WAV2VEC2_MODEL_NAME}', revision='${WAV2VEC2_REVISION}'); Wav2Vec2Processor.from_pretrained('${WAV2VEC2_MODEL_NAME}', revision='${WAV2VEC2_REVISION}')"

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface
COPY src ./src
COPY app ./app
COPY models ./models

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
