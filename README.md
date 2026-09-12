# Voice Gender Classification Microservice

A production-grade machine learning microservice that classifies gender from raw speech audio. The architecture transforms continuous raw waveforms into self-supervised transformer representations using a frozen Wav2Vec 2.0 backbone, followed by a calibrated, efficient classification pipeline served via an asynchronous FastAPI backend.


---

## 1. Executive Summary & Benchmark Results

Traditional voice classification pipelines rely on hand-crafted acoustic metrics (pitch, fundamental frequency, spectral centroids, jitter, shimmer) or spectrogram computer vision baselines. These hand-engineered approaches degrade rapidly in real-world environments due to microphone variances, background noise, and acoustic compression.

This system replaces manual feature extraction with representation learning:
- **Feature Backbone**: Frozen `facebook/wav2vec2-base` extracting temporal representations from an early-middle transformer layer (Layer 6).
- **Classification Head**: Scikit-Learn `Pipeline` bundling `StandardScaler` with cost-sensitive `LogisticRegression`.
- **Validation Standard**: Evaluated against two physically isolated datasets, including an untouched Out-Of-Distribution (OOD) benchmark.

### Benchmark Performance

| Evaluation Split | Dataset | Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Validation (In-Distribution)** | Kaggle Voice Gender (80/20 Stratified) | **100.0%** | **1.000** | **1.000** | **1.000** |
| **OOD Generalization Test** | Google FLEURS (`en_us` Test Split) | **99.17%** | **0.9895** | **0.9895** | **0.9895** |

- **Inference Latency**: Sub-50ms per audio utterance on standard x86 CPU.
- **Serving Memory Footprint**: Under 1 GB RAM at steady-state.

---

## 2. System Architecture & End-to-End Pipeline

The end-to-end processing lifecycle flows through four decoupled stages:

```
[ Ingested Audio File ]
  (WAV, MP3, 16/24/32-bit PCM, Float)
         │
         ▼
[ Ingestion & Sanitization Firewall ] (src/audio_utils.py)
  • In-memory decoding (BytesIO) via torchaudio / soundfile / wave
  • Multi-channel downmixing to mono
  • Sinc-interpolation resampling to 16,000 Hz
  • Minimum duration validation (>= 0.5s)
         │
         ▼
[ Foundation Feature Backbone ] (src/feature_extractor.py)
  • facebook/wav2vec2-base (Frozen weights, eval mode)
  • Hidden states from Layer 6 of 12
  • Temporal mean pooling across all acoustic frames
  • Output shape: (768,) float vector
         │
         ▼
[ Downstream Classifier Pipeline ] (src/train.py)
  • StandardScaler (Z-score feature standardization)
  • LogisticRegression with cost-sensitive class balancing
         │
         ▼
[ Inference & API Serving ] (app/main.py)
  • FastAPI asynchronous handler
  • Pydantic contract validation (app/schemas.py)
  • Sub-50ms JSON response: predicted_label, confidence_score, latency_ms
```

---

## 3. Engineering Decisions & Architectural Rationale

### 3.1 Feature Backbone: Frozen Wav2Vec 2.0 (Layer 6)
- **Why Layer 6 instead of the final layer?** 
  Layer-wise probing research on speech foundation models shows that acoustic properties, vocal tract length, and fundamental pitch concentrate in early-to-middle layers (Layers 4 through 7). Late layers (Layers 10 through 12) specialize in phonetic recognition tuned for speech-to-text objectives, intentionally stripping speaker identity. Layer 6 preserves the maximal speaker-level acoustic signature.
- **Why frozen?**
  Fine-tuning a 95-million-parameter transformer on a small-to-medium dataset introduces high overfitting risk and requires GPU infrastructure. Keeping the backbone completely frozen (`parameter.requires_grad = False`) allows training to complete in minutes and serving to run efficiently on low-cost commodity CPUs.
- **Temporal Mean Pooling**:
  Regardless of whether an uploaded recording is 1 second or 8 seconds, temporal mean pooling over the frame dimension produces a standardized 768-dimensional feature vector.

### 3.2 Classifier: Cost-Sensitive Logistic Regression
- **Why Logistic Regression over LightGBM or Deep Neural Networks?**
  Wav2Vec2 embeddings for pitch and gender are close to linearly separable. A linear model finds the optimal separating hyperplane without the overfitting risks associated with gradient-boosted decision trees. Furthermore, Logistic Regression coefficients provide direct interpretability, and its sigmoid outputs produce well-calibrated probabilities without requiring post-hoc Platt scaling or isotonic regression.
- **Handling Imbalance via Loss Weighting**:
  The primary training set contains approximately 10,000 male samples and 4,700 female samples (a 2.13 to 1 imbalance). Rather than performing destructive downsampling or synthetic oversampling, the classifier uses cost-sensitive loss weighting:
  
  $$\text{weight}_c = \frac{N_{\text{total}}}{N_{\text{classes}} \times N_c}$$

  This assigns a loss penalty of ~1.56 to female misclassifications versus ~0.73 to male misclassifications, forcing the decision boundary to remain acoustically centered.

### 3.3 Audio Preprocessing & Multi-Tier Decoder Resilience
Real-world audio uploads exhibit inconsistent codecs, bit depths, and header structures. The ingestion module (`src/audio_utils.py`) enforces strict validation and multi-tier decoding resilience:
- **In-Memory Decoding**: All byte streams are processed via `io.BytesIO`, avoiding disk I/O bottlenecks during live inference.
- **Multi-Tier Decoding Fallback**:
  1. 1st: `torchaudio.load(io.BytesIO(audio_bytes))` for native stream parsing.
  2. 2nd: `torchaudio` with explicit `format="wav"` header hint.
  3. 3rd: `soundfile` (`libsndfile` backend) to seamlessly decode 24-bit PCM, 32-bit floating point, and extensible WAV headers generated by Audacity and mobile recorders.
  4. 4th: Python standard library `wave` binary chunk parser.
- **Why Heuristic Noise Reduction Was Rejected**:
  Legacy audio projects often apply naive spectral subtraction on the first 0.2 seconds assuming initial silence. In production, if a speaker starts immediately at 0.0 seconds, naive subtraction destroys the initial consonants and introduces phase distortion (musical noise). Wav2Vec2 was pre-trained on thousands of hours of noisy, multi-environment speech, making it inherently robust to stationary noise without destructive pre-filtering.

---

## 4. Experiment Tracking with MLflow

MLflow tracks all training parameters, metrics, and generated artifacts.

### Tracked Metadata
- **Parameters**: `wav2vec2_layer` (6), `classifier` (LogisticRegression), `class_weight` (balanced), random seeds.
- **Metrics**: Accuracy, Precision, Recall, and F1 Score for both in-distribution validation and out-of-distribution evaluation.
- **Artifacts**: Serialized model pipeline (`.joblib`), confusion matrix visualizations (`confusion_matrix.png`), and label schemas (`.json`).

---

## 5. Production API Design (FastAPI & Pydantic)

The inference API is built on FastAPI and Uvicorn:

- **Lifespan Context Management**:
  Backbone weights and the classification pipeline load once during application startup into memory (`app.state`).
- **Pydantic Data Contracts**:
  Endpoints strictly validate inputs and outputs against schemas defined in `app/schemas.py`. Output probabilities are bounded to `[0.0, 1.0]` 
- **Operational Health Checks**:
  Exposes `GET /health` returning `{"status": "ok"}` 
- **Error Mapping**:
  Domain errors (empty payload, audio under 0.5s, unreadable bytes) map to `HTTP 422 Unprocessable Entity` 
---

## 6. Containerization & Reproducibility (Docker)

The application packages into an isolated, self-contained Docker container using a multi-stage build:

1. **Stage 1 (Builder)**:
   - Base: `python:3.11-slim`.
   - Installs build tools, `curl`, and runtime audio libraries (`ffmpeg`, `libsndfile1`).
   - Uses `uv` to resolve and install frozen dependencies from `uv.lock`.
   - Pre-downloads and verifies model backbone weights.
2. **Stage 2 (Runtime)**:
   - Fresh `python:3.11-slim` base containing only system libraries (`ffmpeg`, `libsndfile1`).
   - Copies the pre-built virtual environment (`.venv`), application source code, and local model weights.
   - Excludes compilers, package managers, and development dependencies (`pytest`, `mlflow`).
   - **Zero-Network Cold Starts**: Model weights reside in the container image, guaranteeing that the service boots offline in under 2 seconds without external calls to Hugging Face.

---

## 7. Project Directory Structure

```
Voice-Gender-Classification-Model/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application, lifespan loader, /v1/predict-gender
│   └── schemas.py               # Pydantic data contracts (GenderPredictionResponse, HealthResponse)
├── data/
│   ├── train_val/               # Primary training dataset (male/, female/)
│   └── ood_test/                # Out-of-distribution benchmark dataset (male/, female/)
├── models/
│   ├── wav2vec2-base/           # Local offline Wav2Vec2 backbone weights & configs
│   ├── gender_classifier.joblib # Serialized StandardScaler + LogisticRegression pipeline
│   ├── gender_classifier.json   # Target class label mappings
│   └── README.md                # Models directory documentation
├── scripts/
│   ├── download_ood_test_set.py # Streaming script for Google FLEURS test split
│   └── check_mlflow.py          # Diagnostic verification utility for tracking stores
├── src/
│   ├── __init__.py
│   ├── audio_utils.py           # In-memory audio decoding, resampling, validation firewall
│   ├── config.py                # System constants, sample rates, dynamic path resolution
│   ├── feature_extractor.py     # Frozen Wav2Vec2 Layer 6 embedding extraction
│   └── train.py                 # Training orchestrator, evaluation, and MLflow logging
├── tests/
│   ├── conftest.py              # Test fixtures and path configurations
│   ├── test_api.py              # Integration tests against FastAPI endpoints
│   ├── test_audio_utils.py      # Unit tests for audio resampling, mono, and duration
│   └── test_feature_extractor.py# Unit tests verifying embedding shape (768,) and determinism
├── Dockerfile                   # Multi-stage production container definition
├── pyproject.toml               # Project metadata and dependency constraints
├── uv.lock                      # Cryptographically pinned dependency lockfile
└── README.md                    # Project documentation
```

---

## 8. Quickstart & Local Setup

### 8.1 Prerequisites
- Python 3.11+
- `uv` package manager

Install `uv` (Windows PowerShell):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Install `uv` (macOS / Linux):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 8.2 Installation
Clone the repository and synchronize dependencies:
```bash
git clone https://github.com/your-username/Voice-Gender-Classification-Model.git
cd Voice-Gender-Classification-Model
uv sync
```

### 8.3 Run Automated Tests
Execute unit and integration tests:
```bash
uv run pytest tests/ -v
```

### 8.4 Download Out-of-Distribution Data (Optional)
Stream the Google FLEURS English test split directly into `data/ood_test/` without downloading full training archives:
```bash
uv run python -m scripts.download_ood_test_set
```

### 8.5 Train the Model & Track Experiments
Run the training pipeline:
```bash
uv run python -m src.train
```

Launch the MLflow tracking dashboard:
```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open `http://localhost:5000` to inspect hyperparameters, validation metrics, and confusion matrix artifacts.

### 8.6 Launch the Production API Server
Start Uvicorn with hot-reload enabled:
```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
- Interactive Swagger UI: `http://localhost:8000/docs`
- Health Probe: `http://localhost:8000/health`

### 8.7 Test Inference via cURL
Send an audio file for live gender prediction:
```bash
curl.exe -X POST "http://localhost:8000/v1/predict-gender" \
  -H "accept: application/json" \
  -F "file=@data/ood_test/male/sample.wav"
```

Sample JSON response:
```json
{
  "predicted_label": "male",
  "confidence_score": 0.9942,
  "latency_ms": 45.3
}
```

---

## 9. Container Deployment (Docker)

### 9.1 Build Container Image
```bash
docker build -t voice-gender-service:latest .
```

### 9.2 Run Container
```bash
docker run -d --name voice-gender-api -p 8000:8000 voice-gender-service:latest
```

The container starts up self-contained with offline model weights, serving live predictions at `http://localhost:8000/docs`.
