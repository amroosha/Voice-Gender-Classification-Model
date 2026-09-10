from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
TRAIN_VAL_DIR = DATA_DIR / "train_val"
OOD_TEST_DIR = DATA_DIR / "ood_test"
MODELS_DIR = BASE_DIR / "models"
MODEL_ARTIFACT_PATH = MODELS_DIR / "gender_classifier.joblib"

RANDOM_SEED = 42
SAMPLE_RATE = 16_000
MIN_AUDIO_DURATION_S = 1.0

WAV2VEC2_MODEL_NAME = "facebook/wav2vec2-base"
WAV2VEC2_REVISION = "main"
WAV2VEC2_LAYER = 6

DEVICE = "cpu"
