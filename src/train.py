from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch

from src.config import MODEL_ARTIFACT_PATH, OOD_TEST_DIR, RANDOM_SEED, TRAIN_VAL_DIR, WAV2VEC2_LAYER, NUM_CPU_THREADS
from src.feature_extractor import Wav2Vec2FeatureExtractor

# control the number of threads used by PyTorch to avoid oversubscription and improve performance
torch.set_num_threads(NUM_CPU_THREADS)
LABELS = {"female": 0, "male": 1}


def collect_audio_paths(root_dir: Path) -> tuple[list[Path], list[int]]:
    paths: list[Path] = []
    labels: list[int] = []
    for label_name, label_id in LABELS.items():
        class_dir = root_dir / label_name
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() in {".wav", ".mp3"}:
                paths.append(path)
                labels.append(label_id)
    return paths, labels


def extract_embeddings(
    paths: list[Path], labels: list[int], extractor: Wav2Vec2FeatureExtractor, desc: str = "Extracting"
) -> tuple[np.ndarray, np.ndarray]:
    embeddings = []
    valid_labels = []

    with tqdm(zip(paths, labels), total=len(paths), desc=desc, unit="clip") as pbar:
        for path, label in pbar:
            try:
                emb = extractor.extract_from_bytes(path.read_bytes())
                embeddings.append(emb)
                valid_labels.append(label)
            except Exception as exc:
                pbar.write(f"[{desc}] Skipping {path.name}: {exc}")

    if not embeddings:
        raise RuntimeError(f"No valid audio embeddings extracted for {desc}")
    return np.vstack(embeddings), np.array(valid_labels)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict[str, float]:
    return {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}precision": precision_score(y_true, y_pred),
        f"{prefix}recall": recall_score(y_true, y_pred),
        f"{prefix}f1": f1_score(y_true, y_pred),
    }


def train(experiment_name: str = "voice-gender-classification") -> Path:
    """
    trains using a standard sklearn pipeline with a standard scaler and logistic regression classifier.
    saves the trained model to MODEL_ARTIFACT_PATH and logs the model and metrics to mlflow.
    """

    train_paths, train_labels = collect_audio_paths(TRAIN_VAL_DIR)
    if not train_paths:
        raise RuntimeError(f"No training audio files found in {TRAIN_VAL_DIR}")

    mlflow.set_experiment(experiment_name)
    extractor = Wav2Vec2FeatureExtractor()
    print(f"Found {len(train_paths)} audio files in {TRAIN_VAL_DIR}")
    x_all, y_all = extract_embeddings(train_paths, train_labels, extractor, desc="Train/Val")

    x_train, x_test, y_train, y_test = train_test_split(
        x_all,
        y_all,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y_all,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED, verbose=1)),
        ],
        verbose=True,
    )

    with mlflow.start_run():
        mlflow.log_param("wav2vec2_layer", WAV2VEC2_LAYER)
        mlflow.log_param("classifier", "LogisticRegression")
        mlflow.log_param("class_weight", "balanced")

        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        metrics = evaluate(y_test, y_pred)
        mlflow.log_metrics(metrics)

        cm = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        figure_path = Path("confusion_matrix.png")
        cm.figure_.savefig(figure_path)
        plt.close(cm.figure_)
        mlflow.log_artifact(str(figure_path))
        figure_path.unlink(missing_ok=True)

        ood_paths, ood_labels = collect_audio_paths(OOD_TEST_DIR)
        if ood_paths:
            print(f"Found {len(ood_paths)} OOD test files in {OOD_TEST_DIR}")
            x_ood, y_ood = extract_embeddings(ood_paths, ood_labels, extractor, desc="OOD")
            y_ood_pred = pipeline.predict(x_ood)
            mlflow.log_metrics(evaluate(y_ood, y_ood_pred, prefix="ood_"))

        MODEL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_ARTIFACT_PATH)
        mlflow.log_artifact(str(MODEL_ARTIFACT_PATH))

        metadata_path = MODEL_ARTIFACT_PATH.with_suffix(".json")
        metadata_path.write_text(json.dumps({"classes": list(LABELS.keys())}, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(metadata_path))

    return MODEL_ARTIFACT_PATH


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default="voice-gender-classification")
    args = parser.parse_args()
    artifact_path = train(args.experiment_name)
    print(f"Saved artifact to: {artifact_path}")


if __name__ == "__main__":
    main()
