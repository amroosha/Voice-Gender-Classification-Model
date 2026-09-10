"""
FLEURS ships a `gender` class label per utterance (0=male, 1=female per the
`google/fleurs` dataset card), so we bucket each clip's audio directly into
`data/ood_test/male/` or `data/ood_test/female/` as .wav files, matching the
directory layout `src/config.py` (`OOD_TEST_DIR`) and `src/train.py`
(`collect_audio_paths`) already expect.

Usage:
    uv run python scripts/download_ood_test_set.py
"""


from datasets import Audio, load_dataset
from config import OOD_TEST_DIR

GENDER_INT_TO_LABEL = {0: "male", 1: "female"}


def download_ood_test_set() -> None:

    # streaming=True ensures ONLY the test split is fetched (no train/val downloaded).
    # decode=False leaves raw audio bytes intact so no soundfile or decoding is used so we can use torchaudio in our unified pipeline.

    test_ds = load_dataset(
        "google/fleurs", "en_us", split="test", streaming=True
    ).cast_column("audio", Audio(decode=False))

    for label in GENDER_INT_TO_LABEL.values():
        (OOD_TEST_DIR / label).mkdir(parents=True, exist_ok=True)

    counts = {"male": 0, "female": 0}
    for example in test_ds:
        label = GENDER_INT_TO_LABEL.get(example["gender"])
        if label is None:
            continue

        out_path = OOD_TEST_DIR / label / f"{example['id']}.wav"
        out_path.write_bytes(example["audio"]["bytes"])
        counts[label] += 1

    print(f"Done! Saved {counts['male']} male and {counts['female']} female clips to {OOD_TEST_DIR}")


if __name__ == "__main__":
    download_ood_test_set()

