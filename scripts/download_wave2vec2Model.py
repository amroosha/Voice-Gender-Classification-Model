from transformers import Wav2Vec2Model
from pathlib import Path

save_directory = Path("models/wav2vec2-base")
save_directory.mkdir(parents=True, exist_ok=True)
print("Starting Download")

model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model.save_pretrained(save_directory)

print(f"Saved locally to: {save_directory.resolve()}")