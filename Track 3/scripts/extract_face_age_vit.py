"""Frozen nateraw/vit-age-classifier expected face age (same recipe as the FLAG age cache)."""
import sys, numpy as np, pandas as pd, torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
CENTERS = np.array([1, 6, 15, 25, 35, 45, 55, 65, 80], float) / 100  # placeholder, replaced below
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Track 1/src"))
from track1.age_auxiliary import expected_face_age
dev = "mps" if torch.backends.mps.is_available() else "cpu"
proc = AutoImageProcessor.from_pretrained("nateraw/vit-age-classifier")
model = AutoModelForImageClassification.from_pretrained("nateraw/vit-age-classifier").to(dev).eval()
paths = pd.read_csv(sys.argv[1]).path.tolist(); out = []
with torch.inference_mode():
    for s in range(0, len(paths), 64):
        imgs = [Image.open(p).convert("RGB") for p in paths[s:s+64]]
        pr = torch.softmax(model(**{k: v.to(dev) for k, v in proc(images=imgs, return_tensors="pt").items()}).logits, 1)
        out.append(expected_face_age(pr.float().cpu().numpy().astype(np.float64)))
        if s % 6400 == 0: print(s, flush=True)
np.save(sys.argv[2], np.concatenate(out).astype(np.float32))
