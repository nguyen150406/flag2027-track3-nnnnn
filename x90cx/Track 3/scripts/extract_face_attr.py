"""Pair-local face attribute EMBEDDINGS (not just scalar predictions) for every FLAG face row (train + dev).

  mivolo: MiVOLO-v2 pre-head features = [cls token (head input), mean of patch tokens (aux_head input)] -> 768-d
          run with .tools/mivolo_env/bin/python
  vit:    nateraw/vit-age-classifier classifier input (768-d CLS) + 9 age-bin logits -> 777-d
Each image is processed independently (frozen models, no cross-file statistics).
Usage: python extract_face_attr.py mivolo|vit [--list faces.csv --out file.npy]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Track 3/caches/face_attr"


def flag_paths():
    meta = pd.read_csv(ROOT / "flag_raw_s01_v3_adaface_simam100/metadata.csv")
    return [str(ROOT / "data/train_set/train_set" / r.face_relpath) if r.partition == "train"
            else str(ROOT / "data/dev_set" / r.protocol / r.face_relpath) for r in meta.itertuples()]


def run_mivolo(paths):
    import cv2
    from transformers import AutoModelForImageClassification, AutoImageProcessor
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForImageClassification.from_pretrained(
        "iitolstykh/mivolo_v2", trust_remote_code=True, torch_dtype=torch.float32).to(dev).eval()
    proc = AutoImageProcessor.from_pretrained("iitolstykh/mivolo_v2", trust_remote_code=True)
    net, cap = model.mivolo.model, {}
    net.head.register_forward_hook(lambda m, i, o: cap.__setitem__("cls", i[0].detach()))
    net.aux_head.register_forward_hook(lambda m, i, o: cap.__setitem__("tok", i[0].detach()))
    out = []
    for s in range(0, len(paths), 32):
        imgs = [cv2.imread(p) for p in paths[s:s + 32]]
        assert all(i is not None for i in imgs)
        x = proc(images=imgs)["pixel_values"].to(dev); body = proc(images=[None] * len(imgs))["pixel_values"].to(dev)
        with torch.no_grad():
            model(faces_input=x, body_input=body)
        tok = cap["tok"]; tok = tok.mean(1) if tok.dim() == 3 else tok
        out.append(torch.cat([cap["cls"].reshape(len(imgs), -1), tok.reshape(len(imgs), -1)], 1).float().cpu().numpy())
        if s % 1024 == 0: print(s, out[-1].shape, flush=True)
    return np.concatenate(out)


def run_vit(paths):
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained("nateraw/vit-age-classifier")
    model = AutoModelForImageClassification.from_pretrained("nateraw/vit-age-classifier").to(dev).eval()
    cap = {}
    model.classifier.register_forward_hook(lambda m, i, o: cap.__setitem__("x", i[0].detach()))
    out = []
    with torch.inference_mode():
        for s in range(0, len(paths), 64):
            imgs = [Image.open(p).convert("RGB") for p in paths[s:s + 64]]
            logits = model(**{k: v.to(dev) for k, v in proc(images=imgs, return_tensors="pt").items()}).logits
            x = cap["x"]; x = x[:, 0] if x.dim() == 3 else x
            out.append(torch.cat([x, logits], 1).float().cpu().numpy())
            if s % 6400 == 0: print(s, out[-1].shape, flush=True)
    return np.concatenate(out)


if __name__ == "__main__":
    kind = sys.argv[1]
    if "--list" in sys.argv:
        paths = pd.read_csv(sys.argv[sys.argv.index("--list") + 1]).path.tolist(); dst = Path(sys.argv[sys.argv.index("--out") + 1])
    else:
        paths = flag_paths(); dst = OUT / f"flag_{kind}.npy"
    feats = run_mivolo(paths) if kind == "mivolo" else run_vit(paths)
    dst.parent.mkdir(parents=True, exist_ok=True); np.save(dst, feats.astype(np.float32)); print("saved", dst, feats.shape)
