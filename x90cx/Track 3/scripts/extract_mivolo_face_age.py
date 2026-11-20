"""Pair-local MiVOLO-v2 face age/gender for every FLAG face row (train + dev).

Frozen pretrained face model (iitolstykh/mivolo_v2, Apache-2.0). Each image is
processed independently; no statistics across files. Run with .tools/mivolo_env.
"""
import sys
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch
from transformers import AutoModelForImageClassification, AutoImageProcessor

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Track 3/caches/mivolo_face"
meta = pd.read_csv(ROOT / "flag_raw_s01_v3_adaface_simam100/metadata.csv")


def face_path(r):
    if r.partition == "train":
        return ROOT / "data/train_set/train_set" / r.face_relpath
    return ROOT / "data/dev_set" / r.protocol / r.face_relpath


def main(flip=False, limit=None, paths=None, out=OUT):  # paths: explicit image list (eval phase, --list csv with 'path')
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForImageClassification.from_pretrained(
        "iitolstykh/mivolo_v2", trust_remote_code=True, torch_dtype=torch.float32).to(dev).eval()
    proc = AutoImageProcessor.from_pretrained("iitolstykh/mivolo_v2", trust_remote_code=True)
    rows = ([str(face_path(r)) for r in meta.itertuples()] if paths is None else list(paths))[:limit]
    ages, pf = np.zeros(len(rows), np.float32), np.zeros(len(rows), np.float32)
    for start in range(0, len(rows), 32):
        batch = rows[start:start+32]
        imgs = [cv2.imread(p) for p in batch]  # BGR as model expects
        if any(i is None for i in imgs):
            raise FileNotFoundError("unreadable face image")
        if flip:
            imgs = [i[:, ::-1].copy() for i in imgs]
        x = proc(images=imgs)["pixel_values"].to(dev)
        body = proc(images=[None]*len(imgs))["pixel_values"].to(dev)
        with torch.no_grad():
            o = model(faces_input=x, body_input=body)
        ages[start:start+len(batch)] = o.age_output.float().cpu().numpy().reshape(-1)
        male = o.gender_class_idx.cpu().numpy().reshape(-1) == 0
        p = o.gender_probs.float().cpu().numpy().reshape(-1)
        pf[start:start+len(batch)] = np.where(male, p, 1-p)  # P(male)
        if start % 1024 == 0:
            print(start, flush=True)
    if limit and paths is None:
        print(np.round(ages[:limit], 1), np.round(pf[:limit], 2), meta.identity[:limit].tolist())
        return
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    tag = "_flip" if flip else ""
    np.save(out / f"face_age{tag}.npy", ages); np.save(out / f"face_pmale{tag}.npy", pf)
    if paths is None:
        meta[["row_id", "partition", "protocol", "language", "face_relpath"]].to_csv(out / "metadata.csv", index=False)


if __name__ == "__main__":
    arg = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
    main(flip="--flip" in sys.argv, limit=int(arg("--limit")) if arg("--limit") else None,
         paths=pd.read_csv(arg("--list")).path.tolist() if arg("--list") else None, out=arg("--out") or OUT)
