"""Eval-phase feature extraction for t3_eval.py (user 2026-10-29): every sub60 input recomputed for the pairs of a rows.csv
(t3_eval.py table) with the script/recipe that produced the FLAG caches; files are encoded one by one (images in ViT batches),
no flip, no statistics across files. Each unique face/voice file is encoded once and mapped back to the rows.
  python t3_eval_extract.py run <rows.csv> <feat_dir> [group ...] [--device mps|cuda|cpu] [--limit n]
  python t3_eval_extract.py check <feat_dir> <ref_feat_dir>     cosine / abs diff vs reference features (same rows)
groups (default all sub60 groups): see GROUPS --limit n: n evenly spaced rows only (smoke / parity check; indices kept in <feat_dir>/_rows.npy).
Python: $T3_PY (default this interpreter; audeering needs transformers 4.x), MiVOLO runs in .tools/mivolo_env."""
import os, sys, subprocess
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; SUBS = ROOT / "Track 3/submissions"
PY = os.environ.get("T3_PY", sys.executable); MIVOLO_PY = str(ROOT / ".tools/mivolo_env/bin/python")
LORA_CKPT = {"clipft": SUBS / "t3_sub22/clipft.pt", "clipsa": SUBS / "t3_sub26/clipsa.pt",  # NOT t3_sub23/clipft.pt (another LoRA)
             "clipft12": SUBS / "t3_sub57/clipft12.pt", "clipsa12": SUBS / "t3_sub58/clipsa12.pt"}
import os, json as _json
if os.environ.get("T3_SYSSET"): LORA_CKPT = {k: SUBS / v for k, v in _json.load(open(os.environ["T3_SYSSET"]))["lora_ckpt"].items()}  # clean retrain
LORA_CKPT.update({n: ROOT / "Track 3/caches/ultra_lora" / f"{n}.pt" for n in ("clipnf", "clipnf2")})  # ULTRA 2026-11-11 no-FLAG LoRAs (Kaggle-made -> --turbo)
AUD = {"a24": "audeering/wav2vec2-large-robust-24-ft-age-gender", "a6": "audeering/wav2vec2-large-robust-6-ft-age-gender"}
# JPEG decoder of each face cache: Kaggle-made (FaRL notebook, LoRA kernels) = libjpeg-turbo; Mac-made (CLIP01, ViT age) = anaconda
# Pillow (IJG libjpeg 9); MiVOLO decodes with OpenCV (turbo) in both. Decoded pixels differ by up to ~13 grey levels, and CLIP-family
# embeddings follow (LoRA cos .985 with the wrong decoder; dev EER +0.41 in the 2026-10-30 rehearsal).
TURBO = {"farl": "1", "clip": "0"}
GROUPS = ["farl", "clip", *LORA_CKPT, "sim", "ecapa", "sam100ft", "r293", "rdn", "wsv", "vitage", "mivolo", "a24", "a6"]


def run(rows_csv, fd, groups, device, limit):
    fd = Path(fd); W = fd / "_work"; W.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(rows_csv)
    if limit:
        idx = np.unique(np.linspace(0, len(t) - 1, limit).astype(int)); np.save(fd / "_rows.npy", idx); t = t.iloc[idx]
    uf, fi = np.unique(t.face.astype(str), return_inverse=True); uv, vi = np.unique(t.voice.astype(str), return_inverse=True)
    pd.DataFrame(dict(path=uf)).to_csv(W / "faces.csv", index=False); pd.DataFrame(dict(path=uv, voice=uv)).to_csv(W / "voices.csv", index=False)
    print(f"{len(t)} rows: {len(uf)} faces, {len(uv)} voices -> {fd}", flush=True)
    env = dict(os.environ, T3_DEVICE=device)
    def sh(*cmd, py=PY):
        print("$", " ".join(map(str, cmd))[:220], flush=True); subprocess.run([py, *map(str, cmd)], cwd=HERE, env=env, check=True)
    def put(name, arr, inv):
        arr = np.asarray(arr); assert len(arr) == inv.max() + 1, (name, arr.shape); np.save(fd / f"{name}.npy", arr[inv]); print("  ->", name, arr[inv].shape, flush=True)
    for g in groups:
        o = W / f"{g}.npy"
        if g in ("farl", "clip"):
            sh(__file__, "_t3x", "face", g, W / "faces.csv", o, TURBO[g]); put(f"face_{g}", np.load(o), fi)
        elif g in LORA_CKPT:
            sh("t3_clipft_encode.py", LORA_CKPT[g], "list", W / "faces.csv", "--out", o, "--device", device, "--turbo"); put(f"face_{g}", np.load(o), fi)
        elif g in ("sim", "sam100ft", "r293"):
            sh(__file__, "_t3x", "voice", "simam" if g == "sim" else g, W / "voices.csv", o); put(f"voice_{g}", np.load(o), vi)
        elif g == "ecapa":
            sh(__file__, "_ecapa", W / "voices.csv", W / "ecapa", device); put("voice_ecapa", np.load(W / "ecapa/voice_embeddings.float32.npy"), vi)
        elif g in ("rdn", "wsv"):
            sh("t3_extract_wavlmsv.py", "--name", g, "--list", W / "voices.csv", "--out", W, "--flag_data", ROOT / "data", "--sources", "", "--device", device)
            put(f"voice_{g}", np.load(W / f"list_voice_{g}.float32.npy"), vi)
        elif g == "vitage":
            sh("extract_face_age_vit.py", W / "faces.csv", o); put("vit_age", np.load(o), fi)
        elif g == "mivolo":
            sh("extract_mivolo_face_age.py", "--list", W / "faces.csv", "--out", W / "mivolo", py=MIVOLO_PY)
            put("mivolo_age", np.load(W / "mivolo/face_age.npy"), fi); put("mivolo_pmale", np.load(W / "mivolo/face_pmale.npy"), fi)
        elif g in AUD:
            sh("extract_voice_age.py", AUD[g], W / "voices.csv", W / f"{g}.npz"); z = np.load(W / f"{g}.npz")
            if g == "a24": put("a24_age", z["age"], vi); put("a24_gender", z["gender"], vi)
            put(f"{g}_emb", z["emb"], vi)
        else:
            raise ValueError(g)


def check(fd, ref):
    fd, ref = Path(fd), Path(ref); idx = np.load(fd / "_rows.npy") if (fd / "_rows.npy").exists() else None
    for p in sorted(fd.glob("*.npy")):
        if p.name.startswith("_") or not (ref / p.name).exists(): continue
        x = np.load(p).astype(np.float64); y = np.load(ref / p.name).astype(np.float64); y = y[idx] if idx is not None else y
        if x.ndim == 2 and x.shape[1] > 3:
            c = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-12)
            print(f"{p.stem:15s} cos mean {c.mean():.5f} min {c.min():.5f}  (rows {len(c)})")
        else:
            d = np.abs(x - y); print(f"{p.stem:15s} |diff| mean {d.mean():.2e} max {d.max():.2e}  (rows {len(d)})")


def _t3x(kind, enc, lst, out, turbo="0"):  # worker: t3extract encoders (frozen faces farl/clip, WeSpeaker-style voices)
    sys.path.insert(0, str(HERE)); import t3extract as X
    ps = pd.read_csv(lst).path.tolist(); f = X.face_encoder(enc, turbo == "1") if kind == "face" else X.voice_encoder(enc)
    np.save(out, f(ps, log=print).astype(np.float32))


def _ecapa(lst, out, device):  # worker: extract_wespeaker.py recipe (fixed 30 s chunks, unnormalised mean) used for the ecapa cache
    sys.path.insert(0, str(HERE)); import extract_wespeaker as E
    E.main("Wespeaker/wespeaker-voxceleb-ecapa-tdnn1024-LM", "eval", device, rows=pd.read_csv(lst), out_dir=out)


if __name__ == "__main__":
    a = sys.argv[1:]; opt = lambda k, d: a[a.index(k) + 1] if k in a else d
    pos = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or not a[i - 1].startswith("--"))]
    if pos[0] == "run": run(pos[1], pos[2], pos[3:] or GROUPS, opt("--device", "mps"), int(opt("--limit", 0)))
    elif pos[0] == "check": check(pos[1], pos[2])
    elif pos[0] == "_t3x": _t3x(*pos[1:6])
    elif pos[0] == "_ecapa": _ecapa(*pos[1:4])
    else: raise SystemExit(__doc__)
