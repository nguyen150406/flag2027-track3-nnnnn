"""R3 (user-approved 2026-10-15): AdaFace IR-101 (WebFace12M) face embeddings for the external sources, with EXACTLY the Track 1
RAW-S01-v3 recipe that produced the FLAG 'ada' cache (flag_raw_s01_v3_adaface_simam100): official AdaFace repo pinned at
c60eaa7, its bundled MTCNN aligner (112x112, limit=1; centred-crop fallback when no face), BGR /127.5 - 1, fp32 forward,
L2-normalised feature. Parity gate first: re-embeds FLAG train faces and compares them with the cached FLAG 'ada' rows.
Outputs (<out>): <src>_face_ada.{float32,ok}.npy (rows csv order), <src>_face_ada.align.npy (1 = MTCNN face, 0 = fallback), ada_parity.json
Usage: python t3_extract_adaface.py --adaface_root DIR --weights CKPT --out DIR --roots flag=DIR vox=DIR vox2=DIR vb=DIR lrs3=DIR"""
import argparse, json, os, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from PIL import Image
import t3data as D
from t3lib import META

ap = argparse.ArgumentParser()
ap.add_argument("--adaface_root", required=True); ap.add_argument("--weights", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--roots", nargs="+", default=[]); ap.add_argument("--sources", default="vox,vox2,vb,lrs3"); ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--parity_n", type=int, default=128)
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); T0 = time.time(); dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # AdaFace MTCNN accepts only cuda:0 or cpu (Track 1 ran cuda:0)
ROOTS = {k: [Path(p) for p in v.split(",")] for k, v in (r.split("=", 1) for r in a.roots)}
def log(m): print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 60:6.1f}m] ada:", m, flush=True)
def find(src, rel):  # as t3_ft_clip.find
    for r in ROOTS.get(src, []):
        for p in (r / rel, r / src / rel, r / "train" / rel, r / "test" / rel, r / "dev" / rel):
            if p.exists(): return p
    return None

AR = Path(a.adaface_root); sys.path.insert(0, str(AR))
import net as adaface_net  # noqa: E402
face_model = adaface_net.build_model("ir_101")
ck = torch.load(a.weights, map_location="cpu", weights_only=False); sd = ck.get("state_dict", ck)
ms = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items() if k.startswith("model.") or k in face_model.state_dict()}
missing, unexpected = face_model.load_state_dict(ms, strict=False); assert not missing and not unexpected, (missing[:5], unexpected[:5])
face_model.eval().to(dev)
atp = AR / "face_alignment/mtcnn_pytorch/src/align_trans.py"  # Python-2-era `is` string comparisons (patched as in Track 1)
if atp.is_file():
    s = atp.read_text(); t = s.replace("align_type is 'cv2_affine'", "align_type == 'cv2_affine'").replace("align_type is 'affine'", "align_type == 'affine'")
    if t != s: atp.write_text(t)
cwd = Path.cwd(); os.chdir(AR / "face_alignment")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    from face_alignment import mtcnn as adaface_mtcnn  # noqa: E402
aligner = adaface_mtcnn.MTCNN(device=str(dev), crop_size=(112, 112)); os.chdir(cwd)

def fallback(im):
    im = im.convert("RGB"); w, h = im.size; s = min(w, h); l, t = (w - s) // 2, (h - s) // 2
    return im.crop((l, t, l + s, t + s)).resize((112, 112), Image.Resampling.BILINEAR)
def align(path):
    im = Image.open(path).convert("RGB")
    try:
        _, faces = aligner.align_multi(im, limit=1)
        if faces is not None and len(faces) > 0: return faces[0].convert("RGB"), 1
    except Exception:
        pass
    return fallback(im), 0
def tensor(im):
    rgb = np.asarray(im, dtype=np.float32); assert rgb.shape == (112, 112, 3), rgb.shape
    return torch.from_numpy((rgb[:, :, ::-1].copy() / 127.5 - 1.0).transpose(2, 0, 1))
@torch.inference_mode()
def embed(paths):
    E, S = np.zeros((len(paths), 512), np.float32), np.zeros(len(paths), np.uint8)
    for s in range(0, len(paths), a.bs):
        xs = []
        for i, p in enumerate(paths[s:s + a.bs]):
            im, st = align(p); xs.append(tensor(im)); S[s + i] = st
        f, _ = face_model(torch.stack(xs).to(dev)); E[s:s + len(xs)] = F.normalize(f.float(), dim=1).cpu().numpy()
        if (s // a.bs) % 100 == 0: log(f"  embedded {s + len(xs)}/{len(paths)}")
    return E, S

# parity gate: FLAG train faces vs the cached FLAG 'ada' rows (same recipe -> cos ~1)
tr = META.sort_values("row_id"); tr = tr[tr.partition.eq("train")]
j = np.sort(np.random.default_rng(0).choice(len(tr), a.parity_n, replace=False))
paths = [find("flag", rp) for rp in tr.face_relpath.values[j]]; assert all(p is not None for p in paths), "FLAG train faces not found under flag root"
e, st = embed(paths); c = np.asarray(D.face("ada")[:6485], np.float32)[j]; c = c / np.linalg.norm(c, axis=1, keepdims=True)
cs = (e * c).sum(1); par = dict(mean=float(cs.mean()), min=float(cs.min()), p05=float(np.percentile(cs, 5)), aligned=float(st.mean()))
json.dump(par, open(OUT / "ada_parity.json", "w"), indent=1); log(f"parity vs cached FLAG ada: {par}")
assert cs.mean() > .98, f"AdaFace re-embedding does not reproduce the FLAG cache: {par}"

for s in [x for x in a.sources.split(",") if x]:
    rc = pd.read_csv(D.EXT / f"rows_{s}.csv"); ps = [find(s, f) for f in rc.face]; ok = np.array([p is not None for p in ps])
    up = sorted({str(p) for p in ps if p is not None}); E, S = embed([Path(p) for p in up]); idx = {p: i for i, p in enumerate(up)}
    e = np.zeros((len(ps), 512), np.float32); al = np.zeros(len(ps), np.uint8)
    for r, p in enumerate(ps):
        if p is not None: e[r] = E[idx[str(p)]]; al[r] = S[idx[str(p)]]
    np.save(OUT / f"{s}_face_ada.float32.npy", e); np.save(OUT / f"{s}_face_ada.ok.npy", ok); np.save(OUT / f"{s}_face_ada.align.npy", al)
    log(f"{s}: {ok.sum()}/{len(ok)} rows with an image, MTCNN-aligned {al[ok].mean():.3f}")
log("DONE")
