"""Build the Kaggle notebook that extracts Track 3 frozen embeddings (FLAG voices + MAV v1 + VoxCeleb1)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
cells = []
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s})
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})

md("# FLAG Track 3 — frozen embedding extraction\nFrozen pretrained encoders; every file processed independently (pair-local). "
   "FLAG: voices only (r293, sam100ft, clean unbatched wavlm4). MAV-Celeb v1 English+Urdu **train** split and VoxCeleb1: "
   "faces (farl, clip) and voices (simam, ecapa, campplus, wavlm4, r293, sam100ft). No labels or test statistics are used.")
code(r'''
import os, sys, subprocess, time, shutil, zipfile, json
from pathlib import Path
import numpy as np, pandas as pd
def run(c): print("+", c); subprocess.run(c, shell=True, check=True)
run("pip install -q ftfy regex pyyaml huggingface_hub 2>&1 | tail -1")
W = Path("/kaggle/temp/t3"); W.mkdir(parents=True, exist_ok=True)
OUT = Path("/kaggle/working/t3_out"); OUT.mkdir(parents=True, exist_ok=True)
def find(marker, depth=6):
    base = Path("/kaggle/input"); hits = []
    for root, dirs, files in os.walk(base):
        d = Path(root)
        if len(d.relative_to(base).parts) >= depth or d.name.startswith("id"):
            dirs[:] = []
        if marker in files or marker in dirs:
            hits.append(d / marker)
    return hits
TOOLS_SRC = find("t3extract.py")[0].parent
print("tools", TOOLS_SRC)
T = W / "tools"; (T / "farl").mkdir(parents=True, exist_ok=True); (T / "simam/unpacked/voxblink2_samresnet100").mkdir(parents=True, exist_ok=True)
for a, b in ((TOOLS_SRC / "farl/FaRL-Base-Patch16-LAIONFace20M-ep16.pth", T / "farl/FaRL-Base-Patch16-LAIONFace20M-ep16.pth"),
             (TOOLS_SRC / "simam/avg_model.pt", T / "simam/unpacked/voxblink2_samresnet100/avg_model.pt")):
    if not b.exists(): os.symlink(a, b)
if not (T / "farl/CLIP/clip").exists():
    run(f"git clone -q https://github.com/openai/CLIP.git {T}/farl/CLIP && git -C {T}/farl/CLIP checkout -q d50d76daa670286dd6cacf3bcd80b5e4823fc8e1")
(T / "farl/vendor").mkdir(exist_ok=True)
if not (T / "wespeaker/wespeaker").exists():
    run(f"git clone -q https://github.com/wenet-e2e/wespeaker.git {T}/wespeaker && git -C {T}/wespeaker checkout -q 9fecd6cb4f47475d01761d87c826298dff4ef18c")
os.environ["T3_TOOLS"] = str(T); os.environ["T3_DEVICE"] = "cuda"
sys.path.insert(0, str(TOOLS_SRC))
import t3extract as X
print("device", X.DEV)
''')
code(r'''
FLAG = find("data")  # dataset root containing data/dev_set
FLAG = [p.parent for p in FLAG if (p / "dev_set").is_dir()][0]
V1 = find("meta_v1.csv")[0].parent / "v1"
VOXF = [p for p in find("face") if p.parent.name == "VoxCeleb_1_adapted"]
VOXA = [p for p in find("wav") if p.is_dir() and "voxceleb1" in str(p).lower()]
print(FLAG, V1, VOXF, VOXA)
flag = pd.read_csv(TOOLS_SRC / "rows_flag.csv"); v1 = pd.read_csv(TOOLS_SRC / "rows_v1.csv"); vox = pd.read_csv(TOOLS_SRC / "rows_vox.csv")
def resolve(rel, roots, subs=("",)):
    for r in roots:
        for s in subs:
            p = r / s / rel
            if p.exists(): return p
    return None
flag_voice = [FLAG / p for p in flag.voice]
v1_voice = [V1 / "voices" / p for p in v1.voice]; v1_face = [V1 / "faces" / p for p in v1.face]
vox_face = [resolve(p, VOXF, ("train", "test", "dev", "")) for p in vox.face]
vox_voice = [resolve(p, VOXA) for p in vox.voice]
for name, lst in (("flag_voice", flag_voice), ("v1_voice", v1_voice), ("v1_face", v1_face), ("vox_face", vox_face), ("vox_voice", vox_voice)):
    miss = sum(p is None or not Path(p).exists() for p in lst); print(name, len(lst), "missing", miss)
''')
code(r'''
def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)
def save(tag, arr, ok=None):
    np.save(OUT / f"{tag}.float32.npy", arr.astype(np.float32))
    if ok is not None: np.save(OUT / f"{tag}.ok.npy", ok)
    log(f"saved {tag} {arr.shape}")
def faces(tag, paths, enc):
    if (OUT / f"{tag}.float32.npy").exists(): return
    ok = np.array([p is not None for p in paths]); fn = X.face_encoder(enc)
    arr = np.zeros((len(paths), 512 if enc == "farl" else 768), np.float32)
    arr[ok] = fn([str(p) for p, k in zip(paths, ok) if k], log=log); save(tag, arr, ok)
def voices(tag, paths, enc):
    if (OUT / f"{tag}.float32.npy").exists(): return
    ok = np.array([p is not None for p in paths]); fn = X.voice_encoder(enc)
    e = fn([str(p) for p, k in zip(paths, ok) if k], log=log)
    arr = np.zeros((len(paths), e.shape[1]), np.float32); arr[ok] = e; save(tag, arr, ok)
t0 = time.time()
for enc in ("farl", "clip"):
    faces(f"v1_face_{enc}", v1_face, enc); faces(f"vox_face_{enc}", vox_face, enc)
import torch; torch.cuda.empty_cache()
for enc in ("simam", "ecapa", "campplus", "wavlm4", "sam100ft", "r293"):
    voices(f"v1_voice_{enc}", v1_voice, enc); voices(f"vox_voice_{enc}", vox_voice, enc)
    if enc in ("wavlm4", "sam100ft", "r293"):
        voices(f"flag_voice_{enc}", flag_voice, enc)
    torch.cuda.empty_cache()
log(f"total {(time.time()-t0)/60:.1f} min")
''')
code(r'''
for f in ("rows_flag.csv", "rows_v1.csv", "rows_vox.csv"): shutil.copy(TOOLS_SRC / f, OUT / f)
json.dump(dict(pair_local=True, frozen_encoders=True, labels_used=False, test_statistics=False), open(OUT / "PROVENANCE.json", "w"))
with zipfile.ZipFile("/kaggle/working/t3_embeddings.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()): z.write(p, p.name)
shutil.rmtree(OUT); print("done")
''')
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
d = HERE / "t3_extract"; d.mkdir(exist_ok=True)
json.dump(nb, open(d / "t3_extract.ipynb", "w"), indent=1)
json.dump({"id": "trncnguyn/flag-t3-embedding-extract", "title": "FLAG T3 Embedding Extract", "code_file": "t3_extract.ipynb",
           "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
           "dataset_sources": ["trncnguyn/flag-t3-tools", "trncnguyn/flag-2027-data", "KAGGLE_USER/external-data-mavceleb",
                               "robot12138/voxceleb-1", "kryakrya/voxceleb1train", "kryakrya/voxceleb1test"],
           "competition_sources": [], "kernel_sources": []}, open(d / "kernel-metadata.json", "w"), indent=1)
print("built")
