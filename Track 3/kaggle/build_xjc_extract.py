"""Kaggle notebook (trncnguyn): our frozen encoders on XJ's MAV-Celeb v1/v2 clips (14/11, user "Tiếp tục làm và phát triển cho eval tốt lên").
Faces: clip (OpenAI CLIP ViT-L/14, Pillow decode = as the eval/v1 caches). Voices: simam, sam100ft, r293, ecapa (= cat4). Same t3extract.py and
pinned CLIP / WeSpeaker commits as t3_extract (which made v1_*/vox_* caches). Every file processed independently; no labels, no test data.
Rows = xjc_rows.csv (kind, src, id, lang, video, rel, xj_row): the 8,779 voice clips and 9,520 face images of the teammate's external MAV set."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
cells = []
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s})
code(r'''
import os, sys, subprocess, time, shutil, zipfile, json, glob, traceback
from pathlib import Path
import numpy as np, pandas as pd
T0 = time.time()
def log(m): print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 60:6.1f}m]", m, flush=True)
def run(c): print("+", c, flush=True); subprocess.run(c, shell=True, check=True)
def find_dirs(name, depth=4):   # depth-limited: never walk the media trees
    return [p for d in range(1, depth + 1) for p in glob.glob("/kaggle/input/" + "/".join(["*"] * d)) if os.path.basename(p) == name]
run("pip install -q ftfy regex pyyaml huggingface_hub 2>&1 | tail -1")
W = Path("/kaggle/temp/xjc"); W.mkdir(parents=True, exist_ok=True); OUT = Path("/kaggle/working/xjc_out"); OUT.mkdir(parents=True, exist_ok=True)
TOOLS_SRC = Path([p for p in find_dirs("t3extract.py")][0]).parent; ROWS = Path(find_dirs("xjc_rows.csv")[0])
log(f"tools {TOOLS_SRC} rows {ROWS}")
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
log(f"device {X.DEV}")
''')
code(r'''
try:
    R = pd.read_csv(ROWS); roots = {}
    for top in ("mavceleb_v1_complete-001", "mavceleb_v2_complete-002"):
        h = [p for p in find_dirs(top) if os.path.isdir(p)]; assert h, f"{top} not mounted"; roots[top] = Path(h[0]).parent
    R["path"] = [str(roots[r.split("/")[0]] / r) for r in R.rel]; miss = [p for p in R.path if not os.path.exists(p)]
    log(f"rows {len(R)} {R.groupby(['kind', 'src']).size().to_dict()} missing {len(miss)} roots {roots}"); assert not miss, miss[:5]
    def save(tag, arr, ok):
        np.save(OUT / f"{tag}.float32.npy", arr.astype(np.float32)); np.save(OUT / f"{tag}.ok.npy", ok); log(f"saved {tag} {arr.shape} ok {int(ok.sum())}")
    fp = R[R.kind.eq("face")].path.tolist(); vp = R[R.kind.eq("voice")].path.tolist()
    fn = X.face_encoder("clip"); e = np.asarray(fn(fp, log=log), np.float32); save("xjc_face_clip", e, np.isfinite(e).all(1) & (np.abs(e).sum(1) > 0))
    import torch; torch.cuda.empty_cache()
    for enc in ("simam", "ecapa", "sam100ft", "r293"):
        fn = X.voice_encoder(enc); e = np.asarray(fn(vp, log=log), np.float32); save(f"xjc_voice_{enc}", e, np.isfinite(e).all(1) & (np.abs(e).sum(1) > 0)); torch.cuda.empty_cache()
    shutil.copy(ROWS, OUT / "xjc_rows.csv")
    json.dump(dict(pair_local=True, frozen_encoders=True, labels_used=False, test_statistics=False, encoders=["clip", "simam", "ecapa", "sam100ft", "r293"]), open(OUT / "PROVENANCE.json", "w"))
    with zipfile.ZipFile("/kaggle/working/xjc_embeddings.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()): z.write(p, p.name)
    shutil.rmtree(OUT); log("JOB DONE")
except Exception as ex:
    log(f"VMERR {ex!r}"); traceback.print_exc(); raise
''')
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
d = HERE / "xjc_extract"; d.mkdir(exist_ok=True)
json.dump(nb, open(d / "xjc_extract.ipynb", "w"), indent=1)
json.dump({"id": "trncnguyn/flag-t3-xjc-extract", "title": "FLAG T3 xjc extract", "code_file": "xjc_extract.ipynb", "language": "python", "kernel_type": "notebook",
           "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
           "dataset_sources": ["trncnguyn/flag-t3-tools", "KAGGLE_USER/external-data-mavceleb", "trncnguyn/flag-t3-xjc-rows"],
           "competition_sources": [], "kernel_sources": []}, open(d / "kernel-metadata.json", "w"), indent=1)
print("built")
