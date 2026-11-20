"""MAV-Celeb v1 English-proxy export + Bengali LID (user 2026-11-12: English fix; "đừng copy code bên kia, làm theo ý mình").
The complete MAV v1 media is only mounted on Kaggle (KAGGLE_USER/external-data-mavceleb); the local copy has 28 of 64 faces and no voices.
This kernel (GPU, internet for the public LID models):
  1. copies the media of caches/ext/mavx_subset.csv (2,560 faces + 2,313 English voices, 64 ids) into mavx_media.zip
     -> the Mac extracts every feature with the SAME code path as the eval set (t3_eval_extract.py, t3_imagebind_extract.py);
  2. runs scripts/t3_lid_audit.py (VoxLingua107, Whisper-large-v3, MMS-LID-256) on those voices -> lid_mavx.csv
     (rule applied on the Mac with t3_lid_merge semantics: any model P(bn+as) > .5 on any clip drops the identity).
No training, no FLAG data, no test data.
  python build_t3_mavx_kernel.py <user>      (then kaggle datasets create -p mavx_ds ; kaggle kernels push -p t3_mavx)"""
import json, sys, tarfile
from pathlib import Path
K = Path(__file__).parent; T3 = K.parent; USER = sys.argv[1]
ds = K / "mavx_ds"; ds.mkdir(exist_ok=True)
with tarfile.open(ds / "mavx_files.tar", "w") as t:
    for f in ("caches/ext/mavx_subset.csv", "scripts/t3_lid_audit.py"):
        t.add(T3 / f, arcname=Path(f).name)
json.dump({"title": "flag-t3-mavx", "id": f"{USER}/flag-t3-mavx", "licenses": [{"name": "other"}]}, open(ds / "dataset-metadata.json", "w"))
CODE = r'''
import os, sys, glob, time, shutil, subprocess, zipfile, tarfile, traceback
from pathlib import Path
import pandas as pd
T0 = time.time()
def log(m): print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 60:6.1f}m]", m, flush=True)
W = Path("/kaggle/temp/mavx"); OUT = Path("/kaggle/working"); W.mkdir(parents=True, exist_ok=True)
try:
    hit = [h for h in glob.glob("/kaggle/input/**/mavx_files.tar", recursive=True)]
    if hit: tarfile.open(hit[0]).extractall(W)
    else:
        for n in ("mavx_subset.csv", "t3_lid_audit.py"): shutil.copy([h for h in glob.glob(f"/kaggle/input/**/{n}", recursive=True) if "mavx" in h][0], W / n)
    meta = [p for p in glob.glob("/kaggle/input/**/meta_v1.csv", recursive=True)]; assert meta, "MAV v1 not mounted"
    V1 = Path(meta[0]).parent / "v1"; S = pd.read_csv(W / "mavx_subset.csv"); log(f"MAV root {V1}; subset {len(S)} files")
    src = [V1 / ("faces" if k == "face" else "voices") / p for k, p in zip(S.kind, S.relpath)]; miss = [str(p) for p in src if not p.exists()]
    log(f"missing {len(miss)}"); assert not miss, miss[:5]
    with zipfile.ZipFile(OUT / "mavx_media.zip", "w", zipfile.ZIP_STORED) as z:
        for p, k, r in zip(src, S.kind, S.relpath): z.write(p, f"{'faces' if k == 'face' else 'voices'}/{r}")
    log("STAGE media zip done")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "speechbrain"], check=True)
    v = S[S.kind == "voice"]; pd.DataFrame(dict(src="v1", key=v.relpath, path=[str(V1 / "voices" / p) for p in v.relpath])).to_csv(W / "clips.csv", index=False)
    r = subprocess.run([sys.executable, "-u", str(W / "t3_lid_audit.py"), "--list", str(W / "clips.csv"), "--out", str(W / "lid"), "--device", "cuda", "--models", "vl,wh,mms"])
    assert r.returncode == 0, r.returncode
    outs = sorted((W / "lid").glob("lid_*.csv")); pd.concat([pd.read_csv(o) for o in outs]).to_csv(OUT / "lid_mavx.csv", index=False); log(f"STAGE lid done {len(outs)} files")
    log("JOB DONE")
except Exception as e:
    log(f"VMERR {e!r}"); traceback.print_exc(); raise
'''
compile(CODE, "mavx", "exec")
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": CODE}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
d = K / "t3_mavx"; d.mkdir(exist_ok=True); json.dump(nb, open(d / "t3_mavx.ipynb", "w"), indent=1)
json.dump({"id": f"{USER}/flag-t3-mavx-run", "title": "FLAG T3 mavx run", "code_file": "t3_mavx.ipynb", "language": "python", "kernel_type": "notebook",
           "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
           "dataset_sources": ["KAGGLE_USER/external-data-mavceleb", f"{USER}/flag-t3-mavx"], "competition_sources": [], "kernel_sources": []},
          open(d / "kernel-metadata.json", "w"), indent=1)
print("built", d, ds)
