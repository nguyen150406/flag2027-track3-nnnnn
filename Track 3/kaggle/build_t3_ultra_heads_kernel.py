"""ULTRA heads on Kaggle (user 2026-11-11 "chạy tối đa nhanh nhất", ncamtu06 allowed): scripts/t3_ultra_heads.py on T4 x2.
Inputs: trncnguyn/flag-t3-cv-bundle, -cv-extra (shared) + ncamtu06/flag-t3-overlay50 + overlay66 (base of the SUB66 build, which trained
the same frozen branches) + <user>/flag-t3-ultra-<tag> (overlay: current t3data/t3train/t3lib/t3_ultra_heads, ImageBind-face caches
caches/ibv + caches/ext/{vox2,vb}_face_ibv, exclude_ultra.csv, ultra_holdout.csv, cv_run/w/split.json [+ clipnf arrays for E2]).
One job list per GPU; a 1-epoch smoke model first (fail fast). Output: ultra_<tag>_models.zip (outputs/ultra/models/*.pt + .done).
  python build_t3_ultra_heads_kernel.py <user> <tag> "<gpu0 branches>" "<gpu1 branches>"     (then kaggle datasets create + kernels push)"""
import json, sys, tarfile
from pathlib import Path
K = Path(__file__).parent; T3 = K.parent; ROOT = T3.parent
USER, TAG, G0, G1 = sys.argv[1:5]
FSEEDS = sys.argv[5] if len(sys.argv) > 5 else "2027"; NSEEDS = sys.argv[6] if len(sys.argv) > 6 else "2027,42,3407"   # 13/11: fold / final seeds ("-" = none)
# 13/11: a GPU job may carry its own seeds: "<branches>@<fold seeds>@<final seeds>" (default = FSEEDS / NSEEDS)
G0B, G1B = G0.split("@")[0], G1.split("@")[0]
ds_dir = K / f"ultra_ds_{TAG}"; ds_dir.mkdir(exist_ok=True)
files = ["scripts/t3data.py", "scripts/t3train.py", "scripts/t3lib.py", "scripts/t3_ultra_heads.py", "caches/ext/exclude_ultra.csv",
         "caches/ext/ultra_holdout.csv", "cv_run/w/split.json"]
if any(b.startswith("ibv") for b in (G0B + "," + G1B).split(",")):  # ImageBind-face arrays only when an ibv branch trains (12/11: smaller upload)
    files += ["caches/ibv/face_embeddings.float32.npy"] + [f"caches/ext/{s}_face_ibv.{e}.npy" for s in ("vox2", "vb") for e in ("float32", "ok")]
sys.path.insert(0, str(T3 / "scripts")); import t3_ultra_heads as UH  # branch -> face cache (e.g. clipnf2D_cat4 -> clipnf2)
for nf in sorted({UH.BR[b][0] for b in (G0B + "," + G1B).split(",") if b and UH.BR[b][0].startswith("clipnf")}):  # clipnf / clipnf2 LoRA face arrays
    files += [f"caches/{nf}/face_embeddings.float32.npy"] + [f"caches/ext/{s}_face_{nf}.{e}.npy" for s in ("vox", "vox2", "vb") for e in ("float32", "ok")]
BRS = [b for b in (G0B + "," + G1B).split(",") if b]
if any(b.endswith(("mp", "yp")) for b in BRS): files += ["outputs/ultra/pool_drop10.csv"]   # "p" branches: R2-EP pool drop list
EXCL = "exclude_ultra_mav.csv" if any(b.startswith("clipD_") for b in BRS) else "exclude_ultra.csv"   # 12/11 data ablation: MAV held-out ids excluded too
files += [f"caches/ext/{EXCL}"]
for src in sorted({s_ for b in BRS for s_ in UH.BR[b][3]} - {"vox", "vox2", "vb"}):  # extra external sources (lrs3, avs, avs2, avssa, v1): rows + clip faces + cat4 voices
    files += [f"caches/ext/rows_{src}.csv"] + [f"caches/ext/{src}_face_{fn}.{e}.npy" for fn in sorted({UH.BR[b][0] for b in BRS if src in UH.BR[b][3]}) for e in ("float32", "ok")]
    files += [f"caches/ext/{src}_voice_{v}.{e}.npy" for v in ("simam", "sam100ft", "r293", "ecapa") for e in ("float32", "ok")]
files = list(dict.fromkeys(files))
with tarfile.open(ds_dir / f"ultra_{TAG}.tar", "w") as t:
    for f in files:
        p = T3 / f; assert p.exists(), p; t.add(p, arcname=str(Path("FLAG_2027/Track 3") / f))
json.dump({"title": f"flag-t3-ultra-{TAG}", "id": f"{USER}/flag-t3-ultra-{TAG}", "licenses": [{"name": "other"}]}, open(ds_dir / "dataset-metadata.json", "w"))
CODE = r'''
import os, sys, time, glob, shutil, subprocess, zipfile, traceback
from pathlib import Path
T0 = time.time()
def log(m): print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 60:6.1f}m]", m, flush=True)
TMP = Path("/kaggle/temp"); W = TMP / "w"; OUT = Path("/kaggle/working"); LG = OUT / "logs"
for d in (W, LG): d.mkdir(parents=True, exist_ok=True)
def sh(cmd, name):
    log(f"$ {cmd[:220]}")
    r = subprocess.run(["bash", "-c", f"set -o pipefail; {cmd} 2>&1 | tee -a {LG / name} | (grep --line-buffered -v -i -E 'warn|F_t =' || true)"])
    if r.returncode: raise RuntimeError(f"exit {r.returncode}: {cmd[:150]}")
def tree(ds, tarname):
    hits = [h for h in glob.glob(f"/kaggle/input/**/{tarname}", recursive=True) if ds in h]
    if hits: sh(f"tar xf '{hits[0]}' -C {W}", "extract.log"); return
    trees = [p for p in glob.glob("/kaggle/input/**/FLAG_2027", recursive=True) if ds in p]; assert trees, f"missing {ds}"
    sh(f"cp -a '{trees[0]}/.' {W}/FLAG_2027/", "extract.log")
T3 = W / "FLAG_2027/Track 3"; MD = T3 / "outputs/ultra/models"
def pack(name):
    with zipfile.ZipFile(OUT / name, "w", zipfile.ZIP_DEFLATED) as z:
        for p in MD.glob("*"): z.write(p, p.name)
try:
    import torch; ng = torch.cuda.device_count(); log(f"GPUs {ng}, CPUs {os.cpu_count()}"); assert ng >= 1, "no GPU"
    (W / "FLAG_2027").mkdir(exist_ok=True)
    for ds, tarname in (("flag-t3-cv-bundle", "flag_cv16.tar"), ("flag-t3-cv-extra", "flag_cv_extra.tar"), ("flag-t3-overlay50", "overlay50.tar"),
                        ("flag-t3-overlay66", "overlay66.tar"), ("flag-t3-ultra-__TAG__", "ultra___TAG__.tar")):
        tree(ds, tarname)
    SC = T3 / "scripts"; MD.mkdir(parents=True, exist_ok=True); sh(f"df -h {TMP} | tail -1", "extract.log"); log("STAGE extract done")
    env = f"cd '{SC}' && T3_EXCLUDE='{T3}/caches/ext/__EXCL__' T3_TRAIN_DEVICE=cuda OMP_NUM_THREADS=2"
    b0 = "__G0__".split("@")[0].split(",")[0]
    sh(f"{env} CUDA_VISIBLE_DEVICES=0 T3U_EPOCHS=1 python -u t3_ultra_heads.py {TMP}/smoke {b0} 0 1 2", "smoke.log"); log("STAGE smoke done")
    jobs = [("__G0__", 0), ("__G1__", 1 if ng > 1 else 0)]
    FS, NS = "__FSEEDS__", "__NSEEDS__"
    cmd = lambda br, g, folds, seeds: f"{env} CUDA_VISIBLE_DEVICES={g} python -u t3_ultra_heads.py '{MD}' {br} {folds} {seeds} 2"
    def job(spec, g):
        br, fs, ns = (spec.split("@") + [FS, NS])[:3] if "@" in spec else (spec, FS, NS)
        cs = [c for c in (cmd(br, g, "0,1,2,3,4", fs) if fs != "-" else "", cmd(br, g, "final", ns) if ns != "-" else "") if c]
        log(f"GPU{g} job: {br} folds s{fs} final s{ns}"); return "set -o pipefail; (" + " && ".join(cs) + f") 2>&1 | tee -a {LG}/gpu{g}.log"
    ps = [subprocess.Popen(["bash", "-c", job(br, g)]) for br, g in jobs if br]
    while any(p.poll() is None for p in ps):
        time.sleep(300); log(f"models done: {len(list(MD.glob('*.done')))}"); pack("ultra___TAG___partial.zip")
    assert all(p.returncode == 0 for p in ps), [p.returncode for p in ps]
    pack("ultra___TAG___models.zip"); log(f"JOB DONE {len(list(MD.glob('*.done')))} models")
except Exception as e:
    log(f"VMERR {e!r}"); traceback.print_exc()
    if MD.exists() and any(MD.glob("*.pt")): pack("ultra___TAG___partial.zip")
    for p in LG.glob("*.log"): print(f"----- tail {p.name}"); print(p.read_text()[-2500:])
    raise
'''.replace("__TAG__", TAG).replace("__G0__", G0).replace("__G1__", G1).replace("__EXCL__", EXCL).replace("__FSEEDS__", FSEEDS).replace("__NSEEDS__", NSEEDS)
compile(CODE, "ultra", "exec")
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": CODE}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
d = K / f"t3_ultra_heads_{TAG}"; d.mkdir(exist_ok=True); json.dump(nb, open(d / f"t3_ultra_heads_{TAG}.ipynb", "w"), indent=1)
json.dump({"id": f"{USER}/flag-t3-ultra-heads-{TAG}", "title": f"FLAG T3 ultra heads {TAG}", "code_file": f"t3_ultra_heads_{TAG}.ipynb", "language": "python",
           "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
           "dataset_sources": ["trncnguyn/flag-t3-cv-bundle", "trncnguyn/flag-t3-cv-extra", f"{USER}/flag-t3-overlay50", f"{USER}/flag-t3-overlay66", f"{USER}/flag-t3-ultra-{TAG}"],
           "competition_sources": [], "kernel_sources": []}, open(d / "kernel-metadata.json", "w"), indent=1)
print("built", d, "and", ds_dir, sum((T3 / f).stat().st_size for f in files) / 1e6, "MB")
