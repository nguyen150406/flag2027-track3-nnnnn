"""SUB26 on Kaggle (account ncamtu06, GPU): CLIP ViT-L LoRA with the v1 recipe (1515 steps, FLAG + vox/vox2/vb/lrs3) PLUS a
South-Asian domain batch every step (caches/ext/domain_sa.csv: 450 VoxCeleb2 + 968 VoxBlink speakers of Hindi/Urdu/Tamil/...),
then the 5 clipsa projector branches (25 projectors). Combined locally with SUB21a's 8 frozen branches (scripts/t3_add_branches.py,
replace=clipsa_:clipft_ on t3_sub22b_tta2) after a local flip-TTA re-score -> compare with sub22b_tta2 (18.57).
Inputs: trncnguyn datasets shared privately with ncamtu06 (flag-t3-cv-bundle/-cv-extra = Track tree + caches, flag-t3-ft-local = FLAG/
LRS3 faces, flag-2027-data = FLAG dev faces) + ncamtu06/flag-t3-overlay26 (current code + small build-time files, extracted last);
VoxCeleb1/2 faces + VoxBlink media via temporary per-file links to trncnguyn's kernel outputs (made at build time).
Usage: python build_t3_sub26_kernel.py  (then KAGGLE_API_TOKEN=... kaggle kernels push -p t3_sub26)"""
import json, sys
from pathlib import Path
K = Path(__file__).parent

def links():  # fresh temporary links (trncnguyn account = default ~/.kaggle token)
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    api = KaggleApi(); api.authenticate(); out = {}
    with api.build_kaggle_client() as k:
        for slug, want in (("flag-t3-vox2faces", "vox2_faces.zip"), ("flag-t3-voxblink-prep", "vb_media.zip"), ("flag-t3-vox1faces", "vox1_faces.zip")):
            r = ApiListKernelSessionOutputRequest(); r.user_name = "trncnguyn"; r.kernel_slug = slug
            for f in k.kernels.kernels_api_client.list_kernel_session_output(r).files:
                if f.file_name == want: out[want] = f.url
    assert len(out) == 3, out
    return out

CODE = r'''
import os, sys, json, time, shutil, subprocess, zipfile, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
T0 = time.time()
def log(m): print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 60:6.1f}m]", m, flush=True)
URLS = __URLS__
IN = Path("/kaggle/input"); INDEX = {}
for root, dirs, files in os.walk(IN):  # bounded walk (phase-1 kernel): never descend into big image trees
    d = Path(root)
    for x in dirs + files: INDEX.setdefault(x, []).append(d / x)
    if len(d.relative_to(IN).parts) >= 5 or d.name in ("flag", "lrs3", "img", "vox2", "vox", "faces", "voices"): dirs[:] = []
def find(marker): return INDEX.get(marker, [])
TMP = Path("/kaggle/temp"); W, IMG, DL = TMP / "w", TMP / "img", TMP / "dl"
T3 = W / "FLAG_2027/Track 3"; SC = T3 / "scripts"; OUT = Path("/kaggle/working"); LG = OUT / "logs"
for d in (W, IMG, DL, LG): d.mkdir(parents=True, exist_ok=True)
def sh(cmd, name):  # stream to the Kaggle log AND keep a file; fail fast with the tail of the log
    log(f"$ {cmd[:220]}")
    r = subprocess.run(["bash", "-c", f"set -o pipefail; {cmd} 2>&1 | tee -a {LG / name} | (grep --line-buffered -v -i -E 'warn|UNEXPECTED|text_model|detach' || true)"])
    if r.returncode: raise RuntimeError(f"exit {r.returncode}: {cmd[:150]}")
def fetch(url, out, n=8, chunk=64 << 20):
    r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=120); total = int(r.headers["Content-Range"].split("/")[1])
    with open(out, "wb") as f: f.truncate(total)
    def part(a):
        b = min(a + chunk, total) - 1
        for k in range(6):
            try:
                d = requests.get(url, headers={"Range": f"bytes={a}-{b}"}, timeout=600).content; assert len(d) == b - a + 1
                with open(out, "r+b") as f: f.seek(a); f.write(d)
                return len(d)
            except Exception:
                time.sleep(5 * (k + 1))
        raise RuntimeError(f"range {a}-{b} failed")
    t = time.time()
    with ThreadPoolExecutor(n) as ex: got = sum(ex.map(part, range(0, total, chunk)))
    log(f"downloaded {out.name}: {got / 1e9:.2f} GB in {time.time() - t:.0f} s"); assert got == total; zipfile.ZipFile(out).namelist()
def tree(ds, tarname):  # a tar in a dataset, or the FLAG_2027 folder Kaggle auto-extracted it into
    hits = [h for h in find(tarname) if ds in str(h)]
    if hits:
        sh(f"tar xf '{hits[0]}' -C {W}", "extract.log"); return
    trees = [p for p in find("FLAG_2027") if ds in str(p)]; assert trees, f"missing {ds}"
    sh(f"cp -a '{trees[0]}/.' {W}/FLAG_2027/", "extract.log")
try:
    import torch; ng = torch.cuda.device_count(); log(f"GPUs {ng}: {[torch.cuda.get_device_name(i) for i in range(ng)]}; cpus {os.cpu_count()}; inputs {sorted(p.name for p in IN.iterdir())}")
    assert ng >= 1, "no GPU"
    (W / "FLAG_2027").mkdir(exist_ok=True)
    tree("flag-t3-cv-bundle", "flag_cv16.tar"); tree("flag-t3-cv-extra", "flag_cv_extra.tar"); tree("flag-t3-overlay26", "overlay26.tar")  # overlay last = newest files
    sh(f"df -h {TMP} | tail -1", "extract.log"); log("STAGE extract done")
    def local(zipname, dirname):  # ft-local images: a zip, or the folder Kaggle auto-extracted
        z = find(zipname)
        if z:
            with zipfile.ZipFile(z[0]) as f: f.extractall(IMG / dirname)
            return IMG / dirname
        d = find(dirname); assert d, f"missing {zipname} / {dirname}"; return d[0]
    flag_root = local("flag_faces.zip", "flag_faces"); lrs3_root = local("lrs3_faces.zip", "lrs3_faces")
    # the overlay also carries data/dev_set (test lists only) -> take the dev_set that actually has the face images (flag-2027-data)
    dev = [p for p in find("dev_set") if (p / "gender/English_test/faces/00000.jpg").exists()]; assert dev, "missing dev faces (flag-2027-data)"; dev_root = dev[0]
    log(f"dev faces root {dev_root} (candidates {find('dev_set')})")
    for k in ("vox2_faces.zip", "vb_media.zip", "vox1_faces.zip"): fetch(URLS[k], DL / k)
    with zipfile.ZipFile(DL / "vox2_faces.zip") as z: z.extractall(IMG / "vox2_faces")
    with zipfile.ZipFile(DL / "vox1_faces.zip") as z: z.extractall(IMG / "vox1_faces")
    with zipfile.ZipFile(DL / "vb_media.zip") as z: z.extractall(TMP / "vbm", [m for m in z.namelist() if m.startswith("img/")])
    import pandas as pd
    rv = pd.read_csv(T3 / "caches/ext/rows_vb.csv"); vb_root = IMG / "vbroot"; n = 0
    for f, ci in zip(rv.face, rv.clip_index):
        src = TMP / "vbm/img" / f"{ci}_{f.rsplit('_', 1)[1]}"
        if src.exists():
            dst = vb_root / f; dst.parent.mkdir(parents=True, exist_ok=True); shutil.move(src, dst); n += 1
    log(f"roots flag {flag_root} dev {dev_root} lrs3 {lrs3_root}; VoxBlink rows with image {n}/{len(rv)}")
    shutil.rmtree(DL); shutil.rmtree(TMP / "vbm"); log("STAGE download done")
    pre = ("import json, numpy as np, pandas as pd, t3_build as TB; cfg = json.load(open('../configs/t3_sub26.json')); A = TB.AGE; "
           "TB.age_offset(np.load(A / 'train_face_age.npy'), np.load(A / 'train_voice_age.npy')); "
           "[np.load(A / f'{p}_{l}_{m}_age.npy') for (p, l) in TB.EXPECTED for m in ('face', 'voice')]; "
           "fa, va = TB.age_v2(cfg); pf, pv = TB.gender_v2(); assert np.isfinite(fa).all() and len(pf) == 11349; "
           "d = pd.read_csv('../caches/ext/domain_sa.csv'); e = pd.read_csv('../caches/ext/exclude_bengali.csv'); print('preflight ok', len(d), 'domain ids', len(e), 'excluded')")
    sh(f'cd "{SC}" && python -c "{pre}"', "preflight.log"); log("STAGE preflight done")
    roots = f"flag={flag_root} flagdev={dev_root} lrs3={lrs3_root} vox={IMG / 'vox1_faces'} vox2={IMG / 'vox2_faces'} vb={vb_root}"
    FT = f"python -u t3_ft_clip.py --fold -1 --domain ../caches/ext/domain_sa.csv --tag clipsa --device cuda --workers 4 --roots {roots}"
    t = time.time(); sh(f"cd '{SC}' && T3_FT_DEBUG_N=256 {FT} --steps 20 --out {TMP}/smoke", "smoke.log")
    log(f"STAGE smoke done: 20 steps + debug encode in {(time.time() - t) / 60:.1f} min")
    sh(f"cd '{SC}' && {FT} --out {TMP}/ft", "ft.log")
    (T3 / "caches/clipsa").mkdir(exist_ok=True); shutil.copyfile(f"{TMP}/ft/flag_face_clipsa.float32.npy", T3 / "caches/clipsa/face_embeddings.float32.npy")
    for s in ("vox", "vox2", "vb", "lrs3"):
        for ext in ("float32", "ok"): shutil.copyfile(f"{TMP}/ft/{s}_face_clipsa.{ext}.npy", T3 / f"caches/ext/{s}_face_clipsa.{ext}.npy")
    log("STAGE ft done")
    sh(f"cd '{SC}' && T3_TRAIN_DEVICE=cuda T3_AUX_DEVICE=0 T3_NGPU={ng} OMP_NUM_THREADS=1 python -u t3_build_par.py ../configs/t3_sub26.json ../submissions/t3_sub26 4", "build.log")
    log("STAGE build done")
    S = T3 / "submissions"; shutil.copy(f"{TMP}/ft/clipsa.pt", S / "t3_sub26"); shutil.copy(f"{TMP}/ft/flag_face_clipsa.float32.npy", S / "t3_sub26")
    with zipfile.ZipFile(OUT / "sub26_out.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for d in (S / "t3_sub26", S / "t3_sub26.models"):
            for p in d.rglob("*"):
                if p.is_file(): z.write(p, p.relative_to(T3))
    log(f"JOB DONE -> {OUT / 'sub26_out.zip'} {(OUT / 'sub26_out.zip').stat().st_size / 1e6:.0f} MB")
except Exception as e:
    log(f"VMERR {e!r}"); traceback.print_exc()
    for p in LG.glob("*.log"): print(f"----- tail {p.name}"); print(p.read_text()[-2500:])
    raise
'''

urls = links()
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": CODE.replace("__URLS__", json.dumps(urls))}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
d = K / "t3_sub26"; d.mkdir(exist_ok=True)
json.dump(nb, open(d / "t3_sub26.ipynb", "w"), indent=1)
json.dump({"id": "ncamtu06/flag-t3-sub26", "title": "FLAG T3 sub26", "code_file": "t3_sub26.ipynb", "language": "python", "kernel_type": "notebook",
           "is_private": True, "enable_gpu": True, "enable_internet": True, "dataset_sources": ["trncnguyn/flag-t3-cv-bundle", "trncnguyn/flag-t3-cv-extra", "trncnguyn/flag-t3-ft-local", "trncnguyn/flag-2027-data", "ncamtu06/flag-t3-overlay26"],
           "competition_sources": [], "kernel_sources": []}, open(d / "kernel-metadata.json", "w"), indent=1)
print("built", d, sorted(urls))
