"""XJ-1+IB features (15/11, user "Ý XJ-1 + ImageBind: thêm khối ImageBind vào XJ-1 ... thử đi", prereg PREREG_gate_all.txt 22:27 sub109a).
Kaggle notebook ncamtu06/flag-t3-xj1ib-extract: frozen ImageBind-huge face (vision) + voice (audio) embeddings of XJ-1's external pool
  faces : MAV xjc set (9,520, xjc_rows order) + the unique VoxCeleb1 rows_vox faces (12,423, first-appearance order)
  voices: MAV xjc set (8,779, xjc_rows order) + the VoxCeleb1 rows_vox voices (12,423, rows_vox order)
with OUR extractor scripts/t3_imagebind_extract.py (embedded, sha-checked) = the code of the FLAG-train XP65 / eval / mav_en2 IB caches:
public ImageBind commit 53680b02 (git clone, 4 model files sha-checked vs our copy), checkpoint HF seung275/imagebind_huge.pth (sha256 checked),
fp32, libjpeg-turbo decode, 3 fixed 2-s audio crops. PARITY GATE first (fail fast): MAV v1 items shared with outputs/mav_en2 (our local IB caches of
the same files; bytes checked on samples) -> mean cos > .98 per modality, else the worker stops. Two workers (GPU0 faces, GPU1 voices); the voice
worker loads its model after the face worker has loaded (RAM). Each file is encoded on its own (pair-local); no labels; nothing is trained.
usage: python build_xj1ib_extract.py  -> kaggle/xj1ib_extract/  (then KAGGLE_API_TOKEN="$(< ~/.kaggle_acc2/access_token)" kaggle kernels push -p xj1ib_extract)"""
import json, gzip, base64, hashlib, re
from pathlib import Path
import numpy as np, pandas as pd

K = Path(__file__).resolve().parent; T3 = K.parent; SC = T3 / "scripts"; IBD = T3 / "caches/imagebind"
COMMIT = "53680b02d7e37b19b124fa37bae4b6c98c38f5be"; CKPT_SHA = "d6f6c22bedcc90708448d5d2fbb7b2db9c73f505dc89bd0b2e09b23af1b62157"
b64 = lambda b: base64.b64encode(gzip.compress(b, 9)).decode()
f16 = lambda a: b64(np.ascontiguousarray(np.asarray(a, np.float32)).astype(np.float16).tobytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
USER = "ncamtu06"
suf = lambda s: "/".join(str(s).split("/")[-4:])

# ------------------------------------------------------------------ rows
R = pd.read_csv(T3 / "kaggle/xjc_rows/xjc_rows.csv"); fr, vr = R[R.kind.eq("face")].reset_index(drop=True), R[R.kind.eq("voice")].reset_index(drop=True)
vox = pd.read_csv(T3 / "caches/ext/rows_vox.csv"); uf = vox.face.drop_duplicates().tolist()
F_ROWS = pd.concat([pd.DataFrame(dict(src=fr.src, rel=fr.rel)), pd.DataFrame(dict(src="vox1", rel=uf))], ignore_index=True)
V_ROWS = pd.concat([pd.DataFrame(dict(src=vr.src, rel=vr.rel)), pd.DataFrame(dict(src="vox1", rel=vox.voice))], ignore_index=True)
assert len(F_ROWS) == len(fr) + len(uf) and len(V_ROWS) == len(vr) + len(vox) and V_ROWS.rel.is_unique and F_ROWS.rel.is_unique

# ------------------------------------------------------------------ parity items: MAV v1 rows also in outputs/mav_en2 (local IB caches, same extractor)
t = pd.read_csv(T3 / "outputs/mav_en2/rows.csv"); IBF, IBV = np.load(T3 / "outputs/mav_en2/ultra/ib_face.npy"), np.load(T3 / "outputs/mav_en2/ultra/ib_voice.npy")
assert len(IBF) == len(IBV) == len(t)
fpos = {suf(p): i for i, p in list(enumerate(t.face))[::-1]}; vpos = {suf(p): i for i, p in list(enumerate(t.voice))[::-1]}
fc = [i for i in range(len(fr)) if fr.src[i] == "v1" and suf(fr.rel[i]) in fpos]; vc = [i for i in range(len(vr)) if vr.src[i] == "v1" and suf(vr.rel[i]) in vpos]
fs = np.array(fc)[np.unique(np.linspace(0, len(fc) - 1, min(64, len(fc))).astype(int))]; vs = np.array(vc)[np.unique(np.linspace(0, len(vc) - 1, min(64, len(vc))).astype(int))]
# bytes of the local files behind the references == the teammate's files (checked where both exist locally: faces in v1_complete, one voice downloaded)
loc = T3.parent / "external_mavceleb/v1_complete/mavceleb_v1_complete-001"
nb = [(loc / fr.rel[i].split("/", 1)[1]) for i in fs]; nb = [(p, Path(t.face[fpos[suf(fr.rel[i])]])) for p, i in zip(nb, fs) if p.exists()]
assert all(a.read_bytes() == b.read_bytes() for a, b in nb), "local face bytes differ"
PAR = dict(fidx=fs.tolist(), vidx=vs.tolist(), face=f16(IBF[[fpos[suf(fr.rel[i])] for i in fs]]), voice=f16(IBV[[vpos[suf(vr.rel[i])] for i in vs]]))
SRC = (SC / "t3_imagebind_extract.py").read_bytes(); MODF = {f: sha(IBD / f"ImageBind-{COMMIT}/imagebind/models/{f}.py") for f in ("imagebind_model", "helpers", "multimodal_preprocessors", "transformer")}

COMMON = r'''
import os, sys, subprocess, time, shutil, zipfile, json, glob, traceback, hashlib, gzip, base64, io
from pathlib import Path
import numpy as np, pandas as pd
T0 = time.time()
def log(m): print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 60:6.1f}m]", m, flush=True)
def run(c): print("+", c, flush=True); subprocess.run(c, shell=True, check=True)
def find_dirs(name, depth=4):   # depth-limited: never walk the media trees
    return [p for d in range(1, depth + 1) for p in glob.glob("/kaggle/input/" + "/".join(["*"] * d)) if os.path.basename(p) == name]
def fsha(p, bs=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""): h.update(b)
    return h.hexdigest()
ub = lambda s: gzip.decompress(base64.b64decode(s))
W = Path("/kaggle/temp/xj1ib"); W.mkdir(parents=True, exist_ok=True); OUT = Path("/kaggle/working/xj1ib_out"); OUT.mkdir(parents=True, exist_ok=True)
ROOT = W / "root"; S = ROOT / "Track 3/scripts"; IBD = ROOT / "Track 3/caches/imagebind"
def run_workers(jobs):
    """jobs = [(name, script_path, gpu)] -> in parallel (one per GPU) when 2 GPUs exist, else one after another; stream the log tails."""
    import torch
    ng = torch.cuda.device_count(); log(f"GPUs {ng}: {[torch.cuda.get_device_name(i) for i in range(ng)]}")
    batches = [jobs] if ng >= 2 else [[j] for j in jobs]
    for b in batches:
        ps = {}
        for name, script, gpu in b:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu if ng >= 2 else 0), PYTHONUNBUFFERED="1", XJ1IB_PAR_WAIT="1" if (ng >= 2 and name == "voices") else "0")
            ps[name] = (subprocess.Popen([sys.executable, str(script)], stdout=open(W / f"{name}.log", "w"), stderr=subprocess.STDOUT, env=env, cwd=str(S)), W / f"{name}.log")
        last = time.time()
        while any(p.poll() is None for p, _ in ps.values()):
            time.sleep(20)
            if any(p.poll() not in (None, 0) for p, _ in ps.values()):          # fail fast: one worker died -> stop the other
                for p, _ in ps.values():
                    if p.poll() is None: p.kill()
            if time.time() - last > 300:
                last = time.time()
                for name, (p, lf) in ps.items():
                    tail = [l for l in open(lf).read().splitlines() if l.strip()][-1:] or ["(no output yet)"]; log(f"  [{name}] {tail[0][:200]}")
        for name, (p, lf) in ps.items():
            txt = open(lf).read(); print(f"===== {name} log (last 40 lines)\n" + "\n".join(txt.splitlines()[-40:]), flush=True)
        for name, (p, lf) in ps.items(): assert p.returncode == 0, f"worker {name} failed (rc {p.returncode})"
'''

WORKER = r'''
import os, sys, time, json
from pathlib import Path
import numpy as np, pandas as pd, torch
KIND = sys.argv[1]; T0 = time.time()
def log(m): print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 60:6.1f}m]", KIND, m, flush=True)
W = Path("/kaggle/temp/xj1ib"); OUT = Path("/kaggle/working/xj1ib_out"); sys.path.insert(0, str(W / "root/Track 3/scripts"))
L = pd.read_csv(W / f"{KIND}s.csv"); paths, good = L.path.tolist(), L.good.to_numpy(bool); P = json.load(open(W / "par.json"))
ref = np.frombuffer(__import__("gzip").decompress(__import__("base64").b64decode(P[KIND])), np.float16).astype(np.float32).reshape(-1, 1024)
pp = [paths[i] for i in P["fidx" if KIND == "face" else "vidx"]]; assert len(pp) == len(ref)
if os.environ.get("XJ1IB_PAR_WAIT") == "1":                       # RAM: load after the face worker has its (pruned) model
    t = time.time()
    while not (W / "face_model_ready").exists() and time.time() - t < 1200: time.sleep(10)
import t3_imagebind_extract as T
dev = torch.device("cuda"); m = T.model(dev); torch.cuda.empty_cache(); log(f"model ready ({torch.cuda.memory_allocated() / 2 ** 30:.1f} GB on GPU)")
if KIND == "face": (W / "face_model_ready").touch()
u = lambda E: E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
cs = (u(T.encode(m, pp, KIND, dev, bs=32)) * u(ref)).sum(1); pj = dict(mean=float(cs.mean()), min=float(cs.min()), p05=float(np.percentile(cs, 5)), n=int(len(cs)))
json.dump(pj, open(OUT / f"ib_{KIND}_parity.json", "w"), indent=1); log(f"PARITY ib {KIND} vs our local mav_en2 cache: {pj}")
assert cs.mean() > .98, f"ib {KIND} does not reproduce our cache: {pj}"
t = time.time(); E = T.encode(m, paths, KIND, dev, bs=32).astype(np.float32); log(f"encoded {len(E)} in {(time.time() - t) / 60:.1f} min")
ok = good & np.isfinite(E).all(1) & (np.abs(E).sum(1) > 0)
np.save(OUT / f"xj1ib_{KIND}.float32.npy", E); np.save(OUT / f"xj1ib_{KIND}.ok.npy", ok); log(f"SAVED {E.shape} ok {int(ok.sum())}/{len(ok)}")
'''

MAIN = r'''
try:
    run("pip install -q ftfy iopath einops 2>&1 | tail -1")
    # ---- our extractor (sha-checked), public ImageBind source at the pinned commit (model files sha-checked vs our copy), HF checkpoint (sha-checked)
    S.mkdir(parents=True, exist_ok=True); src = ub("__SRC__"); assert hashlib.sha256(src).hexdigest() == "__SRCSHA__"; (S / "t3_imagebind_extract.py").write_bytes(src)
    IBD.mkdir(parents=True, exist_ok=True); (IBD / "pylib").mkdir(exist_ok=True); rp = IBD / "ImageBind-__COMMIT__"
    run(f"git clone -q https://github.com/facebookresearch/ImageBind.git '{rp}' && git -C '{rp}' checkout -q __COMMIT__")   # quoted: the path has a space
    for f, h in json.loads(r"""__MODF__""").items(): assert fsha(rp / f"imagebind/models/{f}.py") == h, f"ImageBind {f}.py differs from our copy"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    from huggingface_hub import hf_hub_download
    ck = hf_hub_download("seung275/imagebind_huge.pth", "imagebind_huge.pth", cache_dir=str(W / "hf")); h = fsha(ck); log(f"checkpoint sha256 {h}")
    assert h == "__CKSHA__", "checkpoint differs"; os.symlink(os.path.realpath(ck), IBD / "imagebind_huge.pth")
    # ---- media paths
    roots = {}
    for top in ("mavceleb_v1_complete-001", "mavceleb_v2_complete-002"):
        hh = [p for p in find_dirs(top) if os.path.isdir(p)]; assert hh, f"{top} not mounted"; roots[top] = Path(hh[0]).parent
    vb = [p for p in find_dirs("VoxCeleb_1_adapted", 4) if os.path.isdir(os.path.join(p, "face"))]; assert vb, f"VoxCeleb_1_adapted not mounted: {os.listdir('/kaggle/input')}"
    VF = Path(vb[0]) / "face"; base = {}
    for slug in ("voxceleb1train", "voxceleb1test"):
        hh = [p for p in find_dirs(slug, 3) if os.path.isdir(os.path.join(p, "wav"))]; assert hh, f"{slug} not mounted: {os.listdir('/kaggle/input')}"; base[slug] = Path(hh[0]) / "wav"
    def vface(rel):
        for s in ("train", "test", "dev", ""):
            p = VF / s / rel
            if p.exists(): return str(p)
        return None
    def vvoice(rel):
        for b in base.values():
            p = b / rel
            if p.exists(): return str(p)
        return None
    import cv2, soundfile as sf
    def wok(p):
        try: return p is not None and sf.info(p).frames > 0
        except Exception: return False
    P = json.loads(r"""__PAR__"""); json.dump(P, open(W / "par.json", "w"))
    for kind in ("face", "voice"):
        R = pd.read_csv(io.BytesIO(ub("__FROWS__" if kind == "face" else "__VROWS__")))
        ps = [str(roots[r.split("/")[0]] / r) if s in ("v1", "v2") else (vface(r) if kind == "face" else vvoice(r)) for s, r in zip(R.src, R.rel)]
        good = np.array([p is not None and cv2.imread(p) is not None for p in ps] if kind == "face" else [wok(p) for p in ps])
        for s in ("mav", "vox1"):
            m = R.src.isin(["v1", "v2"]).to_numpy() if s == "mav" else R.src.eq("vox1").to_numpy()
            log(f"{kind}s {s}: {int(m.sum())} usable {int(good[m].sum())}"); assert good[m].mean() > .99, f"too many unusable {s} {kind} files"
        sub = ps[int(np.argmax(good))]
        pd.DataFrame(dict(path=[p if g else sub for p, g in zip(ps, good)], good=good)).to_csv(W / f"{kind}s.csv", index=False); R.to_csv(OUT / f"xj1ib_{kind}_rows.csv", index=False)
    open(W / "worker.py", "w").write(r"""__WORKER__"""); open(W / "faces.py", "w").write("import sys; sys.argv = ['w', 'face']; exec(open('/kaggle/temp/xj1ib/worker.py').read())")
    open(W / "voices.py", "w").write("import sys; sys.argv = ['w', 'voice']; exec(open('/kaggle/temp/xj1ib/worker.py').read())")
    log("setup done -> workers")
    run_workers([("faces", W / "faces.py", 0), ("voices", W / "voices.py", 1)])
    json.dump(dict(pair_local=True, frozen_encoders=True, labels_used=False, test_statistics=False, prereg="sub109a 15/11 22:27",
                   imagebind=f"facebookresearch/ImageBind@__COMMIT__ (model files sha == our copy), checkpoint HF seung275/imagebind_huge.pth sha256 __CKSHA__",
                   extractor="scripts/t3_imagebind_extract.py sha256 __SRCSHA__ (fp32, turbo decode, 3 fixed 2-s audio crops)",
                   parity={k: json.load(open(OUT / f"ib_{k}_parity.json")) for k in ("face", "voice")}), open(OUT / "PROVENANCE.json", "w"), indent=1)
    with zipfile.ZipFile("/kaggle/working/xj1ib.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()): z.write(p, p.name)
    shutil.rmtree(OUT); log("JOB DONE")
except Exception as ex:
    log(f"VMERR {ex!r}"); traceback.print_exc(); raise
'''

if __name__ == "__main__":
    assert '"""' not in WORKER; compile(WORKER, "worker", "exec")
    rep = {"__SRC__": b64(SRC), "__SRCSHA__": hashlib.sha256(SRC).hexdigest(), "__COMMIT__": COMMIT, "__MODF__": json.dumps(MODF), "__CKSHA__": CKPT_SHA,
           "__PAR__": json.dumps(PAR), "__FROWS__": b64(F_ROWS.to_csv(index=False).encode()), "__VROWS__": b64(V_ROWS.to_csv(index=False).encode()), "__WORKER__": WORKER}
    src = MAIN
    for a, b in rep.items(): assert a in src, a; src = src.replace(a, b)
    left = set(re.findall(r"__[A-Z0-9]+__", src)); assert not left, left
    cells = [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s} for s in (COMMON, src)]
    for c in cells: compile(c["source"], "cell", "exec")
    d = K / "xj1ib_extract"; d.mkdir(exist_ok=True)
    json.dump({"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5},
              open(d / "xj1ib_extract.ipynb", "w"))
    json.dump({"id": f"{USER}/flag-t3-xj1ib-extract", "title": "FLAG T3 xj1ib extract", "code_file": "xj1ib_extract.ipynb", "language": "python", "kernel_type": "notebook",
               "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
               "dataset_sources": ["KAGGLE_USER/external-data-mavceleb", "robot12138/voxceleb-1", "kryakrya/voxceleb1train", "kryakrya/voxceleb1test"],
               "competition_sources": [], "kernel_sources": []}, open(d / "kernel-metadata.json", "w"), indent=1)
    (d / "worker.py").write_text(WORKER); F_ROWS.to_csv(d / "face_rows_local.csv", index=False); V_ROWS.to_csv(d / "voice_rows_local.csv", index=False)
    json.dump(PAR | {"face": "(f16 b64)", "voice": "(f16 b64)"}, open(d / "par_local.json", "w"))
    print(f"built {d} | notebook {(d / 'xj1ib_extract.ipynb').stat().st_size / 1e6:.2f} MB | faces {len(F_ROWS)} voices {len(V_ROWS)} | parity faces {len(fs)} "
          f"voices {len(vs)} | face bytes checked {len(nb)}")
