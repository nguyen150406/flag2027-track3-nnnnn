"""IB-LoRA training kernel (16/11, prereg PREREG_gate_all.txt 20:10 "PREREG sub118a").
Kaggle notebook <user>/flag-t3-iblora (env IBL_USER, default ncamtu06): trains LoRA adapters on ImageBind-huge with OUR scripts/t3_iblora.py
(+ scripts/t3_imagebind_extract.py for model / preprocessing; both embedded, sha-checked) on the XJ external pool:
  faces / voices = the xj1ib rows (MAV xjc set + VoxCeleb1 rows_vox) that our frozen-IB Kaggle extraction marked ok, voices through the clip-level
  Bengali LID rule exactly as XJ (xj_ext.clip_lid: MAV clips all checked + not flagged, VoxCeleb1 India-nationality ids clip-checked), identity keys
  as XJ ("mavceleb_v1/id0001", "voxceleb1/id10001"), genders from xj/ext_gender.json (u = unknown -> never in gender-pure batches). NO FLAG data (MAV ids that are FLAG train people, xj_ext.OVERLAP, removed - review 16/11).
  hold-out: 40 VoxCeleb1 ids (20 m / 20 f) + 10 MAV ids (5 m / 5 f), 4 faces + 4 voices each (split 'val'); 256 + 256 training items (split 'ctr')
  give the centring means for the val score. Step-0 parity: val embeddings vs the frozen xj1ib arrays of the same files (cos > .99, fail fast).
ImageBind source = public commit 53680b02 (git clone; 4 model files sha-checked vs our copy), checkpoint HF seung275/imagebind_huge.pth (sha256).
usage: python build_iblora.py -> kaggle/iblora/ (then KAGGLE_API_TOKEN="$(< ~/.kaggle_acc2/access_token)" kaggle kernels push -p kaggle/iblora)"""
import os, sys, json, gzip, base64, hashlib, re
from pathlib import Path
import numpy as np, pandas as pd

K = Path(__file__).resolve().parent; T3 = K.parent; SC = T3 / "scripts"; IBD = T3 / "caches/imagebind"; XD = T3 / "xj"
sys.path.insert(0, str(XD)); import xj_ext as XE
COMMIT = "53680b02d7e37b19b124fa37bae4b6c98c38f5be"; CKPT_SHA = "d6f6c22bedcc90708448d5d2fbb7b2db9c73f505dc89bd0b2e09b23af1b62157"
b64 = lambda b: base64.b64encode(gzip.compress(b, 9)).decode()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
USER = os.environ.get("IBL_USER", "ncamtu06"); OUTD = K / os.environ.get("IBL_DIR", "iblora")

# ------------------------------------------------------------------ items
R = pd.read_csv(T3 / "kaggle/xjc_rows/xjc_rows.csv"); fr, vr = R[R.kind.eq("face")].reset_index(drop=True), R[R.kind.eq("voice")].reset_index(drop=True)
vox = pd.read_csv(T3 / "caches/ext/rows_vox.csv"); uf = vox.face.drop_duplicates().tolist()
XI = XD / "xj1ib"; IFr, IVr = pd.read_csv(XI / "xj1ib_face_rows.csv"), pd.read_csv(XI / "xj1ib_voice_rows.csv")
IF, IV = np.load(XI / "xj1ib_face.float32.npy"), np.load(XI / "xj1ib_voice.float32.npy"); IFok, IVok = np.load(XI / "xj1ib_face.ok.npy"), np.load(XI / "xj1ib_voice.ok.npy")
okr = lambda a: np.isfinite(a).all(1) & (np.abs(a).sum(1) > 0)
IFok, IVok = IFok & okr(IF), IVok & okr(IV)
assert IFr.iloc[:len(fr)].reset_index(drop=True).equals(fr[["src", "rel"]]) and IVr.iloc[:len(vr)].reset_index(drop=True).equals(vr[["src", "rel"]]), "xj1ib MAV rows"
assert (IVr.iloc[len(vr):].rel.values == vox.voice.values).all() and (IFr.iloc[len(fr):].rel.values == np.array(uf)).all(), "xj1ib vox rows"
chk, fl = XE.clip_lid(); ok_lid = lambda keys: np.array([k in chk and k not in fl for k in keys])
ex = pd.read_csv(T3 / "caches/ext/exclude_bengali_v3.csv"); india = set(ex[ex.src.eq("vox")].identity.astype(str))
assert not vr.lang.astype(str).str.lower().isin(["bengali", "bangla", "bn"]).any()
mv = IVok[:len(vr)] & ok_lid(vr.src + ":" + vr.rel.str.split("/voices/").str[1])
xv = IVok[len(vr):] & (~vox.identity.astype(str).isin(india).to_numpy() | ok_lid("vox1:" + vox.voice))
V = pd.concat([pd.DataFrame(dict(kind="voice", src=vr.src, rel=vr.rel, k="mavceleb_" + vr.src + "/" + vr.id, ref=np.arange(len(vr))))[mv],
               pd.DataFrame(dict(kind="voice", src="vox1", rel=vox.voice, k="voxceleb1/" + vox.identity.astype(str), ref=len(vr) + np.arange(len(vox))))[xv]])
Fc = pd.concat([pd.DataFrame(dict(kind="face", src=fr.src, rel=fr.rel, k="mavceleb_" + fr.src + "/" + fr.id, ref=np.arange(len(fr))))[IFok[:len(fr)]],
                pd.DataFrame(dict(kind="face", src="vox1", rel=uf, k=["voxceleb1/" + f.split("/")[0] for f in uf], ref=len(fr) + np.arange(len(uf))))[IFok[len(fr):]]])
both = (set(V.k) & set(Fc.k)) - set(XE.OVERLAP)   # review 16/11: MAV identities that ARE FLAG train people (xj_ext.OVERLAP) never enter IB-LoRA
V, Fc = V[V.k.isin(both)], Fc[Fc.k.isin(both)]
G = XE.gender_map(); L = pd.concat([Fc, V], ignore_index=True); L["gender"] = L.k.map(G).fillna("u")
nf, nv = Fc.k.value_counts(), V.k.value_counts(); rng = np.random.default_rng(2027)
cand = [k for k in sorted(both) if G.get(k) in ("m", "f") and nf[k] >= 4 and nv[k] >= 4]
val = []
for pre, n in (("voxceleb1/", 20), ("mavceleb_", 5)):
    for g in ("m", "f"):
        c = [k for k in cand if k.startswith(pre) and G[k] == g]; val += list(rng.choice(c, n, replace=False))
L["split"] = np.where(L.k.isin(val), "drop", "train")
for k in val:   # 4 faces + 4 voices of each val id
    for kind in ("face", "voice"):
        ix = L.index[(L.k == k) & (L.kind == kind)]; L.loc[rng.choice(ix, 4, replace=False), "split"] = "val"
L = L[L.split.ne("drop")].reset_index(drop=True)
ctr = pd.concat([L[(L.split == "train") & (L.kind == kind)].sample(256, random_state=7) for kind in ("face", "voice")]).assign(split="ctr")
L = pd.concat([L, ctr], ignore_index=True)
vf, vv = L[(L.split == "val") & (L.kind == "face")], L[(L.split == "val") & (L.kind == "voice")]
REF = dict(face=IF[vf.ref.values[:32]].astype(np.float16), voice=IV[vv.ref.values[:32]].astype(np.float16))   # parity on the first 32 val items (notebook size)
assert not L.k.isin(set(XE.OVERLAP)).any() and not L[L.split.eq("train")].k.isin(val).any() and L.k.str.match(r"^(mavceleb_v[12]|voxceleb1)/id\d+$").all()
tr = L[L.split.eq("train")]
SUMMARY = (f"train ids {tr.k.nunique()} (mav {tr.k.str.startswith('mavceleb').groupby(tr.k).first().sum()}), faces {int((tr.kind == 'face').sum())}, "
           f"voices {int((tr.kind == 'voice').sum())} | genders {tr.drop_duplicates('k').gender.value_counts().to_dict()} | val ids {len(val)} "
           f"faces {len(vf)} voices {len(vv)} | MAV voices dropped by LID/ok {int((~mv).sum())}/{len(vr)}, vox1 voices dropped {int((~xv).sum())}/{len(vox)}")

SRC = {n: (SC / n).read_bytes() for n in ("t3_iblora.py", "t3_imagebind_extract.py")}
MODF = {f: sha(IBD / f"ImageBind-{COMMIT}/imagebind/models/{f}.py") for f in ("imagebind_model", "helpers", "multimodal_preprocessors", "transformer")}

CELL = r'''
import os, sys, subprocess, time, shutil, zipfile, json, glob, traceback, hashlib, gzip, base64, io
from pathlib import Path
import numpy as np, pandas as pd
T0 = time.time()
def log(m): print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 60:6.1f}m]", m, flush=True)
def run(c): print("+", c, flush=True); subprocess.run(c, shell=True, check=True)
def find_dirs(name, depth=4): return [p for d in range(1, depth + 1) for p in glob.glob("/kaggle/input/" + "/".join(["*"] * d)) if os.path.basename(p) == name]
def fsha(p, bs=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""): h.update(b)
    return h.hexdigest()
ub = lambda s: gzip.decompress(base64.b64decode(s))
W = Path("/kaggle/temp/iblora"); W.mkdir(parents=True, exist_ok=True); OUT = Path("/kaggle/working/iblora_out"); OUT.mkdir(parents=True, exist_ok=True)
ROOT = W / "root"; S = ROOT / "Track 3/scripts"; IBD = ROOT / "Track 3/caches/imagebind"
try:
    run("pip install -q ftfy iopath einops 2>&1 | tail -1")
    S.mkdir(parents=True, exist_ok=True)
    for n, (s, h) in json.loads(r"""__SRC__""").items():
        b = ub(s); assert hashlib.sha256(b).hexdigest() == h, n; (S / n).write_bytes(b)
    IBD.mkdir(parents=True, exist_ok=True); rp = IBD / "ImageBind-__COMMIT__"
    run(f"git clone -q https://github.com/facebookresearch/ImageBind.git '{rp}' && git -C '{rp}' checkout -q __COMMIT__")
    for f, h in json.loads(r"""__MODF__""").items(): assert fsha(rp / f"imagebind/models/{f}.py") == h, f"ImageBind {f}.py differs from our copy"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    from huggingface_hub import hf_hub_download
    ck = hf_hub_download("seung275/imagebind_huge.pth", "imagebind_huge.pth", cache_dir=str(W / "hf")); h = fsha(ck); log(f"checkpoint sha256 {h}")
    assert h == "__CKSHA__", "checkpoint differs"; os.symlink(os.path.realpath(ck), IBD / "imagebind_huge.pth")
    # ---- media paths (same resolution as the xj1ib extraction kernel)
    roots = {}
    for top in ("mavceleb_v1_complete-001", "mavceleb_v2_complete-002"):
        hh = [p for p in find_dirs(top) if os.path.isdir(p)]; assert hh, f"{top} not mounted"; roots[top] = Path(hh[0]).parent
    vb = [p for p in find_dirs("VoxCeleb_1_adapted", 4) if os.path.isdir(os.path.join(p, "face"))]; assert vb, f"VoxCeleb_1_adapted not mounted"
    VF = Path(vb[0]) / "face"; base = {}
    for slug in ("voxceleb1train", "voxceleb1test"):
        hh = [p for p in find_dirs(slug, 3) if os.path.isdir(os.path.join(p, "wav"))]; assert hh, f"{slug} not mounted"; base[slug] = Path(hh[0]) / "wav"
    def resolve(kind, src, rel):
        if src in ("v1", "v2"): return str(roots[rel.split("/")[0]] / rel)
        cands = [VF / s / rel for s in ("train", "test", "dev", "")] if kind == "face" else [b / rel for b in base.values()]
        for p in cands:
            if p.exists(): return str(p)
        return None
    import lzma; L = pd.read_csv(io.BytesIO(lzma.decompress(base64.b64decode("__ITEMS__"))), keep_default_na=False)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(32) as ex: P = list(ex.map(lambda r: resolve(*r), zip(L.kind, L.src, L.rel)))
    ok = np.array([p is not None and os.path.getsize(p) > 0 for p in P]); log(f"items {len(L)} resolved {int(ok.sum())} | by split/kind missing {L[~ok].groupby(['split', 'kind']).size().to_dict()}")
    assert ok.mean() > .995 and ok[L.split.isin(['val', 'ctr']).to_numpy()].all(), "missing media"
    L["path"] = P; L = L[ok].reset_index(drop=True); L.to_csv(W / "items.csv", index=False); L.drop(columns="path").to_csv(OUT / "items_rel.csv", index=False)
    z = json.loads(r"""__REF__"""); np.savez(W / "ref.npz", **{k: np.frombuffer(ub(v), np.float16).astype(np.float32).reshape(-1, 1024) for k, v in z.items()})
    import torch; log(f"GPUs {torch.cuda.device_count()}: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]} | setup done -> training")
    lf = OUT / "train.log"; p = subprocess.Popen([sys.executable, str(S / "t3_iblora.py"), "train", str(W / "items.csv"), str(OUT), "--ref", str(W / "ref.npz"), "--steps", "__STEPS__"],
                                               stdout=open(lf, "w"), stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUNBUFFERED="1"), cwd=str(S))
    seen, last_sz, last_t = 0, -1, time.time()
    while p.poll() is None:
        time.sleep(30)
        sz = os.path.getsize(lf)
        if sz != last_sz: last_sz, last_t = sz, time.time()
        elif time.time() - last_t > 1200: p.kill(); p.wait(); raise RuntimeError('train.log silent for 20 min -> killed (stall watchdog)')
        L_ = [l for l in open(lf).read().splitlines() if l.strip() and "Warning" not in l]
        for l in L_[seen:]:
            if "VAL" in l or "PARITY" in l or "step " in l and (" 0 loss" in l or "00 loss" in l) or "LoRA" in l or "train ids" in l or "Error" in l: print("   ", l[:300], flush=True)
        seen = len(L_)
    txt = open(lf).read(); print("===== train.log tail\n" + "\n".join(txt.splitlines()[-30:]), flush=True); assert p.returncode == 0, f"training failed rc {p.returncode}"
    json.dump(dict(system="IB-LoRA (prereg PREREG_gate_all.txt 16/11 20:10 sub118a)", data="XJ external pool (MAV xjc set + VoxCeleb1 rows_vox), clip-level Bengali LID rule, no FLAG people",
                   imagebind=f"facebookresearch/ImageBind@__COMMIT__ (model files sha == our copy), checkpoint HF seung275/imagebind_huge.pth sha256 __CKSHA__",
                   code={n: h for n, (_, h) in json.loads(r"""__SRC__""").items()}, summary="__SUMMARY__", labels_used="external identity labels only (training)",
                   test_statistics=False), open(OUT / "PROVENANCE.json", "w"), indent=1)
    with zipfile.ZipFile("/kaggle/working/iblora.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for q in sorted(OUT.iterdir()): zf.write(q, q.name)
    shutil.rmtree(OUT); log("JOB DONE")
except Exception as ex_:
    log(f"VMERR {ex_!r}"); traceback.print_exc(); raise
'''

if __name__ == "__main__":
    rep = {"__SRC__": json.dumps({n: (b64(b), hashlib.sha256(b).hexdigest()) for n, b in SRC.items()}), "__COMMIT__": COMMIT, "__MODF__": json.dumps(MODF),
           "__CKSHA__": CKPT_SHA, "__ITEMS__": base64.b64encode(__import__("lzma").compress(L.to_csv(index=False).encode(), preset=9)).decode(), "__REF__": json.dumps({k: b64(v.tobytes()) for k, v in REF.items()}),
           "__STEPS__": os.environ.get("IBL_STEPS", "1600"), "__SUMMARY__": SUMMARY.replace('"', "'")}
    src = CELL
    for a, b in rep.items(): assert a in src, a; src = src.replace(a, b)
    left = set(re.findall(r"__[A-Z0-9]+__", src)); assert not left, left
    compile(src, "cell", "exec")
    OUTD.mkdir(exist_ok=True)
    json.dump({"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}],
               "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5},
              open(OUTD / "iblora.ipynb", "w"))
    json.dump({"id": f"{USER}/flag-t3-iblora", "title": "FLAG T3 iblora", "code_file": "iblora.ipynb", "language": "python", "kernel_type": "notebook",
               "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
               "dataset_sources": ["KAGGLE_USER/external-data-mavceleb", "robot12138/voxceleb-1", "kryakrya/voxceleb1train", "kryakrya/voxceleb1test"],
               "competition_sources": [], "kernel_sources": []}, open(OUTD / "kernel-metadata.json", "w"), indent=1)
    L.to_csv(OUTD / "items_local.csv", index=False); np.savez(OUTD / "ref_local.npz", **REF)
    print(SUMMARY); print(f"built {OUTD} | notebook {(OUTD / 'iblora.ipynb').stat().st_size / 1e6:.2f} MB | user {USER}")
