"""HB + IB-LoRA face block (17/11, user "thử luôn xem ... thêm được một hệ thật mới"): Kaggle notebook trnlhvy/flag-t3-hbibl-extract.
IB-LoRA (ImageBind-huge + LoRA trained on the XJ ext pool = VoxCeleb1 + MAV, NO FLAG; lora_best.pt sha 60d816e4..., same file the Mac used for
ibl_flag_face / eval ibl_face) MERGED (scripts/t3_iblora.py load_merged) -> face embeddings of every HB pool face: VoxCeleb2 (vox2_faces.zip) and
VoxBlink (vb_media.zip img/), both from trncnguyn kernel outputs via temporary links made at build time. HB's ids (vox2 + VoxBlink) are DISJOINT from
IB-LoRA's training ids (VoxCeleb1 + MAV), so the block is not train-fitted on HB's own items (the sub118b failure mode).
Same extractor as the Mac (scripts/t3_imagebind_extract.py: cv2/libjpeg-turbo decode, 224 bicubic + centre crop, fp32, L2).
GATE (fail fast, worker 0, before the merged model): FROZEN ImageBind on 32 vox2 + 32 VoxBlink faces must reproduce the cached
caches/ext/{vox2,vb}_face_ibv rows (min cos > .99) = decode / preprocessing / file-mapping parity.
Output hbibl.zip: hbibl_face.f16.npy (N x 1024, L2), hbibl_face_names.csv (zip member names, same order), parity json, PROVENANCE.
usage: python build_hbibl_extract.py   (links from the default trncnguyn token) -> kaggle/hbibl_extract/ ; then push with the trnlhvy token at once."""
import os, json, hashlib, re, sys, gzip, base64
from pathlib import Path
import numpy as np, pandas as pd
K = Path(__file__).resolve().parent; sys.path.insert(0, str(K)); T3 = K.parent; CX = T3 / "caches/ext"
import build_xj1ib_extract as B   # COMMON helpers, ImageBind pins, extractor source
USER = "trnlhvy"; LSHA = hashlib.sha256((T3 / "xj/iblora/lora_best.pt").read_bytes()).hexdigest(); assert LSHA.startswith("60d816e4"), LSHA
SRC_IL = (B.SC / "t3_iblora.py").read_bytes()
COMMON = B.COMMON.replace('"/kaggle/temp/xj1ib"', '"/kaggle/temp/hbibl"').replace('"/kaggle/working/xj1ib_out"', '"/kaggle/working/hbibl_out"')
assert COMMON.count("hbibl") == 2
f16 = lambda a: B.b64(np.asarray(a, np.float16).tobytes())

# parity references: frozen ImageBind cached rows (first 32 usable rows of each source)
PAR = {}
for src, pref in (("vox2", "vox2/"), ("vb", None)):
    r = pd.read_csv(CX / f"rows_{src}.csv", keep_default_na=False); ok = np.load(CX / f"{src}_face_ibv.ok.npy"); E = np.load(CX / f"{src}_face_ibv.float32.npy", mmap_mode="r")
    idx = np.flatnonzero(ok)[:32]
    names = [pref + r.face[i] if pref else f"img/{r.clip_index[i]}_{r.face[i].rsplit('_', 1)[1]}" for i in idx]
    PAR[src] = dict(names=names, ref=f16(E[idx]))

WORKER = r'''
import os, sys, time, json
from pathlib import Path
import numpy as np, pandas as pd, torch
SH = int(sys.argv[1]); T0 = time.time()
def log(m): print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 60:6.1f}m]", f"shard{SH}", m, flush=True)
W = Path("/kaggle/temp/hbibl"); OUT = Path("/kaggle/working/hbibl_out"); sys.path.insert(0, str(W / "root/Track 3/scripts"))
L = pd.read_csv(W / "faces.csv"); L = L.iloc[SH::2]; paths = L.path.tolist()
import t3_imagebind_extract as T, t3_iblora as IL
dev = torch.device("cuda"); u = lambda E: E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
if SH == 0:   # parity gate with the FROZEN model on the cached ibv rows
    P = json.load(open(W / "par.json")); m0 = T.model(dev); res = {}
    for src, d in P.items():
        ref = np.frombuffer(__import__("gzip").decompress(__import__("base64").b64decode(d["ref"])), np.float16).reshape(-1, 1024).astype(np.float32)
        e = T.encode(m0, [str(W / "media" / n) for n in d["names"]], "face", dev, bs=16); c = (u(e) * u(ref)).sum(1); res[src] = dict(min=float(c.min()), mean=float(c.mean()))
    json.dump(res, open(OUT / "frozen_parity.json", "w"), indent=1); log(f"FROZEN PARITY vs cached ibv rows: {res}")
    assert all(v["min"] > .99 for v in res.values()), f"frozen parity failed {res}"
    del m0; torch.cuda.empty_cache(); (W / "parity_ok").touch()
else:
    t = time.time()
    while not (W / "parity_ok").exists():
        assert time.time() - t < 2400, "shard 0 parity never finished"; time.sleep(10)
m, C = IL.load_merged(str(W / "lora/lora_best.pt"), dev); torch.cuda.empty_cache()
log(f"merged IB-LoRA ready: step {C['step']} val {C['val']} ({torch.cuda.memory_allocated() / 2 ** 30:.1f} GB on GPU)")
t = time.time(); E = T.encode(m, paths, "face", dev, bs=32).astype(np.float32); log(f"encoded {len(E)} in {(time.time() - t) / 60:.1f} min")
assert np.isfinite(E).all() and (np.abs(E).sum(1) > 0).all()
np.save(OUT / f"shard{SH}.f16.npy", E.astype(np.float16)); log(f"SAVED shard {E.shape}")
'''

MAIN = r'''
try:
    import requests
    from concurrent.futures import ThreadPoolExecutor
    URLS = json.loads(r"""__URLS__""")
    for k, u_ in URLS.items():   # fail fast if the temporary links already expired (rebuild + re-push)
        r_ = requests.get(u_, headers={"Range": "bytes=0-3"}, timeout=120); log(f"link {k}: HTTP {r_.status_code} {r_.content[:2]!r}"); assert r_.status_code == 206 and r_.content[:2] == b"PK", f"link {k} dead"
    run("pip install -q ftfy iopath einops 2>&1 | tail -1")
    S.mkdir(parents=True, exist_ok=True)
    for n, (s_, h_) in json.loads(r"""__SRCS__""").items():
        b_ = ub(s_); assert hashlib.sha256(b_).hexdigest() == h_, n; (S / n).write_bytes(b_)
    IBD.mkdir(parents=True, exist_ok=True); (IBD / "pylib").mkdir(exist_ok=True); rp = IBD / "ImageBind-__COMMIT__"
    run(f"git clone -q https://github.com/facebookresearch/ImageBind.git '{rp}' && git -C '{rp}' checkout -q __COMMIT__")
    for f, h in json.loads(r"""__MODF__""").items(): assert fsha(rp / f"imagebind/models/{f}.py") == h, f"ImageBind {f}.py differs from our copy"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    from huggingface_hub import hf_hub_download
    ck = hf_hub_download("seung275/imagebind_huge.pth", "imagebind_huge.pth", cache_dir=str(W / "hf")); h = fsha(ck); log(f"checkpoint sha256 {h}")
    assert h == "__CKSHA__", "checkpoint differs"; os.symlink(os.path.realpath(ck), IBD / "imagebind_huge.pth")
    # ---- IB-LoRA checkpoint (training kernel output trnlhvy/flag-t3-iblora), sha == the file used on the Mac
    NEED = ("lora_best.pt",); zz, dd, seen = [], [], []
    for top in sorted(os.listdir("/kaggle/input")):
        if top == "datasets": continue
        r0 = os.path.join("/kaggle/input", top)
        for r, ds, fs in os.walk(r0, followlinks=True):
            if r[len(r0):].count(os.sep) >= 6: ds[:] = []
            seen.append(r)
            if "iblora.zip" in fs: zz.append(os.path.join(r, "iblora.zip"))
            if all(n in fs for n in NEED): dd.append(r)
    log(f"lora search: zips {zz} dirs {dd}"); assert zz or dd, f"iblora.zip not mounted; walked {seen[:60]}"
    (W / "lora").mkdir(exist_ok=True)
    if zz:
        with zipfile.ZipFile(zz[0]) as z: z.extract("lora_best.pt", W / "lora")
    else: shutil.copy(os.path.join(dd[0], "lora_best.pt"), W / "lora/lora_best.pt")
    lh = fsha(W / "lora/lora_best.pt"); log(f"lora_best.pt sha256 {lh}"); assert lh == "__LSHA__", "LoRA checkpoint != the one used on the Mac"
    # ---- media (parallel range download, faces only)
    DL = Path("/kaggle/temp/dl"); DL.mkdir(exist_ok=True); MED = W / "media"; MED.mkdir(exist_ok=True)
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
                except Exception: time.sleep(5 * (k + 1))
            raise RuntimeError(f"range {a}-{b} failed")
        t = time.time()
        with ThreadPoolExecutor(n) as ex: got = sum(ex.map(part, range(0, total, chunk)))
        log(f"downloaded {out.name}: {got / 1e9:.2f} GB in {time.time() - t:.0f} s"); assert got == total
    names = []
    for k in ("vox2_faces.zip", "vb_media.zip"):
        fetch(URLS[k], DL / k)
        with zipfile.ZipFile(DL / k) as z:
            mem = [m for m in z.namelist() if m.endswith(".jpg")]; z.extractall(MED, mem); names += mem; log(f"{k}: {len(mem)} faces extracted")
        (DL / k).unlink()
    import cv2
    good = np.array([cv2.imread(str(MED / n)) is not None for n in names]); log(f"faces {len(names)} decodable {int(good.sum())}"); assert good.mean() > .995
    sub = str(MED / names[int(np.argmax(good))])
    pd.DataFrame(dict(path=[str(MED / n) if g else sub for n, g in zip(names, good)], good=good)).to_csv(W / "faces.csv", index=False)
    pd.DataFrame(dict(name=names, good=good)).to_csv(OUT / "hbibl_face_names.csv", index=False)
    json.dump(json.loads(r"""__PAR__"""), open(W / "par.json", "w"))
    open(W / "worker.py", "w").write(r"""__WORKER__""")
    for i in (0, 1): open(W / f"w{i}.py", "w").write(f"import sys; sys.argv = ['w', '{i}']; exec(open('/kaggle/temp/hbibl/worker.py').read())")
    log("setup done -> workers"); run_workers([("w0", W / "w0.py", 0), ("w1", W / "w1.py", 1)])
    E = np.empty((len(names), 1024), np.float16); E[0::2] = np.load(OUT / "shard0.f16.npy"); E[1::2] = np.load(OUT / "shard1.f16.npy")
    np.save(OUT / "hbibl_face.f16.npy", E); (OUT / "shard0.f16.npy").unlink(); (OUT / "shard1.f16.npy").unlink()
    json.dump(dict(pair_local=True, labels_used=False, test_statistics=False, lora_sha256="__LSHA__", n_faces=len(names),
                   imagebind=f"facebookresearch/ImageBind@__COMMIT__, checkpoint HF seung275/imagebind_huge.pth sha256 __CKSHA__ + IB-LoRA merged (scripts/t3_iblora.py load_merged)",
                   extractor="scripts/t3_imagebind_extract.py (fp32, turbo decode)", sources=json.loads(r"""__SHAS__"""),
                   frozen_parity=json.load(open(OUT / "frozen_parity.json"))), open(OUT / "PROVENANCE.json", "w"), indent=1)
    with zipfile.ZipFile("/kaggle/working/hbibl.zip", "w", zipfile.ZIP_STORED) as z:
        for p in sorted(OUT.iterdir()): z.write(p, p.name)
    shutil.rmtree(OUT); log("JOB DONE")
except Exception as ex:
    log(f"VMERR {ex!r}"); traceback.print_exc(); raise
'''

if __name__ == "__main__":
    assert '"""' not in WORKER; compile(WORKER, "worker", "exec")
    src_ = open(K / "build_t3_sub26_kernel.py").read(); ns = {}; exec(src_[src_.index("def links():"):src_.index("CODE = r'''")], ns)
    urls = {k: v for k, v in ns["links"]().items() if k in ("vox2_faces.zip", "vb_media.zip")}; assert len(urls) == 2
    srcs = {"t3_imagebind_extract.py": B.SRC, "t3_iblora.py": SRC_IL}
    rep = {"__URLS__": json.dumps(urls), "__SRCS__": json.dumps({n: [B.b64(b), hashlib.sha256(b).hexdigest()] for n, b in srcs.items()}), "__COMMIT__": B.COMMIT,
           "__MODF__": json.dumps(B.MODF), "__CKSHA__": B.CKPT_SHA, "__LSHA__": LSHA, "__PAR__": json.dumps(PAR), "__WORKER__": WORKER,
           "__SHAS__": json.dumps({n: hashlib.sha256(b).hexdigest() for n, b in srcs.items()})}
    src = MAIN
    for a, b in rep.items(): assert a in src, a; src = src.replace(a, b)
    left = set(re.findall(r"__[A-Z0-9]+__", src)); assert not left, left
    cells = [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s} for s in (COMMON, src)]
    for c in cells: compile(c["source"], "cell", "exec")
    d = K / "hbibl_extract"; d.mkdir(exist_ok=True)
    json.dump({"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5},
              open(d / "hbibl_extract.ipynb", "w"))
    json.dump({"id": f"{USER}/flag-t3-hbibl-extract", "title": "FLAG T3 hbibl extract", "code_file": "hbibl_extract.ipynb", "language": "python", "kernel_type": "notebook",
               "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
               "dataset_sources": [], "competition_sources": [], "kernel_sources": [f"{USER}/flag-t3-iblora"]}, open(d / "kernel-metadata.json", "w"), indent=1)
    (d / "worker.py").write_text(WORKER)
    print(f"built {d} | notebook {(d / 'hbibl_extract.ipynb').stat().st_size / 1e6:.2f} MB | lora sha {LSHA[:16]} | parity rows {[len(v['names']) for v in PAR.values()]}")
