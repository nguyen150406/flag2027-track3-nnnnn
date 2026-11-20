"""Kaggle notebook caohuhoan/flag-t3-xjb-extract (14/11, user "Oke làm thử đi xem sao" = XJ-B, prereg sub99a): features for XJ-B.
 - ArcFace (InsightFace buffalo_l: SCRFD det_10g det_size 320 thresh .5, largest box; retry on a 25%-zero-padded copy; last resort plain resize 112;
   5-point norm_crop 112; w600k_r50; L2, 512) on (a) 128 FLAG train faces = PARITY GATE vs a reference computed by THIS file locally on CPU with the
   same code (mean cos > .98, else stop), (b) all 6,485 FLAG train faces, (c) the 2,326 eval faces (caohuhoan/flag-t3-evalmedia), (d) the teammate's
   MAV-Celeb v1+v2 face images (xjc_rows, 9,520), (e) the VoxCeleb1 rows_vox faces (fresh link to trncnguyn/flag-t3-vox1faces output).
 - ReDimNet b6 ft_lm vox2 (scripts/t3_extract_wavlmsv.py --name rdn recipe) on the xjc_rows MAV voices (8,779); parity gate on 128 FLAG train voices
   vs the cached FLAG rdn rows (mean cos > .98).
Every file is processed on its own (pair-local); no labels; nothing is trained. Rows / parity data are embedded (gzip+base64).
usage: python build_xjb_extract.py   (then KAGGLE_API_TOKEN="$(< ~/.kaggle_acc5/access_token)" kaggle kernels push -p xjb_extract)"""
import json, gzip, base64, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd
K = Path(__file__).resolve().parent; T3 = K.parent; ROOT = T3.parent
sys.path.insert(0, str(T3 / "scripts")); import t3data as D
b64 = lambda b: base64.b64encode(gzip.compress(b, 9)).decode()
DET_SHA, REC_SHA = "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91", "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"
RDN_SHA = "e0f3034320c9cceaa8ed7c917a36e0affb501a922631d55c1007660eb1bbedea"
# ---- the ArcFace recipe: ONE source string, executed locally (parity reference) and in the notebook
ARC = r'''
import numpy as np, cv2, onnxruntime
from ifx.retinaface import RetinaFace          # = insightface 0.7.3 model_zoo/retinaface.py (what model_zoo.get_model returns for det_10g)
from ifx.arcface_onnx import ArcFaceONNX       # = insightface 0.7.3 model_zoo/arcface_onnx.py
from ifx import face_align                     # = insightface 0.7.3 utils/face_align.py
def arc_models(det_path, rec_path, providers):
    det = RetinaFace(model_file=str(det_path), session=onnxruntime.InferenceSession(str(det_path), providers=providers)); det.prepare(ctx_id=0, input_size=(320, 320), det_thresh=0.5)
    rec = ArcFaceONNX(model_file=str(rec_path), session=onnxruntime.InferenceSession(str(rec_path), providers=providers)); rec.prepare(ctx_id=0)
    return det, rec
def arc_one(det, rec, img):   # img = BGR uint8; returns (unit 512 vector, status 1 direct / 2 padded / 0 resize fallback)
    for pad in (0, 1):
        x = img if not pad else cv2.copyMakeBorder(img, *(4 * [int(.25 * max(img.shape[:2]))]), cv2.BORDER_CONSTANT, value=0)
        b, k = det.detect(x, max_num=0, metric="default")
        if b is not None and len(b):
            i = int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))
            f = rec.get_feat(face_align.norm_crop(x, landmark=k[i], image_size=112))[0]; return f / np.linalg.norm(f), 1 + pad
    f = rec.get_feat(cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR))[0]; return f / np.linalg.norm(f), 0
'''
import insightface, tempfile
IFP = Path(insightface.__file__).parent; assert insightface.__version__ == "0.7.3"
VEND = {"retinaface.py": (IFP / "model_zoo/retinaface.py").read_text(), "face_align.py": (IFP / "utils/face_align.py").read_text(),
        "arcface_onnx.py": (IFP / "model_zoo/arcface_onnx.py").read_text().replace("from ..utils import face_align", "from . import face_align"), "__init__.py": ""}
assert "from . import face_align" in VEND["arcface_onnx.py"] and ".." not in "".join(l for l in VEND["retinaface.py"].splitlines() if l.startswith(("from", "import")))
_vd = Path(tempfile.mkdtemp()); (_vd / "ifx").mkdir()
for n, t in VEND.items(): (_vd / "ifx" / n).write_text(t)
sys.path.insert(0, str(_vd))
meta = pd.read_csv(ROOT / "flag_raw_s01_v3_adaface_simam100/metadata.csv").sort_values("row_id"); tr = meta[meta.partition.eq("train")]
assert len(tr) == 6485 and (tr.row_id.values == np.arange(6485)).all()
j = np.sort(np.random.default_rng(0).choice(len(tr), 128, replace=False))
IM = Path.home() / ".insightface/models/buffalo_l"
assert hashlib.sha256((IM / "det_10g.onnx").read_bytes()).hexdigest() == DET_SHA and hashlib.sha256((IM / "w600k_r50.onnx").read_bytes()).hexdigest() == REC_SHA
ns = {}; exec(ARC, ns); import cv2
det, rec = ns["arc_models"](IM / "det_10g.onnx", IM / "w600k_r50.onnx", ["CPUExecutionProvider"])
ref = np.stack([ns["arc_one"](det, rec, cv2.imread(str(ROOT / "data/train_set/train_set" / r)))[0] for r in tr.face_relpath.values[j]]).astype(np.float32)
vref = np.asarray(D.voice_full("rdn")[:6485], np.float32)[j]
par = dict(face_rel=tr.face_relpath.values[j].tolist(), voice_rel=tr.voice_relpath.values[j].tolist(), face_ref=b64(ref.astype(np.float16).tobytes()), voice_ref=b64(np.ascontiguousarray(vref).astype(np.float16).tobytes()))   # fp16: < 1e-3 cos error, keeps the notebook small
rows = (K / "xjc_rows/xjc_rows.csv").read_bytes(); vox = pd.read_csv(T3 / "caches/ext/rows_vox.csv")[["face"]].to_csv(index=False).encode()   # only the face column is needed
ev = pd.read_csv(T3 / "outputs/eval_final/rows.csv"); erel = [p.split("/eval_final/root/")[1] for p in ev.face]; assert len(erel) == 2326
frel = tr.face_relpath.tolist()
src = (K / "build_t3_sub26_kernel.py").read_text(); lk = {}; exec(src[src.index("def links():"):src.index("CODE = r'''")], lk)
fetch_src = src[src.index("def fetch(url, out"):src.index("\ndef tree(")]; assert "log(f\"downloaded" in fetch_src
cells = []
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s})
code(r'''
import os, sys, subprocess, time, shutil, zipfile, json, glob, traceback, gzip, base64, io, math, hashlib, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
T0 = time.time()
def log(m): print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 60:6.1f}m]", m, flush=True)
def run(c): print("+", c, flush=True); subprocess.run(c, shell=True, check=True)
def find_dirs(name, depth=4):   # depth-limited: never walk the media trees
    return [p for d in range(1, depth + 1) for p in glob.glob("/kaggle/input/" + "/".join(["*"] * d)) if os.path.basename(p) == name]
ub = lambda s: gzip.decompress(base64.b64decode(s))
ROWS = pd.read_csv(io.BytesIO(ub("__ROWS__"))); VOX = pd.read_csv(io.BytesIO(ub("__VOX__"))); PAR = json.loads(r"""__PAR__"""); EREL = json.loads(ub("__EREL__"))
FREL = json.loads(ub("__FREL__")); VOX1_URL = r"""__VOX1URL__"""
W = Path("/kaggle/temp/xjb"); W.mkdir(parents=True, exist_ok=True); OUT = Path("/kaggle/working/xjb_out"); OUT.mkdir(parents=True, exist_ok=True)
__FETCH__
log(f"rows {len(ROWS)} vox {len(VOX)} eval faces {len(EREL)} flag faces {len(FREL)}")
''')
code(r'''
try:
    # ---------------- paths
    roots = {}
    for top in ("mavceleb_v1_complete-001", "mavceleb_v2_complete-002"):
        h = [p for p in find_dirs(top) if os.path.isdir(p)]; assert h, f"{top} not mounted"; roots[top] = Path(h[0]).parent
    ROWS["path"] = [str(roots[r.split("/")[0]] / r) for r in ROWS.rel]; miss = [p for p in ROWS.path if not os.path.exists(p)]; assert not miss, miss[:5]
    def inside(base_name, leaf, ok, depth=5):   # search only inside one mounted dataset (never the MAV media trees)
        bases = [p for p in find_dirs(base_name, 4) if os.path.isdir(p)]; assert bases, f"{base_name} not mounted: {os.listdir('/kaggle/input')}"
        hits = [Path(q) for b in bases for d in range(1, depth + 1) for q in glob.glob(b + "/" + "/".join(["*"] * d)) if os.path.basename(q) == leaf and ok(q)]
        assert hits, f"{leaf} not found in {bases}"; return hits[0]
    FLAGTR = inside("flag-2027-data", "train_set", lambda q: os.path.isdir(os.path.join(q, "faces")))
    EROOT = inside("flag-t3-evalmedia", "root", lambda q: os.path.isdir(os.path.join(q, "no_gender")), 3)
    pflag = [FLAGTR / r for r in FREL]; pev = [EROOT / r for r in EREL]; pvp = [FLAGTR / r for r in PAR["voice_rel"]]
    assert all(p.exists() for p in pflag) and all(p.exists() for p in pev) and all(p.exists() for p in pvp), "FLAG / eval media missing"
    log(f"paths ok: MAV {roots} FLAG {FLAGTR} eval {EROOT}")
    fetch(VOX1_URL, W / "vox1_faces.zip")
    with zipfile.ZipFile(W / "vox1_faces.zip") as z: z.extractall(W / "vox1_faces")
    (W / "vox1_faces.zip").unlink()
    def vfind(rel):
        r = W / "vox1_faces"
        for p in (r / rel, r / "vox" / rel, r / "train" / rel, r / "test" / rel, r / "dev" / rel):
            if p.exists(): return p
        return None
    uf = VOX.face.drop_duplicates().tolist(); pvox = [vfind(f) for f in uf]; log(f"vox1 faces found {sum(p is not None for p in pvox)}/{len(uf)}")
    assert sum(p is not None for p in pvox) > .99 * len(uf), "vox1 faces missing"
    # ---------------- ArcFace (buffalo_l, onnxruntime) — set up and parity-checked FIRST (fail fast)
    import torch
    (W / "ifx").mkdir(exist_ok=True)
    for n, t in json.loads(ub("__VEND__")).items(): (W / "ifx" / n).write_text(t)
    sys.path.insert(0, str(W)); open(W / "cons.txt", "w").write(f"numpy=={np.__version__}\n")
    run("pip uninstall -y -q onnxruntime onnxruntime-gpu || true"); run(f"pip install -q onnx onnxruntime-gpu -c {W}/cons.txt 2>&1 | tail -3")
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        try: ort.preload_dlls()
        except Exception as ex: log(f"preload_dlls: {ex!r}")
    run(f"cd {W} && wget -q https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip && unzip -q -o buffalo_l.zip -d buffalo_l")
    det_p, rec_p = (next((W / "buffalo_l").rglob(n)) for n in ("det_10g.onnx", "w600k_r50.onnx"))
    assert hashlib.sha256(det_p.read_bytes()).hexdigest() == "__DETSHA__" and hashlib.sha256(rec_p.read_bytes()).hexdigest() == "__RECSHA__", "buffalo_l files differ"
    ns = {}; exec(r"""__ARC__""", ns)
    det, rec = ns["arc_models"](det_p, rec_p, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    prov = [det.session.get_providers()[0], rec.session.get_providers()[0]]; log(f"STAGE arcface ready, providers {prov} available {ort.get_available_providers()}")
    import cv2
    def fencode(paths, tag):
        E, S = np.zeros((len(paths), 512), np.float32), np.full(len(paths), 255, np.uint8)
        rd = lambda p: cv2.imread(str(p)) if p is not None else None
        with ThreadPoolExecutor(8) as ex:
            for i, img in enumerate(ex.map(rd, paths)):
                if img is None: continue
                E[i], S[i] = ns["arc_one"](det, rec, img)
                if i % 3000 == 0: log(f"  arc {tag} {i}/{len(paths)}")
        return E, S
    e, st = fencode([FLAGTR / r for r in PAR["face_rel"]], "parity"); ref = np.frombuffer(ub(PAR["face_ref"]), np.float16).astype(np.float32).reshape(-1, 512); ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    cs = (e * ref).sum(1); pf = dict(mean=float(cs.mean()), min=float(cs.min()), p05=float(np.percentile(cs, 5)), direct=float((st == 1).mean()), providers=prov)
    json.dump(pf, open(OUT / "arc_parity.json", "w"), indent=1); log(f"PARITY arcface vs local CPU reference: {pf}"); assert cs.mean() > .98, f"ArcFace differs from the local reference: {pf}"
    # ---------------- ReDimNet (t3_extract_wavlmsv rdn recipe)
    import torch, torch.nn.functional as F, torchaudio, wave
    assert torch.cuda.is_available(); dev = torch.device("cuda:0"); log(f"gpu {torch.cuda.get_device_name(0)}")
    rdn = torch.hub.load("IDRnD/ReDimNet", "ReDimNet", model_name="b6", train_type="ft_lm", dataset="vox2", trust_repo=True).to(dev).eval()
    ck = Path(torch.hub.get_dir()) / "checkpoints/b6-vox2-ft_lm.pt"; hsh = hashlib.sha256(ck.read_bytes()).hexdigest()
    hub = [p for p in Path(torch.hub.get_dir()).glob("IDRnD_ReDimNet_*/hubconf.py")]; hubsha = hashlib.sha256(hub[0].read_bytes()).hexdigest() if hub else "?"
    log(f"rdn weights sha {hsh[:16]} hubconf sha {hubsha[:16]}"); assert hsh == "__RDNSHA__", "ReDimNet weights differ from the cached FLAG/eval/vox1 extraction"
    def read(p):   # float waveform in [-1, 1] at 16 kHz (rdn recipe; multi-channel -> channel mean)
        with wave.open(str(p)) as w:
            assert w.getsampwidth() == 2, p; sr, ch = w.getframerate(), w.getnchannels(); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        x = x.reshape(-1, ch).mean(1) if ch > 1 else x
        return torchaudio.functional.resample(torch.from_numpy(x)[None], sr, 16000)[0].numpy() if sr != 16000 else x
    def prep(p):
        x = read(p); c = [x[s:s + 30 * 16000] for s in range(0, len(x), 30 * 16000)]; c = [q for q in c if len(q) >= 16000] or [x]
        return [torch.from_numpy(q)[None] for q in c]
    pool = ThreadPoolExecutor(8)
    @torch.no_grad()
    def vencode(paths, tag, block=256):
        out = np.zeros((len(paths), 192), np.float32); ok = np.zeros(len(paths), bool)
        def safe(p):
            try: return prep(p)
            except Exception as ex: log(f"voice read fail {p}: {ex!r}"); return None
        for s in range(0, len(paths), block):
            for i, xs in enumerate(pool.map(safe, paths[s:s + block])):
                if xs is None: continue
                with torch.autocast("cuda", dtype=torch.float16):
                    e = torch.stack([rdn(x.to(dev))[0].float() for x in xs]).mean(0)
                out[s + i] = F.normalize(e, dim=0).cpu().numpy(); ok[s + i] = np.isfinite(out[s + i]).all()
            if (s // block) % 8 == 0: log(f"  rdn {tag} {min(s + block, len(paths))}/{len(paths)}")
        return out, ok
    e, _ = vencode(pvp, "parity"); ref = np.frombuffer(ub(PAR["voice_ref"]), np.float16).astype(np.float32).reshape(-1, 192); ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    cs = (e * ref).sum(1); pv = dict(mean=float(cs.mean()), min=float(cs.min()), p05=float(np.percentile(cs, 5)))
    json.dump(pv, open(OUT / "rdn_parity.json", "w"), indent=1); log(f"PARITY rdn vs cached FLAG rdn: {pv}"); assert cs.mean() > .98, f"ReDimNet does not reproduce the FLAG cache: {pv}"
    vp = ROWS[ROWS.kind.eq("voice")].path.tolist(); V, okv = vencode(vp, "mav")
    np.save(OUT / "xjb_voice_rdn.float32.npy", V); np.save(OUT / "xjb_voice_rdn.ok.npy", okv); log(f"STAGE mav voices done ok {okv.sum()}/{len(vp)}")
    del rdn; torch.cuda.empty_cache()
    # ---------------- ArcFace bulk
    for tag, paths, name in (("flag", pflag, "flag_face_arc"), ("eval", pev, "eval_face_arc")):
        E, S = fencode(paths, tag); assert (S != 255).all(); np.save(OUT / f"{name}.float32.npy", E); np.save(OUT / f"{name}.align.npy", S)
        log(f"STAGE {tag} faces done {E.shape} direct {(S == 1).mean():.3f} padded {(S == 2).mean():.3f} fallback {(S == 0).mean():.3f}")
    fp = ROWS[ROWS.kind.eq("face")].path.tolist(); E, S = fencode(fp, "mav")
    np.save(OUT / "xjb_face_arc.float32.npy", E); np.save(OUT / "xjb_face_arc.ok.npy", S != 255); np.save(OUT / "xjb_face_arc.align.npy", S)
    log(f"STAGE mav faces done {E.shape} direct {(S == 1).mean():.3f} fallback {(S == 0).mean():.3f} unread {(S == 255).sum()}")
    E, S = fencode(pvox, "vox1"); m = dict(zip(uf, range(len(uf)))); ix = VOX.face.map(m).to_numpy()
    np.save(OUT / "vox_face_arc.float32.npy", E[ix]); np.save(OUT / "vox_face_arc.ok.npy", (S != 255)[ix]); np.save(OUT / "vox_face_arc.align.npy", S[ix])
    log(f"STAGE vox1 faces done {len(uf)} unique -> {len(ix)} rows, direct {(S == 1).mean():.3f} fallback {(S == 0).mean():.3f} unread {(S == 255).sum()}")
    ROWS.drop(columns=["path"]).to_csv(OUT / "xjb_rows.csv", index=False); VOX.to_csv(OUT / "vox_faces_order.csv", index=False)
    json.dump(dict(pair_local=True, frozen_encoders=True, labels_used=False, test_statistics=False,
                   arcface=f"InsightFace buffalo_l det_10g sha {'__DETSHA__'[:16]} + w600k_r50 sha {'__RECSHA__'[:16]}, insightface 0.7.3, det 320", arc_parity=pf,
                   redimnet=f"torch.hub IDRnD/ReDimNet b6 ft_lm vox2, weights sha {hsh[:16]}, hubconf sha {hubsha[:16]}, fp16 autocast", rdn_parity=pv), open(OUT / "PROVENANCE.json", "w"), indent=1)
    with zipfile.ZipFile("/kaggle/working/xjb_embeddings.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()): z.write(p, p.name)
    shutil.rmtree(OUT); log("JOB DONE")
except Exception as ex:
    log(f"VMERR {ex!r}"); traceback.print_exc(); raise
''')
s0 = cells[0]["source"]
for a, b in (("__ROWS__", b64(rows)), ("__VOX__", b64(vox)), ("__PAR__", json.dumps(par)), ("__EREL__", b64(json.dumps(erel).encode())), ("__FREL__", b64(json.dumps(frel).encode())),
             ("__VOX1URL__", lk["links"]()["vox1_faces.zip"]), ("__FETCH__", fetch_src)):
    assert s0.count(a) == 1, a; s0 = s0.replace(a, b)
cells[0]["source"] = s0
s1 = cells[1]["source"]
for a, b in (("__ARC__", ARC), ("__VEND__", b64(json.dumps(VEND).encode())), ("__DETSHA__", DET_SHA), ("__RECSHA__", REC_SHA), ("__RDNSHA__", RDN_SHA)): assert a in s1, a; s1 = s1.replace(a, b)
cells[1]["source"] = s1
for c in cells: compile(c["source"], "cell", "exec")
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
d = K / "xjb_extract"; d.mkdir(exist_ok=True); json.dump(nb, open(d / "xjb_extract.ipynb", "w"))
json.dump({"id": "caohuhoan/flag-t3-xjb-extract", "title": "FLAG T3 xjb extract", "code_file": "xjb_extract.ipynb", "language": "python", "kernel_type": "notebook",
           "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
           "dataset_sources": ["KAGGLE_USER/external-data-mavceleb", "trncnguyn/flag-2027-data", "caohuhoan/flag-t3-evalmedia"],
           "competition_sources": [], "kernel_sources": []}, open(d / "kernel-metadata.json", "w"), indent=1)
np.save(K / "xjb_extract/local_arc_ref.npy", ref)   # local copy of the parity reference (not uploaded: only the notebook is pushed)
print("built", d, "notebook bytes", (d / "xjb_extract.ipynb").stat().st_size)
