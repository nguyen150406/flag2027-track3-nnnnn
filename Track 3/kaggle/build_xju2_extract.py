"""Two Kaggle notebooks for XJ-U2 (14/11 20:55 prereg sub100a, user "tiếp tục cải thiện cái kia" = hướng 1: XJ-U with the XJ-SA external pool).
  faces  (ncamtu06/flag-t3-xju2-faces):  AdaFace IR-101 (xja recipe, GPU0) + ArcFace buffalo_l (xjb recipe, GPU1) on the teammate's VoxCeleb1 REAL face
          images (43,523; the 6,517 voice_spectrogram PNGs of ext_feats are excluded) + the MAVX face rows (2,115).
  voices (trnlhvy/flag-t3-xju2-voices):  WeSpeaker CAM++-LM (xja recipe, GPU0) + ReDimNet b6 ft_lm vox2 (xjb recipe, GPU1) on the teammate's VoxCeleb1
          voice clips (50,040) + the MAVX voice rows (4,761).
Recipe code is copied from build_xja_extract.py / build_xjb_extract.py. Gates (bug checks): every encoder vs our own caches on 128 items shared with
rows_vox (same file): AdaFace vs caches/ext/vox_face_ada, ArcFace vs xj/xjb/vox_face_arc, CAM++ vs caches/ext/vox_voice_campplus, ReDimNet vs
caches/ext/vox_voice_rdn; mean cos > .98 else the worker stops. Each file is encoded on its own (pair-local); no labels; nothing is trained.
usage: python build_xju2_extract.py  -> xju2_faces/, xju2_voices/  (then kaggle kernels push -p <dir> with the account's token)"""
import json, gzip, base64, sys, hashlib, tempfile
from pathlib import Path
import numpy as np, pandas as pd
K = Path(__file__).resolve().parent; T3 = K.parent; ROOT = T3.parent; XJ = T3 / "xj"
b64 = lambda b: base64.b64encode(gzip.compress(b, 9)).decode()
f16 = lambda a: b64(np.ascontiguousarray(np.asarray(a, np.float32)).astype(np.float16).tobytes())
DET_SHA, REC_SHA = "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91", "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"
RDN_SHA = "e0f3034320c9cceaa8ed7c917a36e0affb501a922631d55c1007660eb1bbedea"
ADA_SHA = "0e7a3238d2a50f3fe3860782534928ac7cb2598977cf897f6869fd5ac2493fd0"

# ------------------------------------------------------------------ rows
fm, vm = pd.read_csv(XJ / "ext_feats/faces_meta.csv"), pd.read_csv(XJ / "ext_feats/voices_meta.csv")
fv, vv = fm[fm.source.eq("voxceleb1")], vm[vm.source.eq("voxceleb1")]
real = fv.path.str.contains("/VoxCeleb_1_adapted/face/"); assert real.sum() + fv.path.str.contains("/voice_spectrogram/").sum() == len(fv)
fv = fv[real]; assert len(fv) == 43523, len(fv)
F_ROWS = pd.concat([pd.DataFrame(dict(src="vox1", rel=fv.path.str.split("VoxCeleb_1_adapted/").str[1].values, xj_row=fv.index.values)),
                    *[pd.DataFrame(dict(src=f"mavx{sh}", rel=(c := pd.read_csv(XJ / f"mavx/mavx{sh}_face_rows.csv")).rel.values, xj_row=np.arange(len(c)))) for sh in (0, 1)]], ignore_index=True)
vparts = vv.path.str.split("/"); assert vparts.str[2].isin(["voxceleb1train", "voxceleb1test"]).all()
V_ROWS = pd.concat([pd.DataFrame(dict(src="vox1", rel=(vparts.str[2] + "/" + vparts.str[3:].str.join("/")).values, xj_row=vv.index.values)),
                    *[pd.DataFrame(dict(src=f"mavx{sh}", rel=(c := pd.read_csv(XJ / f"mavx/mavx{sh}_voice_rows.csv")).rel.values, xj_row=np.arange(len(c)))) for sh in (0, 1)]], ignore_index=True)
assert len(F_ROWS) == 43523 + 2115 and len(V_ROWS) == 50040 + 4761, (len(F_ROWS), len(V_ROWS))

# ------------------------------------------------------------------ parity items (shared with rows_vox) + references from our caches
r = pd.read_csv(T3 / "caches/ext/rows_vox.csv"); fo = pd.read_csv(XJ / "xjb/vox_faces_order.csv"); assert (fo.face.values == r.face.values).all()
fkey = F_ROWS.rel.where(F_ROWS.src.ne("vox1")).fillna(F_ROWS.rel.str.split("/").str[-3:].str.join("/"))
rf = dict(zip(r.face.values[::-1], np.arange(len(r))[::-1]))   # face -> its first rows_vox row (duplicate faces share one encoding)
fs = np.flatnonzero(F_ROWS.src.eq("vox1").to_numpy() & fkey.isin(rf).to_numpy()); assert len(fs) >= 1600, len(fs)
fs = np.sort(np.random.default_rng(0).choice(fs, 128, replace=False)); fri = np.array([rf[fkey[i]] for i in fs])
ada_ref, arc_ref = np.load(T3 / "caches/ext/vox_face_ada.float32.npy")[fri], np.load(XJ / "xjb/vox_face_arc.float32.npy")[fri]
assert np.load(T3 / "caches/ext/vox_face_ada.ok.npy")[fri].all() and np.load(XJ / "xjb/vox_face_arc.ok.npy")[fri].all()
vkey = V_ROWS.rel.str.split("/").str[-3:].str.join("/"); rv = {v: i for i, v in enumerate(r.voice)}
vs = np.flatnonzero(V_ROWS.src.eq("vox1").to_numpy() & vkey.isin(rv).to_numpy()); assert len(vs) >= 8000, len(vs)
vs = np.sort(np.random.default_rng(0).choice(vs, 128, replace=False)); vri = np.array([rv[vkey[i]] for i in vs])
cp_ref, rdn_ref = np.load(T3 / "caches/ext/vox_voice_campplus.float32.npy")[vri], np.load(T3 / "caches/ext/vox_voice_rdn.float32.npy")[vri]
assert np.load(T3 / "caches/ext/vox_voice_campplus.ok.npy")[vri].all() and np.load(T3 / "caches/ext/vox_voice_rdn.ok.npy")[vri].all()
FPAR = dict(idx=fs.tolist(), ada=f16(ada_ref), arc=f16(arc_ref)); VPAR = dict(idx=vs.tolist(), campplus=f16(cp_ref), rdn=f16(rdn_ref))

# ------------------------------------------------------------------ vendored insightface 0.7.3 (as build_xjb_extract.py)
import insightface
IFP = Path(insightface.__file__).parent; assert insightface.__version__ == "0.7.3"
VEND = {"retinaface.py": (IFP / "model_zoo/retinaface.py").read_text(), "face_align.py": (IFP / "utils/face_align.py").read_text(),
        "arcface_onnx.py": (IFP / "model_zoo/arcface_onnx.py").read_text().replace("from ..utils import face_align", "from . import face_align"), "__init__.py": ""}
assert "from . import face_align" in VEND["arcface_onnx.py"]
xjb_src = (K / "build_xjb_extract.py").read_text(); ARC = xjb_src[xjb_src.index("ARC = r'''") + 10:xjb_src.index("'''", xjb_src.index("ARC = r'''") + 10)]
assert "def arc_one" in ARC and "input_size=(320, 320)" in ARC

COMMON = r'''
import os, sys, subprocess, time, shutil, zipfile, json, glob, traceback, gzip, base64, io, math, hashlib, types, warnings
from pathlib import Path
import numpy as np, pandas as pd
T0 = time.time()
def log(m): print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 60:6.1f}m]", m, flush=True)
def run(c): print("+", c, flush=True); subprocess.run(c, shell=True, check=True)
def find_dirs(name, depth=4):   # depth-limited: never walk the media trees
    return [p for d in range(1, depth + 1) for p in glob.glob("/kaggle/input/" + "/".join(["*"] * d)) if os.path.basename(p) == name]
ub = lambda s: gzip.decompress(base64.b64decode(s))
W = Path("/kaggle/temp/xju2"); W.mkdir(parents=True, exist_ok=True); OUT = Path("/kaggle/working/xju2_out"); OUT.mkdir(parents=True, exist_ok=True)
def mav_roots():
    roots = {}
    for top in ("mavceleb_v1_complete-001", "mavceleb_v2_complete-002"):
        h = [p for p in find_dirs(top) if os.path.isdir(p)]; assert h, f"{top} not mounted"; roots[top] = Path(h[0]).parent
    return roots
def run_workers(jobs):
    """jobs = [(name, script_path, gpu)] -> run in parallel (one per GPU) when 2 GPUs exist, else one after another; stream the log tails."""
    import torch
    ng = torch.cuda.device_count(); log(f"GPUs {ng}: {[torch.cuda.get_device_name(i) for i in range(ng)]}")
    batches = [jobs] if ng >= 2 else [[j] for j in jobs]
    for b in batches:
        ps = {}
        for name, script, gpu in b:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu if ng >= 2 else 0), PYTHONUNBUFFERED="1")
            ps[name] = (subprocess.Popen([sys.executable, str(script)], stdout=open(W / f"{name}.log", "w"), stderr=subprocess.STDOUT, env=env), W / f"{name}.log")
        last = time.time()
        while any(p.poll() is None for p, _ in ps.values()):
            time.sleep(20)
            if time.time() - last > 300:
                last = time.time()
                for name, (p, lf) in ps.items():
                    tail = [l for l in open(lf).read().splitlines() if l.strip()][-1:] or ["(no output yet)"]; log(f"  [{name}] {tail[0][:200]}")
        for name, (p, lf) in ps.items():
            txt = open(lf).read(); print(f"===== {name} log (last 40 lines)\n" + "\n".join(txt.splitlines()[-40:]), flush=True)
            assert p.returncode == 0, f"worker {name} failed (rc {p.returncode})"
'''

WORKER_HEAD = r'''
import os, sys, time, json, gzip, base64, math, hashlib, types, warnings
from pathlib import Path
import numpy as np, pandas as pd
T0 = time.time()
def log(m): print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 60:6.1f}m]", m, flush=True)
ub = lambda s: gzip.decompress(base64.b64decode(s))
W = Path("/kaggle/temp/xju2"); OUT = Path("/kaggle/working/xju2_out")
PAR = json.load(open(W / "par.json")); paths = pd.read_csv(W / "paths.csv").path.tolist()
def gate(E, key, d, name):
    ref = np.frombuffer(ub(PAR[key]), np.float16).astype(np.float32).reshape(-1, d); ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    e = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12); cs = (e * ref).sum(1)
    pj = dict(mean=float(cs.mean()), min=float(cs.min()), p05=float(np.percentile(cs, 5)), n=int(len(cs)))
    json.dump(pj, open(OUT / f"{name}_parity.json", "w"), indent=1); log(f"PARITY {name} vs our cache on shared rows_vox items: {pj}")
    assert cs.mean() > .98, f"{name} does not reproduce our cache: {pj}"
    return pj
import torch, torch.nn.functional as F
assert torch.cuda.is_available(); dev = torch.device("cuda:0"); torch.set_num_threads(2); log(f"worker gpu {torch.cuda.get_device_name(0)}, {len(paths)} files")
'''

ADA = WORKER_HEAD + r'''
from PIL import Image
AR = W / "AdaFace"; sys.path.insert(0, str(AR)); import net as adaface_net
ckpt = W / "adaw/adaface_weights.ckpt"
face_model = adaface_net.build_model("ir_101"); ck = torch.load(ckpt, map_location="cpu", weights_only=False); sd = ck.get("state_dict", ck)
ms = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items() if k.startswith("model.") or k in face_model.state_dict()}
missing, unexpected = face_model.load_state_dict(ms, strict=False); assert not missing and not unexpected, (missing[:5], unexpected[:5]); face_model.eval().to(dev)
cwd = Path.cwd(); os.chdir(AR / "face_alignment")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=SyntaxWarning); from face_alignment import mtcnn as adaface_mtcnn
aligner = adaface_mtcnn.MTCNN(device="cuda:0", crop_size=(112, 112)); os.chdir(cwd)
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
def embed(ps, tag, bs=64):
    E, S = np.zeros((len(ps), 512), np.float32), np.full(len(ps), 255, np.uint8)
    for s in range(0, len(ps), bs):
        xs, ii = [], []
        for i, p in enumerate(ps[s:s + bs]):
            try: im, st = align(p)
            except Exception as ex: log(f"face read fail {p}: {ex!r}"); continue
            xs.append(tensor(im)); ii.append(s + i); S[s + i] = st
        if xs:
            f, _ = face_model(torch.stack(xs).to(dev)); E[ii] = F.normalize(f.float(), dim=1).cpu().numpy()
        if (s // bs) % 50 == 0: log(f"  ada {tag} {s + len(xs)}/{len(ps)}")
    return E, S
e, _ = embed([paths[i] for i in PAR["idx"]], "parity"); pj = gate(e, "ada", 512, "ada")
E, S = embed(paths, "all"); np.save(OUT / "xju2_face_ada.float32.npy", E); np.save(OUT / "xju2_face_ada.align.npy", S)
log(f"ADA DONE {E.shape} aligned {(S == 1).mean():.3f} fallback {(S == 0).mean():.3f} unread {(S == 255).sum()}")
'''

ARCW = WORKER_HEAD + r'''
import cv2
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(W))
import onnxruntime as ort
if hasattr(ort, "preload_dlls"):
    try: ort.preload_dlls()
    except Exception as ex: log(f"preload_dlls: {ex!r}")
ns = {}; exec(open(W / "arc_recipe.py").read(), ns)
det_p, rec_p = (next((W / "buffalo_l").rglob(n)) for n in ("det_10g.onnx", "w600k_r50.onnx"))
det, rec = ns["arc_models"](det_p, rec_p, ["CUDAExecutionProvider", "CPUExecutionProvider"])
prov = [det.session.get_providers()[0], rec.session.get_providers()[0]]; log(f"arcface providers {prov}"); assert prov[0] == "CUDAExecutionProvider", prov
def fencode(ps, tag):
    E, S = np.zeros((len(ps), 512), np.float32), np.full(len(ps), 255, np.uint8)
    with ThreadPoolExecutor(6) as ex:
        for i, img in enumerate(ex.map(lambda p: cv2.imread(str(p)), ps)):
            if img is None: continue
            E[i], S[i] = ns["arc_one"](det, rec, img)
            if i % 3000 == 0: log(f"  arc {tag} {i}/{len(ps)}")
    return E, S
e, _ = fencode([paths[i] for i in PAR["idx"]], "parity"); pj = gate(e, "arc", 512, "arc")
E, S = fencode(paths, "all"); np.save(OUT / "xju2_face_arc.float32.npy", E); np.save(OUT / "xju2_face_arc.align.npy", S)
log(f"ARC DONE {E.shape} direct {(S == 1).mean():.3f} padded {(S == 2).mean():.3f} fallback {(S == 0).mean():.3f} unread {(S == 255).sum()}")
'''

CPW = WORKER_HEAD + r'''
import yaml, torchaudio, wave
from huggingface_hub import hf_hub_download
root = W / "wespeaker/wespeaker"; pkg = types.ModuleType("wespeaker"); pkg.__path__ = [str(root)]; sys.modules["wespeaker"] = pkg
fe = types.ModuleType("wespeaker.frontend"); fe.__path__ = [str(root / "frontend")]; sys.modules["wespeaker.frontend"] = fe
from wespeaker.models.speaker_model import get_speaker_model
cfg = yaml.safe_load(open(hf_hub_download("Wespeaker/wespeaker-voxceleb-campplus-LM", "config.yaml"))); wts = hf_hub_download("Wespeaker/wespeaker-voxceleb-campplus-LM", "avg_model.pt")
model = get_speaker_model(cfg["model"])(**cfg["model_args"]); missing, _ = model.load_state_dict(torch.load(wts, map_location="cpu"), strict=False)
assert not [k for k in missing if "projection" not in k], missing; model.eval().to(dev)
def pcm16(path):
    with wave.open(str(path)) as w:
        sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes(); assert w.getsampwidth() == 2, path
        x = np.frombuffer(w.readframes(n), "<i2").reshape(n, ch).T.astype(np.float32)
    x = x.mean(0, keepdims=True) if ch > 1 else x
    if sr != 16000:
        from scipy.signal import resample_poly
        g = math.gcd(sr, 16000); x = resample_poly(x, 16000 // g, sr // g, axis=1).astype(np.float32)
    return x
def one(path):
    x = torch.from_numpy(pcm16(path)); n = x.shape[1]; k = max(1, math.ceil(n / (30 * 16000))); bounds = np.linspace(0, n, k + 1, dtype=np.int64); embs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        fb = torchaudio.compliance.kaldi.fbank(x[:, a:b], num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0, sample_frequency=16000, window_type="hamming")
        fb = fb - fb.mean(0, keepdim=True)
        with torch.inference_mode(): e = model(fb[None].to(dev))
        e = e[-1] if isinstance(e, (tuple, list)) else e; e = e[0].float().cpu(); embs.append(e / e.norm())
    e = torch.stack(embs).mean(0); return (e / e.norm()).numpy()
def vencode(ps, tag):
    V = np.zeros((len(ps), 512), np.float32); ok = np.zeros(len(ps), bool)
    for i, p in enumerate(ps):
        try: V[i] = one(p); ok[i] = np.isfinite(V[i]).all()
        except Exception as ex: log(f"voice fail {p}: {ex!r}")
        if i % 2000 == 0: log(f"  campplus {tag} {i}/{len(ps)}")
    return V, ok
e, _ = vencode([paths[i] for i in PAR["idx"]], "parity"); pj = gate(e, "campplus", 512, "campplus")
V, ok = vencode(paths, "all"); np.save(OUT / "xju2_voice_campplus.float32.npy", V); np.save(OUT / "xju2_voice_campplus.ok.npy", ok)
log(f"CAMPPLUS DONE ok {ok.sum()}/{len(ok)}")
'''

RDNW = WORKER_HEAD + r'''
import torchaudio, wave
from concurrent.futures import ThreadPoolExecutor
rdn = torch.hub.load("IDRnD/ReDimNet", "ReDimNet", model_name="b6", train_type="ft_lm", dataset="vox2", trust_repo=True).to(dev).eval()
ck = Path(torch.hub.get_dir()) / "checkpoints/b6-vox2-ft_lm.pt"; hsh = hashlib.sha256(ck.read_bytes()).hexdigest()
log(f"rdn weights sha {hsh[:16]}"); assert hsh == "__RDNSHA__", "ReDimNet weights differ from the cached extraction"
def read(p):
    with wave.open(str(p)) as w:
        assert w.getsampwidth() == 2, p; sr, ch = w.getframerate(), w.getnchannels(); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    x = x.reshape(-1, ch).mean(1) if ch > 1 else x
    return torchaudio.functional.resample(torch.from_numpy(x)[None], sr, 16000)[0].numpy() if sr != 16000 else x
def prep(p):
    x = read(p); c = [x[s:s + 30 * 16000] for s in range(0, len(x), 30 * 16000)]; c = [q for q in c if len(q) >= 16000] or [x]
    return [torch.from_numpy(q)[None] for q in c]
pool = ThreadPoolExecutor(6)
@torch.no_grad()
def vencode(ps, tag, block=256):
    out = np.zeros((len(ps), 192), np.float32); ok = np.zeros(len(ps), bool)
    def safe(p):
        try: return prep(p)
        except Exception as ex: log(f"voice read fail {p}: {ex!r}"); return None
    for s in range(0, len(ps), block):
        for i, xs in enumerate(pool.map(safe, ps[s:s + block])):
            if xs is None: continue
            with torch.autocast("cuda", dtype=torch.float16):
                e = torch.stack([rdn(x.to(dev))[0].float() for x in xs]).mean(0)
            out[s + i] = F.normalize(e, dim=0).cpu().numpy(); ok[s + i] = np.isfinite(out[s + i]).all()
        if (s // block) % 16 == 0: log(f"  rdn {tag} {min(s + block, len(ps))}/{len(ps)}")
    return out, ok
e, _ = vencode([paths[i] for i in PAR["idx"]], "parity"); pj = gate(e, "rdn", 192, "rdn")
V, ok = vencode(paths, "all"); np.save(OUT / "xju2_voice_rdn.float32.npy", V); np.save(OUT / "xju2_voice_rdn.ok.npy", ok)
log(f"RDN DONE ok {ok.sum()}/{len(ok)}")
'''.replace("__RDNSHA__", RDN_SHA)

FACES_MAIN = r'''
try:
    ROWS = pd.read_csv(io.BytesIO(ub("__ROWS__"))); PAR = json.loads(r"""__PAR__""")
    vb = [p for p in find_dirs("VoxCeleb_1_adapted", 4) if os.path.isdir(os.path.join(p, "face"))]; assert vb, f"VoxCeleb_1_adapted not mounted: {os.listdir('/kaggle/input')}"
    roots = mav_roots(); VB = Path(vb[0])
    ROWS["path"] = [str(VB / r) if s == "vox1" else str(roots[r.split("/")[0]] / r) for s, r in zip(ROWS.src, ROWS.rel)]
    miss = [p for p in ROWS.path if not os.path.exists(p)]; log(f"faces {len(ROWS)} missing {len(miss)} {miss[:3]}"); assert len(miss) < .001 * len(ROWS), "too many faces missing"
    ROWS[["path"]].to_csv(W / "paths.csv", index=False); json.dump(PAR, open(W / "par.json", "w"))
    # AdaFace code + checkpoint (as build_xja_extract.py)
    run(f"git init -q {W}/AdaFace && cd {W}/AdaFace && git remote add origin https://github.com/mk-minchul/AdaFace.git && "
        f"git fetch -q --depth 1 origin c60eaa786a42c03444f3df7096dbaf9d57ae010d && git checkout -q --detach FETCH_HEAD && git rev-parse HEAD")
    from huggingface_hub import hf_hub_download
    ckpt = hf_hub_download("marcelo-victor/adaface_ir101_webface12m", "adaface_weights.ckpt", local_dir=str(W / "adaw"))
    h = hashlib.sha256(); f = open(ckpt, "rb")
    for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    assert h.hexdigest() == "__ADASHA__", "AdaFace checkpoint hash differs from Track 1"
    atp = W / "AdaFace/face_alignment/mtcnn_pytorch/src/align_trans.py"
    s = atp.read_text(); t = s.replace("align_type is 'cv2_affine'", "align_type == 'cv2_affine'").replace("align_type is 'affine'", "align_type == 'affine'")
    if t != s: atp.write_text(t)
    # ArcFace: vendored insightface 0.7.3 + onnxruntime-gpu + buffalo_l (as build_xjb_extract.py)
    (W / "ifx").mkdir(exist_ok=True)
    for n, t in json.loads(ub("__VEND__")).items(): (W / "ifx" / n).write_text(t)
    open(W / "arc_recipe.py", "w").write(r"""__ARC__"""); open(W / "cons.txt", "w").write(f"numpy=={np.__version__}\n")
    run("pip uninstall -y -q onnxruntime onnxruntime-gpu || true"); run(f"pip install -q onnx onnxruntime-gpu -c {W}/cons.txt 2>&1 | tail -3")
    run(f"cd {W} && wget -q https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip && unzip -q -o buffalo_l.zip -d buffalo_l")
    det_p, rec_p = (next((W / "buffalo_l").rglob(n)) for n in ("det_10g.onnx", "w600k_r50.onnx"))
    assert hashlib.sha256(det_p.read_bytes()).hexdigest() == "__DETSHA__" and hashlib.sha256(rec_p.read_bytes()).hexdigest() == "__RECSHA__", "buffalo_l files differ"
    open(W / "ada_worker.py", "w").write(r"""__ADAW__"""); open(W / "arc_worker.py", "w").write(r"""__ARCW__""")
    log("setup done -> workers")
    run_workers([("ada", W / "ada_worker.py", 0), ("arc", W / "arc_worker.py", 1)])
    ROWS.drop(columns=["path"]).to_csv(OUT / "xju2_face_rows.csv", index=False)
    json.dump(dict(pair_local=True, frozen_encoders=True, labels_used=False, test_statistics=False, prereg="sub100a 14/11 20:55",
                   adaface=f"mk-minchul/AdaFace@c60eaa7 + HF marcelo-victor/adaface_ir101_webface12m sha256 {'__ADASHA__'[:16]} (xja recipe)",
                   arcface=f"InsightFace buffalo_l det_10g {'__DETSHA__'[:16]} + w600k_r50 {'__RECSHA__'[:16]}, insightface 0.7.3, det 320 (xjb recipe)",
                   parity={n: json.load(open(OUT / f"{n}_parity.json")) for n in ("ada", "arc")}), open(OUT / "PROVENANCE.json", "w"), indent=1)
    with zipfile.ZipFile("/kaggle/working/xju2_faces.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()): z.write(p, p.name)
    shutil.rmtree(OUT); log("JOB DONE")
except Exception as ex:
    log(f"VMERR {ex!r}"); traceback.print_exc(); raise
'''

VOICES_MAIN = r'''
try:
    ROWS = pd.read_csv(io.BytesIO(ub("__ROWS__"))); PAR = json.loads(r"""__PAR__""")
    base = {}
    for slug in ("voxceleb1train", "voxceleb1test"):
        h = [p for p in find_dirs(slug, 3) if os.path.isdir(os.path.join(p, "wav"))]; assert h, f"{slug} not mounted: {os.listdir('/kaggle/input')}"; base[slug] = Path(h[0])
    roots = mav_roots()
    ROWS["path"] = [str(base[r.split("/")[0]] / r.split("/", 1)[1]) if s == "vox1" else str(roots[r.split("/")[0]] / r) for s, r in zip(ROWS.src, ROWS.rel)]
    miss = [p for p in ROWS.path if not os.path.exists(p)]; log(f"voices {len(ROWS)} missing {len(miss)} {miss[:3]}"); assert len(miss) < .001 * len(ROWS), "too many voices missing"
    ROWS[["path"]].to_csv(W / "paths.csv", index=False); json.dump(PAR, open(W / "par.json", "w"))
    run(f"git clone -q https://github.com/wenet-e2e/wespeaker.git {W}/wespeaker && git -C {W}/wespeaker checkout -q 9fecd6cb4f47475d01761d87c826298dff4ef18c")
    open(W / "cp_worker.py", "w").write(r"""__CPW__"""); open(W / "rdn_worker.py", "w").write(r"""__RDNW__""")
    log("setup done -> workers")
    run_workers([("campplus", W / "cp_worker.py", 0), ("rdn", W / "rdn_worker.py", 1)])
    ROWS.drop(columns=["path"]).to_csv(OUT / "xju2_voice_rows.csv", index=False)
    json.dump(dict(pair_local=True, frozen_encoders=True, labels_used=False, test_statistics=False, prereg="sub100a 14/11 20:55",
                   campplus="Wespeaker/wespeaker-voxceleb-campplus-LM, wespeaker@9fecd6cb (xja recipe)",
                   redimnet=f"torch.hub IDRnD/ReDimNet b6 ft_lm vox2 weights sha {'__RDNSHA__'[:16]}, fp16 autocast (xjb recipe)",
                   parity={n: json.load(open(OUT / f"{n}_parity.json")) for n in ("campplus", "rdn")}), open(OUT / "PROVENANCE.json", "w"), indent=1)
    with zipfile.ZipFile("/kaggle/working/xju2_voices.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()): z.write(p, p.name)
    shutil.rmtree(OUT); log("JOB DONE")
except Exception as ex:
    log(f"VMERR {ex!r}"); traceback.print_exc(); raise
'''


def build(name, user, main, rep, datasets):
    for w in ("__ADAW__", "__ARCW__", "__CPW__", "__RDNW__"):   # worker sources are inserted inside r"""...""" -> must not contain triple double quotes
        assert '"""' not in rep.get(w, ""), w
    src = main
    for a, b in rep.items():
        assert a in src, a; src = src.replace(a, b)
    import re; left = set(re.findall(r"__[A-Z0-9]+__", src)); assert not left, f"unreplaced placeholders {left}"
    cells = [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s} for s in (COMMON, src)]
    for c in cells: compile(c["source"], "cell", "exec")
    for w in ("__ADAW__", "__ARCW__", "__CPW__", "__RDNW__"):
        if w in rep: compile(rep[w], w, "exec")
    d = K / name; d.mkdir(exist_ok=True)
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(d / f"{name}.ipynb", "w"))
    json.dump({"id": f"{user}/flag-t3-{name.replace('_', '-')}", "title": f"FLAG T3 {name.replace('_', ' ')}", "code_file": f"{name}.ipynb", "language": "python",
               "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
               "dataset_sources": datasets, "competition_sources": [], "kernel_sources": []}, open(d / "kernel-metadata.json", "w"), indent=1)
    print("built", d, "notebook bytes", (d / f"{name}.ipynb").stat().st_size)


build("xju2_faces", "ncamtu06", FACES_MAIN,
      {"__ROWS__": b64(F_ROWS[["src", "rel"]].to_csv(index=False).encode()), "__PAR__": json.dumps(FPAR), "__VEND__": b64(json.dumps(VEND).encode()),
       "__ARC__": ARC, "__ADASHA__": ADA_SHA, "__DETSHA__": DET_SHA, "__RECSHA__": REC_SHA, "__ADAW__": ADA, "__ARCW__": ARCW},
      ["robot12138/voxceleb-1", "KAGGLE_USER/external-data-mavceleb"])
build("xju2_voices", "trnlhvy", VOICES_MAIN,
      {"__ROWS__": b64(V_ROWS[["src", "rel"]].to_csv(index=False).encode()), "__PAR__": json.dumps(VPAR), "__CPW__": CPW, "__RDNW__": RDNW, "__RDNSHA__": RDN_SHA},
      ["kryakrya/voxceleb1train", "kryakrya/voxceleb1test", "KAGGLE_USER/external-data-mavceleb"])
F_ROWS.to_csv(K / "xju2_faces/xju2_face_rows_local.csv", index=False); V_ROWS.to_csv(K / "xju2_voices/xju2_voice_rows_local.csv", index=False)   # local copies with xj_row
