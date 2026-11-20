"""Kaggle notebook caohuhoan/flag-t3-xja-extract (14/11, user "làm hết đi, chạy kaggle cho nhanh"): features for XJ-A.
 - AdaFace IR-101 WebFace12M (Track 1 RAW-S01-v3 recipe = scripts/t3_extract_adaface.py: official repo @c60eaa7, bundled MTCNN 112x112 limit=1,
   centred-crop fallback, BGR /127.5-1, fp32, L2) on (a) 128 FLAG train faces = PARITY GATE vs the cached FLAG 'ada' rows (mean cos > .98, else stop),
   (b) the 2,326 eval faces (caohuhoan/flag-t3-evalmedia), (c) the teammate's MAV-Celeb v1+v2 face images (xjc_rows, 9,520).
 - WeSpeaker CAM++-LM (t3extract._wespeaker recipe = the rows_v1 / rows_vox campplus caches) on the xjc_rows MAV voice clips (8,779).
Every file is processed on its own (pair-local); no labels; nothing is trained. Rows/parity data are embedded (gzip+base64)."""
import json, gzip, base64, sys
from pathlib import Path
import numpy as np, pandas as pd
K = Path(__file__).resolve().parent; T3 = K.parent; ROOT = T3.parent
sys.path.insert(0, str(T3 / "scripts")); import t3data as D
b64 = lambda b: base64.b64encode(gzip.compress(b, 9)).decode()
rows = (K / "xjc_rows/xjc_rows.csv").read_bytes()
meta = pd.read_csv(ROOT / "flag_raw_s01_v3_adaface_simam100/metadata.csv").sort_values("row_id"); tr = meta[meta.partition.eq("train")]
assert len(tr) == 6485 and (tr.row_id.values == np.arange(6485)).all()
j = np.sort(np.random.default_rng(0).choice(len(tr), 128, replace=False))     # = t3_extract_adaface.py parity rows
par = dict(rel=tr.face_relpath.values[j].tolist(), ref=b64(np.ascontiguousarray(np.asarray(D.face("ada")[:6485], np.float32)[j]).tobytes()))
ev = pd.read_csv(T3 / "outputs/eval_final/rows.csv"); erel = [p.split("/eval_final/root/")[1] for p in ev.face]; assert len(erel) == 2326
cells = []
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s})
code(r'''
import os, sys, subprocess, time, shutil, zipfile, json, glob, traceback, gzip, base64, io, math, types, hashlib, warnings
from pathlib import Path
import numpy as np, pandas as pd
T0 = time.time()
def log(m): print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 60:6.1f}m]", m, flush=True)
def run(c): print("+", c, flush=True); subprocess.run(c, shell=True, check=True)
def find_dirs(name, depth=4):   # depth-limited: never walk the media trees
    return [p for d in range(1, depth + 1) for p in glob.glob("/kaggle/input/" + "/".join(["*"] * d)) if os.path.basename(p) == name]
ub = lambda s: gzip.decompress(base64.b64decode(s))
ROWS = pd.read_csv(io.BytesIO(ub("__ROWS__"))); PAR = json.loads(r"""__PAR__"""); EREL = json.loads(r"""__EREL__""")
W = Path("/kaggle/temp/xja"); W.mkdir(parents=True, exist_ok=True); OUT = Path("/kaggle/working/xja_out"); OUT.mkdir(parents=True, exist_ok=True)
import torch, torch.nn.functional as F
from PIL import Image
assert torch.cuda.is_available(); dev = torch.device("cuda:0"); log(f"gpu {torch.cuda.get_device_name(0)} rows {len(ROWS)} eval faces {len(EREL)}")
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
    ppar = [FLAGTR / r for r in PAR["rel"]]; pev = [EROOT / r for r in EREL]
    assert all(p.exists() for p in ppar) and all(p.exists() for p in pev), "FLAG parity / eval faces missing"
    log(f"paths ok: MAV {roots} FLAG {FLAGTR} eval {EROOT}")
    # ---------------- AdaFace (Track 1 RAW-S01-v3 recipe)
    run(f"git init -q {W}/AdaFace && cd {W}/AdaFace && git remote add origin https://github.com/mk-minchul/AdaFace.git && "
        f"git fetch -q --depth 1 origin c60eaa786a42c03444f3df7096dbaf9d57ae010d && git checkout -q --detach FETCH_HEAD && git rev-parse HEAD")
    from huggingface_hub import hf_hub_download
    ckpt = hf_hub_download("marcelo-victor/adaface_ir101_webface12m", "adaface_weights.ckpt", local_dir=str(W / "adaw"))
    h = hashlib.sha256(); f = open(ckpt, "rb")
    for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    assert h.hexdigest() == "0e7a3238d2a50f3fe3860782534928ac7cb2598977cf897f6869fd5ac2493fd0", "AdaFace checkpoint hash differs from Track 1"
    AR = W / "AdaFace"; sys.path.insert(0, str(AR)); import net as adaface_net
    face_model = adaface_net.build_model("ir_101"); ck = torch.load(ckpt, map_location="cpu", weights_only=False); sd = ck.get("state_dict", ck)
    ms = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items() if k.startswith("model.") or k in face_model.state_dict()}
    missing, unexpected = face_model.load_state_dict(ms, strict=False); assert not missing and not unexpected, (missing[:5], unexpected[:5]); face_model.eval().to(dev)
    atp = AR / "face_alignment/mtcnn_pytorch/src/align_trans.py"
    s = atp.read_text(); t = s.replace("align_type is 'cv2_affine'", "align_type == 'cv2_affine'").replace("align_type is 'affine'", "align_type == 'affine'")
    if t != s: atp.write_text(t)
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
    def embed(paths, tag, bs=64):
        E, S = np.zeros((len(paths), 512), np.float32), np.zeros(len(paths), np.uint8)
        for s in range(0, len(paths), bs):
            xs = []
            for i, p in enumerate(paths[s:s + bs]):
                im, st = align(p); xs.append(tensor(im)); S[s + i] = st
            f, _ = face_model(torch.stack(xs).to(dev)); E[s:s + len(xs)] = F.normalize(f.float(), dim=1).cpu().numpy()
            if (s // bs) % 25 == 0: log(f"  ada {tag} {s + len(xs)}/{len(paths)}")
        return E, S
    log("STAGE adaface ready (checkpoint hash = Track 1)")
    e, st = embed(ppar, "parity"); ref = np.frombuffer(ub(PAR["ref"]), np.float32).reshape(-1, 512); ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    cs = (e * ref).sum(1); pj = dict(mean=float(cs.mean()), min=float(cs.min()), p05=float(np.percentile(cs, 5)), aligned=float(st.mean()))
    json.dump(pj, open(OUT / "ada_parity.json", "w"), indent=1); log(f"PARITY vs cached FLAG ada: {pj}"); assert cs.mean() > .98, f"AdaFace does not reproduce the FLAG cache: {pj}"
    E, S = embed(pev, "eval"); np.save(OUT / "eval_face_ada.float32.npy", E); np.save(OUT / "eval_face_ada.align.npy", S); log(f"eval faces done aligned {S.mean():.3f}")
    fp = ROWS[ROWS.kind.eq("face")].path.tolist(); E, S = embed(fp, "mav")
    np.save(OUT / "xja_face_ada.float32.npy", E); np.save(OUT / "xja_face_ada.ok.npy", np.isfinite(E).all(1) & (np.abs(E).sum(1) > 0)); np.save(OUT / "xja_face_ada.align.npy", S)
    log(f"STAGE mav faces done {E.shape} aligned {S.mean():.3f}"); del face_model; torch.cuda.empty_cache()
    # ---------------- CAM++ (t3extract._wespeaker recipe; wespeaker pinned as in t3_extract)
    run(f"git clone -q https://github.com/wenet-e2e/wespeaker.git {W}/wespeaker && git -C {W}/wespeaker checkout -q 9fecd6cb4f47475d01761d87c826298dff4ef18c")
    import yaml, torchaudio, wave
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
    vp = ROWS[ROWS.kind.eq("voice")].path.tolist(); V = np.zeros((len(vp), 512), np.float32); okv = np.zeros(len(vp), bool)
    for i, p in enumerate(vp):
        try: V[i] = one(p); okv[i] = np.isfinite(V[i]).all()
        except Exception as ex: log(f"voice fail {p}: {ex!r}")
        if i % 1000 == 0: log(f"  campplus {i}/{len(vp)}")
    np.save(OUT / "xja_voice_campplus.float32.npy", V); np.save(OUT / "xja_voice_campplus.ok.npy", okv); log(f"STAGE mav voices done ok {okv.sum()}/{len(vp)}")
    ROWS.drop(columns=["path"]).to_csv(OUT / "xja_rows.csv", index=False)
    json.dump(dict(pair_local=True, frozen_encoders=True, labels_used=False, test_statistics=False, adaface="mk-minchul/AdaFace@c60eaa7 + HF marcelo-victor/adaface_ir101_webface12m sha256 0e7a3238",
                   campplus="Wespeaker/wespeaker-voxceleb-campplus-LM, wespeaker@9fecd6cb", parity=pj), open(OUT / "PROVENANCE.json", "w"), indent=1)
    with zipfile.ZipFile("/kaggle/working/xja_embeddings.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()): z.write(p, p.name)
    shutil.rmtree(OUT); log("JOB DONE")
except Exception as ex:
    log(f"VMERR {ex!r}"); traceback.print_exc(); raise
''')
src = cells[0]["source"].replace("__ROWS__", b64(rows)).replace("__PAR__", json.dumps(par)).replace("__EREL__", json.dumps(erel)); cells[0]["source"] = src
for c in cells: compile(c["source"], "cell", "exec")
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
d = K / "xja_extract"; d.mkdir(exist_ok=True); json.dump(nb, open(d / "xja_extract.ipynb", "w"))
json.dump({"id": "caohuhoan/flag-t3-xja-extract", "title": "FLAG T3 xja extract", "code_file": "xja_extract.ipynb", "language": "python", "kernel_type": "notebook",
           "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
           "dataset_sources": ["KAGGLE_USER/external-data-mavceleb", "trncnguyn/flag-2027-data", "caohuhoan/flag-t3-evalmedia"],
           "competition_sources": [], "kernel_sources": []}, open(d / "kernel-metadata.json", "w"), indent=1)
print("built", d, "notebook bytes", (d / "xja_extract.ipynb").stat().st_size)
