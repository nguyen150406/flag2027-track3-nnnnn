"""XJ-1 (15/11, user: "1 hệ duy nhất nghĩa là bangla english chung một hệ duy nhất ... gom các encoder lại như bạn đã làm cho xj-u"):
ONE XJ model for all four cells whose inputs gather XJ-U's six encoders AND sub90c's frozen encoders. XJ-U's external MAV set (xjc_rows.csv:
9,520 faces / 8,779 voices of the teammate's MAV-Celeb v1+v2 clips) has no sub90c features yet -> this Kaggle notebook encodes it with the exact
sub90c recipes (t3_eval_extract.py groups):
  faces : clipnf, clipnf2 (CLIP ViT-L/14 + our no-FLAG LoRAs, t3_clipft_encode.py, libjpeg-turbo decode), farl (t3extract, turbo decode)
  voices: simam, sam100ft, r293 (t3extract WeSpeaker recipe), ecapa (extract_wespeaker.py recipe = the 'ecapa' cache)
Parity gate (mean cos > .98, else the worker stops) on MAV v1 items that are also in our rows_v1 caches (made locally with the same scripts).
Frozen encoders, every file independently, no labels, no FLAG / test data inside the kernel.
usage: python build_xj1_extract.py -> kaggle/xj1_tools/ (private dataset ncamtu06/flag-t3-xj1-tools) + kaggle/xj1_mav/ (kernel ncamtu06/flag-t3-xj1-mav)"""
import json, gzip, base64, hashlib, os, shutil
from pathlib import Path
import numpy as np, pandas as pd

K = Path(__file__).resolve().parent; T3 = K.parent; ROOT = T3.parent; SC = T3 / "scripts"; CX = T3 / "caches/ext"
b64 = lambda b: base64.b64encode(gzip.compress(b, 9)).decode()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
USER = "ncamtu06"

# ------------------------------------------------------------------ rows (XJ-U's MAV set) + parity items shared with rows_v1
R = pd.read_csv(T3 / "kaggle/xjc_rows/xjc_rows.csv")
fr, vr = R[R.kind.eq("face")].reset_index(drop=True), R[R.kind.eq("voice")].reset_index(drop=True)
v1 = pd.read_csv(CX / "rows_v1.csv")
fpos = {f: i for i, f in list(enumerate(v1.face))[::-1]}; vpos = {f: i for i, f in list(enumerate(v1.voice))[::-1]}
FE, VE = ("clipnf", "clipnf2", "farl"), ("simam", "sam100ft", "r293", "ecapa")
fok = np.logical_and.reduce([np.load(CX / f"v1_face_{e}.ok.npy") for e in FE]); vok = np.logical_and.reduce([np.load(CX / f"v1_voice_{e}.ok.npy") for e in VE])
fk = fr.rel.str.split("/faces/").str[1]; vk = vr.rel.str.split("/voices/").str[1]
fc = [i for i, k in enumerate(fk) if fr.src[i] == "v1" and k in fpos and fok[fpos[k]]]
vc = [i for i, k in enumerate(vk) if vr.src[i] == "v1" and k in vpos and vok[vpos[k]]]
fs = np.array(fc)[np.unique(np.linspace(0, len(fc) - 1, min(128, len(fc))).astype(int))]
vs = np.array(vc)[np.unique(np.linspace(0, len(vc) - 1, 128).astype(int))]
PAR = dict(fidx=fs, vidx=vs)
for e in FE: PAR[f"face_{e}"] = np.load(CX / f"v1_face_{e}.float32.npy")[[fpos[fk[i]] for i in fs]].astype(np.float32)
for e in VE: PAR[f"voice_{e}"] = np.load(CX / f"v1_voice_{e}.float32.npy")[[vpos[vk[i]] for i in vs]].astype(np.float32)

# ------------------------------------------------------------------ private tools dataset
TD = K / "xj1_tools"; TD.mkdir(exist_ok=True)
WEIGHTS = {"avg_model.pt": ROOT / ".tools/simam/unpacked/voxblink2_samresnet100/avg_model.pt",
           "clipnf.pt": T3 / "caches/ultra_lora/clipnf.pt", "clipnf2.pt": T3 / "caches/ultra_lora/clipnf2.pt"}
for n, p in WEIGHTS.items():
    q = TD / n
    if not q.exists(): os.link(p, q)          # hard link, no copy
    assert sha(q) == sha(p), n
fr[["src", "rel"]].to_csv(TD / "xj1_face_rows.csv", index=False); vr[["src", "rel"]].to_csv(TD / "xj1_voice_rows.csv", index=False)
if (TD / "xj1_par.npz").exists():   # 15/11 rerun: keep the uploaded file (np.savez embeds a timestamp -> sha would change)
    _P = np.load(TD / "xj1_par.npz"); assert set(_P.files) == set(PAR) and all(np.array_equal(_P[k], PAR[k]) for k in PAR), "parity refs changed"
else: np.savez(TD / "xj1_par.npz", **PAR)
for n in ("t3extract.py", "t3_clipft_encode.py", "extract_wespeaker.py"): shutil.copy(SC / n, TD / n)
SHAS = {n: sha(TD / n) for n in sorted(os.listdir(TD)) if n != "dataset-metadata.json"}
json.dump({"title": "flag-t3-xj1-tools", "id": f"{USER}/flag-t3-xj1-tools", "licenses": [{"name": "other"}]}, open(TD / "dataset-metadata.json", "w"), indent=1)
FARL_SHA = sha(ROOT / ".tools/farl/FaRL-Base-Patch16-LAIONFace20M-ep16.pth")

COMMON = r'''
import os, sys, subprocess, time, shutil, zipfile, json, glob, traceback, hashlib, wave
from pathlib import Path
import numpy as np, pandas as pd
T0 = time.time()
def log(m): print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 60:6.1f}m]", m, flush=True)
def run(c): print("+", c, flush=True); subprocess.run(c, shell=True, check=True)
def find_dirs(name, depth=4):   # depth-limited: never walk the media trees
    return [p for d in range(1, depth + 1) for p in glob.glob("/kaggle/input/" + "/".join(["*"] * d)) if os.path.basename(p) == name]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
W = Path("/kaggle/temp/xj1"); W.mkdir(parents=True, exist_ok=True); OUT = Path("/kaggle/working/xj1_out"); OUT.mkdir(parents=True, exist_ok=True)
ROOT = W / "root"; S = ROOT / "Track 3/scripts"; TL = ROOT / ".tools"
def run_workers(jobs):
    """jobs = [(name, script_path, gpu)] -> in parallel (one per GPU) when 2 GPUs exist, else one after another; stream the log tails."""
    import torch
    ng = torch.cuda.device_count(); log(f"GPUs {ng}: {[torch.cuda.get_device_name(i) for i in range(ng)]}")
    batches = [jobs] if ng >= 2 else [[j] for j in jobs]
    for b in batches:
        ps = {}
        for name, script, gpu in b:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu if ng >= 2 else 0), PYTHONUNBUFFERED="1", T3_TOOLS=str(TL), T3_DEVICE="cuda")
            ps[name] = (subprocess.Popen([sys.executable, str(script)], stdout=open(W / f"{name}.log", "w"), stderr=subprocess.STDOUT, env=env, cwd=str(S)), W / f"{name}.log")
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
import os, sys, time, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
T0 = time.time()
def log(m): print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 60:6.1f}m]", m, flush=True)
W = Path(os.environ.get("XJ1_W", "/kaggle/temp/xj1")); OUT = Path(os.environ.get("XJ1_OUT", "/kaggle/working/xj1_out")); ROOT = W / "root"
sys.path.insert(0, str(ROOT / "Track 3/scripts"))
PAR = np.load(W / "xj1_par.npz")
def unit(E): E = np.asarray(E, np.float32); return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
def gate(E, key, name):
    cs = (unit(E) * unit(PAR[key])).sum(1); pj = dict(mean=float(cs.mean()), min=float(cs.min()), p05=float(np.percentile(cs, 5)), n=int(len(cs)))
    json.dump(pj, open(OUT / f"{name}_parity.json", "w"), indent=1); log(f"PARITY {name} vs our rows_v1 cache: {pj}")
    assert cs.mean() > .98, f"{name} does not reproduce our cache: {pj}"
def save(tag, E, ok):
    E = np.asarray(E, np.float32); ok = np.asarray(ok, bool) & np.isfinite(E).all(1) & (np.abs(E).sum(1) > 0)
    np.save(OUT / f"xj1_{tag}.float32.npy", E); np.save(OUT / f"xj1_{tag}.ok.npy", ok); log(f"SAVED {tag} {E.shape} ok {int(ok.sum())}/{len(ok)}")
'''

FACES_W = WORKER_HEAD + r'''
import torch
L = pd.read_csv(W / "faces.csv"); paths, good = L.path.tolist(), L.good.to_numpy(bool); dev = torch.device(os.environ.get("XJ1_DEV", "cuda"))
pp = [paths[i] for i in PAR["fidx"]]
import t3_clipft_encode as CE
for tag in ("clipnf", "clipnf2"):
    clip = CE.build(str(ROOT / "Track 3/caches/ultra_lora" / f"{tag}.pt"), dev)
    gate(CE.encode(clip, pp, dev, turbo=True), f"face_{tag}", tag)
    save(f"face_{tag}", CE.encode(clip, paths, dev, turbo=True), good); del clip; torch.cuda.empty_cache() if dev.type == "cuda" else None
import t3extract as X
fn = X.face_encoder("farl", True); gate(fn(pp), "face_farl", "farl"); save("face_farl", fn(paths, log=log), good)
log("FACES DONE")
'''

VOICES_W = WORKER_HEAD + r'''
import torch
L = pd.read_csv(W / "voices.csv"); paths, good = L.path.tolist(), L.good.to_numpy(bool); dev = os.environ.get("XJ1_DEV", "cuda")
pp = [paths[i] for i in PAR["vidx"]]
import t3extract as X
for enc in ("simam", "sam100ft", "r293"):
    fn = X.voice_encoder(enc); gate(fn(pp), f"voice_{enc}", enc); save(f"voice_{enc}", fn(paths, log=log), good)
    del fn; torch.cuda.empty_cache() if dev == "cuda" else None
import extract_wespeaker as EW, wave, torchaudio
def _load_pcm(path):   # 15/11 v2: EW.load_pcm asserts mono; same result for mono files, channel mean for multi-channel (as t3extract._pcm16)
    with wave.open(str(path)) as w:
        assert w.getsampwidth() == 2, path; sr, ch = w.getframerate(), w.getnchannels(); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
    x = x.reshape(-1, ch).mean(1) if ch > 1 else x
    x = torch.from_numpy(np.ascontiguousarray(x))[None]
    return torchaudio.functional.resample(x, sr, 16000) if sr != 16000 else x
EW.load_pcm = _load_pcm
EC = "Wespeaker/wespeaker-voxceleb-ecapa-tdnn1024-LM"
EW.main(EC, "par", dev, rows=pd.DataFrame(dict(path=pp)), out_dir=W / "ecapa_par"); gate(np.load(W / "ecapa_par/voice_embeddings.float32.npy"), "voice_ecapa", "ecapa")
EW.main(EC, "all", dev, rows=pd.DataFrame(dict(path=paths)), out_dir=W / "ecapa_all"); save("voice_ecapa", np.load(W / "ecapa_all/voice_embeddings.float32.npy"), good)
log("VOICES DONE")
'''

MAIN = r'''
try:
    TD = Path(find_dirs("xj1_par.npz")[0]).parent; SH = json.loads(r"""__SHAS__""")
    for n, h in SH.items(): assert sha(TD / n) == h, f"tools file changed: {n}"
    S.mkdir(parents=True, exist_ok=True)
    for n in ("t3extract.py", "t3_clipft_encode.py", "extract_wespeaker.py"): shutil.copy(TD / n, S / n)
    (ROOT / "flag_raw_s01_v3_adaface_simam100").mkdir(parents=True, exist_ok=True)     # extract_wespeaker.py reads this at import (FLAG rows: not used here)
    pd.DataFrame(columns=["row_id", "partition", "protocol", "voice_relpath", "face_relpath"]).to_csv(ROOT / "flag_raw_s01_v3_adaface_simam100/metadata.csv", index=False)
    (TL / "simam/unpacked/voxblink2_samresnet100").mkdir(parents=True, exist_ok=True); (ROOT / "Track 3/caches/ultra_lora").mkdir(parents=True, exist_ok=True)
    os.symlink(TD / "avg_model.pt", TL / "simam/unpacked/voxblink2_samresnet100/avg_model.pt")
    for n in ("clipnf.pt", "clipnf2.pt"): os.symlink(TD / n, ROOT / "Track 3/caches/ultra_lora" / n)
    shutil.copy(TD / "xj1_par.npz", W / "xj1_par.npz")
    (TL / "farl").mkdir(exist_ok=True); fp = TL / "farl/FaRL-Base-Patch16-LAIONFace20M-ep16.pth"
    run(f"wget -q -O {fp} https://github.com/FacePerceiver/FaRL/releases/download/pretrained_weights/FaRL-Base-Patch16-LAIONFace20M-ep16.pth")
    assert sha(fp) == "__FARLSHA__", "FaRL weights differ from ours"
    run(f"git clone -q https://github.com/openai/CLIP.git {TL}/farl/CLIP && git -C {TL}/farl/CLIP checkout -q d50d76daa670286dd6cacf3bcd80b5e4823fc8e1")
    (TL / "farl/vendor").mkdir(exist_ok=True); run("pip install -q ftfy regex 2>&1 | tail -1")
    run(f"git clone -q https://github.com/wenet-e2e/wespeaker.git {TL}/wespeaker && git -C {TL}/wespeaker checkout -q 9fecd6cb4f47475d01761d87c826298dff4ef18c")
    roots = {}
    for top in ("mavceleb_v1_complete-001", "mavceleb_v2_complete-002"):
        h = [p for p in find_dirs(top) if os.path.isdir(p)]; assert h, f"{top} not mounted"; roots[top] = Path(h[0]).parent
    import cv2
    for kind in ("face", "voice"):
        R = pd.read_csv(TD / f"xj1_{kind}_rows.csv"); ps = [str(roots[r.split("/")[0]] / r) for r in R.rel]
        if kind == "face": good = [os.path.exists(p) and cv2.imread(p) is not None for p in ps]
        else:
            def wok(p):
                try:
                    with wave.open(p) as w: return w.getsampwidth() == 2 and w.getnframes() > 0   # 15/11 v2: multi-channel allowed (channel mean, as t3extract / XJ-B rdn)
                except Exception: return False
            good = [wok(p) for p in ps]
            def nch(p):
                try:
                    with wave.open(p) as w: return w.getnchannels()
                except Exception: return -1
            C = pd.Series([nch(p) for p in ps]).value_counts().to_dict(); log(f"voice channel counts (-1 = unreadable) {C}")
        good = np.array(good); sub = ps[int(np.argmax(good))]; log(f"{kind}s {len(ps)} usable {int(good.sum())}; unusable files are encoded as a stand-in and marked ok=False")
        assert good.mean() > .99, f"too many unusable {kind} files"
        pd.DataFrame(dict(path=[p if g else sub for p, g in zip(ps, good)], good=good)).to_csv(W / f"{kind}s.csv", index=False)
        R.to_csv(OUT / f"xj1_{kind}_rows.csv", index=False)
    open(W / "faces_worker.py", "w").write(r"""__FACESW__"""); open(W / "voices_worker.py", "w").write(r"""__VOICESW__""")
    log("setup done -> workers")
    run_workers([("faces", W / "faces_worker.py", 0), ("voices", W / "voices_worker.py", 1)])
    json.dump(dict(pair_local=True, frozen_encoders=True, labels_used=False, test_statistics=False, purpose="XJ-1 (one model, all cells): sub90c encoders on XJ-U's MAV set",
                   faces="clipnf, clipnf2 (t3_clipft_encode.py, turbo), farl (t3extract, turbo)", voices="simam, sam100ft, r293 (t3extract), ecapa (extract_wespeaker.py)",
                   tools_sha=SH, farl_sha="__FARLSHA__", clip_repo="openai/CLIP@d50d76d", wespeaker="wespeaker@9fecd6cb",
                   parity={n: json.load(open(OUT / f"{n}_parity.json")) for n in ("clipnf", "clipnf2", "farl", "simam", "sam100ft", "r293", "ecapa")}),
              open(OUT / "PROVENANCE.json", "w"), indent=1)
    with zipfile.ZipFile("/kaggle/working/xj1_mav.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()): z.write(p, p.name)
    shutil.rmtree(OUT); log("JOB DONE")
except Exception as ex:
    log(f"VMERR {ex!r}"); traceback.print_exc(); raise
'''

if __name__ == "__main__":
    for w in (FACES_W, VOICES_W): assert '"""' not in w; compile(w, "worker", "exec")
    src = MAIN.replace("__SHAS__", json.dumps(SHAS)).replace("__FARLSHA__", FARL_SHA).replace("__FACESW__", FACES_W).replace("__VOICESW__", VOICES_W)
    import re; assert not re.findall(r"__[A-Z]+__", src), re.findall(r"__[A-Z]+__", src)
    cells = [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s} for s in (COMMON, src)]
    for c in cells: compile(c["source"], "cell", "exec")
    d = K / "xj1_mav"; d.mkdir(exist_ok=True)
    json.dump({"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5},
              open(d / "xj1_mav.ipynb", "w"))
    json.dump({"id": f"{USER}/flag-t3-xj1-mav", "title": "FLAG T3 xj1 mav", "code_file": "xj1_mav.ipynb", "language": "python", "kernel_type": "notebook",
               "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
               "dataset_sources": [f"{USER}/flag-t3-xj1-tools", "KAGGLE_USER/external-data-mavceleb"], "competition_sources": [], "kernel_sources": []},
              open(d / "kernel-metadata.json", "w"), indent=1)
    (d / "faces_worker.py").write_text(FACES_W); (d / "voices_worker.py").write_text(VOICES_W)   # copies for the local smoke test
    print(f"built {d} | tools {TD} files {list(SHAS)} | parity faces {len(fs)} voices {len(vs)}")
