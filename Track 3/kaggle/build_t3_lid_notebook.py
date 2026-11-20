"""Kaggle (CPU) notebook: spoken-language ID for every VoxCeleb clip used to train Track 3 projectors, so that any
Bengali speech can be removed (FLAG 2027 FAQ: Bengali speech must not be used to train/adapt the audio side).
  * VoxCeleb2 subset: re-creates exactly the clip selection of t3_vox2 (same listing code + seed), audio only.
  * VoxCeleb1: handled by nationality in scripts/t3_lid_filter.py (no dataset mounts -> fast start).
LID model: speechbrain/lang-id-voxlingua107-ecapa (frozen; used only as a filter, never for training)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
vox2 = json.load(open(HERE / "t3_vox2/t3_vox2.ipynb"))
vox2_code = [c["source"] for c in vox2["cells"] if c["cell_type"] == "code"]
LISTING = vox2_code[1]          # remote zip listing + identical clip selection (K videos, seed 2027)
assert "vox2_manifest.csv" in LISTING and "rng = np.random.default_rng(2027)" in LISTING

cells = []
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s})
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
md("# FLAG Track 3 — language ID filter for VoxCeleb training clips (CPU)")
code(r'''
import os, sys, subprocess, time, shutil, zipfile, json, io, struct, zlib, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd, requests, torch
def run(c): print("+", c); subprocess.run(c, shell=True, check=True)
run("pip install -q speechbrain imageio-ffmpeg 2>&1 | tail -1")
T0 = time.time()
def log(m): print(time.strftime("%H:%M:%S"), f"[{(time.time()-T0)/60:5.1f}m]", m, flush=True)
W = Path("/kaggle/temp/lid"); (W / "wav").mkdir(parents=True, exist_ok=True)
OUT = Path("/kaggle/working/lid_out"); OUT.mkdir(parents=True, exist_ok=True)
FFMPEG = shutil.which("ffmpeg")
if FFMPEG is None:
    import imageio_ffmpeg; FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
torch.set_num_threads(os.cpu_count() or 4)
def find(marker, depth=6):
    base = Path("/kaggle/input"); hits = []
    for root, dirs, files in os.walk(base):
        d = Path(root)
        if len(d.relative_to(base).parts) >= depth or d.name.startswith("id"): dirs[:] = []
        if marker in files or marker in dirs: hits.append(d / marker)
    return hits
log(f"ffmpeg {FFMPEG} cpus {os.cpu_count()}")
''')
code(LISTING)
code(r'''
URLS = {}; url_lock = threading.Lock(); tls = threading.local()
def zip_url(zi, refresh=False):
    with url_lock:
        if refresh or zi not in URLS: URLS[zi] = requests.head(BASE + ZIPS[zi], allow_redirects=True, timeout=60).url
        return URLS[zi]
def session():
    if not hasattr(tls, "s"): tls.s = requests.Session()
    return tls.s
def fetch(r):
    n_ = len(r.fname.encode()); err = None
    for k in range(4):
        try:
            url = zip_url(r.zip, refresh=k > 0); start = int(r.off); end = start + 30 + n_ + 1024 + int(r.csize)
            resp = session().get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120); resp.raise_for_status()
            b = resp.content; assert b[:4] == b"PK\x03\x04"
            n, m = struct.unpack("<HH", b[26:30]); off = 30 + n + m; data = b[off:off + int(r.csize)]
            if len(data) < r.csize:
                data += session().get(url, headers={"Range": f"bytes={start+off+len(data)}-{start+off+int(r.csize)-1}"}, timeout=120).content
            data = zlib.decompressobj(-15).decompress(data) if r.ctype == 8 else data
            assert len(data) == r.size; return data
        except Exception as e:
            err = e; time.sleep(2 ** k)
    raise err
def work(i):
    r = REC[i]; wav = W / "wav" / f"{i}.wav"
    if wav.exists(): return i, True
    try:
        tmp = W / f"tmp_{i}.mp4"; tmp.write_bytes(fetch(r))
        p = subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(tmp), "-map", "0:a:0", "-ac", "1", "-ar", "16000",
                            "-sample_fmt", "s16", "-t", "12", str(wav)], capture_output=True); tmp.unlink()
        return i, p.returncode == 0 and wav.exists()
    except Exception:
        return i, False
REC = list(man.itertuples(index=False))
ok = np.zeros(len(man), bool)
with ThreadPoolExecutor(32) as ex:
    for n, fu in enumerate(as_completed([ex.submit(work, i) for i in range(len(man))])):
        i, good = fu.result(); ok[i] = good
        if n % 3000 == 0: log(f"audio {n}/{len(man)} ok {ok.sum()}")
log(f"vox2 audio ok {ok.sum()}/{len(man)}")
''')
code(r'''
from speechbrain.inference.classifiers import EncoderClassifier
import wave
lid = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="/kaggle/temp/lid_model")
labels = [lid.hparams.label_encoder.ind2lab[i] for i in range(len(lid.hparams.label_encoder.ind2lab))]
want = {k: [i for i, l in enumerate(labels) if l.split(":")[0].strip() == k][0] for k in ("bn", "hi", "ur", "en", "pa", "ta", "te")}
log(f"{len(labels)} languages; tracked {want}")
def read16(p, max_s=12):
    with wave.open(str(p)) as w:
        sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        x = np.frombuffer(w.readframes(min(n, max_s * sr)), "<i2").reshape(-1, ch).mean(1).astype(np.float32) / 32768.0
    if sr != 16000:
        import torchaudio; x = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy()
    return x
def classify(paths, tag):
    rec = []
    with torch.inference_mode():
        for j, p in enumerate(paths):
            if p is None or not Path(p).exists(): rec.append(dict(top="", **{f"p_{k}": np.nan for k in want})); continue
            lp = lid.classify_batch(torch.from_numpy(read16(p))[None])[0][0]
            pr = torch.softmax(lp, 0) if lp.max() > 0 or lp.sum() > 0 else lp.exp()
            pr = pr / pr.sum()
            rec.append(dict(top=labels[int(pr.argmax())], **{f"p_{k}": float(pr[i]) for k, i in want.items()}))
            if j % 3000 == 0: log(f"{tag} LID {j}/{len(paths)}")
    return pd.DataFrame(rec)
v2 = classify([W / "wav" / f"{i}.wav" if ok[i] else None for i in range(len(man))], "vox2")
pd.concat([man[["identity", "video", "clip", "gender"]], v2], axis=1).to_csv(OUT / "lid_vox2.csv", index=False)
log(f"vox2 top languages: {v2.top.value_counts().head(8).to_dict()}  bn-top: {(v2.top.str.startswith('bn')).sum()}")
# VoxCeleb1: no audio LID here (mounting the large VoxCeleb1 datasets left the kernel stuck at start-up, empty logs, twice);
# t3_lid_filter.py excludes VoxCeleb1 people by nationality instead.
with zipfile.ZipFile("/kaggle/working/t3_lid.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()): z.write(p, p.name)
shutil.rmtree(OUT); log("done")
''')
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
d = HERE / "t3_lid"; d.mkdir(exist_ok=True)
json.dump(nb, open(d / "t3_lid.ipynb", "w"), indent=1)
json.dump({"id": "trncnguyn/flag-t3-voxceleb-language-filter", "title": "FLAG T3 VoxCeleb Language Filter", "code_file": "t3_lid.ipynb",
           "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
           "dataset_sources": [],
           "competition_sources": [], "kernel_sources": []}, open(d / "kernel-metadata.json", "w"), indent=1)
print("built", d)
