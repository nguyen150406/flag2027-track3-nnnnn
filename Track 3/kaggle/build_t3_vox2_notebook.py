"""Build the Kaggle notebook that turns a VoxCeleb2-dev subset into Track 3 auxiliary embeddings.

Selective download from the public Hugging Face mirror (ProgramComputer/voxceleb, cc-by-4.0): zip central
directories are read remotely and each chosen clip is fetched with one HTTP range request (no full download).
Per identity: up to K videos x 1 clip; per clip: 16 kHz mono audio + 2 face frames (VoxCeleb2 mp4 are 224x224
face tracks, same size as FLAG face frames). Frozen encoders identical to the FLAG / VoxCeleb1 caches
(t3extract.py): faces farl, clip; voices simam, sam100ft, ecapa, r293; plus audeering-24 age/gender (hidden)
and ViT-age (CLS + logits) for the attribute / voice-age directions. Every file is processed independently."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
K_VIDEOS = 5
cells = []
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s})
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})

md("# FLAG Track 3 — VoxCeleb2 auxiliary embeddings\n"
   "Selective range-download of VoxCeleb2-dev clips (HF mirror), frozen encoders, pair-local. No FLAG data is used here.")
code(r'''
import os, sys, subprocess, time, shutil, zipfile, json, io, struct, zlib, threading, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd, requests
def run(c): print("+", c); subprocess.run(c, shell=True, check=True)
run("pip install -q ftfy regex pyyaml huggingface_hub imageio-ffmpeg 2>&1 | tail -1")
T0 = time.time()
def log(m): print(time.strftime("%H:%M:%S"), f"[{(time.time()-T0)/60:5.1f}m]", m, flush=True)
W = Path("/kaggle/temp/vox2"); (W / "wav").mkdir(parents=True, exist_ok=True); (W / "img").mkdir(parents=True, exist_ok=True)
OUT = Path("/kaggle/working/vox2_out"); OUT.mkdir(parents=True, exist_ok=True)
FFMPEG = shutil.which("ffmpeg")
if FFMPEG is None:
    import imageio_ffmpeg; FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
log(f"ffmpeg {FFMPEG}")
def find(marker, depth=6):
    base = Path("/kaggle/input"); hits = []
    for root, dirs, files in os.walk(base):
        d = Path(root)
        if len(d.relative_to(base).parts) >= depth: dirs[:] = []
        if marker in files or marker in dirs: hits.append(d / marker)
    return hits
TOOLS_SRC = find("t3extract.py")[0].parent
T = Path("/kaggle/temp/t3tools"); (T / "farl").mkdir(parents=True, exist_ok=True); (T / "simam/unpacked/voxblink2_samresnet100").mkdir(parents=True, exist_ok=True)
for a, b in ((TOOLS_SRC / "farl/FaRL-Base-Patch16-LAIONFace20M-ep16.pth", T / "farl/FaRL-Base-Patch16-LAIONFace20M-ep16.pth"),
             (TOOLS_SRC / "simam/avg_model.pt", T / "simam/unpacked/voxblink2_samresnet100/avg_model.pt")):
    if not b.exists(): os.symlink(a, b)
if not (T / "farl/CLIP/clip").exists():
    run(f"git clone -q https://github.com/openai/CLIP.git {T}/farl/CLIP && git -C {T}/farl/CLIP checkout -q d50d76daa670286dd6cacf3bcd80b5e4823fc8e1")
(T / "farl/vendor").mkdir(exist_ok=True)
if not (T / "wespeaker/wespeaker").exists():
    run(f"git clone -q https://github.com/wenet-e2e/wespeaker.git {T}/wespeaker && git -C {T}/wespeaker checkout -q 9fecd6cb4f47475d01761d87c826298dff4ef18c")
os.environ["T3_TOOLS"] = str(T); os.environ["T3_DEVICE"] = "cuda"
sys.path.insert(0, str(TOOLS_SRC))
import t3extract as X
log(f"device {X.DEV}")
''')
code(r'''
# ---------------------------------------------------------------- remote zip listing + clip selection
BASE = "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox2/"
ZIPS = [f"vox2_mp4_{i}.zip" for i in range(1, 7)]
class HTTPFile(io.RawIOBase):
    def __init__(self, url):
        self.s = requests.Session(); r = self.s.head(url, allow_redirects=True, timeout=60); r.raise_for_status()
        self.url, self.size, self.pos = r.url, int(r.headers["Content-Length"]), 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off; return self.pos
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        if n == 0 or self.pos >= self.size: return b""
        end = min(self.size, self.pos + n) - 1
        for k in range(6):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=120); r.raise_for_status(); break
            except Exception as e:
                if k == 5: raise
                time.sleep(2 ** k)
        self.pos += len(r.content); return r.content
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)
recs = []
for zi, name in enumerate(ZIPS):
    f = HTTPFile(BASE + name); z = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 20))
    for i in z.infolist():
        if i.filename.endswith(".mp4"):
            p = i.filename.split("/")
            recs.append((zi, i.filename, p[-3], p[-2], p[-1][:-4], i.header_offset, i.compress_size, i.file_size, i.compress_type))
    log(f"{name}: {len(recs)} clips so far")
allc = pd.DataFrame(recs, columns=["zip", "fname", "identity", "video", "clip", "off", "csize", "size", "ctype"])
meta = pd.read_csv(io.StringIO(requests.get(BASE + "vox2_meta.csv", timeout=60).text))
meta.columns = [c.strip().replace("﻿", "") for c in meta.columns]
gender = dict(zip(meta["VoxCeleb2 ID"].str.strip(), meta["Gender"].str.strip()))
rng = np.random.default_rng(2027); pick = []
for ident, g in allc.groupby("identity"):
    vids = g.video.unique(); vids = rng.choice(vids, min(len(vids), K_VIDEOS), replace=False)
    for v in vids:
        gv = g[g.video == v]; pick.append(gv.index[rng.integers(len(gv))])
man = allc.loc[sorted(pick)].reset_index(drop=True)
man["gender"] = man.identity.map(gender).fillna("u")
man.to_csv(OUT / "vox2_manifest.csv", index=False)
log(f"identities {man.identity.nunique()} clips {len(man)} MB {man.csize.sum()/1e6:.0f}  gender {man.drop_duplicates('identity').gender.value_counts().to_dict()}")
''')
cells[-1]["source"] = cells[-1]["source"].replace("K_VIDEOS", str(K_VIDEOS))
code(r'''
# ---------------------------------------------------------------- parallel fetch (1 range request / clip) + decode
URLS = {}
url_lock = threading.Lock()
def zip_url(zi, refresh=False):
    with url_lock:
        if refresh or zi not in URLS:
            URLS[zi] = requests.head(BASE + ZIPS[zi], allow_redirects=True, timeout=60).url
        return URLS[zi]
tls = threading.local()
def session():
    if not hasattr(tls, "s"): tls.s = requests.Session()
    return tls.s
def fetch(r):
    fname_len = len(r.fname.encode())
    for k in range(4):
        try:
            url = zip_url(r.zip, refresh=k > 0)
            start = int(r.off); end = start + 30 + fname_len + 1024 + int(r.csize)
            resp = session().get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120); resp.raise_for_status()
            b = resp.content; assert b[:4] == b"PK\x03\x04"
            n, m = struct.unpack("<HH", b[26:30]); off = 30 + n + m; data = b[off:off + int(r.csize)]
            if len(data) < r.csize:
                data += session().get(url, headers={"Range": f"bytes={start+off+len(data)}-{start+off+int(r.csize)-1}"}, timeout=120).content
            data = zlib.decompressobj(-15).decompress(data) if r.ctype == 8 else data
            assert len(data) == r.size
            return data
        except Exception as e:
            err = e; time.sleep(2 ** k)
    raise err
def work(i):
    r = REC[i]; wav = W / "wav" / f"{i}.wav"; img = W / "img" / f"{i}_%d.jpg"
    if wav.exists() and (W / "img" / f"{i}_1.jpg").exists(): return i, True, ""
    try:
        data = fetch(r); tmp = W / f"tmp_{i}.mp4"; tmp.write_bytes(data)
        cmd = [FFMPEG, "-v", "error", "-y", "-i", str(tmp), "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(wav),
               "-map", "0:v:0", "-vf", r"select='eq(n\,20)+eq(n\,60)'", "-vsync", "0", "-q:v", "2", str(img)]
        p = subprocess.run(cmd, capture_output=True, text=True); tmp.unlink()
        if p.returncode != 0 or not (W / "img" / f"{i}_1.jpg").exists(): return i, False, p.stderr[-200:]
        return i, True, ""
    except Exception as e:
        return i, False, repr(e)[:200]
REC = list(man.itertuples(index=False))
ok = np.zeros(len(man), bool); errs = []
with ThreadPoolExecutor(32) as ex:
    futs = [ex.submit(work, i) for i in range(len(man))]
    for n, fu in enumerate(as_completed(futs)):
        i, good, msg = fu.result(); ok[i] = good
        if not good:
            errs.append((i, msg))
            if len(errs) <= 5: log(f'error {i}: {msg}')
        if n % 500 == 0: log(f"fetched {n}/{len(man)} ok {ok.sum()} errors {len(errs)}")
log(f"fetch done ok {ok.sum()}/{len(man)}; first errors {errs[:3]}")
man["ok"] = ok; man["n_frames"] = [int((W / "img" / f"{i}_2.jpg").exists()) + int((W / "img" / f"{i}_1.jpg").exists()) for i in range(len(man))]
man.to_csv(OUT / "vox2_manifest.csv", index=False)
''')
code(r'''
# ---------------------------------------------------------------- rows (one per face frame) + frozen identity encoders
rows = []
for i, r in zip(man.index[man.ok], man[man.ok].itertuples(index=False)):
    for k in range(1, r.n_frames + 1):
        rows.append(dict(identity=r.identity, gender=r.gender, video=r.video, clip_index=i,
                         voice=f"{r.identity}/{r.video}/{r.clip}.wav", face=f"{r.identity}/{r.video}/{r.clip}_{k}.jpg",
                         face_path=str(W / "img" / f"{i}_{k}.jpg")))
rows = pd.DataFrame(rows); clips = np.array(sorted(rows.clip_index.unique()))
cpos = {c: j for j, c in enumerate(clips)}; ridx = rows.clip_index.map(cpos).values
log(f"rows {len(rows)} clips {len(clips)} identities {rows.identity.nunique()}")
import torch
def save(tag, arr):
    np.save(OUT / f"vox2_{tag}.float32.npy", arr.astype(np.float32)); np.save(OUT / f"vox2_{tag}.ok.npy", np.ones(len(arr), bool))
    log(f"saved {tag} {arr.shape}")
for enc in ("farl", "clip"):
    if (OUT / f"vox2_face_{enc}.float32.npy").exists(): continue
    save(f"face_{enc}", X.face_encoder(enc)(list(rows.face_path), log=log)); torch.cuda.empty_cache()
wavs = [str(W / "wav" / f"{c}.wav") for c in clips]
for enc in ("simam", "sam100ft", "ecapa", "r293"):
    if (OUT / f"vox2_voice_{enc}.float32.npy").exists(): continue
    e = X.voice_encoder(enc)(wavs, log=log); save(f"voice_{enc}", e[ridx]); torch.cuda.empty_cache()
rows.drop(columns=["face_path"]).to_csv(OUT / "rows_vox2.csv", index=False)
''')
code(r'''
# ---------------------------------------------------------------- attributes (time-guarded): audeering-24 voice, ViT-age face
import torch.nn as nn
from transformers import Wav2Vec2Processor, AutoImageProcessor, AutoModelForImageClassification
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel
from PIL import Image
class Head(nn.Module):
    def __init__(self, c, n):
        super().__init__(); self.dense = nn.Linear(c.hidden_size, c.hidden_size); self.dropout = nn.Dropout(c.final_dropout); self.out_proj = nn.Linear(c.hidden_size, n)
    def forward(self, x): return self.out_proj(self.dropout(torch.tanh(self.dense(self.dropout(x)))))
class AgeGender(Wav2Vec2PreTrainedModel):
    def __init__(self, c):
        super().__init__(c); self.wav2vec2 = Wav2Vec2Model(c); self.age = Head(c, 1); self.gender = Head(c, 3); self.init_weights()
    def forward(self, x):
        h = self.wav2vec2(x).last_hidden_state.mean(1); return self.age(h), torch.softmax(self.gender(h), 1), h
if time.time() - T0 < 8.5 * 3600 and not (OUT / "vox2_face_vit.npy").exists():
    proc = AutoImageProcessor.from_pretrained("nateraw/vit-age-classifier")
    model = AutoModelForImageClassification.from_pretrained("nateraw/vit-age-classifier").to("cuda").eval(); cap = {}
    model.classifier.register_forward_hook(lambda m, i, o: cap.__setitem__("x", i[0].detach()))
    out = []
    with torch.inference_mode():
        for s in range(0, len(rows), 128):
            imgs = [Image.open(p).convert("RGB") for p in rows.face_path[s:s + 128]]
            lg = model(**{k: v.to("cuda") for k, v in proc(images=imgs, return_tensors="pt").items()}).logits
            x = cap["x"]; x = x[:, 0] if x.dim() == 3 else x; out.append(torch.cat([x, lg], 1).float().cpu().numpy())
    np.save(OUT / "vox2_face_vit.npy", np.concatenate(out).astype(np.float16)); log("saved face vit"); del model; torch.cuda.empty_cache()
if time.time() - T0 < 9.0 * 3600 and not (OUT / "vox2_voice_aud24.npz").exists():
    repo = "audeering/wav2vec2-large-robust-24-ft-age-gender"
    proc = Wav2Vec2Processor.from_pretrained(repo); model = AgeGender.from_pretrained(repo).to("cuda").eval()
    ages, gen, emb = [], [], []
    with torch.inference_mode():
        for j, p in enumerate(wavs):
            if time.time() - T0 > 10.5 * 3600: log("audeering stopped by time guard"); break
            w = X._pcm16(p)[0] / 32768.0
            x = proc(w, sampling_rate=16000, return_tensors="pt").input_values.to("cuda")
            a, g, h = model(x); ages.append(float(a.clamp(0, 1))); gen.append(g[0].cpu().numpy()); emb.append(h[0].float().cpu().numpy())
            if j % 5000 == 0: log(f"audeering {j}/{len(wavs)}")
    n = len(ages)
    np.savez(OUT / "vox2_voice_aud24.npz", age=np.array(ages, np.float32), gender=np.stack(gen).astype(np.float32),
             emb=np.stack(emb).astype(np.float16), clip_index=clips[:n])
    log(f"saved audeering for {n} clips")
''')
code(r'''
json.dump(dict(source="VoxCeleb2 dev (HF ProgramComputer/voxceleb, cc-by-4.0)", k_videos_per_id=int(K_VIDEOS), frames_per_clip=2,
               pair_local=True, frozen_encoders=True, flag_data_used=False), open(OUT / "PROVENANCE.json", "w"))
with zipfile.ZipFile("/kaggle/working/t3_vox2.zip", "w", zipfile.ZIP_STORED) as z:
    for p in sorted(OUT.iterdir()): z.write(p, p.name)
shutil.rmtree(OUT); log("done")
'''.replace("K_VIDEOS", str(K_VIDEOS)))
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
d = HERE / "t3_vox2"; d.mkdir(exist_ok=True)
json.dump(nb, open(d / "t3_vox2.ipynb", "w"), indent=1)
json.dump({"id": "trncnguyn/flag-t3-voxceleb2-extract", "title": "FLAG T3 VoxCeleb2 Extract", "code_file": "t3_vox2.ipynb",
           "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
           "dataset_sources": ["trncnguyn/flag-t3-tools"], "competition_sources": [], "kernel_sources": []},
          open(d / "kernel-metadata.json", "w"), indent=1)
print("built", d)
