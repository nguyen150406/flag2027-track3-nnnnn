"""VoxBlink2 as new external identities (user 2026-10-27 "oke đồng ý" to the idea sweep). Official VoxBlink2 = annotations only and
YouTube blocks bulk downloads, so: AUDIO from the HF mirror adollahamini1998/VoxBlink2-Audio-6-{0..5} (tar shards of <spk>/<video>.mp3,
the first 60 s of each video; every official utterance ends before 60 s) cut to the longest official utterance of the video
(caches/vb2/utt_best.csv from the official vb2_meta.tar.gz); FACES from YouTube's automatic frames i.ytimg.com/vi/<video>/maxres{1,2,3}.jpg
of other videos of the same channel owner, YuNet face boxes framed like the FLAG crops (YuNet on 1,500 FLAG train crops: box h = .70 of
the crop, centre y .378, x .50). Bengali speakers excluded (caches/vb2/exclude_bengali.csv, 5,823). All on trncnguyn only (CC BY-NC-SA,
no sharing):
  trncnguyn/flag-t3-vb2-prep (CPU) : random shards (seed 2027) until N_SPK speakers have >= 1 clip; <= 2 clips/speaker (<= 6 s, centred),
                                     <= 3 face videos x 3 frames, <= 2 faces/frame (h >= 64 px, score >= .85) -> vb2_media.zip + csvs
  trncnguyn/flag-t3-vb2-enc  (GPU) : FaRL/CLIP faces; identity filter (keep a crop only if it matches a crop of ANOTHER video of the
                                     speaker above the 99th percentile of cross-speaker cosines, FaRL); SimAM/sam100ft/r293/ecapa voices
                                     -> t3_ext_vb2.zip (rows_vb2.csv + vb2_* arrays, the vb layout) for the CV kernel kvb2.
Usage: python build_t3_vb2_kernels.py   (dataset trncnguyn/flag-t3-vb2-meta must hold utt_best.csv.gz, vid_status.csv.gz, yunet.onnx)"""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
N_SPK = 15000

prep = r'''
import os, sys, subprocess, time, shutil, zipfile, json, tarfile, threading, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd, requests, cv2
def run(c): print("+", c); subprocess.run(c, shell=True, check=True)
run("pip install -q imageio-ffmpeg 2>&1 | tail -1")
T0 = time.time()
def log(m): print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 60:6.1f}m]", m, flush=True)
W = Path("/kaggle/temp/vb2"); (W / "wav").mkdir(parents=True, exist_ok=True); (W / "img").mkdir(parents=True, exist_ok=True)
OUT = Path("/kaggle/working")
FFMPEG = shutil.which("ffmpeg")
if FFMPEG is None:
    import imageio_ffmpeg; FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
def find(name):  # Kaggle may store an uploaded x.csv.gz decompressed as x.csv
    for root, dirs, files in os.walk("/kaggle/input"):
        for n in (name, name.removesuffix(".gz")):
            if n in files: return Path(root) / n
    raise FileNotFoundError(name + " " + str([str(Path(r) / f) for r, _, fs in os.walk("/kaggle/input") for f in fs][:20]))
N_SPK, MAXC, MAXS, AUDIO_MIN, FACE_MIN = __N_SPK__, 2, 6.0, 50, 120  # stage deadlines (min): HF throughput can collapse
utt = pd.read_csv(find("utt_best.csv.gz")); utt = utt[utt.dur >= 3.0]
U = {(s, v): (u, a, b) for s, v, u, a, b in zip(utt.identity, utt.video, utt.utt.astype(str).str.zfill(5), utt.f0, utt.f1)}
vs = pd.read_csv(find("vid_status.csv.gz")); upl = vs[vs.uploaded].groupby("speaker_id").video_id.apply(list).to_dict()
log(f"utterance table {len(U)} videos / {utt.identity.nunique()} speakers; uploaded-video lists {len(upl)}; ffmpeg {FFMPEG}")
# ---------------------------------------------------------------- audio: stream random shards, cut the longest utterance of wanted videos
HF = "https://huggingface.co/datasets/adollahamini1998"
shards = []
for p in range(6):
    d = requests.get(f"https://huggingface.co/api/datasets/adollahamini1998/VoxBlink2-Audio-6-{p}?blobs=true", timeout=60).json()
    shards += [(p, x["rfilename"], x["size"]) for x in d["siblings"] if x["rfilename"].endswith(".tar")]
random.Random(2027).shuffle(shards); log(f"{len(shards)} shards, {sum(s[2] for s in shards) / 1e12:.2f} TB")
lock = threading.Lock(); clips = {}; rec = []; seen = set(); stop = threading.Event(); nbytes = [0]
def cut(data, out, a, b):
    t0, t1 = a / 25.0, (b + 1) / 25.0
    if t1 - t0 > MAXS: c = (t0 + t1) / 2; t0, t1 = c - MAXS / 2, c + MAXS / 2
    pr = subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", f"{t0:.3f}", "-t", f"{t1 - t0:.3f}", "-i", "pipe:0", "-ac", "1", "-ar", "16000",
                         "-sample_fmt", "s16", str(out)], input=data, capture_output=True)
    return pr.returncode == 0 and out.exists() and out.stat().st_size > 32000
def shard(job):
    p, name, size = job
    if stop.is_set(): return 0
    for k in range(4):
        try:
            got = 0
            with requests.get(f"{HF}/VoxBlink2-Audio-6-{p}/resolve/main/{name}", stream=True, timeout=300) as r:
                r.raise_for_status()
                with tarfile.open(fileobj=r.raw, mode="r|") as t:
                    for m in t:
                        if stop.is_set() or time.time() - t_audio > AUDIO_MIN * 60: stop.set(); break
                        if not m.isfile() or not m.name.endswith(".mp3"): continue
                        q = m.name.split("/"); spk, vid = q[-2], q[-1][:-4]
                        if (spk, vid) not in U: continue
                        with lock:
                            if (spk, vid) in seen or len(clips.get(spk, [])) >= MAXC or (spk not in clips and len(clips) >= N_SPK): continue
                            seen.add((spk, vid)); i = len(rec); rec.append(None); clips.setdefault(spk, []).append(i)
                        u, a, b = U[(spk, vid)]; data = t.extractfile(m).read()
                        ok = cut(data, W / "wav" / f"{i}.wav", a, b)
                        with lock:
                            rec[i] = (spk, vid, u, a, b, ok)
                            if not ok: clips[spk].remove(i)
                        got += ok
            with lock: nbytes[0] += size
            return got
        except Exception as e:
            log(f"retry {name}: {e!r}"[:200]); time.sleep(10 * (k + 1))
    return 0
t = t_audio = time.time(); done = 0
with ThreadPoolExecutor(8) as ex:
    futs = []
    for j, sh in enumerate(shards[:1600]):
        futs.append(ex.submit(shard, sh))
    for fu in as_completed(futs):
        fu.result(); done += 1
        with lock: full = sum(len(v) > 0 for v in clips.values())
        if full >= N_SPK and not stop.is_set(): stop.set(); log("target reached, stopping new shards")
        if done % 25 == 0: log(f"shards {done}, {nbytes[0] / 1e9:.0f} GB, {nbytes[0] / 1e6 / (time.time() - t):.0f} MB/s, speakers {full}, clips {sum(r is not None and r[5] for r in rec)}")
A = pd.DataFrame([r for r in rec if r is not None and r[5]], columns=["identity", "video", "utt", "f0", "f1", "ok"]).drop(columns="ok")
A["clip"] = [i for i, r in enumerate(rec) if r is not None and r[5]]
log(f"AUDIO done: {len(A)} clips, {A.identity.nunique()} speakers, {A.groupby('identity').size().value_counts().to_dict()}")
# ---------------------------------------------------------------- faces: YouTube automatic frames of other videos, YuNet, FLAG framing
YN = str(find("yunet.onnx")); tls = threading.local()
def det():
    if not hasattr(tls, "d"): tls.d = cv2.FaceDetectorYN.create(YN, "", (640, 360), 0.85, 0.3, 50); tls.s = requests.Session()
    return tls.d, tls.s
def crop(fr, x, y, w, h):
    s = h / 0.70; cx, cy = x + w / 2, y + h / 2; left, top = cx - s / 2, cy - 0.378 * s
    M = np.float32([[224 / s, 0, -left * 224 / s], [0, 224 / s, -top * 224 / s]])
    return cv2.warpAffine(fr, M, (224, 224), flags=cv2.INTER_AREA if s > 224 else cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
def frames(job):
    spk, vid = job; out = []
    if time.time() - t > FACE_MIN * 60: return 0
    d, s = det()
    for k in (1, 2, 3):
        im = None
        for q in (f"maxres{k}", f"sd{k}", f"hq{k}"):
            try:
                r = s.get(f"https://i.ytimg.com/vi/{vid}/{q}.jpg", timeout=30)
                if r.status_code == 200 and len(r.content) > 2000:
                    im = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR); break
            except Exception:
                pass
        if im is None: continue
        sc = 640 / im.shape[1]; small = cv2.resize(im, (640, round(im.shape[0] * sc)), interpolation=cv2.INTER_AREA)
        d.setInputSize((small.shape[1], small.shape[0])); _, f = d.detect(small)
        if f is None: continue
        f = f[np.argsort(-f[:, 3])][:2]
        for j, r_ in enumerate(f):
            x, y, w, h = r_[:4] / sc
            if h >= 64:
                c = crop(im, x, y, w, h)
                with lock: i = len(F); F.append((spk, vid, k, j, float(h), float(r_[14]), i))
                cv2.imwrite(str(W / "img" / f"{i}.jpg"), c, [cv2.IMWRITE_JPEG_QUALITY, 95]); out.append(i)
    return len(out)
rng = random.Random(2027); jobs = []
for spk in sorted(A.identity.unique()):
    v = list(upl.get(spk, [])); rng.shuffle(v); jobs += [(spk, x) for x in v[:3]]
log(f"face jobs {len(jobs)} videos"); F = []; t = time.time()
with ThreadPoolExecutor(32) as ex:
    for n, _ in enumerate(as_completed([ex.submit(frames, jb) for jb in jobs])):
        if n % 5000 == 0: log(f"videos {n}/{len(jobs)}, crops {len(F)}, {n / max(1, time.time() - t):.1f} videos/s")
F = pd.DataFrame(F, columns=["identity", "video", "frame", "face", "h", "score", "img"])
log(f"FACES done: {len(F)} crops, {F.identity.nunique()} speakers, {F.groupby('identity').video.nunique().value_counts().sort_index().to_dict()} videos/speaker")
A.to_csv(OUT / "vb2_audio.csv", index=False); F.to_csv(OUT / "vb2_faces.csv", index=False)
json.dump(dict(source="VoxBlink2: audio = HF mirror adollahamini1998/VoxBlink2-Audio-6-* cut to the longest official utterance (official "
                      "vb2_meta timestamps); faces = YouTube automatic frames (i.ytimg.com maxres1-3) of other videos of the speaker, YuNet, "
                      "FLAG-calibrated framing", bengali_speakers_excluded=True, flag_data_used=False, test_data_used=False),
          open(OUT / "PROVENANCE_VB2.json", "w"))
with zipfile.ZipFile(OUT / "vb2_media.zip", "w", zipfile.ZIP_STORED) as z:
    for i in A["clip"]: z.write(W / "wav" / f"{i}.wav", f"wav/{i}.wav")
    for i in F.img: z.write(W / "img" / f"{i}.jpg", f"img/{i}.jpg")
log(f"zip {os.path.getsize(OUT / 'vb2_media.zip') / 1e9:.2f} GB; done")
'''.replace("__N_SPK__", str(N_SPK))

B = json.load(open(HERE / "t3_vox2b/t3_vox2b.ipynb"))
enc_setup = [c["source"] for c in B["cells"] if c["cell_type"] == "code"][0]
enc_setup = enc_setup.replace('W = Path("/kaggle/temp/vox2")', 'W = Path("/kaggle/temp/vb2")').replace('OUT = Path("/kaggle/working/vox2_out")', 'OUT = Path("/kaggle/working/vb2_out")')
assert "/kaggle/temp/vb2" in enc_setup and "vb2_out" in enc_setup

enc = r'''
# ---------------------------------------------------------------- media, faces, identity filter, voices -> rows_vb2.csv + vb2_* arrays
import torch
SRC = find("vb2_audio.csv")[0].parent
A = pd.read_csv(SRC / "vb2_audio.csv", dtype={"utt": str, "video": str}); Fc = pd.read_csv(SRC / "vb2_faces.csv", dtype={"video": str})
if (SRC / "vb2_media.zip").exists():
    with zipfile.ZipFile(SRC / "vb2_media.zip") as z: z.extractall(W)
else:  # Kaggle mounted the zip already unpacked
    media = [d for d in (SRC, SRC / "vb2_media") if (d / "wav").exists()][0]
    for sub in ("wav", "img"): shutil.rmtree(W / sub, ignore_errors=True); os.symlink(media / sub, W / sub)
log(f"media: {len(A)} clips / {A.identity.nunique()} speakers, {len(Fc)} crops / {Fc.identity.nunique()} speakers")
Fc = Fc[Fc.identity.isin(set(A.identity))].reset_index(drop=True)
paths = [str(W / "img" / f"{i}.jpg") for i in Fc.img]
EF = {e: np.asarray(X.face_encoder(e)(paths, log=log), np.float32) for e in ("farl", "clip")}; torch.cuda.empty_cache()
E = EF["farl"] / np.linalg.norm(EF["farl"], axis=1, keepdims=True)
rng = np.random.default_rng(2027); a_ = rng.integers(0, len(E), 400000); b_ = rng.integers(0, len(E), 400000)
x = Fc.identity.values[a_] != Fc.identity.values[b_]; thr = float(np.quantile((E[a_[x]] * E[b_[x]]).sum(1), .99))
keep = np.zeros(len(Fc), bool)
for spk, g in Fc.groupby("identity").groups.items():
    g = np.asarray(g); S = E[g] @ E[g].T; vid = Fc.video.values[g]; other = vid[:, None] != vid[None, :]
    keep[g] = ((S >= thr) & other).any(1)
Fc["keep"] = keep; kept = Fc[Fc.keep]
log(f"identity filter: cross-speaker FaRL cos 99th pct {thr:.3f}; kept {keep.sum()}/{len(keep)} crops, speakers with >= 2 kept crops "
    f"{(kept.groupby('identity').size() >= 2).sum()}")
good = set(kept.groupby("identity").size().loc[lambda s: s >= 2].index)
kept = kept[kept.identity.isin(good)]; A = A[A.identity.isin(good)].reset_index(drop=True)
rows = []; pos = {m: i for i, m in enumerate(Fc.img.values)}; ga = dict(list(A.groupby("identity")))
for spk, g in kept.groupby("identity"):
    a = ga[spk]
    for n in range(max(len(g), len(a))):
        f = g.iloc[n % len(g)]; v = a.iloc[n % len(a)]
        rows.append(dict(identity=spk, gender="u", language="?", video=v.video, clip_index=int(v["clip"]),
                         voice=f"{spk}/{v.video}/{v.utt}.wav", face=f"{spk}/{f.video}/{f.frame}_{f.face}.jpg", fi=pos[f.img]))
rows = pd.DataFrame(rows); log(f"rows {len(rows)}, identities {rows.identity.nunique()}, clips {rows.clip_index.nunique()}")
def save(tag, arr):
    np.save(OUT / f"vb2_{tag}.float32.npy", np.asarray(arr, np.float32)); np.save(OUT / f"vb2_{tag}.ok.npy", np.ones(len(arr), bool)); log(f"saved {tag} {arr.shape}")
for e in ("farl", "clip"): save(f"face_{e}", EF[e][rows.fi.values])
clips = np.array(sorted(rows.clip_index.unique())); cpos = {c: j for j, c in enumerate(clips)}; ridx = rows.clip_index.map(cpos).values
wavs = [str(W / "wav" / f"{c}.wav") for c in clips]
for v in ("simam", "sam100ft", "r293", "ecapa"):
    e = X.voice_encoder(v)(wavs, log=log); save(f"voice_{v}", np.asarray(e)[ridx]); torch.cuda.empty_cache()
rows.drop(columns=["fi"]).to_csv(OUT / "rows_vb2.csv", index=False); Fc.to_csv(OUT / "vb2_faces_filtered.csv", index=False)
with zipfile.ZipFile("/kaggle/working/t3_ext_vb2.zip", "w", zipfile.ZIP_STORED) as z:
    for p in sorted(OUT.iterdir()): z.write(p, p.name)
shutil.rmtree(OUT); log("done")
'''


def nb(cells, title):
    out = [{"cell_type": "markdown", "metadata": {}, "source": f"# {title}"}]
    out += [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s} for s in cells]
    return {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}


def write(dirname, notebook, meta):
    d = HERE / dirname; d.mkdir(exist_ok=True); json.dump(notebook, open(d / f"{dirname}.ipynb", "w"), indent=1)
    json.dump(dict(dict(code_file=f"{dirname}.ipynb", language="python", kernel_type="notebook", is_private=True, enable_internet=True,
                        competition_sources=[], dataset_sources=[], kernel_sources=[]), **meta), open(d / "kernel-metadata.json", "w"), indent=1)
    print("built", d, meta["id"])


for code in (prep, enc_setup + enc): compile(code, "vb2", "exec")
write("t3_vb2prep", nb([prep], "FLAG Track 3 - VoxBlink2 prep (CPU)"),
      dict(id="trncnguyn/flag-t3-vb2-prep", title="FLAG T3 VB2 Prep", enable_gpu=False, dataset_sources=["trncnguyn/flag-t3-vb2-meta"]))
write("t3_vb2enc", nb([enc_setup, enc], "FLAG Track 3 - VoxBlink2 encode"),
      dict(id="trncnguyn/flag-t3-vb2-enc", title="FLAG T3 VB2 Enc", enable_gpu=True, dataset_sources=["trncnguyn/flag-t3-tools"],
           kernel_sources=["trncnguyn/flag-t3-vb2-prep"]))
