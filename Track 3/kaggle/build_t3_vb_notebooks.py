"""VoxBlink (clean subset, 18k YouTube speakers) as new external identities for the Track 3 projector.
Two CPU 'prep' kernels (one per account; identical deterministic selection) + two GPU 'enc' kernels that mount them:
  prep : HF mirror adollahamini1998/VoxBlink-Video-clean-8-* (tar shards, full videos) -> tar header walk by HTTP range,
         K=2 videos per speaker (Bengali speakers excluded via official spk2lan), longest clean utterance per video,
         audio cut to the official utterance timestamps, 2 face frames cropped with the official face-track bbox
         (framing calibrated to FLAG crops: face box h = 0.72 of the crop, face centre at 0.38 height).
  enc  : frozen FLAG Track 3 encoders on the prepared media -> vb_* arrays + rows_vb.csv (VoxCeleb2 batch format).
Writes kaggle/t3_vbprep_{main,team}/ and kaggle/t3_vbenc_{team,main}/."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
K_VIDEOS = 2

prep_setup = r'''
import os, sys, subprocess, time, shutil, zipfile, json, io, tarfile, threading, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd, requests, cv2
def run(c): print("+", c); subprocess.run(c, shell=True, check=True)
run("pip install -q imageio-ffmpeg 2>&1 | tail -1")
T0 = time.time()
def log(m): print(time.strftime("%H:%M:%S"), f"[{(time.time()-T0)/60:5.1f}m]", m, flush=True)
W = Path("/kaggle/temp/vb"); (W / "wav").mkdir(parents=True, exist_ok=True); (W / "img").mkdir(parents=True, exist_ok=True)
OUT = Path("/kaggle/working"); META = W / "meta"; META.mkdir(exist_ok=True)
FFMPEG = shutil.which("ffmpeg")
if FFMPEG is None:
    import imageio_ffmpeg; FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
HF = "https://huggingface.co/datasets/adollahamini1998"
def get(url, path, tries=6):
    for k in range(tries):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for ch in r.iter_content(1 << 22): f.write(ch)
            return path
        except Exception as e:
            log(f"retry {url}: {e!r}"[:200]); time.sleep(2 ** k)
    raise RuntimeError(url)
log(f"ffmpeg {FFMPEG} cv2 {cv2.__version__}")
'''

prep_select = r'''
# ---------------------------------------------------------------- official VoxBlink metadata + mirror progress -> selection
R = HF + "/ScriptsForVoxBlinkResources/resolve/main/"
for f in ("meta/spk2gender", "meta/spk2lan", "meta/utt2dur", "data/utt_clean.txt", "timestamp.tar.gz"):
    get(R + f, META / Path(f).name)
for p in range(8):
    get(f"{HF}/VoxBlinkProgressVideos/resolve/main/progress/download_progress_clean_8_{p}.json", META / f"prog{p}.json")
log("metadata downloaded")
lan = {l.split()[0]: l.split()[1] for l in open(META / "spk2lan") if len(l.split()) >= 2}
gen = {l.split()[0]: l.split()[1][0].lower() for l in open(META / "spk2gender") if len(l.split()) >= 2}
dur = {l.split()[0]: float(l.split()[1]) for l in open(META / "utt2dur") if len(l.split()) == 2}
best = {}                                   # (spk, video) -> (dur, utt) of the longest clean utterance
for l in open(META / "utt_clean.txt"):
    u = l.strip()
    if not u: continue
    spk, vid, utt = u[:7], u[8:-6], u[-5:]
    d = dur.get(u, 0.0)
    if d > best.get((spk, vid), (-1, ""))[0]: best[(spk, vid)] = (d, utt)
part_of, uploaded = {}, {}
for p in range(8):
    for spk, x in json.load(open(META / f"prog{p}.json"))["completed"].items():
        part_of[spk] = p; uploaded[spk] = sorted(v for v, s in x["videos"].items() if s["status"] == "uploaded")
rng = np.random.default_rng(2027); sel = []
excluded_bn = 0
for spk in sorted(uploaded):
    if lan.get(spk, "").lower() == "bengali": excluded_bn += 1; continue
    cand = [v for v in uploaded[spk] if (spk, v) in best]
    if not cand: continue
    good = [v for v in cand if best[(spk, v)][0] >= 3.0]
    pool = good if len(good) >= K else cand
    for v in rng.choice(pool, min(K, len(pool)), replace=False):
        d, utt = best[(spk, v)]; sel.append((spk, gen.get(spk, "u"), lan.get(spk, "?"), part_of[spk], v, utt, d))
man = pd.DataFrame(sel, columns=["identity", "gender", "language", "part", "video", "utt", "dur"])
log(f"selected {len(man)} clips, {man.identity.nunique()} speakers (excluded Bengali speakers: {excluded_bn}); "
    f"gender {man.drop_duplicates('identity').gender.value_counts().to_dict()}; mean dur {man.dur.mean():.2f}s")
'''.replace("len(good) >= K", f"len(good) >= {K_VIDEOS}").replace("min(K,", f"min({K_VIDEOS},")

prep_tracks = r'''
# ---------------------------------------------------------------- official face tracks for the selected utterances
want = {(r.identity, r.video, r.utt) for r in man.itertuples(index=False)}; tracks = {}
with tarfile.open(META / "timestamp.tar.gz", "r|gz") as t:
    for m in t:
        if not m.isfile(): continue
        p = m.name.split("/")
        key = (p[-3], p[-2], p[-1][:-4])
        if key in want:
            L = [l.split() for l in t.extractfile(m).read().decode().splitlines()]
            tracks[key] = np.array([[float(v) for v in l] for l in L if len(l) == 5 and l[0].isdigit()])
man["has_track"] = [(r.identity, r.video, r.utt) in tracks for r in man.itertuples(index=False)]
log(f"tracks found {man.has_track.sum()}/{len(man)}")
'''

prep_walk = r'''
# ---------------------------------------------------------------- tar header walk (1 range request / header) on all mirror shards
shards = []
for p in range(8):
    d = requests.get(f"https://huggingface.co/api/datasets/adollahamini1998/VoxBlink-Video-clean-8-{p}?blobs=true", timeout=60).json()
    shards += [(p, x["rfilename"], x["size"]) for x in d["siblings"] if x["rfilename"].endswith(".tar")]
log(f"{len(shards)} shards, {sum(s[2] for s in shards) / 1e9:.0f} GB")
wantv = {(r.identity, r.video) for r in man.itertuples(index=False) if r.has_track}
tls = threading.local()
def session():
    if not hasattr(tls, "s"): tls.s = requests.Session()
    return tls.s
URLS, url_lock = {}, threading.Lock()
def cdn(p, name, refresh=False):
    with url_lock:
        if not refresh and (p, name) in URLS: return URLS[(p, name)]
    for k in range(8):
        try:
            r = session().get(f"{HF}/VoxBlink-Video-clean-8-{p}/resolve/main/{name}", headers={"Range": "bytes=0-0"}, timeout=60)
            if r.status_code in (200, 206):
                with url_lock: URLS[(p, name)] = r.url
                return r.url
        except Exception:
            pass
        time.sleep(min(60, 2 ** k))
    raise IOError(f"resolve {p}/{name}")
def rget(state, a, b):
    for k in range(6):
        try:
            r = session().get(state["url"], headers={"Range": f"bytes={a}-{b}"}, timeout=120)
            if r.status_code in (403, 404, 410): raise IOError(r.status_code)
            r.raise_for_status(); return r.content
        except Exception as e:
            time.sleep(min(30, 2 ** k)); state["url"] = cdn(*state["key"], refresh=True)
    raise IOError(f"range {state['key']} {a}")
def walk(sh):
    p, name, size = sh; st = {"key": (p, name)}; st["url"] = cdn(p, name); off = 0; hits = []; n = 0
    while off + 512 <= size:
        h = rget(st, off, off + 511)
        nm = h[:100].split(b"\0")[0].decode(errors="replace")
        if not nm: break
        sz = int(h[124:136].split(b"\0")[0].strip() or b"0", 8); typ = h[156:157]
        pre = h[345:500].split(b"\0")[0].decode(errors="replace") if h[257:262] == b"ustar" else ""
        full = (pre + "/" + nm) if pre else nm
        if typ in (b"0", b"\0") and full.endswith(".mp4"):
            n += 1; q = full.split("/")
            if len(q) >= 2 and (q[-2], q[-1][:-4]) in wantv: hits.append((q[-2], q[-1][:-4], p, name, off + 512, sz))
        off += 512 + ((sz + 511) // 512) * 512
    return hits, n
loc = {}; nmem = 0
with ThreadPoolExecutor(64) as ex:
    futs = [ex.submit(walk, sh) for sh in shards]
    for j, fu in enumerate(as_completed(futs)):
        try:
            hits, n = fu.result(); nmem += n
            for spk, v, p, name, o, sz in hits: loc[(spk, v)] = (p, name, o, sz)
        except Exception as e:
            log(f"walk failed: {e!r}"[:200])
        if j % 50 == 0: log(f"walked {j + 1}/{len(shards)} shards, {nmem} videos, located {len(loc)}/{len(wantv)}")
log(f"walk done: {nmem} videos in mirror, located {len(loc)}/{len(wantv)} selected")
'''

prep_fetch = r'''
# ---------------------------------------------------------------- fetch selected members, cut utterance audio, crop 2 face frames
FPS = 25.0; MAXF = 200
def crop(fr, x, y, w, h):
    S = max(h, w / 0.76) / 0.72                                      # FLAG framing: face box h = 0.72 of the crop (w/h = 0.76)
    cx, cy = x + w / 2, y + h / 2; left, top = cx - S / 2, cy - 0.38 * S
    M = np.float32([[224 / S, 0, -left * 224 / S], [0, 224 / S, -top * 224 / S]])
    return cv2.warpAffine(fr, M, (224, 224), flags=cv2.INTER_AREA if S > 224 else cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
REC = list(man.itertuples(index=False))
cache_url = {}
def work(i):
    r = REC[i]; wav = W / "wav" / f"{i}.wav"
    if (W / "img" / f"{i}_2.jpg").exists() and wav.exists(): return i, 2, ""
    if not r.has_track or (r.identity, r.video) not in loc: return i, 0, "not located"
    p, name, o, sz = loc[(r.identity, r.video)]; st = {"key": (p, name)}; st["url"] = cdn(p, name)
    tmp = W / f"tmp_{i}.mp4"
    try:
        tmp.write_bytes(rget(st, o, o + sz - 1))
        tr = tracks[(r.identity, r.video, r.utt)]
        if len(tr) > MAXF: s0 = (len(tr) - MAXF) // 2; tr = tr[s0:s0 + MAXF]     # centred 8 s window (FLAG voices ~6 s)
        f0, f1 = tr[0, 0], tr[-1, 0]
        cmd = [FFMPEG, "-v", "error", "-y", "-ss", f"{f0 / FPS:.3f}", "-i", str(tmp), "-t", f"{(f1 + 1 - f0) / FPS:.3f}",
               "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(wav)]
        pr = subprocess.run(cmd, capture_output=True, text=True)
        if pr.returncode != 0 or not wav.exists(): return i, 0, pr.stderr[-160:]
        cap = cv2.VideoCapture(str(tmp)); nf = 0
        for k, q in enumerate((0.3, 0.7), 1):
            row = tr[min(len(tr) - 1, int(q * len(tr)))]
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(row[0])); ok, fr = cap.read()
            if ok and fr is not None:
                cv2.imwrite(str(W / "img" / f"{i}_{nf + 1}.jpg"), crop(fr, *row[1:]), [cv2.IMWRITE_JPEG_QUALITY, 95]); nf += 1
        cap.release()
        return i, nf, ""
    except Exception as e:
        return i, 0, repr(e)[:160]
    finally:
        if tmp.exists(): tmp.unlink()
nfr = np.zeros(len(man), int); errs = []
with ThreadPoolExecutor(24) as ex:
    futs = [ex.submit(work, i) for i in range(len(man))]
    for n, fu in enumerate(as_completed(futs)):
        i, k, msg = fu.result(); nfr[i] = k
        if msg and msg != "not located":
            errs.append((i, msg))
            if len(errs) <= 5: log(f"error {i}: {msg}")
        if n % 2000 == 0: log(f"processed {n}/{len(man)} ok {int((nfr > 0).sum())} errors {len(errs)}")
man["n_frames"] = nfr; man["ok"] = nfr > 0
log(f"media done: {int(man.ok.sum())}/{len(man)} clips, {man[man.ok].identity.nunique()} speakers, {int(nfr.sum())} face frames; errors {len(errs)}")
'''

prep_final = r'''
man.to_csv(OUT / "vb_manifest.csv", index=False)
json.dump(dict(source="VoxBlink clean subset (official metadata/timestamps: HF adollahamini1998/ScriptsForVoxBlinkResources; videos: "
                      "HF mirror adollahamini1998/VoxBlink-Video-clean-8-*)", videos_per_speaker=K_VIDEOS_, bengali_speakers_excluded=True,
               audio="official clean-utterance timestamps, centred window <= 8 s", faces="official face-track bbox, FLAG-calibrated framing",
               flag_data_used=False), open(OUT / "PROVENANCE_VB.json", "w"))
with zipfile.ZipFile(OUT / "vb_media.zip", "w", zipfile.ZIP_STORED) as z:
    for i in np.flatnonzero(man.ok.values):
        z.write(W / "wav" / f"{i}.wav", f"wav/{i}.wav")
        for k in range(1, man.n_frames.iat[i] + 1): z.write(W / "img" / f"{i}_{k}.jpg", f"img/{i}_{k}.jpg")
log(f"zip {os.path.getsize(OUT / 'vb_media.zip') / 1e9:.2f} GB; done")
'''.replace("K_VIDEOS_", str(K_VIDEOS))

enc_setup_tmpl = None  # filled from the VoxCeleb2 batch-B notebook setup cell (same frozen encoders)
B = json.load(open(HERE / "t3_vox2b/t3_vox2b.ipynb"))
enc_setup = [c["source"] for c in B["cells"] if c["cell_type"] == "code"][0]
enc_setup = enc_setup.replace('W = Path("/kaggle/temp/vox2")', 'W = Path("/kaggle/temp/vb")').replace('OUT = Path("/kaggle/working/vox2_out")', 'OUT = Path("/kaggle/working/vb_out")')
assert "/kaggle/temp/vb" in enc_setup and "vb_out" in enc_setup

enc_body = r'''
# ---------------------------------------------------------------- unpack prepared media, rows (one per face frame), frozen encoders
SRC = find("vb_manifest.csv")[0].parent; man = pd.read_csv(SRC / "vb_manifest.csv", dtype={"utt": str, "video": str})
if (SRC / "vb_media.zip").exists():
    with zipfile.ZipFile(SRC / "vb_media.zip") as z: z.extractall(W)
else:                                               # input mounted already unpacked
    media = [d for d in [SRC, SRC / "vb_media"] if (d / "wav").exists()][0]
    for sub in ("wav", "img"):
        shutil.rmtree(W / sub, ignore_errors=True); os.symlink(media / sub, W / sub)
log(f"media ready from {SRC}")
rows = []
for i in np.flatnonzero(man.ok.values):
    r = man.iloc[i]
    for k in range(1, int(r.n_frames) + 1):
        rows.append(dict(identity=r.identity, gender=r.gender, language=r.language, video=r.video, clip_index=int(i),
                         voice=f"{r.identity}/{r.video}/{r.utt}.wav", face=f"{r.identity}/{r.video}/{r.utt}_{k}.jpg",
                         face_path=str(W / "img" / f"{i}_{k}.jpg")))
rows = pd.DataFrame(rows); clips = np.array(sorted(rows.clip_index.unique()))
cpos = {c: j for j, c in enumerate(clips)}; ridx = rows.clip_index.map(cpos).values
log(f"rows {len(rows)} clips {len(clips)} identities {rows.identity.nunique()}")
import torch
def save(tag, arr):
    np.save(OUT / f"vb_{tag}.float32.npy", arr.astype(np.float32)); np.save(OUT / f"vb_{tag}.ok.npy", np.ones(len(arr), bool))
    log(f"saved {tag} {arr.shape}")
for enc in FACE_ENCS:
    save(f"face_{enc}", X.face_encoder(enc)(list(rows.face_path), log=log)); torch.cuda.empty_cache()
wavs = [str(W / "wav" / f"{c}.wav") for c in clips]
for enc in VOICE_ENCS:
    e = X.voice_encoder(enc)(wavs, log=log); save(f"voice_{enc}", e[ridx]); torch.cuda.empty_cache()
rows.drop(columns=["face_path"]).to_csv(OUT / "rows_vb.csv", index=False)
with zipfile.ZipFile(f"/kaggle/working/t3_vb_{TAG}.zip", "w", zipfile.ZIP_STORED) as z:
    for p in sorted(OUT.iterdir()): z.write(p, p.name)
shutil.rmtree(OUT); log("done")
'''


def nb(cells, title):
    out = [{"cell_type": "markdown", "metadata": {}, "source": f"# {title}"}]
    out += [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s} for s in cells]
    return {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 5}


def write(dirname, notebook, meta):
    d = HERE / dirname; d.mkdir(exist_ok=True)
    json.dump(notebook, open(d / f"{dirname}.ipynb", "w"), indent=1)
    meta = dict(dict(code_file=f"{dirname}.ipynb", language="python", kernel_type="notebook", is_private=True,
                     enable_internet=True, competition_sources=[], dataset_sources=[], kernel_sources=[]), **meta)
    json.dump(meta, open(d / "kernel-metadata.json", "w"), indent=1)
    print("built", d, meta["id"])


prep = nb([prep_setup, prep_select, prep_tracks, prep_walk, prep_fetch, prep_final], "FLAG Track 3 — VoxBlink prep (CPU)")
for acct, d in (("trncnguyn", "t3_vbprep_main"), ("KAGGLE_USER", "t3_vbprep_team")):
    write(d, prep, dict(id=f"{acct}/flag-t3-voxblink-prep", title="FLAG T3 VoxBlink Prep", enable_gpu=False))
for acct, d, tag, faces, voices in (("KAGGLE_USER", "t3_vbenc_team", "a", ["farl", "clip"], ["simam", "sam100ft"]),
                                    ("trncnguyn", "t3_vbenc_main", "b", [], ["ecapa", "r293"])):
    body = enc_body.replace("FACE_ENCS", repr(faces)).replace("VOICE_ENCS", repr(voices)).replace("{TAG}", tag)
    write(d, nb([enc_setup, body], f"FLAG Track 3 — VoxBlink encode {tag}"),
          dict(id=f"{acct}/flag-t3-voxblink-enc-{tag}", title=f"FLAG T3 VoxBlink Enc {tag.upper()}", enable_gpu=True,
               dataset_sources=[f"{acct}/flag-t3-tools"], kernel_sources=[f"{acct}/flag-t3-voxblink-prep"]))
