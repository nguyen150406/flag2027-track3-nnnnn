"""SUB53 fix (2026-10-18 20:55): the Kaggle extraction kernel sat 'Queued' for 3 h while its signed HF URLs (~1 h) expired, so the audio
is fetched HERE (token stays on the Mac, URLs re-signed per request batch) and uploaded as a private Kaggle dataset; the kernel then
only needs a GPU. Writes <out>/avs_raw/<src>/<clip_index><ext> (original AVSpeech audio bytes, decoded in the kernel exactly like the
pipeline) and <out>/vox/<voice relpath> (VoxCeleb1 wavs). Resumable (skips files already written).
Usage: python t3_fetch_audio_local.py <out dir>"""
import json, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd, requests, pyarrow.parquet as pq
from t3_httprange import RangeFile
from t3_fetch_audio import vox1_member
from t3lib import T3
OUT = Path(sys.argv[1]); SP = Path("/tmp/scratch")
tok = (Path.home() / ".hf_lrs3_token").read_text().strip()
def sign(p):
    for k in range(5):
        try:
            r = requests.head(f"https://huggingface.co/datasets/{p}", allow_redirects=False, timeout=60, headers={"Authorization": f"Bearer {tok}"})
            if r.status_code in (301, 302, 307): return r.headers["Location"]
        except Exception: pass
        time.sleep(3 * (k + 1))
    raise RuntimeError(f"cannot sign {p}")
EXT = T3 / "caches/ext"; keys = {}
for src in ("avs", "avs2"):
    r = pd.read_csv(EXT / f"rows_{src}.csv").drop_duplicates("voice")
    for v, ci in zip(r.voice, r.clip_index): keys.setdefault(v.split("/")[-1][:-4], []).append((src, int(ci)))
def avs_job(f):
    done = OUT / "avs_raw/.done" / f.replace("/", "_")
    if done.exists(): return 0
    tab = pq.ParquetFile(RangeFile(sign("ProgramComputer/avspeech-visual-audio/resolve/main/" + f))).read(columns=["clip_id", "audio"]).to_pylist(); n = 0
    for r in tab:
        k = r["clip_id"].replace("/", "_")
        if k not in keys or not r["audio"] or not r["audio"].get("bytes"): continue
        ext = Path(r["audio"].get("path") or "a.m4a").suffix or ".m4a"
        for src, ci in keys[k]:
            p = OUT / "avs_raw" / src / f"{ci}{ext}"; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(r["audio"]["bytes"]); n += 1
    done.parent.mkdir(parents=True, exist_ok=True); done.touch(); return n
files = json.load(open(SP / "avs_shuffled_files.json"))[:200]; t = time.time(); got = 0
with ThreadPoolExecutor(8) as ex:
    for i, fu in enumerate(as_completed([ex.submit(avs_job, f) for f in files])):
        try: got += fu.result()
        except Exception as e: print("avs file failed:", repr(e)[:200], flush=True)
        if i % 20 == 0: print(f"avs {i + 1}/{len(files)} files, {got} clips, {(time.time() - t) / 60:.1f} min", flush=True)
print("AVS DONE", {s: len(list((OUT / "avs_raw" / s).glob("*"))) for s in ("avs", "avs2")}, flush=True)
m = pd.read_csv(EXT / "vox1_manifest.csv"); m = m[[not (OUT / "vox" / v).exists() for v in m.voice]]
URLS = {"t": 0, "u": None}; lock = threading.Lock(); tl = threading.local()
def urls():
    with lock:
        if time.time() - URLS["t"] > 1800:
            URLS["u"] = [sign("ProgramComputer/voxceleb/resolve/main/vox1/vox1_dev_wav.zip"), sign("ProgramComputer/voxceleb/resolve/main/vox1/vox1_test_wav.zip")]; URLS["t"] = time.time()
        return URLS["u"]
def vjob(r):
    if not hasattr(tl, "s"): tl.s = requests.Session()
    for k in range(4):
        try: return vox1_member(urls(), r, OUT, tl.s)
        except Exception as e:
            err = e; time.sleep(2 ** k)
    raise err
ok = 0; t = time.time()
with ThreadPoolExecutor(32) as ex:
    for i, fu in enumerate(as_completed([ex.submit(vjob, r) for r in m.itertuples()])):
        try: fu.result(); ok += 1
        except Exception as e: print("vox1 failed:", repr(e)[:150], flush=True)
        if i % 2000 == 0: print(f"vox1 {i}/{len(m)} ok {ok} {(time.time() - t) / 60:.1f} min", flush=True)
print("VOX1 DONE", len(list((OUT / "vox").rglob("*.wav"))), flush=True)
