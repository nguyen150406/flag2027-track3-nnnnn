"""Phase A of the clean retrain (user 2026-11-05: "lọc lại bộ data của chúng ta thật là kĩ", big run from the Saturday 07/11 reset):
3-model Bengali LID audit (scripts/t3_lid_audit.py: VoxLingua107, Whisper large-v3, MMS-LID-256, every 30-s window) of every external
clip our training rows use, plus FLAG train as the NEGATIVE CONTROL (known English, South-Asian speakers: calibrates false flags).
Two kernels (two accounts, one big kernel each):
  part vox : FLAG train 6,485 (flag-2027-data) + LRS3 7,336 (wavs in the files dataset) + VoxCeleb1 12,423 (HF zips, range reads)
             + VoxCeleb2 ~29.5k (HF mp4 zips, range reads; same fetch code as the whs kernel)
  part web : VoxBlink ~29k (vb_media.zip, trncnguyn/flag-t3-voxblink-prep output) + AVSpeech avs 15k + avs2 40k (gated parquet audio)
Identities already in caches/ext/exclude_bengali_strict.csv are skipped (they stay excluded). Smoke on 3 clips per source first (all
three models, fails fast), then two shards on the two T4s. Output: lid_<part>.zip (lid_*.csv + clips_<part>.csv + fetch.json).
Signed URLs expire ~1 h: push right after building. Merge locally: scripts/t3_lid_merge.py.
Usage: python build_t3_lidaudit_kernel.py <user> vox|web [--stage]   (--stage = (re)build the files dataset folder kaggle/lid_files_<user>/)"""
import json, random, shutil, sys, zipfile
from pathlib import Path
import requests
K = Path(__file__).parent; T3 = K.parent; EXT = T3 / "caches/ext"
USER, PART = sys.argv[1], sys.argv[2]; assert PART in ("vox", "web")
TOK = (Path.home() / ".hf_lrs3_token").read_text().strip()


def sign(p):
    r = requests.head(f"https://huggingface.co/datasets/{p}", allow_redirects=False, timeout=60, headers={"Authorization": f"Bearer {TOK}"})
    assert r.status_code in (301, 302, 307), (p, r.status_code); return r.headers["Location"]


def stage():  # small private dataset: current audit code + row tables + manifests + strict list (+ LRS3 wavs for part vox)
    d = K / f"lid_files_{USER}"; d.mkdir(exist_ok=True)
    for f in ("t3_lid_audit.py", "t3_fetch_audio.py", "t3_httprange.py"): shutil.copy(T3 / "scripts" / f, d / f)
    for s in ("vox", "vox2", "vb", "lrs3", "avs", "avs2"): shutil.copy(EXT / f"rows_{s}.csv", d / f"rows_{s}.csv")
    for f in ("vox1_manifest.csv", "vox2_manifest.csv", "exclude_bengali_strict.csv"): shutil.copy(EXT / f, d / f)
    z = d / "lrs3_wav.zip"
    if not z.exists():
        with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as f:
            for p in sorted((T3 / "caches/lrs3_media/wav_x/wav").glob("*.wav")): f.write(p, f"lrs3_wav/{p.name}")
    json.dump({"title": "flag-t3-lid-files", "id": f"{USER}/flag-t3-lid-files", "licenses": [{"name": "other"}]}, open(d / "dataset-metadata.json", "w"))
    print("staged", d, "-> create: kaggle datasets create -p", d, "--dir-mode zip   (or: version -m update)")


def vb_link():  # fresh link of the VoxBlink media zip (trncnguyn's kernel output, default ~/.kaggle token)
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    api = KaggleApi(); api.authenticate()
    with api.build_kaggle_client() as k:
        r = ApiListKernelSessionOutputRequest(); r.user_name = "trncnguyn"; r.kernel_slug = "flag-t3-voxblink-prep"
        out = [f.url for f in k.kernels.kernels_api_client.list_kernel_session_output(r).files if f.file_name == "vb_media.zip"]
    assert len(out) == 1, out; return out[0]


def avs_files():  # the seed-2027 shuffle of the mirror's train parquet files (colab/avs/pipeline.py); [:200] = avs pilot + avs2 chunks
    from huggingface_hub import HfApi
    f = sorted(x for x in HfApi(token=TOK).list_repo_files("ProgramComputer/avspeech-visual-audio", repo_type="dataset")
               if x.startswith("train/") and x.endswith(".parquet"))
    random.Random(2027).shuffle(f); return f[:200]


src = (K / "build_t3_sub35_kernel.py").read_text(); VOX2 = src[src.index("VOX2 = r'''") + 11:src.index("'''\nhead = head[:dl0]")]
VOX2 = "".join("    " + l if l.strip() else l for l in VOX2.splitlines(True))  # one level deeper (inside if PART == "vox")
assert "exclude_bengali.csv" in VOX2 and "__HFURLS__" in VOX2
CODE = r'''
import os, sys, json, time, glob, shutil, subprocess, zipfile, traceback, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
T0 = time.time()
def log(m): print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 60:6.1f}m]", m, flush=True)
PART = "__PART__"; VOX1 = __VOX1__; AVS = __AVS__; VB = __VB__
TMP = Path("/kaggle/temp"); DL, AUD, OUT = TMP / "dl", TMP / "aud", Path("/kaggle/working"); LG = OUT / "logs"
T3 = TMP / "t3"; SC = T3 / "scripts"; EXT = T3 / "caches/ext"
for d in (DL, AUD, LG, SC, EXT): d.mkdir(parents=True, exist_ok=True)
def sh(cmd, name):
    log(f"$ {cmd[:220]}")
    r = subprocess.run(["bash", "-c", f"set -o pipefail; {cmd} 2>&1 | tee -a {LG / name} | (grep --line-buffered -v -i -E 'warn|stft|_VF' || true)"])
    if r.returncode: raise RuntimeError(f"exit {r.returncode}: {cmd[:150]}")
try:
    import torch; ng = torch.cuda.device_count(); log(f"GPUs {ng}: {[torch.cuda.get_device_name(i) for i in range(ng)]}; cpus {os.cpu_count()}")
    assert ng >= 1, "no GPU"
    FILES = [Path(p).parent for p in glob.glob("/kaggle/input/**/t3_lid_audit.py", recursive=True) if "flag-t3-lid-files" in p]; assert FILES, "missing flag-t3-lid-files"
    F = FILES[0]
    for f in ("t3_lid_audit.py", "t3_fetch_audio.py", "t3_httprange.py"): shutil.copy(F / f, SC / f)
    for f in F.glob("*.csv"): shutil.copy(f, EXT / f.name)
    shutil.copy(EXT / "exclude_bengali_strict.csv", EXT / "exclude_bengali.csv")  # the reused VoxCeleb2 fetch skips the excluded speakers
    sh("pip install -q speechbrain imageio-ffmpeg 2>&1 | tail -1", "setup.log"); import imageio_ffmpeg, pandas as pd; FF = imageio_ffmpeg.get_ffmpeg_exe()
    sys.path.insert(0, str(SC)); from t3_fetch_audio import avs_file, vox1_member
    exc = pd.read_csv(EXT / "exclude_bengali_strict.csv"); EXC = {s: set(exc.identity[exc.src.eq(s)].astype(str)) for s in exc.src.unique()}
    def src_rows(s):  # one row per clip of rows_<s>.csv, identities already excluded skipped
        r = pd.read_csv(EXT / f"rows_{s}.csv").drop_duplicates("voice"); return r[~r.identity.astype(str).isin(EXC.get(s, set()))]
    L, FETCH = [], {}
    log("STAGE setup done")
    if PART == "vox":
        tr = sorted(glob.glob("/kaggle/input/**/train_set/train_set/voices/English/*/*.wav", recursive=True)); assert len(tr) > 6000, len(tr)
        L += [("flag", "flag:" + "/".join(p.split("/")[-2:]), p, p.split("/")[-2]) for p in tr]; FETCH["flag"] = [len(tr), len(tr)]
        z = glob.glob("/kaggle/input/**/lrs3_wav.zip", recursive=True)
        if z:
            with zipfile.ZipFile(z[0]) as f: f.extractall(AUD)
        else:  # Kaggle may have auto-extracted the zip (possibly nested): link the folder that holds the wavs
            w = glob.glob("/kaggle/input/**/lrs3_wav/*_s1.wav", recursive=True); assert w, "missing LRS3 wavs"; os.symlink(Path(w[0]).parent, AUD / "lrs3_wav")
        r = src_rows("lrs3"); ps = [AUD / "lrs3_wav" / f"{v.split('/')[1]}_{Path(v).stem}.wav" for v in r.voice]
        L += [("lrs3", "lrs3:" + v, str(p), i) for v, p, i in zip(r.voice, ps, r.identity)]; FETCH["lrs3"] = [len(r), sum(p.exists() for p in ps)]
        m = pd.read_csv(EXT / "vox1_manifest.csv"); r = src_rows("vox"); m = m[m.voice.isin(set(r.voice))]; tl = threading.local(); t = time.time()
        def vjob(x):
            if not hasattr(tl, "s"): tl.s = requests.Session()
            for k in range(4):
                try: return vox1_member(VOX1, x, AUD, tl.s)
                except Exception as e:
                    err = e; time.sleep(2 ** k)
            raise err
        ok = 0
        with ThreadPoolExecutor(32) as ex:
            for i, fu in enumerate(as_completed([ex.submit(vjob, x) for x in m.itertuples()])):
                try: fu.result(); ok += 1
                except Exception as e:
                    if i - ok < 5: log(f"vox1 member failed: {e!r}"[:200])
        log(f"STAGE vox1 audio: {ok}/{len(m)} in {(time.time() - t) / 60:.1f} min")
        L += [("vox", "vox:" + v, str(AUD / "vox" / v), i) for v, i in zip(r.voice, r.identity)]; FETCH["vox"] = [len(r), ok]
__VOX2__        r = src_rows("vox2")
        L += [("vox2", "vox2:" + v, str(AUD / "vox2" / f"{ci}.wav"), i) for v, ci, i in zip(r.voice, r.clip_index, r.identity)]; FETCH["vox2"] = [len(r), okc]
    else:
        t = time.time(); u = VB
        sz = int(requests.get(u, headers={"Range": "bytes=0-0"}, timeout=120).headers["Content-Range"].split("/")[1])
        with open(DL / "vb_media.zip", "wb") as f: f.truncate(sz)
        def part(a):  # 64 MB range reads with retries (a single 7 GB stream broke at 5.8 GB on 07/11)
            b = min(a + (64 << 20), sz) - 1
            for k in range(6):
                try:
                    d = requests.get(u, headers={"Range": f"bytes={a}-{b}"}, timeout=600).content; assert len(d) == b - a + 1
                    with open(DL / "vb_media.zip", "r+b") as f: f.seek(a); f.write(d)
                    return
                except Exception: time.sleep(5 * (k + 1))
            raise RuntimeError(f"vb range {a}-{b} failed")
        with ThreadPoolExecutor(8) as ex: list(ex.map(part, range(0, sz, 64 << 20)))
        with zipfile.ZipFile(DL / "vb_media.zip") as z: z.extractall(TMP / "vbm", [n for n in z.namelist() if n.startswith("wav/")])
        shutil.move(str(TMP / "vbm/wav"), str(AUD / "vb")); (DL / "vb_media.zip").unlink(); log(f"STAGE vb audio in {(time.time() - t) / 60:.1f} min")
        r = src_rows("vb"); ps = [AUD / "vb" / f"{ci}.wav" for ci in r.clip_index]
        L += [("vb", "vb:" + v, str(p), i) for v, p, i in zip(r.voice, ps, r.identity)]; FETCH["vb"] = [len(r), sum(p.exists() for p in ps)]
        keys = {}; R = {s: src_rows(s) for s in ("avs", "avs2")}
        for s, r in R.items():
            for v, ci in zip(r.voice, r.clip_index): keys.setdefault(v.split("/")[-1][:-4], []).append((s, int(ci)))
        t = time.time(); got = 0
        with ThreadPoolExecutor(8) as ex:
            futs = {ex.submit(avs_file, u, keys, AUD, FF): f for f, u in AVS.items()}
            for i, fu in enumerate(as_completed(futs)):
                try: got += fu.result()[0]
                except Exception as e: log(f"avs file failed {futs[fu]}: {e!r}"[:300])
                if i % 25 == 0: log(f"avs files {i + 1}/{len(AVS)}: {got} wavs ({(time.time() - t) / 60:.1f} min)")
        for s, r in R.items():
            ps = [AUD / s / f"{ci}.wav" for ci in r.clip_index]
            L += [(s, f"{s}:" + v, str(p), i) for v, p, i in zip(r.voice, ps, r.identity)]; FETCH[s] = [len(r), sum(p.exists() for p in ps)]
    log(f"STAGE audio done: clips needed / fetched {FETCH}")
    C = pd.DataFrame(L, columns=["src", "key", "path", "identity"]); C.to_csv(OUT / f"clips_{PART}.csv", index=False)
    json.dump(FETCH, open(OUT / "fetch.json", "w"))
    X = f"python -X faulthandler -u t3_lid_audit.py --list {OUT}/clips_{PART}.csv --device cuda --workers 4"
    t = time.time(); sh(f"cd '{SC}' && CUDA_VISIBLE_DEVICES=0 {X} --limit 3 --out {TMP}/smoke", "smoke.log")
    log(f"STAGE smoke done in {(time.time() - t) / 60:.1f} min (3 clips per source, 3 models incl. load)"); sh(f"cat {TMP}/smoke/lid_0of1.csv | head -30", "smoke.log")
    n = min(ng, 2)
    procs = [subprocess.Popen(["bash", "-c", f"cd '{SC}' && CUDA_VISIBLE_DEVICES={g} timeout -s ABRT 30000 {X} --shard {g}/{n} --out {OUT}/lid > {LG}/lid{g}.log 2>&1"]) for g in range(n)]
    while any(p.poll() is None for p in procs):
        time.sleep(600)
        for g in range(n): log(f"lid{g}: " + ((LG / f"lid{g}.log").read_text().strip().splitlines() or ["(no output yet)"])[-1][-160:])
    assert all(p.returncode == 0 for p in procs), [p.returncode for p in procs]
    with zipfile.ZipFile(OUT / f"lid_{PART}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in list((OUT / "lid").glob("*.csv")) + [OUT / f"clips_{PART}.csv", OUT / "fetch.json"]: z.write(p, p.name)
    log(f"JOB DONE -> {OUT / f'lid_{PART}.zip'}")
except Exception as e:
    log(f"VMERR {e!r}"); traceback.print_exc()
    for p in LG.glob("*.log"): print(f"----- tail {p.name}"); print(p.read_text()[-2500:])
    raise
'''

if "--stage" in sys.argv: stage()
if PART == "vox":
    vox1 = [sign("ProgramComputer/voxceleb/resolve/main/vox1/vox1_dev_wav.zip"), sign("ProgramComputer/voxceleb/resolve/main/vox1/vox1_test_wav.zip")]
    hfv2 = [sign(f"ProgramComputer/voxceleb/resolve/main/vox2/vox2_mp4_{i}.zip") for i in range(1, 7)]
    code = CODE.replace("__VOX2__", VOX2.replace("__HFURLS__", json.dumps(hfv2))).replace("__VOX1__", json.dumps(vox1)).replace("__AVS__", "{}").replace("__VB__", "''")
else:
    code = CODE.replace("__VOX2__", "").replace("__VOX1__", "[]").replace("__AVS__", json.dumps({f: sign("ProgramComputer/avspeech-visual-audio/resolve/main/" + f) for f in avs_files()})).replace("__VB__", json.dumps(vb_link()))
code = code.replace("__PART__", PART); compile(code, "lid", "exec")
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
d = K / f"t3_lid_{PART}"; d.mkdir(exist_ok=True); json.dump(nb, open(d / f"t3_lid_{PART}.ipynb", "w"), indent=1)
json.dump({"id": f"{USER}/flag-t3-lid-{PART}", "title": f"FLAG T3 lid {PART}", "code_file": f"t3_lid_{PART}.ipynb", "language": "python", "kernel_type": "notebook",
           "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
           "dataset_sources": [f"{USER}/flag-t3-lid-files"] + (["trncnguyn/flag-2027-data"] if PART == "vox" else []), "competition_sources": [], "kernel_sources": []},
          open(d / "kernel-metadata.json", "w"), indent=1)
print("built", d, "(push within ~1 h: signed URLs expire)")
