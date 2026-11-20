"""VoxBlink2 step 2 (runs on the user's machine): download each queued speaker's videos from YouTube (official VoxBlink2
procedure: yt-dlp, from t=0, reference format = best video <= 720p, whose coordinates the official face tracks use) at about
half the reference resolution to save bandwidth (face-track boxes rescaled), until 2 videos per speaker succeed.
Per video: the longest official utterance (centred window <= 8 s) -> 16 kHz mono wav + 2 face crops (official box,
FLAG framing) -> caches/vb2/media/{wav,img}; the video file is deleted immediately. Resumable; stops on YouTube bot checks.
Usage: python vb2_fetch.py <workers> [max_speakers]"""
import os, sys, time, pickle, random, subprocess, threading, csv
from pathlib import Path
SCRATCH = "/tmp/scratch/pylib"
sys.path.insert(0, SCRATCH)
import static_ffmpeg; static_ffmpeg.add_paths()
import shutil, numpy as np, pandas as pd, cv2, yt_dlp
from yt_dlp.utils import download_range_func
from concurrent.futures import ThreadPoolExecutor

D = Path(os.environ.get("VB2_DIR", Path(__file__).resolve().parents[1] / "caches/vb2")); M = D / "media"; TMP = D / "tmp"
for d in (M / "wav", M / "img", TMP): d.mkdir(parents=True, exist_ok=True)
WORKERS = int(sys.argv[1]); MAXS = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9
K, MAX_TRY, MAXF = 2, 5, 200
FF = shutil.which("ffmpeg")
tracks = pickle.load(open(D / "tracks.pkl", "rb"))
queue = pd.read_csv(D / "queue.csv", dtype=str).fillna("")
MAN = D / "fetch_manifest.csv"; COLS = ["identity", "video", "utt", "dur", "ok", "n_frames", "ref_h", "dl_h", "err"]
done = pd.read_csv(MAN, dtype=str) if MAN.exists() else pd.DataFrame(columns=COLS)
lock = threading.Lock(); stats = dict(ok=0, fail=0, bot=0, t0=time.time(), spk_done=0); STOP = threading.Event()
if not MAN.exists():
    with open(MAN, "w", newline="") as f: csv.writer(f).writerow(COLS)


def record(row):
    with lock:
        with open(MAN, "a", newline="") as f: csv.writer(f).writerow([row.get(c, "") for c in COLS])


def crop(fr, x, y, w, h):
    S = max(h, w / 0.76) / 0.72; cx, cy = x + w / 2, y + h / 2; left, top = cx - S / 2, cy - 0.38 * S
    Mx = np.float32([[224 / S, 0, -left * 224 / S], [0, 224 / S, -top * 224 / S]])
    return cv2.warpAffine(fr, Mx, (224, 224), flags=cv2.INTER_AREA if S > 224 else cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def fetch(spk, vid):
    dur, utt, tr = tracks[(spk, vid)]
    if len(tr) > MAXF: s0 = (len(tr) - MAXF) // 2; tr = tr[s0:s0 + MAXF]
    f0, f1 = int(tr[0, 0]), int(tr[-1, 0]); end = (f1 + 1) / 25.0 + 1.0
    base = dict(quiet=True, no_warnings=True, noplaylist=True, js_runtimes={"node": {}}, socket_timeout=30)
    if COOKIES:  # per-thread copy of the user-exported cookies.txt (yt-dlp rewrites its cookie file on exit)
        ck = TMP / f"cookies_{threading.get_ident()}.txt"
        if not ck.exists(): shutil.copy(COOKIES, ck); os.chmod(ck, 0o600)
        base["cookiefile"] = str(ck)
    with yt_dlp.YoutubeDL(base) as y:
        info = y.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False, process=False)
    fmts = [f for f in info.get("formats", []) if f.get("vcodec") not in (None, "none") and f.get("height")]
    ref = [f for f in fmts if f["height"] <= 720]
    if not ref: raise RuntimeError("no <=720 video format")
    ref_h = max(f["height"] for f in ref)
    small = sorted([f for f in ref if 0.45 * ref_h <= f["height"] <= ref_h and f.get("acodec") in (None, "none")],
                   key=lambda f: (f["height"], f.get("ext") != "mp4", f.get("tbr") or 0))
    vf = small[0]["format_id"] if small else "bestvideo[height<=720]"
    out = TMP / f"{spk}_{vid}"
    opts = dict(base, format=f"{vf}+bestaudio/best", outtmpl=str(out) + ".%(ext)s", merge_output_format="mkv",
                download_ranges=download_range_func(None, [(0, end)]))
    with yt_dlp.YoutubeDL(opts) as y:
        y.process_ie_result(info, download=True)
    files = [p for p in TMP.glob(f"{spk}_{vid}.*") if p.suffix in (".mkv", ".mp4", ".webm")]
    if not files: raise RuntimeError("no output file")
    p = files[0]
    try:
        cap = cv2.VideoCapture(str(p)); dl_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT); dl_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        sc = max(dl_h, dl_w) / max(ref_h, max(f.get("width") or 0 for f in ref if f["height"] == ref_h) or ref_h)
        key = f"{spk}_{vid}"; nf = 0
        for k, q in enumerate((0.3, 0.7), 1):
            row = tr[min(len(tr) - 1, int(q * len(tr)))]
            cap.set(cv2.CAP_PROP_POS_MSEC, row[0] / 25.0 * 1000 + 1); ok, fr = cap.read()
            if ok and fr is not None:
                cv2.imwrite(str(M / "img" / f"{key}_{nf + 1}.jpg"), crop(fr, *(row[1:] * sc)), [cv2.IMWRITE_JPEG_QUALITY, 95]); nf += 1
        cap.release()
        r = subprocess.run([FF, "-v", "error", "-y", "-ss", f"{f0 / 25:.3f}", "-i", str(p), "-t", f"{(f1 + 1 - f0) / 25:.3f}",
                            "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(M / "wav" / f"{key}.wav")], capture_output=True, text=True)
        okw = r.returncode == 0 and (M / "wav" / f"{key}.wav").exists()
        return dict(identity=spk, video=vid, utt=utt, dur=dur, ok=int(okw and nf > 0), n_frames=nf, ref_h=ref_h, dl_h=int(dl_h),
                    err="" if okw else r.stderr[-120:])
    finally:
        for f in TMP.glob(f"{spk}_{vid}.*"): f.unlink(missing_ok=True)


MAX_H = float(os.environ.get("VB2_MAX_HOURS", "1e9"))
COOKIES = os.environ.get("VB2_COOKIES")  # optional path to a cookies.txt exported by the user
PAUSE = float(os.environ.get("VB2_SLEEP", "0"))  # optional per-video pause (s) per worker


def speaker(row):
    if STOP.is_set() or time.time() - stats["t0"] > MAX_H * 3600: return
    spk = row.identity; vids = [v for v in row.videos.split(",") if v]
    prev = done[done.identity == spk]; good = int((prev.ok == "1").sum()); tried = set(prev.video)
    rng = random.Random(hash(spk) % 2 ** 32)
    vids = sorted(vids, key=lambda v: (tracks[(spk, v)][0] < 3.0, rng.random()))
    for vid in vids:
        if good >= K or STOP.is_set() or len(tried) >= MAX_TRY: break
        if vid in tried: continue
        tried.add(vid)
        try:
            res = fetch(spk, vid)
        except Exception as e:
            msg = str(e).replace("\n", " ")[:160]
            res = dict(identity=spk, video=vid, ok=0, err=msg)
            if "confirm you" in msg.lower() or "429" in msg or "too many requests" in msg.lower():
                with lock:
                    stats["bot"] += 1; nb = stats["bot"]
                    if nb >= (60 if COOKIES else 15): STOP.set()
                time.sleep(600 if COOKIES else 60)  # back off 10 min when logged in
        record(res); good += int(res.get("ok", 0))
        if PAUSE: time.sleep(PAUSE * random.uniform(.5, 1.5))
        with lock:
            stats["ok" if res.get("ok") else "fail"] += 1; n = stats["ok"] + stats["fail"]
            if n % 200 == 0:
                el = time.time() - stats["t0"]
                print(f"{time.strftime('%H:%M')} videos {n} ok {stats['ok']} fail {stats['fail']} bot {stats['bot']} | "
                      f"{n / el * 3600:.0f} videos/h | speakers done {stats['spk_done']}", flush=True)
    with lock: stats["spk_done"] += 1


todo = [r for r in queue.itertuples(index=False)][:MAXS]
print(f"queue {len(todo)} speakers, already in manifest {done.identity.nunique()}", flush=True)
with ThreadPoolExecutor(WORKERS) as ex:
    list(ex.map(speaker, todo))
print(f"finished: ok {stats['ok']} fail {stats['fail']} bot {stats['bot']} stopped={STOP.is_set()}", flush=True)
for ck in TMP.glob("cookies_*.txt"): ck.unlink(missing_ok=True)
