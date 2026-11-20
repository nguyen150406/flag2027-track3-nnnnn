"""VoxBlink2 step 1: choose speakers and keep the official face tracks of their best utterances.
Speakers: all VoxBlink2 speakers minus every speaker flagged by scripts/vb2_bn_filter.py (Whisper bn/as utterance,
Bengali-script transcript, YouTube bn/as language, Bengali script or Bengal keywords in channel/tags; 5,823 removed), shuffled (seed 2027); the first N_SPK form the download queue.
Streams the official vb2_meta.tar.gz (10.6 GB, Google Drive) once and keeps, per (speaker, video), the longest utterance
track (frame, x, y, w, h at 25 fps) -> caches/vb2/tracks.pkl + queue.csv."""
import sys, tarfile, pickle, random
from pathlib import Path
import numpy as np, pandas as pd, requests

T3 = Path(__file__).resolve().parents[1]; D = T3 / "caches/vb2"
N_SPK = int(sys.argv[1]) if len(sys.argv) > 1 else 25000
spk2vid = {l.split()[0]: l.split()[1:] for l in open(D / "spk2videos") if l.strip()}
bn = set(pd.read_csv(D / "exclude_bengali.csv").identity)  # strict multi-signal Bengali filter (scripts/vb2_bn_filter.py)
elig = sorted(s for s in spk2vid if s not in bn)
random.Random(2027).shuffle(elig); queue = elig[:N_SPK]; want = set(queue)
print(f"speakers {len(spk2vid)}, Bengali-flagged excluded {len(bn & set(spk2vid))}, eligible {len(elig)}, queued {len(queue)}", flush=True)

URL = "https://drive.usercontent.google.com/download?id=1oYpxR7Bn_wKj2Ma6dBx_n5dznBybfubc&export=download&confirm=t"
class Stream:  # file-like wrapper over a streamed HTTP body for tarfile 'r|gz'
    def __init__(self, url):
        self.r = requests.get(url, stream=True, timeout=120); self.r.raise_for_status(); self.it = self.r.raw; self.n = 0
    def read(self, k=-1):
        b = self.it.read(k if k and k > 0 else 1 << 20); self.n += len(b); return b
best, n_files = {}, 0
LOCAL = D / "vb2_meta.tar.gz"
class Local:
    def __init__(self, p): self.f = open(p, "rb"); self.n = 0
    def read(self, k=-1):
        b = self.f.read(k if k and k > 0 else 1 << 20); self.n += len(b); return b
s = Local(LOCAL) if (LOCAL.exists() and Path(str(LOCAL) + ".complete").exists()) else Stream(URL)
with tarfile.open(fileobj=s, mode="r|gz") as t:
    for m in t:
        if not m.isfile(): continue
        p = m.name.split("/")
        if len(p) < 3 or p[-3] not in want: continue
        spk, vid, utt = p[-3], p[-2], p[-1][:-4]
        L = t.extractfile(m).read().decode(errors="replace").splitlines(); n_files += 1
        sd = [l for l in L if l.startswith("Start and Duration")]
        dur = float(sd[0].split()[-1]) if sd else 0.0
        if dur <= best.get((spk, vid), (0.0,))[0]: continue
        tr = np.array([[int(x) for x in l.split()] for l in L if len(l.split()) == 5 and l.split()[0].isdigit()], np.int32)
        if len(tr): best[(spk, vid)] = (dur, utt, tr)
        if n_files % 200000 == 0: print(f"  {s.n / 1e9:.1f} GB read, {n_files} utterance files of queued speakers, {len(best)} videos", flush=True)
pickle.dump(best, open(D / "tracks.pkl", "wb"))
rows = [(i, spk, ",".join(v for v in spk2vid[spk] if (spk, v) in best)) for i, spk in enumerate(queue)]
pd.DataFrame(rows, columns=["order", "identity", "videos"]).to_csv(D / "queue.csv", index=False)
print(f"done: {s.n / 1e9:.1f} GB streamed, tracks for {len(best)} videos of {len({k[0] for k in best})} speakers", flush=True)
