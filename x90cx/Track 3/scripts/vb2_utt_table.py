"""VoxBlink2 (2026-10-27, user "oke đồng ý" to the VoxBlink2 pilot): longest official utterance per (speaker, video) for EVERY
non-Bengali speaker (caches/vb2/exclude_bengali.csv), from the local official vb2_meta.tar.gz -> caches/vb2/utt_best.csv
(identity, video, utt, dur, f0, f1; frames at 25 fps). The HF audio mirror stores the first 60 s of each video, and every official
utterance ends before 60 s (checked on 20k tracks), so f0/25 .. (f1+1)/25 cuts the utterance from the mp3. Face boxes are not kept."""
import tarfile
from pathlib import Path
import pandas as pd
D = Path(__file__).resolve().parents[1] / "caches/vb2"
bn = set(pd.read_csv(D / "exclude_bengali.csv").identity)
best, n = {}, 0
with tarfile.open(D / "vb2_meta.tar.gz", "r|gz") as t:
    for m in t:
        if not m.isfile(): continue
        p = m.name.split("/")
        if len(p) < 3 or p[-3] in bn: continue
        spk, vid, utt = p[-3], p[-2], p[-1][:-4]
        L = t.extractfile(m).read().decode(errors="replace").splitlines(); n += 1
        sd = [l for l in L if l.startswith("Start and Duration")]
        dur = float(sd[0].split()[-1]) if sd else 0.0
        if dur <= best.get((spk, vid), (0.0,))[0]: continue
        fr = [int(l.split()[0]) for l in L if len(l.split()) == 5 and l.split()[0].isdigit()]
        if fr: best[(spk, vid)] = (dur, utt, fr[0], fr[-1])
        if n % 500000 == 0: print(f"{n} utterance files, {len(best)} videos", flush=True)
df = pd.DataFrame([(s, v, u, d, a, b) for (s, v), (d, u, a, b) in best.items()], columns=["identity", "video", "utt", "dur", "f0", "f1"])
df.to_csv(D / "utt_best.csv", index=False)
print(f"done: {n} files, {len(df)} videos, {df.identity.nunique()} speakers; max end {df.f1.max() / 25:.1f}s", flush=True)
