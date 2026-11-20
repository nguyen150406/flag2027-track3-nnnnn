"""LRS3-TED trainval (4,004 TED/TEDx talks, English; one talk ~ one speaker) -> external identities for the Track 3 projector.
Input: caches/lrs3_raw/trainval.tar.gz (HF mirror TheNHz/ellipsis-lrs3-raw, gated). LRS3 face tracks are already 224x224 with
the same framing as FLAG faces (YuNet face box w/h/cx/cy = .53/.71/.50/.37 vs FLAG .53/.70/.50/.38), so frames are used as-is.
Per talk: up to 2 samples, each = consecutive sentence clips from one half of the talk, concatenated until >= 5 s (max 3 clips,
capped at 8 s); faces = frame at 30% of the sample's first clip and 70% of its last clip.
Output: caches/lrs3_media/{wav,img}/ + lrs3_manifest.csv + lrs3_media.zip (for the Kaggle encode kernel)."""
import os, sys, tarfile, subprocess, wave, zipfile, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd, cv2

T3 = Path(__file__).resolve().parents[1]
RAW, OUT = T3 / "caches/lrs3_raw", T3 / "caches/lrs3_media"
FF = sys.argv[1] if len(sys.argv) > 1 else "ffmpeg"
CLIPS = RAW / "clips"
for d in (CLIPS, OUT / "wav", OUT / "img"): d.mkdir(parents=True, exist_ok=True)

if not (RAW / "extracted.ok").exists():  # 1) stream the tar.gz once, keep only the mp4 clips
    n = 0
    with tarfile.open(RAW / "trainval.tar.gz", "r|gz") as t:
        for m in t:
            if m.isfile() and m.name.endswith(".mp4"):
                p = m.name.split("/"); dst = CLIPS / p[-2]; dst.mkdir(exist_ok=True)
                (dst / p[-1]).write_bytes(t.extractfile(m).read()); n += 1
                if n % 5000 == 0: print("extracted", n, flush=True)
    (RAW / "extracted.ok").write_text(str(n)); print("extracted", n, flush=True)


def dur(p):
    c = cv2.VideoCapture(str(p)); n, f = c.get(cv2.CAP_PROP_FRAME_COUNT), c.get(cv2.CAP_PROP_FPS) or 25.0; c.release()
    return n / f


def pcm(p):
    r = subprocess.run([FF, "-v", "error", "-i", str(p), "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"], capture_output=True)
    return np.frombuffer(r.stdout, np.int16)


def frame(p, q):
    c = cv2.VideoCapture(str(p)); n = int(c.get(cv2.CAP_PROP_FRAME_COUNT)); c.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(q * n) - 1))
    ok, fr = c.read(); c.release()
    return fr if ok else None


def talk(vid):
    clips = sorted((CLIPS / vid).glob("*.mp4"))
    halves = [clips[: (len(clips) + 1) // 2], clips[(len(clips) + 1) // 2:]] if len(clips) > 1 else [clips]
    out = []
    for k, half in enumerate(halves, 1):
        sel, tot = [], 0.0
        for c in half:
            sel.append(c); tot += dur(c)
            if tot >= 5.0 or len(sel) == 3: break
        if not sel: continue
        a = np.concatenate([pcm(c) for c in sel])[: 8 * 16000]
        if len(a) < 1.0 * 16000: continue
        key = f"{vid}_s{k}"
        with wave.open(str(OUT / "wav" / f"{key}.wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(a.tobytes())
        nf = 0
        for j, (c, q) in enumerate(((sel[0], .3), (sel[-1], .7)), 1):
            fr = frame(c, q)
            if fr is not None:
                cv2.imwrite(str(OUT / "img" / f"{key}_{j}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 95]); nf += 1
        out.append(dict(identity=f"lrs3_{vid}", gender="u", language="english", video=vid, utt=f"s{k}", key=key,
                        n_clips=len(sel), dur=len(a) / 16000, n_frames=nf, ok=nf > 0))
    return out


vids = sorted(p.name for p in CLIPS.iterdir() if p.is_dir())
rows = []
with ThreadPoolExecutor(8) as ex:
    for i, fu in enumerate(as_completed([ex.submit(talk, v) for v in vids])):
        rows += fu.result()
        if i % 500 == 0: print(f"talks {i}/{len(vids)} samples {len(rows)}", flush=True)
man = pd.DataFrame(rows).sort_values("key").reset_index(drop=True)
man.to_csv(OUT / "lrs3_manifest.csv", index=False)
print(f"samples {int(man.ok.sum())}/{len(man)} from {man[man.ok].identity.nunique()} talks; dur mean {man.dur.mean():.2f}s "
      f"median {man.dur.median():.2f}s; frames {int(man.n_frames.sum())}", flush=True)
with zipfile.ZipFile(OUT / "lrs3_media.zip", "w", zipfile.ZIP_STORED) as z:
    z.write(OUT / "lrs3_manifest.csv", "lrs3_manifest.csv")
    for r in man[man.ok].itertuples(index=False):
        z.write(OUT / "wav" / f"{r.key}.wav", f"wav/{r.key}.wav")
        for j in range(1, 3):
            p = OUT / "img" / f"{r.key}_{j}.jpg"
            if p.exists(): z.write(p, f"img/{r.key}_{j}.jpg")
json.dump(dict(source="LRS3-TED trainval (Afouras, Chung, Zisserman 2018; official page: CC BY 4.0; TED/TEDx source media) via HF mirror "
                      "TheNHz/ellipsis-lrs3-raw", identity="one TED talk = one speaker (no explicit identity labels)", language="English only",
               samples=int(man.ok.sum()), talks=int(man[man.ok].identity.nunique()), flag_data_used=False),
          open(OUT / "PROVENANCE_LRS3.json", "w"), indent=1)
print("zip", round(os.path.getsize(OUT / "lrs3_media.zip") / 1e9, 2), "GB")
