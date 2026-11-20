"""Bengali LID audit of every external training clip (user 2026-11-05: "lọc lại bộ data của chúng ta thật là kĩ ... trước khi train").
Three independent LID models, run on EVERY 30-s window of each clip (the whole clip, not just its start):
  vl  = SpeechBrain VoxLingua107 ECAPA        (labels 'bn: Bengali', 'as: Assamese')
  wh  = Whisper large-v3 language detection   (softmax over the language tokens after <|startoftranscript|>; 'bn', 'as')
  mms = Meta MMS-LID-256                      (ISO 639-3 'ben', 'asm')
Per clip and model: top-1 labels over windows ('|'-joined, short codes) and the max over windows of P(bn) and P(bn) + P(as).
The flag rule is applied later (t3_lid_merge.py); this script only measures. Filter use only: no model here is trained or adapted.
Usage: python t3_lid_audit.py --list clips.csv (src,key,path) --out DIR [--shard k/n] [--device cuda] [--models vl,wh,mms]
       [--whisper openai/whisper-large-v3[@commit]] [--limit N]   -> DIR/lid_<k>of<n>.csv (resumable: rows already written are skipped)"""
import argparse, csv, os, re, time, wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, torch

WIN = 30 * 16000


def read16(p):  # mono float32 at 16 kHz in [-1, 1]
    with wave.open(str(p)) as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth(); x = w.readframes(w.getnframes())
    assert sw == 2, (p, sw)
    x = np.frombuffer(x, np.int16).astype(np.float32) / 32768
    if ch > 1: x = x.reshape(-1, ch).mean(1)
    if sr != 16000:
        import torchaudio; x = torchaudio.functional.resample(torch.from_numpy(x)[None], sr, 16000)[0].numpy()
    return x


def windows(x):  # non-overlapping 30-s windows; a tail under 1 s is dropped unless it is the whole clip
    w = [x[s:s + WIN] for s in range(0, len(x), WIN)]
    return [q for q in w if len(q) >= 16000] or [x]


class VL:
    name = "vl"
    def __init__(s, dev):
        from speechbrain.inference.classifiers import EncoderClassifier
        s.m = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir=str(Path.home() / ".cache/sb_vl107"), run_opts={"device": str(dev)})
        lab = s.m.hparams.label_encoder.ind2lab; s.lab = [lab[i].split(":")[0].strip() for i in range(len(lab))]
        s.bn, s.as_ = s.lab.index("bn"), s.lab.index("as")
    @torch.no_grad()
    def __call__(s, ws):
        out = []
        for q in ws:
            p = s.m.classify_batch(torch.from_numpy(q)[None])[0][0].exp().float().cpu().numpy()  # log posteriors -> probabilities
            out.append((s.lab[int(p.argmax())], float(p[s.bn]), float(p[s.bn] + p[s.as_])))
        return out


class WH:
    name = "wh"
    def __init__(s, dev, model):
        from transformers import GenerationConfig, WhisperFeatureExtractor, WhisperForConditionalGeneration
        model, _, rev = model.partition("@"); kw = {"revision": rev} if rev else {}  # name[@commit]
        s.dev = dev; s.fe = WhisperFeatureExtractor.from_pretrained(model, **kw)
        s.m = WhisperForConditionalGeneration.from_pretrained(model, **kw).to(dev).eval()
        if dev.type == "cuda": s.m = s.m.half()  # T4: fp16
        g = GenerationConfig.from_pretrained(model, **kw); l2i = {k[2:-2]: v for k, v in g.lang_to_id.items()}  # token ids from the model repo
        assert {"bn", "as", "en"} <= set(l2i) and len(l2i) >= 99, len(l2i)
        s.codes = sorted(l2i); s.ids = torch.tensor([l2i[c] for c in s.codes], device=dev); s.sot = g.decoder_start_token_id
        s.bn, s.as_ = s.codes.index("bn"), s.codes.index("as")
    @torch.no_grad()
    def __call__(s, ws, bs=16):  # every window is padded to Whisper's fixed 30 s, so batching does not change any result
        out = []
        for i in range(0, len(ws), bs):
            f = s.fe(ws[i:i + bs], sampling_rate=16000, return_tensors="pt").input_features.to(s.dev, s.m.dtype)
            d = torch.full((len(f), 1), s.sot, device=s.dev)
            p = s.m(input_features=f, decoder_input_ids=d).logits[:, -1, s.ids].float().softmax(-1).cpu().numpy()
            out += [(s.codes[int(r.argmax())], float(r[s.bn]), float(r[s.bn] + r[s.as_])) for r in p]
        return out


class MMS:
    name = "mms"
    def __init__(s, dev):
        from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
        s.dev = dev; s.fe = AutoFeatureExtractor.from_pretrained("facebook/mms-lid-256")
        s.m = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/mms-lid-256").to(dev).eval()
        if dev.type == "cuda": s.m = s.m.half()
        lab = s.m.config.id2label; s.lab = [lab[i] for i in range(len(lab))]; s.bn, s.as_ = s.lab.index("ben"), s.lab.index("asm")
    @torch.no_grad()
    def __call__(s, ws):
        out = []
        for q in ws:  # one window at a time (no padding interactions)
            x = s.fe(q, sampling_rate=16000, return_tensors="pt").input_values.to(s.dev, next(s.m.parameters()).dtype)
            p = s.m(x).logits[0].float().softmax(-1).cpu().numpy(); top = s.lab[int(p.argmax())]
            out.append(({"ben": "bn", "asm": "as", "eng": "en"}.get(top, top), float(p[s.bn]), float(p[s.bn] + p[s.as_])))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True); ap.add_argument("--out", required=True); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--models", default="vl,wh,mms"); ap.add_argument("--whisper", default="openai/whisper-large-v3")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(); T0 = time.time(); dev = torch.device(a.device)
    log = lambda m: print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 60:6.1f}m] lid:", m, flush=True)
    k, n = map(int, a.shard.split("/")); L = pd.read_csv(a.list).iloc[k::n]
    if a.limit: L = L.groupby("src").head(a.limit)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); f = out / f"lid_{k}of{n}.csv"
    done = set(pd.read_csv(f).key.astype(str)) if f.exists() else set(); L = L[~L.key.astype(str).isin(done)]
    M = [{"vl": lambda: VL(dev), "wh": lambda: WH(dev, a.whisper), "mms": lambda: MMS(dev)}[m]() for m in a.models.split(",")]
    log(f"{len(L)} clips to do ({len(done)} already in {f.name}), models {[m.name for m in M]}")
    cols = ["src", "key", "dur", "n_win"] + [f"{m.name}_{c}" for m in M for c in ("top", "bn", "bnas")]
    new = not f.exists(); fh = open(f, "a", newline=""); wr = csv.writer(fh)
    if new: wr.writerow(cols)
    def load(r):
        try: return r, read16(r.path)
        except Exception as e: return r, e
    bad = 0; t = time.time(); rows = list(L.itertuples()); CH = 16  # clips per chunk: Whisper batches the chunk's windows together
    starts = list(range(0, len(rows), CH)); AHEAD = 4  # bounded prefetch: only a few chunks of audio in memory
    with ThreadPoolExecutor(a.workers) as pool:
        q = [[pool.submit(load, r) for r in rows[s:s + CH]] for s in starts[:AHEAD]]
        for ci, c0 in enumerate(starts):
            futs = q.pop(0)
            if ci + AHEAD < len(starts): q.append([pool.submit(load, r) for r in rows[starts[ci + AHEAD]:starts[ci + AHEAD] + CH]])
            chunk = [fu.result() for fu in futs]; ws, own = [], []
            for j, (r, x) in enumerate(chunk):
                if isinstance(x, Exception):
                    bad += 1; log(f"unreadable {r.path}: {x!r}"[:200]) if bad <= 5 else None
                    wr.writerow([r.src, r.key, -1, 0] + ["ERR", 1.0, 1.0] * len(M)); continue  # unreadable = flagged (cannot be verified)
                w = windows(x); ws += w; own += [j] * len(w)
            res = {m.name: m(ws) for m in M} if ws else {}; own = np.array(own)
            for j, (r, x) in enumerate(chunk):
                if isinstance(x, Exception): continue
                idx = np.flatnonzero(own == j); row = [r.src, r.key, round(len(x) / 16000, 2), len(idx)]
                for m in M:
                    wq = [res[m.name][i] for i in idx]  # not q: that is the prefetch queue
                    row += ["|".join(sorted({t for t, _, _ in wq})), round(max(b for _, b, _ in wq), 6), round(max(c for _, _, c in wq), 6)]
                wr.writerow(row)
            if (c0 // CH) % 30 == 0: fh.flush(); log(f"{c0 + len(chunk)}/{len(rows)} clips ({(c0 + len(chunk)) / (time.time() - t):.1f}/s), unreadable {bad}")
    fh.close(); log(f"DONE {len(rows)} clips, unreadable {bad} -> {f}")


if __name__ == "__main__":
    main()
