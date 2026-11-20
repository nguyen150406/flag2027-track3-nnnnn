"""Dominant-speaker ("keynote speaker") front-end, pair-local (each file processed alone).

Step 1 (mask): split a clip into 1.5 s windows (hop 0.75 s), embed each window with the frozen
WeSpeaker ECAPA-1024 (VoxCeleb) model, cluster the windows of THIS file (average-linkage cosine,
threshold `--thr`), keep the largest cluster. Clips < 2.5 s are kept whole.
Step 2 (embed): concatenate the kept samples and re-embed with a voice encoder (t3extract recipe).
Usage: python extract_dominant.py mask <list.csv> <out.pkl>
       python extract_dominant.py embed <list.csv> <mask.pkl> <encoder> <out.npy>
"""
import os, sys, math, pickle, numpy as np, pandas as pd, torch, torchaudio
os.environ.setdefault("T3_DEVICE", "cpu")
import t3extract as X
from sklearn.cluster import AgglomerativeClustering

SR, WIN, HOP = 16000, int(1.5 * 16000), int(0.75 * 16000)


def windows(n):
    if n < int(2.5 * SR):
        return [(0, n)]
    starts = list(range(0, n - WIN + 1, HOP))
    if starts[-1] + WIN < n: starts.append(n - WIN)
    return [(s, s + WIN) for s in starts]


def mask_file(x, model, thr):
    n = x.shape[1]; w = windows(n)
    if len(w) == 1:
        return [(0, n)], 1, 1.0
    fbs = []
    for a, b in w:
        fb = torchaudio.compliance.kaldi.fbank(x[:, a:b], num_mel_bins=80, frame_length=25, frame_shift=10,
                                               dither=0.0, sample_frequency=SR, window_type="hamming")
        fbs.append(fb - fb.mean(0, keepdim=True))
    with torch.inference_mode():
        e = model(torch.stack(fbs).to(X.DEV)); e = e[-1] if isinstance(e, (tuple, list)) else e
    e = torch.nn.functional.normalize(e.float(), dim=1).cpu().numpy()
    lab = AgglomerativeClustering(n_clusters=None, distance_threshold=thr, metric="cosine", linkage="average").fit_predict(e)
    u, c = np.unique(lab, return_counts=True); keep = lab == u[np.argmax(c)]
    segs = [w[i] for i in np.flatnonzero(keep)]
    # merge overlapping kept windows
    merged = []
    for a, b in sorted(segs):
        if merged and a <= merged[-1][1]: merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else: merged.append((a, b))
    kept = sum(b - a for a, b in merged) / n
    return merged, len(u), kept


def main():
    mode = sys.argv[1]; paths = pd.read_csv(sys.argv[2]).path.tolist()
    if mode == "mask":
        thr = float(os.environ.get("THR", "0.6"))
        one = X._wespeaker("ecapa")  # loads model; we need raw model access
        import types
        model = one.__closure__[[c.cell_contents.__class__.__name__ for c in one.__closure__].index(
            next(c.cell_contents.__class__.__name__ for c in one.__closure__ if hasattr(c.cell_contents, "forward")))].cell_contents
        out = []
        for i, p in enumerate(paths):
            x = torch.from_numpy(X._pcm16(p)); out.append(mask_file(x, model, thr))
            if i % 1000 == 0: print("mask", i, flush=True)
        pickle.dump(out, open(sys.argv[3], "wb"))
    else:
        masks = pickle.load(open(sys.argv[3], "rb")); enc = sys.argv[4]
        one = X._wespeaker(enc) if enc != "wavlm4" else X._wavlm4()
        import tempfile, wave
        res = []
        for i, (p, (segs, k, kept)) in enumerate(zip(paths, masks)):
            x = X._pcm16(p)
            y = np.concatenate([x[:, a:b] for a, b in segs], axis=1)
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                with wave.open(f.name, "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(y.astype(np.int16).tobytes())
                res.append(one(f.name))
            if i % 1000 == 0: print("embed", enc, i, flush=True)
        np.save(sys.argv[5], X._unit(np.stack(res)))


if __name__ == "__main__":
    main()
