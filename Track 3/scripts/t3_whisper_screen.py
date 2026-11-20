"""Whisper voice branch, step 1 screen (user 2026-11-03 "oke làm đi"): on FLAG train only, is a frozen Whisper encoder a
voice view that differs from our speaker encoders (sim/rdn/wsv) and still carries usable identity information?
Model: openai/whisper-large-v3-turbo @41f01f3f (same 32-layer large-v3 encoder family, half the download of large-v3;
pretraining includes Bengali, allowed frozen per the 17/10 organizer answer; nothing here is trained on Bengali).
  python t3_whisper_screen.py extract [n]   one file at a time (pair-local), <=30 s chunks (>=1 s), each padded to Whisper's
                                            fixed 30 s window; per layer mean/std over VALID frames only, chunks averaged
                                            -> caches/whisper/flag_train_{mean,std}.f16.npy [6485, len(LAYERS), 1280]
  python t3_whisper_screen.py screen        per-layer SV EER on FLAG train pairs (same pairs for sim/rdn/wsv), score
                                            correlations; writes CV candidates caches/voice/whs{a,b}/voice_embeddings.float32.npy"""
import sys, time, wave
from pathlib import Path
import numpy as np, torch
from t3lib import TRAIN, ROOT, T3, SPK, GENDER

MODEL, REV = "openai/whisper-large-v3-turbo", "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
LAYERS = list(range(0, 33, 2))  # hidden_states index (0 = after conv + positions, 32 = last block)
OUT = T3 / "caches/whisper"


def read(p):
    with wave.open(str(p)) as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1 and w.getframerate() == 16000, p
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def extract(n=0):
    from transformers import WhisperFeatureExtractor, WhisperModel
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    fe = WhisperFeatureExtractor.from_pretrained(MODEL, revision=REV)
    enc = WhisperModel.from_pretrained(MODEL, revision=REV, torch_dtype=torch.float16).encoder.to(dev).eval()
    paths = [ROOT / "data/train_set/train_set" / p for p in TRAIN.voice_relpath][: int(n) or None]
    M = np.zeros((len(paths), len(LAYERS), 1280), np.float16); S = np.zeros_like(M); t0 = time.time()
    for i, p in enumerate(paths):
        x = read(p); ch = [x[s:s + 480000] for s in range(0, len(x), 480000)]; ch = [c for c in ch if len(c) >= 16000] or [x]
        ms, ss = [], []
        for c in ch:
            f = fe(c, sampling_rate=16000, return_tensors="pt").input_features.to(dev, torch.float16)  # padded to 3000 frames
            with torch.no_grad():
                hs = enc(f, output_hidden_states=True).hidden_states
            k = min(1500, -(-len(c) // 320))  # encoder frames covering real audio (10 ms mel hop, conv stride 2)
            h = torch.stack([hs[l][0, :k].float() for l in LAYERS])  # [L, k, 1280]
            ms.append(h.mean(1)); ss.append(h.std(1) if k > 1 else torch.zeros_like(h[:, 0]))
        M[i] = torch.stack(ms).mean(0).cpu().numpy(); S[i] = torch.stack(ss).mean(0).cpu().numpy()
        if i % 250 == 0 or i == len(paths) - 1:
            print(f"{i + 1}/{len(paths)} {(time.time() - t0) / 60:.1f} min ({(time.time() - t0) / (i + 1):.2f} s/file)", flush=True)
    assert np.isfinite(M.astype(np.float32)).all() and np.isfinite(S.astype(np.float32)).all()
    OUT.mkdir(parents=True, exist_ok=True); sfx = "" if not int(n) else f".smoke{n}"
    np.save(OUT / f"flag_train_mean{sfx}.f16.npy", M); np.save(OUT / f"flag_train_std{sfx}.f16.npy", S); print("saved", OUT, M.shape)


def eer(pos, neg):
    t = np.sort(np.concatenate([pos, neg])); fr = np.searchsorted(np.sort(pos), t) / len(pos)
    fa = 1 - np.searchsorted(np.sort(neg), t, side="right") / len(neg); j = np.argmin(np.abs(fr - fa)); return 100 * (fr[j] + fa[j]) / 2


def screen():
    import t3data as D
    M = np.load(OUT / "flag_train_mean.f16.npy").astype(np.float32); S = np.load(OUT / "flag_train_std.f16.npy").astype(np.float32)
    r = np.random.default_rng(0); n = len(SPK); g = np.array([GENDER[s] for s in SPK])
    a, b = r.integers(n, size=400000), r.integers(n, size=400000); a, b = a[a != b], b[a != b]
    same = SPK[a] == SPK[b]; sg = (g[a] == g[b]) & ~same
    P = np.flatnonzero(same)[:20000]; N = np.flatnonzero(sg)[:20000]  # same-gender negatives: the hard condition
    idx = np.concatenate([P, N]); lab = np.r_[np.ones(len(P)), np.zeros(len(N))]
    def cos(X):  # train-mean centred cosine (the mean is a property of the encoder, fitted on train rows only)
        X = X - X.mean(0); X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9; return (X[a[idx]] * X[b[idx]]).sum(1)
    ref = {v: cos(np.asarray(D.voice_full(v)[:6485], np.float32)) for v in ("sim", "rdn", "wsv")}
    print(f"pairs: {len(P)} same-ID, {len(N)} same-gender different-ID (FLAG train, 70 IDs)")
    for v, s in ref.items(): print(f"  {v:9s} SV EER {eer(s[lab == 1], s[lab == 0]):5.2f}")
    print("  corr sim~rdn %.3f  sim~wsv %.3f" % (np.corrcoef(ref["sim"], ref["rdn"])[0, 1], np.corrcoef(ref["sim"], ref["wsv"])[0, 1]))
    res = []
    for j, l in enumerate(LAYERS):
        s = cos(np.hstack([M[:, j], S[:, j]])); e = eer(s[lab == 1], s[lab == 0])
        c = [np.corrcoef(s, ref[v])[0, 1] for v in ("sim", "rdn", "wsv")]; res.append((e, l, j))
        print(f"  whisper L{l:02d} mean|std SV EER {e:5.2f}  corr sim {c[0]:.3f} rdn {c[1]:.3f} wsv {c[2]:.3f}")
    best = min(res)[2]; band = [j for j in range(len(LAYERS)) if abs(j - best) <= 2]
    cand = {"whsa": np.hstack([M[:, best], S[:, best]]),  # best single layer
            "whsb": np.hstack([M[:, band].mean(1), S[:, band].mean(1)])}  # +-4 layers around it
    for k, X in cand.items():
        s = cos(X); d = T3 / "caches/voice" / k; d.mkdir(parents=True, exist_ok=True)
        np.save(d / "voice_embeddings.float32.npy", X.astype(np.float32))
        print(f"  {k}: layers {[LAYERS[j] for j in ([best] if k == 'whsa' else band)]} SV EER {eer(s[lab == 1], s[lab == 0]):5.2f} "
              f"corr sim {np.corrcoef(s, ref['sim'])[0, 1]:.3f} rdn {np.corrcoef(s, ref['rdn'])[0, 1]:.3f} -> {d}")


if __name__ == "__main__":
    {"extract": extract, "screen": screen}[sys.argv[1]](*sys.argv[2:])
