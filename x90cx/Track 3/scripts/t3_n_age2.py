"""T3-N: better age compatibility.
Face age: ViT (FairFace bins), MiVOLO-v2, or their mean.  Voice age: audeering-6, audeering-24,
their mean, or a ridge regression on [aud6 hidden, aud24 hidden] trained to predict the
identity-median face age of fold-fit IDs (FLAG train only; optional MAV v3)."""
import pickle, numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from t3lib import *
AGE = ROOT / "Track 1/age_compatibility_experiments/cache"
vit = np.load(AGE / "train_face_age.npy"); mvl = np.load(T3 / "caches/mivolo_face/face_age.npy")[:6485] / 100
import os; _VS = os.environ.get("T3_VAGE", "")  # e.g. "_seg3": fixed 3-s sub-segment averaging (extract_voice_age.py seg 3)
a24 = np.load(T3 / f"caches/flag_voiceage24{_VS}.npz"); v24 = a24["age"][:6485]
_p6 = T3 / f"caches/flag_voiceage6{_VS}.npz"
if _p6.exists():
    a6 = np.load(_p6); v6 = a6["age"][:6485]
    H = np.hstack([a6["emb"][:6485].astype(np.float32), a24["emb"][:6485].astype(np.float32)])
else:
    v6 = np.load(AGE / "train_voice_age.npy"); H = a24["emb"][:6485].astype(np.float32)
H = (H - H.mean(0)) / (H.std(0) + 1e-6)   # train-row statistics only
FACES = {"vit": vit, "mivolo": mvl, "mean": (vit + mvl) / 2}
a = pickle.load(open(T3 / "outputs/oof_j_v1a_v3.pkl", "rb")); b = pickle.load(open(T3 / "outputs/oof_j_v3only.pkl", "rb"))
o = dict(b); o.update(a)
W = {"farl_sim": .4, "clip_sim": .4, "farl_ecapa": .2, "clip_ecapa": .2, "farl_campplus": .2, "clip_campplus": .2}
br = lambda f, p: sum(W[k] * o[k][f, p] for k in W)
def idmed(x): return pd.Series(x).groupby(SPK).median().loc[SPK].values
def offset(f, v, m): return float(np.median([np.median(f[m & (SPK == s)]) - np.median(v[m & (SPK == s)]) for s in np.unique(SPK[m])]))
cache = {}
def vage(kind, face, fold):
    k = (kind, face, fold)
    if k in cache: return cache[k]
    fit, _ = fold_ids(fold); m = np.isin(SPK, fit); fa = FACES[face]
    if kind in ("a6", "a24", "amean"):
        raw = dict(a6=v6, a24=v24, amean=(v6 + v24) / 2)[kind]
        out = np.clip(raw + offset(fa, raw, m), 0, 1)
    elif kind == "ridge":
        r = RidgeCV(alphas=np.logspace(0, 5, 12)).fit(H[m], idmed(fa)[m]); out = r.predict(H)
    elif kind == "ridge+a24":
        out = .5 * vage("ridge", face, fold) + .5 * vage("a24", face, fold)
    cache[k] = out; return out
if __name__ == "__main__":
    print("MAE vs id-median face age (years), full train:")
    for face in FACES:
        for kind in ("a6", "a24", "amean"):
            m = np.ones(6485, bool)
            print(f"  face={face:6s} {kind:6s} {100*np.mean(np.abs(np.clip(dict(a6=v6,a24=v24,amean=(v6+v24)/2)[kind] + offset(FACES[face], dict(a6=v6,a24=v24,amean=(v6+v24)/2)[kind], m),0,1) - idmed(FACES[face]))):.2f}")
    for face in FACES:
        for kind in ("a6", "a24", "amean", "ridge", "ridge+a24"):
            fa = FACES[face]
            r0 = evaluate(lambda f, p, fr, v: -np.abs(fa[fr.face_row] - vage(kind, face, f)[fr.voice_row]))
            best = None
            for w in (.6, 1.0, 1.5):
                r = evaluate(lambda f, p, fr, v: br(f, p) - w * np.abs(fa[fr.face_row] - vage(kind, face, f)[fr.voice_row]))
                if best is None or r["mean"] < best[1]["mean"]: best = (w, r)
            w, r = best
            print(f"face={face:6s} voice={kind:9s} AGE-ONLY sg {r0['same_gender']:.2f} std {r0['standard']:.2f} | fused(w={w}) std {r['standard']:.2f} sg {r['same_gender']:.2f} mean {r['mean']:.2f}", flush=True)
