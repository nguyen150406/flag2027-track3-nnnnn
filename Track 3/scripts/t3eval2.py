"""T3 evaluation v2: identity-disjoint folds with CROSS-SESSION positives.

Pseudo-sessions (≈videos) per identity: average-linkage clustering of [CLIP face, SimAM voice]
of each train row (same row = same segment), cosine distance threshold 0.2 (~3 large
clusters/ID, matching ~2.6 videos/ID in the dataset description). Positives pair a voice
with a face of the same ID from a different pseudo-session; negatives pair it with a face
of another held ID (standard) or another same-gender held ID (same_gender). Built once
(seed 0) and cached; only FLAG train rows are used.
"""
import pickle, numpy as np, pandas as pd
from t3lib import *

PATH = T3 / "outputs/eval2_trials.pkl"


def sessions():
    from sklearn.cluster import AgglomerativeClustering
    import t3data as D
    C = np.asarray(D.face("clip")[:6485]); S = np.asarray(D.voice("sim")[:6485])
    X = np.hstack([C, S]) / np.sqrt(2)
    sess = np.empty(6485, object)
    for s in np.unique(SPK):
        r = np.flatnonzero(SPK == s)
        lab = AgglomerativeClustering(n_clusters=None, distance_threshold=0.2, metric="cosine",
                                      linkage="average").fit_predict(X[r])
        sess[r] = [f"{s}:{l}" for l in lab]
    return sess


def build(per_voice=1, seed=0):
    rng = np.random.default_rng(seed); sess = sessions(); cells = {}
    for fold in range(4):
        held = GROUPS[fold + 1]
        for proto in ("standard", "same_gender"):
            rows = []
            for s in held:
                vr = np.flatnonzero(SPK == s)
                vr = rng.choice(vr, min(len(vr), 60), replace=False)
                others = [t for t in held if t != s and (proto == "standard" or GENDER[t] == GENDER[s])]
                for v in vr:
                    pos = np.flatnonzero((SPK == s) & (sess != sess[v]))
                    if not len(pos):
                        continue
                    rows.append((v, rng.choice(pos), 1, s))
                    t = rng.choice(others); rows.append((v, rng.choice(np.flatnonzero(SPK == t)), 0, s))
            fr = pd.DataFrame(rows, columns=["voice_row", "face_row", "label", "voice_identity"])
            cells[fold, proto] = fr
    pickle.dump(dict(cells=cells, sessions=sess), open(PATH, "wb"))
    return cells


def cells():
    return pickle.load(open(PATH, "rb"))["cells"] if PATH.exists() else build()


def evaluate2(score_fn, verbose=True, tag=""):
    """score_fn(fold, protocol, frame) -> scores."""
    c = cells(); out = {p: [] for p in ("standard", "same_gender")}
    for f in range(4):
        for p in out:
            out[p].append(eer(c[f, p], score_fn(f, p, c[f, p])))
    r = {p: float(np.mean(v)) for p, v in out.items()}; r["mean"] = (r["standard"] + r["same_gender"]) / 2
    if verbose:
        print(f"{tag:28s} XS std {r['standard']:.2f} sg {r['same_gender']:.2f} mean {r['mean']:.2f}", flush=True)
    return r


if __name__ == "__main__":
    c = build()
    for k, v in c.items():
        print(k, len(v), v.voice_identity.nunique())
