"""ENX2-R V0 screen (13/11): MAV-Celeb v2 identities vs FLAG train, MAV v1 (MAV-ALL validation) and the EN/XL hold-out identities,
by clipnf face centroids. Threshold = 5th percentile of v2 same-identity split-half centroid cosine (videos split in two halves);
a v2 identity whose best match exceeds it is reported (and dropped from training, with the LID-flagged ones)."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; T3 = HERE.parent; EXT = T3 / "caches/ext"; sys.path.insert(0, str(HERE))
U = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)
from t3lib import SPK
fa = pd.read_csv(T3 / "kaggle/enx2r_v2data/faces/v2_faces.csv"); Fv = U(np.load(T3 / "kaggle/enx2r_v2data/faces/v2_face_clipnf.f16.npy").astype(np.float32))
cent = lambda F, ids: (lambda u: (u, U(np.stack([F[ids == i].mean(0) for i in u]))))(np.unique(ids))
vi, Cv = cent(Fv, fa.identity.values)
same = []   # split-half same-identity centroid cos
for i in vi:
    m = fa.identity.values == i; vids = sorted(fa.video[m].unique())
    if len(vids) < 2: continue
    a = np.isin(fa.video.values, vids[::2]) & m; b = np.isin(fa.video.values, vids[1::2]) & m
    same.append(float(U(Fv[a].mean(0, keepdims=True)) @ U(Fv[b].mean(0, keepdims=True)).T))
thr = float(np.percentile(same, 5)); print(f"v2 same-identity split-half cos: median {np.median(same):.3f}, p5 {thr:.3f} (n {len(same)})")
refs = {}
Ff = U(np.load(T3 / "caches/clipnf/face_embeddings.float32.npy")[:6485]); refs["FLAG-train"] = cent(Ff, SPK)
r1 = pd.read_csv(EXT / "rows_v1.csv"); X1 = np.load(EXT / "v1_face_clipnf.float32.npy"); k1 = np.abs(X1).sum(1) > 0   # 10133 rows never encoded (zeros)
refs["MAV-v1"] = cent(U(X1[k1]), r1.identity.astype(str).values[k1])
ho = pd.read_csv(EXT / "ultra_holdout.csv")
for s in ("vox2", "vb"):
    r = pd.read_csv(EXT / f"rows_{s}.csv"); m = r.identity.isin(ho.identity[ho.src.eq(s)]).values
    refs[f"{s}-EN/XL"] = cent(U(np.load(EXT / f"{s}_face_clipnf.float32.npy")[m]), r.identity.values[m])
drop, rep = set(), {}
for nm, (ri, Cr) in refs.items():
    S = Cv @ Cr.T; best = S.max(1); j = S.argmax(1)
    hits = [(vi[k], ri[j[k]], round(float(best[k]), 3)) for k in np.flatnonzero(best > thr)]
    rep[nm] = dict(n_ref=len(ri), max=float(best.max()), p99_other=float(np.percentile(S, 99)), hits=hits); drop |= {h[0] for h in hits}
    print(f"{nm:12s} refs {len(ri):5d} | v2 best-match max {best.max():.3f} | pairs p99 {np.percentile(S, 99):.3f} | above thr: {hits}")
json.dump(dict(threshold=thr, report=rep, drop=sorted(drop)), open(T3 / "outputs/enx2r/v2_overlap.json", "w"), indent=1)
print("v2 identities to drop (overlap):", sorted(drop))
