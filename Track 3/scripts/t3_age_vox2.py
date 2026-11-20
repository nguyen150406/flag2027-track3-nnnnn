"""Voice age from VoxCeleb2 face-age pseudo-labels (face -> voice age transfer, no new data).
VoxCeleb2 A+B clips (58k, 6k identities) have audeering-24 hidden states (voice) and ViT-age logits on 2 frames (face).
Target per clip = ViT expected age of its frames, mapped to the FLAG face-age scale (mean of ViT and MiVOLO) by a linear
fit on FLAG train rows. Ridge on the 1024-d audeering hidden state -> FLAG voice age (+ per-fold median offset on fit IDs).
Evaluated inside the SUB09 fused CV (OOF oof_j_vox2vb_*) by swapping only the voice-age input of the age term."""
import sys, io, contextlib, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from t3lib import *
sys.argv = ["x", "vox2vb"]
with contextlib.redirect_stdout(io.StringIO()):
    import t3_fuse_vs09 as Fz
import t3_n_age2 as A

C = np.array([1, 6, 14.5, 24.5, 34.5, 44.5, 54.5, 64.5, 75]) / 100
def vit_age(logits):
    lg = np.asarray(logits, np.float32); p = np.exp(lg - lg.max(1, keepdims=True)); return (p / p.sum(1, keepdims=True)) @ C

EXT = T3 / "caches/ext"
X, Y, G = [], [], []
for tag, rows_f, vit_f in (("vox2a", "rows_vox2.csv", "vox2_face_vit.npy"), ("vox2b", "rows_vox2b.csv", "vox2b_face_vit.npy")):
    a = np.load(EXT / f"{tag}_voice_aud24.npz"); r = pd.read_csv(EXT / rows_f, usecols=["identity", "clip_index"])
    fa = vit_age(np.load(EXT / vit_f, mmap_mode="r")[:, -9:]); r["fa"] = fa
    per = r.groupby("clip_index").agg(fa=("fa", "mean"), identity=("identity", "first"))
    keep = np.isin(a["clip_index"], per.index)
    X.append(a["emb"][keep].astype(np.float32)); Y.append(per.loc[a["clip_index"][keep], "fa"].to_numpy()); G.append(per.loc[a["clip_index"][keep], "identity"].to_numpy())
X, Y, G = np.vstack(X), np.concatenate(Y), np.concatenate(G)
# map ViT-only age to the FLAG face-age definition (mean of ViT and MiVOLO), fitted on FLAG train rows
b1, b0 = np.polyfit(A.vit, A.FACES["mean"], 1); Ym = b1 * Y + b0
print(f"VoxCeleb2 clips {len(X)}, identities {len(set(G))}; ViT->FLAG-mean map {b1:.3f}x+{b0:.3f}; target mean {100 * Ym.mean():.1f}y sd {100 * Ym.std():.1f}y")
mu, sd = X.mean(0), X.std(0) + 1e-6
FL = np.load(T3 / "caches/flag_voiceage24.npz")["emb"].astype(np.float32)
preds = {}
for target in ("clip", "idmed"):
    y = Ym if target == "clip" else pd.Series(Ym).groupby(G).transform("median").to_numpy()
    for alpha in (10.0, 100.0, 1000.0):
        r = Ridge(alpha=alpha).fit((X - mu) / sd, y); preds[f"vox2ridge_{target}_a{int(alpha)}"] = r.predict((FL - mu) / sd)[:6485]

def offset_cal(raw, fold):
    fit, _ = fold_ids(fold); m = np.isin(SPK, fit)
    return np.clip(raw + A.offset(A.FACES["mean"], raw, m), 0, 1)

face_id = A.idmed(A.FACES["mean"])
print("MAE (years) vs identity-median face age, FLAG train (fold 0 offset):")
base = {"a24 (+offset)": A.vage("a24", "mean", 0), "ridge FLAG": A.vage("ridge", "mean", 0), "ridge+a24 (current)": A.vage("ridge+a24", "mean", 0)}
for k, v in list(base.items()) + [(k, offset_cal(v, 0)) for k, v in preds.items()]:
    print(f"  {k:28s} {100 * np.mean(np.abs(v - face_id)):.2f}")

VA = {}  # voice-age variants per fold
for k in preds:
    VA[k] = lambda f, k=k: offset_cal(preds[k], f)
    VA[k + "+cur"] = lambda f, k=k: .5 * offset_cal(preds[k], f) + .5 * A.vage("ridge+a24", "mean", f)
VA["current"] = lambda f: A.vage("ridge+a24", "mean", f)

src, W0 = Fz.s5, Fz.W0
def run(name, va, wa=.9, wg=.05):
    tot = sum(W0.values())
    def score(f, p, fr, v):
        s = sum(W0[k] * src[k][f, p] for k in W0) / tot * 1.6 - wa * np.abs(Fz.fa[fr.face_row] - va(f)[fr.voice_row])
        if p == "standard":
            x, y = Fz.mv[fr.face_row], Fz.pv[fr.voice_row]; s = s + wg * np.log(np.clip(x * y + (1 - x) * (1 - y), 1e-4, 1))
        return s
    r = evaluate(score, verbose=False)
    print(f"{name:34s} wa {wa:.1f}  std {r['standard']:.2f} sg {r['same_gender']:.2f} mean {r['mean']:.2f}  {r['folds']}", flush=True)
    return r
run("current (SUB09 age term)", VA["current"])
for k in VA:
    if k != "current":
        for wa in (.9, 1.2):
            run(k, VA[k], wa)
