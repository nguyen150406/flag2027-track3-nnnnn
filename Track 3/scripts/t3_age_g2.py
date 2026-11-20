"""G2 voice age (teammate report, user 2026-10-26 "thử đi"): ridge 'audeering-24 hidden -> identity-median face age' fitted on
IDENTITY MEANS of FLAG fold-fit IDs + VoxCeleb2 A+B (Bengali-excluded); hidden mean/std over all those rows; offset = median over
identities of (median face age - median audeering age); voice age = .5*ridge(row) + .5*clip(a24 + offset). VoxCeleb2 has ViT face
age only -> mapped to the FLAG face-age scale (mean of ViT, MiVOLO) by a linear fit on the fold's FLAG fit rows. Held-out IDs are
never used. Variants (VAGE[name](fold) -> voice age of the 6485 train rows):
  cur = production ridge (t3_n_age2 ridge+a24), g2flag = G2 recipe on FLAG fit IDs only (the report's baseline),
  g2 = + VoxCeleb2, g2mix = .5 cur + .5 g2.
Usage: python t3_age_g2.py (per-fold MAE on held-out IDs); fused CV: t3_cv_boot.py kgdrop2 kgdrop2@g2 ..."""
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from t3lib import T3, SPK, fold_ids
import t3_n_age2 as A

EXT = T3 / "caches/ext"
C = np.array([1, 6, 14.5, 24.5, 34.5, 44.5, 54.5, 64.5, 75]) / 100  # ViT age bin centres (as t3_age_vox2.py)
def vit_age(lg):
    lg = np.asarray(lg, np.float32); p = np.exp(lg - lg.max(1, keepdims=True)); return (p / p.sum(1, keepdims=True)) @ C

FL_H = np.load(T3 / "caches/flag_voiceage24.npz")["emb"][:6485].astype(np.float32)
_vox = None
def vox2():
    """per-frame (identity, ViT face age) and per-clip (identity, hidden, audeering age), A+B batches, Bengali IDs dropped."""
    global _vox
    if _vox is None:
        bn = set(pd.read_csv(EXT / "exclude_bengali.csv").query("src == 'vox2'").identity)
        fid, fage, vid, H, aud = [], [], [], [], []
        for tag, rows_f, vit_f in (("vox2a", "rows_vox2.csv", "vox2_face_vit.npy"), ("vox2b", "rows_vox2b.csv", "vox2b_face_vit.npy")):
            r = pd.read_csv(EXT / rows_f, usecols=["identity", "clip_index"]); r["fa"] = vit_age(np.load(EXT / vit_f, mmap_mode="r")[:, -9:])
            a = np.load(EXT / f"{tag}_voice_aud24.npz"); who = r.groupby("clip_index").identity.first()
            keep = np.isin(a["clip_index"], who.index) & ~np.isin(who.reindex(a["clip_index"]).to_numpy(), list(bn))
            ok = ~r.identity.isin(bn)
            fid.append(r.identity[ok].to_numpy()); fage.append(r.fa[ok].to_numpy())
            vid.append(who.loc[a["clip_index"][keep]].to_numpy()); H.append(a["emb"][keep].astype(np.float32)); aud.append(a["age"][keep])
        _vox = [np.concatenate(x) for x in (fid, fage, vid)] + [np.vstack(H), np.concatenate(aud)]
    return _vox

def fit_voice_age(parts, alphas=np.logspace(0, 5, 12)):
    """parts: dicts(face_id, face_age, voice_id, hid, aud_age); the report's fit_voice_age (identity-level samples)."""
    H = np.concatenate([p["hid"] for p in parts]); mu, sd = H.mean(0), H.std(0) + 1e-6
    fa = pd.concat([pd.Series(p["face_age"]).groupby(p["face_id"]).median() for p in parts])
    aa = pd.concat([pd.Series(p["aud_age"]).groupby(p["voice_id"]).median() for p in parts])
    assert fa.index.is_unique and aa.index.is_unique
    ids = fa.index[fa.index.isin(aa.index)]
    Z = pd.concat([pd.DataFrame((p["hid"] - mu) / sd).groupby(p["voice_id"]).mean() for p in parts])
    ridge = RidgeCV(alphas=alphas).fit(Z.loc[ids].values, fa.loc[ids].values)
    off = float(np.median(fa.loc[ids].values - aa.loc[ids].values))
    return lambda hid, aud: .5 * ridge.predict((hid - mu) / sd) + .5 * np.clip(aud + off, 0, 1), dict(n_ids=len(ids), alpha=float(ridge.alpha_), offset=off)

_cache, INFO = {}, {}
def g2(fold, with_vox):
    k = (fold, with_vox)
    if k not in _cache:
        fit, _ = fold_ids(fold); m = np.isin(SPK, fit); fa = A.FACES["mean"]
        parts = [dict(face_id=SPK[m], face_age=fa[m], voice_id=SPK[m], hid=FL_H[m], aud_age=A.v24[m])]
        if with_vox:
            b1, b0 = np.polyfit(A.vit[m], fa[m], 1); fid, fage, vid, H, aud = vox2()
            parts.append(dict(face_id=fid, face_age=b1 * fage + b0, voice_id=vid, hid=H, aud_age=aud))
        fn, INFO[k] = fit_voice_age(parts); _cache[k] = fn(FL_H, A.v24)
    return _cache[k]

cur = lambda f: A.vage("ridge+a24", "mean", f)
VAGE = {"cur": cur, "g2flag": lambda f: g2(f, False), "g2": lambda f: g2(f, True), "g2mix": lambda f: .5 * cur(f) + .5 * g2(f, True)}

if __name__ == "__main__":
    fid, _, vid, H, _ = vox2(); print(f"VoxCeleb2 (Bengali-excluded): {len(set(fid))} face IDs, {len(set(vid))} voice IDs, {len(H)} clips")
    face_id = pd.Series(A.FACES["mean"]).groupby(SPK).median()
    for f in range(4):
        _, held = fold_ids(f); line = []
        for n, fn in VAGE.items():
            v = pd.Series(fn(f)).groupby(SPK).median().loc[held]; t = face_id.loc[held]
            line.append(f"{n} MAE {100 * np.mean(np.abs(v - t)):.2f}y r {np.corrcoef(v, t)[0, 1]:.2f}")
        print(f"fold {f}: " + " | ".join(line), {k[1]: v for k, v in INFO.items() if k[0] == f})
