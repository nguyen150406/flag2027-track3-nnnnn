"""ULTRA scorecard (user 2026-11-11): every expert scored on the SAME person-held-out trials, train data only.
  FLAG OOF : split folds k=0..4 (cv_run/w/split.json), trial sampler t3_cv_protocol.make_trials (3000 + 3000 per protocol), scored with fold-k models
  reserve  : the 10 reserve persons of the split (never in any fold model), 3000 + 3000, mean over the 5 fold models
  EXT EN/XL: caches/ext/ultra_holdout.csv persons (300 English cross-video / 250 South-Asian-language), positives = face and voice
             from DIFFERENT videos, negatives any (no_gender) or same-gender (gender) other held-out person; mean over fold models
  python t3_ultra_eval.py trials | attr | heads <models_dir> <branch,...>"""
import os, sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
os.environ.setdefault("T3_EXCLUDE", str(HERE.parent / "caches/ext/exclude_ultra.csv"))
import numpy as np, pandas as pd, torch
from t3lib import SPK, T3, META
import t3data as D
from t3train import pair_cos
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

O = T3 / "outputs/ultra"; O.mkdir(parents=True, exist_ok=True); EXT = T3 / "caches/ext"
import t3_cv_protocol as CV
SP = json.load(open(T3 / "cv_run/w/split.json")); PR = ("no_gender", "gender"); N = 3000


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr; i = np.nanargmin(np.abs(fnr - fpr)); return 100 * float(fpr[i] + fnr[i]) / 2


def v4():
    return CV.load_v4(T3 / "cv_run/b", T3 / "cv_run/w")


# ------------------------------------------------------------------ trials
def ext_tables():
    H = pd.read_csv(EXT / "ultra_holdout.csv"); tabs = {}
    for src in ("vox2", "vb"):
        r = pd.read_csv(EXT / f"rows_{src}.csv"); r["row"] = np.arange(len(r)); r["src"] = src; tabs[src] = r
    return H, tabs


def ext_trials(group, rng):
    H, tabs = ext_tables(); h = H[H.group.eq(group)]
    rows = pd.concat([tabs[s][tabs[s].identity.astype(str).isin(set(h.identity[h.src.eq(s)].astype(str)))] for s in ("vox2", "vb")])
    rows["g"] = rows.gender.astype(str).str.lower().str[0]; per = {k: g for k, g in rows.groupby("identity")}
    ids = np.array([k for k, g in per.items() if g.video.nunique() >= 2]); gid = {k: per[k].g.iloc[0] for k in ids}
    out = {}
    for pr in PR:
        F, V, Y = [], [], []
        for _ in range(N):  # positives: different videos of one person
            g = per[rng.choice(ids)]; v1, v2 = rng.choice(g.video.unique(), 2, replace=False)
            F.append(g[g.video.eq(v1)].sample(1, random_state=rng.integers(1 << 30)).iloc[0]); V.append(g[g.video.eq(v2)].sample(1, random_state=rng.integers(1 << 30)).iloc[0]); Y.append(1)
        for _ in range(N):
            a = rng.choice(ids); cand = ids[(ids != a) & ((np.array([gid[x] for x in ids]) == gid[a]) if pr == "gender" else True)]; b = rng.choice(cand)
            F.append(per[a].sample(1, random_state=rng.integers(1 << 30)).iloc[0]); V.append(per[b].sample(1, random_state=rng.integers(1 << 30)).iloc[0]); Y.append(0)
        out[pr] = dict(fsrc=np.array([f.src for f in F]), frow=np.array([f.row for f in F]), vsrc=np.array([v.src for v in V]), vrow=np.array([v.row for v in V]),
                       y=np.array(Y), vid=np.array([str(v.identity) for v in V]))
    return out


def trials():
    V4 = v4(); rng = np.random.default_rng(20261111); T = {}
    for pr in PR:
        for k, fold in enumerate(SP["folds"]):
            fi, vi, y = CV.make_trials(V4, fold, SP["gender"], N, pr, rng); T[f"f{k}_{pr}"] = dict(fi=fi, vi=vi, y=y)
        fi, vi, y = CV.make_trials(V4, SP["reserve"], SP["gender"], N, pr, rng); T[f"res_{pr}"] = dict(fi=fi, vi=vi, y=y)
    for grp in ("EN", "XL"):
        for pr, d in ext_trials(grp, rng).items(): T[f"{grp}_{pr}"] = d
    np.savez(O / "trials.npz", **{f"{k}__{n}": v for k, d in T.items() for n, v in d.items()})
    print({k: (len(d["y"]), int(d["y"].sum())) for k, d in T.items()})


def load_trials():
    Z = np.load(O / "trials.npz", allow_pickle=True); T = {}
    for key in Z.files:
        k, n = key.split("__"); T.setdefault(k, {})[n] = Z[key]
    return T


# ------------------------------------------------------------------ attributes: gender, age (fold-refit)
def attr():
    import t3_build as TB
    T = load_trials()
    pf = np.load(T3 / "caches/mivolo_face/face_pmale.npy"); g24 = np.load(T3 / "caches/flag_voiceage24.npz")["gender"]; pv = g24[:, 1] / (g24[:, 0] + g24[:, 1])
    a24, a6 = np.load(T3 / "caches/flag_voiceage24.npz"), np.load(T3 / "caches/flag_voiceage6.npz")
    fa = (TB.all_rows_face_vit() + np.load(T3 / "caches/mivolo_face/face_age.npy") / 100) / 2; H = TB.age_hidden(a24["emb"], a6["emb"])[:6485]; v24 = a24["age"][:6485]
    from sklearn.linear_model import RidgeCV
    def vage(rows):  # TB.voice_age_v2 fitted on the given FLAG rows only
        mu, sd = H[rows].mean(0), H[rows].std(0) + 1e-6; tgt = pd.Series(fa[rows]).groupby(SPK[rows]).median().loc[SPK[rows]].values
        r = RidgeCV(alphas=np.logspace(0, 5, 12)).fit((H[rows] - mu) / sd, tgt)
        off = float(np.median([np.median(fa[rows][SPK[rows] == s]) - np.median(v24[rows][SPK[rows] == s]) for s in np.unique(SPK[rows])]))
        return .5 * r.predict((H - mu) / sd) + .5 * np.clip(v24 + off, 0, 1)
    out = {}
    for k in list(range(5)) + ["res"]:
        rows = np.flatnonzero(np.isin(SPK, [p for j, f in enumerate(SP["folds"]) if (k == "res" or j != k) for p in f])); va = vage(rows)
        for pr in PR:
            key = f"f{k}_{pr}" if k != "res" else f"res_{pr}"; t = T[key]; fi, vi = t["fi"], t["vi"]
            out[f"{key}__gender"] = np.log(np.clip(pf[fi] * pv[vi] + (1 - pf[fi]) * (1 - pv[vi]), 1e-4, 1))
            out[f"{key}__age"] = -np.abs(fa[fi] - va[vi])
    np.savez(O / "attr.npz", **out); print("attr done")


# ------------------------------------------------------------------ heads
def _ext_face(src, fn):
    return np.load(EXT / f"{src}_face_{fn}.float32.npy", mmap_mode="r")


def _ext_voice(src, vn):
    names = D.CAT.get(vn, [vn]); arr = [D._l2(np.load(EXT / f"{src}_voice_{D._VOICE_KEY.get(n, n)}.float32.npy", mmap_mode="r")) for n in names]
    return np.hstack(arr) if len(arr) > 1 else np.load(EXT / f"{src}_voice_{D._VOICE_KEY.get(vn, vn)}.float32.npy", mmap_mode="r")


def heads(models, branches, tag=""):   # tag (13/11): save as head_<b><tag>.npz (e.g. "_s3" = 3 seeds per fold from models_s3)
    import t3_ultra_heads as UH
    T = load_trials(); models = Path(models)
    for b in branches.split(","):
        fn, vn, _, _ = UH.BR[b]; F, V = np.asarray(D.face(fn)), np.asarray(D.voice_full(vn)); out = {}
        ms = {k: [torch.load(p, weights_only=False) for p in sorted(models.glob(f"{b}__{k}__s*.pt"))] for k in range(5)}
        assert all(ms[k] for k in range(5)), (b, {k: len(v) for k, v in ms.items()})
        sc = lambda mm, F_, V_, fi, vi: np.mean([pair_cos(m, F_, V_, fi, vi) for m in mm], 0)
        for pr in PR:
            for k in range(5): t = T[f"f{k}_{pr}"]; out[f"f{k}_{pr}"] = sc(ms[k], F, V, t["fi"], t["vi"])
            t = T[f"res_{pr}"]; out[f"res_{pr}"] = np.mean([sc(ms[k], F, V, t["fi"], t["vi"]) for k in range(5)], 0)
            for grp in ("EN", "XL"):
                t = T[f"{grp}_{pr}"]; s = np.zeros(len(t["y"]))
                for fs in ("vox2", "vb"):
                    for vs in ("vox2", "vb"):
                        m_ = (t["fsrc"] == fs) & (t["vsrc"] == vs)
                        if not m_.any(): continue
                        Fe, Ve = _ext_face(fs, fn), _ext_voice(vs, vn); fr, vr = t["frow"][m_], t["vrow"][m_]
                        uf, fi = np.unique(fr, return_inverse=True); uv, vi = np.unique(vr, return_inverse=True)
                        s[m_] = np.mean([sc(ms[k], np.asarray(Fe[uf], np.float32), np.asarray(Ve[uv], np.float32), fi, vi) for k in range(5)], 0)
                out[f"{grp}_{pr}"] = s
        np.savez(O / f"head_{b}{tag}.npz", **out); print("scored", b, tag, {k: len(v) for k, v in ms.items()}, flush=True)


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    {"trials": trials, "attr": attr, "heads": heads}[cmd](*a)
