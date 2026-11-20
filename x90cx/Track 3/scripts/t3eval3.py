"""Track 3 evaluation v3 — closer to the dev protocol.

Per fold k (4 folds) one model is trained with FLAG fold-fit IDs + external aux whose held-out
IDs are removed (MAV v1 held fold k, MAV v3 held fold k; VoxCeleb1 always in). It is scored on:
  FLAG-CAN : canonical cells (any-session positives)
  FLAG-XS  : FLAG cross-session positives (t3eval2)
  v3-XL / v1-XL : held external IDs, voice in one language, faces ONLY from the ID's videos in the
                  other language (EN<->DE for v3, EN<->Urdu for v1)  [mimics shared face pools in dev]
  v3-XV / v1-XV : same language, face from a different video
Protocols: standard and same_gender (v1 gender from meta_v1.csv; v3 gender from a FLAG-trained
FaRL gender head, ID majority). Summary metric EVAL3 = mean(FLAG-XS, v1-XL, v3-XL).
"""
from __future__ import annotations
import pickle, sys, json, numpy as np, pandas as pd
from t3lib import *
from t3train import train_projector, pair_cos
import t3data as D
import t3eval2 as E2

RNG_SEED = 0
V1META = pd.read_csv(ROOT / "external_mavceleb/v1_complete/meta_v1.csv")
V1G = dict(zip(V1META.ids, V1META.gender.str[0].str.lower()))


def ext_full(src, fn, vn):
    return D.ext_source(src, fn, vn)  # all faces, all voices, all languages


def id_folds(names, k=4, seed=0):
    u = np.array(sorted(set(names))); r = np.random.default_rng(seed).permutation(u)
    return [set(x) for x in np.array_split(r, k)]


_v3g = None
def v3_gender():
    global _v3g
    if _v3g is None:
        from t3_build import gender_head
        male = np.array([GENDER[s] == "m" for s in SPK], int)
        h = gender_head(np.asarray(D.face("farl")[:6485]), male, SPK)
        a = ext_full("v3", "farl", "sim"); p = h.predict_proba(a["faces"])[:, 1]
        _v3g = {n: ("m" if p[a["face_names"] == n].mean() > .5 else "f") for n in set(a["face_names"])}
    return _v3g


def ext_trials(src, held, other_lang, seed):
    """Trials over the uncapped external tables: list of (cond, proto) -> (face_idx, voice_idx, label, voice_name)."""
    a = ext_full(src, "farl", "sim")
    gender = V1G if src == "v1" else v3_gender()
    rng = np.random.default_rng(seed); out = {}
    hv = np.flatnonzero(np.isin(a["voice_names"], list(held)))
    hf = np.flatnonzero(np.isin(a["face_names"], list(held)))
    for cond in ("XL", "XV"):
        for proto in ("standard", "same_gender"):
            fr, vr, lab, spk = [], [], [], []
            for name in sorted(held):
                vs = hv[a["voice_names"][hv] == name]
                vs = rng.choice(vs, min(len(vs), 40), replace=False) if len(vs) else vs
                for v in vs:
                    L, vid = a["voice_lang"][v], a["voice_video"][v]
                    if cond == "XL":
                        pool = hf[a["face_lang"][hf] != L]
                    else:
                        pool = hf[(a["face_lang"][hf] == L) & (a["face_video"][hf] != vid)]
                    pos = pool[a["face_names"][pool] == name]
                    negn = sorted({n for n in a["face_names"][pool] if n != name and (proto == "standard" or gender.get(n) == gender.get(name))})
                    if not len(pos) or not negn:
                        continue
                    neg = pool[a["face_names"][pool] == negn[rng.integers(len(negn))]]
                    fr += [rng.choice(pos), rng.choice(neg)]; vr += [v, v]; lab += [1, 0]; spk += [name, name]
            out[cond, proto] = (np.array(fr), np.array(vr), np.array(lab), np.array(spk))
    return out


def eer_t(lab, spk, s):
    return eer(pd.DataFrame(dict(label=lab, voice_identity=spk.astype(str))), s)


def run(tag, fn, vn, kw, seeds=(2027,), use_v1a=None):
    path = T3 / f"outputs/eval3_{tag}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    F, V = np.asarray(D.face(fn)[:6485]), np.asarray(D.voice_full(vn)[:6485])
    use_v1a = (vn in ("sim", "wav")) if use_v1a is None else use_v1a
    v1base = D.mav(fn, vn)[0] if use_v1a else D.ext_source("v1", fn, vn, langs=["English"])
    v3cap = D.ext_source("v3", fn, vn, max_faces_per_video=20); vox = D.ext_source("vox", fn, vn)
    v1f, v3f = id_folds(ext_full("v1", "farl", "sim")["voice_names"]), id_folds(ext_full("v3", "farl", "sim")["voice_names"])
    tables = {"v1": ext_full("v1", fn, vn), "v3": ext_full("v3", fn, vn)}
    c2 = E2.cells(); res = {}
    for k in range(4):
        fit, _ = fold_ids(k); rows = np.flatnonzero(np.isin(SPK, fit))
        aux = [D.drop_ids(v1base, v1f[k]), D.drop_ids(v3cap, v3f[k]), vox]
        ms = [train_projector(F, V, rows, SPK, seed=s, aux=aux, **kw) for s in seeds]
        for p in ("standard", "same_gender"):
            fr, _ = cell(k, p)
            res["FLAG-CAN", p, k] = np.mean([pair_cos(m, F, V, fr.face_row.values, fr.voice_row.values) for m in ms], 0)
            fr = c2[k, p]
            res["FLAG-XS", p, k] = np.mean([pair_cos(m, F, V, fr.face_row.values, fr.voice_row.values) for m in ms], 0)
        for src, held in (("v1", v1f[k]), ("v3", v3f[k])):
            t = tables[src]; tr = ext_trials(src, held, None, seed=100 + k)
            for (cond, p), (fi, vi, lab, spk) in tr.items():
                res[f"{src}-{cond}", p, k] = np.mean([pair_cos(m, t["faces"], t["voices"], fi, vi) for m in ms], 0)
        print(tag, "fold", k, "done", flush=True)
    pickle.dump(res, open(path, "wb"))
    return res


_trials = {}
def trial_info(src, k):
    if (src, k) not in _trials:
        f = id_folds(ext_full(src, "farl", "sim")["voice_names"])[k]
        _trials[src, k] = ext_trials(src, f, None, seed=100 + k)
    return _trials[src, k]


def summarize(res_or_fn, verbose=True, tag=""):
    """res_or_fn: dict from run() (or callable (key, proto, fold) -> scores) -> metrics dict."""
    get = res_or_fn if callable(res_or_fn) else (lambda key, p, k: res_or_fn[key, p, k])
    c2 = E2.cells(); m = {}
    for key in ("FLAG-CAN", "FLAG-XS", "v1-XL", "v1-XV", "v3-XL", "v3-XV"):
        for p in ("standard", "same_gender"):
            e = []
            for k in range(4):
                s = get(key, p, k)
                if key == "FLAG-CAN":
                    e.append(eer(cell(k, p)[0], s))
                elif key == "FLAG-XS":
                    e.append(eer(c2[k, p], s))
                else:
                    src, cond = key.split("-"); fi, vi, lab, spk = trial_info(src, k)[cond, p]
                    e.append(eer_t(lab, spk, s))
            m[key, p] = float(np.mean(e))
    mm = {key: (m[key, "standard"] + m[key, "same_gender"]) / 2 for key in ("FLAG-CAN", "FLAG-XS", "v1-XL", "v1-XV", "v3-XL", "v3-XV")}
    mm["EVAL3"] = (mm["FLAG-XS"] + mm["v1-XL"] + mm["v3-XL"]) / 3
    if verbose:
        print(f"{tag:34s} " + "  ".join(f"{k} {v:.2f}" for k, v in mm.items()), flush=True)
    return mm, m


COMBO = dict(dim=512, w_proto=.6, w_con=.5, margin=.1, voice_dropout=.55, temp=.2, face_dropout=.85)
if __name__ == "__main__":
    fn, vn, variant = sys.argv[1], sys.argv[2], sys.argv[3]
    kw = dict(COMBO)
    if variant == "xv":
        kw.update(decouple=True, aux_xvideo=True)
    elif variant == "dec":
        kw.update(decouple=True)
    r = run(f"{fn}_{vn}_{variant}", fn, vn, kw)
    summarize(r, tag=f"{fn}_{vn}_{variant}")
