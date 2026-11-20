"""Validation for the XJ improvement loop (decided 14/11, before any variant result):
 V1 = XJ-protocol FLAG CV (7 gender-balanced id folds x 3 reps, fixed trials/seeds; paired deltas over 21 runs).
 V3 = MAV-Celeb v1 ENGLISH cross-video hold-out: 24 v1 ids (our 16 mav_holdout MV ids + 8 more, gender-balanced, seed 7) removed from
      training (all languages); trials = each held-out English voice clip x (face of the same id from ANOTHER video = 1, face of another
      held-out id = 0; gender protocol: same gender). 5 seeds averaged. Bootstrap over held-out ids for the paired delta SE.
usage: python xj_val.py <name> [key=val ...]; keys: epochs ratio mask(0/1) bn_ext(0/1, default 1) flag(none|all3|any) nseed_v3
       langs (comma list of ext clip languages; english,hindi,urdu,unk; voxceleb1 = unk; default all) gpure(0/1) hidden outd"""
import sys, json, time, numpy as np, pandas as pd, xj_lib as X, xj_ext as XE
from pathlib import Path
H = Path(__file__).resolve().parent; T3 = H.parent


def cfg(kv):
    return dict(epochs=int(kv.get("epochs", 3)), ratio=int(kv.get("ratio", 4)), mask=kv.get("mask", "0") == "1", bn_ext=kv.get("bn_ext", "1") == "1",
                flag=kv.get("flag", "none"), nseed_v3=int(kv.get("nseed_v3", 5)), fresh=int(kv.get("fresh", "0")), langs=kv.get("langs"), gpure=kv.get("gpure", "0") == "1",
                hidden=int(kv.get("hidden", 256)), outd=int(kv.get("outd", 128)), vb=kv.get("vb", "0") == "1", vb_frac=float(kv.get("vb_frac", 1.0)),
                mav_lid=kv.get("mav_lid", "0") == "1", vox2=kv.get("vox2", "0") == "1", mavx=kv.get("mavx", "0") == "1",
                bs=int(kv.get("bs", 128)), drop=float(kv.get("drop", 0.7)), mav_w=float(kv.get("mav_w", 1.0)), bnclip=kv.get("bnclip", "0") == "1")


def v3_ids():
    """FROZEN 14/11 (bug-check): the hold-out list is read from val_B0.json, because later Bengali-list updates (e.g. id0040 flagged by the
    Urdu/Hindi LID) would otherwise silently change the V3 trial set and break paired comparisons. Hold-out ids never enter training."""
    ids = pd.read_csv(XE.D / "ids.csv"); g = ids[ids.source.eq("mavceleb")].set_index("id").gender
    fz = H / "val_B0.json"
    if fz.exists(): return sorted(json.load(open(fz))["hold"]), g
    B = XE.bengali_keys(); ok = [i for i in g.index if g.get(i) in ("m", "f") and i in B["v1_cov"] and i not in B["v1_bad"]]
    hold = [i for i in pd.read_csv(T3 / "caches/ext/mav_holdout.csv").identity if i in ok]; rng = np.random.default_rng(7)
    rest = [i for i in ok if i not in hold]; need = {"m": 4, "f": 4}
    for i in rng.permutation(rest):
        if need[g[i]] > 0 and len(hold) < 24: hold.append(i); need[g[i]] -= 1
    return sorted(hold), g


def v3_trials(E, hold, g, seed=7):
    fm, vm = E.fm_all, E.vm_all; rng = np.random.default_rng(seed); keys = {f"mavceleb_v1/{i}": i for i in hold}; out = {"no_gender": [], "gender": []}
    fby = {k: d for k, d in fm[fm.k.isin(keys)].groupby("k")}
    for vi in np.flatnonzero(vm.k.isin(keys).to_numpy() & vm.lang.eq("english").to_numpy()):
        k, vid = vm.k.iat[vi], vm.video.iat[vi]; pos = fby[k][fby[k].video != vid]
        if not len(pos): continue
        fp = pos.index.values[rng.integers(len(pos))]
        for pr in out:
            cand = [c for c in keys if c != k and (pr == "no_gender" or g[keys[c]] == g[keys[k]])]; c = cand[rng.integers(len(cand))]
            out[pr] += [(vi, fp, 1, keys[k]), (vi, fby[c].index.values[rng.integers(len(fby[c]))], 0, keys[k])]
    return {p: pd.DataFrame(v, columns=["vi", "fi", "y", "pid"]) for p, v in out.items()}


def run(name, c):
    t0 = time.time(); L, F, V = X.load_flag(); Fp, Vp = X.prep_face(F), X.prep_voice(V)
    keep = ~X.flag_bengali_mask(L, c["flag"]) if c["flag"] != "none" else np.ones(len(L), bool)
    tr_args = dict(epochs=c["epochs"], ext_ratio=c["ratio"], mask_same=c["mask"], hidden=c["hidden"], outd=c["outd"], gpure=c["gpure"], bs=c.get("bs", 128), drop=c.get("drop", 0.7),
                   ext_gender=XE.gender_map() if c["gpure"] else None)
    langs = c["langs"].split(",") if c["langs"] else None
    # V1
    ek = dict(langs=langs, bengali_filter=c["bn_ext"], vb=c["vb"], vb_frac=c["vb_frac"], mav_lid_ok=c["mav_lid"], vox2=c["vox2"], mavx=c["mavx"], mav_w=c.get("mav_w", 1.0), bengali_clip=c.get("bnclip", False)); E = XE.Ext(**ek); R = []; S1, Y1, RI = ({"no_gender": [], "gender": []} for _ in range(3))
    f = int(c["fresh"]); reps = range(3 * f, 3 * f + 3)   # fresh = confirmation folds/trials/seeds never used for any decision (1 = conf set, 2 = fresh-2; 0/1 identical to before)
    for rep in reps:
        for k, te in enumerate(X.folds7(L, rep)):
            tr = (~L.id.isin(te).to_numpy()) & keep; s = rep * 7 + k + 1000 * f; T = X.trials(L, te, seed=s)
            m = X.train(Fp[tr], Vp[tr], seed=s, ext=E.sampler(E.drop_for(te)), ids=L.id.values[tr], genders=L.g.values[tr], **tr_args)
            sc = {p: X.score(m, Fp[T[p][:, 1]], Vp[T[p][:, 0]]) for p in ("no_gender", "gender")}
            for p in sc: S1[p].append(sc[p]); Y1[p].append(T[p][:, 2]); RI[p].append(np.full(len(sc[p]), len(R)))
            R.append([X.eer(T[p][:, 2], sc[p]) for p in ("no_gender", "gender")])
    R = np.array(R); del E
    # V3
    hold, g = v3_ids(); E3 = XE.Ext(**ek, exclude_keys=[f"mavceleb_v1/{i}" for i in hold])
    fm, vm = pd.read_csv(XE.D / "faces_meta.csv"), pd.read_csv(XE.D / "voices_meta.csv"); fm["k"], vm["k"] = XE.key(fm), XE.key(vm); E3.fm_all, E3.vm_all = fm, vm
    T3v = v3_trials(E3, hold, g, seed=7 + f); Fa = X.prep_face(np.load(XE.D / "faces.npy", mmap_mode="r").astype(np.float32)); Va = X.prep_voice(np.load(XE.D / "voices.npy").astype(np.float32))
    ms = [X.train(Fp[keep], Vp[keep], seed=500 + 200 * f + j, ext=E3.sampler(), ids=L.id.values[keep], genders=L.g.values[keep], **tr_args) for j in range(c["nseed_v3"])]
    s3 = {p: np.mean([X.score(m, Fa[t.fi.values], Va[t.vi.values]) for m in ms], 0) for p, t in T3v.items()}
    v3 = {p: X.eer(t.y.values, s3[p]) for p, t in T3v.items()}
    np.savez(H / f"val_{name}.npz", R=R, **{f"v1s_{p}": np.concatenate(S1[p]) for p in S1}, **{f"v1y_{p}": np.concatenate(Y1[p]) for p in Y1},
             **{f"v1run_{p}": np.concatenate(RI[p]) for p in RI}, **{f"v3_{p}": s3[p] for p in s3}, **{f"v3y_{p}": T3v[p].y.values for p in T3v}, **{f"v3pid_{p}": T3v[p].pid.values for p in T3v})
    json.dump(dict(name=name, cfg=c, v1=dict(ng=R[:, 0].mean(), g=R[:, 1].mean()), v3=v3, v3_trials={p: len(t) for p, t in T3v.items()}, hold=hold),
              open(H / f"val_{name}.json", "w"), indent=1)
    print(f"{name:14s} V1 ng {R[:,0].mean():.2f} g {R[:,1].mean():.2f} | V3 ng {v3['no_gender']:.2f} g {v3['gender']:.2f} (n {len(T3v['no_gender'])}) ({time.time()-t0:.0f}s)", flush=True)


def compare(a, b, nb=1000):
    """paired: V1 delta mean +- SE over 21 runs; V3 delta by bootstrap over held-out ids"""
    A, B = np.load(H / f"val_{a}.npz", allow_pickle=True), np.load(H / f"val_{b}.npz", allow_pickle=True); d = B["R"] - A["R"]; se = d.std(0, ddof=1) / np.sqrt(len(d))
    out = [f"{b} vs {a}: V1 ng {d[:,0].mean():+.2f} +- {se[0]:.2f} g {d[:,1].mean():+.2f} +- {se[1]:.2f}"]
    for p in ("no_gender", "gender"):
        y, pid = A[f"v3y_{p}"], A[f"v3pid_{p}"]; assert (y == B[f"v3y_{p}"]).all() and (pid == B[f"v3pid_{p}"]).all(), "V3 trial sets differ"; u = np.unique(pid); rng = np.random.default_rng(0); ds = []
        for _ in range(nb):
            ix = np.concatenate([np.flatnonzero(pid == x) for x in rng.choice(u, len(u))]); ds.append(X.eer(y[ix], B[f"v3_{p}"][ix]) - X.eer(y[ix], A[f"v3_{p}"][ix]))
        out.append(f"V3 {p} {X.eer(y, B[f'v3_{p}']) - X.eer(y, A[f'v3_{p}']):+.2f} +- {np.std(ds):.2f}")
    print(" | ".join(out))


def ensemble(name, members):
    """score-level mean of already-run configs (same folds/trials/seeds): V1 EER per run from the averaged trial scores; V3 from averaged scores."""
    Z = [np.load(H / f"val_{m}.npz", allow_pickle=True) for m in members]; out = {}
    for p in ("no_gender", "gender"):
        for k in (f"v1y_{p}", f"v1run_{p}", f"v3y_{p}", f"v3pid_{p}"): assert all((z[k] == Z[0][k]).all() for z in Z), k
        out[f"v1s_{p}"] = np.mean([z[f"v1s_{p}"] for z in Z], 0); out[f"v3_{p}"] = np.mean([z[f"v3_{p}"] for z in Z], 0)
        for k in (f"v1y_{p}", f"v1run_{p}", f"v3y_{p}", f"v3pid_{p}"): out[k] = Z[0][k]
    R = np.array([[X.eer(out[f"v1y_{p}"][out[f"v1run_{p}"] == r], out[f"v1s_{p}"][out[f"v1run_{p}"] == r]) for p in ("no_gender", "gender")]
                  for r in range(len(Z[0]["R"]))])
    for z in Z:   # bit-check: per-run EER of each member recomputed from its saved scores equals its stored R
        Rm = np.array([[X.eer(z[f"v1y_{p}"][z[f"v1run_{p}"] == r], z[f"v1s_{p}"][z[f"v1run_{p}"] == r]) for p in ("no_gender", "gender")] for r in range(len(R))])
        assert np.allclose(Rm, z["R"]), "stored R != recomputed"
    np.savez(H / f"val_{name}.npz", R=R, **out)
    v3 = {p: X.eer(out[f"v3y_{p}"], out[f"v3_{p}"]) for p in ("no_gender", "gender")}
    print(f"{name:14s} V1 ng {R[:,0].mean():.2f} g {R[:,1].mean():.2f} | V3 ng {v3['no_gender']:.2f} g {v3['gender']:.2f} (members {members})")


if __name__ == "__main__":
    if sys.argv[1] == "cmp": compare(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "ens": ensemble(sys.argv[2], sys.argv[3].split(","))
    else: run(sys.argv[1], cfg(dict(a.split("=") for a in sys.argv[2:])))
