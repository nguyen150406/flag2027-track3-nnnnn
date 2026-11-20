"""HB (17/11, prereg PREREG_gate_all.txt 09:45 "PREREG sub119a"): a second system FAMILY next to XJ-1+IB (different data,
objectives and fuser).
  data   : VoxCeleb2 + VoxBlink cached features only (no FLAG, no MAV, no VoxCeleb1), clip-level LID v3 Bengali rule + exclude_bengali_v3 ids,
           FLAG-train duplicate guard (ext ids whose SimAM voice centroid has cos > .83 with a FLAG-TRAIN id centroid are dropped)
  blocks : face CLIP-L (frozen) | FaRL ; voice SimAM-R100 | ResNet293 | WavLM-SV   (each L2, concat / sqrt(n))
  S1     : xj_lib.train_nf towers, defaults, 10 seeds, same-video pairs
  S2     : regularised CCA on identity means (reg .1 x mean diag, k 128)
  attrs  : P(male) face / voice = logistic regressions on the pool's m/f labels
  fuser  : per protocol LR on [z1, z2, pmf, pmv, d, z1*d, z2*d], d = |pmf - pmv|, fitted on the kit CV trials (fold-nested for the OOF)
usage: python hb.py run <outdir> <oof.npz>     |   python hb.py gate <oof.npz>   (exit 0 pass / 3 fail)
       HB_SAVE=1 python hb.py run ...  -> also <outdir>/ckpt/   |   python hb.py score <outdir>/ckpt <newdir>  (eval scores from the checkpoint)"""
import os, sys, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
XD = Path(__file__).resolve().parent; T3 = XD.parent; sys.path.insert(0, str(XD))
import xj_lib as X
torch.set_num_threads(int(os.environ.get("HB_THREADS", 4)))
CX, EV = T3 / "caches/ext", T3 / "outputs/eval_final"
FB = tuple(os.environ.get("HB_FB", "clip,farl").split(","))   # 17/11 PM: HB_FB adds frozen ImageBind face "ibv" or IB-LoRA face "ibl" (default = HB)
# 17/11 10:40 variants (env; defaults = HB of 09:45): HB_SRCS sources, HB_VB voice blocks, HB_FLAG=1 adds FLAG-train pairs to S1 (fold-wise OOF)
SRCS = tuple(os.environ.get("HB_SRCS", "vox2,vb").split(",")); VB = tuple(os.environ.get("HB_VB", "simam,r293,wsv").split(",")); HBF = os.environ.get("HB_FLAG", "0") == "1"
MODEL = os.environ.get("HB_MODEL", "HB")
FLAGF, FLAGV = {"clip": "clip", "farl": "farl"}, {"simam": "sim", "r293": "r293", "wsv": "wsv"}
EVF = {"clip": "feat/face_clip.npy", "farl": "feat/face_farl.npy"}
EVV = {"simam": "feat/voice_sim.npy", "r293": "feat/voice_r293.npy", "wsv": "feat/voice_wsv.npy"}
SEEDS, KCCA, REG, DUP = [1000 * i for i in range(10)], 128, .1, .83
T0 = time.time()
def log(m): print(f"[{(time.time() - T0) / 60:5.1f}m] {m}", flush=True)
def U(a): a = np.asarray(a, np.float64); return a / np.linalg.norm(a, axis=1, keepdims=True)
def blk(arrs): return (np.concatenate([U(a) for a in arrs], 1) / np.sqrt(len(arrs))).astype(np.float32)
BNW = int(os.environ.get("HB_BNW", "0"))   # 17/11 21:00 prereg 1: FLAG-train rows with a Bengali-LID voice seen 1 + BNW times by the S1 towers (0 = unchanged)
def bn_train(Ff, Vf, bnm, rows, seed, draw, ratio):
    idx = np.flatnonzero(rows); idx = np.concatenate([idx] + [idx[bnm[idx]]] * BNW)   # duplicated Bengali rows; centre = the unduplicated training rows
    return X.train(Ff[idx], Vf[idx], seed=seed, epochs=3, ext=draw, ext_ratio=ratio, center_rows=(Ff[rows], Vf[rows]))
PEMU = {}
def pe_c(a, side):   # 17/11 19:25: raw PE-AV embeddings share one dominant direction (FLAG-train mean pair cos: face .999, voice .69) -> centre by the
    if side not in PEMU: PEMU[side] = np.load(T3 / f"outputs/ultra/pe_flag_{side}.npy").astype(np.float64).mean(0)   # FLAG-train mean (fixed train constant) before the block L2
    return np.asarray(a, np.float64) - PEMU[side]
CFG = dict(model=MODEL, data=f"cached features of {SRCS}" + (" + FLAG-train pairs in S1" if HBF else "; no FLAG / MAV / VoxCeleb1"), face="+".join(FB), voice="+".join(VB),
           S1="xj_lib.train_nf defaults (256->128, drop .7, 3 epochs, ratio 4), 10 seeds, same-video pairs", S2=f"CCA reg {REG} k {KCCA}",
           attrs="P(male) LR on pool m/f labels", fuser="per-protocol LR [z1,z2,pmf,pmv,d,z1*d,z2*d]", dup_guard=DUP, seeds=SEEDS,
           prereg="PREREG_gate_all.txt 17/11 09:45 sub119a")


def lid_ok():
    parts = [pd.read_csv(T3 / f"outputs/lid_v3/{d}/lid/lid_{i}of2.csv") for d in ("vox", "web") for i in range(2)]
    v = pd.concat(parts); v = v[v.src.isin(list(SRCS))]; cols = ["vl_bnas", "wh_bnas", "mms_bnas"]
    fl = ~(v[cols] <= .5).all(axis=1) | ~(v.dur > 0)
    d = pd.DataFrame({"k": v.key.values, "fl": fl.to_numpy()}).groupby("k").fl.max()
    return set(d.index[~d.values]), len(d), int(d.sum())


def flag():
    sys.path.insert(0, str(T3 / "scripts")); cwd = os.getcwd(); os.chdir(T3 / "scripts")
    import t3data as D
    F = blk([(np.load(T3 / "outputs/ultra/ib_own_flag_face.npy") if n == "ibv" else np.load(T3 / "outputs/ultra/ibl_flag_face.npy") if n == "ibl" else pe_c(np.load(T3 / "outputs/ultra/pe_flag_face.npy"), "face") if n == "pe" else np.asarray(D.face(FLAGF[n]), np.float32)[:6485]) for n in FB]); V = blk([(np.load(T3 / "outputs/ultra/ib_own_flag_voice.npy") if n == "ibv" else np.load(T3 / "outputs/ultra/ibl_flag_voice.npy") if n == "ibl" else pe_c(np.load(T3 / "outputs/ultra/pe_flag_voice.npy"), "voice") if n == "pe" else np.asarray(D.voice_full(FLAGV[n]), np.float32)[:6485]) for n in VB])
    os.chdir(cwd); L, _, _ = X.load_flag()
    meta = pd.read_csv(T3.parent / "flag_raw_s01_v3_adaface_simam100/metadata.csv").sort_values("row_id"); t = meta[meta.partition.eq("train")]
    assert (L.voice.values == t.voice_relpath.values).all() and F.shape[0] == V.shape[0] == 6485 and np.isfinite(F).all() and np.isfinite(V).all()
    return L, F, V


def eval_feats():
    t = pd.read_csv(EV / "rows.csv")
    F = blk([pe_c(np.load(EV / "ultra/pe_face.npy"), "face") if n == "pe" else np.load(EV / ("ultra/ib_face.npy" if n == "ibv" else "ultra/ibl_face.npy" if n == "ibl" else EVF[n])) for n in FB]); V = blk([pe_c(np.load(EV / "ultra/pe_voice.npy"), "voice") if n == "pe" else np.load(EV / ("ultra/ib_voice.npy" if n == "ibv" else "ultra/ibl_voice.npy" if n == "ibl" else EVV[n])) for n in VB]); assert len(F) == len(V) == len(t) and np.isfinite(F).all() and np.isfinite(V).all()
    return t, F, V


class Pool:
    def __init__(self, L, Vflag):
        ok_clip, nchk, nfl = lid_ok(); ex = pd.read_csv(CX / "exclude_bengali_v3.csv"); FR, VR, FA, VA = [], [], [], []
        for src in SRCS:
            r = pd.read_csv(CX / f"rows_{src}.csv", keep_default_na=False); n = len(r)
            if "gender" not in r: r["gender"] = "u"
            fa = [pe_c(np.load(CX / f"{src}_face_{b}.float32.npy"), "face") if b == "pe" else np.load(CX / f"{src}_face_{b}.float32.npy", mmap_mode="r") for b in FB]; va = [pe_c(np.load(CX / f"{src}_voice_{b}.float32.npy"), "voice") if b == "pe" else np.load(CX / f"{src}_voice_{b}.float32.npy", mmap_mode="r") for b in VB]
            okf = np.logical_and.reduce([np.load(CX / f"{src}_face_{b}.ok.npy") for b in FB]); okv = np.logical_and.reduce([np.load(CX / f"{src}_voice_{b}.ok.npy") for b in VB])
            assert all(len(a) == n for a in fa + va) and len(okf) == len(okv) == n, src
            bad = set(ex[ex.src.eq(src)].identity.astype(str)); keep_id = ~r.identity.astype(str).isin(bad).to_numpy()
            k = (src + "/" + r.identity.astype(str)).to_numpy(); lk = (src + ":" + r.voice.astype(str)).to_numpy()
            lidok = np.array([x in ok_clip for x in lk]); vfirst = ~r.voice.duplicated().to_numpy()
            mv = okv & keep_id & lidok & vfirst; mf = okf & keep_id
            log(f"{src}: rows {n}, ids {r.identity.nunique()}, excluded ids {len(bad & set(r.identity.astype(str)))}, voice clips kept {int(mv.sum())} / {int(vfirst.sum())} (LID-cleared {int((lidok & vfirst).sum())})")
            VR.append(pd.DataFrame(dict(k=k[mv], video=r.video.astype(str).values[mv], g=r.gender.values[mv]))); VA.append(blk([a[mv] for a in va]))
            FR.append(pd.DataFrame(dict(k=k[mf], video=r.video.astype(str).values[mf], g=r.gender.values[mf]))); FA.append(blk([a[mf] for a in fa]))
        vm, fm = pd.concat(VR, ignore_index=True), pd.concat(FR, ignore_index=True); V, F = np.concatenate(VA), np.concatenate(FA)
        # FLAG-train duplicate guard on the SimAM block (first 256 dims / sqrt(3) of the voice input -> renormalised)
        assert VB[0] == "simam"; sim = lambda A: U(A[:, :256])
        cf = pd.DataFrame(sim(Vflag)).groupby(L.id.values).mean(); cf = U(cf.to_numpy())
        ce = pd.DataFrame(sim(V)).groupby(vm.k.values).mean(); ids_e = ce.index.to_numpy(); ce = U(ce.to_numpy())
        mx = (ce @ cf.T).max(1); dup = set(ids_e[mx > DUP]); log(f"FLAG-train duplicate guard: {len(dup)} ext ids dropped (max cos {mx.max():.3f}) {sorted(dup)[:10]}")
        keepv = ~vm.k.isin(dup).to_numpy(); keepf = fm.k.isin(set(vm.k[keepv])).to_numpy()
        self.vm, self.V, self.fm, self.F = vm[keepv].reset_index(drop=True), V[keepv], fm[keepf].reset_index(drop=True), F[keepf]
        self.byk = {k: np.asarray(v) for k, v in self.fm.groupby("k").indices.items()}; self.bykv = {k: np.asarray(v) for k, v in self.fm.groupby(["k", "video"]).indices.items()}
        assert set(self.vm.k) <= set(self.byk), "every kept voice identity needs faces"
        self.info = dict(voices=int(len(self.vm)), faces=int(len(self.fm)), ids=int(self.vm.k.nunique()), dup_dropped=sorted(dup), lid_checked=nchk, lid_flagged=nfl)
        log(f"HB pool: {self.info['voices']} voices, {self.info['faces']} faces, {self.info['ids']} ids")

    def sampler(self):
        ks, vids = self.vm.k.to_numpy(), self.vm.video.to_numpy()
        def draw(rng, n):
            vi = rng.choice(len(ks), min(n, len(ks)), replace=False); fi = np.empty(len(vi), int)
            for j, v in enumerate(vi):
                c = self.bykv.get((ks[v], vids[v])); c = c if c is not None and len(c) else self.byk[ks[v]]; fi[j] = c[rng.integers(len(c))]
            return self.F[fi], self.V[vi], ks[vi]
        return draw


def cca_fit(P):
    idsv = P.vm.k.to_numpy(); ids = np.unique(idsv)
    Fm = pd.DataFrame(P.F).groupby(P.fm.k.values).mean().loc[ids].to_numpy(); Vm = pd.DataFrame(P.V).groupby(idsv).mean().loc[ids].to_numpy()
    mx, my = Fm.mean(0), Vm.mean(0); A, B = Fm - mx, Vm - my; n = len(ids)
    Cxx, Cyy, Cxy = A.T @ A / n, B.T @ B / n, A.T @ B / n
    Cxx += REG * np.mean(np.diag(Cxx)) * np.eye(len(Cxx)); Cyy += REG * np.mean(np.diag(Cyy)) * np.eye(len(Cyy))
    def isq(C): w, Q = np.linalg.eigh(C); return Q @ np.diag(w ** -.5) @ Q.T
    Ix, Iy = isq(Cxx), isq(Cyy); Uu, s, Vt = np.linalg.svd(Ix @ Cxy @ Iy)
    Wx, Wy = Ix @ Uu[:, :KCCA], Iy @ Vt[:KCCA].T; log(f"CCA on {n} identities: top canonical corr {s[:5].round(3).tolist()} ... k={KCCA}: {s[KCCA - 1]:.3f}")
    return dict(mx=mx, my=my, Wx=Wx, Wy=Wy)


def cca_proj(C, F, V): return (F - C["mx"]) @ C["Wx"], (V - C["my"]) @ C["Wy"]


def cos_c(zf, zv, cf, cv): a, b = U(zf - cf), U(zv - cv); return (a * b).sum(1)


def run(out, oofp):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    L, Ff, Vf = flag(); te, Fe, Ve = eval_feats(); P = Pool(L, Vf); CFG["pool"] = P.info
    # ---- attributes: P(male)
    gf, gv = P.fm.g.isin(["m", "f"]).to_numpy(), P.vm.g.isin(["m", "f"]).to_numpy()
    lf = LogisticRegression(C=1.0, max_iter=3000).fit(P.F[gf], (P.fm.g.values[gf] == "m").astype(int))
    lv = LogisticRegression(C=1.0, max_iter=3000).fit(P.V[gv], (P.vm.g.values[gv] == "m").astype(int))
    pmf_t, pmv_t, pmf_e, pmv_e = lf.predict_proba(Ff)[:, 1], lv.predict_proba(Vf)[:, 1], lf.predict_proba(Fe)[:, 1], lv.predict_proba(Ve)[:, 1]
    gm = L.g.map({"m": 1, "f": 0}).to_numpy(); log(f"P(male) on FLAG train vs metadata gender: face acc {np.mean((pmf_t > .5) == gm):.3f}, voice acc {np.mean((pmv_t > .5) == gm):.3f}")
    # ---- S1 deep towers (no FLAG); FLAG rows only give the centre
    BNM = X.flag_bengali_mask(L, "any") if BNW else None
    if BNW: log(f"BN-weight: {int(BNM.sum())} Bengali-LID FLAG rows x {1 + BNW}")
    draw = P.sampler(); ms = []
    for i, sd in enumerate(SEEDS):
        ms.append((bn_train(Ff, Vf, BNM, np.ones(len(Ff), bool), sd, draw, 4) if BNW else X.train(Ff, Vf, seed=sd, epochs=3, ext=draw, ext_ratio=4)) if HBF else X.train_nf(Ff, Vf, seed=sd, ext=draw, ext_ratio=4)); log(f"S1 seed {sd} done")
    with torch.no_grad():
        ET = [m.emb(torch.from_numpy(Ff), torch.from_numpy(Vf)) for m in ms]; EE = [m.emb(torch.from_numpy(Fe), torch.from_numpy(Ve)) for m in ms]
    ET = [(a.numpy().astype(np.float64), b.numpy().astype(np.float64)) for a, b in ET]; EE = [(a.numpy().astype(np.float64), b.numpy().astype(np.float64)) for a, b in EE]
    def s1(rows_c, fi, vi, E=ET): return np.mean([cos_c(a[fi], b[vi], a[rows_c].mean(0), b[rows_c].mean(0)) for a, b in E], 0)
    FOLDM = {}
    if HBF:   # fold-wise retraining: FLAG rows of the held-out fold (and the reserve) never train the towers used for that fold's OOF
        J0 = json.load(open(T3 / "release/sub90c/local_eval_kit/folds.json"))
        for k in range(5):
            held = set(J0["reserve"]) | set(J0["folds"][k]); tr_ = ~L.id.isin(held).to_numpy()
            FOLDM[k] = [(bn_train(Ff, Vf, BNM, tr_, 1000 * j + k, draw, 4) if BNW else X.train(Ff[tr_], Vf[tr_], seed=1000 * j + k, epochs=3, ext=draw, ext_ratio=4)) for j in range(len(SEEDS))]; log(f"S1 fold {k} models done")
    def s1e(): return np.mean([cos_c(ea, eb, a.mean(0), b.mean(0)) for (a, b), (ea, eb) in zip(ET, EE)], 0)
    # ---- S2 CCA
    C = cca_fit(P); zfT, zvT = cca_proj(C, Ff.astype(np.float64), Vf.astype(np.float64)); zfE, zvE = cca_proj(C, Fe.astype(np.float64), Ve.astype(np.float64))
    def s2(rows_c, fi, vi): return cos_c(zfT[fi], zvT[vi], zfT[rows_c].mean(0), zvT[rows_c].mean(0))
    # ---- stream scores on the kit CV trials (centre = fold-train FLAG rows, unlabelled)
    sys.path.insert(0, str(T3 / "scripts")); cwd = os.getcwd(); os.chdir(T3 / "scripts")
    import t3_ultra_eval as E
    T = E.load_trials(); os.chdir(cwd); J = json.load(open(T3 / "release/sub90c/local_eval_kit/folds.json")); S = {}
    for k in range(5):
        held = set(J["reserve"]) | set(J["folds"][k]); trr = np.flatnonzero(~L.id.isin(held).to_numpy())
        for pr in E.PR:
            t = T[f"f{k}_{pr}"]; fi, vi = t["fi"], t["vi"]; d = np.abs(pmf_t[fi] - pmv_t[vi])
            if HBF: assert not set(L.id.values[fi]) & set(L.id.values[trr]); s1v = np.mean([X.score(m, Ff[fi], Vf[vi]) for m in FOLDM[k]], 0)
            else: s1v = s1(trr, fi, vi)
            S[(k, pr)] = dict(s1=s1v, s2=s2(trr, fi, vi), pmf=pmf_t[fi], pmv=pmv_t[vi], d=d, y=t["y"].astype(int))
    def feats(s, st): z1, z2 = (s["s1"] - st[0]) / st[1], (s["s2"] - st[2]) / st[3]; return np.c_[z1, z2, s["pmf"], s["pmv"], s["d"], z1 * s["d"], z2 * s["d"]]
    def stats(keys): a1 = np.concatenate([S[q]["s1"] for q in keys]); a2 = np.concatenate([S[q]["s2"] for q in keys]); return (a1.mean(), a1.std(), a2.mean(), a2.std())
    oof, rep, fin = {}, {}, {}
    for pr in E.PR:
        for k in range(5):
            tk = [(j, pr) for j in range(5) if j != k]; st = stats(tk)
            lr = LogisticRegression(C=1.0, max_iter=3000).fit(np.concatenate([feats(S[q], st) for q in tk]), np.concatenate([S[q]["y"] for q in tk]))
            oof[f"f{k}_{pr}"] = lr.decision_function(feats(S[(k, pr)], st))
        ally = np.concatenate([S[(k, pr)]["y"] for k in range(5)]); cat = lambda n: np.concatenate([S[(k, pr)][n] for k in range(5)])
        rep[pr] = dict(S1=round(X.eer(ally, cat("s1")), 2), S2=round(X.eer(ally, cat("s2")), 2), HB_oof=round(X.eer(ally, np.concatenate([oof[f"f{k}_{pr}"] for k in range(5)])), 2))
        st = stats([(k, pr) for k in range(5)]); lr = LogisticRegression(C=1.0, max_iter=3000).fit(np.concatenate([feats(S[(k, pr)], st) for k in range(5)]), ally)
        fin[pr] = (st, lr); rep[pr]["coef"] = dict(zip(["z1", "z2", "pmf", "pmv", "d", "z1d", "z2d"], lr.coef_[0].round(3).tolist())); log(f"{pr}: {rep[pr]}")
    np.savez(oofp, **oof, oof_cfg=json.dumps(dict(model=MODEL, nested="LR fold-nested on kit CV trials", streams=rep)))
    # ---- final eval scores
    e1 = np.mean([X.score(m, Fe, Ve) for m in ms], 0) if HBF else s1e()
    e2 = cos_c(zfE, zvE, zfT.mean(0), zvT.mean(0)); info = {}
    for tr in ("no_gender", "gender"):
        st, lr = fin[tr]; d = np.abs(pmf_e - pmv_e); s = lr.decision_function(feats(dict(s1=e1, s2=e2, pmf=pmf_e, pmv=pmv_e, d=d), st))
        for lang, tag in (("English", "heard"), ("Bangla", "unheard")):
            msk = (te.protocol.eq(tr) & te.language.eq(lang)).to_numpy(); c = s[msk]; assert np.isfinite(c).all()
            ids = te.pair_id.astype(str)[msk].values; ref = [l.split()[0] for l in open(XD / "xj1ib_10" / tr / f"sub_score_v4_{lang}_{tag}.txt")]; assert list(ids) == ref, "eval row order"
            np.save(out / f"cos_{tr}_{lang}.npy", c); p = out / tr / f"sub_score_v4_{lang}_{tag}.txt"; p.parent.mkdir(exist_ok=True)
            p.write_text("".join(f"{a} {-b:.8f}\n" for a, b in zip(ids, c))); info[f"{tr}/{lang}"] = int(len(c))
    CFG["report"] = rep; json.dump(CFG, open(out / "config.json", "w"), indent=1, default=str); log(f"final {info}")
    if os.environ.get("HB_SAVE") == "1": save_ckpt(out, ms, C, zfT, zvT, lf, lv, fin)


def save_ckpt(out, ms, C, zfT, zvT, lf, lv, fin):
    """18/11 (release): checkpoint of everything the FINAL eval scores use (additive; does not change any computation).
    ckpt/s1_m{i}.pt = S1 towers (centre buffers inside), cca.npz = CCA + FLAG-train projected means, heads.pkl = P(male) probes + per-protocol fusers."""
    import pickle
    assert HBF, "checkpoint export implemented for HB_FLAG=1 models (the released ones)"
    d = Path(out) / "ckpt"; d.mkdir(exist_ok=True)
    for i, m in enumerate(ms): torch.save(m.state_dict(), d / f"s1_m{i}.pt")
    np.savez(d / "cca.npz", mx=C["mx"], my=C["my"], Wx=C["Wx"], Wy=C["Wy"], cf=zfT.mean(0), cv=zvT.mean(0))
    pickle.dump(dict(model=MODEL, FB=FB, VB=VB, HBF=HBF, nseed=len(ms), lf=lf, lv=lv, fin=fin), open(d / "heads.pkl", "wb")); log(f"checkpoint -> {d}")


def score(ckdir, out):
    """18/11 (release): eval scores from a saved checkpoint (no training, no pool): must equal the run's cos files exactly."""
    import pickle
    d, out = Path(ckdir), Path(out); out.mkdir(parents=True, exist_ok=True); H = pickle.load(open(d / "heads.pkl", "rb"))
    assert (H["model"], tuple(H["FB"]), tuple(H["VB"]), H["HBF"]) == (MODEL, FB, VB, HBF), f"env does not match checkpoint {H['model']} {H['FB']} {H['VB']}"
    te, Fe, Ve = eval_feats(); ms = []
    for i in range(H["nseed"]):
        m = X.XJ(Fe.shape[1], Ve.shape[1]); m.load_state_dict(torch.load(d / f"s1_m{i}.pt", map_location="cpu")); m.eval(); ms.append(m)
    C = dict(np.load(d / "cca.npz")); e1 = np.mean([X.score(m, Fe, Ve) for m in ms], 0)
    zfE, zvE = cca_proj(C, Fe.astype(np.float64), Ve.astype(np.float64)); e2 = cos_c(zfE, zvE, C["cf"], C["cv"])
    pmf_e, pmv_e = H["lf"].predict_proba(Fe)[:, 1], H["lv"].predict_proba(Ve)[:, 1]
    def feats(s, st): z1, z2 = (s["s1"] - st[0]) / st[1], (s["s2"] - st[2]) / st[3]; return np.c_[z1, z2, s["pmf"], s["pmv"], s["d"], z1 * s["d"], z2 * s["d"]]
    info = {}
    for tr in ("no_gender", "gender"):
        st, lr = H["fin"][tr]; dd = np.abs(pmf_e - pmv_e); s = lr.decision_function(feats(dict(s1=e1, s2=e2, pmf=pmf_e, pmv=pmv_e, d=dd), st))
        for lang, tag in (("English", "heard"), ("Bangla", "unheard")):
            msk = (te.protocol.eq(tr) & te.language.eq(lang)).to_numpy(); c = s[msk]; assert np.isfinite(c).all(); ids = te.pair_id.astype(str)[msk].values
            np.save(out / f"cos_{tr}_{lang}.npy", c); p = out / tr / f"sub_score_v4_{lang}_{tag}.txt"; p.parent.mkdir(exist_ok=True)
            p.write_text("".join(f"{a} {-b:.8f}\n" for a, b in zip(ids, c))); info[f"{tr}/{lang}"] = int(len(c))
    import shutil
    if (d.parent / "config.json").exists(): shutil.copy2(d.parent / "config.json", out / "config.json")   # fuse_ab checks the model name
    log(f"scored from checkpoint {d}: {info}")


def gate(oofp):
    K = pd.read_csv(T3 / "release/sub90c/local_eval_kit/trials.csv.gz", keep_default_na=False); O, XO = np.load(oofp), np.load(XD / "xj1ib_oof_x1ib_10.npz"); ok, res = True, {}
    assert json.loads(str(O["oof_cfg"]))["model"].startswith("HB")
    for tr in ("no_gender", "gender"):
        y = K[K.protocol.eq(tr) & K.set.eq("CV")].y.to_numpy(); a = np.concatenate([O[f"f{k}_{tr}"].astype(float) for k in range(5)]); b = np.concatenate([XO[f"f{k}_{tr}"].astype(float) for k in range(5)])
        ea, eb = X.eer(y, a), X.eer(y, b); assert abs(eb - {"no_gender": 14.49, "gender": 21.58}[tr]) < .015, f"alignment canary {tr} {eb}"
        za, zb = (a - a.mean()) / a.std(), (b - b.mean()) / b.std(); rp = float(np.corrcoef(za[y == 1], zb[y == 1])[0, 1])
        r = dict(HB_oof=round(ea, 2), xj1ib_oof=round(eb, 2), zsum_oof=round(X.eer(y, za + zb), 2), corr_pos=round(rp, 3), GB1=bool(ea <= eb + 3.0), GB2=bool(rp <= .75))
        ok &= r["GB1"] and r["GB2"]; res[tr] = r; print(tr, json.dumps(r), flush=True)
    json.dump(res, open(Path(oofp).with_suffix(".gate.json"), "w"), indent=1); print("GATES-HB", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 3)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "run": run(a[1], a[2])
    elif a[0] == "gate": gate(a[1])
    elif a[0] == "score": score(a[1], a[2])
