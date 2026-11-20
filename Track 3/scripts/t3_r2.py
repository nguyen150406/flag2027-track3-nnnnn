"""R2 (user 2026-11-12): our own cross-video contrastive face-voice member, written from scratch to replace the RF member.
Idea (diagnosis 12/11): every Track 3 head is FLAG-dominated (AAM over 48-70 FLAG identities) and loses 4-6 EER on the 10 reserve
persons; RF's projector learns mainly from thousands of EXTERNAL people with cross-video pairs and keeps its CV level on them.
R2 = per-modality LayerNorm -> Dropout -> Linear(512) projector trained with symmetric InfoNCE on cross-video face/voice pairs of
VoxCeleb2 + VoxBlink identities (512 per step), + a same-gender batch of external identities (gender from a probe fitted on FLAG
train labels only), + optionally a small FLAG same-gender term (w_flag). Inputs (our caches): face [L2 CLIP-L no-FLAG LoRA | L2
ImageBind-vision], voice cat4 [SimAM | SimAM-ft | ResNet293 | ECAPA-1024]. No code from the RF packages.
Held-out external persons (exclude_ultra.csv: Bengali v3 + 550 EN/XL) never train; RF split folds as t3_ultra_heads.
  python t3_r2.py fit <arm> <face clipnf|clipnf2> <folds 0,1,2,3,4|final> [seeds 1,2,3]   -> outputs/ultra/r2/<arm>_<face>__<fold>__s<seed>.pt
  python t3_r2.py score <arm> <face>                                                     -> outputs/ultra/head_r2<arm>_<face>.npz (t3_ultra_eval keys)"""
import os, sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
os.environ.setdefault("T3_EXCLUDE", str(HERE.parent / "caches/ext/exclude_ultra.csv"))
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import t3data as D
from t3lib import SPK, GENDER, T3
from sklearn.linear_model import LogisticRegression

assert "ultra" in os.environ["T3_EXCLUDE"], os.environ["T3_EXCLUDE"]
SPLIT = json.load(open(T3 / "cv_run/w/split.json")); OUT = T3 / "outputs/ultra/r2"; OUT.mkdir(parents=True, exist_ok=True)
CFG = dict(dim=512, drop_f=.7, drop_v=.5, lr=5e-4, wd=1e-4, steps=3000, warmup=100, n_ext=512, tau=.05, K=32,
           w_sgx=.5, n_sgx=256, w_flag=0.0, n_flag=32,
           srcs=("vox2", "vb"), w_hard=0.0, n_hard=256, age_band=.04, sa_w=1.0, xsess=False, w_mav=0.0, n_mav=32, w_sah=0.0, n_sah=256, drop="", vaux=False,
           w_id=0.0, id_m=.2, id_s=30.0, hard_w="", hard_gamma=2.0, ema=0.0,
           mlp=0, sa_thr=.5, sa_clean=False)   # sa_clean (15:10): fold models refit the FLAG-likeness probe on their own fit persons only; 13/11 ENX: mlp = hidden width of a 2-layer tower (0 = the linear R2 projector); sa_thr = FLAG-likeness cut of the EH hard set
ARMS = {"E0": {}, "E1": dict(w_flag=.5),          # E0 = external only (FLAG never trains R2); E1 = + FLAG same-gender term .5
        # 12/11 data / hard-pair screen (user "làm A, B, C"): one change each vs E1
        "Ev2": dict(w_flag=.5, srcs=("vox2",)), "Evb": dict(w_flag=.5, srcs=("vb",)),                      # which external source helps
        "EA": dict(w_flag=.5, w_hard=.5, xsess=True),   # A: + batch of same-gender, same-age-band external ids; FLAG positives from different-looking clips
        "EB": dict(w_flag=.5, sa_w=3.0),                # B: South-Asian-looking external ids sampled 3x more often
        "EC": dict(w_flag=.5, srcs=("vox2", "vb", "v1")),  # C1: + MAV-Celeb v1 (48 ids; 16 held out as the MV validation group)
        # 13/11 English fix at the representation level (corrected MAV proxy: every stream ~26-27 EER on same-gender English pairs)
        "EM": dict(w_flag=.5, srcs=("vox2", "vb", "mavx"), w_mav=.5),   # + 40 MAV English TRAIN people (t3_mavx_source.py) and a same-gender MAV batch
        "EH": dict(w_flag=.5, w_sah=.5),                                  # + batch of same-gender South-Asian-looking external ids (hard negatives)
        "EMH": dict(w_flag=.5, srcs=("vox2", "vb", "mavx"), w_mav=.5, w_sah=.5),
        # 13/11 (user "hãy thử nghiệm hết"): one change each vs E1
        "E1b": dict(w_flag=.5),                                            # = E1, extra seeds (4..10) stored under their own name
        "EP": dict(w_flag=.5, drop="outputs/ultra/pool_drop10.csv"),      # Wen et al. 2021: drop the 10% least learnable pool identities
        "EV": dict(w_flag=.5, vaux=True),                                  # Simicch-style: voice input + audeering age/gender embeddings (a24|a6)
        # 13/11 research loop (web sweep 13/11): one change each vs E1
        "EI": dict(w_flag=.5, w_id=.5),          # + shared face/voice identity classifier (AAM m .2 s 30, learnable prototypes; Wen 2021 implicit alignment)
        "EW": dict(w_flag=.5, drop="outputs/ultra/pool_drop10.csv", hard_w="outputs/ultra/pool_learnability.csv"),  # Wen 2021 full: drop 10% + upweight hard ids (1..3x)
        "ED9": dict(w_flag=.5, drop_f=.9, drop_v=.9),  # Simicch input dropout .9
        # 13/11 EMA .9995 of the projector weights: same recipe as E1, EMA weights saved
        "EE": dict(w_flag=.5, ema=.9995),
        # 13/11 English v2 (session diagnosis: FLAG-trained parts learn a same-recording shortcut; eval English positives look cross-session)
        "EPX": dict(w_flag=.5, drop="outputs/ultra/pool_drop10.csv", xsess=True),     # EP + FLAG positives only from different-looking clips
        "EPXM": dict(w_flag=.5, drop="outputs/ultra/pool_drop10.csv", xsess=True, srcs=("vox2", "vb", "mavx"), w_mav=.5),  # + 40 MAV English TRAIN people
        # 13/11 shortcut trade-off: FLAG same-clip positives teach a same-recording cue (CV) while cross-video pool pairs teach session-robust matching (HARD/MAV/EN)
        "EPL": dict(w_flag=.25, drop="outputs/ultra/pool_drop10.csv"),   # EP with half the FLAG loss weight
        # 13/11 afternoon multi-factor re-check (multi1.txt): EH (same-gender FLAG-like external hard negatives) = only arm with a large gain on the
        # clean MAV-ALL proxy (gender -2.15, 97% of 15-person subsets better). Robustness checks: dose (w_sah 1.0), extra seeds, + EP pool drop.
        "EH2": dict(w_flag=.5, w_sah=1.0),                                  # dose-response: twice the SA hard-negative weight
        "EHb": dict(w_flag=.5, w_sah=.5),                                   # = EH, extra seeds (4..6) under their own name
        "EHP": dict(w_flag=.5, w_sah=.5, drop="outputs/ultra/pool_drop10.csv"),   # EH + EP pool drop
        # 13/11 ENX (new English-only system, design outputs/ultra/ENX_DESIGN.md): one change each vs EH
        "XM": dict(w_flag=.5, w_sah=.5, mlp=1024),     # non-linear 2-layer towers (capacity for the cross-modal map)
        "XA": dict(w_flag=.5, w_sah=.5, w_hard=.5),    # + same-gender age-close external hard batch (identity cues beyond gender / age)
        "XS": dict(w_flag=.5, w_sah=.5, sa_thr=.3),    # wider FLAG-like hard set (more South-Asian-looking ids per batch)
        # 13/11 15:10 leak audit: the EH probe (Pool.sa) saw all 70 FLAG-train persons' faces, incl. the fold's held-out / reserve persons
        "EHc": dict(w_flag=.5, w_sah=.5, sa_clean=True)}   # = EH with a fold-clean FLAG-likeness probe (final model: same 70 persons as EH)
SRCS = ("vox2", "vb")                              # default sources with ImageBind-vision faces
MAV_HO = set(pd.read_csv(T3 / "caches/ext/mav_holdout.csv").identity.astype(str))
U = lambda X: X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
NTR = len(SPK)                                     # FLAG train rows 0..6484


def flag_feats(fn, vaux=False):
    Fm = np.hstack([U(np.asarray(D.face(fn), np.float32)), U(np.asarray(D.face("ibv"), np.float32))]); Vm = np.asarray(D.voice_full("cat4"), np.float32)
    if vaux:   # audeering 24 / 6-layer hidden means of every FLAG row (caches/flag_voiceage{24,6}.npz, same extractor as eval / MAV / pool)
        A = [U(np.asarray(np.load(T3 / f"caches/flag_voiceage{k}.npz")["emb"], np.float32))[:len(Vm)] for k in (24, 6)]; Vm = np.hstack([Vm] + A)
    return Fm, Vm


def vaux_input(cat4, a24, a6):
    """voice input of a vaux arm for any rows: [cat4 | L2(a24 hidden) | L2(a6 hidden)]"""
    return np.hstack([cat4, U(np.asarray(a24, np.float32)), U(np.asarray(a6, np.float32))]).astype(np.float32)


def ext_full(src, fn, vaux=False):  # every row of rows_<src>.csv (scoring held-out trials) + usable masks (training)
    E = D.EXT; rows = pd.read_csv(E / f"rows_{src}.csv")
    Fm = np.hstack([U(np.load(E / f"{src}_face_{fn}.float32.npy")), U(np.load(E / f"{src}_face_ibv.float32.npy"))]).astype(np.float32)
    Vm = np.hstack([D._l2(np.load(E / f"{src}_voice_{D._VOICE_KEY.get(n, n)}.float32.npy")) for n in D.CAT["cat4"]]).astype(np.float32)
    fok = np.load(E / f"{src}_face_{fn}.ok.npy") & np.load(E / f"{src}_face_ibv.ok.npy")
    vok = np.logical_and.reduce([np.load(E / f"{src}_voice_{D._VOICE_KEY.get(n, n)}.ok.npy") for n in D.CAT["cat4"]])
    if vaux:   # caches/ext/<src>_voice_audk{24,6}.float32.npy (kaggle/build_t3_aud_kernel.py via t3_aud_install.py, per row)
        Vm = vaux_input(Vm, *[np.nan_to_num(np.load(E / f"{src}_voice_audk{k}.float32.npy")) for k in (24, 6)])
        vok = vok & np.load(E / f"{src}_voice_audk24.ok.npy") & np.load(E / f"{src}_voice_audk6.ok.npy")
    ex = np.isin(rows.identity.astype(str).values, D.excluded(src).astype(str))
    return dict(rows=rows, F=np.nan_to_num(Fm), V=np.nan_to_num(Vm), fok=fok & ~ex, vok=vok & ~ex,
                ident=rows.identity.astype(str).values, fvid=rows.face.str.split("/").str[-2].values, vvid=rows.voice.str.split("/").str[-2].values)


def pair_table(fid, fvid, vid, vvid, K, rng):
    """identity -> K (face row, voice row) pairs, face from a different video than the voice whenever the identity has one."""
    fg, vg = pd.Series(np.arange(len(fid))).groupby(fid).apply(np.array), pd.Series(np.arange(len(vid))).groupby(vid).apply(np.array)
    names = sorted(set(fg.index) & set(vg.index)); PF, PV = np.zeros((len(names), K), np.int64), np.zeros((len(names), K), np.int64)
    for i, n in enumerate(names):
        fr, vr = fg[n], vg[n]; vs = rng.choice(vr, K)
        for j, v in enumerate(vs):
            other = fr[fvid[fr] != vvid[v]]; PF[i, j] = rng.choice(other if len(other) else fr); PV[i, j] = v
    return names, PF, PV


class Proj(nn.Module):
    def __init__(self, df, dv, c):
        super().__init__()
        h = c.get("mlp", 0)   # 0 -> exactly the original linear towers (same parameter names, old checkpoints load unchanged)
        tower = lambda d, p: (nn.Sequential(nn.LayerNorm(d), nn.Dropout(p), nn.Linear(d, c["dim"])) if not h else
                              nn.Sequential(nn.LayerNorm(d), nn.Dropout(p), nn.Linear(d, h), nn.GELU(), nn.Dropout(.1), nn.Linear(h, c["dim"])))
        self.f = tower(df, c["drop_f"]); self.v = tower(dv, c["drop_v"])


DEV = torch.device(os.environ.get("R2_DEVICE", "cpu"))


def aam(e, W, y, m_, s_):
    """additive angular margin softmax of embeddings e against identity prototypes W (EI arm)"""
    c = F.normalize(e, dim=-1) @ F.normalize(W, dim=-1).T; th = torch.acos(c.gather(1, y[:, None]).clamp(-1 + 1e-6, 1 - 1e-6))
    return F.cross_entropy(c.scatter(1, y[:, None], torch.cos(th + m_)) * s_, y)


def nce(fe, ve, tau):
    fe, ve = F.normalize(fe, dim=-1), F.normalize(ve, dim=-1); lg = fe @ ve.T / tau; y = torch.arange(len(fe), device=fe.device)
    return .5 * (F.cross_entropy(lg, y) + F.cross_entropy(lg.T, y))


class Pool:
    """External training pool (vox2 + VoxBlink minus exclusions): features, cross-video pair table, identity gender."""
    def __init__(self, fn, Ff, Vf, seed=0, srcs=SRCS, drop="", vaux=False):
        rng = np.random.default_rng(seed); Fs, Vs, fid, fvid, vid, vvid = [], [], [], [], [], []
        for src in srcs:
            e = ext_full(src, fn, vaux)
            ho = np.isin(e["ident"], sorted(MAV_HO | {"id0025", "id0028", "id0041"})) if src in ("v1", "mavx") else np.zeros(len(e["ident"]), bool)  # MV + vox2 dups never train
            fi, vi = np.flatnonzero(e["fok"] & ~ho), np.flatnonzero(e["vok"] & ~ho)
            Fs.append(e["F"][fi]); Vs.append(e["V"][vi]); fid += [f"{src}|{x}" for x in e["ident"][fi]]; vid += [f"{src}|{x}" for x in e["ident"][vi]]
            fvid += list(e["fvid"][fi]); vvid += list(e["vvid"][vi])
        self.F, self.V = torch.from_numpy(np.vstack(Fs)), torch.from_numpy(np.vstack(Vs))
        fid, vid, fvid, vvid = map(np.array, (fid, vid, fvid, vvid))
        self.names, self.PF, self.PV = pair_table(fid, fvid, vid, vvid, CFG["K"], rng)
        if drop:   # EP: identities listed in <drop> (column identity = "src|id") never train
            dr = set(pd.read_csv(T3 / drop).identity.astype(str)); keep = [i for i, n in enumerate(self.names) if n not in dr]
            print(f"pool drop: {len(self.names) - len(keep)} of {len(self.names)} identities ({drop})", flush=True)
            self.names = [self.names[i] for i in keep]; self.PF, self.PV = self.PF[keep], self.PV[keep]
        # identity gender from probes fitted on FLAG TRAIN labels only (face and voice), averaged over the identity's rows
        y = np.array([GENDER[s] == "m" for s in SPK]); pf = LogisticRegression(max_iter=2000, C=.5).fit(Ff[:NTR], y); pv = LogisticRegression(max_iter=2000, C=.5).fit(Vf[:NTR], y)
        P = pd.concat([pd.Series(pf.predict_proba(self.F.numpy())[:, 1], index=fid), pd.Series(pv.predict_proba(self.V.numpy())[:, 1], index=vid)]).groupby(level=0).mean()
        self.male = P.reindex(self.names).to_numpy() > .5; self.fid = fid
        # identity age and "FLAG-likeness" (South-Asian-looking) from probes fitted on FLAG train faces only (A / B arms)
        import t3_build as TB
        from sklearn.linear_model import RidgeCV
        fa = (TB.all_rows_face_vit()[:NTR] + np.load(T3 / "caches/mivolo_face/face_age.npy")[:NTR] / 100) / 2
        pa = RidgeCV(alphas=np.logspace(-2, 3, 8)).fit(Ff[:NTR], fa); Fn = self.F.numpy()
        self.age = pd.Series(pa.predict(Fn), index=fid).groupby(level=0).median().reindex(self.names).to_numpy()
        sel = rng.choice(len(Fn), NTR, replace=False); ps = LogisticRegression(max_iter=2000, C=.5).fit(np.vstack([Ff[:NTR], Fn[sel]]), np.r_[np.ones(NTR), np.zeros(NTR)])
        self.sa = pd.Series(ps.predict_proba(Fn)[:, 1], index=fid).groupby(level=0).mean().reindex(self.names).to_numpy()
        print(f"pool sources {srcs}: age median {np.median(self.age) * 100:.1f}, FLAG-like ids {np.mean(self.sa > .5):.3f}", flush=True)
        print(f"pool: {len(self.names)} identities, {len(fid)} faces, {len(vid)} voices, male share {self.male.mean():.2f}", flush=True)


def xsess_pairs(persons, K=64, seed=0):
    rng = np.random.default_rng(seed); Fc = U(np.nan_to_num(np.asarray(D.face("clip"), np.float32)[:NTR])); out = {}
    for s_ in sorted(persons):
        r = np.flatnonzero(SPK == s_); S = Fc[r] @ Fc[r].T; off = ~np.eye(len(r), dtype=bool); thr = np.median(S[off]) if off.any() else 0
        i, j = np.nonzero((S < thr) & off)
        if not len(i): i, j = np.nonzero(off) if off.any() else (np.zeros(1, int), np.zeros(1, int))   # degenerate person: any distinct pair
        k = rng.integers(0, len(i), K); out[s_] = np.stack([r[i[k]], r[j[k]]], 1)
    return out


def fit_one(pool, Ff, Vf, persons, cfg, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); torch.set_num_threads(int(os.environ.get("R2_THREADS", 4)))
    m = Proj(pool.F.shape[1], pool.V.shape[1], cfg).to(DEV).train(); PF_, PV_ = torch.from_numpy(pool.PF).to(DEV), torch.from_numpy(pool.PV).to(DEV)
    if pool.F.device != DEV: pool.F, pool.V = pool.F.to(DEV), pool.V.to(DEV)
    T_ = lambda a: torch.from_numpy(np.asarray(a, np.int64)).to(DEV)
    n = len(pool.names); W = nn.Parameter(torch.randn(n, cfg["dim"], device=DEV) * .01) if cfg["w_id"] else None   # EI prototypes (not saved)
    params = list(m.parameters()) + ([W] if W is not None else [])
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["wd"])
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / cfg["warmup"]))
    n, K = len(pool.names), pool.PF.shape[1]; gid = [np.flatnonzero(pool.male == g) for g in (False, True)]; bad = 0
    if cfg["w_flag"]:
        rows = np.flatnonzero(np.isin(SPK, sorted(persons))); by = pd.Series(rows).groupby(SPK[rows]).apply(np.array)
        fl = {g: [s for s in by.index if (GENDER[s] == "m") == g] for g in (False, True)}; Ft, Vt = torch.from_numpy(Ff[:NTR]).to(DEV), torch.from_numpy(Vf[:NTR]).to(DEV)
        XS = xsess_pairs(by.index, seed=seed) if cfg["xsess"] else None
    pw = np.where(pool.sa > .5, cfg["sa_w"], 1.0); pw = pw / pw.sum() if cfg["sa_w"] != 1.0 else None
    if cfg["hard_w"]:   # EW: sampling weight 1 + gamma * (1 - percentile of the identity's E1 learnability AUC) -> hardest kept ids ~3x
        au = pd.read_csv(T3 / cfg["hard_w"]).set_index("identity").auc.reindex(pool.names); assert au.notna().mean() > .99, au.notna().mean()
        pw = 1 + cfg["hard_gamma"] * (1 - au.rank(pct=True).fillna(.5).to_numpy()); pw = pw / pw.sum(); print(f"    hard weights max/min {pw.max() / pw.min():.2f}", flush=True)
    if cfg["w_mav"]:   # MAV ids by their metadata gender (t3data.gender_map, MAV v1 metadata since 13/11)
        gm = D.gender_map(); mi = np.array([i for i, nm in enumerate(pool.names) if nm.startswith("mavx|")]); assert len(mi) >= 30, len(mi)
        mg = [mi[np.array([bool(gm[pool.names[i].split("|", 1)[1]]) == g for i in mi])] for g in (False, True)]
        print(f"    MAV batch ids f/m {len(mg[0])}/{len(mg[1])}", flush=True)
    if cfg["w_sah"]:
        sa = pool.sa
        if cfg["sa_clean"]:   # probe on this model's fit persons only (own rng: the training rng stream is unchanged)
            r_ = np.flatnonzero(np.isin(SPK, sorted(persons))); Fn = pool.F.cpu().numpy(); sel = np.random.default_rng(0).choice(len(Fn), len(r_), replace=False)
            ps = LogisticRegression(max_iter=2000, C=.5).fit(np.vstack([Ff[:NTR][r_], Fn[sel]]), np.r_[np.ones(len(r_)), np.zeros(len(r_))])
            sa = pd.Series(ps.predict_proba(Fn)[:, 1], index=pool.fid).groupby(level=0).mean().reindex(pool.names).to_numpy()
            print(f"    fold-clean probe on {len(set(SPK[r_]))} persons: overlap with the 70-person set {np.mean((sa > cfg['sa_thr'])[pool.sa > cfg['sa_thr']]):.2f}", flush=True)
        sah = [np.flatnonzero((pool.male == g) & (sa > cfg["sa_thr"])) for g in (False, True)]; print(f"    SA hard ids f/m {len(sah[0])}/{len(sah[1])}", flush=True)
    ema = {k_: v_.detach().clone() for k_, v_ in m.state_dict().items()} if cfg["ema"] else None
    for t in range(cfg["steps"]):
        p = rng.choice(n, min(cfg["n_ext"], n), replace=False, p=pw); k = rng.integers(0, K, len(p))
        p, k = T_(p), T_(k); fe0, ve0 = m.f(pool.F[PF_[p, k]]), m.v(pool.V[PV_[p, k]]); loss = nce(fe0, ve0, cfg["tau"])
        if W is not None: loss = loss + cfg["w_id"] * .5 * (aam(fe0, W, p, cfg["id_m"], cfg["id_s"]) + aam(ve0, W, p, cfg["id_m"], cfg["id_s"]))
        if cfg["w_sgx"]:
            ids = gid[rng.integers(2)]; q = rng.choice(ids, min(cfg["n_sgx"], len(ids)), replace=False); k2 = rng.integers(0, K, len(q))
            q, k2 = T_(q), T_(k2); loss = loss + cfg["w_sgx"] * nce(m.f(pool.F[PF_[q, k2]]), m.v(pool.V[PV_[q, k2]]), cfg["tau"])
        if cfg["w_hard"]:  # A: same gender AND same age band (anchor +- age_band, widened if too few) -> identity cues beyond gender / age
            a = rng.integers(n); band = cfg["age_band"]
            while True:
                c = np.flatnonzero((pool.male == pool.male[a]) & (np.abs(pool.age - pool.age[a]) < band))
                if len(c) >= 64 or band > .5: break
                band *= 2
            q = rng.choice(c, min(cfg["n_hard"], len(c)), replace=False); k3 = rng.integers(0, K, len(q))
            q, k3 = T_(q), T_(k3); loss = loss + cfg["w_hard"] * nce(m.f(pool.F[PF_[q, k3]]), m.v(pool.V[PV_[q, k3]]), cfg["tau"])
        if cfg["w_mav"]:
            q = mg[rng.integers(2)]; q = rng.choice(q, min(cfg["n_mav"], len(q)), replace=False); k4 = rng.integers(0, K, len(q))
            q, k4 = T_(q), T_(k4); loss = loss + cfg["w_mav"] * nce(m.f(pool.F[PF_[q, k4]]), m.v(pool.V[PV_[q, k4]]), cfg["tau"])
        if cfg["w_sah"]:
            q = sah[rng.integers(2)]; q = rng.choice(q, min(cfg["n_sah"], len(q)), replace=False); k5 = rng.integers(0, K, len(q))
            q, k5 = T_(q), T_(k5); loss = loss + cfg["w_sah"] * nce(m.f(pool.F[PF_[q, k5]]), m.v(pool.V[PV_[q, k5]]), cfg["tau"])
        if cfg["w_flag"]:
            ss = fl[bool(rng.integers(2))]; ss = rng.choice(ss, min(cfg["n_flag"], len(ss)), replace=False)
            if XS is not None:
                pr_ = np.array([XS[s][rng.integers(len(XS[s]))] for s in ss]); fr, vr = pr_[:, 0], pr_[:, 1]
            else:
                fr = np.array([rng.choice(by[s]) for s in ss]); vr = np.array([rng.choice(by[s][by[s] != f] if len(by[s]) > 1 else by[s]) for s, f in zip(ss, fr)])
            loss = loss + cfg["w_flag"] * nce(m.f(Ft[T_(fr)]), m.v(Vt[T_(vr)]), cfg["tau"])
        if not torch.isfinite(loss):  # guard (12/11: one MPS run went NaN; a clean rerun of the same seed did not) -> skip the step, abort if it repeats
            bad += 1; print(f"    step {t}: non-finite loss, skipped ({bad})", flush=True); assert bad <= 20, "R2 training diverged"; continue
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step(); sch.step()
        if ema is not None:   # EE: exponential moving average of the weights (decay ramps in over the first steps)
            d_ = min(cfg["ema"], (1 + t) / (10 + t))
            with torch.no_grad():
                for k_, v_ in m.state_dict().items(): ema[k_].mul_(d_).add_(v_.detach(), alpha=1 - d_) if v_.is_floating_point() else ema[k_].copy_(v_)
        if t % 1000 == 0 or t == cfg["steps"] - 1: print(f"    step {t} loss {loss.item():.3f}", flush=True)
    if ema is not None: m.load_state_dict(ema)
    return m.cpu().eval()


def fold_persons(fold):
    if fold == "final": return set(SPK)
    return {p for j, f in enumerate(SPLIT["folds"]) if j != int(fold) for p in f}


def fit(arm, fn, folds, seeds="1,2,3"):
    cfg = dict(CFG, **ARMS[arm])
    if os.environ.get("R2_STEPS"): cfg["steps"] = int(os.environ["R2_STEPS"])  # smoke only
    Ff, Vf = flag_feats(fn, cfg["vaux"]); pool = Pool(fn, Ff, Vf, srcs=tuple(cfg["srcs"]), drop=cfg["drop"], vaux=cfg["vaux"])
    for fold in folds.split(","):
        for s in map(int, seeds.split(",")):
            p = (OUT / "smoke" if os.environ.get("R2_STEPS") else OUT) / f"{arm}_{fn}__{fold}__s{s}.pt"; p.parent.mkdir(exist_ok=True)
            if p.exists() and not os.environ.get("R2_STEPS"): continue
            t0 = time.time(); m = fit_one(pool, Ff, Vf, fold_persons(fold), cfg, s); torch.save(m.state_dict(), p)
            print(f"{arm} {fn} fold {fold} seed {s}: {(time.time() - t0) / 60:.1f} min", flush=True)


@torch.no_grad()
def cos(models, Fa, Va):
    Fa, Va = torch.from_numpy(np.ascontiguousarray(Fa, np.float32)), torch.from_numpy(np.ascontiguousarray(Va, np.float32))
    return np.mean([(F.normalize(m.f(Fa), dim=-1) * F.normalize(m.v(Va), dim=-1)).sum(1).numpy() for m in models], 0)


def load(arm, fn, fold, df, dv):
    ms = []
    seeds = os.environ.get("R2_SEEDS")  # screening: e.g. "1" -> only seed-1 models
    for p in sorted(OUT.glob(f"{arm}_{fn}__{fold}__s*.pt")):
        if seeds and p.stem.split("__s")[-1] not in seeds.split(","): continue
        m = Proj(df, dv, dict(CFG, **ARMS.get(arm, {}))); m.load_state_dict(torch.load(p)); ms.append(m.eval())   # arm cfg: mlp towers (13/11)
    assert ms, (arm, fn, fold); return ms


def mv_trials(n=3000, seed=20261112):
    """MV = 16 held-out MAV-Celeb v1 identities (caches/ext/mav_holdout.csv), ENGLISH voices; positives = face from a different video
    than the voice; negatives = another held-out identity (no_gender: any; gender: same gender). Our own sampler, fixed seed."""
    r = pd.read_csv(D.EXT / "rows_v1.csv"); g = D.gender_map(); rng = np.random.default_rng(seed); ident = r.identity.astype(str).values
    fvid, vvid = r.face.str.split("/").str[-2].values, r.voice.str.split("/").str[-2].values; ho = sorted(MAV_HO)
    V = {s_: np.flatnonzero((ident == s_) & r.language.eq("English").values) for s_ in ho}; Fr = {s_: np.flatnonzero(ident == s_) for s_ in ho}; T = {}
    for pr in ("no_gender", "gender"):
        fi, vi, y = [], [], []
        while len(y) < n:
            a = ho[rng.integers(len(ho))]; v = rng.choice(V[a]); c = Fr[a][fvid[Fr[a]] != vvid[v]]
            if len(c): fi.append(rng.choice(c)); vi.append(v); y.append(1)
        while len(y) < 2 * n:
            a, b = ho[rng.integers(len(ho))], ho[rng.integers(len(ho))]
            if a == b or (pr == "gender" and bool(g[a]) != bool(g[b])): continue
            fi.append(rng.choice(Fr[b])); vi.append(rng.choice(V[a])); y.append(0)
        T[pr] = dict(fi=np.array(fi), vi=np.array(vi), y=np.array(y))
    return T


def score(arm, fn):
    import t3_ultra_eval as E
    va = dict(CFG, **ARMS[arm])["vaux"]; T = E.load_trials(); Ff, Vf = flag_feats(fn, va); X = {s: ext_full(s, fn, va) for s in SRCS}; out = {}
    M = {k: load(arm, fn, k, Ff.shape[1], Vf.shape[1]) for k in range(5)}; assert len({len(m) for m in M.values()}) == 1, {k: len(m) for k, m in M.items()}
    for pr in E.PR:
        for k in range(5): t = T[f"f{k}_{pr}"]; out[f"f{k}_{pr}"] = cos(M[k], Ff[t["fi"]], Vf[t["vi"]])
        t = T[f"res_{pr}"]; out[f"res_{pr}"] = np.mean([cos(M[k], Ff[t["fi"]], Vf[t["vi"]]) for k in range(5)], 0)
        for grp in ("EN", "XL"):
            t = T[f"{grp}_{pr}"]; Fa = np.vstack([X[s]["F"][r] for s, r in zip(t["fsrc"], t["frow"])]); Va = np.vstack([X[s]["V"][r] for s, r in zip(t["vsrc"], t["vrow"])])
            out[f"{grp}_{pr}"] = np.mean([cos(M[k], Fa, Va) for k in range(5)], 0)
            if va: out[f"{grp}_{pr}_ok"] = np.abs(Va[:, 960:]).sum(1) > 0     # voice rows that have audeering features (missing -> zero block)
    if (D.EXT / f"v1_face_{fn}.float32.npy").exists() and (D.EXT / "v1_face_ibv.float32.npy").exists() and (not va or (D.EXT / "v1_voice_audk24.float32.npy").exists()):  # old v1 MV; vaux arms need v1 audk caches (none -> skip; gate uses outputs/mav_en2)
        X1 = ext_full("v1", fn, va); MV = mv_trials()
        for pr in E.PR: t = MV[pr]; out[f"MV_{pr}"] = np.mean([cos(M[k], X1["F"][t["fi"]], X1["V"][t["vi"]]) for k in range(5)], 0); out[f"MVy_{pr}"] = t["y"]
    tag = f"_s{os.environ['R2_SEEDS'].replace(',', '')}" if os.environ.get("R2_SEEDS") else ""
    np.savez(T3 / f"outputs/ultra/head_r2{arm}_{fn}{tag}.npz", **out); print("scored", arm, fn, tag, flush=True)


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    {"fit": fit, "score": score}[cmd](*a)
