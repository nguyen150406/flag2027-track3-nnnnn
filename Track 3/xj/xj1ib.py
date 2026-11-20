"""XJ-1+IB (15/11, user "Ý XJ-1 + ImageBind: thêm khối ImageBind vào XJ-1, vì đây là loại tín hiệu XJ-1 chưa có. thử đi"; prereg
PREREG_gate_all.txt 22:27 sub109a): XJ-1 (xj1.py) with ONE extra L2 block per side = frozen ImageBind-huge (scripts/t3_imagebind_extract.py):
  face  input = [the 6 XJ-1 face blocks | IB vision] / sqrt(7)
  voice input = [the 7 XJ-1 voice blocks | IB audio] / sqrt(8)
Everything else = XJ-1 unchanged (xj_lib.train defaults, 10 seeds 1000*i, same external pool and clip-level Bengali rule; external items need
ALL encoders of their side incl. IB). IB features: FLAG train = outputs/ultra/ib_own_flag_{face,voice}.npy (XP65 extraction, META train order),
eval = outputs/eval_final/ultra/ib_{face,voice}.npy (rows.csv order), external pool = Kaggle ncamtu06/flag-t3-xj1ib-extract
(kaggle/build_xj1ib_extract.py, parity-gated) unpacked in xj/xj1ib/.
usage: python xj1ib.py final <outdir> [nseed=10]   |   python xj1ib.py oof <tag> [nseed=10]   |   python xj1ib.py score <ckdir> <outdir> [nseed=10]"""
import sys, os, json
from pathlib import Path
import numpy as np, pandas as pd, torch
XD = Path(__file__).resolve().parent; T3 = XD.parent; sys.path.insert(0, str(XD))
import xj_lib as X, xj_ext as XE, xjc as XC, xju as XUm, xj1 as X1M
CX, XA, XB, XV, EXF, X1, XI = X1M.CX, X1M.XA, X1M.XB, X1M.XV, X1M.EXF, X1M.X1, XD / "xj1ib"
U, blk, ok_rows, FN, VN = X1M.U, X1M.blk, X1M.ok_rows, X1M.FN, X1M.VN
DF, DV = X1M.DF + 1024, X1M.DV + 1024


def add_ib(F, V, ibf, ibv, nf=6, nv=7):
    """XJ-1 inputs (concat of nf / nv unit blocks / sqrt(n)) -> append the unit IB block, rescale to / sqrt(n + 1)"""
    F2 = np.hstack([F * np.sqrt(nf), U(ibf)]) / np.sqrt(nf + 1); V2 = np.hstack([V * np.sqrt(nv), U(ibv)]) / np.sqrt(nv + 1)
    return F2.astype(np.float32), V2.astype(np.float32)


def flag():
    L, F, V = X1M.flag()
    ibf, ibv = np.load(T3 / "outputs/ultra/ib_own_flag_face.npy"), np.load(T3 / "outputs/ultra/ib_own_flag_voice.npy")
    assert ibf.shape == ibv.shape == (6485, 1024) and ok_rows(ibf).all() and ok_rows(ibv).all()
    # order check (ib_own = META train sorted by row_id; XJ-1 L = the same order): same-person face pairs are closer than different-person ones
    rng = np.random.default_rng(0); i, j = rng.integers(0, len(L), 20000), rng.integers(0, len(L), 20000); s = L.id.values[i] == L.id.values[j]
    cf = (U(ibf)[i] * U(ibf)[j]).sum(1); cv = (U(ibv)[i] * U(ibv)[j]).sum(1)
    assert cf[s].mean() - cf[~s].mean() > .1 and cv[s].mean() - cv[~s].mean() > .1, (cf[s].mean(), cf[~s].mean(), cv[s].mean(), cv[~s].mean())
    F2, V2 = add_ib(F, V, ibf, ibv); assert F2.shape == (6485, DF) and V2.shape == (6485, DV)
    return L, F2, V2


class ExtIB(XC.ExtC):
    """= xj1.ExtOne with the IB block appended on both sides (MAV: xj1ib rows [0:len(xjc)] in xjc_rows order; VoxCeleb1: voices in rows_vox order,
    faces = unique rows_vox faces). Code copied from xj1.ExtOne; the only changes are marked # IB."""
    def __init__(self):
        chk, fl = XE.clip_lid(); ok_lid = lambda keys: np.array([k in chk and k not in fl for k in keys])
        ex = pd.read_csv(T3 / "caches/ext/exclude_bengali_v3.csv"); india = set(ex[ex.src.eq("vox")].identity.astype(str))
        Eb_f, Eb_v = np.load(EXF / "faces.npy", mmap_mode="r"), np.load(EXF / "voices.npy", mmap_mode="r")
        fmeta, vmeta = pd.read_csv(EXF / "faces_meta.csv"), pd.read_csv(EXF / "voices_meta.csv"); assert len(fmeta) == len(Eb_f) and len(vmeta) == len(Eb_v)
        VR, FR, VA, FA = [], [], [], []
        # IB arrays + rows
        IFr, IVr = pd.read_csv(XI / "xj1ib_face_rows.csv"), pd.read_csv(XI / "xj1ib_voice_rows.csv")
        IF, IV = np.load(XI / "xj1ib_face.float32.npy"), np.load(XI / "xj1ib_voice.float32.npy"); IFok, IVok = np.load(XI / "xj1ib_face.ok.npy"), np.load(XI / "xj1ib_voice.ok.npy")
        assert len(IF) == len(IFr) == len(IFok) and len(IV) == len(IVr) == len(IVok) and IF.shape[1] == IV.shape[1] == 1024
        # ---- MAV xjc set: same rows in xja / xjb / xj1 / xj1ib
        R = pd.read_csv(T3 / "kaggle/xjc_rows/xjc_rows.csv")
        assert R.equals(pd.read_csv(XA / "xja_rows.csv")) and R.equals(pd.read_csv(XB / "xjb_rows.csv")), "xja / xjb rows = xjc_rows"
        vr, fr = R[R.kind.eq("voice")].reset_index(drop=True), R[R.kind.eq("face")].reset_index(drop=True)
        assert pd.read_csv(X1 / "xj1_face_rows.csv").equals(fr[["src", "rel"]]) and pd.read_csv(X1 / "xj1_voice_rows.csv").equals(vr[["src", "rel"]]), "xj1 rows = xjc_rows"
        assert IFr.iloc[:len(fr)].reset_index(drop=True).equals(fr[["src", "rel"]]) and IVr.iloc[:len(vr)].reset_index(drop=True).equals(vr[["src", "rel"]]), "xj1ib MAV rows"  # IB
        assert all(vmeta.path.values[i].endswith(p) for i, p in zip(vr.xj_row, vr.rel)) and all(fmeta.path.values[i].endswith(p) for i, p in zip(fr.xj_row, fr.rel)), "xj_row -> ext_feats"
        bv = np.asarray(Eb_v[vr.xj_row.values], np.float32); bf = np.asarray(Eb_f[fr.xj_row.values], np.float32)
        mv = [np.load(X1 / f"xj1_voice_{n}.float32.npy") for n in VN]; mf = [np.load(X1 / f"xj1_face_{n}.float32.npy") for n in FN]
        Vx = blk(X.prep_voice(bv), U(np.load(XA / "xja_voice_campplus.float32.npy")), U(np.load(XB / "xjb_voice_rdn.float32.npy")), *[U(x) for x in mv],
                 U(IV[:len(vr)]))                                                                                                             # IB
        okv = ok_rows(bv) & np.load(XA / "xja_voice_campplus.ok.npy") & np.load(XB / "xjb_voice_rdn.ok.npy") & np.logical_and.reduce([np.load(X1 / f"xj1_voice_{n}.ok.npy") for n in VN]) \
              & IVok[:len(vr)] & ok_rows(IV[:len(vr)])                                                                                        # IB
        Fx = blk(X.prep_face(bf), U(np.load(XA / "xja_face_ada.float32.npy")), U(np.load(XB / "xjb_face_arc.float32.npy")), *[U(x) for x in mf],
                 U(IF[:len(fr)]))                                                                                                             # IB
        okf = ok_rows(bf) & np.load(XA / "xja_face_ada.ok.npy") & np.load(XB / "xjb_face_arc.ok.npy") & np.logical_and.reduce([np.load(X1 / f"xj1_face_{n}.ok.npy") for n in FN]) \
              & IFok[:len(fr)] & ok_rows(IF[:len(fr)])                                                                                        # IB
        assert len(Vx) == len(vr) and len(Fx) == len(fr) and not vr.lang.astype(str).str.lower().isin(["bengali", "bangla", "bn"]).any()
        lk = vr.src + ":" + vr.rel.str.split("/voices/").str[1]; m = okv & ok_lid(lk)
        VR.append(pd.DataFrame(dict(k="mavceleb_" + vr.src[m] + "/" + vr.id[m], video=vr.video[m]))); VA.append(Vx[m])
        FR.append(pd.DataFrame(dict(k="mavceleb_" + fr.src[okf] + "/" + fr.id[okf], video=fr.video[okf]))); FA.append(Fx[okf])
        print(f"xj1ib MAV: voices kept {m.sum()} / {len(vr)} (LID/ok), faces {okf.sum()} / {len(fr)}", flush=True)
        # ---- VoxCeleb1 rows_vox
        r = pd.read_csv(CX / "rows_vox.csv")
        vv = vmeta[vmeta.source.eq("voxceleb1")]; pos = dict(zip(vv.path.str.split("/").str[-3:].str.join("/"), vv.index))
        has = r.voice.isin(pos).to_numpy(); bv = np.zeros((len(r), 192), np.float32); bv[has] = np.asarray(Eb_v[[pos[v] for v in r.voice[has]]], np.float32)
        nv = [np.load(CX / f"vox_voice_{n}.float32.npy") for n in VN]; nf = [np.load(CX / f"vox_face_{n}.float32.npy") for n in FN]
        assert all(len(x) == len(r) for x in nv + nf), "rows_vox order"
        iv = IVr.iloc[len(vr):]; assert (iv.src == "vox1").all() and (iv.rel.values == r.voice.values).all(), "xj1ib vox voice rows"          # IB
        ibv, ibv_ok = IV[len(vr):], IVok[len(vr):]
        fi = IFr.iloc[len(fr):]; assert (fi.src == "vox1").all(); fp = {f: len(fr) + j for j, f in enumerate(fi.rel)}; ix = np.array([fp[f] for f in r.face])  # IB
        ibf, ibf_ok = IF[ix], IFok[ix]
        Vx = blk(X.prep_voice(bv + (~has)[:, None]), U(np.load(CX / "vox_voice_campplus.float32.npy")), U(np.load(CX / "vox_voice_rdn.float32.npy")), *[U(x) for x in nv],
                 U(ibv))                                                                                                                      # IB
        okv = has & ok_rows(bv) & np.load(CX / "vox_voice_campplus.ok.npy") & np.load(CX / "vox_voice_rdn.ok.npy") & np.logical_and.reduce([np.load(CX / f"vox_voice_{n}.ok.npy") for n in VN]) \
              & ibv_ok & ok_rows(ibv)                                                                                                         # IB
        ind = r.identity.astype(str).isin(india).to_numpy(); m = okv & (~ind | ok_lid("vox1:" + r.voice))
        VR.append(pd.DataFrame(dict(k="voxceleb1/" + r.identity[m], video=r.video[m]))); VA.append(Vx[m])
        fo = pd.read_csv(XB / "vox_faces_order.csv"); assert (fo.face.values == r.face.values).all(), "vox face order"
        bf = np.load(XV / "vox_face_btc.float32.npy"); assert len(bf) == len(r)
        Fx = blk(X.prep_face(bf + (~ok_rows(bf))[:, None]), U(np.load(CX / "vox_face_ada.float32.npy")), U(np.load(XB / "vox_face_arc.float32.npy")), *[U(x) for x in nf],
                 U(ibf))                                                                                                                      # IB
        okf = ok_rows(bf) & np.load(XV / "vox_face_btc.ok.npy") & np.load(CX / "vox_face_ada.ok.npy") & np.load(XB / "vox_face_arc.ok.npy") \
              & np.logical_and.reduce([np.load(CX / f"vox_face_{n}.ok.npy") for n in FN]) & ibf_ok & ok_rows(ibf)                             # IB
        d = ~r.face.duplicated().to_numpy() & okf
        FR.append(pd.DataFrame(dict(k="voxceleb1/" + r.identity[d], video=r.face[d].str.split("/").str[1]))); FA.append(Fx[d])
        print(f"vox1: voices kept {m.sum()} / {len(r)} (BTC match {has.sum()}; India-id clips need LID: {int((ind & okv).sum())}, kept {int((ind & m).sum())}), faces {d.sum()}", flush=True)
        vm = pd.concat(VR, ignore_index=True); V = np.concatenate(VA); fm = pd.concat(FR, ignore_index=True); F = np.concatenate(FA)
        assert np.isfinite(V).all() and np.isfinite(F).all() and V.shape[1] == DV and F.shape[1] == DF
        keep = fm.k.isin(set(vm.k)).to_numpy(); self.fm, self.F = fm[keep].reset_index(drop=True), F[keep]
        kv = vm.k.isin(set(self.fm.k)).to_numpy(); self.vm, self.V = vm[kv].reset_index(drop=True), V[kv]; assert len(self.V) == len(self.vm)
        self.byk = {k: g.index.values for k, g in self.fm.groupby("k")}; self.bykv = {kv_: g.index.values for kv_, g in self.fm.groupby(["k", "video"])}
        print(f"ext 1+IB: voices {len(self.V)} faces {len(self.F)} ids {self.vm.k.nunique()} ({self.vm.k.str.split('/').str[0].value_counts().to_dict()})", flush=True)


def eval_feats():
    ev = T3 / "outputs/eval_final"; t, F, V = X1M.eval_feats()
    assert t.equals(pd.read_csv(ev / "rows.csv")), "eval rows order"
    ibf, ibv = np.load(ev / "ultra/ib_face.npy"), np.load(ev / "ultra/ib_voice.npy"); assert len(ibf) == len(ibv) == len(t) and ok_rows(ibf).all() and ok_rows(ibv).all()
    F2, V2 = add_ib(F, V, ibf, ibv); assert F2.shape[1] == DF and V2.shape[1] == DV and np.isfinite(F2).all() and np.isfinite(V2).all()
    return t, F2, V2


CFG = lambda nseed: dict(model="XJ-1+IB", face="btc+ada+arc+clipnf+clipnf2+farl+imagebind", voice="btc+campplus+rdn+simam+sam100ft+r293+ecapa+imagebind",
                         blocks="each L2, concat / sqrt(n blocks)", ext="XJ-U pool (MAV xjc set + VoxCeleb1 rows_vox), items with all encoders incl. IB",
                         hidden=256, outd=128, epochs=3, ratio=4, drop=0.7, seeds=[1000 * i for i in range(nseed)], v2=True, bnclip=True, nf=False)


def final(out, nseed=10):
    out = Path(out); cfg = CFG(nseed); out.mkdir(parents=True, exist_ok=True); cp = out / "config.json"
    if cp.exists(): assert json.load(open(cp)) == cfg, "config mismatch"
    json.dump(cfg, open(cp, "w"), indent=1)
    L, Ff, Vf = flag(); E = ExtIB(); ms = []
    for i, sd in enumerate(cfg["seeds"]):
        p = out / f"xj1ib_m{i}.pt"
        if p.exists():
            m = X.XJ(DF, DV); m.load_state_dict(torch.load(p, map_location="cpu")); m.eval(); ms.append(m); continue
        m = X.train(Ff, Vf, seed=sd, epochs=3, ext=E.sampler(()), ext_ratio=4); torch.save(m.state_dict(), p); ms.append(m)
        print(f"model {i}: train-pair mean cos {X.score(m, Ff, Vf).mean():.3f}", flush=True)
    t, F, V = eval_feats(); info = {}
    for tr in ("no_gender", "gender"):
        for lang, tag in (("English", "heard"), ("Bangla", "unheard")):
            msk = (t.protocol.eq(tr) & t.language.eq(lang)).to_numpy(); c = np.mean([X.score(m, F[msk], V[msk]) for m in ms], 0); assert np.isfinite(c).all()
            ref = pd.read_csv(T3 / f"submissions/t3_sub90c_eval/{tr}/sub_score_v4_{lang}_{tag}.txt", sep=" ", header=None, names=["id", "s"], dtype={"id": str})
            ids = t.pair_id.astype(str)[msk].values; assert (ids == ref.id.values).all(), "eval row order"
            np.save(out / f"cos_{tr}_{lang}.npy", c); p = out / tr / f"sub_score_v4_{lang}_{tag}.txt"; p.parent.mkdir(exist_ok=True)
            p.write_text("".join(f"{a} {-b:.6f}\n" for a, b in zip(ids, c))); info[f"{tr}/{lang}"] = int(len(c))
    json.dump(info, open(out / "rows.json", "w")); print(info)



def score(ckdir, out, nseed=10):
    """18/11 (release): eval scores from the saved towers <ckdir>/xj1ib_m{i}.pt only (no FLAG / external features needed);
    writes the same cos_*.npy / submission files as final() -> must be identical to the final run's outputs."""
    ck, out = Path(ckdir), Path(out); out.mkdir(parents=True, exist_ok=True); ms = []
    for i in range(nseed):
        m = X.XJ(DF, DV); m.load_state_dict(torch.load(ck / f"xj1ib_m{i}.pt", map_location="cpu")); m.eval(); ms.append(m)
    t, F, V = eval_feats(); info = {}
    for tr in ("no_gender", "gender"):
        for lang, tag in (("English", "heard"), ("Bangla", "unheard")):
            msk = (t.protocol.eq(tr) & t.language.eq(lang)).to_numpy(); c = np.mean([X.score(m, F[msk], V[msk]) for m in ms], 0); assert np.isfinite(c).all()
            ids = t.pair_id.astype(str)[msk].values
            np.save(out / f"cos_{tr}_{lang}.npy", c); p = out / tr / f"sub_score_v4_{lang}_{tag}.txt"; p.parent.mkdir(exist_ok=True)
            p.write_text("".join(f"{a} {-b:.6f}\n" for a, b in zip(ids, c))); info[f"{tr}/{lang}"] = int(len(c))
    import shutil
    if (ck / "config.json").exists(): shutil.copy2(ck / "config.json", out / "config.json")   # fuse_ab checks the model name
    print("scored from checkpoint", ck, info)

def oof(tag, nseed=10):
    sys.path.insert(0, str(T3 / "scripts")); cwd = os.getcwd(); os.chdir(T3 / "scripts")
    import t3_ultra_eval as E
    T = E.load_trials(); os.chdir(cwd)
    J = json.load(open(T3 / "release/sub90c/local_eval_kit/folds.json")); L, Ff, Vf = flag(); Ex = ExtIB(); out = {}
    for k in range(5):
        held = set(J["reserve"]) | set(J["folds"][k]); tr = ~L.id.isin(held).to_numpy()
        ms = [X.train(Ff[tr], Vf[tr], seed=1000 * j + k, epochs=3, ext=Ex.sampler(Ex.drop_for(held)), ext_ratio=4) for j in range(nseed)]
        for pr in E.PR:
            t = T[f"f{k}_{pr}"]; assert not set(L.id.values[t["vi"]]) & set(L.id.values[tr])
            out[f"f{k}_{pr}"] = np.mean([X.score(m, Ff[t["fi"]], Vf[t["vi"]]) for m in ms], 0)
        print("done", k, flush=True)
    oc = dict(model="XJ-1+IB", nseed=nseed, v2=True, nf=False, epochs=3, ratio=4)
    np.savez(XD / f"xj1ib_oof_{tag}.npz", **out, oof_cfg=json.dumps(oc)); print("saved", tag, oc)


if __name__ == "__main__":
    kv = dict(a.split("=") for a in sys.argv[3:] if "=" in a); n = int(kv.get("nseed", 10))
    if sys.argv[1] == "score": score(sys.argv[2], sys.argv[3], n)
    else: (final if sys.argv[1] == "final" else oof)(sys.argv[2], n)
