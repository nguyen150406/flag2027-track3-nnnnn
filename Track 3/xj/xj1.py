"""XJ-1 (15/11, user: "1 hệ duy nhất nghĩa là bangla english chung một hệ duy nhất thôi đấy, gom các encoder lại như bạn đã làm cho xj-u nhưng phải
làm sao cho nó thật tốt và tối ưu"): ONE XJ model scores all four cells (English and Bangla), early fusion of XJ-U's six encoders and sub90c's
frozen encoders (same recipe as xju.py, only the input blocks grow):
  face  input = [sqrt+L2 BTC VGGFace | AdaFace | ArcFace | CLIP-L/14 + LoRA clipnf | CLIP-L/14 + LoRA clipnf2 | FaRL] / sqrt(6)
  voice input = [L2 BTC ECAPA | CAM++ | ReDimNet | SimAM-R100 (VoxBlink2) | SimAM ft (sam100ft) | ResNet293 | ECAPA-1024] / sqrt(7)
(each block L2 = equal a-priori share; sub90c's attribute models and ImageBind are not inputs: no features for the external pool).
XJ recipe unchanged = xj_lib.train defaults (towers 256->128, dropout .7, symmetric InfoNCE, 3 epochs, ext ratio 4, FLAG-train centring), 10 seeds.
External pool = XJ-U's (MAV xjc set + VoxCeleb1 rows_vox), items with ALL encoders of their side; sub90c features of the MAV set from Kaggle
ncamtu06/flag-t3-xj1-mav (kaggle/build_xj1_extract.py, parity-gated) unpacked in xj/xj1/. Bengali rule = clip level, as XJ-U.
usage: python xj1.py final <outdir> [nseed=10]   |   python xj1.py oof <tag> [nseed=10]
15/11 sub105a: env XJ1_GPURE=1 -> XJ-1G = same recipe with gender-pure batches (xj_lib.train gpure; FLAG meta gender, external
xj/ext_gender.json via xj_ext.gender_map(); unknown -> mixed batches). Default (unset) = XJ-1 unchanged."""
import sys, os, json
from pathlib import Path
import numpy as np, pandas as pd, torch
XD = Path(__file__).resolve().parent; T3 = XD.parent; sys.path.insert(0, str(XD))
import xj_lib as X, xj_ext as XE, xjc as XC, xju as XUm
CX = T3 / "caches/ext"; XA, XB, XV, EXF, X1 = XD / "xja", XD / "xjb", XD / "vox1_btc", XD / "ext_feats", XD / "xj1"
U, blk, ok_rows = XC.U, XUm.blk, XUm.ok_rows
torch.set_num_threads(int(os.environ.get("XJA_THREADS", 4)))
GPURE = os.environ.get("XJ1_GPURE", "0") == "1"
FN, VN = ("clipnf", "clipnf2", "farl"), ("simam", "sam100ft", "r293", "ecapa")
DF, DV = 4096 + 512 + 512 + 768 + 768 + 512, 192 + 512 + 192 + 256 * 3 + 192
EVF = {"clipnf": "ultra/face_clipnf.npy", "clipnf2": "ultra/face_clipnf2.npy", "farl": "feat/face_farl.npy",
       "simam": "feat/voice_sim.npy", "sam100ft": "feat/voice_sam100ft.npy", "r293": "feat/voice_r293.npy", "ecapa": "feat/voice_ecapa.npy"}
FLAGV = {"simam": "sim", "sam100ft": "sam100ft", "r293": "r293", "ecapa": "ecapa"}


def flag():
    sys.path.insert(0, str(T3 / "scripts")); cwd = os.getcwd(); os.chdir(T3 / "scripts")
    import t3data as D
    ada, cp, rdn = (np.asarray(a)[:6485] for a in (D.face("ada"), D.voice("campplus"), D.voice_full("rdn")))
    nf = [np.asarray(D.face(n), np.float32)[:6485] for n in FN]; nv = [np.asarray(D.voice_full(FLAGV[n]), np.float32)[:6485] for n in VN]; os.chdir(cwd)
    arc = np.load(XB / "flag_face_arc.float32.npy")
    L, Fb, Vb = X.load_flag()
    meta = pd.read_csv(T3.parent / "flag_raw_s01_v3_adaface_simam100/metadata.csv").sort_values("row_id"); t = meta[meta.partition.eq("train")]
    assert (L.voice.values == t.voice_relpath.values).all() and (L.face.values == t.face_relpath.values).all(), "BTC CSV order = META train order"
    F = blk(X.prep_face(Fb), U(ada), U(arc), *[U(x) for x in nf]); V = blk(X.prep_voice(Vb), U(cp), U(rdn), *[U(x) for x in nv])
    assert F.shape == (6485, DF) and V.shape == (6485, DV) and np.isfinite(F).all() and np.isfinite(V).all() and all(ok_rows(x).all() for x in nf + nv)
    return L, F, V


class ExtOne(XC.ExtC):
    """= xju.ExtU with the sub90c blocks appended (MAV: Kaggle xj1 arrays in xjc_rows order; VoxCeleb1: caches/ext vox_* in rows_vox order)."""
    def __init__(self):
        chk, fl = XE.clip_lid(); ok_lid = lambda keys: np.array([k in chk and k not in fl for k in keys])
        ex = pd.read_csv(T3 / "caches/ext/exclude_bengali_v3.csv"); india = set(ex[ex.src.eq("vox")].identity.astype(str))
        Eb_f, Eb_v = np.load(EXF / "faces.npy", mmap_mode="r"), np.load(EXF / "voices.npy", mmap_mode="r")
        fmeta, vmeta = pd.read_csv(EXF / "faces_meta.csv"), pd.read_csv(EXF / "voices_meta.csv"); assert len(fmeta) == len(Eb_f) and len(vmeta) == len(Eb_v)
        VR, FR, VA, FA = [], [], [], []
        # ---- MAV xjc set: same rows in xja / xjb / xj1
        R = pd.read_csv(T3 / "kaggle/xjc_rows/xjc_rows.csv")
        assert R.equals(pd.read_csv(XA / "xja_rows.csv")) and R.equals(pd.read_csv(XB / "xjb_rows.csv")), "xja / xjb rows = xjc_rows"
        vr, fr = R[R.kind.eq("voice")].reset_index(drop=True), R[R.kind.eq("face")].reset_index(drop=True)
        assert pd.read_csv(X1 / "xj1_face_rows.csv").equals(fr[["src", "rel"]]) and pd.read_csv(X1 / "xj1_voice_rows.csv").equals(vr[["src", "rel"]]), "xj1 rows = xjc_rows"
        assert all(vmeta.path.values[i].endswith(p) for i, p in zip(vr.xj_row, vr.rel)) and all(fmeta.path.values[i].endswith(p) for i, p in zip(fr.xj_row, fr.rel)), "xj_row -> ext_feats"
        bv = np.asarray(Eb_v[vr.xj_row.values], np.float32); bf = np.asarray(Eb_f[fr.xj_row.values], np.float32)
        mv = [np.load(X1 / f"xj1_voice_{n}.float32.npy") for n in VN]; mf = [np.load(X1 / f"xj1_face_{n}.float32.npy") for n in FN]
        Vx = blk(X.prep_voice(bv), U(np.load(XA / "xja_voice_campplus.float32.npy")), U(np.load(XB / "xjb_voice_rdn.float32.npy")), *[U(x) for x in mv])
        okv = ok_rows(bv) & np.load(XA / "xja_voice_campplus.ok.npy") & np.load(XB / "xjb_voice_rdn.ok.npy") & np.logical_and.reduce([np.load(X1 / f"xj1_voice_{n}.ok.npy") for n in VN])
        Fx = blk(X.prep_face(bf), U(np.load(XA / "xja_face_ada.float32.npy")), U(np.load(XB / "xjb_face_arc.float32.npy")), *[U(x) for x in mf])
        okf = ok_rows(bf) & np.load(XA / "xja_face_ada.ok.npy") & np.load(XB / "xjb_face_arc.ok.npy") & np.logical_and.reduce([np.load(X1 / f"xj1_face_{n}.ok.npy") for n in FN])
        assert len(Vx) == len(vr) and len(Fx) == len(fr) and not vr.lang.astype(str).str.lower().isin(["bengali", "bangla", "bn"]).any()
        lk = vr.src + ":" + vr.rel.str.split("/voices/").str[1]; m = okv & ok_lid(lk)
        VR.append(pd.DataFrame(dict(k="mavceleb_" + vr.src[m] + "/" + vr.id[m], video=vr.video[m]))); VA.append(Vx[m])
        FR.append(pd.DataFrame(dict(k="mavceleb_" + fr.src[okf] + "/" + fr.id[okf], video=fr.video[okf]))); FA.append(Fx[okf])
        print(f"xj1 MAV: voices kept {m.sum()} / {len(vr)} (LID/ok), faces {okf.sum()} / {len(fr)}", flush=True)
        # ---- VoxCeleb1 rows_vox
        r = pd.read_csv(CX / "rows_vox.csv")
        vv = vmeta[vmeta.source.eq("voxceleb1")]; pos = dict(zip(vv.path.str.split("/").str[-3:].str.join("/"), vv.index))
        has = r.voice.isin(pos).to_numpy(); bv = np.zeros((len(r), 192), np.float32); bv[has] = np.asarray(Eb_v[[pos[v] for v in r.voice[has]]], np.float32)
        nv = [np.load(CX / f"vox_voice_{n}.float32.npy") for n in VN]; nf = [np.load(CX / f"vox_face_{n}.float32.npy") for n in FN]
        assert all(len(x) == len(r) for x in nv + nf), "rows_vox order"
        Vx = blk(X.prep_voice(bv + (~has)[:, None]), U(np.load(CX / "vox_voice_campplus.float32.npy")), U(np.load(CX / "vox_voice_rdn.float32.npy")), *[U(x) for x in nv])
        okv = has & ok_rows(bv) & np.load(CX / "vox_voice_campplus.ok.npy") & np.load(CX / "vox_voice_rdn.ok.npy") & np.logical_and.reduce([np.load(CX / f"vox_voice_{n}.ok.npy") for n in VN])
        ind = r.identity.astype(str).isin(india).to_numpy(); m = okv & (~ind | ok_lid("vox1:" + r.voice))
        VR.append(pd.DataFrame(dict(k="voxceleb1/" + r.identity[m], video=r.video[m]))); VA.append(Vx[m])
        fo = pd.read_csv(XB / "vox_faces_order.csv"); assert (fo.face.values == r.face.values).all(), "vox face order"
        bf = np.load(XV / "vox_face_btc.float32.npy"); assert len(bf) == len(r)
        Fx = blk(X.prep_face(bf + (~ok_rows(bf))[:, None]), U(np.load(CX / "vox_face_ada.float32.npy")), U(np.load(XB / "vox_face_arc.float32.npy")), *[U(x) for x in nf])
        okf = ok_rows(bf) & np.load(XV / "vox_face_btc.ok.npy") & np.load(CX / "vox_face_ada.ok.npy") & np.load(XB / "vox_face_arc.ok.npy") \
              & np.logical_and.reduce([np.load(CX / f"vox_face_{n}.ok.npy") for n in FN])
        d = ~r.face.duplicated().to_numpy() & okf
        FR.append(pd.DataFrame(dict(k="voxceleb1/" + r.identity[d], video=r.face[d].str.split("/").str[1]))); FA.append(Fx[d])
        print(f"vox1: voices kept {m.sum()} / {len(r)} (BTC match {has.sum()}; India-id clips need LID: {int((ind & okv).sum())}, kept {int((ind & m).sum())}), faces {d.sum()}", flush=True)
        vm = pd.concat(VR, ignore_index=True); V = np.concatenate(VA); fm = pd.concat(FR, ignore_index=True); F = np.concatenate(FA)
        assert np.isfinite(V).all() and np.isfinite(F).all() and V.shape[1] == DV and F.shape[1] == DF
        keep = fm.k.isin(set(vm.k)).to_numpy(); self.fm, self.F = fm[keep].reset_index(drop=True), F[keep]
        kv = vm.k.isin(set(self.fm.k)).to_numpy(); self.vm, self.V = vm[kv].reset_index(drop=True), V[kv]; assert len(self.V) == len(self.vm)
        self.byk = {k: g.index.values for k, g in self.fm.groupby("k")}; self.bykv = {kv_: g.index.values for kv_, g in self.fm.groupby(["k", "video"])}
        print(f"ext 1: voices {len(self.V)} faces {len(self.F)} ids {self.vm.k.nunique()} ({self.vm.k.str.split('/').str[0].value_counts().to_dict()})", flush=True)


def eval_feats():
    ev = T3 / "outputs/eval_final"; t, Fu, Vu = XUm.eval_feats()          # XJ-U blocks (already / sqrt(3)) -> undo the scale, then append ours
    ef = [np.load(ev / EVF[n]) for n in FN]; evv = [np.load(ev / EVF[n]) for n in VN]
    assert all(len(x) == len(t) for x in ef + evv) and all(ok_rows(x).all() for x in ef + evv), "eval sub90c features per rows.csv row"
    F = np.hstack([Fu * np.sqrt(3)] + [U(x) for x in ef]) / np.sqrt(6); V = np.hstack([Vu * np.sqrt(3)] + [U(x) for x in evv]) / np.sqrt(7)
    F, V = F.astype(np.float32), V.astype(np.float32); assert F.shape[1] == DF and V.shape[1] == DV and np.isfinite(F).all() and np.isfinite(V).all()
    return t, F, V


CFG = lambda nseed: dict(model="XJ-1", face="btc+ada+arc+clipnf+clipnf2+farl", voice="btc+campplus+rdn+simam+sam100ft+r293+ecapa",
                         blocks="each L2, concat / sqrt(n blocks)", ext="XJ-U pool (MAV xjc set + VoxCeleb1 rows_vox)", hidden=256, outd=128, epochs=3,
                         ratio=4, drop=0.7, seeds=[1000 * i for i in range(nseed)], v2=True, bnclip=True, nf=False, **({"gpure": True} if GPURE else {}))


def gp(genders):
    """extra X.train kwargs for XJ-1G (gender-pure batches); {} for XJ-1"""
    if not GPURE: return {}
    G = XE.gender_map(); assert set(np.unique(genders)) <= {"m", "f"}
    return dict(gpure=True, genders=np.asarray(genders), ext_gender=G)


def final(out, nseed=10):
    out = Path(out); cfg = CFG(nseed); out.mkdir(parents=True, exist_ok=True); cp = out / "config.json"
    if cp.exists(): assert json.load(open(cp)) == cfg, "config mismatch"
    json.dump(cfg, open(cp, "w"), indent=1)
    L, Ff, Vf = flag(); E = ExtOne(); ms = []
    for i, sd in enumerate(cfg["seeds"]):
        p = out / f"xj1_m{i}.pt"
        if p.exists():
            m = X.XJ(DF, DV); m.load_state_dict(torch.load(p, map_location="cpu")); m.eval(); ms.append(m); continue
        m = X.train(Ff, Vf, seed=sd, epochs=3, ext=E.sampler(()), ext_ratio=4, **gp(L.g.values)); torch.save(m.state_dict(), p); ms.append(m)
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


def oof(tag, nseed=10):
    sys.path.insert(0, str(T3 / "scripts")); cwd = os.getcwd(); os.chdir(T3 / "scripts")
    import t3_ultra_eval as E
    T = E.load_trials(); os.chdir(cwd)
    J = json.load(open(T3 / "release/sub90c/local_eval_kit/folds.json")); L, Ff, Vf = flag(); Ex = ExtOne(); out = {}
    for k in range(5):
        held = set(J["reserve"]) | set(J["folds"][k]); tr = ~L.id.isin(held).to_numpy()
        ms = [X.train(Ff[tr], Vf[tr], seed=1000 * j + k, epochs=3, ext=Ex.sampler(Ex.drop_for(held)), ext_ratio=4, **gp(L.g.values[tr])) for j in range(nseed)]
        for pr in E.PR:
            t = T[f"f{k}_{pr}"]; assert not set(L.id.values[t["vi"]]) & set(L.id.values[tr])
            out[f"f{k}_{pr}"] = np.mean([X.score(m, Ff[t["fi"]], Vf[t["vi"]]) for m in ms], 0)
        print("done", k, flush=True)
    oc = dict(model="XJ-1", nseed=nseed, v2=True, nf=False, epochs=3, ratio=4, **({"gpure": True} if GPURE else {}))
    np.savez(XD / f"xj1_oof_{tag}.npz", **out, oof_cfg=json.dumps(oc)); print("saved", tag, oc)


if __name__ == "__main__":
    kv = dict(a.split("=") for a in sys.argv[3:]); n = int(kv.get("nseed", 10))
    (final if sys.argv[1] == "final" else oof)(sys.argv[2], n)
