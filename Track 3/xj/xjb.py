"""XJ-B (14/11, user "Tôi đang muốn phải xuống tầm 16.8" -> "Oke làm thử đi xem sao"; prereg sub99a): the teammate's XJ recipe (xj_lib.train defaults:
towers 256->128, dropout .7, symmetric InfoNCE, 3 epochs, ext ratio 4, FLAG-train centring, mean of 10 seeds) on a THIRD encoder pair used by no
other XJ system: faces = ArcFace (InsightFace buffalo_l w600k_r50, SCRFD det 320 + 5-point norm_crop 112; 512, L2), voices = ReDimNet b6 ft_lm vox2
(192, L2). External = the XJ-A set: the teammate's MAV-Celeb v1+v2 clip set (xjb/ = Kaggle caohuhoan/flag-t3-xjb-extract on xjc_rows) + VoxCeleb1
rows_vox (ReDimNet cache + ArcFace from the same notebook). Bengali rule = clip level (xj_ext.clip_lid), exactly as xja.ExtA.
usage: python xjb.py final <outdir> [nseed=10]   |   python xjb.py oof <tag> [nseed=10]"""
import sys, os, json
from pathlib import Path
import numpy as np, pandas as pd, torch
XD = Path(__file__).resolve().parent; T3 = XD.parent; sys.path.insert(0, str(XD))
import xj_lib as X, xj_ext as XE, xjc as XC
CX = T3 / "caches/ext"; XB = XD / "xjb"   # XB = unpacked xjb_embeddings.zip
U = XC.U
torch.set_num_threads(int(os.environ.get("XJA_THREADS", 4)))
DF, DV = 512, 192


def flag():
    sys.path.insert(0, str(T3 / "scripts")); cwd = os.getcwd(); os.chdir(T3 / "scripts")
    import t3data as D
    V = np.asarray(D.voice_full("rdn"))[:6485]; os.chdir(cwd)
    F = np.load(XB / "flag_face_arc.float32.npy")
    L, _, _ = X.load_flag(); assert V.shape == (6485, DV) and F.shape == (6485, DF) and len(L) == 6485 and np.isfinite(F).all()
    return L, U(F), U(V)


class ExtB(XC.ExtC):
    def __init__(self):
        chk, fl = XE.clip_lid(); ok_lid = lambda keys: np.array([k in chk and k not in fl for k in keys])
        ex = pd.read_csv(T3 / "caches/ext/exclude_bengali_v3.csv"); india = set(ex[ex.src.eq("vox")].identity.astype(str))
        VR, FR, VA, FA = [], [], [], []
        R = pd.read_csv(XB / "xjb_rows.csv"); assert R.equals(pd.read_csv(T3 / "kaggle/xjc_rows/xjc_rows.csv")), "xjb rows = xjc_rows"
        vr, fr = R[R.kind.eq("voice")].reset_index(drop=True), R[R.kind.eq("face")].reset_index(drop=True)
        Vx = U(np.load(XB / "xjb_voice_rdn.float32.npy")); okv = np.load(XB / "xjb_voice_rdn.ok.npy")
        Fx = U(np.load(XB / "xjb_face_arc.float32.npy")); okf = np.load(XB / "xjb_face_arc.ok.npy")
        assert len(Vx) == len(vr) == len(okv) and len(Fx) == len(fr) == len(okf) and not vr.lang.astype(str).str.lower().isin(["bengali", "bangla", "bn"]).any()
        lk = vr.src + ":" + vr.rel.str.split("/voices/").str[1]; m = okv & ok_lid(lk)
        VR.append(pd.DataFrame(dict(k="mavceleb_" + vr.src[m] + "/" + vr.id[m], video=vr.video[m]))); VA.append(Vx[m])
        FR.append(pd.DataFrame(dict(k="mavceleb_" + fr.src[okf] + "/" + fr.id[okf], video=fr.video[okf]))); FA.append(Fx[okf])
        print(f"xjb MAV: voices kept {m.sum()} / {len(vr)} (LID/ok), faces {okf.sum()} / {len(fr)}", flush=True)
        r = pd.read_csv(CX / "rows_vox.csv"); Vx = U(np.load(CX / "vox_voice_rdn.float32.npy")); okv = np.load(CX / "vox_voice_rdn.ok.npy")
        assert len(Vx) == len(r) == len(okv)
        ind = r.identity.astype(str).isin(india).to_numpy(); m = okv & (~ind | ok_lid("vox1:" + r.voice))
        VR.append(pd.DataFrame(dict(k="voxceleb1/" + r.identity[m], video=r.video[m]))); VA.append(Vx[m])
        fo = pd.read_csv(XB / "vox_faces_order.csv"); assert (fo.face.values == r.face.values).all(), "vox face order"
        Fx = U(np.load(XB / "vox_face_arc.float32.npy")); okf = np.load(XB / "vox_face_arc.ok.npy"); assert len(Fx) == len(r) == len(okf)
        d = ~r.face.duplicated().to_numpy() & okf
        FR.append(pd.DataFrame(dict(k="voxceleb1/" + r.identity[d], video=r.face[d].str.split("/").str[1]))); FA.append(Fx[d])
        print(f"vox1: voices kept {m.sum()} / {len(r)} (India-id clips need LID: {int((ind & okv).sum())}, kept {int((ind & m).sum())}), faces {d.sum()}", flush=True)
        vm = pd.concat(VR, ignore_index=True); V = np.concatenate(VA); fm = pd.concat(FR, ignore_index=True); F = np.concatenate(FA)
        keep = fm.k.isin(set(vm.k)).to_numpy(); self.fm, self.F = fm[keep].reset_index(drop=True), F[keep]
        kv = vm.k.isin(set(self.fm.k)).to_numpy(); self.vm, self.V = vm[kv].reset_index(drop=True), V[kv]; assert len(self.V) == len(self.vm)
        self.byk = {k: g.index.values for k, g in self.fm.groupby("k")}; self.bykv = {kv_: g.index.values for kv_, g in self.fm.groupby(["k", "video"])}
        print(f"ext B: voices {len(self.V)} faces {len(self.F)} ids {self.vm.k.nunique()} ({self.vm.k.str.split('/').str[0].value_counts().to_dict()})", flush=True)


def eval_feats():
    ev = T3 / "outputs/eval_final"; t = pd.read_csv(ev / "rows.csv")
    F = U(np.load(XB / "eval_face_arc.float32.npy")); V = U(np.load(ev / "feat/voice_rdn.npy")); assert len(F) == len(V) == len(t) and np.isfinite(F).all()
    return t, F, V


def final(out, nseed=10):
    out = Path(out); cfg = dict(model="XJ-B", face="arc", voice="rdn", hidden=256, outd=128, epochs=3, ratio=4, drop=0.7, seeds=[1000 * i for i in range(nseed)],
                                v2=True, bnclip=True, nf=False)
    out.mkdir(parents=True, exist_ok=True); cp = out / "config.json"
    if cp.exists(): assert json.load(open(cp)) == cfg, "config mismatch"
    json.dump(cfg, open(cp, "w"), indent=1)
    L, Ff, Vf = flag(); E = ExtB(); ms = []
    for i, sd in enumerate(cfg["seeds"]):
        p = out / f"xjb_m{i}.pt"
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


def oof(tag, nseed=10):
    sys.path.insert(0, str(T3 / "scripts")); cwd = os.getcwd(); os.chdir(T3 / "scripts")
    import t3_ultra_eval as E
    T = E.load_trials(); os.chdir(cwd)
    J = json.load(open(T3 / "release/sub90c/local_eval_kit/folds.json")); L, Ff, Vf = flag(); Ex = ExtB(); out = {}
    for k in range(5):
        held = set(J["reserve"]) | set(J["folds"][k]); tr = ~L.id.isin(held).to_numpy()
        ms = [X.train(Ff[tr], Vf[tr], seed=1000 * j + k, epochs=3, ext=Ex.sampler(Ex.drop_for(held)), ext_ratio=4) for j in range(nseed)]
        for pr in E.PR:
            t = T[f"f{k}_{pr}"]; assert not set(L.id.values[t["vi"]]) & set(L.id.values[tr])
            out[f"f{k}_{pr}"] = np.mean([X.score(m, Ff[t["fi"]], Vf[t["vi"]]) for m in ms], 0)
        print("done", k, flush=True)
    oc = dict(model="XJ-B", nseed=nseed, v2=True, nf=False, epochs=3, ratio=4)
    np.savez(XD / f"xjb_oof_{tag}.npz", **out, oof_cfg=json.dumps(oc)); print("saved", tag, oc)


if __name__ == "__main__":
    kv = dict(a.split("=") for a in sys.argv[3:]); n = int(kv.get("nseed", 10))
    (final if sys.argv[1] == "final" else oof)(sys.argv[2], n)
