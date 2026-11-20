"""XJ-U2 (14/11 20:55 prereg sub100a; user "ngoài ra tiếp tục cải thiện cái kia" = hướng 1): XJ-U (xju.py: same encoders, same blocks, same XJ recipe)
with the XJ-SA external pool on all six encoders:
  MAV xjc set (exactly as XJ-U)
  + the teammate's FULL VoxCeleb1 set (ext_feats voxceleb1 rows: 50,040 voice clips; 43,523 real face images — the 6,517 voice-spectrogram PNGs that
    ext_feats lists as "faces" are excluded)
  + MAVX (2 shards; a voice clip only if LID v3 cleared it — fail closed: a NaN probability counts as flagged; faces with ok rows)
New encoders for VoxCeleb1 / MAVX: Kaggle ncamtu06/flag-t3-xju2-faces (AdaFace + ArcFace) and trnlhvy/flag-t3-xju2-voices (CAM++ + ReDimNet), unpacked
in xj/xju2/ (kaggle/build_xju2_extract.py). BTC VGGFace / ECAPA = the teammate's ext_feats rows (VoxCeleb1) and the xjx MAVX run. Bengali rule = clip
level (xj_ext.clip_lid) as XJ-U: MAV clips need LID; VoxCeleb1 India-nationality ids keep only LID-checked unflagged clips.
usage: python xju2.py final <outdir> [nseed=10]   |   python xju2.py oof <tag> [nseed=10]"""
import sys, os, json
from pathlib import Path
import numpy as np, pandas as pd, torch
XD = Path(__file__).resolve().parent; T3 = XD.parent; sys.path.insert(0, str(XD))
import xj_lib as X, xj_ext as XE, xjc as XC, xju as XU
CX = T3 / "caches/ext"; XA, XB, X2, EXF, MXX = XD / "xja", XD / "xjb", XD / "xju2", XD / "ext_feats", XD / "mavx"
U, blk, ok_rows, DF, DV = XC.U, XU.blk, XU.ok_rows, XU.DF, XU.DV
torch.set_num_threads(int(os.environ.get("XJA_THREADS", 4)))
BN = ["bengali", "bangla", "bn"]; LIDC = ["vl_bnas", "wh_bnas", "mms_bnas"]


def new_feats():
    """Kaggle outputs, row-aligned with the local row lists (which carry xj_row = row in ext_feats / position in the MAVX shard csv)."""
    fl, vl = pd.read_csv(T3 / "kaggle/xju2_faces/xju2_face_rows_local.csv"), pd.read_csv(T3 / "kaggle/xju2_voices/xju2_voice_rows_local.csv")
    assert pd.read_csv(X2 / "xju2_face_rows.csv").equals(fl[["src", "rel"]]) and pd.read_csv(X2 / "xju2_voice_rows.csv").equals(vl[["src", "rel"]]), "Kaggle rows = local rows"
    ada, sa = np.load(X2 / "xju2_face_ada.float32.npy"), np.load(X2 / "xju2_face_ada.align.npy")
    arc, sr = np.load(X2 / "xju2_face_arc.float32.npy"), np.load(X2 / "xju2_face_arc.align.npy")
    cp, okc = np.load(X2 / "xju2_voice_campplus.float32.npy"), np.load(X2 / "xju2_voice_campplus.ok.npy")
    rd, okr = np.load(X2 / "xju2_voice_rdn.float32.npy"), np.load(X2 / "xju2_voice_rdn.ok.npy")
    assert len(ada) == len(arc) == len(sa) == len(sr) == len(fl) and len(cp) == len(rd) == len(okc) == len(okr) == len(vl)
    fl["ok"] = (sa != 255) & (sr != 255) & ok_rows(ada) & ok_rows(arc); vl["ok"] = okc & okr & ok_rows(cp) & ok_rows(rd)
    return fl, U(ada), U(arc), vl, U(cp), U(rd)


class ExtU2(XC.ExtC):
    def __init__(self):
        chk, fl = XE.clip_lid(); ok_lid = lambda keys: np.array([k in chk and k not in fl for k in keys])
        ex = pd.read_csv(CX / "exclude_bengali_v3.csv"); india = set(ex[ex.src.eq("vox")].identity.astype(str))
        Eb_f, Eb_v = np.load(EXF / "faces.npy", mmap_mode="r"), np.load(EXF / "voices.npy", mmap_mode="r")
        fmeta, vmeta = pd.read_csv(EXF / "faces_meta.csv"), pd.read_csv(EXF / "voices_meta.csv"); assert len(fmeta) == len(Eb_f) and len(vmeta) == len(Eb_v)
        VR, FR, VA, FA = [], [], [], []
        # ---- MAV xjc set: verbatim XJ-U (xju.ExtU)
        R = pd.read_csv(T3 / "kaggle/xjc_rows/xjc_rows.csv")
        assert R.equals(pd.read_csv(XA / "xja_rows.csv")) and R.equals(pd.read_csv(XB / "xjb_rows.csv")), "xja / xjb rows = xjc_rows"
        vr, fr = R[R.kind.eq("voice")].reset_index(drop=True), R[R.kind.eq("face")].reset_index(drop=True)
        assert all(vmeta.path.values[i].endswith(p) for i, p in zip(vr.xj_row, vr.rel)) and all(fmeta.path.values[i].endswith(p) for i, p in zip(fr.xj_row, fr.rel)), "xj_row -> ext_feats"
        bv = np.asarray(Eb_v[vr.xj_row.values], np.float32); bf = np.asarray(Eb_f[fr.xj_row.values], np.float32)
        Vx = blk(X.prep_voice(bv), U(np.load(XA / "xja_voice_campplus.float32.npy")), U(np.load(XB / "xjb_voice_rdn.float32.npy")))
        okv = ok_rows(bv) & np.load(XA / "xja_voice_campplus.ok.npy") & np.load(XB / "xjb_voice_rdn.ok.npy")
        Fx = blk(X.prep_face(bf), U(np.load(XA / "xja_face_ada.float32.npy")), U(np.load(XB / "xjb_face_arc.float32.npy")))
        okf = ok_rows(bf) & np.load(XA / "xja_face_ada.ok.npy") & np.load(XB / "xjb_face_arc.ok.npy")
        assert len(Vx) == len(vr) and len(Fx) == len(fr) and not vr.lang.astype(str).str.lower().isin(BN).any()
        lk = vr.src + ":" + vr.rel.str.split("/voices/").str[1]; m = okv & ok_lid(lk)
        VR.append(pd.DataFrame(dict(k="mavceleb_" + vr.src[m] + "/" + vr.id[m], video=vr.video[m]))); VA.append(Vx[m])
        FR.append(pd.DataFrame(dict(k="mavceleb_" + fr.src[okf] + "/" + fr.id[okf], video=fr.video[okf]))); FA.append(Fx[okf])
        print(f"xju2 MAV: voices kept {m.sum()} / {len(vr)} (LID/ok), faces {okf.sum()} / {len(fr)}", flush=True)
        # ---- the teammate's full VoxCeleb1 set (replaces rows_vox)
        frl, Ada, Arc, vrl, Cp, Rd = new_feats()
        v = vrl[vrl.src.eq("vox1")]; vm1 = vmeta.loc[v.xj_row.values]
        assert vm1.source.eq("voxceleb1").all() and all(p.endswith("/" + r) for p, r in zip(vm1.path, v.rel)), "vox1 voice rows -> ext_feats"
        bv = np.asarray(Eb_v[v.xj_row.values], np.float32); okb = ok_rows(bv); i = v.index.values
        Vx = blk(X.prep_voice(bv), Cp[i], Rd[i]); okv = okb & v.ok.to_numpy()
        key = v.rel.str.split("/").str[-3:].str.join("/").values; ids = vm1.id.astype(str).values
        ind = np.isin(ids, list(india)); m = okv & (~ind | ok_lid("vox1:" + pd.Series(key)))
        VR.append(pd.DataFrame(dict(k="voxceleb1/" + ids[m], video=vm1.video.values[m]))); VA.append(Vx[m])
        f = frl[frl.src.eq("vox1")]; fm1 = fmeta.loc[f.xj_row.values]
        assert fm1.source.eq("voxceleb1").all() and all(p.endswith("VoxCeleb_1_adapted/" + r) for p, r in zip(fm1.path, f.rel)) and f.rel.str.startswith("face/").all(), "vox1 face rows"
        bf = np.asarray(Eb_f[f.xj_row.values], np.float32); j = f.index.values
        Fx = blk(X.prep_face(bf), Ada[j], Arc[j]); okf = ok_rows(bf) & f.ok.to_numpy()
        FR.append(pd.DataFrame(dict(k="voxceleb1/" + fm1.id.astype(str).values[okf], video=fm1.video.values[okf]))); FA.append(Fx[okf])
        print(f"vox1 full: voices kept {m.sum()} / {len(v)} (encoders ok {okv.sum()}; India-id clips {int((ind & okv).sum())}, kept {int((ind & m).sum())}), "
              f"faces {okf.sum()} / {len(f)}", flush=True)
        # ---- MAVX (xj_ext.add_mavx rows; LID fail-closed)
        for sh in (0, 1):
            vv = pd.read_csv(MXX / f"mavx{sh}_voice_rows.csv"); Vb = np.load(MXX / f"mavx{sh}_voice.f32.npy"); assert len(vv) == len(Vb)
            d = pd.read_csv(MXX / f"mavx{sh}_lid.csv"); okl = (d[LIDC] <= .5).all(axis=1) & (d.dur > 0)
            lk = vv.src.str.replace("mavceleb_", "") + ":" + vv.rel.str.split("/voices/").str[1]; assert set(lk) == set(d.key), f"shard {sh}: LID keys != clips"
            nv = vrl[vrl.src.eq(f"mavx{sh}")]; assert (nv.rel.values == vv.rel.values).all() and (nv.xj_row.values == np.arange(len(vv))).all()
            Vx = blk(X.prep_voice(Vb), Cp[nv.index.values], Rd[nv.index.values])
            m = vv.ok.to_numpy() & lk.isin(set(d.key[okl])).to_numpy() & nv.ok.to_numpy() & ok_rows(Vb)
            assert not vv.lang[m].astype(str).str.lower().isin(BN).any()
            VR.append(pd.DataFrame(dict(k=vv.k.values[m], video=vv.video.values[m]))); VA.append(Vx[m])
            ff = pd.read_csv(MXX / f"mavx{sh}_face_rows.csv"); Fb = np.load(MXX / f"mavx{sh}_face.f16.npy").astype(np.float32); assert len(ff) == len(Fb)
            nf = frl[frl.src.eq(f"mavx{sh}")]; assert (nf.rel.values == ff.rel.values).all() and (nf.xj_row.values == np.arange(len(ff))).all()
            Fx = blk(X.prep_face(Fb), Ada[nf.index.values], Arc[nf.index.values]); okf = ff.ok.to_numpy() & nf.ok.to_numpy() & ok_rows(Fb)
            FR.append(pd.DataFrame(dict(k=ff.k.values[okf], video=ff.video.values[okf]))); FA.append(Fx[okf])
            print(f"mavx{sh}: voices kept {m.sum()} / {len(vv)} (LID cleared {int(okl.sum())}), faces {okf.sum()} / {len(ff)}", flush=True)
        vm = pd.concat(VR, ignore_index=True); V = np.concatenate(VA); fm = pd.concat(FR, ignore_index=True); F = np.concatenate(FA)
        assert np.isfinite(V).all() and np.isfinite(F).all() and V.shape[1] == DV and F.shape[1] == DF
        keep = fm.k.isin(set(vm.k)).to_numpy(); self.fm, self.F = fm[keep].reset_index(drop=True), F[keep]
        kv = vm.k.isin(set(self.fm.k)).to_numpy(); self.vm, self.V = vm[kv].reset_index(drop=True), V[kv]; assert len(self.V) == len(self.vm)
        self.byk = {k: g.index.values for k, g in self.fm.groupby("k")}; self.bykv = {kv_: g.index.values for kv_, g in self.fm.groupby(["k", "video"])}
        print(f"ext U2: voices {len(self.V)} faces {len(self.F)} ids {self.vm.k.nunique()} ({self.vm.k.str.split('/').str[0].value_counts().to_dict()})", flush=True)


CFG = lambda nseed: dict(model="XJ-U2", face="btc+ada+arc", voice="btc+campplus+rdn", blocks="each L2, concat / sqrt(3)",
                         ext="MAV xjc + teammate VoxCeleb1 full (real faces) + MAVX", hidden=256, outd=128, epochs=3, ratio=4, drop=0.7,
                         seeds=[1000 * i for i in range(nseed)], v2=True, bnclip=True, nf=False)


def final(out, nseed=10):
    out = Path(out); cfg = CFG(nseed); out.mkdir(parents=True, exist_ok=True); cp = out / "config.json"
    if cp.exists(): assert json.load(open(cp)) == cfg, "config mismatch"
    json.dump(cfg, open(cp, "w"), indent=1)
    L, Ff, Vf = XU.flag(); E = ExtU2(); ms = []
    for i, sd in enumerate(cfg["seeds"]):
        p = out / f"xju2_m{i}.pt"
        if p.exists():
            m = X.XJ(DF, DV); m.load_state_dict(torch.load(p, map_location="cpu")); m.eval(); ms.append(m); continue
        m = X.train(Ff, Vf, seed=sd, epochs=3, ext=E.sampler(()), ext_ratio=4); torch.save(m.state_dict(), p); ms.append(m)
        print(f"model {i}: train-pair mean cos {X.score(m, Ff, Vf).mean():.3f}", flush=True)
    t, F, V = XU.eval_feats(); info = {}
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
    J = json.load(open(T3 / "release/sub90c/local_eval_kit/folds.json")); L, Ff, Vf = XU.flag(); Ex = ExtU2(); out = {}
    for k in range(5):
        held = set(J["reserve"]) | set(J["folds"][k]); tr = ~L.id.isin(held).to_numpy()
        ms = [X.train(Ff[tr], Vf[tr], seed=1000 * j + k, epochs=3, ext=Ex.sampler(Ex.drop_for(held)), ext_ratio=4) for j in range(nseed)]
        for pr in E.PR:
            t = T[f"f{k}_{pr}"]; assert not set(L.id.values[t["vi"]]) & set(L.id.values[tr])
            out[f"f{k}_{pr}"] = np.mean([X.score(m, Ff[t["fi"]], Vf[t["vi"]]) for m in ms], 0)
        print("done", k, flush=True)
    oc = dict(model="XJ-U2", nseed=nseed, v2=True, nf=False, epochs=3, ratio=4)
    np.savez(XD / f"xju2_oof_{tag}.npz", **out, oof_cfg=json.dumps(oc)); print("saved", tag, oc)


if __name__ == "__main__":
    kv = dict(a.split("=") for a in sys.argv[3:]); n = int(kv.get("nseed", 10))
    (final if sys.argv[1] == "final" else oof)(sys.argv[2], n)
