"""XJ-C (14/11, user "Tiếp tục làm và phát triển cho eval tốt lên"): the teammate's XJ recipe (2 MLP towers 256->128, dropout .7, symmetric
InfoNCE, joint FLAG train + 4x external voice clips per epoch, 3 epochs, FLAG-train centring, mean of seeds; xj_lib.train) on OUR frozen encoders
instead of the BTC ones: faces = OpenAI CLIP ViT-L/14 (768), voices = cat4 (SimAM-R100 | SimAM-R100-VB2ft | ResNet293 | ECAPA1024; each L2-normalised,
stacked, 960). Goal: an XJ-type member with DIFFERENT features (new information for the fusion), same external data idea.
External: MAV-Celeb v1 from the rows_v1 caches (v2=0)  or  XJ's exact MAV-Celeb v1+v2 clips from the xjc Kaggle extraction (v2=1);
          VoxCeleb1 from the rows_vox caches (<= 10 clips per id).
Bengali rule = clip level (xj_ext.clip_lid, user 14/11): every MAV clip and every clip of the 26 India-nationality VoxCeleb1 ids must be LID-checked
and not flagged (unchecked = dropped); faces carry no speech. No Bengali-labelled data (asserted).
usage: python xjc.py final <outdir> [nseed=10] [v2=0|1]   |   python xjc.py oof <tag> [nseed=10] [v2=0|1]"""
import sys, os, json
from pathlib import Path
import numpy as np, pandas as pd, torch
XD = Path(__file__).resolve().parent; T3 = XD.parent; sys.path.insert(0, str(XD))
import xj_lib as X, xj_ext as XE
CX = T3 / "caches/ext"; XC = XD / "xjc"   # XC = unpacked xjc_embeddings.zip (Kaggle trncnguyn/flag-t3-xjc-extract)
VENC = ["simam", "sam100ft", "r293", "ecapa"]; EVV = ["sim", "sam100ft", "r293", "ecapa"]
U = lambda a: np.asarray(a, np.float32) / np.linalg.norm(np.asarray(a, np.float32), axis=1, keepdims=True).clip(1e-12)
cat = lambda arrs: U(np.hstack([U(a) for a in arrs]))


def flag():
    sys.path.insert(0, str(T3 / "scripts")); cwd = os.getcwd(); os.chdir(T3 / "scripts")
    import t3data as D
    F, V = np.asarray(D.face("clip"))[:6485], D.voice_full("cat4")[:6485]; os.chdir(cwd)
    L, _, _ = X.load_flag(); assert V.shape == (6485, 960) and F.shape == (6485, 768)
    return L, U(F), U(V)   # cat4 = per-encoder L2 stacked (t3data convention); extra L2 = same direction


class ExtC:
    def __init__(self, v2=False):
        chk, fl = XE.clip_lid(); ok_lid = lambda keys: np.array([k in chk and k not in fl for k in keys])
        ex = pd.read_csv(T3 / "caches/ext/exclude_bengali_v3.csv"); india = set(ex[ex.src.eq("vox")].identity.astype(str))
        VR, FR, VA, FA = [], [], [], []
        if v2:   # XJ's MAV v1+v2 clips, our encoders
            R = pd.read_csv(XC / "xjc_rows.csv"); vr, fr = R[R.kind.eq("voice")].reset_index(drop=True), R[R.kind.eq("face")].reset_index(drop=True)
            Vx = cat([np.load(XC / f"xjc_voice_{e}.float32.npy") for e in VENC]); Fx = U(np.load(XC / "xjc_face_clip.float32.npy"))
            okv = np.all([np.load(XC / f"xjc_voice_{e}.ok.npy") for e in VENC], 0); okf = np.load(XC / "xjc_face_clip.ok.npy")
            assert len(Vx) == len(vr) and len(Fx) == len(fr) and not vr.lang.astype(str).str.lower().isin(["bengali", "bangla", "bn"]).any()
            lk = vr.src + ":" + vr.rel.str.split("/voices/").str[1]; m = okv & ok_lid(lk)
            VR.append(pd.DataFrame(dict(k="mavceleb_" + vr.src[m] + "/" + vr.id[m], video=vr.video[m]))); VA.append(Vx[m])
            FR.append(pd.DataFrame(dict(k="mavceleb_" + fr.src[okf] + "/" + fr.id[okf], video=fr.video[okf]))); FA.append(Fx[okf])
            print(f"xjc MAV: voices kept {m.sum()} / {len(vr)} (LID/ok), faces {okf.sum()}", flush=True)
        else:    # rows_v1 caches (MAV v1 English + Urdu)
            r = pd.read_csv(CX / "rows_v1.csv"); assert not r.language.str.lower().isin(["bengali", "bangla"]).any()
            Vx = cat([np.load(CX / f"v1_voice_{e}.float32.npy") for e in VENC]); okv = np.all([np.load(CX / f"v1_voice_{e}.ok.npy") for e in VENC], 0)
            m = okv & ok_lid("v1:" + r.voice); VR.append(pd.DataFrame(dict(k="mavceleb_v1/" + r.identity[m], video=r.video[m]))); VA.append(Vx[m])
            Fx = U(np.load(CX / "v1_face_clip.float32.npy")); okf = np.load(CX / "v1_face_clip.ok.npy"); d = ~r.face.duplicated().to_numpy() & okf
            FR.append(pd.DataFrame(dict(k="mavceleb_v1/" + r.identity[d], video=r.face[d].str.split("/").str[2]))); FA.append(Fx[d])
            print(f"rows_v1 MAV: voices kept {m.sum()} / {len(r)} (LID-checked & clean), faces {d.sum()}", flush=True)
        r = pd.read_csv(CX / "rows_vox.csv"); Vx = cat([np.load(CX / f"vox_voice_{e}.float32.npy") for e in VENC]); okv = np.all([np.load(CX / f"vox_voice_{e}.ok.npy") for e in VENC], 0)
        ind = r.identity.astype(str).isin(india).to_numpy(); m = okv & (~ind | ok_lid("vox1:" + r.voice))
        VR.append(pd.DataFrame(dict(k="voxceleb1/" + r.identity[m], video=r.video[m]))); VA.append(Vx[m])
        Fx = U(np.load(CX / "vox_face_clip.float32.npy")); okf = np.load(CX / "vox_face_clip.ok.npy"); d = ~r.face.duplicated().to_numpy() & okf
        FR.append(pd.DataFrame(dict(k="voxceleb1/" + r.identity[d], video=r.face[d].str.split("/").str[1]))); FA.append(Fx[d])
        print(f"vox1: voices kept {m.sum()} / {len(r)} (India-id clips need LID: {int((ind & okv).sum())}, kept {int((ind & m).sum())}), faces {d.sum()}", flush=True)
        self.vm = pd.concat(VR, ignore_index=True); self.V = np.concatenate(VA); fm = pd.concat(FR, ignore_index=True); F = np.concatenate(FA)
        keep = fm.k.isin(set(self.vm.k)).to_numpy(); self.fm, self.F = fm[keep].reset_index(drop=True), F[keep]
        self.vm = self.vm[self.vm.k.isin(set(self.fm.k))].reset_index(drop=True); self.V = self.V[np.flatnonzero(pd.concat(VR, ignore_index=True).k.isin(set(self.fm.k)).to_numpy())]
        assert len(self.V) == len(self.vm)
        self.byk = {k: g.index.values for k, g in self.fm.groupby("k")}; self.bykv = {kv: g.index.values for kv, g in self.fm.groupby(["k", "video"])}
        print(f"ext C: voices {len(self.V)} faces {len(self.F)} ids {self.vm.k.nunique()} ({self.vm.k.str.split('/').str[0].value_counts().to_dict()})", flush=True)

    def sampler(self, drop_keys=()):   # = xj_ext.Ext.sampler (uniform)
        ok = np.flatnonzero(~self.vm.k.isin(set(drop_keys)).to_numpy()); ks, vids = self.vm.k.to_numpy(), self.vm.video.astype(str).to_numpy()
        def draw(rng, n):
            vi = rng.choice(ok, min(n, len(ok)), replace=False); fi = np.empty(len(vi), int)
            for j, v in enumerate(vi):
                c = self.bykv.get((ks[v], vids[v])); c = c if c is not None and len(c) else self.byk[ks[v]]; fi[j] = c[rng.integers(len(c))]
            return self.F[fi], self.V[vi], ks[vi]
        return draw

    def drop_for(self, test_ids): return [k for k, f in XE.OVERLAP.items() if f in set(test_ids)]


def eval_feats():
    ev = T3 / "outputs/eval_final"; t = pd.read_csv(ev / "rows.csv")
    F = U(np.load(ev / "feat/face_clip.npy")); V = cat([np.load(ev / f"feat/voice_{e}.npy") for e in EVV]); assert len(F) == len(V) == len(t)
    return t, F, V


def final(out, nseed=10, v2=False):
    out = Path(out); cfg = dict(model="XJ-C", face="clip", voice="cat4", hidden=256, outd=128, epochs=3, ratio=4, drop=0.7, seeds=[1000 * i for i in range(nseed)], v2=bool(v2), bnclip=True)
    out.mkdir(parents=True, exist_ok=True); cp = out / "config.json"
    if cp.exists(): assert json.load(open(cp)) == cfg, "config mismatch"
    json.dump(cfg, open(cp, "w"), indent=1)
    L, Ff, Vf = flag(); E = ExtC(v2); ms = []
    for i, sd in enumerate(cfg["seeds"]):
        p = out / f"xjc_m{i}.pt"
        if p.exists():
            m = X.XJ(768, 960); m.load_state_dict(torch.load(p, map_location="cpu")); m.eval(); ms.append(m); continue
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


def oof(tag, nseed=10, v2=False):
    sys.path.insert(0, str(T3 / "scripts")); cwd = os.getcwd(); os.chdir(T3 / "scripts")
    import t3_ultra_eval as E
    T = E.load_trials(); os.chdir(cwd)
    J = json.load(open(T3 / "release/sub90c/local_eval_kit/folds.json")); L, Ff, Vf = flag(); Ex = ExtC(v2); out = {}
    for k in range(5):
        held = set(J["reserve"]) | set(J["folds"][k]); tr = ~L.id.isin(held).to_numpy()
        ms = [X.train(Ff[tr], Vf[tr], seed=1000 * j + k, epochs=3, ext=Ex.sampler(Ex.drop_for(held)), ext_ratio=4) for j in range(nseed)]
        for pr in E.PR:
            t = T[f"f{k}_{pr}"]; assert not set(L.id.values[t["vi"]]) & set(L.id.values[tr])
            out[f"f{k}_{pr}"] = np.mean([X.score(m, Ff[t["fi"]], Vf[t["vi"]]) for m in ms], 0)
        print("done", k, flush=True)
    oc = dict(model="XJ-C", nseed=nseed, v2=bool(v2), epochs=3, ratio=4)
    np.savez(XD / f"xjc_oof_{tag}.npz", **out, oof_cfg=json.dumps(oc)); print("saved", tag, oc)


if __name__ == "__main__":
    kv = dict(a.split("=") for a in sys.argv[3:]); n, v2 = int(kv.get("nseed", 10)), kv.get("v2", "0") == "1"
    (final if sys.argv[1] == "final" else oof)(sys.argv[2], n, v2)
