"""fuse_lid (15/11 sub107a, user "làm thử cái này đi", prereg PREREG_gate_all.txt 19:20): clip-language routing inside the ENGLISH cells.
Per English trial: if its voice clip is flagged Bengali by ALL THREE LID models of the LID v3 tool (scripts/t3_lid_audit.py: VoxLingua107,
whisper-large-v3, mms-lid-256; max over 30-s windows of P(bn)+P(as) > .5 each; the all3 rule of the FLAG-train filter) -> score from sub90c (the
system of our Bangla files), else from XJ-1 (= sub101b English). Both are put on one scale by a per-protocol logistic (Platt) calibration fitted on
FLAG-train out-of-fold trials (kit CV trials, 5 RF folds): XJ-1 OOF cos and sub90c OOF logit (kit column). File value = -(calibrated log-odds).
Non-routed trials keep XJ-1's order exactly. Bangla files = bytes of <bn_sub> (sub99b = sub90c). Each clip's LID is computed on that clip alone
(per-clip feature); no eval statistic and no identity information enter the rule.
usage: python fuse_lid.py <name> <xj dir> <xj oof.npz> <lid dir> <bn_sub> [xjref]
   e.g. python fuse_lid.py sub107a xj1_10 xj1_oof_x1_10.npz ../outputs/lid_eval_en sub99b
15/11 sub109a: xjref = "sub101b" (default: the XJ dir must reproduce sub101b English = XJ-1) or "self" (another XJ model, e.g. XJ-1+IB: its cos must
reproduce the English files written by its own final run). The XJ model name in the provenance comes from <xj dir>/config.json."""
import os, sys, json, zipfile, hashlib, tempfile
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
XD = Path(__file__).resolve().parent; T3 = XD.parent; SUB = T3 / "submissions"; sys.path.insert(0, str(XD))
import xj_lib as X
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(); shab = lambda b: hashlib.sha256(b).hexdigest()
TRS = ("no_gender", "gender"); EN = "sub_score_v4_English_heard.txt"; BN = "sub_score_v4_Bangla_unheard.txt"
read = lambda p: pd.read_csv(p, sep=" ", header=None, names=["id", "s"], dtype={"id": str})


def platt(s, y):
    m = LogisticRegression(C=1e6, class_weight="balanced", max_iter=1000).fit(s[:, None], y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def lid_flags(lid_dir):
    D = None
    for m in ("vl", "wh", "mms"):
        d = pd.read_csv(Path(lid_dir) / m / "lid_0of1.csv"); assert d.key.is_unique and (d.dur > 0).all() and not (d[f"{m}_top"] == "ERR").any(), m
        D = d[["key", "dur", f"{m}_bnas"]] if D is None else D.merge(d[["key", f"{m}_bnas"]], on="key", how="inner", validate="1:1")
    clips = pd.read_csv(Path(lid_dir) / "clips.csv"); assert set(D.key) == set(clips.key) and len(D) == len(clips), "LID rows != clip list"
    D["all3"] = (D[["vl_bnas", "wh_bnas", "mms_bnas"]] > .5).all(axis=1); D["any"] = (D[["vl_bnas", "wh_bnas", "mms_bnas"]] > .5).any(axis=1)
    return D.set_index("key")


def main(name, xjdir, xj_oof, lid_dir, bn_sub, xjref="sub101b"):
    xjdir, xj_oof, lid_dir = (XD / xjdir).resolve(), (XD / xj_oof).resolve(), (XD / lid_dir).resolve()
    out = SUB / f"t3_{name}"; assert not out.exists(), out
    assert xjref in ("sub101b", "self"), xjref; XN = json.load(open(xjdir / "config.json"))["model"]
    K = pd.read_csv(T3 / "release/sub90c/local_eval_kit/trials.csv.gz", keep_default_na=False); O = np.load(xj_oof)
    cwd = os.getcwd(); os.chdir(T3 / "scripts"); sys.path.insert(0, str(T3 / "scripts")); import t3_ultra_eval as E; T = E.load_trials(); os.chdir(cwd)
    t = pd.read_csv(T3 / "outputs/eval_final/rows.csv"); FL = lid_flags(lid_dir)
    ref = zipfile.ZipFile(SUB / "t3_sub101b/sub101b.zip"); bz = zipfile.ZipFile(SUB / f"t3_{bn_sub}/{bn_sub}.zip")
    tmp = Path(tempfile.mkdtemp(dir=SUB, prefix=f".t3_{name}_")); files, rep = {}, {}
    for tr in TRS:
        # FLAG-train OOF calibration (kit CV trials; XJ-1 OOF in the same order, checked through the labels as t3_90cx_gate)
        c = K[K.protocol.eq(tr) & K.set.eq("CV")].reset_index(drop=True); y = c.y.to_numpy()
        assert (np.concatenate([T[f"f{k}_{tr}"]["y"] for k in range(5)]) == y).all(), "kit order"
        sx = np.concatenate([O[f"f{k}_{tr}"] for k in range(5)]).astype(float); s9 = c.sub90c.to_numpy(float)
        ax, bx = platt(sx, y); a9, b9 = platt(s9, y); assert ax > 0 and a9 > 0, "calibration slope"
        e_x, e_9 = X.eer(y, sx), X.eer(y, s9)
        # eval
        msk = (t.protocol.eq(tr) & t.language.eq("English")).to_numpy(); te = t[msk].reset_index(drop=True)
        cos = np.load(xjdir / f"cos_{tr}_English.npy").astype(float); b9f = read(SUB / "t3_sub90c_eval" / tr / EN); rb = ref.read(f"{tr}/{EN}").decode().splitlines()
        ids101 = [l.split()[0] for l in rb]; v101 = np.array([float(l.split()[1]) for l in rb])
        assert len(cos) == len(te) == len(b9f) == len(ids101) and (te.pair_id.astype(str).values == b9f.id.values).all() and list(b9f.id) == ids101, "ids/order"
        if xjref == "sub101b": assert np.corrcoef(-cos, v101)[0, 1] > .999999, "XJ-1 cos != sub101b English"
        else:
            own = read(xjdir / tr / EN); assert list(own.id) == ids101 and np.allclose(-own.s.to_numpy(float), cos, rtol=0, atol=2e-6), f"{XN} cos != its own English file"
        l9 = -b9f.s.to_numpy(float)                                   # sub90c eval logit (file = -logit)
        key = tr + "/English/" + te.voice.str.split("/").str[-1]; f3 = key.map(FL.all3); assert f3.notna().all(), "LID missing for some trials"
        route = f3.to_numpy(bool)
        zx, z9 = ax * cos + bx, a9 * l9 + b9
        fused = np.where(route, z9, zx); assert np.isfinite(fused).all()
        p = tmp / tr / EN; p.parent.mkdir(exist_ok=True); p.write_text("".join(f"{i} {-v:.8f}\n" for i, v in zip(ids101, fused)))
        files[f"{tr}/{EN}"] = dict(rows=int(len(fused)), sha256=sha(p), source=f"{XN} ({xjdir.name}) | sub90c on voices flagged Bengali by all 3 LID models; Platt on FLAG-train OOF")
        nr = ~route
        rep[tr] = dict(calib=dict(xj=dict(a=ax, b=bx), s90c=dict(a=a9, b=b9)), oof_eer=dict(xj=round(e_x, 3), s90c=round(e_9, 3)),
                       routed=dict(n=int(route.sum()), of=int(len(route)), share=round(float(route.mean()), 4),
                                   clips_any=int(key.map(FL["any"]).sum())),
                       health=dict(oof_logodds_mean=dict(xj=round(float((ax * sx + bx).mean()), 3), s90c=round(float((a9 * s9 + b9).mean()), 3)),
                                   eval_logodds_mean_all_trials=dict(xj=round(float(zx.mean()), 3), s90c=round(float(z9.mean()), 3)),
                                   eval_logodds_sd_all_trials=dict(xj=round(float(zx.std()), 3), s90c=round(float(z9.std()), 3)),
                                   corr_fused_vs_101b=round(float(np.corrcoef(-fused, v101)[0, 1]), 5),
                                   nonrouted_order_kept=bool(np.array_equal(np.argsort(zx[nr], kind="stable"), np.argsort(cos[nr], kind="stable")))))
        print(f"{tr}: calib XJ a={ax:.3f} b={bx:.3f} | 90c a={a9:.3f} b={b9:.3f} | OOF EER {XN} {e_x:.2f} 90c {e_9:.2f} | routed {route.sum()}/{len(route)} "
              f"(any-rule {rep[tr]['routed']['clips_any']}) | eval log-odds mean XJ {zx.mean():.2f} 90c {z9.mean():.2f} (OOF {rep[tr]['health']['oof_logodds_mean']}) "
              f"| corr vs 101b {rep[tr]['health']['corr_fused_vs_101b']}")
        for n in (f"{tr}/{BN}",):
            b = bz.read(n); (tmp / n).write_bytes(b); files[n] = dict(rows=len(b.decode().splitlines()), sha256=shab(b), source=f"{bn_sub} bytes (= sub90c)")
    zp = tmp / f"{name}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for tr in TRS:
            for n in (f"{tr}/{EN}", f"{tr}/{BN}"): z.write(tmp / n, n)
    rel = lambda q: str(Path(q).resolve().relative_to(T3.parent)) if Path(q).resolve().is_relative_to(T3.parent) else str(Path(q).resolve())
    json.dump(dict(system=("two-branch. English = " + XN + " (" + rel(xjdir) + "), except trials whose voice clip all three LID models (VoxLingua107, whisper-large-v3, "
                           "mms-lid-256; P(bn)+P(as) > .5 on any 30-s window) flag as Bengali, which take sub90c's score; both scores mapped to log-odds by a logistic "
                           f"calibration fitted on FLAG-train out-of-fold trials. Bangla = {bn_sub} bytes (= sub90c)."),
                   method=__doc__.split("usage")[0].strip(), protocols=rep, files=files,
                   input_sha256={rel(q): sha(q) for q in [xj_oof, xjdir / "config.json", T3 / "release/sub90c/local_eval_kit/trials.csv.gz", T3 / "outputs/eval_final/rows.csv",
                                                          lid_dir / "clips.csv"] + [lid_dir / m / "lid_0of1.csv" for m in ("vl", "wh", "mms")]
                                 + [xjdir / f"cos_{tr}_English.npy" for tr in TRS] + [SUB / "t3_sub90c_eval" / tr / EN for tr in TRS]},
                   code_sha256={"xj/fuse_lid.py": sha(XD / "fuse_lid.py"), "scripts/t3_lid_audit.py": sha(T3 / "scripts/t3_lid_audit.py")},
                   zip_sha256=sha(zp), pair_local=True, test_statistics_used=False, weights_tuned_on_eval=False), open(tmp / "provenance.json", "w"), indent=1)
    tmp.rename(out); print("wrote", out / f"{name}.zip", sha(out / f"{name}.zip")[:16])


if __name__ == "__main__":
    main(*sys.argv[1:])
