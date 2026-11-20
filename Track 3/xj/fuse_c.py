"""Tickets of the two-directions plan (user 14/11 "Thử phát triển 2 cái của 95a theo 2 hướng" -> "Có làm đi 2 điều luôn").
English = 0.5 * z(XJ member) + 0.5 * z(90c member), weights fixed a priori (as sub95a); Bangla = sub90c bit-exact. file = -fused (lower = match).
  XJ member  = mean centred cosine of an XJ-family model dir (cos_<tr>_English.npy);
               mu/sd = its train-OOF scores on the RF-split FLAG trials (xj_oof.py npz: the SAME trials as sub90c's OOF; recipe checked vs the dir)
  90c member = sub90c English logit (its file)                       -> mu/sd from sub90c's train OOF (fuse_b.oof_90c)
            or 90c-remix (remix_90c.py json: weights over sub90c's components, fitted on train OOF with XJ fixed)
               -> u = beta . components; eval components recomputed with the sub90c pipeline and bit-checked (sub90c's stored LRs on them
               reproduce the sub90c file); mu/sd of u from the train OOF
No eval statistic is used: each eval score = f(that pair's component scores; constants fixed from FLAG-train OOF). The eval z-drift printed at
the end is a health report only. Review 14/11 fixes: all inputs resolved/checked before anything is written, build in a temp dir renamed at the end,
OOF recipe == model-dir recipe, remix tied to the OOF file by sha + constants, bit-check rtol=0, sign asserts, fuller provenance.
usage: python fuse_c.py <name> <xj model dir> <xj_oof.npz> [remix.json]"""
import os, sys, json, zipfile, hashlib, shutil, importlib.util, tempfile
from pathlib import Path
import numpy as np, pandas as pd
XD = Path(__file__).resolve().parent; T3 = XD.parent; SUB = T3 / "submissions"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mod(n):
    s = importlib.util.spec_from_file_location(n, XD / f"{n}.py"); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
fb, rm = mod("fuse_b"), mod("remix_90c"); RB = fb.RB
NX = lambda tr: f"{tr}/sub_score_v4_English_heard.txt"; NB = lambda tr: f"{tr}/sub_score_v4_Bangla_unheard.txt"
TRS = ("no_gender", "gender")


def read(p): return pd.read_csv(p, sep=" ", header=None, names=["id", "s"], dtype={"id": str})


def check_recipe(cfg, oc):
    """the OOF npz must come from the same XJ recipe as the model dir (xj_oof.py can only vary these keys; everything else must be default)."""
    assert oc["nseed"] == len(cfg["seeds"]) and oc["epochs"] == cfg["epochs"] and oc["ratio"] == cfg["ratio"], (oc, cfg)
    assert oc["langs"] == cfg["langs"] and oc["bn"] == cfg["bengali"] and oc["bnclip"] == cfg.get("bnclip", False) and oc["mavx"] == cfg["mavx"] and not oc["mask"], (oc, cfg)
    assert cfg["hidden"] == 256 and cfg["outd"] == 128 and cfg["flag"] == "none" and not cfg["vb"] and not cfg["vox2"] and not cfg["mav_lid"] and not cfg["gpure"], cfg
    assert not ({"bs", "drop", "mav_w"} & set(cfg)), cfg
    assert cfg.get("bnclip") and cfg["mavx"], "provenance text below describes the clip-level filter + MAVX recipe"


def main(name, xjdir, xj_oof, remix=None):
    xjdir, xj_oof = Path(xjdir).resolve(), Path(xj_oof).resolve(); remix = Path(remix).resolve() if remix else None
    out = SUB / f"t3_{name}"; assert not out.exists(), out
    cfg = json.load(open(xjdir / "config.json")); Z = np.load(xj_oof); assert "oof_cfg" in Z.files, "OOF npz without its recipe"; oc = json.loads(str(Z["oof_cfg"]))
    check_recipe(cfg, oc)
    R = json.load(open(remix)) if remix else None
    if R: assert Path(R["xj_oof"]).resolve() == xj_oof and R["xj_oof_sha256"] == sha(xj_oof), "remix was fitted on a different XJ OOF file"
    heads = [O for b in RB.NF + [RB.FARL, "r2E1_clipnf"] for O in [RB.O / f"head_{b}.npz"]]
    inputs = [xj_oof, T3 / "outputs/ultra/trials.npz", XD / "lid_xjx3.csv", XD / "lid_xjx_urdu_hindi.csv", XD / "lid_xjx_mav_english.csv",
              T3.parent / "external_mavceleb/mavx_dl/lid_mavx.csv", T3 / "caches/ext/exclude_bengali_v3.csv",
              XD / "ext_feats/faces.npy", XD / "ext_feats/voices.npy", XD / "ext_feats/faces_meta.csv", XD / "ext_feats/voices_meta.csv"] + sorted((XD / "mavx").glob("mavx*")) \
             + [xjdir / f"cos_{tr}_English.npy" for tr in TRS] + [xjdir / NX(tr) for tr in TRS] + [xjdir / "config.json"] \
             + [SUB / "t3_sub90c_eval" / n for tr in TRS for n in (NX(tr), NB(tr))] + [SUB / "t3_sub90c_eval/provenance.json", RB.O / "attr.npz", RB.O / "ib_own.npz"] + heads \
             + ([remix, T3 / "outputs/eval_final/rows.csv"] if remix else [])
    miss = [str(q) for q in inputs if not Path(q).exists()]; assert not miss, f"inputs missing: {miss}"
    rel = lambda q: str(Path(q).resolve().relative_to(T3.parent))
    in_sha = {rel(q): sha(q) for q in inputs}
    T = RB.E.load_trials()
    if R:   # eval components of sub90c, all rows; bit-checked against the sub90c files below
        cwd = os.getcwd(); os.chdir(T3 / "scripts"); ev = T3 / "outputs/eval_final"; t, col, nr2 = RB.eval_cols("E1", ev, ev / "rows.csv"); os.chdir(cwd)
    tmp = Path(tempfile.mkdtemp(dir=SUB, prefix=f".t3_{name}_")); files, const, drift, comp_sha = {}, {}, {}, {}
    for tr in TRS:
        y = np.concatenate([T[f"f{k}_{tr}"]["y"] for k in range(5)]).astype(int); xo = np.concatenate([Z[f"f{k}_{tr}"] for k in range(5)]).astype(float)
        mx, sx = float(xo.mean()), float(xo.std())
        b = read(SUB / "t3_sub90c_eval" / NX(tr)); a = read(xjdir / NX(tr)); cos = np.load(xjdir / f"cos_{tr}_English.npy").astype(float)
        assert len(a) == len(b) == len(cos) and (a.id.values == b.id.values).all() and np.allclose(np.round(-cos, 6), a.s.values, rtol=0, atol=1e-6), "XJ dir files/cos/ids"
        if R is None:
            m_eval = -b.s.values.astype(float); s9, y9 = fb.oof_90c(tr); assert (y9 == y).all(); oo = s9; how = "sub90c logit"
        else:
            P = R["protocols"][tr]; assert P["comps"] == rm.comps(tr) and np.isclose(P["xj_mu"], mx, rtol=0, atol=1e-12) and np.isclose(P["xj_sd"], sx, rtol=0, atol=1e-12)
            msk = (t.protocol.eq(tr) & t.language.eq("English")).to_numpy(); assert (t.pair_id.astype(str)[msk].values == b.id.values).all(), "eval row order"
            M = np.stack([col[q][msk] for q in P["comps"]], 1).astype(float); assert np.isfinite(M).all(); comp_sha[tr] = hashlib.sha256(np.ascontiguousarray(M).tobytes()).hexdigest()
            b0, c0 = rm.base_beta(tr); assert np.allclose(b0, P["beta_base_raw"], rtol=0, atol=1e-12)
            assert np.allclose(M @ b0 + c0, -b.s.values.astype(float), rtol=0, atol=2e-6), "recomputed components do not reproduce the sub90c file"
            beta = np.array(P["beta_raw"]); assert (beta >= 0).all(); m_eval = M @ beta
            Xo, yo, xo2, _ = rm.load(tr, xj_oof); assert (yo == y).all() and np.allclose(xo2, xo, rtol=0, atol=0); oo = Xo @ beta; how = "90c-remix"
            assert np.isclose(oo.mean(), P["u_mu"], rtol=0, atol=1e-9) and np.isclose(oo.std(), P["u_sd"], rtol=0, atol=1e-9)
        mm, sm = float(oo.mean()), float(oo.std())
        f = 0.5 * (cos - mx) / sx + 0.5 * (m_eval - mm) / sm; assert np.isfinite(f).all()
        fo = 0.5 * (xo - mx) / sx + 0.5 * (oo - mm) / sm
        ex, e9, ef = fb.eer(y, xo), fb.eer(y, oo), fb.eer(y, fo); assert max(ex, e9, ef) < 50, "sign: higher must mean same person on train OOF"
        zx, z9 = (cos - mx) / sx, (m_eval - mm) / sm
        drift[tr] = dict(xj_eval_z_mean=round(float(zx.mean()), 3), xj_eval_z_sd=round(float(zx.std()), 3), m90_eval_z_mean=round(float(z9.mean()), 3), m90_eval_z_sd=round(float(z9.std()), 3),
                         eval_corr=round(float(np.corrcoef(cos, m_eval)[0, 1]), 3))
        const[tr] = dict(xj=dict(mu=mx, sd=sx, oof_eer=round(ex, 3)), m90=dict(kind=how, mu=mm, sd=sm, oof_eer=round(e9, 3)), fused_oof_eer=round(ef, 3), n_trials=int(len(y)))
        print(f"{tr}: train-OOF EER XJ {ex:.2f} | {how} {e9:.2f} | fused {ef:.2f} ; eval health (z mean/sd vs train 0/1): XJ {zx.mean():+.2f}/{zx.std():.2f} "
              f"{how} {z9.mean():+.2f}/{z9.std():.2f}, corr {drift[tr]['eval_corr']:.3f}")
        (tmp / tr).mkdir(); p = tmp / NX(tr); p.write_text("".join(f"{i} {-v:.8f}\n" for i, v in zip(b.id.values, f)))
        files[NX(tr)] = dict(rows=len(f), sha256=sha(p), source=f"0.5/0.5 fusion: XJ ({xjdir.name}) + {how}, train-OOF standardisation")
        shutil.copyfile(SUB / "t3_sub90c_eval" / NB(tr), tmp / NB(tr)); assert sha(tmp / NB(tr)) == sha(SUB / "t3_sub90c_eval" / NB(tr))
        files[NB(tr)] = dict(rows=sum(1 for _ in open(tmp / NB(tr))), sha256=sha(tmp / NB(tr)), source="sub90c bit-exact")
    zp = tmp / f"{name}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for tr in TRS:
            for n in (NX(tr), NB(tr)): z.write(tmp / n, n)
    code = {q: sha(XD / q) for q in ("fuse_c.py", "remix_90c.py", "fuse_b.py", "xj_lib.py", "xj_ext.py", "final_xj.py", "xj_oof.py", "xj_val.py")}
    code.update({"scripts/t3_r2_build.py": sha(T3 / "scripts/t3_r2_build.py"), "kaggle/xjx_lid3/xjx_lid3.py": sha(T3 / "kaggle/xjx_lid3/xjx_lid3.py"),
                 "kaggle/xjx_lidlist3/xjx_lid_clips.csv": sha(T3 / "kaggle/xjx_lidlist3/xjx_lid_clips.csv")})
    json.dump(dict(system=("two-branch. English = equal-weight (0.5/0.5, fixed a priori) late fusion of (a) an XJ-family system (own re-implementation of "
                           "teammate XJ_ens5: BTC VGGFace+ECAPA, 2 MLP towers 256->128, InfoNCE, FLAG train + external MAV-Celeb v1/v2 (+MAVX extra clips) + VoxCeleb1, "
                           f"clip-level Bengali filter, {len(cfg['seeds'])} seeds) and (b) " + ("sub90c English (our side + R2 system)" if R is None else
                           "90c-remix: sub90c's own English components (NF heads, FaRL head, R2 projector, ImageBind, gender, age) re-weighted on FLAG-train "
                           "out-of-fold trials to complement the fixed XJ member (non-negative weights)") +
                           ". Each member standardised with mean/sd of its own FLAG-train OOF scores on the same RF-split trials. Bangla = sub90c bit-exact."),
                   bengali_filter=("external MAV-Celeb + VoxCeleb1 India-nationality ids: CLIP-level LID v3 (VoxLingua107, Whisper-large-v3, MMS-LID; every 30-s window): "
                                   "a clip any model gives P(bn)+P(as) > .5, an unreadable/NaN clip and an unchecked clip are dropped; identities kept. MAVX: only LID-cleared "
                                   "clips. Other VoxCeleb1 ids: organizer-permitted, identity-level audit 0 flags. FLAG train: all official clips used (as every team system)."),
                   xj_model=dict(dir=rel(xjdir), config=cfg, oof_recipe=oc, checkpoints={p.name: sha(p) for p in sorted(xjdir.glob("xj_m*.pt"))}),
                   remix=(dict(file=rel(remix), protocols={tr: dict(comps=P["comps"], beta_raw=P["beta_raw"], weight_share=P["weight_share"], oof_eer=P["oof_eer"])
                                                          for tr, P in R["protocols"].items()}, eval_component_matrix_sha256=comp_sha) if R else None),
                   constants=const, eval_health_report=drift, code_sha256=code, input_sha256=in_sha,
                   files=files, zip_sha256=sha(zp), pair_local=True, test_statistics_used=False, weights_tuned_on_eval=False), open(tmp / "provenance.json", "w"), indent=1)
    os.rename(tmp, out); print("wrote", out / f"{name}.zip", sha(out / f"{name}.zip")[:16])


if __name__ == "__main__":
    main(*sys.argv[1:])
