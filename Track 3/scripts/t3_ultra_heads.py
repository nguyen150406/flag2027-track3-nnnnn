"""ULTRA heads (user 2026-11-11 "chạy tối đa nhanh nhất"): Track 3 projector branches trained under the RF person split
(cv_run/w/split.json: 5 folds of 12 FLAG persons + 10 reserve persons).
  fold k : FLAG fit rows = persons of the other 4 folds (fold k AND the 10 reserve persons never seen) -> OOF / reserve scoring
  final  : all 70 FLAG persons (as the RF final projector, reserve included)
External sources (own AAM heads) minus caches/ext/exclude_ultra.csv = Bengali list v3 + 550 held-out external persons
(caches/ext/ultra_holdout.csv: 300 English cross-video persons, 250 South-Asian-language persons), so every model here has
never seen the held-out persons. No AVSpeech / LRS3 (Q21 access).
  python t3_ultra_heads.py <out_dir> <branch,...> <fold,...|final> <seed,...> [threads]
Saves <out_dir>/<branch>__<fold>__s<seed>.pt (whole model, torch.save) + a .done marker; skips finished ones (resumable)."""
import os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
os.environ.setdefault("T3_EXCLUDE", str(HERE.parent / "caches/ext/exclude_ultra.csv"))
import json, numpy as np, torch
from t3lib import SPK, T3
import t3data as D
from t3train import train_projector

FROZEN = dict(dim=512, w_proto=.6, w_con=.5, margin=.1, voice_dropout=.4, temp=.2, face_dropout=.7, w_sg=.5, epochs=40)   # t3_c38 trainer
LORA = dict(FROZEN, face_dropout=.55)                                                                                     # t3_c33a trainer
BR = {"farl_cat4": ("farl", "cat4", FROZEN, ("vox", "vox2", "vb")), "clip_cat4": ("clip", "cat4", FROZEN, ("vox", "vox2", "vb")),
      "clipnf_cat4": ("clipnf", "cat4", LORA, ("vox", "vox2", "vb")), "clipnf_sim": ("clipnf", "sim", LORA, ("vox", "vox2", "vb")),
      "clipnf2_cat4": ("clipnf2", "cat4", LORA, ("vox", "vox2", "vb")), "clipnf2_sim": ("clipnf2", "sim", LORA, ("vox", "vox2", "vb")),
      # R2 arm D (2026-11-12): same head, FLAG down-weighted (main losses x .25) + cross-video/cross-row positives (FLAG decouple, aux_xvideo)
      "clipnfD_cat4": ("clipnf", "cat4", dict(LORA, w_main=.25, decouple=True, aux_xvideo=True), ("vox", "vox2", "vb")),
      # 12/11 arm D won 8/8 cells (CV, reserve, EN, XL) on clipnf_cat4 -> the same change for every head of our side
      **{f"{f}D_{v}": (f, v, dict(LORA, w_main=.25, decouple=True, aux_xvideo=True), ("vox", "vox2", "vb")) for f in ("clipnf", "clipnf2") for v in ("cat4", "sim") if (f, v) != ("clipnf", "cat4")},
      "farlD_cat4": ("farl", "cat4", dict(FROZEN, w_main=.25, decouple=True, aux_xvideo=True), ("vox", "vox2", "vb")),
      # 12/11 data trade-off ablation (frozen CLIP face so every source has features; run with T3_EXCLUDE=exclude_ultra_mav.csv)
      **{f"clipD_{k}": ("clip", "cat4", dict(FROZEN, w_main=.25, decouple=True, aux_xvideo=True), ("vox", "vox2", "vb") + extra)
         for k, extra in (("base", ()), ("l", ("lrs3",)), ("a", ("avs", "avs2")), ("as", ("avssa",)), ("m", ("v1",)))},
      "ibv_cat4": ("ibv", "cat4", FROZEN, ("vox2", "vb")), "ibv_sim": ("ibv", "sim", FROZEN, ("vox2", "vb"))}
BR.update({f"{b}m": (BR[b][0], BR[b][1], BR[b][2], BR[b][3] + ("mavx",))   # 13/11 English fix: + 40 MAV-Celeb v1 English TRAIN people (aux source
           for b in ("clipnfD_cat4", "clipnfD_sim", "clipnf2D_cat4", "clipnf2D_sim", "farlD_cat4")})   # mavx, t3_mavx_source.py; no MV / LID / vox2-dup ids)
# 13/11 "p" branches = the "m" heads with the R2-EP pool drop (outputs/ultra/pool_drop10.csv: ~12% least-learnable vox2/VoxBlink ids by the E1
# cross-video face-voice AUC; R2 arm EP: FLAG CV/HARD/reserve better, XL worse). Same config otherwise.
BR.update({f"{b}p": BR[b] for b in ("clipnfD_cat4m", "clipnfD_simm", "clipnf2D_cat4m", "clipnf2D_simm", "farlD_cat4m")})
# 13/11 final-refit option "y" / "yp": MAV source "mavy" = mavx + the 15 MV hold-out ids (t3_mavx_source.py with T3_MAV_ALL=1; MAV-MV no longer held out)
BR.update({f"{b}{s_}": (BR[b][0], BR[b][1], BR[b][2], BR[b][3] + ("mavy",)) for b in ("clipnfD_cat4", "clipnfD_sim", "clipnf2D_cat4", "clipnf2D_sim", "farlD_cat4") for s_ in ("y", "yp")})
# 13/11 ENX "h" branches = the sub90c D heads + a same-gender FLAG-like (South-Asian-looking) external hard batch per source (R2 arm EH idea:
# gender MAV-ALL -2.15 fused); identities picked by a FLAG-likeness probe (FLAG train faces vs the source's faces, as t3_r2.Pool) + metadata gender.
BR.update({f"{b}h": (BR[b][0], BR[b][1], dict(BR[b][2], w_sah_aux=.5), BR[b][3]) for b in ("clipnfD_cat4", "clipnfD_sim", "clipnf2D_cat4", "clipnf2D_sim", "farlD_cat4")})
SPLIT = json.load(open(T3 / "cv_run/w/split.json"))


def sa_groups(Fflag, a, src, seed=0, thr=.5):
    """[female label ids, male label ids] of the source's FLAG-like identities (probe on this branch's face features, FLAG train rows only)."""
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    U = lambda X: X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    rng = np.random.default_rng(seed); Ff = U(np.nan_to_num(np.asarray(Fflag[:len(SPK)], np.float32))); Fa = U(np.nan_to_num(np.asarray(a["faces"], np.float32)))
    sel = rng.choice(len(Fa), min(len(Ff), len(Fa)), replace=False)
    ps = LogisticRegression(max_iter=2000, C=.5).fit(np.vstack([Ff, Fa[sel]]), np.r_[np.ones(len(Ff)), np.zeros(len(sel))])
    fl = np.asarray(a["face_labels"]); sa = pd.Series(ps.predict_proba(Fa)[:, 1]).groupby(fl).mean()
    names = pd.Series(np.asarray(a["face_names"]).astype(str)).groupby(fl).first(); gm = D.gender_map(); both = set(np.unique(np.asarray(a["labels"])).tolist())
    keep = [int(l) for l in sa[sa > thr].index if int(l) in both and gm.get(names[l]) is not None]
    out = [np.array([l for l in keep if not bool(gm[names[l]])], np.int64), np.array([l for l in keep if bool(gm[names[l]])], np.int64)]
    print(f"    SA hard ids {src}: f/m {len(out[0])}/{len(out[1])} (of {len(sa)} ids)", flush=True); return out


def drop_ids(a, src, path=T3 / "outputs/ultra/pool_drop10.csv"):
    """remove the listed identities of one external source from its aux dict (faces and voices), relabel 0..n-1"""
    import pandas as pd
    d = pd.read_csv(path).identity.astype(str); bad = set(d[d.str.startswith(src + "|")].str.split("|", n=1).str[1])
    if not bad: return a
    fk = ~np.isin(a["face_names"].astype(str), list(bad)); vk = ~np.isin(a["voice_names"].astype(str), list(bad))
    ids = sorted(set(a["face_names"][fk].astype(str)) & set(a["voice_names"][vk].astype(str))); lab = {x: i for i, x in enumerate(ids)}
    fk &= np.isin(a["face_names"].astype(str), ids); vk &= np.isin(a["voice_names"].astype(str), ids); o = dict(a)
    for k in ("faces", "face_names", "face_video", "face_lang", "face_rows"): o[k] = a[k][fk]
    for k in ("voices", "voice_names", "voice_video", "voice_lang", "voice_rows"): o[k] = a[k][vk]
    o["face_labels"] = np.array([lab[x] for x in o["face_names"].astype(str)]); o["labels"] = np.array([lab[x] for x in o["voice_names"].astype(str)])
    print(f"    drop {src}: {len(set(a['voice_names'].astype(str)))} -> {len(ids)} ids", flush=True); return o


def fit_rows(fold):
    if fold == "final": keep = set(SPK)
    else: keep = {p for j, f in enumerate(SPLIT["folds"]) if j != int(fold) for p in f}
    return np.flatnonzero(np.isin(SPK, sorted(keep)))


def main(out, branches, folds, seeds, threads=3):
    torch.set_num_threads(int(threads)); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    assert "ultra" in os.environ["T3_EXCLUDE"], os.environ["T3_EXCLUDE"]
    for b in branches.split(","):
        fn, vn, cfg, srcs = BR[b]
        if os.environ.get("T3U_EPOCHS"): cfg = dict(cfg, epochs=int(os.environ["T3U_EPOCHS"]))  # smoke tests only
        F, V = np.asarray(D.face(fn)), np.asarray(D.voice_full(vn)); aux = [D.ext_source(s, fn, vn) for s in srcs]
        if b.endswith(("mp", "yp")): aux = [drop_ids(x, s) for x, s in zip(aux, srcs)]  # no w_sg_aux in these trainers -> no gender attach (as t3_build)
        if cfg.get("w_sah_aux"): aux = [dict(x, sah=sa_groups(F, x, s)) for x, s in zip(aux, srcs)]   # 13/11 ENX "h" branches
        for fold in folds.split(","):
            rows = fit_rows(fold)
            for s in map(int, seeds.split(",")):
                p = out / f"{b}__{fold}__s{s}.pt"
                if p.with_suffix(".done").exists(): continue
                t = time.time(); m = train_projector(F, V, rows, SPK, seed=s, aux=aux, **cfg); m.cpu().eval()
                torch.save(m, p); p.with_suffix(".done").write_text(f"{time.time() - t:.0f}s ids {len(set(SPK[rows]))}\n")
                print(f"{b} fold {fold} seed {s}: {len(set(SPK[rows]))} FLAG ids, aux ids {[len(set(a['labels'])) for a in aux]}, {(time.time() - t) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
