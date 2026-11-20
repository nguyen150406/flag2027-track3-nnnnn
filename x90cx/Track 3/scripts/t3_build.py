"""Track 3 production builder: train frozen-input projectors on all 70 FLAG train IDs
(+ allowed external MAV/VoxCeleb identities), score every dev/eval pair independently,
write the 4-file submission ZIP (distance: lower = more likely same person).

Pair-local by construction: score_i = f(face_i, voice_i; theta) with theta fit on
training/external data only. No test statistics, normalisation, cohorts or graphs.
Usage: python t3_build.py <config.json> <out_dir>
"""
from __future__ import annotations
import json, sys, zipfile, hashlib
from pathlib import Path
import numpy as np, pandas as pd, torch
from t3lib import *
from t3train import train_projector, pair_cos
import t3data as D

EXPECTED = {("no_gender", "English"): "no_gender/sub_score_v4_English_heard.txt",
            ("no_gender", "Bangla"): "no_gender/sub_score_v4_Bangla_unheard.txt",
            ("gender", "English"): "gender/sub_score_v4_English_heard.txt",
            ("gender", "Bangla"): "gender/sub_score_v4_Bangla_unheard.txt"}
AGE = ROOT / "Track 1/age_compatibility_experiments/cache"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def named_copy(out):
    """<out>/<name>.zip next to submission.zip (t3_sub24a -> sub24a.zip) so leaderboard uploads show which build they are."""
    import shutil; out = Path(out); shutil.copyfile(out / "submission.zip", out / f"{out.name.removeprefix('t3_')}.zip")


def aux_for(fn, vn, spec):
    out = []
    for s in spec:
        if s["src"] == "v1a":  # MAV v1 English train anchors (Track 2 caches)
            out.append(dict(D.mav(fn, vn)[0], weight=s.get("w", 1.0))); continue
        out.append(D.ext_source(s["src"], fn, vn, langs=s.get("langs"), max_faces_per_video=s.get("cap"), w=s.get("w", 1.0)))
    return out


def age_offset(fa, va):
    return float(np.median([np.median(fa[SPK == s]) - np.median(va[SPK == s]) for s in np.unique(SPK)]))


def gender_head(X, y, spk):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    _, inv, cnt = np.unique(spk, return_inverse=True, return_counts=True)
    h = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=2027))
    h.fit(X, y, logisticregression__sample_weight=1.0 / cnt[inv])
    return h


def all_rows_face_vit():
    """ViT face age for all 11349 rows (train cache + per-cell dev caches, manifest order)."""
    out = np.full(len(META), np.nan, np.float32); out[:6485] = np.load(AGE / "train_face_age.npy")
    for (proto, lang) in EXPECTED:
        fr = META[META.partition.eq("dev") & META.protocol.eq(proto) & META.language.eq(lang)].sort_values("manifest_order")
        out[fr.row_id.to_numpy()] = np.load(AGE / f"{proto}_{lang}_face_age.npy")
    assert np.isfinite(out).all(); return out


def voice_age_v2(fa_tr, H_tr, v24_tr, H, v24):
    """age_v2 voice age of any rows (H, v24): standardisation, ridge and offset are fitted on the 6,485 FLAG train rows
    (fa_tr, H_tr, v24_tr) only, then applied row by row (t3_eval.py reuses it on eval rows)."""
    from sklearn.linear_model import RidgeCV
    assert len(fa_tr) == len(H_tr) == len(v24_tr) == len(SPK)
    mu, sd = H_tr.mean(0), H_tr.std(0) + 1e-6
    target = pd.Series(fa_tr).groupby(SPK).median().loc[SPK].values
    ridge = RidgeCV(alphas=np.logspace(0, 5, 12)).fit((H_tr - mu) / sd, target)
    return .5 * ridge.predict((H - mu) / sd) + .5 * np.clip(v24 + age_offset(fa_tr, v24_tr), 0, 1)


def age_hidden(a24, a6):
    """audeering hidden-state input of the voice-age ridge: [a6 | a24] (a6 first, as built since SUB14a6)."""
    return np.hstack(([a6.astype(np.float32)] if a6 is not None else []) + [a24.astype(np.float32)])


def age_v2(cfg):
    """face = mean(ViT, MiVOLO); voice = .5*ridge(audeering hidden -> ID-median face age, 70 train IDs)
    + .5*(audeering-24 age + train offset). Returns (face_age[11349], voice_age[11349])."""
    fa = (all_rows_face_vit() + np.load(T3 / "caches/mivolo_face/face_age.npy") / 100) / 2
    import os; vs = os.environ.get("T3_VAGE", "")  # "_seg3": voice age with fixed 3-s sub-segment averaging (research A1, 2026-11-04); gender_v2 unchanged
    a24 = np.load(T3 / f"caches/flag_voiceage24{vs}.npz")
    a6 = np.load(T3 / f"caches/flag_voiceage6{vs}.npz")["emb"] if cfg.get("use_a6_hidden", True) and (T3 / f"caches/flag_voiceage6{vs}.npz").exists() else None
    H = age_hidden(a24["emb"], a6); v24 = a24["age"]
    return fa, voice_age_v2(fa[:6485], H[:6485], v24[:6485], H, v24)


def vox2_voice_age():
    """Face->voice age transfer: ridge (alpha 100) on audeering-24 hidden states of 58k VoxCeleb2 clips (A+B), target =
    ViT expected face age of the clip's frames mapped to the FLAG face-age scale (linear fit ViT -> mean(ViT, MiVOLO) on
    FLAG train rows). Returns raw predictions for all 11349 FLAG voice rows (no FLAG labels used in the ridge)."""
    from sklearn.linear_model import Ridge
    C = np.array([1, 6, 14.5, 24.5, 34.5, 44.5, 54.5, 64.5, 75]) / 100
    def vit_age(lg):
        lg = np.asarray(lg, np.float32); q = np.exp(lg - lg.max(1, keepdims=True)); return (q / q.sum(1, keepdims=True)) @ C
    EXT = T3 / "caches/ext"; X, Y = [], []
    for tag, rows_f, vit_f in (("vox2a", "rows_vox2.csv", "vox2_face_vit.npy"), ("vox2b", "rows_vox2b.csv", "vox2b_face_vit.npy")):
        a = np.load(EXT / f"{tag}_voice_aud24.npz"); r = pd.read_csv(EXT / rows_f, usecols=["identity", "clip_index"])
        r["fa"] = vit_age(np.load(EXT / vit_f, mmap_mode="r")[:, -9:]); per = r.groupby("clip_index").fa.mean()
        keep = np.isin(a["clip_index"], per.index); X.append(a["emb"][keep].astype(np.float32)); Y.append(per.loc[a["clip_index"][keep]].to_numpy())
    X, Y = np.vstack(X), np.concatenate(Y)
    vit = np.load(AGE / "train_face_age.npy"); mean = (vit + np.load(T3 / "caches/mivolo_face/face_age.npy")[:6485] / 100) / 2
    b1, b0 = np.polyfit(vit, mean, 1); mu, sd = X.mean(0), X.std(0) + 1e-6
    r = Ridge(alpha=100.0).fit((X - mu) / sd, b1 * Y + b0)
    return r.predict((np.load(T3 / "caches/flag_voiceage24.npz")["emb"].astype(np.float32) - mu) / sd)


def age_v3(cfg):
    """age_v2 with the voice age averaged with the VoxCeleb2 face->voice transfer estimate (+ train-ID median offset)."""
    fa, va2 = age_v2(cfg); vx = vox2_voice_age(); off = age_offset(fa[:6485], vx[:6485])
    return fa, .5 * va2 + .5 * np.clip(vx + off, 0, 1)


def age_g2(cfg, mix=False):
    """G2 (teammate report, 2026-10-26; CV in t3_age_g2.py): voice age = .5*ridge + .5*clip(a24 + offset), ridge/offset/normalisation
    fitted on IDENTITY means of the 70 FLAG train IDs + Bengali-filtered VoxCeleb2 A+B (ViT face age mapped to the FLAG scale).
    mix: .5*v2 + .5*G2."""
    import t3_age_g2 as G
    fa, va2 = age_v2(cfg); a24 = np.load(T3 / "caches/flag_voiceage24.npz"); H = a24["emb"].astype(np.float32)
    b1, b0 = np.polyfit(np.load(AGE / "train_face_age.npy"), fa[:6485], 1); fid, fage, vid, Hv, aud = G.vox2()
    fn, info = G.fit_voice_age([dict(face_id=SPK, face_age=fa[:6485], voice_id=SPK, hid=H[:6485], aud_age=a24["age"][:6485]),
                                dict(face_id=fid, face_age=b1 * fage + b0, voice_id=vid, hid=Hv, aud_age=aud)])
    print("G2 voice age fit:", info, flush=True); g2 = fn(H, a24["age"])
    return fa, (.5 * va2 + .5 * g2) if mix else g2


def gender_v2():
    """P(male): face MiVOLO-v2, voice audeering-24 gender head (female/male/child -> male share)."""
    g = np.load(T3 / "caches/flag_voiceage24.npz")["gender"]
    return np.load(T3 / "caches/mivolo_face/face_pmale.npy"), g[:, 1] / (g[:, 0] + g[:, 1])


def main(cfg_path, out_dir):
    cfg = json.loads(Path(cfg_path).read_text()); out = Path(out_dir)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    rows = np.arange(6485)  # all FLAG train rows (70 identities)
    meta = META
    models = {}
    for b in cfg["branches"]:
        fn, vn = b["face"], b["voice"]
        F, V = np.asarray(D.face(fn)), np.asarray(D.voice_full(vn))
        aux = aux_for(fn, vn, b.get("aux", cfg.get("aux", [])))
        if cfg.get("trainer", {}).get("w_sg_aux"):  # same-gender contrast on external identities needs their gender labels
            aux = [D.attach_gender(a) for a in aux]
        models[b["name"]] = (F, V, [train_projector(F, V, rows, SPK, seed=s, aux=aux, **cfg.get("trainer", {}))
                                    for s in cfg["seeds"]])
        print("trained", b["name"], flush=True)
    male = np.array([GENDER[s] == "m" for s in SPK], int)
    gF = np.asarray(D.face(cfg["gender_face"]))[:6485]; gV = np.asarray(D.voice_full(cfg["gender_voice"]))[:6485]
    fh, vh = gender_head(gF, male, SPK), gender_head(gV, male, SPK)
    tfa, tva = np.load(AGE / "train_face_age.npy"), np.load(AGE / "train_voice_age.npy")
    off = age_offset(tfa, tva)
    if cfg.get("age_mode") == "v2":
        FA2, VA2 = age_v2(cfg)
    elif cfg.get("age_mode") == "v3":
        FA2, VA2 = age_v3(cfg)
    if cfg.get("gender_mode") == "v2":
        PF2, PV2 = gender_v2()
    W = cfg["weights"]
    files = {}
    for (proto, lang), name in EXPECTED.items():
        ids = [l.split()[0] for l in (ROOT / "data/dev_set" / proto / f"{lang}_test.txt").read_text().splitlines()]
        fr = meta[meta.partition.eq("dev") & meta.protocol.eq(proto) & meta.language.eq(lang)].sort_values("manifest_order")
        assert fr.pair_id.tolist() == ids
        r = fr.row_id.to_numpy()
        score = np.zeros(len(r)); comp = {}
        for b in cfg["branches"]:
            F, V, ms = models[b["name"]]
            FS = [np.asarray(D.face(n)) for n in b["face_dev"]] if b.get("face_dev") else [F]  # cross-fit LoRA: average over the per-LoRA test embeddings
            comp[b["name"]] = np.mean([pair_cos(m, Fx, V, r, r) for m in ms for Fx in FS], 0)
            score += W[b["name"]] * comp[b["name"]]
        if cfg.get("age_mode") in ("v2", "v3"):
            comp["age"] = -np.abs(FA2[r] - VA2[r])
            score += W["age"] * comp["age"]
        else:
            fa = np.load(AGE / f"{proto}_{lang}_face_age.npy"); va = np.load(AGE / f"{proto}_{lang}_voice_age.npy")
            score += W["age"] * -np.abs(fa - np.clip(va + off, 0, 1))
        if proto == "no_gender" and W.get("gender", 0):
            if cfg.get("gender_mode") == "v2":
                pf, pv = PF2[r], PV2[r]
            else:
                pf = fh.predict_proba(np.asarray(D.face(cfg["gender_face"]))[r])[:, 1]
                pv = vh.predict_proba(np.asarray(D.voice_full(cfg["gender_voice"]))[r])[:, 1]
            comp["gender"] = np.log(np.clip(pf * pv + (1 - pf) * (1 - pv), 1e-4, 1))
            score += W["gender"] * comp["gender"]
        # pair-locality check: scoring a single pair alone gives the same value
        b0 = cfg["branches"][0]; F, V, ms = models[b0["name"]]
        assert abs(pair_cos(ms[0], F, V, r[:1], r[:1])[0] - pair_cos(ms[0], F, V, r, r)[0]) < 1e-5
        np.savez(out / f"components_{proto}_{lang}.npz", ids=np.array(ids), **comp)
        dist = -score
        assert np.isfinite(dist).all()
        p = out / name; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(f"{i} {d:.8f}\n" for i, d in zip(ids, dist)))
        files[name] = dict(rows=len(ids), sha256=sha(p))
    with zipfile.ZipFile(out / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in EXPECTED.values():
            z.write(out / name, name)
    json.dump(dict(config=cfg, files=files, zip_sha256=sha(out / "submission.zip"), pair_local=True,
                   test_statistics_used=False, train_ids=70), open(out / "provenance.json", "w"), indent=1)
    named_copy(out); print("wrote", out / "submission.zip")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
