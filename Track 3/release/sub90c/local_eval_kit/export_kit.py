"""Export the team's LOCAL evaluation (the sets behind t3_final.py gate_all / outputs/ultra/desc/multi.py) as system-agnostic trial lists
plus the per-trial scores of sub90c, so any system can be scored on the SAME trials and compared with sub90c (score_local.py).
Written 13/11/2026 for the sub90c full pack. Run once, from Track 3/scripts:
  python ../release/sub90c/local_eval_kit/export_kit.py      -> local_eval_kit/trials.csv.gz (+ prints the sub90c numbers as a self-check)
Sets per protocol (no_gender / gender):
  CV      FLAG-train OOF trials of the 5 RF-split folds (3000 pos + 3000 neg per fold); sub90c = OOF score (heads = fold models, LR refit
          without the test fold).  HARD = subset flag inside CV (cross-session-like positives + age-close negatives).
  RES     10 reserve FLAG-train people never in any fold; sub90c = fold-model heads + LR fitted on all 5 folds.
  MAV     MAV-Celeb v1 English voices, 55 people (split MV = 16 never used by any arm, TR = the rest); FINAL models, as on eval.
  EN, XL  held-out VoxCeleb2 / VoxBlink2 people (EN = English, XL = South-Asian-language VoxBlink); sub90c column = PARTIAL system
          LR[NF, farlD, r2] fitted on FLAG OOF (the ImageBind / gender / age columns do not exist for these rows), as in multi.py.
Media references (no media in this kit): flag:<relpath in the official FLAG train data>, mav1:<relpath in MAV-Celeb v1>,
  vox2:<id/video/clip>.wav (+ our face frame <clip>_<k>.jpg), vb:<id/video/clip>.wav (VoxBlink2)."""
import os, sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; SCRIPTS = HERE.parents[2] / "scripts"; sys.path.insert(0, str(SCRIPTS)); os.chdir(SCRIPTS)
import t3_final as F, t3_ultra_eval as E
from t3lib import TRAIN, SPK
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

T = F._T; EXT = E.T3 / "caches/ext"; ROWS = {s: pd.read_csv(EXT / f"rows_{s}.csv") for s in ("vox2", "vb")}
fl = lambda col, ix: ("flag:" + TRAIN[col].astype(str).to_numpy()[ix]).astype(object)


def mav_rel(p): return "mav1:" + p.split("/mavx/", 1)[1]


def partial_scores(pr):
    fc = F.flag_cols(F.BASE); cols = ["NF", "farlD", "r2"]
    X = np.vstack([np.stack([fc[c][f"f{k}_{pr}"] for c in cols], 1) for k in range(5)]); y = np.concatenate([T[f"f{k}_{pr}"]["y"] for k in range(5)])
    sc = StandardScaler().fit(X); m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(X), y)
    return {g: m.decision_function(sc.transform(np.stack([fc[c][f"{g}_{pr}"] for c in cols], 1))) for g in ("EN", "XL")}


def main():
    out = []
    for pr in E.PR:
        b = F.evaluate(F.BASE, pr); S, q25, _ = F.session(pr)
        y = np.concatenate([T[f"f{k}_{pr}"]["y"] for k in range(5)]); fs = np.concatenate([S[f"f{k}_{pr}"][0] for k in range(5)])
        gap = np.concatenate([S[f"f{k}_{pr}"][1] for k in range(5)]); hard = ((y == 1) & (fs < q25)) | ((y == 0) & (gap < 5))
        fi = np.concatenate([T[f"f{k}_{pr}"]["fi"] for k in range(5)]); vi = np.concatenate([T[f"f{k}_{pr}"]["vi"] for k in range(5)])
        fold = np.concatenate([np.full(len(T[f"f{k}_{pr}"]["y"]), k) for k in range(5)]); assert (b["cv"][0] == y).all()
        out.append(pd.DataFrame(dict(protocol=pr, set="CV", fold=fold, hard=hard.astype(int), y=y, face_ref=fl("face_relpath", fi), voice_ref=fl("voice_relpath", vi),
                                     face_person=SPK[fi], voice_person=SPK[vi], sub90c=b["cv"][1])))
        t = T[f"res_{pr}"]; assert (b["res"][0] == t["y"]).all()
        out.append(pd.DataFrame(dict(protocol=pr, set="RES", fold=-1, hard=0, y=t["y"], face_ref=fl("face_relpath", t["fi"]), voice_ref=fl("voice_relpath", t["vi"]),
                                     face_person=SPK[t["fi"]], voice_person=SPK[t["vi"]], sub90c=b["res"][1])))
        tm, _ = F.row_cols(F.BASE, F.MAV); ma = tm.protocol.eq(pr).to_numpy(); tm = tm[ma]; assert (tm.y.to_numpy() == b["all"][0]).all()
        out.append(pd.DataFrame(dict(protocol=pr, set="MAV", fold=-1, hard=0, y=tm.y.to_numpy(), face_ref=tm.face.map(mav_rel).to_numpy(), voice_ref=tm.voice.map(mav_rel).to_numpy(),
                                     face_person=tm.identity_f.to_numpy(), voice_person=tm.identity_v.to_numpy(), sub90c=b["all"][1], mav_split=tm.split.to_numpy())))
        ps = partial_scores(pr)
        for g in ("EN", "XL"):
            t = T[f"{g}_{pr}"]
            ref = lambda src, row, col: np.array([f"{s}:{ROWS[s][col].iloc[r]}" for s, r in zip(src, row)], object)
            per = lambda src, row: np.array([ROWS[s].identity.iloc[r] for s, r in zip(src, row)], object)
            out.append(pd.DataFrame(dict(protocol=pr, set=g, fold=-1, hard=0, y=t["y"], face_ref=ref(t["fsrc"], t["frow"], "face"), voice_ref=ref(t["vsrc"], t["vrow"], "voice"),
                                         face_person=per(t["fsrc"], t["frow"]), voice_person=per(t["vsrc"], t["vrow"]), sub90c=ps[g])))
            assert (out[-1].voice_person.to_numpy() == t["vid"]).all()
    D = pd.concat(out, ignore_index=True); D["mav_split"] = D["mav_split"].fillna("")
    D.insert(0, "trial_id", [f"{p}|{s}|{i}" for p, s, i in zip(D.protocol, D.set, D.groupby(["protocol", "set"]).cumcount())])
    D.to_csv(HERE / "trials.csv.gz", index=False, float_format="%.9g")
    print("wrote", HERE / "trials.csv.gz", len(D), "trials"); print(D.groupby(["protocol", "set"]).size().to_string())
    json.dump(dict(folds=E.SP["folds"], reserve=E.SP["reserve"], seed=E.SP["seed"], note="FLAG-train people; CV fold k = trials whose face AND voice people are in folds[k]"),
              open(HERE / "folds.json", "w"), indent=1)
    H = []
    for g in ("MAV", "EN", "XL"):
        d = D[D.set.eq(g)]
        for p in sorted(set(d.face_person) | set(d.voice_person)):
            src = "mav1" if g == "MAV" else ("vb" if str(p).startswith("vb_") else "vox2")
            H.append(dict(set=g, source=src, person=p, mav_split=(d.mav_split[d.voice_person.eq(p)].iloc[0] if g == "MAV" and (d.voice_person == p).any() else "")))
    pd.DataFrame(H).drop_duplicates(["source", "person"]).to_csv(HERE / "heldout_people.csv", index=False)
    print("wrote folds.json, heldout_people.csv", len(H))
    for pr in E.PR:   # self-check vs multi.py (outputs/ultra/desc/multi7.txt): no_gender 13.38/20.34/15.32/18.19/17.75/10.23/21.08, gender 20.56/29.95/24.88/26.37/24.95/13.95/22.55
        d = D[D.protocol.eq(pr)]; e = lambda m: E.eer(d.y[m], d.sub90c[m])
        print(pr, "sub90c CV %.2f HARD %.2f RES %.2f MAV-ALL %.2f MV %.2f EN %.2f XL %.2f" % (e(d.set.eq("CV")), e(d.set.eq("CV") & d.hard.eq(1)), e(d.set.eq("RES")),
              e(d.set.eq("MAV")), e(d.set.eq("MAV") & d.mav_split.eq("MV")), e(d.set.eq("EN")), e(d.set.eq("XL"))))


if __name__ == "__main__":
    main()
