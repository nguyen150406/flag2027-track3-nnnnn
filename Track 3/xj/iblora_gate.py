"""IB-LoRA gate + XJ-style member dir (16/11, prereg PREREG_gate_all.txt 20:10 "PREREG sub118a").
score = unit(f - mu_f) . unit(v - mu_v); OOF on the kit CV trials with fold-fit FLAG-train means; eval with the full FLAG-train means (no eval statistic).
G1 (prereg numbers, full-train centring as the 16/11 zero-shot screen): IB-LoRA gender <= 30.4 and no_gender <= 23.94 (frozen IB reproduced alongside).
G2: equal z-sum with XJ-1+IB (constants = each member's OOF) not worse than XJ-1+IB OOF (14.49 / 21.58) by > 1.0 in either protocol.
Writes xj/ibl_10 (config.json model "IB-LoRA", cos_<tr>_<lang>.npy, <tr>/sub_score_v4_<lang>.txt = -cos) and xj/ibl_oof_10.npz (oof_cfg) for fuse_ab.
exit 0 = G1 and G2 pass, 3 = a gate failed."""
import os, sys, json
from pathlib import Path
import numpy as np, pandas as pd
XD = Path(__file__).resolve().parent; T3 = XD.parent; sys.path.insert(0, str(T3 / "scripts")); cwd = os.getcwd(); os.chdir(T3 / "scripts")
import t3_ultra_eval as E
TR = E.load_trials(); SP = E.SP; os.chdir(cwd)
from t3lib import SPK
EN = {"English": "sub_score_v4_English_heard.txt", "Bangla": "sub_score_v4_Bangla_unheard.txt"}
U = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
o = T3 / "outputs/ultra"; Fl, Vl = np.load(o / "ibl_flag_face.npy").astype(np.float64), np.load(o / "ibl_flag_voice.npy").astype(np.float64)
Fi, Vi = np.load(o / "ib_own_flag_face.npy").astype(np.float64), np.load(o / "ib_own_flag_voice.npy").astype(np.float64)
assert Fl.shape == Vl.shape == Fi.shape == (6485, 1024)
res, oof, ok = {}, {}, True
for pr in E.PR:
    y = np.concatenate([TR[f"f{k}_{pr}"]["y"] for k in range(5)]); r = {}
    for nm, F_, V_ in (("IB-LoRA", Fl, Vl), ("frozen IB", Fi, Vi)):
        A, B = U(F_ - F_.mean(0)), U(V_ - V_.mean(0)); r[f"{nm} full-centred"] = round(E.eer(y, np.concatenate([(A[TR[f"f{k}_{pr}"]["fi"]] * B[TR[f"f{k}_{pr}"]["vi"]]).sum(1) for k in range(5)])), 2)
        sc = []
        for k in range(5):
            fit = np.isin(SPK, [p for j, f in enumerate(SP["folds"]) if j != k for p in f]); t = TR[f"f{k}_{pr}"]
            sc.append((U(F_[t["fi"]] - F_[fit].mean(0)) * U(V_[t["vi"]] - V_[fit].mean(0))).sum(1))
        r[f"{nm} fold-centred"] = round(E.eer(y, np.concatenate(sc)), 2)
        if nm == "IB-LoRA": oof.update({f"f{k}_{pr}": sc[k] for k in range(5)})
    X = np.load(XD / "xj1ib_oof_x1ib_10.npz"); xs = [X[f"f{k}_{pr}"].astype(float) for k in range(5)]; ls = [oof[f"f{k}_{pr}"] for k in range(5)]
    z = lambda L_: [(a - np.concatenate(L_).mean()) / np.concatenate(L_).std() for a in L_]
    zx, zl = z(xs), z(ls); r["XJ-1+IB OOF"] = round(E.eer(y, np.concatenate(xs)), 2); r["z-sum OOF"] = round(E.eer(y, np.concatenate([a + b for a, b in zip(zx, zl)])), 2)
    assert abs(r["frozen IB full-centred"] - {"no_gender": 23.94, "gender": 33.41}[pr]) < .015 and abs(r["XJ-1+IB OOF"] - {"no_gender": 14.49, "gender": 21.58}[pr]) < .015, f"alignment canary failed {r}"   # review 16/11
    g1 = r["IB-LoRA full-centred"] <= (30.4 if pr == "gender" else 23.94); g2 = r["z-sum OOF"] <= r["XJ-1+IB OOF"] + 1.0
    r["G1"], r["G2"] = bool(g1), bool(g2); ok &= g1 and g2; res[pr] = r; print(pr, json.dumps(r), flush=True)
# ---- member dir for fuse_ab (eval rows order = rows.csv per protocol / language, ids == xj1ib_10)
t = pd.read_csv(T3 / "outputs/eval_final/rows.csv"); ef, ev = np.load(T3 / "outputs/eval_final/ultra/ibl_face.npy").astype(np.float64), np.load(T3 / "outputs/eval_final/ultra/ibl_voice.npy").astype(np.float64)
assert len(ef) == len(ev) == len(t) and np.isfinite(ef).all() and np.isfinite(ev).all()
s_all = (U(ef - Fl.mean(0)) * U(ev - Vl.mean(0))).sum(1)
D = XD / "ibl_10"; D.mkdir(exist_ok=True)
for pr in E.PR:
    (D / pr).mkdir(exist_ok=True)
    for lang, fn in EN.items():
        m = (t.protocol.eq(pr) & t.language.eq(lang)).to_numpy(); ids = t.pair_id[m].astype(str).tolist(); s = s_all[m]
        ref = [l.split()[0] for l in open(XD / "xj1ib_10" / pr / fn)]; assert ids == ref, f"ids/order {pr} {lang}"
        np.save(D / f"cos_{pr}_{lang}.npy", s); (D / pr / fn).write_text("".join(f"{i} {-v:.8f}\n" for i, v in zip(ids, s)))
json.dump(dict(model="IB-LoRA", score="unit(f - mu_f).unit(v - mu_v), FLAG-train means", gates=res, prereg="PREREG_gate_all.txt 16/11 20:10 sub118a"), open(D / "config.json", "w"), indent=1)
np.savez(XD / "ibl_oof_10.npz", **oof, oof_cfg=json.dumps(dict(model="IB-LoRA", centring="fold-fit FLAG-train means", nseed=1)))
print("GATES", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 3)
