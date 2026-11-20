"""Option B (user 14/11 "B đi"): English = equal-weight late fusion of two COMPLETE team systems, Bangla = sub90c bit-exact.
  match_xj  = cos of sub94a's English branch (xj_bnfilt_ens5, file = -cos)
  match_90c = logit of sub90c's English branch (file = -logit; our side + R2 LRs, log-odds 1:1)
  fused     = 0.5 * (match_xj - mu_xj) / sd_xj + 0.5 * (match_90c - mu_90c) / sd_90c ; file = -fused (lower = match)
Weights 0.5/0.5 fixed a priori. mu/sd per protocol from FLAG-TRAIN out-of-fold trials only (XJ: val_B0c = the sub94a recipe, 21 held-out-
identity runs; 90c: its 5 RF-split OOF folds pushed through the exact sub90c LRs, coefficients checked against its provenance).
No eval statistic is used: every eval score is f(that pair's two component scores; fixed constants).
usage: python fuse_b.py <name> [sd=total|within]"""
import os, sys, json, zipfile, hashlib, shutil
from pathlib import Path
import numpy as np, pandas as pd
T3 = Path(__file__).resolve().parents[1]; SUB = T3 / "submissions"; XD = Path(__file__).resolve().parent
os.environ["T3R2_OURS"] = "D"; os.environ["T3R2_NOGG"] = "1"          # the sub90c settings (its provenance: D heads, no gender column in gender)
sys.path.insert(0, str(T3 / "scripts")); cwd = os.getcwd(); os.chdir(T3 / "scripts")
import t3_r2_build as RB
os.chdir(cwd)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
P90 = json.load(open(SUB / "t3_sub90c_eval/provenance.json"))


def eer(y, s):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr; i = np.nanargmin(np.abs(fnr - fpr)); return 100 * float(fpr[i] + fnr[i]) / 2


def oof_90c(pr):
    """sub90c's train-OOF log-odds for protocol pr, using its stored LR constants; refit check against the provenance."""
    colf = RB.train_cols("E1"); T = RB.E.load_trials(); y = np.concatenate([T[f"f{k}_{pr}"]["y"] for k in range(5)]); tot = np.zeros(len(y))
    for nm, cols in (("ours", RB.OURS), ("r2sys", RB.R2C)):
        cols = [c for c in cols if not (pr == "gender" and c == "gender")]; c = P90["coefs"][pr][nm]; assert c["cols"] == cols, (pr, nm, cols, c["cols"])
        X = np.vstack([np.stack([colf(q, f"f{k}_{pr}") for q in cols], 1) for k in range(5)])
        sc, m = RB.fit_lr(cols, colf, pr)                                         # same fit as the sub90c build -> must give the stored constants
        assert np.allclose(sc.mean_, c["mean"]) and np.allclose(sc.scale_, c["scale"]) and np.allclose(m.coef_[0], c["coef"]) and np.isclose(m.intercept_[0], c["b"]), (pr, nm)
        tot += ((X - np.array(c["mean"])) / np.array(c["scale"])) @ np.array(c["coef"]) + c["b"]
    return tot, y


def stats(s, y, how):
    if how == "total": return float(s.mean()), float(s.std())
    return float(s.mean()), float(np.sqrt((np.var(s[y == 1]) + np.var(s[y == 0])) / 2))


def main(name, how="total"):
    out = SUB / f"t3_{name}"; assert not out.exists(), out; out.mkdir()
    Z = np.load(XD / "val_B0c.npz", allow_pickle=True); cfgc = json.load(open(XD / "val_B0c.json"))["cfg"]
    assert cfgc["flag"] == "none" and cfgc["langs"] is None and cfgc["hidden"] == 256 and cfgc["outd"] == 128 and not cfgc["vb"] and not cfgc["vox2"], "B0c must be the sub94a recipe"
    files, const = {}, {}
    for tr in ("no_gender", "gender"):
        sx, yx = Z[f"v1s_{tr}"].astype(float), Z[f"v1y_{tr}"].astype(int); s9, y9 = oof_90c(tr)
        assert eer(yx, sx) < 50 and eer(y9, s9) < 50, "sign: higher match score must mean same person on train OOF"
        mx, dx = stats(sx, yx, how); m9, d9 = stats(s9, y9, how)
        const[tr] = dict(xj=dict(mu=mx, sd=dx, n_trials=int(len(sx)), oof_eer=round(eer(yx, sx), 3)), c90=dict(mu=m9, sd=d9, n_trials=int(len(s9)), oof_eer=round(eer(y9, s9), 3)))
        print(f"{tr}: XJ OOF EER {eer(yx, sx):.2f} mu {mx:.4f} sd {dx:.4f} | 90c OOF EER {eer(y9, s9):.2f} mu {m9:.4f} sd {d9:.4f}")
        # eval components
        nx = f"{tr}/sub_score_v4_English_heard.txt"
        a = pd.read_csv(SUB / "t3_sub94a" / nx, sep=" ", header=None, names=["id", "s"], dtype={"id": str})
        b = pd.read_csv(SUB / "t3_sub90c_eval" / nx, sep=" ", header=None, names=["id", "s"], dtype={"id": str})
        cos = np.load(XD / f"xj_bnfilt_ens5/cos_{tr}_English.npy"); assert len(cos) == len(a) == len(b) and (a.id.values == b.id.values).all()
        assert np.allclose(np.round(-cos, 6), a.s.values, atol=1e-6), "sub94a file = -cos of xj_bnfilt_ens5"
        lg = -b.s.values.astype(float)
        f = 0.5 * (cos - mx) / dx + 0.5 * (lg - m9) / d9; assert np.isfinite(f).all()
        (out / tr).mkdir(exist_ok=True); p = out / nx; p.write_text("".join(f"{i} {-v:.8f}\n" for i, v in zip(a.id.values, f)))
        files[nx] = dict(rows=len(f), sha256=sha(p), source="0.5/0.5 fusion of sub94a English (XJ) and sub90c English, train-OOF standardisation")
        nb = f"{tr}/sub_score_v4_Bangla_unheard.txt"; shutil.copyfile(SUB / "t3_sub90c_eval" / nb, out / nb); assert sha(out / nb) == sha(SUB / "t3_sub90c_eval" / nb)
        files[nb] = dict(rows=sum(1 for _ in open(out / nb)), sha256=sha(out / nb), source="sub90c bit-exact")
    zp = out / f"{name}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for n in ("no_gender/sub_score_v4_English_heard.txt", "no_gender/sub_score_v4_Bangla_unheard.txt", "gender/sub_score_v4_English_heard.txt", "gender/sub_score_v4_Bangla_unheard.txt"): z.write(out / n, n)
    json.dump(dict(system=("two-branch. English = equal-weight (0.5/0.5, fixed a priori) late fusion of two complete team systems: (a) XJ-family "
                           "(sub94a English: re-implementation of teammate XJ_ens5, BTC VGGFace+ECAPA, 5 seeds, external MAV-Celeb v1/v2 + VoxCeleb1 with "
                           "the sub94a Bengali identity filter) and (b) sub90c English (our side + R2 system, log-odds 1:1). Each system's match score is "
                           "standardised with mean/sd of its OWN FLAG-train out-of-fold trial scores (no eval statistic). Bangla = sub90c bit-exact."),
                   standardisation=how, constants=const, component_files={k: sha(SUB / d / f"{tr}/sub_score_v4_English_heard.txt") for d, k in (("t3_sub94a", "sub94a"), ("t3_sub90c_eval", "sub90c")) for tr in ("no_gender", "gender")},
                   component_provenance=["submissions/t3_sub94a/provenance.json", "submissions/t3_sub90c_eval/provenance.json"],
                   code_sha256={"fuse_b.py": sha(__file__), "scripts/t3_r2_build.py": sha(T3 / "scripts/t3_r2_build.py")},
                   input_sha256={"xj/val_B0c.npz": sha(XD / "val_B0c.npz"), "xj/xj_bnfilt_ens5/cos_no_gender_English.npy": sha(XD / "xj_bnfilt_ens5/cos_no_gender_English.npy"),
                                 "xj/xj_bnfilt_ens5/cos_gender_English.npy": sha(XD / "xj_bnfilt_ens5/cos_gender_English.npy"), "outputs/ultra/trials.npz": sha(T3 / "outputs/ultra/trials.npz")},
                   files=files, zip_sha256=sha(zp), pair_local=True, test_statistics_used=False, weights_tuned_on_eval=False), open(out / "provenance.json", "w"), indent=1)
    print("wrote", zp)


if __name__ == "__main__":
    main(sys.argv[1], *(a.split("=")[1] for a in sys.argv[2:]))
