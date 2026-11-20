"""Direction 2 (user 14/11 "Có làm đi 2 điều luôn"): 90c-remix = re-weight sub90c's own English components so that they COMPLEMENT the
XJ member, using FLAG-train out-of-fold trials only (Q5: weights from training data).
  sub90c English logit = LR_ours[NF, FaRL, gender, age] + LR_r2[r2, ImageBind, gender, age]  (= one linear map of the union of components)
  fused (as in sub95a) = 0.5 * z_xj + 0.5 * z_90 ;  z = (score - mu) / sd with train-OOF constants
  remix: keep that a-priori 0.5/0.5 fusion and XJ fixed, and choose the 90c direction beta (union components) that maximises the balanced
  logistic likelihood of a * (0.5 * z_xj + 0.5 * z(beta . c)) + b on the 30k RF-split OOF trials per protocol (5 folds). Start = sub90c's own
  direction; component weights constrained >= 0; 5-fold CV over the RF folds reported as a diagnostic (fit on 4 folds, score the 5th).
  The gender protocol has no gender column (sub90c NOGG setting).
usage: python remix_90c.py <xj_oof.npz> <out.json>"""
import os, sys, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
T3 = Path(__file__).resolve().parents[1]
os.environ["T3R2_OURS"] = "D"; os.environ["T3R2_NOGG"] = "1"
sys.path.insert(0, str(T3 / "scripts")); _cwd = os.getcwd(); os.chdir(T3 / "scripts")
import t3_r2_build as RB
os.chdir(_cwd)
P90 = json.load(open(T3 / "submissions/t3_sub90c_eval/provenance.json"))


def eer(y, s):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr; i = np.nanargmin(np.abs(fnr - fpr)); return 100 * float(fpr[i] + fnr[i]) / 2


def comps(pr): return ["NF", RB.FARL, "age", "r2", "ibo"] + ([] if pr == "gender" else ["gender"])


def base_beta(pr):
    """sub90c's English logit as ONE linear map of the union components: raw-space weights + constant (from its stored LR constants)."""
    w = dict.fromkeys(comps(pr), 0.0); c0 = 0.0
    for nm in ("ours", "r2sys"):
        c = P90["coefs"][pr][nm]
        for col, m, s, k in zip(c["cols"], c["mean"], c["scale"], c["coef"]): w[col] += k / s; c0 -= k * m / s
        c0 += c["b"]
    return np.array([w[q] for q in comps(pr)]), c0


def load(pr, xj_npz, folds=range(5)):
    T = RB.E.load_trials(); colf = RB.train_cols("E1"); Z = np.load(xj_npz)
    X = np.vstack([np.stack([colf(q, f"f{k}_{pr}") for q in comps(pr)], 1) for k in folds]).astype(float)
    y = np.concatenate([T[f"f{k}_{pr}"]["y"] for k in folds]).astype(int); x = np.concatenate([Z[f"f{k}_{pr}"] for k in folds]).astype(float)
    f = np.concatenate([np.full(len(T[f"f{k}_{pr}"]["y"]), k) for k in folds]); assert len(X) == len(y) == len(x) and np.isfinite(X).all() and np.isfinite(x).all()
    return X, y, x, f


def fused(X, x, beta, cx, cu):
    """0.5 z_xj + 0.5 z_u with constants cx = (mu, sd) of xj and cu = (mu, sd) of u = X @ beta (both from the fitting trials)."""
    u = X @ beta; return 0.5 * (x - cx[0]) / cx[1] + 0.5 * (u - cu[0]) / cu[1]


def fit(X, y, x, beta0):
    """returns raw-space beta maximising the balanced logistic likelihood of a*fused + b (fused uses constants of these trials)."""
    mx, sx = X.mean(0), X.std(0); Zs = (X - mx) / sx; zx = (x - x.mean()) / x.std(); w = np.where(y == 1, len(y) / (2 * y.sum()), len(y) / (2 * (len(y) - y.sum())))
    t = 2 * y - 1
    def loss(p):
        b_, a, c = p[:-2], p[-2], p[-1]; u = Zs @ b_; su = u.std()
        if su < 1e-9: return 1e9
        s = 0.5 * zx + 0.5 * (u - u.mean()) / su; m = t * (a * s + c); return float(np.sum(w * np.logaddexp(0, -m)) / len(y))
    b0 = beta0 * sx; b0 = b0 / np.linalg.norm(b0); best = None
    for a0 in (1.0, 2.0, 4.0):
        # every component is a similarity score (higher = same person): weights kept >= 0 (no suppressor terms; decided 14/11 on the dev fit,
        # train data only: an unconstrained fit put a NEGATIVE weight on FaRL, a FLAG-specific suppression that is fragile under domain shift)
        r = minimize(loss, np.r_[b0, a0, 0.0], method="L-BFGS-B", bounds=[(0, None)] * len(b0) + [(None, None)] * 2)
        if best is None or r.fun < best.fun: best = r
    lb = min(minimize(lambda q: loss(np.r_[b0, q]), np.array([a0, 0.0]), method="L-BFGS-B").fun for a0 in (1.0, 2.0, 4.0))   # base at its own best (a, c)
    bz = best.x[:-2]; bz = bz / np.linalg.norm(bz); return bz / sx, best.fun, lb


def main(xj_npz, out_json):
    import hashlib; Zc = np.load(xj_npz); assert "oof_cfg" in Zc.files, "XJ OOF file without its recipe (rerun xj_oof.py)"
    res = dict(xj_oof=str(Path(xj_npz).resolve()), xj_oof_sha256=hashlib.sha256(Path(xj_npz).read_bytes()).hexdigest(), xj_oof_cfg=json.loads(str(Zc["oof_cfg"])),
               method=__doc__.split("usage")[0].strip(), protocols={})
    for pr in ("no_gender", "gender"):
        X, y, x, f = load(pr, xj_npz); b0, c0 = base_beta(pr)
        l90 = X @ b0 + c0; s90 = None
        # bit-check: the union map reproduces sub90c's train-OOF log-odds (fuse_b.oof_90c recomputation)
        import importlib.util; spec = importlib.util.spec_from_file_location("fb", Path(__file__).parent / "fuse_b.py"); fb = importlib.util.module_from_spec(spec); spec.loader.exec_module(fb)
        s90, y90 = fb.oof_90c(pr); assert (y90 == y).all() and np.allclose(s90, l90, atol=1e-8), "union map != sub90c OOF logit"
        beta, l_new, l_base = fit(X, y, x, b0)
        cx, cu, cb = (x.mean(), x.std()), ((X @ beta).mean(), (X @ beta).std()), ((X @ b0).mean(), (X @ b0).std())
        e = dict(xj=eer(y, x), c90=eer(y, l90), remix_alone=eer(y, X @ beta), fused_base=eer(y, fused(X, x, b0, cx, cb)), fused_remix=eer(y, fused(X, x, beta, cx, cu)))
        cv = {"fused_base": [], "fused_remix": []}
        for k in range(5):   # diagnostic only: fit on 4 RF folds, score the 5th (constants from the 4 training folds)
            tr, te = f != k, f == k; bk, _, _ = fit(X[tr], y[tr], x[tr], b0)
            cxk = (x[tr].mean(), x[tr].std()); cuk = ((X[tr] @ bk).mean(), (X[tr] @ bk).std()); cbk = ((X[tr] @ b0).mean(), (X[tr] @ b0).std())
            cv["fused_base"].append(eer(y[te], fused(X[te], x[te], b0, cxk, cbk))); cv["fused_remix"].append(eer(y[te], fused(X[te], x[te], bk, cxk, cuk)))
        d = np.array(cv["fused_remix"]) - np.array(cv["fused_base"])
        nb0, nb = b0 * X.std(0), beta * X.std(0); nb0, nb = nb0 / np.abs(nb0).sum(), nb / np.abs(nb).sum()
        print(f"{pr}: OOF EER xj {e['xj']:.2f} | 90c {e['c90']:.2f} | remix alone {e['remix_alone']:.2f} | fused base {e['fused_base']:.2f} -> remix {e['fused_remix']:.2f} "
              f"(balanced logloss {l_base:.4f} -> {l_new:.4f}) | CV delta {d.mean():+.2f} +- {d.std(ddof=1) / np.sqrt(5):.2f} per fold {np.round(d, 2).tolist()}")
        print("   standardised weight share (|w| sums to 1):", {q: (round(a, 3), round(b, 3)) for q, a, b in zip(comps(pr), nb0, nb)}, "(base, remix)")
        res["protocols"][pr] = dict(comps=comps(pr), beta_raw=beta.tolist(), beta_base_raw=b0.tolist(), base_const=c0, u_mu=float(cu[0]), u_sd=float(cu[1]),
                                    xj_mu=float(cx[0]), xj_sd=float(cx[1]), n_trials=int(len(y)), oof_eer={k: round(v, 3) for k, v in e.items()},
                                    cv_fold_eer={k: [round(v, 3) for v in vv] for k, vv in cv.items()}, weight_share=dict(base=dict(zip(comps(pr), nb0.round(4).tolist())), remix=dict(zip(comps(pr), nb.round(4).tolist()))))
    json.dump(res, open(out_json, "w"), indent=1); print("wrote", out_json)


if __name__ == "__main__":
    main(*sys.argv[1:3])
