"""90c-X soft local gate (prereg 15/11, descriptive; user "chấm tạm local cũng được nhưng không quá tin"). Run INSIDE the clone after
t3_90cx_chain.sh:  python t3_90cx_gate.py sub90cx
New-system FLAG-train OOF = its own stored LR constants (provenance of submissions/t3_<name>) applied to its OOF columns (same as
xj/fuse_b.oof_90c), compared with sub90c's OOF (kit column) on the kit CV trials: CV, HARD, LID-Bengali FLAG clips (rule any), rest (EN).
Reference = sub90c's OOF computed THE SAME WAY (stored LR constants on its own OOF columns), saved BEFORE the chain overwrites the
clone's head columns:  python t3_90cx_gate.py --save-ref sub90c_clonecheck   (-> ../outputs/ultra/ref_oof_sub90c.npz)
Gate: new - 90c <= max(0.5, 2 SE) (person bootstrap, voice side) on every subset, both protocols; else recommend not submitting."""
import os, sys, json
from pathlib import Path
import numpy as np, pandas as pd
os.environ["T3R2_OURS"] = "D"; os.environ["T3R2_NOGG"] = "1"
HERE = Path(__file__).resolve().parent; T3 = HERE.parent; assert "FLAG_2027_90cx" in str(T3), "run inside the 90c-X clone"
cwd = os.getcwd(); os.chdir(HERE); sys.path.insert(0, str(HERE)); import t3_r2_build as RB; os.chdir(cwd)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "xj")); import xj_lib as X   # FLAG LID mask (read only)
SAVE = sys.argv[1] == "--save-ref"; name = sys.argv[-1]; P = json.load(open(T3 / f"submissions/t3_{name}/provenance.json")); NB = int(os.environ.get("NBOOT", 500))
K = pd.read_csv(T3 / "release/sub90c/local_eval_kit/trials.csv.gz", keep_default_na=False)
L, _, _ = X.load_flag(); bn = dict(zip(L.voice, X.flag_bengali_mask(L, "any")))
colf = RB.train_cols("E1"); T = RB.E.load_trials()


def oof(pr):
    tot = 0
    for nm, cols in (("ours", RB.OURS), ("r2sys", RB.R2C)):
        cols = [c for c in cols if not (pr == "gender" and c == "gender")]; c = P["coefs"][pr][nm]; assert c["cols"] == cols, (pr, nm)
        Xm = np.vstack([np.stack([colf(q, f"f{k}_{pr}") for q in cols], 1) for k in range(5)])
        tot = tot + ((Xm - np.array(c["mean"])) / np.array(c["scale"])) @ np.array(c["coef"]) + c["b"]
    return tot


def boot(y, s1, s0, ids, seed=0):
    rng = np.random.default_rng(seed); U = np.unique(ids); idx = {u: np.flatnonzero(ids == u) for u in U}; d = []
    for _ in range(NB):
        ix = np.concatenate([idx[u] for u in rng.choice(U, len(U))])
        if y[ix].min() != y[ix].max(): d.append(X.eer(y[ix], s1[ix]) - X.eer(y[ix], s0[ix]))
    return float(np.std(d))


REF = T3 / "outputs/ultra/ref_oof_sub90c.npz"
if SAVE:
    assert not REF.exists(); np.savez(REF, **{pr: oof(pr) for pr in ("no_gender", "gender")}); print("saved", REF); sys.exit()
R0 = np.load(REF)
ok = True
print(f"{'protocol':9s} {'subset':6s} {'n':>6s} {'new':>6s} {'90c':>6s} {'delta':>6s} {'SE':>5s} gate")
for pr in ("no_gender", "gender"):
    c = K[K.protocol.eq(pr) & K.set.eq("CV")].reset_index(drop=True); y = c.y.to_numpy(); s9 = R0[pr]; assert len(s9) == len(y)
    y2 = np.concatenate([T[f"f{k}_{pr}"]["y"] for k in range(5)]); assert (y2 == y).all(), "kit order"
    sn = oof(pr); b = c.voice_ref.str.replace("flag:", "", regex=False).map(bn).to_numpy(bool); pid = c.voice_person.to_numpy()
    for nm, m in (("CV", np.ones(len(y), bool)), ("HARD", c.hard.to_numpy(bool)), ("BN", b), ("EN", ~b)):
        en, e9 = X.eer(y[m], sn[m]), X.eer(y[m], s9[m]); se = boot(y[m], sn[m], s9[m], pid[m]); p = en - e9 <= max(.5, 2 * se); ok &= p
        print(f"{pr:9s} {nm:6s} {m.sum():6d} {en:6.2f} {e9:6.2f} {en - e9:+6.2f} {se:5.2f} {'pass' if p else 'FAIL'}")
print("SOFT GATE (non-inferiority vs sub90c OOF):", "PASS" if ok else "FAIL")
