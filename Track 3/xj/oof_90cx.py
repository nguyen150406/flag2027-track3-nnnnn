"""FLAG-train OOF of a 90c-X-family build on the kit CV trials (17/11, prereg 19:16 B), = the s90cx_oof.npz method (t3_90cx_gate.oof):
the build's stored LR constants (submissions/t3_<name>/provenance.json coefs) applied to its OOF columns, both LRs summed (log-odds).
  python oof_90cx.py <clone Track 3 dir> <build name> <out.npz> <model label>
Check: run on the 90c-X clone with sub90cx -> must equal xj/s90cx_oof.npz."""
import os, sys, json
from pathlib import Path
import numpy as np
os.environ["T3R2_OURS"] = "D"; os.environ["T3R2_NOGG"] = "1"
C, name, out, label = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]).resolve(), sys.argv[4]
os.chdir(C / "scripts"); sys.path.insert(0, str(C / "scripts")); import t3_r2_build as RB
assert str(RB.E.T3) == str(C), (RB.E.T3, C)
P = json.load(open(C / f"submissions/t3_{name}/provenance.json")); colf = RB.train_cols("E1"); res = {}
for pr in ("no_gender", "gender"):
    for k in range(5):
        tot = 0
        for nm, cols in (("ours", RB.OURS), ("r2sys", RB.R2C)):
            cols = [c for c in cols if not (pr == "gender" and c == "gender")]; c = P["coefs"][pr][nm]; assert c["cols"] == cols, (pr, nm)
            Xm = np.stack([colf(q, f"f{k}_{pr}") for q in cols], 1)
            tot = tot + ((Xm - np.array(c["mean"])) / np.array(c["scale"])) @ np.array(c["coef"]) + c["b"]
        assert len(tot) == 6000; res[f"f{k}_{pr}"] = tot
np.savez(out, **res, oof_cfg=json.dumps(dict(model=label, source=f"{name} provenance LR constants on its OOF columns (xj/oof_90cx.py, clone {C.parent.name})")))
print("wrote", out)
