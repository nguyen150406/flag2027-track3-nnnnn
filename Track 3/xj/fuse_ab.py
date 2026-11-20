"""fuse_ab (16/11, prereg PREREG_gate_all.txt "PREREG sub114a"): XJ-A+B = equal-weight z-sum of two complete XJ systems that use disjoint encoders
  A = XJ-U (xju10, OOF xju_oof_u10.npz)      identity encoders
  B = XJ-S (xjs_10, OOF xj1ibs_oof_s10.npz)  CLIP-LoRA / FaRL / ImageBind | SimAM / sam100ft / R293 / ECAPA-1024 / ImageBind audio
  s = (cos_A - mu_A) / sd_A + (cos_B - mu_B) / sd_B, per protocol; mu / sd = each system's FLAG-train OOF scores on the kit CV trials (no eval statistic).
Writes a model dir xjab_10 (config.json, cos_<tr>_English.npy, <tr>/sub_score_v4_English_heard.txt = -s) and OOF xjab_oof_ab10.npz (same z on the OOF),
so that fuse_lid.py can route / calibrate it exactly as XJ-1+IB in sub110a. Prints label-free eval diversity (Spearman) and descriptive OOF EERs.
usage: python fuse_ab.py                      (sub114a defaults)
       python fuse_ab.py <outdir> <oof.npz> <model name> <dirA>:<oofA>:<nameA> <dirB>:<oofB>:<nameB>"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
XD = Path(__file__).resolve().parent; T3 = XD.parent; sys.path.insert(0, str(XD))
import xj_lib as X
TRS, EN = ("no_gender", "gender"), "sub_score_v4_English_heard.txt"
MEM = {"A": ("xju10", "xju_oof_u10.npz", "XJ-U"), "B": ("xjs_10", "xj1ibs_oof_s10.npz", "XJ-S")}
OUT, OOFN, NAME = XD / "xjab_10", XD / "xjab_oof_ab10.npz", "XJ-A+B"
import os
WB = {k: float(v) for k, v in (x.split("=") for x in os.environ.get("AB_W", "").split(",") if x)}
if len(sys.argv) > 1:
    OUT, OOFN, NAME = XD / sys.argv[1], XD / sys.argv[2], sys.argv[3]
    MEM = {m: tuple(a.split(":")) for m, a in zip("AB", sys.argv[4:6])}; assert all(len(v) == 3 for v in MEM.values()), MEM


def ids_of(d, tr): return [l.split()[0] for l in open(XD / d / tr / EN)]


def main():
    assert not OUT.exists() and not OOFN.exists(), "outputs exist"
    K = pd.read_csv(T3 / "release/sub90c/local_eval_kit/trials.csv.gz", keep_default_na=False)
    O = {m: np.load(XD / f) for m, (_, f, _) in MEM.items()}
    for m, (d, _, nm) in MEM.items(): assert json.load(open(XD / d / "config.json"))["model"] == nm and json.loads(str(O[m]["oof_cfg"]))["model"] == nm
    ref = {tr: ids_of("xj1ib_10", tr) for tr in TRS}; cos_ib = {tr: np.load(XD / f"xj1ib_10/cos_{tr}_English.npy") for tr in TRS}
    OUT.mkdir(); fo, const, rep = {}, {}, {}
    for tr in TRS:
        y = K[K.protocol.eq(tr) & K.set.eq("CV")].y.to_numpy(); z_oof, z_ev, raw = {}, {}, {}
        for m, (d, _, nm) in MEM.items():
            o = [O[m][f"f{k}_{tr}"].astype(float) for k in range(5)]; oc = np.concatenate(o); assert len(oc) == len(y)
            mu, sd = float(oc.mean()), float(oc.std()); assert sd > 0; const[f"{tr}/{m}"] = dict(mu=mu, sd=sd)
            c = np.load(XD / d / f"cos_{tr}_English.npy").astype(float); assert ids_of(d, tr) == ref[tr] and len(c) == len(ref[tr]) and np.isfinite(c).all()
            own = np.array([float(l.split()[1]) for l in open(XD / d / tr / EN)]); assert np.allclose(-own, c, atol=2e-6), f"{d} cos != its file"
            z_oof[m] = [(x - mu) / sd for x in o]; z_ev[m] = (c - mu) / sd; raw[m] = c
        wb = WB.get(tr)   # 17/11 sub119b: optional per-protocol train-OOF weight (env AB_W "no_gender=.4,gender=.25"); absent = equal z-sum (unchanged)
        cA, cB = (1.0, 1.0) if wb is None else (1.0 - wb, wb)
        s = cA * z_ev["A"] + cB * z_ev["B"]; assert np.isfinite(s).all()
        for k in range(5): fo[f"f{k}_{tr}"] = cA * z_oof["A"][k] + cB * z_oof["B"][k]
        np.save(OUT / f"cos_{tr}_English.npy", s); (OUT / tr).mkdir()
        (OUT / tr / EN).write_text("".join(f"{i} {-v:.8f}\n" for i, v in zip(ref[tr], s)))
        r = lambda a, b: round(float(spearmanr(a, b)[0]), 4)
        e = lambda sc: round(X.eer(y, np.concatenate(sc) if isinstance(sc, list) else sc), 2)
        rep[tr] = dict(eval_spearman=dict(A_vs_B=r(raw["A"], raw["B"]), AB_vs_XJ1IB=r(s, cos_ib[tr]), A_vs_XJ1IB=r(raw["A"], cos_ib[tr]), B_vs_XJ1IB=r(raw["B"], cos_ib[tr])),
                       oof_eer_descriptive=dict(A=e(z_oof["A"]), B=e(z_oof["B"]), AB=e([fo[f"f{k}_{tr}"] for k in range(5)]),
                                                XJ1IB=e([np.load(XD / "xj1ib_oof_x1ib_10.npz")[f"f{k}_{tr}"] for k in range(5)])))
        print(tr, json.dumps(rep[tr]), flush=True)
    np.savez(OOFN, **fo, oof_cfg=json.dumps(dict(model=NAME, members={m: v[2] for m, v in MEM.items()}, nseed=10)))
    json.dump(dict(model=NAME, members={m: dict(dir=v[0], oof=v[1], name=v[2]) for m, v in MEM.items()}, fusion=("z_A + z_B, equal a-priori weights" if not WB else f"(1-w) z_A + w z_B, w per protocol {WB} (train-OOF nested median)"),
                   constants_from="each member's FLAG-train OOF on the kit CV trials", constants=const, report=rep), open(OUT / "config.json", "w"), indent=1)
    print("wrote", OUT, OOFN)


if __name__ == "__main__":
    main()
