"""C1/C2 quick re-tune on existing OOF (no training): frozen 8 branches (SUB21a recipe OOF) + the phase-1 per-fold CLIP LoRA branches
(kaggle/outft, LoRA trained without the held-out fold's IDs -> the dev situation), fused like production (ID part total 1.6, age |d|,
gender term on the standard protocol). Questions: share of the LoRA group, share of the FaRL group, age weight, gender weight.
Pre-registered decision rule: change a weight only if CV mean improves >= 0.15 AND >= 3/4 folds agree AND it does not contradict
official evidence. Usage: python t3_fuse_retune.py [frozen_tag=gdrop]"""
import sys, io, contextlib, pickle, numpy as np
from pathlib import Path
TAG = sys.argv[1] if len(sys.argv) > 1 else "gdrop"
sys.argv = ["x", "vox2vb"]
with contextlib.redirect_stdout(io.StringIO()):
    import t3_age_vox2 as AV
from t3lib import evaluate, T3
Fz = AV.Fz
frozen = Fz.load([f"{TAG}_{s}" for s in ("sim", "oa", "ob", "other", "a", "b")] if TAG == "gdrop" else [f"{TAG}_other"])
W8 = dict(AV.W0, clip_cat4=.4, farl_cat4=.4); assert set(W8) <= set(frozen), sorted(frozen)
lora = {}
for p in sorted((T3 / "kaggle/outft").glob("**/ft_f*/oof_fold*_proj.pkl")):
    for b, d in pickle.load(open(p, "rb")).items(): lora.setdefault("lora_" + b.removeprefix("clip_"), {}).update(d)
assert all(len(d) == 8 for d in lora.values()), {k: len(d) for k, d in lora.items()}
WL = {"lora_" + b.removeprefix("clip_"): W8[b] for b in W8 if b.startswith("clip_")}  # same .4/.2 voice pattern as the frozen CLIP group
src = dict(frozen, **lora)
BASE = None

def run(name, w, age=.9, gender=.05):
    global BASE
    tot = sum(w.values())
    def score(f, p, fr, v):
        s = sum(w[k] * src[k][f, p] for k in w) / tot * 1.6 - age * np.abs(Fz.fa[fr.face_row] - AV.VA["current"](f)[fr.voice_row])
        if p == "standard" and gender:
            x, y = Fz.mv[fr.face_row], Fz.pv[fr.voice_row]; s = s + gender * np.log(np.clip(x * y + (1 - x) * (1 - y), 1e-4, 1))
        return s
    with contextlib.redirect_stdout(io.StringIO()):
        r = evaluate(score)
    fm = np.array([(a + b) / 2 for a, b in zip(r["folds"]["standard"], r["folds"]["same_gender"])])
    if BASE is None: BASE = (r["mean"], fm)
    d, wins = r["mean"] - BASE[0], int((fm < BASE[1] - 1e-9).sum())
    print(f"{name:44s} std {r['standard']:6.2f} sg {r['same_gender']:6.2f} mean {r['mean']:6.2f}  d {d:+.2f}  better folds {wins}/4  folds {np.round(fm - BASE[1], 2).tolist()}", flush=True)
    return r

def scaled(group, m, w):  # multiply every weight of one group by m (the total is renormalised to 1.6 inside run)
    return {k: v * (m if k.startswith(group) else 1) for k, v in w.items()}

W13 = dict(W8, **WL)
print(f"frozen OOF = {TAG}; LoRA OOF = phase-1 per-fold ({len(lora)} branches)")
print("== reference = deployed rule: frozen 8 + LoRA 5, raw .4/.2, ID total 1.6, age .9, gender .05")
run("REF  frozen8 + lora5 (current rule)", W13)
run("     frozen8 only", W8)
print("== C1a: share of the LoRA group (x m)")
for m in (.5, .75, 1.5, 2, 3): run(f"     lora x{m}", scaled("lora_", m, W13))
print("== C1b: share of the FaRL group (x m)")
for m in (0, .5, 1.5): run(f"     farl x{m}", scaled("farl_", m, W13))
print("== C1c: share of the frozen CLIP group (x m)")
for m in (.5, 1.5): run(f"     frozen clip x{m}", {k: v * (m if k.startswith("clip_") else 1) for k, v in W13.items()})
print("== C1d: voice pattern inside every group: equal instead of .4/.2")
run("     equal per branch", {k: 1.0 for k in W13})
print("== C2a: age weight (gender .05)")
for a in (.5, .7, 1.1, 1.3, 1.6): run(f"     age {a}", W13, age=a)
print("== C2b: gender weight (age .9)")
for g in (0, .1, .2): run(f"     gender {g}", W13, gender=g)
print("== C2 on the frozen-only system (does a stronger ID part move the optimum?)")
BASE = None; run("     frozen8 age .9", W8)
for a in (.7, 1.1): run(f"     frozen8 age {a}", W8, age=a)
