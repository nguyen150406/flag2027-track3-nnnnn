"""Fused CV: SUB05 (linear projector) vs alternative projector OOFs, and per-branch averages of both.
Usage: python t3_fuse_proj.py <tag prefix, e.g. lr128>   (reads outputs/oof_j_<prefix>_{sim,oa,ob}.pkl)"""
import sys, pickle, numpy as np
from t3lib import *
import t3_n_age2 as A
mv = np.load(T3 / "caches/mivolo_face/face_pmale.npy")[:6485]
g24 = np.load(T3 / "caches/flag_voiceage24.npz")["gender"][:6485]; pv = g24[:, 1] / (g24[:, 0] + g24[:, 1])
fa = A.FACES["mean"]
W0 = dict(farl_sim=.4, clip_sim=.4, clip_ecapa=.2, clip_sam100ft=.2, farl_ecapa=.2, clip_r293=.2)


def load(tags):
    o = {}
    for t in tags:
        p = T3 / f"outputs/oof_j_{t}.pkl"
        if p.exists(): o.update(pickle.load(open(p, "rb")))
    return o


def side(f, p, fr, wa=.9, wg=.05):
    s = -wa * np.abs(fa[fr.face_row] - A.vage("ridge+a24", "mean", f)[fr.voice_row])
    if p == "standard":
        x, y = mv[fr.face_row], pv[fr.voice_row]; s = s + wg * np.log(np.clip(x * y + (1 - x) * (1 - y), 1e-4, 1))
    return s


def run(name, src, W=W0):
    tot = sum(W.values())
    r = evaluate(lambda f, p, fr, v: sum(W[k] * src[k][f, p] for k in W) / tot * 1.6 + side(f, p, fr), verbose=False)
    print(f"{name:34s} std {r['standard']:.2f} sg {r['same_gender']:.2f} mean {r['mean']:.2f}  {r['folds']}", flush=True)
    return r


s5 = load(["vox2a_sim", "vox2a_oa", "vox2a_ob"])
pre = sys.argv[1]
alt = load([f"{pre}_sim", f"{pre}_oa", f"{pre}_ob", f"{pre}_other"])
print("branches:", sorted(alt))
run("SUB07 (vox2a)", s5)
if set(alt) >= set(W0):
    run(f"{pre} all branches", alt)
# scale-matched: rescale each alt branch so its OOF score SD equals SUB05's (keeps the age/gender weight's relative strength)
def sd(o, k): return np.concatenate([o[k][f, "standard"] for f in range(4)]).std()
if set(alt) >= set(W0):
    alt_m = {k: {key: alt[k][key] * sd(s5, k) / sd(alt, k) for key in alt[k]} for k in W0}
    print("score SD ratio alt/SUB05:", {k: round(sd(alt, k) / sd(s5, k), 3) for k in W0})
    run(f"{pre} all, SD-matched", alt_m)
mix = {k: s5[k] for k in W0}
for k in alt: mix[k] = alt[k]
run(f"SUB05 with {pre} swapped in ({len(alt)})", mix)
avg = {}
for k in W0:
    if k in alt:
        avg[k] = {key: .5 * (s5[k][key] + alt[k][key]) for key in s5[k]}
    else:
        avg[k] = s5[k]
run(f"avg(linear, {pre}) per branch", avg)
for k in alt:
    one = dict(s5); one[k] = alt[k]
    run(f"  swap only {k}", one)
