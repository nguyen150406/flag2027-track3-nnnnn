"""Fused 6-branch CV of OOF sets (outputs/oof_j_<tag>_*.pkl) under the SUB09/10 fusion, with the current (v2) and the
VoxCeleb2-transfer (v3) voice age. Usage: python t3_cmp.py <tag> [<tag> ...]"""
import sys, io, contextlib, numpy as np
tags = sys.argv[1:]; sys.argv = ["x", "vox2vb"]
with contextlib.redirect_stdout(io.StringIO()):
    import t3_age_vox2 as AV
from t3lib import evaluate
VA = {"v2": AV.VA["current"], "v3": AV.VA["vox2ridge_clip_a100+cur"]}
for tag in tags:
    src = AV.Fz.load([f"{tag}_sim", f"{tag}_oa", f"{tag}_ob", f"{tag}_other", f"{tag}_a", f"{tag}_b"])
    if set(src) < set(AV.W0): print(tag, "missing branches", sorted(set(AV.W0) - set(src))); continue
    for age, va in VA.items():
        tot = sum(AV.W0.values())
        def score(f, p, fr, v):
            s = sum(AV.W0[k] * src[k][f, p] for k in AV.W0) / tot * 1.6 - .9 * np.abs(AV.Fz.fa[fr.face_row] - va(f)[fr.voice_row])
            if p == "standard":
                x, y = AV.Fz.mv[fr.face_row], AV.Fz.pv[fr.voice_row]; s = s + .05 * np.log(np.clip(x * y + (1 - x) * (1 - y), 1e-4, 1))
            return s
        r = evaluate(score, verbose=False)
        print(f"{tag:16s} age {age}  std {r['standard']:.2f} sg {r['same_gender']:.2f} mean {r['mean']:.2f}  {r['folds']}", flush=True)
