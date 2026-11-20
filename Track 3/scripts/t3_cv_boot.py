"""t3_cv_boot.py (2026-10-20, review): identity-cluster bootstrap of fused-CV deltas (review point 3): per fold resample the 14 held identities with replacement;
trial weight = speaker weight x multiplicity (positives m_id, negatives m_voice*m_face); paired (same resample for both systems)."""
import sys, io, contextlib, pickle, numpy as np
ARGS = sys.argv[:]; sys.argv = ["x", "vox2vb"]
with contextlib.redirect_stdout(io.StringIO()):
    import t3_age_vox2 as AV
import t3lib as L
from track2.core import equal_error_rate, speaker_weights
W = dict(AV.W0, clip_cat4=.4, farl_cat4=.4)
def system(tags, w, va=None, extra={}):
    src = dict(extra); va = va or AV.VA["current"]
    for t in tags: src.update(AV.Fz.load([f"{t}_other"]))
    tot = sum(w.values())
    def score(f, p, fr):
        s = sum(w[k] * src[k][f, p] for k in w) / tot * 1.6 - .9 * np.abs(AV.Fz.fa[fr.face_row] - va(f)[fr.voice_row])
        if p == "standard":
            x, y = AV.Fz.mv[fr.face_row], AV.Fz.pv[fr.voice_row]; s = s + .05 * np.log(np.clip(x * y + (1 - x) * (1 - y), 1e-4, 1))
        return s
    return score
swap = {("clip_rdn" if k == "clip_ecapa" else "farl_rdn" if k == "farl_ecapa" else k): v for k, v in W.items()}
def spec(s):  # "<tag>[+lora][@<voice age of t3_age_g2.VAGE>]": +lora = phase-1 per-fold CLIP LoRA OOF (t3_fuse_retune.py 13-branch analog)
    s, _, age = s.partition("@"); tag, _, plus = s.partition("+"); lora = {}
    if plus == "lora":
        for p in sorted((L.T3 / "kaggle/outft").glob("**/ft_f*/oof_fold*_proj.pkl")):
            for b, d in pickle.load(open(p, "rb")).items(): lora.setdefault("lora_" + b.removeprefix("clip_"), {}).update(d)
    w = dict(W, **{"lora_" + b.removeprefix("clip_"): W[b] for b in W if b.startswith("clip_") and lora})
    return system([tag], w, __import__("t3_age_g2").VAGE[age] if age else None, lora)
sys.argv = ARGS
if len(sys.argv) > 2:  # t3_cv_boot.py <base spec> <spec> [spec ...]: same 8 branches and weights, recipe changes only
    SYS = {f"base {sys.argv[1]}": spec(sys.argv[1]), **{t: spec(t) for t in sys.argv[2:]}}
else:
    SYS = {"base kgdrop2": system(["kgdrop2"], W), "kavs2 (+avs2 scale-up)": system(["kavs2"], W),
           "knoavs (no AVSpeech)": system(["knoavs"], W), "ecapa->rdn fair (sub54c analog)": system(["kgdrop2", "lrdnfull"], swap)}
cells = {(f, p): L.cell(f, p)[0] for f in range(4) for p in ("standard", "same_gender")}
S = {n: {k: fn(*k, fr) for k, fr in cells.items()} for n, fn in SYS.items()}
def fused(n, mult=None, folds=False):
    e = {}
    for (f, p), fr in cells.items():
        w = speaker_weights(fr.voice_identity.to_numpy())
        if mult is not None: w = w * mult[(f, p)]
        e[(f, p)] = 100 * equal_error_rate(fr.label.to_numpy(), S[n][(f, p)], w)
    m = np.array([(e[(f, "standard")] + e[(f, "same_gender")]) / 2 for f in range(4)])
    return m if folds else m.mean()
rng = np.random.default_rng(2027); B = 1000; names = list(SYS)
point = {n: fused(n) for n in names}; boot = {n: [] for n in names}
for b in range(B):
    mult = {}
    for f in range(4):
        ids = sorted(set(cells[(f, "standard")].voice_identity) | set(cells[(f, "standard")].face_identity))
        m = dict(zip(ids, np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))))
        for p in ("standard", "same_gender"):
            fr = cells[(f, p)]; mv = fr.voice_identity.map(m).to_numpy(); mf = fr.face_identity.map(m).to_numpy()
            mult[(f, p)] = np.where(fr.label.to_numpy() == 1, mv, mv * mf).astype(float)
    for n in names: boot[n].append(fused(n, mult))
print(f"held identities per fold: {[len(set(cells[(f, 'standard')].voice_identity)) for f in range(4)]}")
for n in names[1:]:
    d = np.array(boot[n]) - np.array(boot[names[0]])
    wins = int((fused(n, folds=True) < fused(names[0], folds=True) - 1e-9).sum())
    print(f"{n:34s} point d {point[n] - point[names[0]]:+.2f}  boot SE {d.std():.2f}  95% CI [{np.percentile(d, 2.5):+.2f}, {np.percentile(d, 97.5):+.2f}]  P(worse) {np.mean(d > 0):.2f}  better folds {wins}/4")
print(f"base fused {point[names[0]]:.2f}, boot SE of the level {np.std(boot[names[0]]):.2f}")
