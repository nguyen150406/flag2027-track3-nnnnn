"""R2 eval build (2026-11-12): NEW = our side + R2 system, log-odds 1:1, with NO component from the RF code.
  our side  = LR (balanced) on [NF (mean of the 4 no-FLAG LoRA heads), farl_cat4, gender, age]
  R2 system = LR (balanced) on [R2 projector (t3_r2.py), own frozen ImageBind stream, gender, age]
Both LRs are fitted on all FLAG-train OOF trials of the RF-split folds (outputs/ultra), standardisation from train only; applied pair by
pair to the eval features (no eval statistic). Inputs: eval feat/ (our extractors), ultra/face_clipnf*.npy, ultra/ib_face.npy +
ultra/ib_voice.npy (t3_imagebind_extract.py), FLAG-train ImageBind caches (Track 2 XP65, same extractor).
  python t3_r2_build.py <out_name> <R2 arm E0|E1> [<eval dir> <rows.csv>]"""
import os, sys, json, zipfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
os.environ.setdefault("T3_EXCLUDE", str(HERE.parent / "caches/ext/exclude_ultra.csv"))
import numpy as np, pandas as pd, torch
import t3_ultra_eval as E, t3_build as TB, t3data as D, t3_ultra_heads as UH, t3_r2 as R2
from t3train import pair_cos
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

O = E.O; R2C = ["r2", "ibo", "gender", "age"]
D_ = os.environ.get("T3R2_OURS") == "D"   # sub90b (12/11, gate_90b.py PASS): our side uses the arm-D heads (FLAG main losses x .25 + cross-video positives)
NF = ["clipnfD_cat4", "clipnfD_sim", "clipnf2D_cat4", "clipnf2D_sim"] if D_ else ["clipnf_cat4", "clipnf_sim", "clipnf2_cat4", "clipnf2_sim"]
FARL = "farlD_cat4" if D_ else "farl_cat4"; OURS = ["NF", FARL, "gender", "age"]
U = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)


def train_cols(arm):
    A = dict(np.load(O / "attr.npz")); IBO = dict(np.load(O / "ib_own.npz")); H = {b: dict(np.load(O / f"head_{b}.npz")) for b in NF + [FARL, f"r2{arm}_clipnf"]}
    def col(n, key):
        if n in ("gender", "age"): return A[f"{key}__{n}"]
        if n == "NF": return np.mean([H[b][key] for b in NF], 0)
        if n == "ibo": return IBO[key]
        if n == "r2": return H[f"r2{arm}_clipnf"][key]
        return H[n][key]
    return col


def eval_cols(arm, ev, rows_csv):
    t = pd.read_csv(rows_csv); fd = ev / "feat"; col = {}
    pf = np.load(fd / "mivolo_pmale.npy"); g = np.load(fd / "a24_gender.npy"); pv = g[:, 1] / (g[:, 0] + g[:, 1])
    col["gender"] = np.log(np.clip(pf * pv + (1 - pf) * (1 - pv), 1e-4, 1))
    a24, a6 = np.load(E.T3 / "caches/flag_voiceage24.npz"), np.load(E.T3 / "caches/flag_voiceage6.npz")
    fa_tr = (TB.all_rows_face_vit()[:6485] + np.load(E.T3 / "caches/mivolo_face/face_age.npy")[:6485] / 100) / 2
    va = TB.voice_age_v2(fa_tr, TB.age_hidden(a24["emb"][:6485], a6["emb"][:6485]), a24["age"][:6485], TB.age_hidden(np.load(fd / "a24_emb.npy"), np.load(fd / "a6_emb.npy")), np.load(fd / "a24_age.npy"))
    col["age"] = -np.abs((np.load(fd / "vit_age.npy") + np.load(fd / "mivolo_age.npy") / 100) / 2 - va)
    FACE = {"farl": np.load(fd / "face_farl.npy"), "clipnf": np.load(ev / "ultra/face_clipnf.npy"), "clipnf2": np.load(ev / "ultra/face_clipnf2.npy")}
    voice = lambda vn: np.hstack([D._l2(np.load(fd / f"voice_{n}.npy")) for n in D.CAT[vn]]) if vn in D.CAT else np.load(fd / f"voice_{vn}.npy")
    r = np.arange(len(t)); hs = {}
    for b in NF + [FARL]:
        fn, vn, _, _ = UH.BR[b]; ms = [torch.load(p, weights_only=False) for p in sorted((O / "models").glob(f"{b}__final__s*.pt"))]; assert len(ms) == 3, (b, len(ms))
        hs[b] = np.mean([pair_cos(m, FACE[fn], voice(vn), r, r) for m in ms], 0)
    col["NF"] = np.mean([hs[b] for b in NF], 0); col[FARL] = hs[FARL]
    ibf, ibv = np.load(ev / "ultra/ib_face.npy"), np.load(ev / "ultra/ib_voice.npy"); assert len(ibf) == len(t) == len(ibv)
    trf, trv = np.load(O / "ib_own_flag_face.npy"), np.load(O / "ib_own_flag_voice.npy")      # FLAG train rows (XP65), centring constants
    col["ibo"] = (U(ibf - trf.mean(0)) * U(ibv - trv.mean(0))).sum(1)
    Fe = np.hstack([U(FACE["clipnf"]), U(ibf)]).astype(np.float32); Ve = voice("cat4").astype(np.float32)
    folds = ["final"] if arm == "E1" else ["0", "1", "2", "3", "4"]      # E0 never sees FLAG: all its models share the same training data
    ms = [m for f in folds for m in R2.load(arm, "clipnf", f, Fe.shape[1], Ve.shape[1])]
    col["r2"] = R2.cos(ms, Fe, Ve)
    return t, col, len(ms)


def fit_lr(cols, colf, pr):
    T = E.load_trials(); X = np.vstack([np.stack([colf(c, f"f{k}_{pr}") for c in cols], 1) for k in range(5)]); y = np.concatenate([T[f"f{k}_{pr}"]["y"] for k in range(5)])
    sc = StandardScaler().fit(X); m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(X), y); return sc, m


def main(name, arm, ev="../outputs/eval_final", rows_csv=None):
    ev = Path(ev); t, col, nr2 = eval_cols(arm, ev, rows_csv or ev / "rows.csv"); colf = train_cols(arm)
    out = E.T3 / "submissions" / f"t3_{name}"; assert not out.exists(), out; out.mkdir(parents=True); files, coefs = {}, {}
    for (proto, lang), n in TB.EXPECTED.items():
        msk = (t.protocol.eq(proto) & t.language.eq(lang)).to_numpy(); logit = np.zeros(msk.sum()); coefs[proto] = {}
        NOGG = os.environ.get("T3R2_NOGG") == "1" and proto == "gender"   # 12/11 review: in the gender protocol the gender column only measures estimator
        for nm, cols in (("ours", OURS), ("r2sys", R2C)):                  # disagreement (negative weight, hits uncertain English voices) -> dropped
            cols = [c for c in cols if not (NOGG and c == "gender")]
            sc, m = fit_lr(cols, colf, proto); M = np.stack([col[c][msk] for c in cols], 1); assert np.isfinite(M).all(), (nm, proto)
            logit += m.decision_function(sc.transform(M)); coefs[proto][nm] = dict(cols=cols, mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=m.coef_[0].tolist(), b=float(m.intercept_[0]))
        p = out / n; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("".join(f"{i} {-v:.8f}\n" for i, v in zip(t.pair_id.astype(str)[msk], logit)))
        files[n] = dict(rows=int(msk.sum()), sha256=TB.sha(p))
    zp = out / f"{name}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        for n in TB.EXPECTED.values(): zf.write(out / n, n)
    json.dump(dict(system="our side + R2 system (1:1 log-odds); no RF code or features", r2_training="CPU (12/11: MPS-trained R2 models diverged)", gender_col_in_gender_protocol=os.environ.get("T3R2_NOGG") != "1", our_side_heads=NF + [FARL], r2_arm=arm, r2_models=nr2, coefs=coefs, files=files,
                   zip_sha256=TB.sha(zp), pair_local=True, test_statistics_used=False), open(out / "provenance.json", "w"), indent=1)
    print("wrote", zp)


if __name__ == "__main__":
    main(*sys.argv[1:])
