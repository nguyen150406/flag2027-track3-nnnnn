"""Sanity checks on the MAVX Kaggle outputs (xj/mavx/): row/array alignment, LID coverage + flags, no Bengali, clips new vs the
teammate's set, ids subset of keep_ids, and feature sanity = cross-video same-identity AUC of the NEW clips against the identity's
EXISTING ext_feats embeddings (should match the existing pool's own cross-video AUC, ~.97-.99 for voice)."""
import numpy as np, pandas as pd, xj_lib as X, xj_ext as XE
from pathlib import Path
from sklearn.metrics import roc_auc_score
import sys; M = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "mavx"; KD = Path(__file__).resolve().parents[1] / "kaggle/xjx_mavxds"
keep = set(pd.read_csv(KD / "keep_ids.csv").key); exist = set(pd.read_csv(KD / "existing_rel.csv").rel)
fm, vm = pd.read_csv(XE.D / "faces_meta.csv"), pd.read_csv(XE.D / "voices_meta.csv"); fm["k"], vm["k"] = XE.key(fm), XE.key(vm)
Fa = np.load(XE.D / "faces.npy", mmap_mode="r"); Va = np.load(XE.D / "voices.npy", mmap_mode="r")
B = XE.bengali_keys(); allv, allf = [], []
for sh in (0, 1):
    v = pd.read_csv(M / f"mavx{sh}_voice_rows.csv"); Vx = np.load(M / f"mavx{sh}_voice.f32.npy"); assert len(v) == len(Vx) and (v.vrow.values == np.arange(len(v))).all()
    f = pd.read_csv(M / f"mavx{sh}_face_rows.csv"); Fx = np.load(M / f"mavx{sh}_face.f16.npy", mmap_mode="r"); assert len(f) == len(Fx)
    d = pd.read_csv(M / f"mavx{sh}_lid.csv"); lk = v.src.str.replace("mavceleb_", "") + ":" + v.rel.str.split("/voices/").str[1]
    assert set(lk) == set(d.key) and d.key.is_unique, "LID keys != selected clips"
    fl = (d[["vl_bnas", "wh_bnas", "mms_bnas"]] > .5).any(axis=1) | (d.dur <= 0)
    assert not v.lang.isin(["bengali", "bangla"]).any() and not v.rel.isin(exist).any() and not f.rel.isin(exist).any() and set(v.k) <= keep
    print(f"shard {sh}: ids {v.k.nunique()} clips {len(v)} voice ok {v.ok.sum()} faces {len(f)} ok {f.ok.sum()} | langs {v.lang.value_counts().to_dict()}")
    print(f"  LID flagged {int(fl.sum())} (all3 {int((d[['vl_bnas','wh_bnas','mms_bnas']] > .5).all(axis=1).sum())}), unreadable {int((d.dur <= 0).sum())}; flagged ids:",
          sorted(set(d.key[fl].str.extract(r'^(v[12]):(id\d{4})/').agg(':'.join, axis=1))))
    vn = np.linalg.norm(Vx[v.ok.values], axis=1); print(f"  voice norm mean {vn.mean():.1f} (min {vn.min():.1f}); dur>0 rows {(v.dur > 0).sum()}")
    allv.append((v, Vx)); allf.append((f, Fx))
dropped = {f"mavceleb_v1/{i}" for i in B["v1_bad"]} | {f"mavceleb_v2/{i}" for i in B["v2_bad"]} | {f"mavceleb_v1/{i}" for i in set(fm.id[fm.k.str.startswith("mavceleb_v1")]) if i not in B["v1_cov"]}
print("keep ids now dropped by the Bengali rule (incl. MAVX flags):", sorted(keep & dropped))
# feature sanity: new clip vs mean of the identity's existing embeddings (same id) vs other ids' means
for kind, data, A, meta, prep in (("voice", allv, Va, vm, X.prep_voice), ("face", allf, Fa, fm, X.prep_face)):
    ks = sorted({k for t, _ in data for k in t.k[t.ok]}); idx = {k: np.flatnonzero(meta.k.values == k) for k in ks}
    cen = np.stack([prep(np.asarray(A[idx[k]], np.float32)).mean(0) for k in ks]); cen /= np.linalg.norm(cen, axis=1, keepdims=True); kpos = {k: j for j, k in enumerate(ks)}
    E = np.concatenate([prep(np.asarray(Z[t.ok.values], np.float32)) for t, Z in data]); kk = np.concatenate([t.k[t.ok].values for t, _ in data])
    own = np.array([kpos[k] for k in kk]); S = E @ cen.T; pos = S[np.arange(len(E)), own]
    rng = np.random.default_rng(0); jn = rng.integers(len(ks), size=len(E)); ok = jn != own; neg = S[np.arange(len(E))[ok], jn[ok]]   # random other identity
    print(f"{kind}: new clips {len(E)} vs existing centroids of {len(ks)} ids: same-id AUC {roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]):.3f}")
