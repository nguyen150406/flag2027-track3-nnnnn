"""Install the Kaggle audeering embeddings (kaggle/build_t3_aud_kernel.py output, untrusted: only the .npz arrays are read, as data) as
per-row caches: caches/ext/<src>_voice_audk{24,6}.{float32,ok}.npy (new names: the Oct-15 vox2_voice_aud24 cache is kept) aligned with caches/ext/rows_<src>.csv (row -> clip_index).
Port check: FLAG-train-style sanity -> the 24-layer age of the same model on MAV/eval was extracted locally; here we check finite values,
emb dim 1024, age in [0, 1], gender probabilities sum to 1, and coverage per source.
  python t3_aud_install.py <download dir>"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
T3 = Path(__file__).resolve().parents[1]; EXT = T3 / "caches/ext"; src_dir = Path(sys.argv[1])
for s in ("vox2", "vb"):
    rows = pd.read_csv(EXT / f"rows_{s}.csv")
    for k in (24, 6):
        z = np.load(next(src_dir.rglob(f"{s}_a{k}.npz"))); ci, emb = z["clip_index"].astype(int), z["emb"].astype(np.float32)
        assert emb.shape[1] == 1024 and len(ci) == len(emb) and len(set(ci)) == len(ci), (s, k, emb.shape)
        fin = np.isfinite(emb).all(1); assert fin.mean() > .99, (s, k, fin.mean())
        assert ((z["age"] >= 0) & (z["age"] <= 1)).all() and np.allclose(z["gender"].sum(1), 1, atol=1e-3)
        pos = pd.Series(np.arange(len(ci)), index=ci); idx = pos.reindex(rows.clip_index.astype(int)).to_numpy()
        ok = ~np.isnan(idx); A = np.zeros((len(rows), 1024), np.float32); A[ok] = emb[idx[ok].astype(int)]; ok &= np.isfinite(A).all(1) & (np.abs(A).sum(1) > 0)
        np.save(EXT / f"{s}_voice_audk{k}.float32.npy", A); np.save(EXT / f"{s}_voice_audk{k}.ok.npy", ok)
        print(f"{s} a{k}: {len(ci)} clips -> {ok.sum()}/{len(rows)} rows ok ({ok.mean():.3f}); age mean {z['age'].mean():.3f}")
