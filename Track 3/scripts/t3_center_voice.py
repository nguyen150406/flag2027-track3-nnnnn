"""Standardised voice arrays for concatenation (user 2026-11-04 "làm một cat mà thêm whisper với wavlm"): raw Whisper vectors are nearly
constant (random-pair cos .995 on FLAG train, WavLM-SV .746, sam100ft .08), so an L2-normalised raw block would put ~all of its share of
a cat on a constant offset. Per-dimension mean/std are fitted ONLY on FLAG train rows + vox2 + vb (strict Bengali list dropped), saved, and
applied row by row: no dev/eval statistic. Writes caches/ext/{flag,vox2,vb}_voice_<out>.float32.npy (+ ok) and caches/ext/stats_<out>.npz.
  T3_EXCLUDE=<strict csv> python t3_center_voice.py whs:whc wsv:wsc"""
import sys, shutil
import numpy as np, pandas as pd
import t3data as D

E = D.EXT; assert "strict" in str(D.EXCLUDE), D.EXCLUDE
for spec in sys.argv[1:]:
    src, out = spec.split(":"); fl = np.load(E / f"flag_voice_{src}.float32.npy", mmap_mode="r"); sel = {}
    for s in ("vox2", "vb"):
        ok = np.load(E / f"{s}_voice_{src}.ok.npy"); ids = pd.read_csv(E / f"rows_{s}.csv", usecols=["identity"]).identity.astype(str).to_numpy()
        sel[s] = ok & ~np.isin(ids, D.excluded(s))
    n = 6485 + sum(int(k.sum()) for k in sel.values()); m = np.zeros(fl.shape[1]); q = np.zeros(fl.shape[1])
    parts = [np.asarray(fl[:6485], np.float64)] + [np.load(E / f"{s}_voice_{src}.float32.npy", mmap_mode="r")[k] for s, k in sel.items()]
    for x in parts: m += np.asarray(x, np.float64).sum(0); q += (np.asarray(x, np.float64) ** 2).sum(0)
    mu = m / n; sd = np.sqrt(np.maximum(q / n - mu ** 2, 0)) + 1e-6
    np.savez(E / f"stats_{out}.npz", mu=mu.astype(np.float32), sd=sd.astype(np.float32), rows=n, source=src)
    f = lambda x: ((np.asarray(x, np.float32) - mu.astype(np.float32)) / sd.astype(np.float32)).astype(np.float32)
    np.save(E / f"flag_voice_{out}.float32.npy", f(fl))
    for s in ("vox2", "vb"):
        a = np.load(E / f"{s}_voice_{src}.float32.npy", mmap_mode="r"); ok = np.load(E / f"{s}_voice_{src}.ok.npy")
        y = np.zeros(a.shape, np.float32); y[ok] = f(a[ok]); np.save(E / f"{s}_voice_{out}.float32.npy", y); shutil.copyfile(E / f"{s}_voice_{src}.ok.npy", E / f"{s}_voice_{out}.ok.npy")
    X = np.load(E / f"flag_voice_{out}.float32.npy", mmap_mode="r"); X = np.asarray(X[:6485]); X = X / np.linalg.norm(X, axis=1, keepdims=True)
    r = np.random.default_rng(0); a, b = r.integers(6485, size=20000), r.integers(6485, size=20000)
    print(f"{src}->{out}: fit on {n} rows, dim {fl.shape[1]}; FLAG train random-pair cos now {(X[a] * X[b]).sum(1).mean():.3f}", flush=True)
