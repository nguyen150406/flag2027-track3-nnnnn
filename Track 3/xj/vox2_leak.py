"""leak check for iteration 2: VoxBlink ids vs FLAG train ids and vs the 24 V3 hold-out MAV v1 ids (mean-face cosine, BTC VGGFace prepared).
Same person in this space ~.9 (known v2 id0073 = FLAG id039: .908). Writes vox2_leak_drop.json = VB ids with max cos > .85."""
import json, numpy as np, pandas as pd, xj_lib as X, xj_ext as XE
L, F, _ = X.load_flag(); Fp = X.prep_face(F)
fr = pd.read_csv(XE.V2X / "vox2_btc_face_rows.csv"); Fv = X.prep_face(np.load(XE.V2X / "vox2_btc_face.f16.npy").astype(np.float32))
fr["vbid"] = fr.identity; ok = fr.ok.values
def means(Fx, keys):
    u = pd.unique(keys); M = np.stack([Fx[keys == k].mean(0) for k in u]); return list(u), M / np.linalg.norm(M, axis=1, keepdims=True)
vu, Mv = means(Fv[ok], fr.vbid.values[ok]); fu, Mf = means(Fp, L.id.values)
fm = pd.read_csv(XE.D / "faces_meta.csv"); fm["k"] = XE.key(fm); hold = ["mavceleb_v1/" + i for i in json.load(open("val_B0.json"))["hold"]]
Fa = X.prep_face(np.load(XE.D / "faces.npy", mmap_mode="r").astype(np.float32)); sel = fm.k.isin(hold).values; hu, Mh = means(Fa[sel], fm.k.values[sel])
S1, S2 = Mv @ Mf.T, Mv @ Mh.T; drop = {}
for name, S, u in (("FLAG", S1, fu), ("V3hold", S2, hu)):
    mx = S.max(1); print(f"VOX2 vs {name}: max-cos pct 50/99/99.9/max {np.percentile(mx, [50, 99, 99.9]).round(3)} {mx.max():.3f}; > .85: {(mx > .85).sum()}")
    for i in np.flatnonzero(mx > .85): drop[vu[i]] = f"{name} {u[int(S[i].argmax())]} {mx[i]:.3f}"
json.dump(drop, open("vox2_leak_drop.json", "w"), indent=1); print("drop", len(drop), list(drop.items())[:10])
