"""VoxCeleb1 rows_vox face images (vox1_faces.zip = trncnguyn/flag-t3-vox1faces output: vox/<id>/<video>/<frame>.jpg) -> BTC VGGFace fc7
(btc_feat.VGGFace, starter-kit load_img), CPU, inference only (14/11, XJ-U prereg sub99b). Output rows = rows_vox.csv order (duplicate faces share
one encoding). Parity: the images the teammate also encoded (ext_feats/faces.npy, voxceleb1 rows, same relative path) must give cos > .98.
Run in ~/.venvs/xjsb (its Pillow decodes like the BTC CSV).   usage: python vox1_faces_btc.py"""
import sys, io, zipfile, time, numpy as np, pandas as pd, torch
from pathlib import Path
H = Path(__file__).resolve().parent; T3 = H.parent; sys.path.insert(0, str(H)); import btc_feat as B
torch.set_num_threads(6)
z = zipfile.ZipFile(H / "vox1_faces_dl/vox1_faces.zip"); have = set(z.namelist())
r = pd.read_csv(T3 / "caches/ext/rows_vox.csv"); uf = r.face.drop_duplicates().tolist(); names = ["vox/" + f for f in uf]
ok = np.array([n in have for n in names]); print(f"unique faces {len(uf)}, in zip {ok.sum()}", flush=True)
m = B.VGGFace.from_h5(H / "models/vgg_face_weights.h5").eval()
F = np.zeros((len(uf), 4096), np.float32); idx = np.flatnonzero(ok); t0 = time.time()
with torch.no_grad():
    for s in range(0, len(idx), 64):
        ii = idx[s:s + 64]; F[ii] = m(torch.from_numpy(np.stack([B.load_img(io.BytesIO(z.read(names[i]))) for i in ii]))).numpy()
        if s % 1920 == 0: print(f"{s}/{len(idx)} {time.time() - t0:.0f}s", flush=True)
assert np.isfinite(F).all()
# parity vs the teammate's encodings of the same images
fm = pd.read_csv(H / "ext_feats/faces_meta.csv"); fv = fm[fm.source.eq("voxceleb1")]; key = fv.path.str.split("/").str[-3:].str.join("/")
pos = dict(zip(key, fv.index)); j = [i for i, f in enumerate(uf) if f in pos and ok[i]]
E = np.load(H / "ext_feats/faces.npy", mmap_mode="r"); ref = np.asarray(E[[pos[uf[i]] for i in j]], np.float32)
a, b = np.sqrt(np.maximum(F[j], 0)), np.sqrt(np.maximum(ref, 0))   # XJ preprocessing (prep_face) before the cosine
c = (a * b).sum(1) / np.linalg.norm(a, axis=1).clip(1e-12) / np.linalg.norm(b, axis=1).clip(1e-12)
print(f"PARITY vs teammate ext_feats on {len(j)} shared images: mean {c.mean():.5f} min {c.min():.5f} p05 {np.percentile(c, 5):.5f}", flush=True)
assert c.mean() > .98, "BTC VGGFace does not reproduce the teammate's vox1 encodings"
ix = r.face.map(dict(zip(uf, range(len(uf))))).to_numpy()
(H / "vox1_btc").mkdir(exist_ok=True); np.save(H / "vox1_btc/vox_face_btc.float32.npy", F[ix]); np.save(H / "vox1_btc/vox_face_btc.ok.npy", ok[ix])
print(f"done {ok.sum()} faces in {time.time() - t0:.0f}s -> vox1_btc/vox_face_btc (rows_vox order, {len(ix)} rows)")
