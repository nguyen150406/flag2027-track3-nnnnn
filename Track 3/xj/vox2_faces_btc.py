"""VoxCeleb2 face frames (vox2_faces.zip from trncnguyn/flag-t3-vox2faces: vox2/<id>/<video>/<clip>_<k>.jpg, frames n=20/60 of the ENX2 manifest
clips) -> BTC VGGFace fc7 (btc_feat.VGGFace, starter-kit load_img). Inference only. Run in ~/.venvs/xjsb (its Pillow decodes like the BTC CSV).
usage: python vox2_faces_btc.py [check|run]"""
import sys, io, zipfile, time, numpy as np, pandas as pd, torch
from pathlib import Path
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H)); import btc_feat as B
torch.set_num_threads(8); z = zipfile.ZipFile(H / "vox2_faces_dl/vox2_faces.zip"); names = sorted(n for n in z.namelist() if n.endswith(".jpg"))
m = B.VGGFace.from_h5(H / "models/vgg_face_weights.h5")
def batch(ns): return torch.from_numpy(np.stack([B.load_img(io.BytesIO(z.read(n))) for n in ns]))
if sys.argv[1] == "check":
    x = batch(names[:16])
    with torch.no_grad(): a = m(x).numpy(); b = m.to("mps")(x.to("mps")).cpu().numpy()
    c = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1); print(f"{len(names)} frames; MPS vs CPU cos min {c.min():.7f}, max abs {np.abs(a-b).max():.3g}")
else:
    dev = sys.argv[2] if len(sys.argv) > 2 else "mps"; m = m.to(dev); F = np.zeros((len(names), 4096), np.float16); t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(names), 64):
            F[s:s + 64] = m(batch(names[s:s + 64]).to(dev)).cpu().numpy()
            if s % 6400 == 0: print(f"{s}/{len(names)} {time.time()-t0:.0f}s", flush=True)
    assert np.isfinite(F.astype(np.float32)).all()
    p = [n.split("/") for n in names]
    r = pd.DataFrame(dict(identity=[q[1] for q in p], video=[q[2] for q in p], clip=[q[3].rsplit("_", 1)[0] for q in p], k=[int(q[3].rsplit("_", 1)[1][:-4]) for q in p], ok=True))
    (H / "vox2_btc").mkdir(exist_ok=True); np.save(H / "vox2_btc/vox2_btc_face.f16.npy", F); r.to_csv(H / "vox2_btc/vox2_btc_face_rows.csv", index=False)
    print(f"done {len(names)} frames in {time.time()-t0:.0f}s; nonzero {(F[:1000] > 0).mean():.3f} norm {np.linalg.norm(F[:1000].astype(np.float32), axis=1).mean():.1f}")
