"""ULTRA (2026-11-11): install a finished no-FLAG CLIP LoRA kernel output (kaggle/build_t3_ultra_lora.py) into the caches.
  python t3_ultra_install_lora.py <download dir> <tag clipnf|clipnf2>
<download dir> = `kaggle kernels output` of <user>/flag-t3-ultra-<job> (untrusted: only .pt / .npy / the faces zip are read, as data).
  <tag>.pt                         -> caches/ultra_lora/<tag>.pt            (eval faces: t3_eval_extract.py group <tag>)
  flag_face_<tag>.float32.npy      -> caches/<tag>/face_embeddings.float32.npy
  <tag>_faces.zip: {vox,vox2,vb}_face_<tag>.{float32,ok}.npy -> caches/ext/
Checks: FLAG rows = those of the clipft12 cache (11,349, same row order), 768-d, finite, unit norm; ext rows match the clip caches."""
import sys, zipfile, shutil
from pathlib import Path
import numpy as np

T3 = Path(__file__).resolve().parents[1]; C = T3 / "caches"
src, tag = Path(sys.argv[1]), sys.argv[2]; assert tag in ("clipnf", "clipnf2"), tag


def check(a, ref, name):
    a = np.asarray(a); assert a.shape == ref.shape, (name, a.shape, ref.shape)
    ok = np.isfinite(a).all(1); n = np.linalg.norm(a[ok], axis=1); assert ok.mean() > .99 and np.allclose(n, 1, atol=1e-3), (name, ok.mean(), n.min(), n.max())
    print(f"  ok {name} {a.shape}")


pt = next(src.rglob(f"{tag}.pt")); ff = next(src.rglob(f"flag_face_{tag}.float32.npy")); zf = next(src.rglob(f"{tag}_faces.zip"))
sys.path.insert(0, str(T3 / "scripts")); import t3data as D
ref = D.face("clipft12"); check(np.load(ff), ref, ff.name)
(C / "ultra_lora").mkdir(exist_ok=True); shutil.copy(pt, C / "ultra_lora" / f"{tag}.pt")
(C / tag).mkdir(exist_ok=True); shutil.copy(ff, C / tag / "face_embeddings.float32.npy")
with zipfile.ZipFile(zf) as z:
    want = {f"{s}_face_{tag}.{e}.npy" for s in ("vox", "vox2", "vb") for e in ("float32", "ok")}; names = set(z.namelist())
    assert want <= names, want - names
    for n in sorted(want): (C / "ext" / n).write_bytes(z.read(n))
    for n in sorted(want):
        out = C / "ext" / n
        if n.endswith(".float32.npy"):
            s = n.split("_face_")[0]; a = np.load(out, mmap_mode="r"); r = np.load(C / "ext" / f"{s}_face_clip.float32.npy", mmap_mode="r")
            assert a.shape[0] == r.shape[0] and a.shape[1] == 768, (n, a.shape, r.shape)
            okm = np.load(C / "ext" / n.replace(".float32.", ".ok.")); assert np.isfinite(a[okm]).all(), n
            print(f"  ok {n} {a.shape} ok-rows {okm.mean():.3f}")
print("installed", tag)
