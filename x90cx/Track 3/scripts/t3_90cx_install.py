"""90c-X (15/11): install a finished MAV-augmented no-FLAG LoRA kernel output (kaggle/build_t3_lnfm.py: tag clipnfm / clipnfm2) into the
CLONED work root (~/FLAG_2027_90cx) UNDER THE OLD NAMES clipnf / clipnf2, so the documented sub90c chain
(README_SUB90C 4.3) runs unchanged on the new LoRAs. Refuses to run inside the original FLAG_2027 tree.
  python t3_90cx_install.py <download dir> clipnfm clipnf <mavsrc dir>     |  ... clipnfm2 clipnf2 <mavsrc dir>
15/11 option B: also installs the "mav" aux source for the heads (rows_mav.csv + cat4 voice parts from kaggle/mavsrc_<user>, padded with
zeros / ok False for the face rows; MAV faces of THIS LoRA from the kernel zip, ok = zip ok AND face row AND mavf_ok).
Same checks as t3_ultra_install_lora.py (FLAG rows 11,349 x 768, finite, unit norm; ext rows = clip caches) + a health print of the
cosine between the new and the old (sub90c) embeddings of the same FLAG rows."""
import sys, io, zipfile, shutil
from pathlib import Path
import numpy as np

T3 = Path(__file__).resolve().parents[1]; C = T3 / "caches"
assert "FLAG_2027_90cx" in str(T3), f"refusing to install into {T3} (only the 90c-X clone)"
src, stag, dtag, msrc = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
assert (stag, dtag) in (("clipnfm", "clipnf"), ("clipnfm2", "clipnf2")), (stag, dtag)
sys.path.insert(0, str(T3 / "scripts")); import t3data as D


def check(a, ref, name):
    a = np.asarray(a); assert a.shape == ref.shape, (name, a.shape, ref.shape)
    ok = np.isfinite(a).all(1); n = np.linalg.norm(a[ok], axis=1); assert ok.mean() > .99 and np.allclose(n, 1, atol=1e-3), (name, ok.mean(), n.min(), n.max())
    print(f"  ok {name} {a.shape}")


pt = next(src.rglob(f"{stag}.pt")); ff = next(src.rglob(f"flag_face_{stag}.float32.npy")); zf = next(src.rglob(f"{stag}_faces.zip"))
new = np.load(ff); check(new, D.face("clipft12"), ff.name)
old = np.load(C / dtag / "face_embeddings.float32.npy"); cs = (new * old).sum(1)
print(f"  health: cos(new {stag}, old {dtag}) on the same FLAG rows mean {cs.mean():.4f} min {cs.min():.4f} (1.0 would mean no change)")
assert cs.mean() < .9999, "new LoRA embeddings identical to the old ones?"
shutil.copy(pt, C / "ultra_lora" / f"{dtag}.pt"); shutil.copy(ff, C / dtag / "face_embeddings.float32.npy")
with zipfile.ZipFile(zf) as z:
    names = set(z.namelist())
    for s in ("vox", "vox2", "vb"):
        for e in ("float32", "ok"):
            n = f"{s}_face_{stag}.{e}.npy"; assert n in names, n
            (C / "ext" / f"{s}_face_{dtag}.{e}.npy").write_bytes(z.read(n))
        a = np.load(C / "ext" / f"{s}_face_{dtag}.float32.npy", mmap_mode="r"); r = np.load(C / "ext" / f"{s}_face_clip.float32.npy", mmap_mode="r")
        okm = np.load(C / "ext" / f"{s}_face_{dtag}.ok.npy"); assert a.shape[0] == r.shape[0] and a.shape[1] == 768 and np.isfinite(a[okm]).all(), (s, a.shape, r.shape)
        print(f"  ok {s}_face_{dtag} {a.shape} ok-rows {okm.mean():.3f}")
import pandas as pd
E = C / "ext"; rows = pd.read_csv(msrc / "rows_mav.csv"); vok = np.load(msrc / "mavv_ok.npy"); fok = np.load(msrc / "mavf_ok.npy"); nv, nf = len(vok), len(fok)
assert len(rows) == nv + nf and not rows.language.astype(str).str.lower().isin(["bengali", "bangla", "bn"]).any()
rows.to_csv(E / "rows_mav.csv", index=False)
for e in ("simam", "sam100ft", "r293", "ecapa"):
    v = np.load(msrc / f"mavv_{e}.float32.npy"); assert len(v) == nv and np.isfinite(v).all(), e
    np.save(E / f"mav_voice_{e}.float32.npy", np.vstack([v, np.zeros((nf, v.shape[1]), np.float32)])); np.save(E / f"mav_voice_{e}.ok.npy", np.concatenate([vok, np.zeros(nf, bool)]))
with zipfile.ZipFile(zf) as z:
    mf = np.load(io.BytesIO(z.read(f"mav_face_{stag}.float32.npy"))); mo = np.load(io.BytesIO(z.read(f"mav_face_{stag}.ok.npy")))
assert mf.shape == (nv + nf, 768) and len(mo) == nv + nf, (mf.shape, len(mo))
ok = mo & np.concatenate([np.zeros(nv, bool), fok]); assert ok[nv:].mean() > .9 and np.isfinite(mf[ok]).all() and np.allclose(np.linalg.norm(mf[ok], axis=1), 1, atol=1e-3)
np.save(E / f"mav_face_{dtag}.float32.npy", mf.astype(np.float32)); np.save(E / f"mav_face_{dtag}.ok.npy", ok)
a = D.ext_source("mav", dtag, "cat4")
print(f"  ok mav source for {dtag}: {len(set(a['labels']))} identities, {len(a['faces'])} faces, {len(a['voices'])} voices, voice languages "
      f"{pd.Series(a['voice_lang']).value_counts().to_dict()}")
print("installed", stag, "as", dtag, "in", T3)
