"""verify_sub126c (FLAG 2027 Track 3 release, 18/11/2026): compare a rebuilt artefact with the shipped reference.
  python verify_sub126c.py dirs <ref dir> <new dir> [tol]      component scores: cos_<protocol>_<lang>.npy (+ submission txt files if present)
  python verify_sub126c.py subs <ref sub dir|zip> <new sub dir|zip>   the 4 submission files: IDENTICAL (bytes) / SAME RANKING / DIFFERENT
  python verify_sub126c.py oof <ref.npz> <new.npz> [tol]       FLAG-train out-of-fold scores (keys f<fold>_<protocol>)
Exit code 0 = identical (or within tol for 'dirs'), 1 = different."""
import sys, zipfile
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

FILES = [f"{t}/sub_score_v4_{l}_{g}.txt" for t in ("no_gender", "gender") for l, g in (("English", "heard"), ("Bangla", "unheard"))]


def read(src, name):
    src = Path(src)
    if src.suffix == ".zip":
        z = zipfile.ZipFile(src); return z.read(name) if name in z.namelist() else None
    p = src / name
    if not p.exists():
        zs = sorted(src.glob("*.zip"))
        if zs: return read(zs[0], name)
        return None
    return p.read_bytes()


def subs(ref, new):
    ok = True
    for f in FILES:
        a, b = read(ref, f), read(new, f)
        if a is None or b is None: print(f"{f}: MISSING ({'ref' if a is None else 'new'})"); ok = False; continue
        if a == b: print(f"{f}: IDENTICAL"); continue
        la, lb = a.decode().split("\n"), b.decode().split("\n"); la, lb = [l for l in la if l], [l for l in lb if l]
        ida, idb = [l.split()[0] for l in la], [l.split()[0] for l in lb]
        if ida != idb: print(f"{f}: DIFFERENT (ids / order)"); ok = False; continue
        sa, sb = np.array([float(l.split()[1]) for l in la]), np.array([float(l.split()[1]) for l in lb])
        r = spearmanr(sa, sb)[0]; d = float(np.abs(sa - sb).max())
        same = bool((np.argsort(sa, kind="stable") == np.argsort(sb, kind="stable")).all())
        print(f"{f}: {'SAME RANKING' if same else 'DIFFERENT'} (max |diff| {d:.3g}, Spearman {r:.6f})"); ok &= same
    print("RESULT:", "OK" if ok else "DIFFERENT"); return ok


def dirs(ref, new, tol=0.0):
    ok = True
    for t in ("no_gender", "gender"):
        for l in ("English", "Bangla"):
            pa, pb = Path(ref) / f"cos_{t}_{l}.npy", Path(new) / f"cos_{t}_{l}.npy"
            if not pa.exists(): continue
            if not pb.exists(): print(f"cos_{t}_{l}: MISSING in new"); ok = False; continue
            a, b = np.load(pa), np.load(pb); d = float(np.abs(a - b).max()) if a.shape == b.shape else float("inf")
            r = spearmanr(a, b)[0] if a.shape == b.shape else float("nan")
            st = "IDENTICAL" if d == 0 else ("WITHIN TOL" if d <= tol else "DIFFERENT"); print(f"cos_{t}_{l}: {st} (max |diff| {d:.3g}, Spearman {r:.6f})"); ok &= d <= tol
    print("RESULT:", "OK" if ok else "DIFFERENT"); return ok


def oof(ref, new, tol=0.0):
    A, B = np.load(ref), np.load(new); ks = sorted(k for k in A.files if k.startswith("f")); ok = True
    for k in ks:
        if k not in B.files: print(f"{k}: MISSING in new"); ok = False; continue
        a, b = A[k].astype(float), B[k].astype(float); d = float(np.abs(a - b).max()) if a.shape == b.shape else float("inf")
        if d > tol: print(f"{k}: DIFFERENT (max |diff| {d:.3g}, Spearman {spearmanr(a, b)[0]:.6f})"); ok = False
    print(f"OOF {len(ks)} arrays:", "IDENTICAL" if ok and tol == 0 else ("WITHIN TOL" if ok else "DIFFERENT")); print("RESULT:", "OK" if ok else "DIFFERENT"); return ok


if __name__ == "__main__":
    a = sys.argv[1:]; tol = float(a[3]) if len(a) > 3 else 0.0
    ok = subs(a[1], a[2]) if a[0] == "subs" else oof(a[1], a[2], tol) if a[0] == "oof" else dirs(a[1], a[2], tol)
    sys.exit(0 if ok else 1)
