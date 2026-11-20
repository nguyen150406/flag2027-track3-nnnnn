"""17/11 evening: hbpef.zip (Kaggle trnlhvy/flag-t3-hbpef-extract, frozen PE-AV face embeddings of the HB pool) -> caches/ext/{vox2,vb}_face_pe.{float32,ok}.npy
aligned with rows_{vox2,vb}.csv (same member keys as hbibl_ingest). Checks: model sha, synthetic parity, coverage, norm/cos stats vs the Mac FLAG / eval PE faces.
usage: python hbpef_ingest.py"""
import json, zipfile, io
from pathlib import Path
import numpy as np, pandas as pd
XD = Path(__file__).resolve().parent; T3 = XD.parent; CX = T3 / "caches/ext"; Z = zipfile.ZipFile(XD / "hbpef_dl/hbpef.zip")
prov = json.loads(Z.read("PROVENANCE.json")); assert "c87718fb" in prov["model"]; print("parity", prov["parity"]); assert all(v["min"] > .999 for v in prov["parity"].values())
names = pd.read_csv(io.BytesIO(Z.read("hbpef_face_names.csv"))); E = np.load(io.BytesIO(Z.read("hbpef_face.f16.npy"))).astype(np.float32)
assert len(E) == len(names) and E.shape[1] == 1024 and np.isfinite(E).all(); pos = {n: i for i, n in enumerate(names.name)}; good = names.good.to_numpy(bool)
nf = lambda a: np.percentile(np.linalg.norm(a, axis=1), [1, 50, 99]).round(3).tolist()
for src in ("vox2", "vb"):
    r = pd.read_csv(CX / f"rows_{src}.csv", keep_default_na=False)
    key = ("vox2/" + r.face) if src == "vox2" else ("img/" + r.clip_index.astype(str) + "_" + r.face.str.rsplit("_", n=1).str[1])
    idx = np.array([pos.get(k, -1) for k in key]); ok = (idx >= 0) & good[np.maximum(idx, 0)]
    out = np.zeros((len(r), 1024), np.float32); out[ok] = E[idx[ok]]
    np.save(CX / f"{src}_face_pe.float32.npy", out); np.save(CX / f"{src}_face_pe.ok.npy", ok)
    print(f"{src}: rows {len(r)} covered {int(ok.sum())} | norm pct 1/50/99 {nf(out[ok])}")
print("FLAG (Mac) PE face norm pct", nf(np.load(T3 / "outputs/ultra/pe_flag_face.npy")), "| eval", nf(np.load(T3 / "outputs/eval_final/ultra/pe_face.npy")))
# 17/11 19:30 sanity vs the FLAG-train mean direction (PE blocks are centred by it): pool / FLAG / eval rows should look alike
_mu = np.load(T3 / "outputs/ultra/pe_flag_face.npy").astype(np.float64).mean(0); _cm = lambda a: float(np.median((a / np.linalg.norm(a, axis=1, keepdims=True)) @ (_mu / np.linalg.norm(_mu))))
print("median cos to FLAG-train mean: FLAG", round(_cm(np.load(T3 / "outputs/ultra/pe_flag_face.npy")), 4), "| eval", round(_cm(np.load(T3 / "outputs/eval_final/ultra/pe_face.npy")), 4),
      "| pool", {s: round(_cm(np.load(CX / f"{s}_face_pe.float32.npy")[np.load(CX / f"{s}_face_pe.ok.npy")]), 4) for s in ("vox2", "vb")})
