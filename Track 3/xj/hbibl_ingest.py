"""17/11: hbibl.zip (Kaggle trnlhvy/flag-t3-hbibl-extract, IB-LoRA merged face embeddings of the HB pool) -> caches/ext/{vox2,vb}_face_ibl.{float32,ok}.npy
aligned with rows_{vox2,vb}.csv (vox2 member = 'vox2/' + face; vb member = f'img/{clip_index}_{face suffix}'). Checks: lora sha, frozen parity, coverage,
and that the IB-LoRA face differs from the frozen ImageBind face (cos stats vs caches/ext/*_face_ibv) as expected for a LoRA-adapted encoder.
usage: python hbibl_ingest.py"""
import json, zipfile, io
from pathlib import Path
import numpy as np, pandas as pd
XD = Path(__file__).resolve().parent; T3 = XD.parent; CX = T3 / "caches/ext"; Z = zipfile.ZipFile(XD / "hbibl_dl/hbibl.zip")
prov = json.loads(Z.read("PROVENANCE.json")); assert prov["lora_sha256"].startswith("60d816e4"), prov["lora_sha256"]
assert all(v["min"] > .99 for v in prov["frozen_parity"].values()), prov["frozen_parity"]; print("frozen parity", prov["frozen_parity"])
names = pd.read_csv(io.BytesIO(Z.read("hbibl_face_names.csv"))); E = np.load(io.BytesIO(Z.read("hbibl_face.f16.npy"))).astype(np.float32)
assert len(E) == len(names) and E.shape[1] == 1024 and np.isfinite(E).all(); pos = {n: i for i, n in enumerate(names.name)}; good = names.good.to_numpy(bool)
u = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
for src in ("vox2", "vb"):
    r = pd.read_csv(CX / f"rows_{src}.csv", keep_default_na=False)
    key = ("vox2/" + r.face) if src == "vox2" else ("img/" + r.clip_index.astype(str) + "_" + r.face.str.rsplit("_", n=1).str[1])
    idx = np.array([pos.get(k, -1) for k in key]); ok = (idx >= 0) & good[np.maximum(idx, 0)]
    out = np.zeros((len(r), 1024), np.float32); out[ok] = E[idx[ok]]
    np.save(CX / f"{src}_face_ibl.float32.npy", out); np.save(CX / f"{src}_face_ibl.ok.npy", ok)
    ib = np.load(CX / f"{src}_face_ibv.float32.npy"); okb = np.load(CX / f"{src}_face_ibv.ok.npy") & ok
    c = (u(out[okb]) * u(ib[okb])).sum(1)
    print(f"{src}: rows {len(r)} covered {int(ok.sum())} | cos(IB-LoRA, frozen IB) pct 1/50/99 {np.percentile(c, [1, 50, 99]).round(3).tolist()}")
