"""17/11 evening: hbpe.zip (Kaggle ncamtu06/flag-t3-hbpe-extract, frozen PE-AV audio embeddings of the HB pool) -> caches/ext/{vox2,vb}_voice_pe.{float32,ok}.npy
aligned row-by-row with rows_{vox2,vb}.csv via clip_index (as hbibv_ingest). Checks: model sha, synthetic parity, coverage, norm stats vs the Mac FLAG / eval PE voices.
usage: python hbpe_ingest.py"""
import json, zipfile, io
from pathlib import Path
import numpy as np, pandas as pd
XD = Path(__file__).resolve().parent; T3 = XD.parent; CX = T3 / "caches/ext"; Z = zipfile.ZipFile(XD / "hbpe_dl/hbpe.zip")
prov = json.loads(Z.read("PROVENANCE.json")); assert "c87718fb" in prov["model"]; print("parity", prov["parity"]); assert all(v["min"] > .999 for v in prov["parity"].values())
L = pd.read_csv(io.BytesIO(Z.read("hbpe_voices.csv"))); E = np.load(io.BytesIO(Z.read("pe_voice.f16.npy"))).astype(np.float32)
assert len(E) == len(L) and E.shape[1] == 1024 and np.isfinite(E).all()
nf = lambda a: np.percentile(np.linalg.norm(a, axis=1), [1, 50, 99]).round(3).tolist()
for src in ("vox2", "vb"):
    r = pd.read_csv(CX / f"rows_{src}.csv", keep_default_na=False); Ls = L[L.src.eq(src)]; pos = dict(zip(Ls.clip_index.astype(int), Ls.index)); gd = dict(zip(Ls.clip_index.astype(int), Ls.good))
    idx = np.array([pos.get(int(c), -1) for c in r.clip_index]); ok = np.array([i >= 0 and bool(gd.get(int(c), False)) for i, c in zip(idx, r.clip_index)])
    out = np.zeros((len(r), 1024), np.float32); out[ok] = E[idx[ok]]; np.save(CX / f"{src}_voice_pe.float32.npy", out); np.save(CX / f"{src}_voice_pe.ok.npy", ok)
    print(f"{src}: rows {len(r)} covered {int(ok.sum())} | norm pct 1/50/99 {nf(out[ok])}")
print("FLAG (Mac) PE voice norm pct", nf(np.load(T3 / "outputs/ultra/pe_flag_voice.npy")), "| eval", nf(np.load(T3 / "outputs/eval_final/ultra/pe_voice.npy")))
# 17/11 19:30 sanity vs the FLAG-train mean direction (PE blocks are centred by it): pool / FLAG / eval rows should look alike
_mu = np.load(T3 / "outputs/ultra/pe_flag_voice.npy").astype(np.float64).mean(0); _cm = lambda a: float(np.median((a / np.linalg.norm(a, axis=1, keepdims=True)) @ (_mu / np.linalg.norm(_mu))))
print("median cos to FLAG-train mean: FLAG", round(_cm(np.load(T3 / "outputs/ultra/pe_flag_voice.npy")), 4), "| eval", round(_cm(np.load(T3 / "outputs/eval_final/ultra/pe_voice.npy")), 4),
      "| pool", {s: round(_cm(np.load(CX / f"{s}_voice_pe.float32.npy")[np.load(CX / f"{s}_voice_pe.ok.npy")]), 4) for s in ("vox2", "vb")})
