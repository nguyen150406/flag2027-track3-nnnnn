"""17/11: hbibv.zip (Kaggle ncamtu06/flag-t3-hbibv-extract: frozen ImageBind + merged IB-LoRA audio of the HB pool) -> caches/ext/{vox2,vb}_voice_{ibv,ibl}.{float32,ok}.npy
aligned row-by-row with rows_{vox2,vb}.csv via clip_index (voice caches are per row, like *_voice_simam). Checks: lora sha, synthetic-wav parity (both models), coverage,
and cos(IB-LoRA voice, frozen IB voice) on the pool vs the same statistic on the Mac FLAG-train files (consistency of the Kaggle merge with the Mac one).
usage: python hbibv_ingest.py"""
import json, zipfile, io
from pathlib import Path
import numpy as np, pandas as pd
XD = Path(__file__).resolve().parent; T3 = XD.parent; CX = T3 / "caches/ext"; Z = zipfile.ZipFile(XD / "hbibv_dl/hbibv.zip")
prov = json.loads(Z.read("PROVENANCE.json")); assert prov["lora_sha256"].startswith("60d816e4"); print("parity", prov["parity"]); assert all(v["min"] > .999 for v in prov["parity"].values())
L = pd.read_csv(io.BytesIO(Z.read("hbibv_voices.csv"))); E = {k: np.load(io.BytesIO(Z.read(f"{'ib' if k == 'ibv' else 'ibl'}_voice.f16.npy"))).astype(np.float32) for k in ("ibv", "ibl")}
assert all(len(e) == len(L) and e.shape[1] == 1024 and np.isfinite(e).all() for e in E.values())
u = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
for src in ("vox2", "vb"):
    r = pd.read_csv(CX / f"rows_{src}.csv", keep_default_na=False); Ls = L[L.src.eq(src)]; pos = dict(zip(Ls.clip_index.astype(int), Ls.index)); gd = dict(zip(Ls.clip_index.astype(int), Ls.good))
    idx = np.array([pos.get(int(c), -1) for c in r.clip_index]); ok = np.array([i >= 0 and bool(gd.get(int(c), False)) for i, c in zip(idx, r.clip_index)])
    for k, e in E.items():
        out = np.zeros((len(r), 1024), np.float32); out[ok] = e[idx[ok]]; np.save(CX / f"{src}_voice_{k}.float32.npy", out); np.save(CX / f"{src}_voice_{k}.ok.npy", ok)
    c = (u(E["ibl"][idx[ok]]) * u(E["ibv"][idx[ok]])).sum(1)
    print(f"{src}: rows {len(r)} covered {int(ok.sum())} | cos(IB-LoRA voice, frozen IB voice) pct 1/50/99 {np.percentile(c, [1, 50, 99]).round(3).tolist()}")
f = (u(np.load(T3 / "outputs/ultra/ibl_flag_voice.npy").astype(float)) * u(np.load(T3 / "outputs/ultra/ib_own_flag_voice.npy").astype(float))).sum(1)
print("FLAG train (Mac) cos(IB-LoRA voice, frozen IB voice) pct 1/50/99", np.percentile(f, [1, 50, 99]).round(3).tolist())
