"""Merge the two VoxBlink encode outputs (team: farl/clip/simam/sam100ft, main: ecapa/r293) into external source "vb".
Rows are aligned on (voice, face) keys (the two prep runs are deterministic but fetch failures may differ).
Identities get a "vb_" prefix: VoxBlink ids (idNNNNN) collide with VoxCeleb2 ids. Usage: python merge_vb.py <dirA> [<dirB> ...]"""
import sys, numpy as np, pandas as pd
from pathlib import Path
EXT = Path(__file__).resolve().parents[1] / "caches/ext"
parts = []
for d in map(Path, sys.argv[1:]):
    r = pd.read_csv(d / "rows_vb.csv", dtype=str); r["key"] = r.voice + "|" + r.face
    arrs = {p.name[3:-len(".float32.npy")]: np.load(p) for p in d.glob("vb_*.float32.npy")}
    print(d, len(r), "rows,", sorted(arrs)); parts.append((r, arrs))
keys = set(parts[0][0].key)
for r, _ in parts[1:]: keys &= set(r.key)
base = parts[0][0][parts[0][0].key.isin(keys)].reset_index(drop=True)
print(f"aligned rows {len(base)} (of {[len(r) for r, _ in parts]}), identities {base.identity.nunique()}")
for r, arrs in parts:
    pos = pd.Series(np.arange(len(r)), index=r.key)[base.key].to_numpy()
    for tag, a in arrs.items():
        np.save(EXT / f"vb_{tag}.float32.npy", a[pos].astype(np.float32)); np.save(EXT / f"vb_{tag}.ok.npy", np.ones(len(pos), bool))
        print("  vb_" + tag, a[pos].shape)
out = base.drop(columns=["key"]).copy()
for c in ("identity",): out[c] = "vb_" + out[c]
out["voice"] = "vb_" + out.voice; out["face"] = "vb_" + out.face
out.to_csv(EXT / "rows_vb.csv", index=False)
print("wrote", EXT / "rows_vb.csv")
