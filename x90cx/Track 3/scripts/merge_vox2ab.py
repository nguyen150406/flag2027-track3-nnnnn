"""Merge VoxCeleb2 batch A (caches/ext/vox2_*) and batch B (Kaggle output, vox2b_*) into one external source "vox2ab"
(same identities -> one AAM head). Also copies batch B as source "vox2b". Usage: python merge_vox2ab.py <dir with vox2b_* files>"""
import sys, shutil, numpy as np, pandas as pd
from pathlib import Path
EXT = Path(__file__).resolve().parents[1] / "caches/ext"
B = Path(sys.argv[1])
ra, rb = pd.read_csv(EXT / "rows_vox2.csv"), pd.read_csv(B / "rows_vox2b.csv")
assert list(ra.columns) == list(rb.columns), (ra.columns, rb.columns)
overlap = set(zip(ra.identity, ra.video)) & set(zip(rb.identity, rb.video))
print(f"A rows {len(ra)} ids {ra.identity.nunique()} | B rows {len(rb)} ids {rb.identity.nunique()} | shared videos {len(overlap)}")
assert not overlap, "batch B must be video-disjoint from batch A"
for p in sorted(B.glob("vox2b_*")) + [B / "rows_vox2b.csv"]:
    shutil.copy2(p, EXT / p.name)
pd.concat([ra, rb], ignore_index=True).to_csv(EXT / "rows_vox2ab.csv", index=False)
for kind in ("face_clip", "face_farl", "voice_simam", "voice_sam100ft", "voice_ecapa", "voice_r293"):
    for suf in (".float32.npy", ".ok.npy"):
        a, b = np.load(EXT / f"vox2_{kind}{suf}"), np.load(B / f"vox2b_{kind}{suf}")
        assert len(a) == len(ra) and len(b) == len(rb), (kind, suf, a.shape, b.shape)
        np.save(EXT / f"vox2ab_{kind}{suf}", np.concatenate([a, b]))
    ok = np.load(EXT / f"vox2ab_{kind}.ok.npy")
    print(f"vox2ab_{kind}: {ok.sum()}/{len(ok)} ok")
if (B / "vox2b_face_vit.npy").exists():
    np.save(EXT / "vox2ab_face_vit.npy", np.concatenate([np.load(EXT / "vox2_face_vit.npy"), np.load(B / "vox2b_face_vit.npy")]))
print("merged ->", EXT)
