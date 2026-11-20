"""IB-LoRA embeddings on the Mac (16/11, prereg PREREG_gate_all.txt 20:10 "PREREG sub118a"): inference only, LoRA merged into fp32 weights,
our extractor preprocessing (scripts/t3_imagebind_extract.py). CPU-vs-MPS parity on 8 items first (fail fast).
  FLAG train (META train rows sorted by row_id) -> outputs/ultra/ibl_flag_{face,voice}.npy
  eval (outputs/eval_final/rows.csv order)      -> outputs/eval_final/ultra/ibl_{face,voice}.npy
usage: python iblora_extract.py <lora.pt>"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
T3 = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(T3 / "scripts"))
import t3_iblora as IL, t3_imagebind_extract as T
from t3lib import META

ck = sys.argv[1]; dev, cpu = torch.device("mps"), torch.device("cpu")
tr = META[META.partition.eq("train")].sort_values("row_id"); assert len(tr) == 6485; base = T3.parent / "data/train_set/train_set"
fp, vp = [str(base / p) for p in tr.face_relpath], [str(base / p) for p in tr.voice_relpath]
m, C = IL.load_merged(ck, dev); IL.log(f"LoRA {ck}: step {C['step']} val {C['val']} cfg {C['cfg']}")
mc, _ = IL.load_merged(ck, cpu)
for kind, ps in (("face", fp[::811][:8]), ("voice", vp[::811][:8])):
    a, b = T.encode(mc, ps, kind, cpu), T.encode(m, ps, kind, dev); c = (a * b).sum(1); IL.log(f"PARITY cpu vs mps {kind}: min cos {c.min():.6f}")
    assert c.min() > .999, f"MPS parity failed for {kind}"
del mc
# review 16/11: the Mac (merged fp32) must reproduce Kaggle's own embeddings of the chosen checkpoint (parametrised fp16) on val items we have locally
d = Path(ck).parent; it = pd.read_csv(d / "items_rel.csv", keep_default_na=False); loc = T3.parent / "external_mavceleb/v1_complete"
for kind in ("face", "voice"):
    v = it[it.split.eq("val") & it.kind.eq(kind)].reset_index(drop=True); K_ = np.load(d / f"val_{kind}_best.npy"); assert len(K_) == len(v), (kind, len(K_), len(v))
    j = [i for i, (s_, r) in enumerate(zip(v.src, v.rel)) if s_ == "v1" and (loc / r).exists()]
    if len(j) < 4: IL.log(f"KAGGLE-MAC parity {kind}: only {len(j)} local val items -> skipped"); continue
    e = T.encode(m, [str(loc / v.rel[i]) for i in j], kind, dev); c = (e * (K_[j] / np.linalg.norm(K_[j], axis=1, keepdims=True))).sum(1)
    IL.log(f"KAGGLE-MAC parity {kind}: n {len(j)} mean cos {c.mean():.5f} min {c.min():.5f}"); assert c.min() > .99, f"Kaggle-vs-Mac parity failed for {kind}"
o = T3 / "outputs/ultra"
for kind, ps in (("face", fp), ("voice", vp)):
    E = T.encode(m, ps, kind, dev).astype(np.float32); assert E.shape == (6485, 1024) and np.isfinite(E).all(); np.save(o / f"ibl_flag_{kind}.npy", E); IL.log(f"FLAG {kind} {E.shape}")
t = pd.read_csv(T3 / "outputs/eval_final/rows.csv"); oe = T3 / "outputs/eval_final/ultra"
for kind in ("face", "voice"):
    u = t[kind].drop_duplicates().tolist(); E = T.encode(m, u, kind, dev).astype(np.float32); pos = {p: i for i, p in enumerate(u)}
    Er = E[[pos[p] for p in t[kind]]]; assert Er.shape == (len(t), 1024) and np.isfinite(Er).all(); np.save(oe / f"ibl_{kind}.npy", Er); IL.log(f"eval {kind} unique {len(u)} -> rows {Er.shape}")
IL.log("EXTRACT DONE")
