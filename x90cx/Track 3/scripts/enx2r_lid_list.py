"""ENX2-R (13/11): v2 training exclusion list = LID-flagged identities (smoke kernel audit) + identity-overlap screen (enx2r_overlap.py).
LID rule (prereg 18:05, LID v3): a clip is flagged when ANY model gives P(bn)+P(as) > .5 on ANY 30-s window; a clip that is missing from the
audit or has a non-finite score counts as flagged; one flagged clip excludes the whole identity.
usage: python enx2r_lid_list.py <dir with lid_*.csv>   -> outputs/enx2r/v2_lid_exclude.json {"identities", "lid", "overlap", ...}"""
import sys, json, glob
from pathlib import Path
import numpy as np, pandas as pd
T3 = Path(__file__).resolve().parent.parent; O = T3 / "outputs/enx2r"
v2 = pd.read_csv(T3 / "kaggle/enx2r_v2data/v2/v2_clips.csv"); v2["key"] = [f"v2|{i}" for i in range(len(v2))]
fs = sorted(glob.glob(str(Path(sys.argv[1]) / "lid_*.csv"))); assert fs, "no LID csv"
L = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates("key", keep="last")
cols = [c for c in L.columns if c.endswith("_bnas")]; assert sorted(cols) == ["mms_bnas", "vl_bnas", "wh_bnas"], cols
m = v2.merge(L, on="key", how="left")
S = m[cols].to_numpy(float); bad = ~np.isfinite(S).all(1) | (np.nan_to_num(S, nan=1.0) > .5).any(1)
print(f"clips {len(v2)} audited {m[cols[0]].notna().sum()} | flagged {bad.sum()} (missing/non-finite {(~np.isfinite(S).all(1)).sum()}) "
      f"| per model >.5: " + ", ".join(f"{c} {(m[c] > .5).sum()}" for c in cols))
print("top-1 labels (wh):", m.wh_top.str.split("|").explode().value_counts().head(8).to_dict())
lid_ids = sorted(m.identity[bad].astype(str).unique())
# ENX2R_DESIGN.md R1 (approved 17:40): "bất kỳ clip nào bị LID v3 gắn cờ Bengali hoặc Hindi" -> clip-level Hindi drop, same "any model, any window"
# logic as LID v3 (top-1 label of vl / wh / mms on any window == "hi"); restored 13/11 21:33 before any full-run training.
tops = [c for c in L.columns if c.endswith("_top")]; assert sorted(tops) == ["mms_top", "vl_top", "wh_top"], tops
hi = np.array([any("hi" in str(v).split("|") for v in row) for row in m[tops].itertuples(index=False)])
hindi_clips = [int(i) for i in np.flatnonzero(hi)]
ov = json.load(open(O / "v2_overlap.json"))["drop"]
ids = sorted(set(lid_ids) | set(ov))
json.dump(dict(identities=ids, lid=lid_ids, overlap=ov, hindi_clips=hindi_clips, n_v2_ids=int(v2.identity.nunique()),
               rule="identity out: any model P(bn)+P(as)>.5 any window, missing=flagged; clip out: any model top-1 hi any window",
               lid_files=[Path(f).name for f in fs]), open(O / "v2_lid_exclude.json", "w"), indent=1)
keep = v2[~v2.identity.astype(str).isin(ids) & ~hi]
print(f"LID ids {len(lid_ids)} {lid_ids} | overlap ids {len(ov)} | excluded {len(ids)}/{v2.identity.nunique()} | Hindi clips {hi.sum()} -> v2 train ids {keep.identity.nunique()}, clips {len(keep)}, min clips/id {keep.groupby('identity').size().min()}")
