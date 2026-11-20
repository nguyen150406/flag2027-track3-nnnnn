"""Strict Bengali exclusion list (user 2026-11-02/03: "đảm bảo ... clean"): the union of
  - the current list (caches/ext/exclude_bengali.csv: VoxCeleb1 India nationality 26, VoxCeleb2 VoxLingua107 rule 45)
  - VoxCeleb2 speakers with any of OUR clips at P(bn) >= .05 (kaggle/outlid/lid_vox2.csv)
  - AVSpeech videos flagged by our Whisper rule in ANY download chunk (fixes the per-chunk bug: 15 flagged videos were kept)
VoxBlink's 51 officially Bengali-labelled speakers never entered rows_vb.csv. Use: T3_EXCLUDE=<this csv> (t3data.excluded).
  python t3_strict_exclude.py"""
import sys, json
import pandas as pd
from t3lib import T3

EXT = T3 / "caches/ext"
rows = [pd.read_csv(EXT / "exclude_bengali.csv")]
lid = pd.read_csv(T3 / "kaggle/outlid/lid_vox2.csv")
rows.append(pd.DataFrame(dict(src="vox2", identity=sorted(set(lid.identity[lid.p_bn >= .05])), reason="our VoxLingua107 LID: a clip with P(bn) >= .05")))
avs = pd.concat([pd.read_csv(p, usecols=["youtube_id", "bengali"]) for p in
                 (T3 / "caches/avs/pilot/lid.csv", T3 / "kaggle/out_avs2dl/avs2a/lid.csv", T3 / "kaggle/out_avs2dl/avs2b/lid.csv")])
bad = sorted("avs_" + v for v in set(avs.youtube_id[avs.bengali.astype(bool)]))
for s in ("avs", "avs2"): rows.append(pd.DataFrame(dict(src=s, identity=bad, reason="Whisper LID flag in any download chunk")))
e = pd.concat(rows).drop_duplicates(["src", "identity"])  # keeps the first (strongest documented) reason
e.to_csv(EXT / "exclude_bengali_strict.csv", index=False)
for s in ("vox", "vox2", "vb", "avs", "avs2"):
    ids = set(pd.read_csv(EXT / f"rows_{s}.csv", usecols=["identity"]).identity.astype(str))
    x = set(e.identity[e.src.eq(s)]); old = set(rows[0].identity[rows[0].src.eq(s)])
    print(f"{s:5s} listed {len(x):4d} (old {len(old)}), present in rows_{s}: {len(x & ids):4d} (old {len(old & ids)}) of {len(ids)} identities")
print("->", EXT / "exclude_bengali_strict.csv")
