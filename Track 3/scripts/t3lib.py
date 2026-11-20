"""Track 3 research harness: canonical identity-disjoint CV on FLAG English train.

Uses the Track 2 canonical 4-fold trial lists (14 held IDs/fold, balanced
positives/negatives, standard + same_gender protocols) and Track 2 OOF branch
scores as the XP19-style baseline. Only FLAG train rows are read.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
T2 = ROOT / "Track 2"
T3 = ROOT / "Track 3"
sys.path[:0] = [str(T2 / "src"), str(T2 / "scripts")]
from track2.core import equal_error_rate, speaker_weights  # noqa: E402

SPLIT = json.loads((ROOT / "Track 1/data_manifests/speaker_splits.json").read_text())
GENDER = SPLIT["gender_by_speaker"]
GROUPS = [g["speakers"] for g in SPLIT["groups"]]
META = pd.read_csv(ROOT / "flag_raw_s01_v3_adaface_simam100/metadata.csv")
TRAIN = META[META.partition.eq("train")].sort_values("manifest_order").reset_index(drop=True)
assert (TRAIN.row_id.to_numpy() == np.arange(len(TRAIN))).all()
SPK = TRAIN.identity.astype(str).to_numpy()
BRANCHES = ("farl_sim", "ada_sim", "farl_wav", "ada_wav", "clip_sim")
XP19_W = dict(farl_sim=.3, ada_sim=.1, farl_wav=.3, ada_wav=.1, clip_sim=.2)


def fold_ids(fold):
    """fit = all other 4 groups (incl. Group0), held = groups[fold+1]."""
    held = set(GROUPS[fold + 1])
    return sorted(set(SPK) - held), sorted(held)


_cells = {}
def cell(fold, protocol):
    """Canonical trial frame + Track 2 OOF branch scores (3-seed mean) + old age signal."""
    key = (fold, protocol)
    if key not in _cells:
        from tune_t2_xp09_fusion_train_only import canonical_cell
        _cells[key] = canonical_cell(fold, protocol)
    return _cells[key]


def eer(frame, score):
    y = frame.label.to_numpy()
    return 100 * equal_error_rate(y, score, speaker_weights(frame.voice_identity.to_numpy()))


def evaluate(score_fn, folds=range(4), verbose=False):
    """score_fn(fold, protocol, frame, values) -> scores. Returns per-protocol mean SB EER."""
    out = {p: [] for p in ("standard", "same_gender")}
    for f in folds:
        for p in out:
            fr, v = cell(f, p)
            out[p].append(eer(fr, score_fn(f, p, fr, v)))
    res = {p: float(np.mean(e)) for p, e in out.items()}
    res["mean"] = (res["standard"] + res["same_gender"]) / 2
    res["folds"] = {p: np.round(e, 2).tolist() for p, e in out.items()}
    if verbose:
        print(f"std {res['standard']:.2f}  sg {res['same_gender']:.2f}  mean {res['mean']:.2f}  {res['folds']}")
    return res


def branch_sum(v, w=XP19_W):
    return sum(w[b] * v[b] for b in BRANCHES)
