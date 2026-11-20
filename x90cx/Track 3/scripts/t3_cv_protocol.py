"""FLAG-train cross-validation protocol of sub126c: person split and trial sampler.

The person split and trial sampler used for every FLAG-train cross-validation number of the development phase. It draws
the same random numbers in the same order as the original runs, so it reproduces `cv_run/w/split.json` and the FLAG folds of `outputs/ultra/trials.npz` exactly.

  load_v4(bundle, work)                          FLAG-train row metadata (identity, gender, video) from <bundle>/v4.npz
  split(V4)                                      70 identities -> 10 reserve + 5 gender-balanced folds of 12
  make_trials(V4, persons, gender, n, proto, rng)  n positive + n negative (face row, voice row) trials over `persons`
"""
from pathlib import Path
import numpy as np

SPLIT_SEED = 20261103     # seed of cv_run/w/split.json
PAIRS_PER_PERSON = 64     # candidate positive pairs drawn per identity before sampling the trials


ROW_KEYS = ("f_spk", "f_gender", "f_vid", "v_spk", "v_vid")   # the only columns the split and the sampler read


def load_v4(bundle, work=None):
    """Face and voice row metadata of the FLAG training set, in the row order that the trial lists index.
    `work` is accepted for call compatibility and not used."""
    with np.load(Path(bundle) / "v4.npz") as z:
        return {k: z[k] for k in ROW_KEYS}


def split(V4, seed=SPLIT_SEED, n_reserve=10, n_folds=5):
    """Person-disjoint split: per gender the sorted identities are shuffled, the first ones of each gender (in proportion)
    form the reserve, and the rest, ordered by gender, are dealt round-robin into the folds."""
    gender = {}
    for spk, g in zip(V4["f_spk"], V4["f_gender"]):
        gender[str(spk)] = int(g)
    rng = np.random.default_rng(seed)
    by_gender = []
    for g in (0, 1):
        ids = sorted(p for p, x in gender.items() if x == g)
        rng.shuffle(ids)
        by_gender.append(ids)
    if len(gender) >= 20:
        r0 = round(n_reserve * len(by_gender[0]) / len(gender))
        take = (r0, n_reserve - r0)
    else:
        take = (0, 0)
    reserve = by_gender[0][:take[0]] + by_gender[1][:take[1]]
    rest = sorted(by_gender[0][take[0]:] + by_gender[1][take[1]:], key=gender.get)
    folds = [rest[i::n_folds] for i in range(n_folds)]
    return dict(seed=seed, reserve=sorted(reserve), folds=[sorted(f) for f in folds], gender=gender)


def _candidate_pairs(f_ids, f_vids, v_ids, v_vids, k, rng):
    """For every identity that has both faces and voices (sorted by id): k random (face, voice) index pairs, restricted
    to pairs from two different videos when the identity has any such pair."""
    faces, voices = {}, {}
    for i, p in enumerate(f_ids):
        faces.setdefault(p, []).append(i)
    for i, p in enumerate(v_ids):
        voices.setdefault(p, []).append(i)
    names = sorted(set(faces) & set(voices))
    PF = np.zeros((len(names), k), dtype=np.int64)
    PV = np.zeros((len(names), k), dtype=np.int64)
    for j, p in enumerate(names):
        fi, vi = np.asarray(faces[p]), np.asarray(voices[p])
        allowed = f_vids[fi][:, None] != v_vids[vi][None, :]
        if not allowed.any():
            allowed[:] = True
        a, b = np.nonzero(allowed)
        pick = rng.integers(0, len(a), k)
        PF[j], PV[j] = fi[a[pick]], vi[b[pick]]
    return names, PF, PV


def make_trials(V4, persons, gender, n, proto, rng):
    """n positives (one of the identity's candidate pairs, identity and pair drawn uniformly) followed by n negatives
    (face of identity a, voice of another identity b; in the gender protocol b has a's gender). Returns row indices into
    V4's face and voice tables and the labels."""
    keep = set(map(str, persons))
    f_sel = np.nonzero([str(s) in keep for s in V4["f_spk"]])[0]
    v_sel = np.nonzero([str(s) in keep for s in V4["v_spk"]])[0]
    pair_rng = np.random.default_rng(int(rng.integers(1 << 30)))
    names, PF, PV = _candidate_pairs(V4["f_spk"][f_sel], V4["f_vid"][f_sel], V4["v_spk"][v_sel], V4["v_vid"][v_sel],
                                     PAIRS_PER_PERSON, pair_rng)
    PF, PV = f_sel[PF], v_sel[PV]
    who = rng.integers(0, len(names), n)
    col = rng.integers(0, PAIRS_PER_PERSON, n)
    face, voice = list(PF[who, col]), list(PV[who, col])
    face_rows = {p: np.nonzero(V4["f_spk"] == p)[0] for p in names}
    voice_rows = {p: np.nonzero(V4["v_spk"] == p)[0] for p in names}
    for _ in range(n):
        a = names[rng.integers(len(names))]
        others = [b for b in names if b != a]
        if proto == "gender":
            others = [b for b in others if gender[str(b)] == gender[str(a)]] or others
        b = others[rng.integers(len(others))]
        face.append(rng.choice(face_rows[a]))
        voice.append(rng.choice(voice_rows[b]))
    return np.array(face), np.array(voice), np.r_[np.ones(n, dtype=np.int64), np.zeros(n, dtype=np.int64)]
