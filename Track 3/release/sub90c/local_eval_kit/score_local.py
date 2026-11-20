"""Score ANY system on the team's local evaluation and compare it with sub90c on the same trials (paired person bootstrap).
Needs only numpy, pandas, scikit-learn. No access to our features or models.
  python score_local.py <your_scores.csv> [--sets CV,HARD,RES,MAV,MV,EN,XL] [--boot 1000]
your_scores.csv: columns trial_id,score (higher = same person). Score any subset of trial_ids from trials.csv.gz; sets you did not score
are skipped, sets you scored partially are compared on your trials only (and flagged).
How to produce the scores (otherwise the numbers are not comparable):
  CV   out-of-fold: for fold k train on the people of the other 4 folds only (folds.json), score fold k. Reserve people (RES) must not be in training.
  RES  train on the people of all 5 folds (not the 10 reserve people), score RES.
  MAV / MV / EN / XL  your final model; these people must NOT be in your training data (lists in heldout_people.csv)."""
import sys, argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_curve

HERE = Path(__file__).resolve().parent


def eer(y, s):   # identical to Track 3/scripts/t3_ultra_eval.eer
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr; i = np.nanargmin(np.abs(fnr - fpr)); return 100 * float(fpr[i] + fnr[i]) / 2


def boot(y, s1, s0, ids, n=1000, seed=0):
    """person bootstrap of EER(s1) - EER(s0): resample voice-side people with replacement; returns mean, SE, P(s1 better)"""
    rng = np.random.default_rng(seed); U = np.unique(ids); idx = {u: np.flatnonzero(ids == u) for u in U}; d = []
    for _ in range(n):
        ix = np.concatenate([idx[u] for u in rng.choice(U, len(U))])
        if y[ix].min() != y[ix].max(): d.append(eer(y[ix], s1[ix]) - eer(y[ix], s0[ix]))
    d = np.array(d); return d.mean(), d.std(), np.mean(d < 0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("scores"); ap.add_argument("--sets", default="CV,HARD,RES,MAV,MV,EN,XL"); ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    K = pd.read_csv(HERE / "trials.csv.gz", keep_default_na=False); S = pd.read_csv(a.scores)
    assert {"trial_id", "score"} <= set(S.columns), "need columns trial_id,score"; assert S.trial_id.is_unique, "duplicate trial_id"
    unk = set(S.trial_id) - set(K.trial_id); assert not unk, f"{len(unk)} unknown trial_ids, e.g. {sorted(unk)[:3]}"
    assert np.isfinite(S.score.to_numpy(float)).all(), "non-finite scores"
    D = K.merge(S, on="trial_id", how="inner")
    sel = {"CV": lambda d: d.set.eq("CV"), "HARD": lambda d: d.set.eq("CV") & d.hard.eq(1), "RES": lambda d: d.set.eq("RES"), "MAV": lambda d: d.set.eq("MAV"),
           "MV": lambda d: d.set.eq("MAV") & d.mav_split.eq("MV"), "EN": lambda d: d.set.eq("EN"), "XL": lambda d: d.set.eq("XL")}
    print(f"{'protocol':10s} {'set':5s} {'trials':>13s} {'yours':>6s} {'sub90c':>6s} {'delta':>6s} {'boot SE':>7s} {'P(better)':>9s}")
    for pr in ("no_gender", "gender"):
        for name in a.sets.split(","):
            full = K[K.protocol.eq(pr) & sel[name](K)]; d = D[D.protocol.eq(pr) & sel[name](D)]
            if len(d) == 0 or d.y.min() == d.y.max(): continue
            y, s1, s0 = d.y.to_numpy(), d.score.to_numpy(float), d.sub90c.to_numpy(float); e1, e0 = eer(y, s1), eer(y, s0)
            m, se, p = boot(y, s1, s0, d.voice_person.to_numpy(), a.boot)
            part = "" if len(d) == len(full) else f"  (PARTIAL {len(d)}/{len(full)})"
            print(f"{pr:10s} {name:5s} {len(d):6d}/{len(full):<6d} {e1:6.2f} {e0:6.2f} {e1 - e0:+6.2f} {se:7.2f} {p:9.2f}{part}")
    print("delta < 0 = your system better than sub90c. EN / XL sub90c = partial system (see README_LOCAL_EVAL.md). Local deltas did NOT predict\n"
          "eval English deltas between our own systems (LEV-cal 13/11): treat them as bug checks and large-effect signals, not as eval forecasts.")


if __name__ == "__main__":
    main()
