"""Phase A merge: the 3-model LID audit (t3_lid_audit.py outputs of the lid kernels) -> caches/ext/exclude_bengali_v3.csv.
Rule (revised 07/11, user choice): flag_m = P(bn)+P(as) of model m > 0.5 in any window; a clip is flagged when ANY model flags it.
  The first rule calibrated tau_m on FLAG train as an English negative control, but FLAG train's 'English' folder holds real
  Bengali speech (~11% of clips P > .5 under all 3 models, 24/70 ids), so tau went to 1.0 and nothing was flagged. FLAG train
  stats are still printed as a diagnostic only.
  an unreadable or never-audited clip counts as flagged (it cannot be verified)
  one flagged clip excludes the whole identity of that source; the strict list stays excluded; the speaker-level Bengali flags
  of lid_lists/speaker_bn_flags.json (bn_vl VoxLingua107 / bn_wl Whisper-large-v3, run on other copies of the same speakers' clips) are added as well.
Usage: python t3_lid_merge.py <dir with lid_*.csv and clips_*.csv> [more dirs]   -> exclude_bengali_v3.csv + outputs/lid_audit_report.*"""
import sys, json
import numpy as np, pandas as pd
from t3lib import T3

EXT = T3 / "caches/ext"; M = ("vl", "wh", "mms")
lid = pd.concat([pd.read_csv(p) for d in sys.argv[1:] for p in sorted(__import__("pathlib").Path(d).glob("lid_*of*.csv"))]).drop_duplicates("key", keep="last")
clips = pd.concat([pd.read_csv(p) for d in sys.argv[1:] for p in sorted(__import__("pathlib").Path(d).glob("clips_*.csv"))]).drop_duplicates("key")
D = clips.merge(lid.drop(columns="src"), on="key", how="left"); D["audited"] = D.dur.notna() & D.dur.ge(0)
has_bn = lambda s: s.fillna("").astype(str).str.split("|").map(lambda t: bool({"bn", "as"} & set(t)))

ctl = D[D.src.eq("flag") & D.audited]; assert len(ctl) > 6000, f"negative control too small: {len(ctl)}"
cal = {}
for m in M:
    fp_top = float(has_bn(ctl[f"{m}_top"]).mean()); q = ctl[f"{m}_bnas"].quantile([.99, .999]).tolist()
    cal[m] = dict(tau=.5, use_top=False, flag_top1_rate=fp_top, q99=q[0], q999=q[1], max=float(ctl[f"{m}_bnas"].max()))
print("negative control (FLAG train, English):", json.dumps(cal, indent=1))

X = D[D.src.ne("flag")].copy(); why = pd.Series("", index=X.index)
NEED = {"vox", "vox2", "vb", "lrs3", "avs", "avs2"}  # every external source a training config reads; a missing part would pass silently
assert NEED <= set(X.src), f"sources missing from the audit (unzip both lid_vox.zip and lid_web.zip): {sorted(NEED - set(X.src))}"
for m in M:
    c = cal[m]; hit = X[f"{m}_bnas"].gt(c["tau"]) | (has_bn(X[f"{m}_top"]) & c["use_top"])
    why[hit] = why[hit] + f"{m}(top " + X.loc[hit, f"{m}_top"].astype(str) + ", p " + X.loc[hit, f"{m}_bnas"].round(3).astype(str) + ") "
X["flag"] = why.ne("") | ~X.audited; X["why"] = np.where(X.audited, why.str.strip(), "audio not verified (unreadable / not fetched)")
old = pd.read_csv(EXT / "exclude_bengali_strict.csv"); rows = [old]; rep = []
for s in sorted(X.src.unique()):
    x = X[X.src.eq(s)]; ids = x.identity.astype(str); bad = x[x.flag].groupby("identity").why.first()
    r = pd.read_csv(EXT / f"rows_{s}.csv").drop_duplicates("voice"); r = r[~r.identity.astype(str).isin(set(old.identity[old.src.eq(s)].astype(str)))]
    missing = sorted(set(r.identity.astype(str)) - set(ids))  # identities with rows but no clip in the audit list
    rows.append(pd.DataFrame(dict(src=s, identity=bad.index.astype(str), reason="LID v3: " + bad.values)))
    rows.append(pd.DataFrame(dict(src=s, identity=missing, reason="LID v3: no clip audited")))
    rep.append(dict(src=s, clips=len(x), audited=int(x.audited.sum()), identities=int(ids.nunique()), flagged_ids=len(bad),
                    unverified_clips=int((~x.audited).sum()), missing_ids=len(missing),
                    flagged_ids_at_tau05=int(x[x[[f"{m}_bnas" for m in M]].max(1).gt(.05)].identity.nunique())))
# speaker-level Bengali flags (lid_lists/speaker_bn_flags.json)
part = json.load(open(T3 / "lid_lists/speaker_bn_flags.json"))
LAB = {"bn_vl": "VoxLingua107", "bn_wl": "Whisper-large-v3"}
for cat in ("bn_vl", "bn_wl"):
    for src, pre in (("vox2", ""), ("vb", "vb_")):
        rows.append(pd.DataFrame(dict(src=src, identity=[pre + i for i in part[cat][src]], reason=f"speaker-level LID flag ({LAB[cat]})")))
E = pd.concat(rows).drop_duplicates(["src", "identity"]); E.to_csv(EXT / "exclude_bengali_v3.csv", index=False)
R = pd.DataFrame(rep); print(R.to_string(index=False)); print(f"-> {EXT / 'exclude_bengali_v3.csv'}: {len(E)} rows (strict list {len(old)})")
O = T3 / "outputs"; R.to_csv(O / "lid_audit_report.csv", index=False); json.dump(cal, open(O / "lid_audit_calibration.json", "w"), indent=1)
X.assign(pmax=X[[f"{m}_bnas" for m in M]].max(1)).sort_values("pmax", ascending=False).head(60).to_csv(O / "lid_audit_top60.csv", index=False)
