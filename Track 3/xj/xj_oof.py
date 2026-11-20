"""XJ scores on OUR local trial tables (RF folds, folds.json): fold-k models trained on the other 4 folds (no reserve ids; ext twins of held-out
ids dropped), RES by models on all 5 folds. nseed models averaged. Saves xj_oof_<tag>.npz with keys f{k}_{pr}, res_{pr}.
usage: python xj_oof.py <tag> [key=val: epochs ratio mask langs nseed bn bnclip mavx res(0/1)]"""
import sys, json, numpy as np, torch, xj_lib as X, xj_ext as XE
from pathlib import Path
torch.set_num_threads(4)
T3 = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(T3 / "scripts"))
import t3_ultra_eval as E
tag, kv = sys.argv[1], dict(a.split("=") for a in sys.argv[2:])
ep, ratio, mask, nseed = int(kv.get("epochs", 3)), int(kv.get("ratio", 4)), kv.get("mask", "0") == "1", int(kv.get("nseed", 5))
langs = kv["langs"].split(",") if "langs" in kv else None
nf = kv.get("nf", "0") == "1"   # 14/11: no FLAG pairs in training (xj_lib.train_nf); stored in oof_cfg only when true
J = json.load(open(T3 / "release/sub90c/local_eval_kit/folds.json")); T = E.load_trials()
L, F, V = X.load_flag(); Fp, Vp = X.prep_face(F), X.prep_voice(V); out = {}
# 14/11 (direction 2): same external-data options as final_xj.py (defaults = the old unfiltered call, so xj_oof_base is unchanged)
Ex = XE.Ext(langs=langs, bengali_filter=kv.get("bn", "0") == "1", bengali_clip=kv.get("bnclip", "0") == "1", mavx=kv.get("mavx", "0") == "1")
for k in list(range(5)) + (["res"] if kv.get("res", "1") == "1" else []):
    held = set(J["reserve"]) | (set(J["folds"][k]) if k != "res" else set())
    tr = ~L.id.isin(held).to_numpy(); drop = Ex.drop_for(held)
    ms = [(X.train_nf if nf else X.train)(Fp[tr], Vp[tr], seed=1000 * j + (k if k != "res" else 9), epochs=ep, ext=Ex.sampler(drop), ext_ratio=ratio, ids=L.id.values[tr], mask_same=mask) for j in range(nseed)]
    for pr in E.PR:
        key = f"f{k}_{pr}" if k != "res" else f"res_{pr}"; t = T[key]
        assert not set(L.id.values[t["vi"]]) & set(L.id.values[tr]), key   # held-out people only
        out[key] = np.mean([X.score(m, Fp[t["fi"]], Vp[t["vi"]]) for m in ms], 0)
    print("done", k, flush=True)
# review 14/11: store the recipe so fuse_c can check it against the XJ model dir it is paired with
oof_cfg = dict(epochs=ep, ratio=ratio, mask=mask, nseed=nseed, langs=kv.get("langs"), bn=kv.get("bn", "0") == "1", bnclip=kv.get("bnclip", "0") == "1", mavx=kv.get("mavx", "0") == "1")
if nf: oof_cfg["nf"] = True
np.savez(T3 / f"xj/xj_oof_{tag}.npz", **out, oof_cfg=json.dumps(oof_cfg)); print("saved", tag, oof_cfg)
