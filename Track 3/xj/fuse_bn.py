"""fuse_bn (17/11, prereg PREREG_gate_all.txt 11:30 "PREREG sub119d"): Bangla = (1-w) z(90c-X) + w z(HB) per protocol.
90c-X = sub90cx Bangla files (log-odds = -file value); HB = hb_10/cos_<tr>_Bangla.npy. z constants = each member's FLAG-train OOF on the kit CV trials
(xj/s90cx_oof.npz, xj/hb_oof_10.npz). w = median of the 5 fold-nested choices (grid .05) on the kit CV trials for this pair (train validation only).
Writes submissions/t3_<name> (Bangla = fused, English = <en_sub> bytes) -> then fuse_h.py stitches the final ticket.
usage: python fuse_bn.py <name> <en_sub>        e.g. python fuse_bn.py sub119h sub119b"""
import os, sys, json, zipfile, hashlib, tempfile
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
XD = Path(__file__).resolve().parent; T3 = XD.parent; SUB = T3 / "submissions"; sys.path.insert(0, str(XD))
import xj_lib as X
sha = lambda b: hashlib.sha256(b).hexdigest(); TRS = ("no_gender", "gender"); EN, BN = "sub_score_v4_English_heard.txt", "sub_score_v4_Bangla_unheard.txt"
W = np.round(np.arange(0, 1.01, .05), 2)
parse = lambda b: ([l.split()[0] for l in b.decode().splitlines()], np.array([float(l.split()[1]) for l in b.decode().splitlines()]))


def main(name, en_sub):
    out = SUB / f"t3_{name}"; assert not out.exists(), out
    K = pd.read_csv(T3 / "release/sub90c/local_eval_kit/trials.csv.gz", keep_default_na=False)
    O9, OH = np.load(XD / "s90cx_oof.npz"), np.load(XD / "hb_oof_10.npz")
    assert json.loads(str(O9["oof_cfg"]))["model"] == "90c-X" and json.loads(str(OH["oof_cfg"]))["model"] == "HB"
    z9x, zen = zipfile.ZipFile(SUB / "t3_sub90cx/sub90cx.zip"), zipfile.ZipFile(SUB / f"t3_{en_sub}/{en_sub}.zip")
    assert sha((SUB / "t3_sub90cx/sub90cx.zip").read_bytes()).startswith("d8652d4b")
    L, _, _ = X.load_flag(); bnm = dict(zip(L.voice, X.flag_bengali_mask(L, "any"))); data, rep = {}, {}
    for tr in TRS:
        Kt = K[K.protocol.eq(tr) & K.set.eq("CV")]; y = Kt.y.to_numpy(); assert (Kt.fold.to_numpy() == np.repeat(np.arange(5), 6000)).all()
        a = [O9[f"f{k}_{tr}"].astype(float) for k in range(5)]; h = [OH[f"f{k}_{tr}"].astype(float) for k in range(5)]; ys = [y[6000 * k: 6000 * (k + 1)] for k in range(5)]
        assert abs(X.eer(y, np.concatenate(a)) - {"no_gender": 13.03, "gender": 20.93}[tr]) < .015, "90c-X canary"
        ws, pred = [], []
        for k in range(5):
            tk = [j for j in range(5) if j != k]; at, ht, yt = (np.concatenate([q[j] for j in tk]) for q in (a, h, ys))
            za_, zh_ = (lambda q: (q - at.mean()) / at.std()), (lambda q: (q - ht.mean()) / ht.std())
            wb = min(W, key=lambda w: X.eer(yt, (1 - w) * za_(at) + w * zh_(ht))); ws.append(float(wb)); pred.append((1 - wb) * za_(a[k]) + wb * zh_(h[k]))
        w = float(np.median(ws)); ac, hc = np.concatenate(a), np.concatenate(h); m9, s9, mh, sh = ac.mean(), ac.std(), hc.mean(), hc.std()
        bn = Kt.voice_ref.str.replace("flag:", "", regex=False).map(bnm).to_numpy(bool); pn = np.concatenate(pred); fz = (1 - w) * (ac - m9) / s9 + w * (hc - mh) / sh
        rep[tr] = dict(w=w, ws=ws, oof_all=dict(x90=round(X.eer(y, ac), 2), hb=round(X.eer(y, hc), 2), nested_fused=round(X.eer(y, pn), 2)),
                       oof_bengali_lid_subset=dict(n=int(bn.sum()), x90=round(X.eer(y[bn], ac[bn]), 2), hb=round(X.eer(y[bn], hc[bn]), 2), fused=round(X.eer(y[bn], fz[bn]), 2)))
        ids, v9 = parse(z9x.read(f"{tr}/{BN}")); l9 = -v9
        hbids = [l.split()[0] for l in open(XD / "hb_10" / tr / BN)]; c = np.load(XD / f"hb_10/cos_{tr}_Bangla.npy"); assert hbids == ids and len(c) == len(ids)
        s = (1 - w) * (l9 - m9) / s9 + w * (c - mh) / sh; assert np.isfinite(s).all()
        rep[tr]["eval_spearman_bangla"] = dict(new_vs_90cx=round(float(spearmanr(s, l9)[0]), 4), hb_vs_90cx=round(float(spearmanr(c, l9)[0]), 4))
        data[f"{tr}/{BN}"] = "".join(f"{i} {-v:.8f}\n" for i, v in zip(ids, s)).encode(); data[f"{tr}/{EN}"] = zen.read(f"{tr}/{EN}")
        print(tr, json.dumps(rep[tr]), flush=True)
    tmp = Path(tempfile.mkdtemp(dir=SUB, prefix=f".t3_{name}_")); zp = tmp / f"{name}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for tr in TRS:
            for n in (f"{tr}/{EN}", f"{tr}/{BN}"): z.writestr(n, data[n])
    json.dump(dict(system=(f"Bangla (both protocols) = (1-w) z(90c-X: sub90cx Bangla log-odds) + w z(HB: hb_10 Bangla), w per protocol "
                           f"{ {tr: rep[tr]['w'] for tr in TRS} } = median of fold-nested choices on the FLAG-train kit CV OOF of this pair; z constants = each member's "
                           f"FLAG-train OOF. English (both protocols) = {en_sub} English files (bytes unchanged)."),
                   report=rep, files={n: dict(rows=len(b.decode().splitlines()), sha256=sha(b)) for n, b in data.items()},
                   code_sha256={"fuse_bn.py": sha((XD / "fuse_bn.py").read_bytes()), "hb.py": sha((XD / "hb.py").read_bytes())},
                   zip_sha256=sha(zp.read_bytes()), test_statistics_used=False, weights_tuned_on_eval=False), open(tmp / "provenance.json", "w"), indent=1)
    os.rename(tmp, out); print("wrote", out / f"{name}.zip", sha((out / f"{name}.zip").read_bytes())[:16])


if __name__ == "__main__":
    main(*sys.argv[1:])
