"""XJ-family member = mean of the standardised scores of several XJ-type models (14/11): z_i = (score_i - mu_i) / sd_i with mu/sd of model i's
own FLAG-train OOF on the RF-split trials; family score = mean_i z_i (equal weights, fixed a priori). Writes a 'virtual model dir' (config.json,
cos_<tr>_<lang>.npy = family score, sub files = -score) and the family OOF npz (same combination on the OOF trials, recipe stored) so the
family can be standardised / complemented / fused exactly like a single member. No eval statistic is used.
usage: python make_family.py <out_dir> <oof_out.npz> <model_dir>:<oof.npz> [<model_dir>:<oof.npz> ...]"""
import sys, json, hashlib, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
XD = Path(__file__).resolve().parent; T3 = XD.parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
s = importlib.util.spec_from_file_location("fuse_c", XD / "fuse_c.py"); fc = importlib.util.module_from_spec(s); s.loader.exec_module(fc)
TRS = ("no_gender", "gender"); LANGS = (("English", "heard"), ("Bangla", "unheard"))


def check(cfg, oc):
    if cfg.get("model") == "XJ-C": assert oc["model"] == "XJ-C" and oc["nseed"] == len(cfg["seeds"]) and oc["v2"] == cfg["v2"] and cfg["bnclip"], (cfg, oc)
    elif cfg.get("model") in ("XJ-A", "XJ-B", "XJ-U", "XJ-U2", "XJ-1"):   # 14/11 AdaFace + CAM++ (xja.py) / ArcFace + ReDimNet (xjb.py) / all six (xju.py) / all six + XJ-SA pool (xju2.py)
        assert oc["model"] == cfg["model"] and oc["nseed"] == len(cfg["seeds"]) and oc["v2"] == cfg["v2"] and oc["nf"] == cfg["nf"] and cfg["bnclip"], (cfg, oc)
        assert (cfg["hidden"], cfg["outd"], cfg["epochs"], cfg["ratio"], cfg["drop"]) == (256, 128, 3, 4, 0.7) == (256, 128, oc["epochs"], oc["ratio"], 0.7), cfg
    else:
        fc.check_recipe(cfg, oc)   # BTC XJ recipe (XJ-SA)
        assert cfg.get("nf", False) == oc.get("nf", False), (cfg, oc)


nf_of = lambda cfg: bool(cfg.get("nf", False))


def main(out, oof_out, *members):
    out, oof_out = Path(out).resolve(), Path(oof_out).resolve(); assert not out.exists() and not oof_out.exists()
    T = fc.RB.E.load_trials(); mem, zo, ze = [], {}, {}
    for spec in members:
        d, o = (Path(x).resolve() for x in spec.split(":")); cfg = json.load(open(d / "config.json")); Z = np.load(o); oc = json.loads(str(Z["oof_cfg"])); check(cfg, oc)
        info = dict(dir=str(d.relative_to(T3)), oof=str(o.relative_to(T3)), oof_sha256=sha(o), config=cfg, oof_cfg=oc, const={})
        for tr in TRS:
            xo = np.concatenate([Z[f"f{k}_{tr}"] for k in range(5)]).astype(float); mu, sd = float(xo.mean()), float(xo.std()); info["const"][tr] = dict(mu=mu, sd=sd)
            for k in range(5): zo.setdefault(f"f{k}_{tr}", []).append((Z[f"f{k}_{tr}"].astype(float) - mu) / sd)
            for lang, _ in LANGS: ze.setdefault((tr, lang), []).append((np.load(d / f"cos_{tr}_{lang}.npy").astype(float) - mu) / sd)
            y = np.concatenate([T[f"f{k}_{tr}"]["y"] for k in range(5)]); assert fc.fb.eer(y, xo) < 50, "sign"
        mem.append(info)
    assert len({nf_of(m['config']) for m in mem}) == 1, 'family members must agree on FLAG / no-FLAG training'
    out.mkdir(parents=True)
    fam_oof = {k: np.mean(v, 0) for k, v in zo.items()}; oc = dict(model="family", members=[dict(dir=m["dir"], oof_sha256=m["oof_sha256"]) for m in mem])
    np.savez(oof_out, **fam_oof, oof_cfg=json.dumps(oc))
    for tr in TRS:
        for lang, tag in LANGS:
            c = np.mean(ze[(tr, lang)], 0); np.save(out / f"cos_{tr}_{lang}.npy", c)
            ref = pd.read_csv(T3 / f"submissions/t3_sub90c_eval/{tr}/sub_score_v4_{lang}_{tag}.txt", sep=" ", header=None, names=["id", "s"], dtype={"id": str})
            a = pd.read_csv(Path(T3 / mem[0]["dir"]) / tr / f"sub_score_v4_{lang}_{tag}.txt", sep=" ", header=None, names=["id", "s"], dtype={"id": str})
            assert len(c) == len(ref) and (a.id.values == ref.id.values).all()
            (out / tr).mkdir(exist_ok=True); (out / tr / f"sub_score_v4_{lang}_{tag}.txt").write_text("".join(f"{i} {-v:.6f}\n" for i, v in zip(ref.id.values, c)))
        y = np.concatenate([T[f"f{k}_{tr}"]["y"] for k in range(5)]); fo = np.concatenate([fam_oof[f"f{k}_{tr}"] for k in range(5)])
        mo = [np.concatenate([zo[f"f{j}_{tr}"][i] for j in range(5)]) for i in range(len(mem))]
        print(f"{tr}: family train-OOF EER {fc.fb.eer(y, fo):.2f} | members " + " ".join(f"{fc.fb.eer(y, m):.2f}" for m in mo)
              + " | member corr OOF " + " ".join(f"{np.corrcoef(mo)[i, j]:.3f}" for i in range(len(mo)) for j in range(i + 1, len(mo)))
              + " eval En " + " ".join(f"{np.corrcoef(ze[(tr, 'English')])[i, j]:.3f}" for i in range(len(mo)) for j in range(i + 1, len(mo))))
    json.dump(dict(model="family", members=mem, oof_file=str(oof_out.relative_to(T3)), oof_sha256=sha(oof_out), oof_cfg=oc,
                   rule="family score = mean of member z-scores, member mu/sd from its own FLAG-train OOF on the RF-split trials; equal weights fixed a priori"),
              open(out / "config.json", "w"), indent=1)
    print("wrote", out, oof_out)


if __name__ == "__main__":
    main(*sys.argv[1:])
