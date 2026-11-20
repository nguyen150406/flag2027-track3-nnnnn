"""XJ-family final models: nseed seeds (0, 1000, ...; default 5) on all FLAG ids (minus filtered clips) + external ids; eval score = mean centred cosine;
file stores -cosine (6 decimals). Saves checkpoints, config.json and per-file raw cosines.
usage: python final_xj.py <outdir> [key=val ...]; keys: bengali(0/1 ext identity filter) bnclip(0/1 ext clip-level filter, 14/11) flag(none|all3|any) langs epochs ratio gpure hidden outd
Resume (14/11 review fix): an existing checkpoint is reused only when <outdir>/config.json equals this run's config."""
import sys, json, numpy as np, torch, xj_lib as X, xj_ext as XE
from pathlib import Path
out = Path(sys.argv[1]); kv = dict(a.split("=") for a in sys.argv[2:])
cfg = dict(bengali=kv.get("bengali", "0") == "1", flag=kv.get("flag", "none"), langs=kv.get("langs"), epochs=int(kv.get("epochs", 3)),
           ratio=int(kv.get("ratio", 4)), gpure=kv.get("gpure", "0") == "1", hidden=int(kv.get("hidden", 256)), outd=int(kv.get("outd", 128)),
           seeds=[1000 * i for i in range(int(kv.get("nseed", 5)))], vb=kv.get("vb", "0") == "1", mav_lid=kv.get("mav_lid", "0") == "1", vox2=kv.get("vox2", "0") == "1", mavx=kv.get("mavx", "0") == "1")
for k, typ, dflt in (("bs", int, 128), ("drop", float, 0.7), ("mav_w", float, 1.0), ("bnclip", lambda v: v == "1", False), ("nf", lambda v: v == "1", False)):   # nf (14/11): no FLAG pairs in training   # 14/11 iteration 8 knobs; stored only when non-default (old config.json stay comparable)
    if k in kv and typ(kv[k]) != dflt: cfg[k] = typ(kv[k])
assert not (cfg["bengali"] and cfg.get("bnclip")), "choose ONE Bengali rule (bengali or bnclip)"   # review 14/11: before config.json is written
out.mkdir(parents=True, exist_ok=True); cp = out / "config.json"
if cp.exists() and json.load(open(cp)) != cfg: sys.exit(f"config mismatch with existing {cp}: refuse to mix checkpoints")
if not cp.exists() and list(out.glob("xj_m*.pt")): sys.exit(f"{out} has checkpoints but no config.json: refuse to resume")
json.dump(cfg, open(cp, "w"), indent=1)
L, F, V = X.load_flag(); Fp, Vp = X.prep_face(F), X.prep_voice(V)
keep = ~X.flag_bengali_mask(L, cfg["flag"]) if cfg["flag"] != "none" else np.ones(len(L), bool); print(f"FLAG rows kept {keep.sum()} / {len(L)}", flush=True)
E = None; ms = []
for i, sd in enumerate(cfg["seeds"]):
    if (out / f"xj_m{i}.pt").exists():
        m = X.XJ(h=cfg["hidden"], o=cfg["outd"]); m.load_state_dict(torch.load(out / f"xj_m{i}.pt", map_location="cpu")); m.eval(); ms.append(m); continue
    E = E or XE.Ext(langs=cfg["langs"].split(",") if cfg["langs"] else None, bengali_filter=cfg["bengali"], vb=cfg["vb"], mav_lid_ok=cfg["mav_lid"], vox2=cfg["vox2"], mavx=cfg["mavx"], mav_w=cfg.get("mav_w", 1.0), bengali_clip=cfg.get("bnclip", False))
    m = (X.train_nf if cfg.get("nf") else X.train)(Fp[keep], Vp[keep], seed=sd, epochs=cfg["epochs"], ext=E.sampler(()), ext_ratio=cfg["ratio"], hidden=cfg["hidden"], outd=cfg["outd"],
                gpure=cfg["gpure"], genders=L.g.values[keep], ext_gender=XE.gender_map() if cfg["gpure"] else None, bs=cfg.get("bs", 128), drop=cfg.get("drop", 0.7))
    torch.save(m.state_dict(), out / f"xj_m{i}.pt"); ms.append(m)
    print(f"model {i}: train-pair mean cos {X.score(m, Fp[keep], Vp[keep]).mean():.3f}", flush=True)
info = {}
for tr in ("no_gender", "gender"):
    for lang, tag in (("English", "heard"), ("Bangla", "unheard")):
        Le, Fe, Ve = X.load_eval(tr, lang); c = np.mean([X.score(m, X.prep_face(Fe), X.prep_voice(Ve)) for m in ms], 0); assert np.isfinite(c).all()
        np.save(out / f"cos_{tr}_{lang}.npy", c); p = out / tr / f"sub_score_v4_{lang}_{tag}.txt"; p.parent.mkdir(exist_ok=True)
        p.write_text("".join(f"{a} {-b:.6f}\n" for a, b in zip(Le.pid, c))); info[f"{tr}/{lang}"] = int(len(c))
json.dump(info, open(out / "rows.json", "w")); print(info)
