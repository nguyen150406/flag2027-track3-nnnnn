"""check_90cx (15/11, prereg PREREG_gate_all.txt 22:12): HARD checks + health report for a 90c-X build before it becomes a ticket.
usage: python check_90cx.py <name> <clone Track 3 dir> [--lgrl]      e.g. python check_90cx.py sub90cx "/Users/.../FLAG_2027_90cx/Track 3"
HARD (exit 1 on any failure): zip has the 4 files, finite, ids/order == sub107a; LR coefficient columns and signs == the clone's sub90c rebuild
(t3_sub90c_clonecheck); the 4 LoRA D heads x (5 folds seed 2027 + 3 finals) all present, every .done lists the mav source, and (with --lgrl)
carries "w_lgrl 0.05", else "w_lgrl 0.0".
HEALTH (reported; flags -> "not recommended"): Bangla eval scores vs sub90c's Bangla files: Spearman, mean / sd of the log-odds."""
import sys, json, zipfile, hashlib
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
XD = Path(__file__).resolve().parent; T3 = XD.parent; SUB = T3 / "submissions"
FILES = [f"{p}/sub_score_v4_{l}.txt" for p in ("no_gender", "gender") for l in ("English_heard", "Bangla_unheard")]
HEADS = ["clipnfD_cat4", "clipnfD_sim", "clipnf2D_cat4", "clipnf2D_sim"]; MODELS = [f"{f}__s2027" for f in range(5)] + [f"final__s{s}" for s in (2027, 3407, 42)]


def parse(b):
    L = b.decode().splitlines(); return [l.split()[0] for l in L], np.array([float(l.split()[1]) for l in L])


def main(name, clone, lgrl):
    clone = Path(clone); hard, health = [], []
    zp = clone / f"submissions/t3_{name}/{name}.zip"; z = zipfile.ZipFile(zp); ref = zipfile.ZipFile(SUB / "t3_sub107a/sub107a.zip")
    if sorted(z.namelist()) != sorted(FILES): hard.append(f"zip files {z.namelist()}")
    b90 = {n: (SUB / "t3_sub90c_eval" / n).read_bytes() for n in FILES if "Bangla" in n}
    for n in FILES:
        ids, v = parse(z.read(n)); rid, _ = parse(ref.read(n))
        if not np.isfinite(v).all(): hard.append(f"{n} not finite")
        if ids != rid: hard.append(f"{n} ids/order != sub107a")
        if "Bangla" in n:
            i9, v9 = parse(b90[n]); assert i9 == ids
            rho = float(spearmanr(v, v9)[0]); m, s, m9, s9 = -v.mean(), v.std(), -v9.mean(), v9.std()
            flag = rho < .7 or not (.5 <= s / s9 <= 2) or abs(m - m9) > s9
            health.append(dict(file=n, spearman_vs_90c=round(rho, 4), logodds_mean=round(m, 3), logodds_sd=round(s, 3), mean_90c=round(m9, 3),
                               sd_90c=round(s9, 3), flag=bool(flag)))
    P = json.load(open(clone / f"submissions/t3_{name}/provenance.json")); R = json.load(open(clone / "submissions/t3_sub90c_clonecheck/provenance.json"))
    for pr in R["coefs"]:
        for nm in R["coefs"][pr]:
            a, r = P["coefs"][pr][nm], R["coefs"][pr][nm]
            if a["cols"] != r["cols"]: hard.append(f"coef cols {pr}/{nm}")
            elif not np.array_equal(np.sign(a["coef"]), np.sign(r["coef"])): hard.append(f"coef sign flip {pr}/{nm}: {a['coef']} vs {r['coef']}")
    md = clone / "outputs/ultra/models"; want = "w_lgrl 0.05" if lgrl else "w_lgrl 0.0"
    for h in HEADS:
        for m in MODELS:
            d = md / f"{h}__{m}.done"
            if not d.exists() or not (md / f"{h}__{m}.pt").exists(): hard.append(f"missing {d.name}"); continue
            t = d.read_text()
            if "mav" not in t: hard.append(f"{d.name} without the mav source")
            if want not in t: hard.append(f"{d.name} without '{want}'")
    out = dict(name=name, zip=str(zp), zip_sha256=hashlib.sha256(zp.read_bytes()).hexdigest(), hard_fail=hard, health=health,
               health_flag=any(h["flag"] for h in health), coefs={pr: {nm: [round(c, 3) for c in P["coefs"][pr][nm]["coef"]] for nm in P["coefs"][pr]} for pr in P["coefs"]})
    print(json.dumps(out, indent=1)); print("HARD:", "FAIL" if hard else "PASS", "| HEALTH:", "FLAG" if out["health_flag"] else "ok")
    sys.exit(1 if hard else 0)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--lgrl"]; main(a[0], a[1], "--lgrl" in sys.argv)
