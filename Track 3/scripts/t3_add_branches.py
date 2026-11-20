"""Add the branches of a separately built config to an existing submission by re-fusing saved components (no training):
raw config weights of both configs are merged and the branch part is rescaled to a total of 1.6 (CV scale); age/gender
weights from the base config. Checks: the base files are reproduced from their components, and both builds computed the
same age/gender components.
Usage: python t3_add_branches.py <base sub dir> <base cfg> <add build dir> <add cfg> <out dir> [replace=farlft_:farl_]
   replace=A:B drops each base branch named like an added branch with prefix A swapped for B (e.g. farl_sim -> farlft_sim)
   e.g. t3_sub23b_tta2 ../configs/t3_sub23.json t3_sub24 ../configs/t3_sub24.json t3_sub24b"""
import sys, json, shutil, zipfile, numpy as np
from pathlib import Path
import t3_build as TB
from t3lib import T3

S = T3 / "submissions"; base, add, out = S / sys.argv[1], S / sys.argv[3], S / sys.argv[5]
bc, ac = json.loads(Path(sys.argv[2]).read_text()), json.loads(Path(sys.argv[4]).read_text())
pb = json.load(open(base / "provenance.json")); bw = pb["config"]["weights"] if "config" in pb else pb["weights"]  # weights that produced the base files (build/rescore or an earlier add)
rep = sys.argv[6].split("=", 1)[1].split(":") if len(sys.argv) > 6 else None
drop = {b["name"].replace(rep[0], rep[1], 1) for b in ac["branches"]} if rep else set()
assert drop <= {b["name"] for b in bc["branches"]}, drop
raw = {b["name"]: bc["weights"][b["name"]] for b in bc["branches"] if b["name"] not in drop} | {b["name"]: ac["weights"][b["name"]] for b in ac["branches"]}
NW = {n: w * 1.6 / sum(raw.values()) for n, w in raw.items()}
if out.exists(): raise FileExistsError(out)
out.mkdir(parents=True); files = {}
for (proto, lang), name in TB.EXPECTED.items():
    c = dict(np.load(base / f"components_{proto}_{lang}.npz")); d = dict(np.load(add / f"components_{proto}_{lang}.npz"))
    lines = (base / name).read_text().splitlines(); assert list(c["ids"]) == list(d["ids"]) == [l.split()[0] for l in lines]
    rest = lambda x: bw["age"] * x["age"] + (bw.get("gender", 0) * x["gender"] if "gender" in x else 0)
    old = sum(bw[b["name"]] * c[b["name"]] for b in bc["branches"]) + rest(c)
    assert np.abs(old + np.array([float(l.split()[1]) for l in lines])).max() < 1e-6  # base file reproduced
    for k in ("age", "gender"):
        if k in c: assert np.allclose(c[k], d[k], atol=1e-5), k  # same age/gender terms in both builds
    comp = c | {b["name"]: d[b["name"]] for b in ac["branches"]}
    s = sum(NW[n] * comp[n] for n in NW) + rest(c); np.savez(out / f"components_{proto}_{lang}.npz", **comp)
    p = out / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("".join(f"{i} {v:.8f}\n" for i, v in zip(c["ids"], -s)))
    files[name] = dict(rows=len(lines), sha256=TB.sha(p), corr_with_base=float(np.corrcoef(old, s)[0, 1]))
    print(f"{name:45s} base reproduced; corr with base {files[name]['corr_with_base']:.4f}")
with zipfile.ZipFile(out / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for name in TB.EXPECTED.values(): z.write(out / name, name)
for f in ("VOXBLINK_PROVENANCE.json", "PROVENANCE_LRS3.json", "PROVENANCE_AVSPEECH.json"): shutil.copy(S / "t3_sub21a_en" / f, out / f)
json.dump(dict(base=str(base), added=str(add), replaced=sorted(drop), weights=NW | {"age": bw["age"], "gender": bw.get("gender", 0)}, files=files,
               zip_sha256=TB.sha(out / "submission.zip"), pair_local=True, test_statistics_used=False), open(out / "provenance.json", "w"), indent=1)
TB.named_copy(out); print("wrote", out / "submission.zip", TB.sha(out / "submission.zip"), {k: round(v, 4) for k, v in NW.items()})
