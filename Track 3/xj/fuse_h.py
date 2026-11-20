"""fuse_h (14/11, user "tôi muốn lấy 98c,d eng còn phần bangla cứ lấy của xj"): stitch a submission from two BUILT submissions, chosen before any of
them was scored: English files (both protocols) from <en_sub>, Bangla files (both protocols) from <bn_sub>; bytes copied from their zips unchanged.
usage: python fuse_h.py <name> <en_sub> <bn_sub>      e.g. python fuse_h.py sub98e sub98c sub98a"""
import os, sys, json, zipfile, hashlib, tempfile
from pathlib import Path
XD = Path(__file__).resolve().parent; T3 = XD.parent; SUB = T3 / "submissions"
sha = lambda b: hashlib.sha256(b).hexdigest()
TRS = ("no_gender", "gender"); EN = "sub_score_v4_English_heard.txt"; BN = "sub_score_v4_Bangla_unheard.txt"


def main(name, en_sub, bn_sub):
    out = SUB / f"t3_{name}"; assert not out.exists(), out
    za, zb = (zipfile.ZipFile(SUB / f"t3_{s}/{s}.zip") for s in (en_sub, bn_sub))
    pa, pb = (json.load(open(SUB / f"t3_{s}/provenance.json")) for s in (en_sub, bn_sub))
    data = {}
    for tr in TRS:
        data[f"{tr}/{EN}"] = za.read(f"{tr}/{EN}"); data[f"{tr}/{BN}"] = zb.read(f"{tr}/{BN}")
        for n, z in ((f"{tr}/{EN}", za), (f"{tr}/{BN}", zb)):   # ids / order = the other source (same eval lists)
            other = (zb if z is za else za).read(n); ida = [l.split()[0] for l in data[n].decode().splitlines()]; idb = [l.split()[0] for l in other.decode().splitlines()]
            assert ida == idb and len(ida) > 0, n
    tmp = Path(tempfile.mkdtemp(dir=SUB, prefix=f".t3_{name}_")); zp = tmp / f"{name}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for tr in TRS:
            for n in (f"{tr}/{EN}", f"{tr}/{BN}"): z.writestr(n, data[n])
    json.dump(dict(system=f"stitched: English (both protocols) = {en_sub} English files; Bangla (both protocols) = {bn_sub} Bangla files; bytes unchanged. "
                          f"Choice fixed by the user before any of the sources was scored.",
                   english_source=dict(submission=en_sub, zip_sha256=sha((SUB / f"t3_{en_sub}/{en_sub}.zip").read_bytes()), system=pa["system"]),
                   bangla_source=dict(submission=bn_sub, zip_sha256=sha((SUB / f"t3_{bn_sub}/{bn_sub}.zip").read_bytes()), system=pb["system"]),
                   files={n: dict(rows=len(b.decode().splitlines()), sha256=sha(b)) for n, b in data.items()},
                   code_sha256={"fuse_h.py": sha((XD / "fuse_h.py").read_bytes())}, zip_sha256=sha(zp.read_bytes()),
                   pair_local=True, test_statistics_used=False, weights_tuned_on_eval=False), open(tmp / "provenance.json", "w"), indent=1)
    os.rename(tmp, out); print("wrote", out / f"{name}.zip", sha((out / f"{name}.zip").read_bytes())[:16])


if __name__ == "__main__":
    main(*sys.argv[1:])
