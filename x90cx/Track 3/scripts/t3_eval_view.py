"""Eval-day input view (2026-11-11): the official evaluation_phase zip names the lists <Lang>_valid.txt (dev: _test) and every line of
gender/English_valid.txt has a space instead of '/' before the face file ("English_valid/faces 00000.jpg"). The official folder is
left untouched; this builds <view>/<proto>/<Lang>_test.txt (same pair ids, same order, paths repaired), a symlink <Lang>_valid -> the
official media folder, and features/<Lang>_test_{faces,voices}.csv symlinks, so the dry-run-validated pipeline runs unchanged.
  python t3_eval_view.py <official root (has no_gender/, gender/)> <view dir>"""
import sys
from pathlib import Path
import t3_build as TB

src, view = Path(sys.argv[1]).resolve(), Path(sys.argv[2]); assert not view.exists(), view
for proto, lang in TB.EXPECTED:
    d = view / proto; (d / "features").mkdir(parents=True, exist_ok=True)
    out, fixed = [], 0
    for l in (src / proto / f"{lang}_valid.txt").read_text().splitlines():
        t = l.split()
        if not t: continue
        if len(t) == 4 and t[2].endswith("/faces"): t = [t[0], t[1], t[2] + "/" + t[3]]; fixed += 1
        assert len(t) == 3, (proto, lang, l)
        for p in t[1:]: assert (src / proto / p).is_file(), (proto, lang, p)
        out.append(" ".join(t) + "\n")
    (d / f"{lang}_test.txt").write_text("".join(out))
    (d / f"{lang}_valid").symlink_to(src / proto / f"{lang}_valid")
    for kind in ("faces", "voices"):
        c = src / proto / "features" / f"{lang}_valid_{kind}.csv"; n = sum(1 for _ in open(c)); assert n == len(out), (c, n, len(out))
        (d / "features" / f"{lang}_test_{kind}.csv").symlink_to(c)
    print(f"{proto}/{lang}: {len(out)} pairs, {fixed} paths repaired, all media + feature rows present")
