"""check_assets (FLAG 2027 Track 3 sub126c release): verify the unpacked release assets against ASSETS_SHA256.txt.
  python "Track 3/release/sub126c/check_assets.py"            (run from anywhere; checks every listed file under the repository root)
Exit 0 = every asset present with the listed sha256, 1 = something missing / different."""
import sys, hashlib
from pathlib import Path
R = Path(__file__).resolve().parents[3]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


bad, n = [], 0
for l in (R / "ASSETS_SHA256.txt").read_text().splitlines():
    if not l.strip() or l.startswith("#"): continue
    h, size, part, rel = l.split(None, 3); p = R / rel; n += 1
    if not p.is_file(): bad.append(f"MISSING  {rel}  ({part})")
    elif p.stat().st_size != int(size) or sha(p) != h: bad.append(f"DIFFERENT  {rel}  ({part})")
print("\n".join(bad[:30]) + ("\n..." if len(bad) > 30 else ""))
print(f"assets: {n - len(bad)}/{n} OK" + ("" if not bad else f", {len(bad)} missing/different"))
sys.exit(1 if bad else 0)
