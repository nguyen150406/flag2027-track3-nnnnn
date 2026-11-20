#!/usr/bin/env bash
# Download, verify and unpack the release assets (features, checkpoints, reference outputs) into this repository.
#   bash fetch_assets.sh            needs the GitHub CLI (gh auth login) with access to this repository
#   bash fetch_assets.sh --local    if you downloaded the assets_part*.tar files of the release by hand into ./_assets/
set -euo pipefail
cd "$(dirname "$0")"
REPO="${REPO:-nguyen150406/flag2027-track3-nnnnn}"; TAG="${TAG:-v1.1}"; PY="${PY:-python}"
mkdir -p _assets
if [ "${1:-}" != "--local" ]; then gh release download "$TAG" -R "$REPO" -D _assets -p 'assets_part*.tar' --skip-existing; fi
"$PY" - <<'EOF'
import hashlib, pathlib, sys
bad = []
for l in pathlib.Path("ASSETS_PARTS_SHA256.txt").read_text().split("\n"):
    if not l.strip(): continue
    h, name = l.split(); p = pathlib.Path("_assets") / name
    if not p.exists(): bad.append(f"missing {name}"); continue
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): d.update(b)
    print(name, "OK" if d.hexdigest() == h else "CORRUPT"); bad += [] if d.hexdigest() == h else [f"corrupt {name}"]
sys.exit("asset download incomplete: " + ", ".join(bad) if bad else 0)
EOF
for t in _assets/assets_part*.tar; do echo "unpack $t"; tar -xf "$t"; done
"$PY" "Track 3/release/sub126c/check_assets.py"
echo "assets ready. Next: put the official FLAG files in data/ (see data/README.md), then: bash run_rebuild.sh"
