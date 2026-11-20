#!/usr/bin/env bash
# Level 2 — RETRAIN the trainable English/Bengali partner models from the shipped features (CPU, ~15 min on an Apple M4):
#   XJ-1+IB (10 final towers + 5-fold FLAG-train OOF), HB-PF2, HB-BV2, HB-FIV (10 tower seeds + CCA + probes + LR fuser, final + OOF),
# then re-fit every fusion constant / weight / calibration from the new OOF and rebuild sub126c from the retrained members.
# 90c and 90c-X are taken from their shipped models here (their retraining = Level 3, see README).
#   bash run_retrain.sh [tag]        (tag = suffix of all new folders, default "re")
# Determinism: torch CPU kernels are deterministic for a fixed thread count; the thread counts below are the production ones (XJ 6, HB 4 / 4 / 5).
# On the original machine (Apple M4, macOS 26, torch 2.x CPU) every model was re-trained bit-exactly (outputs IDENTICAL). On other CPUs /
# BLAS builds tiny float differences are possible: the script then reports SAME RANKING / DIFFERENT with the max difference.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; MAIN_ROOT="${MAIN_ROOT:-$HERE}"; PY="${PY:-python}"; TAG="${1:-re}"
T3="$MAIN_ROOT/Track 3"; VER="$T3/release/sub126c/verify_sub126c.py"; FAIL=0
run() {   # run <trace name> <root> <script> [args...]
  local name="$1" root="$2"; shift 2
  if [ -n "${TRACE_DIR:-}" ]; then PYTHONPYCACHEPREFIX="$TRACE_DIR/pyc" "$PY" "$T3/release/sub126c/trace_open.py" "$TRACE_DIR/$name" "$root" "$@"
  else "$PY" "$@"; fi
}
check() { "$PY" "$VER" "$@" || FAIL=1; }
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }
cd "$T3/xj"

step "1/5 XJ-1+IB: 10 final towers (seeds 0..9000) on FLAG-train + external pool, then 5-fold OOF (50 towers)"
XJA_THREADS=6 run l2_xj1ib_final "$MAIN_ROOT" xj1ib.py final "xj1ib_$TAG"; check dirs xj1ib_10 "xj1ib_$TAG"
XJA_THREADS=6 run l2_xj1ib_oof "$MAIN_ROOT" xj1ib.py oof "$TAG"; check oof xj1ib_oof_x1ib_10.npz "xj1ib_oof_$TAG.npz"

step "2/5 HB-PF2 / HB-FIV (4 threads each, in parallel) and HB-BV2 (5 threads)"
hb() {   # hb <name> <threads> <face blocks> <voice blocks> <model>
  HB_SAVE=1 HB_THREADS=$2 HB_FLAG=1 HB_FB=$3 HB_VB=$4 HB_MODEL=$5 run "l2_$1" "$MAIN_ROOT" hb.py run "$1_$TAG" "$1_oof_$TAG.npz" > "$1_$TAG.log" 2>&1
}
hb hbpf2 4 clip,farl,pe simam,r293,wsv,pe pf2 & p1=$!; hb hbfiv 4 clip,farl,ibl simam,r293,wsv,ibv HB-FIV & p2=$!
wait $p1; wait $p2; hb hbbv2 5 clip,farl,ibl simam,r293,ibv HB-BV2
for m in hbpf2 hbfiv hbbv2; do tail -1 "${m}_$TAG.log"; check dirs "${m}_10" "${m}_$TAG"; check oof "${m}_oof_10.npz" "${m}_oof_$TAG.npz"; done

step "3/5 English: z-fusion weights (nested CV on the new OOF) + LID routing / Platt calibration"
AB_W="no_gender=.35,gender=.30" run l2_fuse_ab "$MAIN_ROOT" fuse_ab.py "hbpf2ab_$TAG" "hbpf2ab_${TAG}_oof.npz" XJ-1+IB+HB-PF2 \
  "xj1ib_$TAG:xj1ib_oof_$TAG.npz:XJ-1+IB" "hbpf2_$TAG:hbpf2_oof_$TAG.npz:pf2"
run l2_fuse_lid "$MAIN_ROOT" fuse_lid.py "sub126b_$TAG" "hbpf2ab_$TAG" "hbpf2ab_${TAG}_oof.npz" ../outputs/lid_eval_en sub99b self

step "4/5 Bengali: 90c-X + HB-BV2 / HB-FIV (fusion weight = median of the 5 fold-nested choices on the new OOF)"
run l2_fuse_bn2 "$MAIN_ROOT" fuse_bn2.py "sub126c_$TAG" "sub126b_$TAG" "hbbv2_$TAG:hbbv2_oof_$TAG.npz:HB-BV2" "hbfiv_$TAG:hbfiv_oof_$TAG.npz:HB-FIV"

step "5/5 RESULT: retrained sub126c vs the submitted sub126c (CodaBench 1057440)"
check subs "$T3/submissions/t3_sub126c/sub126c.zip" "$T3/submissions/t3_sub126c_$TAG"
echo; [ "$FAIL" = 0 ] && echo "LEVEL 2: ALL CHECKS PASSED (bit-exact retrain) -> Track 3/submissions/t3_sub126c_$TAG/sub126c_$TAG.zip" \
  || { echo "LEVEL 2: some outputs differ from the originals (see the max differences above; expected only on a different CPU/BLAS)"; exit 1; }
