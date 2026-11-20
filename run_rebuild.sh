#!/usr/bin/env bash
# Level 1 — rebuild sub126c from the shipped checkpoints and features (CPU, a few minutes). Every component is re-scored from its
# checkpoint and compared with the shipped component outputs, then the fusion chain rebuilds the 4 submission files and compares
# them byte-for-byte with the submitted sub126c.
#   bash run_rebuild.sh [tag]            (tag = suffix for the output folders, default "chk"; reuse needs a new tag)
# Env (only for development): MAIN_ROOT (default: this folder), X90_ROOT (default: $MAIN_ROOT/x90cx), PY (default: python),
#   TRACE_DIR (if set: every python step runs under release/sub126c/trace_open.py and writes its file list there).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; MAIN_ROOT="${MAIN_ROOT:-$HERE}"; X90_ROOT="${X90_ROOT:-$MAIN_ROOT/x90cx}"; PY="${PY:-python}"; TAG="${1:-chk}"
T3="$MAIN_ROOT/Track 3"; X3="$X90_ROOT/Track 3"; VER="$T3/release/sub126c/verify_sub126c.py"; FAIL=0
run() {   # run <trace name> <root> <script> [args...]   (cwd = caller's)
  local name="$1" root="$2"; shift 2
  if [ -n "${TRACE_DIR:-}" ]; then PYTHONPYCACHEPREFIX="$TRACE_DIR/pyc" "$PY" "$T3/release/sub126c/trace_open.py" "$TRACE_DIR/$name" "$root" "$@"
  else "$PY" "$@"; fi
}
check() { "$PY" "$VER" "$@" || FAIL=1; }
step() { echo; echo "=== $*"; }
for f in data/meta_file_train_set.csv data/train_set/features/faces/train_English_faces.csv data/eval_set/evaluation_phase/gender/English_valid.txt \
         data/eval_set/evaluation_phase/no_gender/features/Bangla_valid_voices.csv "Track 3/xj/xj1ib_10/xj1ib_m0.pt" "Track 3/xj/hbpf2_10/ckpt/cca.npz"; do
  [ -f "$MAIN_ROOT/$f" ] || { echo "missing: $f  (official FLAG files -> data/README.md; release assets -> bash fetch_assets.sh)"; exit 2; }
done

step "1/6 90c-X (Bengali base) from its heads / R2 models / LR constants"
cd "$X3/scripts"; T3R2_OURS=D T3R2_NOGG=1 run l1_90cx "$X90_ROOT" t3_r2_build.py "sub90cx_$TAG" E1
check subs "$X3/submissions/t3_sub90cx/sub90cx.zip" "$X3/submissions/t3_sub90cx_$TAG"
cmp -s "$X3/submissions/t3_sub90cx/sub90cx.zip" "$T3/submissions/t3_sub90cx/sub90cx.zip" && echo "shipped sub90cx.zip == copy used by the fusion: OK" || { echo "sub90cx.zip copies differ"; FAIL=1; }

step "2/6 90c (system of the language-routed English trials)"
cd "$T3/scripts"; T3R2_OURS=D T3R2_NOGG=1 run l1_90c "$MAIN_ROOT" t3_r2_build.py "sub90c_$TAG" E1
check subs "$T3/submissions/t3_sub90c_eval" "$T3/submissions/t3_sub90c_$TAG"

step "3/6 XJ-1+IB from its 10 tower checkpoints"
cd "$T3/xj"; run l1_xj1ib "$MAIN_ROOT" xj1ib.py score xj1ib_10 "xj1ib_score_$TAG"; check dirs xj1ib_10 "xj1ib_score_$TAG"

step "4/6 HB-PF2 / HB-BV2 / HB-FIV from their checkpoints"
HB_THREADS=4 HB_FLAG=1 HB_FB=clip,farl,pe  HB_VB=simam,r293,wsv,pe  HB_MODEL=pf2    run l1_hbpf2 "$MAIN_ROOT" hb.py score hbpf2_10/ckpt "hbpf2_score_$TAG"; check dirs hbpf2_10 "hbpf2_score_$TAG"
HB_THREADS=5 HB_FLAG=1 HB_FB=clip,farl,ibl HB_VB=simam,r293,ibv     HB_MODEL=HB-BV2 run l1_hbbv2 "$MAIN_ROOT" hb.py score hbbv2_10/ckpt "hbbv2_score_$TAG"; check dirs hbbv2_10 "hbbv2_score_$TAG"
HB_THREADS=4 HB_FLAG=1 HB_FB=clip,farl,ibl HB_VB=simam,r293,wsv,ibv HB_MODEL=HB-FIV run l1_hbfiv "$MAIN_ROOT" hb.py score hbfiv_10/ckpt "hbfiv_score_$TAG"; check dirs hbfiv_10 "hbfiv_score_$TAG"

step "5/6 English: z-fusion XJ-1+IB + HB-PF2 (w .35/.30), then per-clip LID routing + Platt calibration"
AB_W="no_gender=.35,gender=.30" run l1_fuse_ab "$MAIN_ROOT" fuse_ab.py "hbpf2ab_$TAG" "hbpf2ab_${TAG}_oof.npz" XJ-1+IB+HB-PF2 \
  "xj1ib_score_$TAG:xj1ib_oof_x1ib_10.npz:XJ-1+IB" "hbpf2_score_$TAG:hbpf2_oof_10.npz:pf2"
check dirs hbpf2ab_10 "hbpf2ab_$TAG"
run l1_fuse_lid "$MAIN_ROOT" fuse_lid.py "sub126b_$TAG" "hbpf2ab_$TAG" "hbpf2ab_${TAG}_oof.npz" ../outputs/lid_eval_en sub99b self

step "6/6 Bengali: z-fusion 90c-X + HB-BV2 (no_gender) / HB-FIV (gender), w .20/.20 -> final sub126c"
run l1_fuse_bn2 "$MAIN_ROOT" fuse_bn2.py "sub126c_$TAG" "sub126b_$TAG" "hbbv2_score_$TAG:hbbv2_oof_10.npz:HB-BV2" "hbfiv_score_$TAG:hbfiv_oof_10.npz:HB-FIV"
echo; echo "=== RESULT: rebuilt sub126c vs the submitted sub126c (CodaBench 1057440)"
check subs "$T3/submissions/t3_sub126c/sub126c.zip" "$T3/submissions/t3_sub126c_$TAG"
echo; [ "$FAIL" = 0 ] && echo "LEVEL 1: ALL CHECKS PASSED -> Track 3/submissions/t3_sub126c_$TAG/sub126c_$TAG.zip" || { echo "LEVEL 1: SOME CHECK FAILED (see above)"; exit 1; }
