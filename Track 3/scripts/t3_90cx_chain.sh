#!/bin/bash
# 90c-X chain (15/11, prereg PREREG_gate_all.txt "90c-X"): the documented sub90c retrain (README_SUB90C 4.3) on the MAV-augmented no-FLAG
# LoRAs, run INSIDE the APFS clone $HOME/Documents/FLAG_2027_90cx (the original tree is never written).
#   cd "$HOME/Documents/FLAG_2027_90cx/Track 3/scripts" && bash t3_90cx_chain.sh <download dir lnfm> <download dir lnfm2> <mavsrc dir>
# 15/11 "cứ làm thử hết": LoRAs trained with option A (--xlang mav); option B = the clone's t3_ultra_heads.py adds the "mav" aux source to
# the 4 LoRA D heads (installed below from <mavsrc dir> + the kernels' MAV face arrays). Option C (language adversary) = t3_90cx_chain_c.sh
# in a second clone made from this one AFTER this chain (same LoRAs, eval faces and R2).
# Unchanged from sub90c: trials, attr, ibo, farlD head, LR recipe (T3R2_OURS=D, T3R2_NOGG=1, arm E1). CPU only (never MPS).
set -euo pipefail
here="$(pwd -P)"; [[ "$here" == $HOME/Documents/FLAG_2027_90cx/* ]] || { echo "run inside the 90c-X clone"; exit 1; }
export T3_EXCLUDE="$here/../caches/ext/exclude_ultra.csv"
log() { echo "$(date +%H:%M:%S) $*"; }
log "install"; python t3_90cx_install.py "$1" clipnfm clipnf "$3"; python t3_90cx_install.py "$2" clipnfm2 clipnf2 "$3"
grep -q '"mav",))' t3_ultra_heads.py || { echo "clone heads without the mav source"; exit 1; }
[[ -z "${T3X_LGRL:-}" ]] || { echo "T3X_LGRL must be unset for the A+B chain"; exit 1; }
log "eval faces (LoRA groups, libjpeg-turbo decode as the Kaggle caches)"
python t3_eval_extract.py run ../outputs/eval_final/rows.csv ../outputs/eval_final/feat_x clipnf clipnf2 --device cpu
for g in clipnf clipnf2; do cp ../outputs/eval_final/feat_x/face_$g.npy ../outputs/eval_final/ultra/face_$g.npy; done
log "drop the clone's old 90c models of the 4 LoRA heads + R2 E1 (the resumable trainers would otherwise skip them)"
for b in clipnfD_cat4 clipnfD_sim clipnf2D_cat4 clipnf2D_sim; do rm -f ../outputs/ultra/models/${b}__*; done
rm -f ../outputs/ultra/r2/E1_clipnf__*
log "heads (2 jobs x 4 threads)"
(python -u t3_ultra_heads.py ../outputs/ultra/models clipnfD_cat4,clipnfD_sim 0,1,2,3,4 2027 4 && python -u t3_ultra_heads.py ../outputs/ultra/models clipnfD_cat4,clipnfD_sim final 2027,3407,42 4) > ../x_heads_a.log 2>&1 & pa=$!
(python -u t3_ultra_heads.py ../outputs/ultra/models clipnf2D_cat4,clipnf2D_sim 0,1,2,3,4 2027 4 && python -u t3_ultra_heads.py ../outputs/ultra/models clipnf2D_cat4,clipnf2D_sim final 2027,3407,42 4) > ../x_heads_b.log 2>&1 & pb=$!
wait $pa; wait $pb; tail -2 ../x_heads_a.log ../x_heads_b.log; grep -h "sources" ../x_heads_a.log ../x_heads_b.log | sort -u
log "R2 E1 (CPU)"; python -u t3_r2.py fit E1 clipnf 0,1,2,3,4,final 1,2,3; python -u t3_r2.py score E1 clipnf
log "head OOF columns"; python -u t3_ultra_eval.py heads ../outputs/ultra/models clipnfD_cat4,clipnfD_sim,clipnf2D_cat4,clipnf2D_sim
log "build"; T3R2_OURS=D T3R2_NOGG=1 python -u t3_r2_build.py sub90cx E1
log "CHAIN DONE"
