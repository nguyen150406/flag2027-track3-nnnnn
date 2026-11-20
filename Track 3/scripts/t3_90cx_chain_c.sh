#!/bin/bash
# 90c-X option C (15/11, user "cứ làm thử hết"; prereg PREREG_gate_all.txt "90c-X A/B/C"): the A+B system with ONE change, a language
# adversary (gradient reversal, w_lgrl 0.05 = middle of our old gender-GRL range .01-.1) on the voice embedding of the 4 LoRA D heads.
# Runs in a SECOND APFS clone made from the finished A+B clone, so LoRAs, eval faces, R2 E1, farlD and LR recipe are identical:
#   cp -cR $HOME/Documents/FLAG_2027_90cx $HOME/Documents/FLAG_2027_90cxc
#   cd "$HOME/Documents/FLAG_2027_90cxc/Track 3/scripts" && bash t3_90cx_chain_c.sh
set -euo pipefail
here="$(pwd -P)"; [[ "$here" == $HOME/Documents/FLAG_2027_90cxc/* ]] || { echo "run inside the 90c-X C clone"; exit 1; }
export T3_EXCLUDE="$here/../caches/ext/exclude_ultra.csv"; export T3X_LGRL=0.05
log() { echo "$(date +%H:%M:%S) $*"; }
[[ -f ../submissions/t3_sub90cx/provenance.json ]] || { echo "A+B build missing (clone made too early?)"; exit 1; }
for b in clipnfD_cat4 clipnfD_sim clipnf2D_cat4 clipnf2D_sim; do rm -f ../outputs/ultra/models/${b}__*; done
log "heads with the language adversary (2 jobs x 4 threads)"
(python -u t3_ultra_heads.py ../outputs/ultra/models clipnfD_cat4,clipnfD_sim 0,1,2,3,4 2027 4 && python -u t3_ultra_heads.py ../outputs/ultra/models clipnfD_cat4,clipnfD_sim final 2027,3407,42 4) > ../c_heads_a.log 2>&1 & pa=$!
(python -u t3_ultra_heads.py ../outputs/ultra/models clipnf2D_cat4,clipnf2D_sim 0,1,2,3,4 2027 4 && python -u t3_ultra_heads.py ../outputs/ultra/models clipnf2D_cat4,clipnf2D_sim final 2027,3407,42 4) > ../c_heads_b.log 2>&1 & pb=$!
wait $pa; wait $pb; tail -2 ../c_heads_a.log ../c_heads_b.log; grep -h "language adversary\|sources" ../c_heads_a.log ../c_heads_b.log | sort -u
grep -L "w_lgrl 0.05" ../outputs/ultra/models/{clipnfD_cat4,clipnfD_sim,clipnf2D_cat4,clipnf2D_sim}__*.done | grep . && { echo "a head without the adversary"; exit 1; } || true
log "head OOF columns"; python -u t3_ultra_eval.py heads ../outputs/ultra/models clipnfD_cat4,clipnfD_sim,clipnf2D_cat4,clipnf2D_sim
log "build"; T3R2_OURS=D T3R2_NOGG=1 python -u t3_r2_build.py sub90cxc E1
log "CHAIN C DONE"
