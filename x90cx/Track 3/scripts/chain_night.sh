#!/bin/bash
# Overnight queue: wait for the LRS3 CV run, compare SUB09/SUB10 OOFs with both age terms, then projector experiments.
cd "$HOME/Documents/FLAG_2027/Track 3/scripts"
until ! pgrep -f chain_sub10_fast.sh >/dev/null; do sleep 60; done
echo "== baseline comparison $(date +%H:%M)"; python t3_cmp.py vox2vb vox2vblrs3 2>&1 | grep -v -iE "warn|linalg"
echo "== E1 aux_ids=64 start $(date +%H:%M)"
T3_RECIPE=combo T3_WSG=.5 T3_AUXIDS=64 python -u t3_j_par.py e1aux64 vox2,vb,lrs3 9 > ../outputs/j_e1aux64.log 2>&1
grep -E "^e1aux64|done in|Traceback" ../outputs/j_e1aux64.log; python t3_cmp.py vox2vblrs3 e1aux64 2>&1 | grep -v -iE "warn|linalg"
echo "== E2 epochs=25 start $(date +%H:%M)"
T3_RECIPE=combo T3_WSG=.5 T3_EPOCHS=25 python -u t3_j_par.py e2ep25 vox2,vb,lrs3 9 > ../outputs/j_e2ep25.log 2>&1
grep -E "^e2ep25|done in|Traceback" ../outputs/j_e2ep25.log; python t3_cmp.py vox2vblrs3 e2ep25 2>&1 | grep -v -iE "warn|linalg"
echo "== done $(date +%H:%M)"
