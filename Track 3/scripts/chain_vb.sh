#!/bin/bash
# VoxBlink pipeline: prep (CPU, both accounts) -> enc (GPU: team a = farl/clip/simam/sam100ft, main b = ecapa/r293)
# -> download -> merge -> parallel 6-branch fuse check (SUB07 + vb). Exits non-zero on the first failure.
cd "$HOME/Documents/FLAG_2027/Track 3/kaggle"
TEAM=$HOME/.kaggle_team/access_token
log() { echo "$(date +%H:%M) $*"; }
st() { if [ "$1" = team ]; then KAGGLE_API_TOKEN=$TEAM kaggle kernels status "$2" 2>&1 | grep -v outdated; else kaggle kernels status "$2" 2>&1 | grep -v outdated; fi; }
waitk() {  # acct ref -> 0 on COMPLETE
  while true; do s=$(st "$1" "$2"); case "$s" in *COMPLETE*) log "$2 COMPLETE"; return 0;; *ERROR*|*CANCEL*) log "$2 FAILED: $s"; return 1;; esac; sleep 180; done; }
pushk() {  # acct dir; retry while GPU slots are busy
  for i in $(seq 1 60); do
    if [ "$1" = team ]; then out=$(KAGGLE_API_TOKEN=$TEAM kaggle kernels push -p "$2" 2>&1 | grep -v outdated | tail -1); else out=$(kaggle kernels push -p "$2" 2>&1 | grep -v outdated | tail -1); fi
    echo "$out" | grep -qi "successfully pushed" && { log "pushed $2"; return 0; }
    log "push $2 retry: $out"; sleep 600
  done; return 1; }
getout() {  # acct ref dir
  mkdir -p "$3"; if [ "$1" = team ]; then KAGGLE_API_TOKEN=$TEAM kaggle kernels output "$2" -p "$3" 2>&1 | grep -v outdated | tail -1; else kaggle kernels output "$2" -p "$3" 2>&1 | grep -v outdated | tail -1; fi
  mkdir -p "$3/x"; for z in "$3"/t3_vb_*.zip; do unzip -o -q "$z" -d "$3/x"; done; ls "$3/x"; }
( waitk main trncnguyn/flag-t3-voxblink-prep && pushk main t3_vbenc_main && sleep 120 && waitk main trncnguyn/flag-t3-voxblink-enc-b && getout main trncnguyn/flag-t3-voxblink-enc-b outvb_main ) > ../outputs/chain_vb_main.log 2>&1 &
PM=$!
( waitk team KAGGLE_USER/flag-t3-voxblink-prep && pushk team t3_vbenc_team && sleep 120 && waitk team KAGGLE_USER/flag-t3-voxblink-enc-a && getout team KAGGLE_USER/flag-t3-voxblink-enc-a outvb_team ) > ../outputs/chain_vb_team.log 2>&1 &
PT=$!
wait $PM; RM=$?; wait $PT; RT=$?
cat ../outputs/chain_vb_main.log ../outputs/chain_vb_team.log
[ $RM -eq 0 ] && [ $RT -eq 0 ] || { log "a branch of the Kaggle pipeline failed (main=$RM team=$RT)"; exit 1; }
cd ../scripts && python merge_vb.py ../kaggle/outvb_team/x ../kaggle/outvb_main/x || exit 1
T3_RECIPE=combo T3_WSG=.5 python -u t3_j_par.py vox2vb vox2,vb 9 > ../outputs/j_vox2vb.log 2>&1
grep -E "^vox2vb|done in|Traceback|Error" ../outputs/j_vox2vb.log
python t3_fuse_vs07.py vox2vb 2>&1 | grep -v -i warn
