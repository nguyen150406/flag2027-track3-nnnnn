#!/bin/bash
# After the user's cookie-based fetch: package -> new dataset version (team) -> GPU encode -> CV vs SUB10 -> SUB-VB2.
T3="$HOME/Documents/FLAG_2027/Track 3"; cd "$T3/scripts"; TEAM=$HOME/.kaggle_team/access_token
log() { echo "$(date +%H:%M) $*"; }
until grep -q "FETCH DONE" ../caches/vb2/fetch.log; do sleep 120; done
log "fetch done"; python pack_vb2.py || exit 1
KAGGLE_API_TOKEN=$TEAM kaggle datasets version -p ../caches/vb2/kaggle_ds -m "VoxBlink2 full fetch" 2>&1 | grep -v outdated | tail -1
sleep 300
for i in $(seq 1 60); do s=$(KAGGLE_API_TOKEN=$TEAM kaggle datasets status KAGGLE_USER/flag-t3-vb2-media 2>&1 | grep -v outdated | tail -1); echo "$s" | grep -qi ready && break; sleep 60; done; log "dataset $s"
for i in $(seq 1 60); do out=$(KAGGLE_API_TOKEN=$TEAM kaggle kernels push -p ../kaggle/t3_vb2enc_team 2>&1 | grep -v outdated | tail -1); log "$out"; echo "$out" | grep -qi "successfully pushed" && break; sleep 600; done
sleep 180
while true; do s=$(KAGGLE_API_TOKEN=$TEAM kaggle kernels status KAGGLE_USER/flag-t3-vb2-enc 2>&1 | grep -v outdated); case "$s" in *COMPLETE*) log "$s"; break;; *ERROR*|*CANCEL*) log "$s"; exit 1;; esac; sleep 180; done
rm -rf ../kaggle/outvb2; mkdir -p ../kaggle/outvb2/x; KAGGLE_API_TOKEN=$TEAM kaggle kernels output KAGGLE_USER/flag-t3-vb2-enc -p ../kaggle/outvb2 2>&1 | grep -v outdated | tail -1
unzip -o -q ../kaggle/outvb2/t3_vb2.zip -d ../kaggle/outvb2/x && cp ../kaggle/outvb2/x/* ../caches/ext/ && rm -rf ../kaggle/outvb2/x ../kaggle/outvb2/t3_vb2.zip
python -c "import pandas as pd; r=pd.read_csv('../caches/ext/rows_vb2.csv'); print('vb2 rows', len(r), 'speakers', r.identity.nunique())"
until ! pgrep -f chain_night.sh >/dev/null; do sleep 120; done
log "CV start"; T3_RECIPE=combo T3_WSG=.5 python -u t3_j_par.py vox2vblrs3vb2 vox2,vb,lrs3,vb2 9 > ../outputs/j_vox2vblrs3vb2.log 2>&1
python t3_cmp.py vox2vblrs3 vox2vblrs3vb2 2>&1 | grep -v -iE "warn|linalg" | tee ../outputs/cmp_vb2.txt
b=$(grep "^vox2vblrs3 .*age v2" ../outputs/cmp_vb2.txt | sed -E 's/.*mean ([0-9.]+).*/\1/'); n=$(grep "^vox2vblrs3vb2 .*age v2" ../outputs/cmp_vb2.txt | sed -E 's/.*mean ([0-9.]+).*/\1/')
log "CV: SUB10 $b  +vb2 $n"
python -c "import sys; sys.exit(0 if float('$n') <= float('$b') + 0.02 else 1)" || { log "NOT BUILDING SUB-VB2"; exit 2; }
python - <<'PY'
import json
c = json.load(open("../configs/t3_sub10.json")); c["name"] = "T3-SUB-VB2"
for b in c["branches"]: b["aux"] = b["aux"] + [{"src": "vb2"}]
json.dump(c, open("../configs/t3_sub_vb2.json", "w"), indent=1)
PY
python -u t3_build_par.py ../configs/t3_sub_vb2.json ../submissions/t3_sub_vb2 10 > ../caches/build_sub_vb2.log 2>&1; grep -E "wrote|Traceback" ../caches/build_sub_vb2.log
cd ../submissions && shasum -a 256 t3_sub_vb2/submission.zip && log "SUB-VB2 READY"
