#!/bin/bash
# VoxBlink2: timestamps -> selection -> YouTube fetch (user's machine) -> Kaggle encode (team) -> CV vs SUB10/11 -> SUB12 build.
T3="$HOME/Documents/FLAG_2027/Track 3"; cd "$T3/scripts"; TEAM=$HOME/.kaggle_team/access_token
log() { echo "$(date +%H:%M) $*"; }
until grep -q "^complete" ../caches/vb2/dl_meta.log; do sleep 30; done
grep -q "complete True" ../caches/vb2/dl_meta.log || { log "meta download incomplete"; exit 1; }
touch ../caches/vb2/vb2_meta.tar.gz.complete; log "meta downloaded"
[ -f ../caches/vb2/tracks.pkl ] || nice -n 5 python -u vb2_select.py 25000 || exit 1
log "fetch start"; VB2_MAX_HOURS=${VB2_MAX_HOURS:-10} python -u vb2_fetch.py 16 15000 2>&1 | grep --line-buffered -vE "^\[download\]|^ERROR|deprecat|WARNING" > ../caches/vb2/fetch.log
tail -2 ../caches/vb2/fetch.log; python pack_vb2.py || exit 1
DS=../caches/vb2/kaggle_ds; KAGGLE_API_TOKEN=$TEAM kaggle datasets create -p $DS 2>&1 | grep -v outdated | tail -1
for i in $(seq 1 60); do s=$(KAGGLE_API_TOKEN=$TEAM kaggle datasets status KAGGLE_USER/flag-t3-vb2-media 2>&1 | grep -v outdated | tail -1); echo "$s" | grep -qi ready && break; sleep 60; done; log "dataset $s"
for i in $(seq 1 60); do out=$(KAGGLE_API_TOKEN=$TEAM kaggle kernels push -p ../kaggle/t3_vb2enc_team 2>&1 | grep -v outdated | tail -1); log "$out"; echo "$out" | grep -qi "successfully pushed" && break; sleep 600; done
sleep 120
while true; do s=$(KAGGLE_API_TOKEN=$TEAM kaggle kernels status KAGGLE_USER/flag-t3-vb2-enc 2>&1 | grep -v outdated); case "$s" in *COMPLETE*) log "$s"; break;; *ERROR*|*CANCEL*) log "$s"; exit 1;; esac; sleep 180; done
mkdir -p ../kaggle/outvb2/x; KAGGLE_API_TOKEN=$TEAM kaggle kernels output KAGGLE_USER/flag-t3-vb2-enc -p ../kaggle/outvb2 2>&1 | grep -v outdated | tail -1
unzip -o -q ../kaggle/outvb2/t3_vb2.zip -d ../kaggle/outvb2/x && cp ../kaggle/outvb2/x/* ../caches/ext/ && rm -rf ../kaggle/outvb2/x ../kaggle/outvb2/t3_vb2.zip; ls ../caches/ext | grep -c vb2
until ! pgrep -f chain_night.sh >/dev/null; do sleep 120; done
log "CV start"; T3_RECIPE=combo T3_WSG=.5 python -u t3_j_par.py vox2vblrs3vb2 vox2,vb,lrs3,vb2 9 > ../outputs/j_vox2vblrs3vb2.log 2>&1
python t3_cmp.py vox2vblrs3 vox2vblrs3vb2 2>&1 | grep -v -iE "warn|linalg" | tee ../outputs/cmp_vb2.txt
b=$(grep "^vox2vblrs3 .*age v3" ../outputs/cmp_vb2.txt | sed -E 's/.*mean ([0-9.]+).*/\1/'); n=$(grep "^vox2vblrs3vb2 .*age v3" ../outputs/cmp_vb2.txt | sed -E 's/.*mean ([0-9.]+).*/\1/')
log "CV v3 age: base $b  +vb2 $n"
python -c "import sys; sys.exit(0 if float('$n') <= float('$b') + 0.02 else 1)" || { log "NOT BUILDING SUB12"; exit 2; }
python - <<'PY'
import json
c = json.load(open("../configs/t3_sub10.json")); c["name"] = "T3-SUB12"; c["age_mode"] = "v3"
for b in c["branches"]: b["aux"] = b["aux"] + [{"src": "vb2"}]
json.dump(c, open("../configs/t3_sub12.json", "w"), indent=1)
PY
python -u t3_build_par.py ../configs/t3_sub12.json ../submissions/t3_sub12 10 > ../caches/build_sub12.log 2>&1; grep -E "wrote|Traceback" ../caches/build_sub12.log
cd ../submissions && shasum -a 256 t3_sub12/submission.zip && log "SUB12 READY"
