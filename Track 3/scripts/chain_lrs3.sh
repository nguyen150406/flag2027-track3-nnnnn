#!/bin/bash
# LRS3: wait for trainval.tar.gz -> local prep -> private Kaggle dataset (team) -> GPU encode (team) -> merge -> fuse check vs SUB09.
T3="$HOME/Documents/FLAG_2027/Track 3"; cd "$T3"
TEAM=$HOME/.kaggle_team/access_token
log() { echo "$(date +%H:%M) $*"; }
until grep -q "trainval exit" caches/lrs3_raw/dl.log; do sleep 30; done
grep -q "trainval exit 0" caches/lrs3_raw/dl.log || { log "download failed"; cat caches/lrs3_raw/dl.log; exit 1; }
sz=$(stat -f %z caches/lrs3_raw/trainval.tar.gz); log "trainval.tar.gz $sz bytes"; [ "$sz" -gt 10000000000 ] || { log "size too small"; exit 1; }
FF="$(ls /tmp/scratch/pylib/static_ffmpeg/bin/darwin_arm64/ffmpeg)"
python -u scripts/prep_lrs3.py "$FF" || { log "prep failed"; exit 1; }
DS=caches/lrs3_media/kaggle_ds; mkdir -p $DS; ln -f caches/lrs3_media/lrs3_media.zip $DS/lrs3_media.zip; cp caches/lrs3_media/lrs3_manifest.csv caches/lrs3_media/PROVENANCE_LRS3.json $DS/
printf '{"title": "FLAG T3 LRS3 Media", "id": "KAGGLE_USER/flag-t3-lrs3-media", "licenses": [{"name": "CC-BY-4.0"}]}' > $DS/dataset-metadata.json
KAGGLE_API_TOKEN=$TEAM kaggle datasets create -p $DS 2>&1 | grep -v outdated | tail -2
for i in $(seq 1 60); do s=$(KAGGLE_API_TOKEN=$TEAM kaggle datasets status KAGGLE_USER/flag-t3-lrs3-media 2>&1 | grep -v outdated | tail -1); log "dataset status: $s"; echo "$s" | grep -qi ready && break; sleep 60; done
for i in $(seq 1 60); do out=$(KAGGLE_API_TOKEN=$TEAM kaggle kernels push -p kaggle/t3_lrs3enc_team 2>&1 | grep -v outdated | tail -1); log "$out"; echo "$out" | grep -qi "successfully pushed" && break; sleep 600; done
sleep 120
while true; do s=$(KAGGLE_API_TOKEN=$TEAM kaggle kernels status KAGGLE_USER/flag-t3-lrs3-enc 2>&1 | grep -v outdated); case "$s" in *COMPLETE*) log "$s"; break;; *ERROR*|*CANCEL*) log "$s"; KAGGLE_API_TOKEN=$TEAM kaggle kernels output KAGGLE_USER/flag-t3-lrs3-enc -p kaggle/outlrs3 2>&1 | tail -1; grep -o '"data":"[^"]*' kaggle/outlrs3/*.log | tail -15; exit 1;; esac; sleep 180; done
mkdir -p kaggle/outlrs3/x; KAGGLE_API_TOKEN=$TEAM kaggle kernels output KAGGLE_USER/flag-t3-lrs3-enc -p kaggle/outlrs3 2>&1 | grep -v outdated | tail -1
unzip -o -q kaggle/outlrs3/t3_lrs3.zip -d kaggle/outlrs3/x && cp kaggle/outlrs3/x/* caches/ext/ && ls -la caches/ext | grep lrs3
cd scripts && T3_RECIPE=combo T3_WSG=.5 python -u t3_j_par.py vox2vblrs3 vox2,vb,lrs3 9 > ../outputs/j_vox2vblrs3.log 2>&1
grep -E "^vox2vblrs3|done in|Traceback|Error" ../outputs/j_vox2vblrs3.log
python t3_fuse_vs09.py vox2vblrs3 2>&1 | grep -v -i warn
