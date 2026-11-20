#!/bin/bash
run() { nice -n 5 python -u xj_val.py "$@" 2>&1 | grep -v -E "^ext:|Warning|bengali filter|voxblink:|voxceleb2:"; }
run B0c fresh=1
run NGc fresh=1 flag=all3 langs=english,unk vb=1 mav_lid=1 vox2=1
run Gc fresh=1 flag=all3 langs=english,unk hidden=512 outd=256 mav_lid=1 vox2=1
python xj_val.py ens ENSc NGc,Gc
python xj_val.py cmp B0c ENSc; python xj_val.py cmp B0c Gc
