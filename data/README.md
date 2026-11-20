# Official FLAG 2027 files (not included)

The official FLAG 2027 distribution is not redistributed in this repository. Copy these files from the official release into this folder, keeping the paths below.

Levels 1 and 2 (`run_rebuild.sh`, `run_retrain.sh`) need only these list, metadata and baseline-feature files. They do not need any audio or image.

```
data/meta_file_train_set.csv
data/train_set/train_set/train_English.txt
data/train_set/features/faces/train_English_faces.csv
data/train_set/features/voices/train_English_voices.csv
data/eval_set/evaluation_phase/no_gender/English_valid.txt
data/eval_set/evaluation_phase/no_gender/Bangla_valid.txt
data/eval_set/evaluation_phase/gender/English_valid.txt
data/eval_set/evaluation_phase/gender/Bangla_valid.txt
data/eval_set/evaluation_phase/{no_gender,gender}/features/{English,Bangla}_valid_{faces,voices}.csv
```

`data/eval_set/evaluation_phase/` holds the evaluation-phase lists and baseline feature CSVs exactly as distributed for the evaluation phase.

**Re-extracting features from media (Level 4)** also needs:
- the training media: `data/train_set/...`;
- the evaluation media: the official `evaluation_phase/` folder, linked through `Track 3/outputs/eval_final/rows.csv`.

Recreate `rows.csv` with `Track 3/scripts/t3_eval_view.py` and `t3_eval.py table`, as described in `Track 3/release/sub90c/README_SUB90C.md` §4.2.
