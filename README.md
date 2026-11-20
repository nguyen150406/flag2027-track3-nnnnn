# FLAG 2027 Track 3: team nnnnn, submission `sub126c` (evaluation EER 13.82)

This repository holds the complete system behind our best FLAG 2027 Track 3 evaluation submission: code, configurations, trained checkpoints, precomputed features and scripts that rebuild the submitted score files.

| | no_gender English | no_gender Bengali | gender English | gender Bengali | Average |
|---|---|---|---|---|---|
| Evaluation EER (%), CodaBench submission 1057440 | 11.22 | 10.72 | 14.90 | 18.44 | **13.82** |

- **Method:** [SYSTEM_DESCRIPTION.md](SYSTEM_DESCRIPTION.md).
- **Reference submission:** `Track 3/submissions/t3_sub126c/sub126c.zip` (sha256 `dbd357022e5b10e50a504178a93fde4a44d2c3f8a34febfe8e88f5c86f106504`). Its `provenance.json` records the fusion weights and constants.

## 1. Quick start: rebuild the submission (Level 1, about 6 minutes on CPU)

```bash
gh repo clone nguyen150406/flag2027-track3-nnnnn && cd flag2027-track3-nnnnn
bash fetch_assets.sh                 # downloads the release assets (3 parts, 5.5 GB), checks SHA-256, unpacks them in place
# put the official FLAG 2027 files into data/ (list: data/README.md)
python3.13 -m venv .venv && . .venv/bin/activate && pip install -r env/requirements.txt
bash run_rebuild.sh
```

`run_rebuild.sh` does the following:

1. Re-scores every component from its saved checkpoint: 90c-X, 90c, XJ-1+IB, HB-PF2, HB-BV2 and HB-FIV.
2. Checks each component output against the shipped copy.
3. Runs the fusion chain.
4. Compares the four rebuilt score files with `sub126c.zip`.

The last line must read `LEVEL 1: ALL CHECKS PASSED`. On our machine (Apple M4, macOS 26.6, Python 3.13.5, torch 2.9.1 CPU) every check is **IDENTICAL**: the four files equal the submitted files byte for byte. The zip itself differs only in its timestamps.

## 2. Retrain the trainable models (Level 2, about 15 minutes on CPU)

```bash
bash run_retrain.sh
```

**What it retrains** from the shipped features:
- XJ-1+IB: 10 final towers plus 50 out-of-fold towers.
- HB-PF2, HB-FIV and HB-BV2: towers, CCA, gender probes and LR fusers, final plus out-of-fold.

**Then** it re-fits every fusion weight, z-constant and calibration from the new FLAG-train out-of-fold scores, rebuilds sub126c and compares it with the submission.

**Determinism:**
- Training is deterministic on CPU for a fixed thread count. The script uses the production thread counts.
- On the original machine every retrained model, out-of-fold file and the final four score files are **IDENTICAL** to the originals (bit-exact retrain).
- On a different CPU or BLAS build, small floating-point differences are possible. The script then prints the maximum difference and the rank correlation for each output.

## 3. Repository layout

The tree mirrors our project folder; the code finds its root from its own location.

| Path | Content |
|---|---|
| `Track 3/xj/` | English sub-system and HB family. `xj_lib.py`: towers and InfoNCE training. `xj1.py`, `xj1ib.py`: XJ-1+IB. `xjc.py`, `xju.py`, `xj_ext.py`: external pool loaders and Bengali rule. `hb.py`: HB. `fuse_ab.py`: English z-fusion. `fuse_lid.py`: language routing and Platt calibration. `fuse_bn2.py`: Bengali z-fusion. Trained models: `xj1ib_10/`, `hb*_10/ckpt/`. Out-of-fold scores: `*_oof_*.npz` |
| `Track 3/scripts/` | 90c system and all feature extractors. `t3_r2_build.py`: final LR fusers and file writer. `t3_r2.py`: R2 projector. `t3_ultra_heads.py`: projection heads. `t3_ultra_eval.py`: FLAG-train CV trials and attributes. `t3_cv_protocol.py`: CV split and trial sampler. `t3_eval_extract.py`, `t3_imagebind_extract.py`, `t3_extract_*.py`, `extract_*.py`: features. `t3_lid_audit.py`: language ID. `t3_ft_clip.py`, `t3_iblora.py`: LoRA training |
| `Track 3/outputs/` | Precomputed features. `eval_final/feat`, `eval_final/ultra`: evaluation rows, in the order of `eval_final/rows.csv`. `ultra/`: FLAG-train features, head and R2 models, out-of-fold columns, CV trials. `lid_eval_en/`: per-clip language ID of the English evaluation voices |
| `Track 3/caches/` | FLAG-train and external-pool features (`caches/ext`), identity lists and exclusion lists, demographic caches |
| `x90cx/Track 3/` | The 90c-X tree: the same code base, with the 90c-X heads, R2 models and LoRA features. `x90cx/Track 1`, `Track 2` and `data` are links to `../` |
| `Track 3/submissions/` | Reference submissions used by the chain: `t3_sub126c` (final), `t3_sub90c_eval`, `t3_sub99b`, `t3_sub101b` (ids and row order), `t3_sub90cx` |
| `Track 3/release/sub126c/` | `verify_sub126c.py`, `check_assets.py`, `trace_open.py`, and the traced file lists (`trace_l1/`, `trace_l2/`: every file each step opens) |
| `Track 3/release/sub90c/` | `README_SUB90C.md`: the original 90c documentation, with the retraining chain of the heads and R2. `local_eval_kit/`: FLAG-train CV trials and folds |
| `Track 3/kaggle/build_*.py` | Builders of the Kaggle kernels we used for GPU work: LoRA training, external feature extraction, LID |
| `Track 1/`, `Track 2/` | Small cross-imported modules and caches (face-age cache, speaker splits) |
| `env/` | `requirements.txt` (Levels 1–2), the full package list of the extraction environment, the MiVOLO environment, machine info |

`Track 3/scripts/` and `Track 3/xj/` also keep the experiment scripts of the development phase. Many of them are **not** part of sub126c; some, such as CSLS and score normalisation, were dropped after the organizers' rulings. The exact set of files that sub126c uses is listed in `Track 3/release/sub126c/trace_l1/` (rebuild) and `trace_l2/` (retrain).

## 4. Data

**Official FLAG data.** It is **not** included; we do not redistribute it. Place the official files as listed in [data/README.md](data/README.md). Levels 1 and 2 only need the official list files, the metadata file and the baseline feature CSVs, not the media.

**External data.** Raw media are **not** redistributed; we ship our extracted features only. Datasets (official releases):
- MAV-Celeb v1/v2;
- VoxCeleb1/2;
- VoxBlink and VoxBlink2 (official request form);
- LRS3 (faces, LoRA only).

The media we processed came from copies of these releases: the Kaggle builders in `Track 3/kaggle/` name the Kaggle datasets they mount (`KAGGLE_USER` stands for the account that holds the copy), and VoxCeleb2 and LRS3 media came from public mirrors. Official access to these corpora may require registration.

**Bengali exclusion.**
- External voice clips and identities flagged as Bengali are removed. A clip is flagged when VoxLingua107, Whisper-large-v3 or MMS-LID-256 gives P(bn) + P(as) > .5. The lists are `Track 3/caches/ext/exclude_*.csv` and `Track 3/outputs/lid_v3/`.
- An identity with no audited clip is excluded as unverified.
- The lists also use VoxCeleb1 nationality, VoxBlink official language labels, and speaker-level flags of language-ID runs (VoxLingua107, Whisper-large-v3) on other copies of the same speakers' clips (`Track 3/lid_lists/speaker_bn_flags.json`; the `reason` column says which). These only remove people.
- XJ-1+IB applies the rule per clip.

## 5. Level 3: retrain 90c and 90c-X (documented, long)

These chains need the full external-feature caches (about 30 GB) and, for the LoRA adapters, a GPU (we used Kaggle T4). Code and commands are in this repository.

**LoRA adapters:**
- `Track 3/kaggle/build_t3_ultra_lora.py` and `build_t3_lnfm.py`, which wrap `Track 3/scripts/t3_ft_clip.py --no_flag --sources vox,vox2,vb,lrs3[,mav]` with random horizontal flip.
- IB-LoRA: `Track 3/kaggle/build_iblora.py` and `t3_iblora.py`.

**90c-X:** `x90cx/Track 3/scripts/t3_90cx_chain.sh` runs, in order:
1. LoRA install;
2. evaluation faces;
3. the 4 projection heads (5 folds plus 3 final seeds);
4. R2 E1 (CPU);
5. out-of-fold columns;
6. the build.

**90c:** `Track 3/release/sub90c/README_SUB90C.md`, section 4.3: `t3_ultra_eval.py trials | attr`, `t3_ib_own.py`, `t3_ultra_heads.py`, `t3_r2.py fit/score E1`, `t3_r2_build.py`.

**Training rule:** train on CPU or CUDA only. Apple MPS training silently diverges.

## 6. Level 4: features from raw media

The evaluation features are extracted by `Track 3/scripts/t3_eval_extract.py run <rows.csv> <outdir> <groups...>` and the encoder-specific extractors:
- ImageBind: `t3_imagebind_extract.py`;
- WavLM-SV: `t3_extract_wavlmsv.py`;
- AdaFace: `t3_extract_adaface.py`;
- WeSpeaker: `extract_wespeaker.py`;
- PE-AV: `kaggle/build_hbpe_extract.py`;
- MiVOLO and age: `extract_mivolo_face_age.py`, `extract_face_age_vit.py`, `extract_voice_age.py`.

External-pool features were extracted with the Kaggle builders in `Track 3/kaggle/`. Pretrained model revisions are listed in `SYSTEM_DESCRIPTION.md` and `env/system_original_machine.txt`.

## 7. Compliance summary

- **Trial independence:** every trial is scored independently from its own face and voice.
- **Train-only constants:** every weight, z-constant, calibration and demographic-term weight comes from FLAG-train out-of-fold scores.
- **No test statistics:** no score normalisation with test statistics, no cohort, no CSLS, no TTA, no pseudo-labels.
- **Selection on CodaBench:** each evaluation submission was a complete system fixed before submission; their results guided which complete system to keep (development steps: `SYSTEM_DESCRIPTION.md` Section 6). No weight or constant comes from them.
