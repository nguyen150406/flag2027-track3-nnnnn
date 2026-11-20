# FLAG 2027 Track 3: System Description (submission `sub126c`, evaluation EER 13.82)

## 1. Overview

The submitted system scores each (face, voice) trial independently. It uses frozen pretrained face and voice encoders, small projection models trained by us, and logistic-regression fusers.

The English lists and the Bengali lists are scored by two fixed sub-systems:

| | no_gender English | no_gender Bengali | gender English | gender Bengali | Average |
|---|---|---|---|---|---|
| Evaluation EER (%) | 11.22 | 10.72 | 14.90 | 18.44 | **13.82** |
| FLAG-train 5-fold CV EER (%)¹ | 13.18 | 12.58 | 20.70 | 20.30 | – |

¹ English: the final fusion weights applied to the out-of-fold scores, without routing; with fold-nested weights and routing included, 13.62 (no_gender) and 20.66 (gender). Bengali: fold-nested weights.

```
ENGLISH LISTS                                              BENGALI LISTS
  XJ-1+IB (towers, ext. pool MAV+VoxCeleb1)                  90c-X (heads + R2 projector + gender/age, LR fusers)
  HB-PF2  (towers + CCA + gender probes, ext. pool Vox2+VB)  HB-BV2 (no_gender) / HB-FIV (gender)
        └── (1-w)·z_XJ + w·z_HB, w = .35 / .30                     └── (1-w)·z_90cX + w·z_HB, w = .20 / .20
        └── per-clip language routing: clips that 3 LID models
            all identify as Bengali take the 90c score (Platt-calibrated)
  output = −log-odds (lower = same person)
```

**Training-only principle.** Every weight, z-score constant, calibration constant and demographic-term weight was fitted on FLAG-train out-of-fold (OOF) scores:

- 5 person-disjoint folds over 60 training identities;
- 30,000 balanced trials per protocol;
- negatives in the gender protocol are always same-gender;
- fusion weights are chosen nested inside the folds, then the median of the 5 inner choices is taken.

No statistic of the progress or evaluation data is used anywhere: no score normalisation, no cohort, no TTA, no transductive step.

## 2. Encoders (frozen at scoring time)

**Face encoders:**
- VGGFace fc7, extracted with the baseline pipeline.
- AdaFace IR-101 (WebFace12M) and ArcFace (InsightFace buffalo_l).
- CLIP ViT-L/14, used two ways: frozen, and with LoRA adapters that we trained on external faces only (VoxCeleb1/2, VoxBlink, LRS3; plus MAV-Celeb for 90c-X).
- FaRL.
- ImageBind-huge vision, used frozen and with a LoRA adapter trained on VoxCeleb1 + MAV-Celeb only.
- PE-AV base video tower (`facebook/pe-av-base`), fed each face image as a 1-frame video.

**Voice encoders:**
- ECAPA-TDNN from the baseline pipeline.
- CAM++, ReDimNet-b6, SimAM-ResNet100 (VoxBlink2) and its VoxCeleb2-fine-tuned version, ResNet293 and ECAPA-1024 (WeSpeaker).
- WavLM-Base+-SV.
- ImageBind-huge audio.
- PE-AV audio tower (48 kHz, centre 20 s).

**Demographic models:** MiVOLO v2 and a ViT age classifier for faces; audeering wav2vec2-large-robust-24/6-ft-age-gender for voices.

**Language ID (routing only):** VoxLingua107-ECAPA, Whisper-large-v3 and MMS-LID-256.

## 3. English sub-system

### 3.1 XJ-1+IB

- **Inputs:**
  - Face side: 7 blocks [VGGFace (sqrt, then L2) | AdaFace | ArcFace | CLIP+LoRA ×2 | FaRL | ImageBind].
  - Voice side: 8 blocks [ECAPA | CAM++ | ReDimNet | SimAM | SimAM-ft | ResNet293 | ECAPA-1024 | ImageBind].
  - Each block is L2-normalised; the blocks are concatenated and divided by √n.
- **Model:** two towers (Linear → BatchNorm → ReLU → Dropout .7 → Linear, 256 → 128) trained with symmetric InfoNCE.
  - 3 epochs, batch 128, Adam (lr 1e-3, weight decay 1e-4).
  - Fixed centring with the FLAG-train mean embedding.
- **Data:**
  - All FLAG-train pairs.
  - In each epoch, 4× as many external pairs drawn from MAV-Celeb v1/v2 and VoxCeleb1. Voice clips that language ID identifies as Bengali are excluded.
- **Score:** mean cosine of 10 seeds.

### 3.2 HB-PF2 (second family)

- **Data:**
  - VoxCeleb2 + VoxBlink: 57,394 voices, 114,786 faces, 20,660 identities.
  - Bengali clips removed with language ID.
  - External identities whose voice centroid is too close to a FLAG identity (cosine > .83) are dropped.
  - FLAG-train pairs are added to the tower training.
- **Blocks:**
  - Faces: [CLIP-L | FaRL | PE-AV].
  - Voices: [SimAM | ResNet293 | WavLM-SV | PE-AV].
  - PE-AV embeddings are centred with the FLAG-train mean before L2 normalisation.
- **Streams:**
  - S1: InfoNCE towers (10 seeds; positives are same-video face–voice pairs).
  - S2: regularised CCA on identity means (k = 128).
  - P(male) logistic probes for face and voice, trained on external gender labels.
- **Fuser:** a per-protocol logistic regression on [z1, z2, pmf, pmv, d, z1·d, z2·d], with d = |pmf − pmv|, fitted on FLAG-train CV.

### 3.3 Fusion and language routing

- **Fusion:** s = (1 − w)·z(XJ-1+IB) + w·z(HB-PF2), with w = .35 (no_gender) and .30 (gender). The z-constants come from each member's FLAG-train OOF scores.
- **Language routing, per clip:**
  - Every English-list voice clip is passed through the three LID models.
  - If all three give P(Bengali) + P(Assamese) > .5 on some 30 s window, the trial takes the score of our 90c system.
  - Both scores are mapped to log-odds with a per-protocol Platt calibration fitted on FLAG-train OOF trials.
  - 89/392 (no_gender) and 100/416 (gender) English trials are routed.
  - The decision uses only that clip's audio.

## 4. Bengali sub-system

### 4.1 90c-X

Two logistic-regression branches; their log-odds are summed 1:1.

**First branch:** LR on [NF, FaRL head, gender, age].
- NF is the mean of 4 projection heads. Each head projects CLIP-L/14+LoRA faces against either 4-encoder voice embeddings (SimAM, SimAM-ft, ResNet293, ECAPA-1024) or SimAM alone.
- **LoRA adapters:** trained on external faces only.
  - Sources: VoxCeleb1/2, VoxBlink, LRS3, MAV-Celeb.
  - Same-person positives across languages are forced for MAV-Celeb identities.
  - Random horizontal flip during training.
- **Heads:** trained with FLAG losses (×.25) plus cross-video positives from external identities (VoxCeleb1/2, VoxBlink, MAV-Celeb).

**Second branch:** LR on [R2, ImageBind cosine, gender, age].
- R2 is a LayerNorm → Dropout → Linear(512) projector.
- It is trained with InfoNCE on cross-video face–voice pairs of VoxCeleb2/VoxBlink identities, plus a same-gender batch and a small FLAG same-gender term.
- 3 models, trained on CPU.

**Demographic terms:**
- Gender = log P(same gender), using MiVOLO P(male) for faces and audeering P(male) for voices. It is not used in the gender protocol.
- Age = −|face age − voice age|. Voice age combines the audeering output with a ridge regressor fitted on FLAG-train identities.

### 4.2 Fusion

- s = 0.8·z(90c-X) + 0.2·z(HB).
- The HB partner is HB-BV2 for no_gender and HB-FIV for gender. Both use the HB recipe of 3.2:
  - faces [CLIP-L | FaRL | ImageBind+LoRA];
  - voices [SimAM | ResNet293 | (WavLM-SV) | ImageBind].
- Partner and weight per protocol were chosen on FLAG-train nested CV (lowest nested CV error among the screened candidates).

## 5. Data

- **FLAG:** the complete official training set (70 identities, 6,485 pairs), without filtering.
- **External data:** MAV-Celeb v1/v2, VoxCeleb1, VoxCeleb2, VoxBlink/VoxBlink2, and LRS3 (faces, LoRA only).
- **Bengali rule:**
  - No Bengali speech corpus was used.
  - External voice clips are removed when language ID (3 models) flags them as Bengali; official language labels and nationality lists are also used.
  - The exclusion list also holds speaker-level flags of language-ID runs on other copies of the same speakers' clips (VoxLingua107 / Whisper-large-v3; identity lists only). These ids only remove people from the training pools.
  - All pretrained audio encoders are frozen.
- **Augmentation:** horizontal flip for LoRA training only; none at test time.

## 6. Development and model selection

Every component and constant of Section 1 was selected on FLAG-train cross-validation. The system grew in steps; each step was a complete system, fixed before it was submitted to CodaBench. Our main evaluation submissions:

| CodaBench submission | Change | Evaluation EER (%) |
|---|---|---|
| sub101b | Two-tower model XJ-1 for English; 90c for Bengali | 15.58 |
| sub107a | + per-clip language routing of English trials (Bengali voices to 90c) | 15.46 |
| sub109a | + ImageBind face and voice blocks in XJ-1 (XJ-1+IB) | 15.00 |
| sub110a | Bengali: 90c → 90c-X (cross-language identities) | 14.86 |
| sub119b | + second English family HB (VoxCeleb2 + VoxBlink pool) | 14.59 |
| sub119e | HB towers also trained on the FLAG training pairs (HB-F) | 14.34 |
| **sub126c** | English: HB-F → HB-PF2 (PE-AV blocks); Bengali: + HB partner (HB-BV2 / HB-FIV) | **13.82** |

Variants that did not help and were not kept: XJ-1 on the Bengali lists (sub101a, 20.23), XJ in the Bengali fusion (sub102a, 16.57), age/gender attributes in XJ-1 (sub106a, 17.06), an XJ pool of MAV-Celeb only (sub113a, 16.94), XJ split into two disjoint-encoder halves (sub114a, 15.46), an identity-encoder branch trained on FLAG and MAV-Celeb only (sub116a, 17.44), and HB with more external data (sub119c, 14.97).

- The evaluation results of the submitted systems guided which direction to continue and which complete system to keep.
- No parameter, weight or constant was derived from evaluation scores or data.

## 7. Reproducibility

The repository contains code, configurations, checkpoints and features (release assets), with three levels:

1. **`run_rebuild.sh`:** scores every component from its checkpoint and rebuilds `sub126c.zip` byte-for-byte.
2. **`run_retrain.sh`:** retrains the towers and HB models from features. On CPU with the stated thread counts it is bit-exact on the original machine.
3. **Documented chains** for the 90c/90c-X heads and R2, the LoRA kernels (Kaggle) and feature extraction from raw media.
