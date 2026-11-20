# Gói đầy đủ hệ sub90c: FLAG 2027 Track 3

Ngày đóng gói: 13/11/2026. Viết cho các thành viên trong team.

**sub90c là bản tốt nhất của team trên eval tính đến 13/11.**

| Ô eval | EER % |
|---|---|
| no_gender English | 16.84 |
| no_gender Bangla | 11.47 |
| gender English | 30.29 |
| gender Bangla | 19.83 |
| **Overall** | **19.61** |

- Bangla của sub90c thuộc nhóm tốt nhất bảng.
- English, nhất là gender English, là điểm yếu.

Lịch sử thí nghiệm và các bài học nằm ở [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

> **Bảo mật dữ liệu:** rulebook (Ethics 3) cấm chia sẻ hoặc re-host dữ liệu FLAG thô và feature trích sẵn.
> - Chỉ chia sẻ folder này đích danh cho thành viên team đã đăng ký. Không bật "anyone with the link".
> - Gói không chứa media gốc của VoxCeleb, VoxBlink hay MAV-Celeb. Ai cần thì tải lại từ nguồn chính thức bằng tài khoản của mình.

---

## 1. Hệ sub90c làm gì

Mỗi cặp (mặt, giọng) được chấm **độc lập**. Điểm là tổng log-odds của hai bộ Logistic Regression, mỗi bộ cân bằng lớp:

| Bộ | Cột đầu vào |
|---|---|
| **ours** | `NF` = trung bình 4 head (clipnfD_cat4, clipnfD_sim, clipnf2D_cat4, clipnf2D_sim), `farlD_cat4`, `gender`, `age` |
| **r2sys** | `r2` (projector InfoNCE R2, arm E1), `ibo` (luồng ImageBind frozen của mình), `gender`, `age` |

- **Protocol `gender`:** bỏ cột `gender`, vì trong protocol này nó chỉ đo độ lệch giữa hai bộ ước lượng giới tính.
- **Fit LR:** dùng các trial OOF trên FLAG train. Split là 5 fold theo người cộng 10 người dự trữ, mỗi fold có 3000 cặp dương và 3000 cặp âm cho từng protocol. Chuẩn hoá cũng chỉ tính trên train.
- **Không dùng** bất kỳ thống kê nào của tập test. Không CSLS, không TTA, không score normalization.
- Hệ số đã fit nằm trong `Track 3/submissions/t3_sub90c_eval/provenance.json`.

**Head mặt–giọng (`t3_ultra_heads.py`, arm D):**
- Projector AAM có thêm dữ liệu phụ từ VoxCeleb1, VoxCeleb2 và VoxBlink.
- Loss chính trên FLAG nhân 0.25, có decouple và có cặp dương khác video (aux_xvideo).
- Mỗi head có 5 model fold và 3 model final (3 seed).

**R2 (`t3_r2.py`, arm E1):**
- Mỗi modality đi qua LayerNorm, Dropout, rồi Linear 512.
- Loss InfoNCE đối xứng trên cặp mặt–giọng khác video của người ngoài: 512 người mỗi bước, lấy từ VoxCeleb2 và VoxBlink.
- Thêm một batch người ngoài cùng giới và một term FLAG cùng giới với trọng số 0.5.
- **Train trên CPU.** Bản train trên MPS bị phân kỳ.

**Encoder** (tất cả frozen, trừ LoRA CLIP):

| Loại | Model |
|---|---|
| Mặt | CLIP ViT-L/14 + LoRA **no-FLAG**: `clipnf` (seed 2027) và `clipnf2` (seed 7). LoRA train trên Kaggle chỉ bằng mặt người ngoài |
| Mặt | FaRL |
| Mặt và giọng | ImageBind-huge (commit 53680b02) |
| Giọng (`cat4`) | SimAM-ResNet100 (VoxBlink2), SimAM-ResNet100-ft, WeSpeaker ResNet293, WeSpeaker ECAPA-1024. Mỗi encoder được chuẩn hoá L2 riêng rồi nối lại |
| Tuổi và giới | ViT age classifier, MiVOLO v2, audeering wav2vec2 (24 lớp và 6 lớp) |

**Dữ liệu train:**
- FLAG train: 70 người.
- VoxCeleb1, VoxCeleb2, VoxBlink-clean. Bỏ toàn bộ người bị gắn cờ tiếng Bengali theo luật LID v3, cùng 550 người ngoài giữ lại để kiểm định (`caches/ext/exclude_ultra.csv`).
- LoRA mặt dùng thêm mặt của LRS3.
- **LID v3:** một clip bị gắn cờ khi bất kỳ model nào trong 3 model (VoxLingua107, Whisper-large-v3, MMS-LID) cho P(bn) + P(as) > 0.5. Chỉ cần 1 clip bị cờ là loại cả người.

---

## 2. Các phần của gói

Gói là **một folder** (`FLAG2027_T3_sub90c_fullpack/`), không nén. Folder này chính là thư mục gốc `FLAG_2027/` thu gọn: mọi đường dẫn trong code và trong README đều tính từ đây, và code tự tìm thư mục gốc theo vị trí của chính file. Tải nguyên folder về là chạy được, không cần sắp xếp lại.

> **Folder ẩn `.tools/`:** tên bắt đầu bằng dấu chấm nên Finder không hiện (bấm Cmd+Shift+. để hiện). Sau khi upload, kiểm tra trên Drive có `.tools/` (khoảng 0.8 GB, chứa checkpoint FaRL/SimAM và code WeSpeaker). Nếu thiếu thì upload riêng folder đó vào đúng vị trí.

| Đường dẫn trong folder | Nội dung |
|---|---|
| `README_SUB90C.md`, `EXPERIMENT_LOG.md`, `MANIFEST_SHA256.txt`, `env/`, `FLAG_2027_RULEBOOK.md` | Tài liệu, lịch sử thí nghiệm, sha256 từng file, môi trường, rulebook |
| `Track 3/scripts/`, `configs/`, `tests/`, `kaggle/`, `reports/`, `*.md` | Toàn bộ code Track 3: script, kernel builder `kaggle/*.py, *.sh` + notebook, runbook (`EVAL_RUNBOOK.md`, `RUNBOOK_CLEAN_RETRAIN.md`), design (`ENX2_DESIGN.md`, `ENX2R_DESIGN.md`, `PLAN_ULTRAPROMAX.md`), báo cáo |
| `Track 3/outputs/ultra/` | Script gate và pre-registration (`*.py`, `PREREG_gate_all.txt`); 40 model head (5 head × [5 fold + 3 final]) trong `models/`; 18 model R2 E1 (6 lần chạy × 3 seed) trong `r2/`; các cột OOF (`head_*.npz`, `attr.npz`, `ib_own*.np*`); bảng trial `trials.npz` |
| `Track 3/submissions/t3_sub90c_eval/`, `t3_sub90c_retrace/` | Zip đã nộp và bản dựng lại (giống hệt từng byte) |
| `Track 3/caches/ultra_lora/` | 2 checkpoint LoRA mặt no-FLAG (`clipnf.pt`, `clipnf2.pt`) |
| `Track 3/caches/imagebind/` | ImageBind-huge: repo đúng commit, checkpoint, pylib |
| `Track 3/caches/ext/`, `Track 3/caches/…` | Feature FLAG train của mình (clipnf, clipnf2, ImageBind, FaRL, các giọng, tuổi và giới); feature người ngoài (VoxCeleb1, VoxCeleb2, VoxBlink) cho mọi encoder dùng lúc train; danh sách người và danh sách loại (`*.csv`) |
| `Track 3/cv_run/` | Split `w/split.json`; bundle `b/` để tạo lại trial và attr |
| `Track 3/lid_lists/` | Danh sách cờ LID theo người (VoxLingua107, Whisper-large-v3; chỉ id) |
| `Track 3/outputs/eval_final/` | Feature eval đã trích (`feat/`, `ultra/`) và `rows.csv` |
| `Track 3/release/sub90c/local_eval_kit/` | **Bộ đánh giá local** cho mọi hệ: trial CV, HARD, RES, MAV, MV, EN, XL kèm điểm sub90c, và `score_local.py` để so với sub90c (xem mục 4.4) |
| `Track 3/outputs/mav_en2/`, `Track 3/submissions/t3_sub88b_eval` … `t3_sub92a_eval` | Feature của tập MAV English proxy; các bản đã nộp mà LEV-cal tái lập |
| `Track 1/src/`, `Track 2/src/`, `Track 2/outputs/t2_xp65_imagebind_train/`, `Track 2/img01_b01_cache/` | Code import chéo; feature ImageBind FLAG train của Track 2 (đầu vào của `t3_ib_own.py`); cache ảnh mặt Track 2 mà loader dùng |
| `flag_raw_s01_v3_adaface_simam100/` | `metadata.csv` (thứ tự hàng FLAG chung cho mọi script trích feature) và embedding giọng SimAM FLAG train |
| `.tools/wespeaker/`, `.tools/farl/`, `.tools/simam/` | Code WeSpeaker, code + checkpoint FaRL/CLIP, checkpoint SimAM |
| `third_party/hf_hub/`, `third_party/hf_modules/` | Snapshot HuggingFace: CLIP-L/14, audeering 24/6, MiVOLO v2 (kèm remote code), ViT-age, WeSpeaker ECAPA/ResNet293, SimAM-ft |
| `data/`, `evaluation_phase/` | **Tuỳ chọn, riêng tư:** dữ liệu FLAG gốc của BTC. Chỉ cần khi trích lại feature (mục 4.2). Đồng đội có thể tải lại từ kênh chính thức, nên có thể bỏ 2 folder này khi upload nếu muốn gói nhẹ hơn |

### Danh sách lọc dữ liệu ngoài (đều có trong gói)

Gói chỉ có danh sách người hoặc clip và kết quả LID, không có media. Script tạo ra chúng nằm trong `Track 3/scripts/`: `t3_lid_audit.py`, `t3_lid_filter.py`, `t3_lid_merge.py`, `t3_strict_exclude.py`, `vb2_bn_filter.py`, `enx2r_lid_list.py`. Kernel Kaggle tương ứng: `kaggle/build_t3_lid_notebook.py`, `build_t3_lidaudit_kernel.py`.

| File (trong `Track 3/`) | Nội dung |
|---|---|
| `caches/ext/exclude_bengali_v3.csv` | **LID v3, luật đang dùng.** Loại 993 người ngoài: cả người bị loại nếu bất kỳ model nào trong 3 model (VoxLingua107, Whisper, MMS) cho P(bn) + P(as) > 0.5 ở bất kỳ cửa sổ nào. Cộng thêm 26 người VoxCeleb1 quốc tịch Ấn Độ (không có audio để chạy LID), người không có clip nào được kiểm LID cũng bị loại (không kiểm chứng được), và các cờ Bengali theo người trong `lid_lists/speaker_bn_flags.json` (VoxLingua107, Whisper-large-v3; chỉ danh sách id), gộp vào để chắc chắn hơn |
| `caches/ext/exclude_ultra.csv` | **Danh sách sub90c dùng** (mặc định của `T3_EXCLUDE`): LID v3 cộng 550 người giữ lại cho EN (300) và XL (250) |
| `caches/ext/exclude_ultra_mav.csv` | Như trên, cộng 16 người MAV v1 giữ lại (MV) |
| `caches/ext/ultra_holdout.csv`, `mav_holdout.csv` | Danh sách người giữ lại cho EN/XL và cho MV |
| `outputs/lid_v3/{vox,web}/lid/*.csv` | Kết quả LID v3 thô: xác suất bn, as theo từng model và từng clip. `clips_*.csv` là danh sách clip đã kiểm tra |
| `kaggle/out_enx2r_smoke_v2/lid/lid_0of1.csv`, `outputs/enx2r/v2_lid_exclude.json` | LID v3 cho MAV-Celeb v2 (3143 clip) và danh sách loại cuối cùng (37/84 người, gồm LID và người trùng) |
| `caches/ext/exclude_bengali.csv`, `exclude_bengali_strict.csv`, `outputs/lid_audit_*`, `kaggle/outlid/lid_vox2.csv`, `caches/ext/si_lid.csv` | Các bản lọc cũ (v1, v2), các lần audit và LID cho nguồn SI. Giữ lại để đối chiếu |
| `caches/ext/exclude_strict_a4.csv` | Thí nghiệm A4: loại 7381 người có độ khớp mặt–giọng thấp nhất. **Không** dùng trong sub90c |
| `caches/ext/rows_*.csv`, `*_manifest.csv`, `mavx_subset.csv`, `domain_sa.csv` | Danh sách clip của từng nguồn ngoài; tập con MAV; danh sách người miền Nam Á cho batch domain của LoRA clipsa |

`MANIFEST_SHA256.txt` liệt kê sha256 của từng file.

---

## 3. Môi trường

- Máy dùng để chạy: Apple M4, macOS 26. Python chính là anaconda 3.13 (torch 2.9.1, transformers 4.57.3). Danh sách đầy đủ ở `env/requirements_main_anaconda_py313.txt`.
- MiVOLO chạy trong venv riêng (transformers 4.51): xem `env/requirements_mivolo_env.txt`. `t3_eval_extract.py` gọi nó qua `.tools/mivolo_env/bin/python`, nên phải tạo venv ở đúng đường dẫn đó.
- Dùng model HF offline:
  ```bash
  export HF_HUB_CACHE="$PWD/third_party/hf_hub" HF_HUB_OFFLINE=1
  export HF_MODULES_CACHE="$PWD/third_party/hf_modules"
  ```
- **Không train trên MPS.** R2 và các head trên MPS cho kết quả phân kỳ mà không báo lỗi. Train trên CPU hoặc CUDA; MPS chỉ dùng để suy luận.
- **Decoder JPEG:** cache mặt làm trên Kaggle (FaRL, LoRA) đọc ảnh bằng libjpeg-turbo, còn Pillow của anaconda dùng libjpeg 9. Vì vậy `t3_eval_extract.py` truyền `--turbo` cho các nhóm clipnf.

---

## 4. Cách dùng

### 4.1 Dựng lại sub90c từ feature có sẵn (vài phút, CPU)

```bash
cd "Track 3/scripts"
T3R2_OURS=D T3R2_NOGG=1 python t3_r2_build.py sub90c_check E1
# -> ../submissions/t3_sub90c_check/: 4 file điểm + zip; so sha256 với ../submissions/t3_sub90c_eval/provenance.json
```

Kết quả kiểm tra ngày 13/11 (`release/sub90c/trace_sub90c.py`):
- 4 file điểm dựng lại **giống hệt** bản đã nộp.
- `trials.npz` tạo lại giống hệt; `attr.npz` và `ib_own.npz` tạo lại có sai lệch tối đa 0.0.

### 4.2 Trích lại feature eval từ ảnh và wav

1. Tạo thư mục trung gian cho dữ liệu eval của BTC (sửa đường dẫn trong `gender/English_valid.txt`; file gốc không bị đụng):
   ```bash
   python t3_eval_view.py <evaluation_phase gốc> ../outputs/eval_final/root
   python t3_eval.py table ../outputs/eval_final/root ../outputs/eval_final/rows.csv
   ```
   `rows.csv` trong gói chứa đường dẫn tuyệt đối trên máy người làm, nên phải tạo lại bằng 2 lệnh trên.
2. Trích feature:
   ```bash
   python t3_eval_extract.py run ../outputs/eval_final/rows.csv ../outputs/eval_final/feat \
       farl clipnf clipnf2 sim sam100ft r293 ecapa vitage mivolo a24 a6 --device cpu
   # rồi chuyển face_clipnf*.npy sang ../outputs/eval_final/ultra/
   ```
3. Trích ImageBind (danh sách csv có cột `path`, theo đúng thứ tự dòng của rows.csv):
   ```bash
   python t3_imagebind_extract.py <faces.csv> face ../outputs/eval_final/ultra/ib_face.npy
   python t3_imagebind_extract.py <voices.csv> voice ../outputs/eval_final/ultra/ib_voice.npy
   ```
4. Sau đó chạy lại bước 4.1.

Ngày 13/11 đã trích lại 4 cặp eval cho đủ 13 nhóm feature. Cosine so với feature đã lưu là 1.00000 ở mọi nhóm; tuổi và giới lệch dưới 4e-6.

### 4.3 Train lại từ đầu

Theo thứ tự (chạy trong `Track 3/scripts`):

| Bước | Lệnh | Ghi chú |
|---|---|---|
| LoRA no-FLAG | `kaggle/build_t3_ultra_lora.py` (Kaggle T4, khoảng 5 giờ mỗi seed), rồi `t3_ultra_install_lora.py <output> clipnf\|clipnf2` | sinh cache mặt clipnf/clipnf2 cho FLAG và người ngoài |
| Trial | `python t3_ultra_eval.py trials` | split `cv_run/w/split.json`, seed 20261111 |
| Tuổi và giới | `python t3_ultra_eval.py attr` | ridge tuổi giọng được fit lại theo từng fold |
| ImageBind ibo | `python t3_ib_own.py` | mean để centering lấy từ người FIT của từng fold |
| Head arm D | `python t3_ultra_heads.py ../outputs/ultra/models <head> 0,1,2,3,4 1` và `... final 1,2,3` | cũng có kernel Kaggle `kaggle/build_t3_ultra_heads_kernel.py` |
| Cột OOF của head | `python t3_ultra_eval.py heads ../outputs/ultra/models <head,...>` | sinh `outputs/ultra/head_<b>.npz` |
| R2 E1 | `python t3_r2.py fit E1 clipnf 0,1,2,3,4,final 1,2,3`, rồi `python t3_r2.py score E1 clipnf` | **CPU**, khoảng 1.2 phút mỗi model |
| Build | bước 4.1 | |

Biến môi trường `T3_EXCLUDE` mặc định trỏ tới `caches/ext/exclude_ultra.csv`, tức danh sách Bengali v3 cộng 550 người ngoài giữ lại.

### 4.4 Đánh giá local và so hệ khác với sub90c

Xem [`Track 3/release/sub90c/local_eval_kit/README_LOCAL_EVAL.md`](Track%203/release/sub90c/local_eval_kit/README_LOCAL_EVAL.md).

- Đồng đội chấm hệ của mình trên đúng các trial trong `trials.csv.gz`, rồi chạy `python score_local.py my_scores.csv`. Kết quả là EER từng tập so với sub90c, kèm bootstrap theo người. Không cần feature hay model của mình.
- Hệ của mình chạy lại được trong gói: `desc/multi.py`, `t3_final.py gate_all` và `desc/lev_cal.py`.
- Lưu ý: proxy local **không** dự đoán được English trên eval (LEV-cal). Hệ nào đã train trên MAV v1 thì không dùng được tập MAV/MV.

---

## 5. Ghi chú về nguồn gốc và giới hạn

- **Trial và attr** dùng split `cv_run/w/split.json` và hàm `make_trials` trong `scripts/t3_cv_protocol.py`. Nó cho ra đúng split và trial của các lần chạy gốc. Metadata dòng lấy từ `cv_run/b/v4.npz`.
  - `attr.npz` gồm hai cột `gender` và `age`.
- **Feature ImageBind-vision của người ngoài** (`caches/ext/*_face_ibv`) lấy từ một lần trích trước đây (`cv_run/x face_ibf`, code trích không kèm theo). Muốn tự trích lại thì chạy `t3_imagebind_extract.py` trên ảnh mặt người ngoài, cần GPU.
- **Feature người ngoài** được dựng bằng các kernel Kaggle trước đây (`kaggle/build_*`).
  - Media gốc không có trong gói.
  - Mirror VoxCeleb2 trên HF `ProgramComputer/voxceleb` hiện đã khoá: cần đăng nhập và xin quyền.
  - Bản audio VoxCeleb2 chính thức có trên Kaggle: `omniaayman1/voxceleb2-audio`.
- **Head OOF:** mỗi fold 1 seed, còn model final có 3 seed. Độ lệch chuẩn của điểm khi chạy trên eval bằng 0.78–1.01 lần so với OOF.
- **Hạn chế đã biết:**
  - English (nhất là gender English) yếu hơn các đội top: 30.3 so với 19–24 của họ.
  - Các proxy local (CV, reserve, MAV, EN/XL) **không dự đoán được** điểm English trên eval. Phân tích LEV-cal ngày 13/11 xem ở EXPERIMENT_LOG.

## 6. Tuân thủ luật (tóm tắt)

- Mỗi trial được chấm độc lập: trong `t3_r2_build.py`, mọi phép tính đều làm trên từng dòng; hằng số chỉ lấy từ train (mean ImageBind của FLAG train, tham số chuẩn hoá và hệ số LR). Bài test đảo thứ tự / tập con / từng trial trước đây được chạy cho scorer sub60c (08/11), chưa chạy riêng cho sub90c.
- Không thống kê test, không CSLS hay cohort, không TTA.
- Trọng số fit trên OOF của train. Leaderboard chỉ dùng để chọn giữa các hệ hoàn chỉnh.
- Không dùng tiếng Bengali để train (lọc bằng LID v3). Mọi dữ liệu ngoài và model pretrained đều đã liệt kê ở trên để khai báo trong system description.
