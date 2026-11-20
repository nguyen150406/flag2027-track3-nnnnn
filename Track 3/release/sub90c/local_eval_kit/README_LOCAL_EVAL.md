# Bộ đánh giá local của team (Track 3)

Viết cho các thành viên trong team, ngày 13/11/2026.

Bộ này cho phép chấm **bất kỳ hệ nào** trên đúng các trial mà team dùng để đánh giá local, rồi so với sub90c (bản tốt nhất trên eval, 19.61) trên **cùng các trial**. Không cần feature hay model của mình. Chỉ cần numpy, pandas và scikit-learn.

| File | Nội dung |
|---|---|
| `trials.csv.gz` | 103.828 trial (2 protocol × 5 tập), kèm điểm từng trial của sub90c |
| `score_local.py` | Chấm file điểm của bạn, so với sub90c, có bootstrap theo người |
| `folds.json` | 5 fold người FLAG-train và 10 người reserve (split RF, seed 20261103) |
| `heldout_people.csv` | Những người ngoài (MAV, EN, XL) mà hệ của bạn **không được** dùng để train |
| `export_kit.py` | Script đã tạo ra các file trên từ hệ của mình (chạy lại được trong gói) |

---

## 1. Các tập

Mỗi tập có 50% cặp dương, 50% cặp âm. Ở protocol gender, cặp âm luôn cùng giới.

| Tập | Trial mỗi protocol | Người | Cách tạo | sub90c no_gender / gender |
|---|---|---|---|---|
| **CV** | 30.000 | 60 người FLAG-train, 5 fold | Mỗi fold có 3000 cặp dương và 3000 cặp âm, mặt và giọng đều là người trong fold. Điểm sub90c là điểm out-of-fold | 13.38 / 20.56 |
| **HARD** | 7.930 / 8.211 | Tập con của CV | Cặp dương có mặt ít giống mặt trong chính clip giọng nhất (gần với khác phiên ghi), và cặp âm lệch tuổi dưới 5 năm. Ngày 12/11, mức điểm HARD khớp với mức English trên eval | 20.34 / 29.95 |
| **RES** | 6.000 | 10 người reserve của FLAG-train | Không thuộc fold nào | 15.32 / 24.88 |
| **MAV** (MAV-ALL) | 3.914 | 55 người MAV-Celeb **v1**, giọng English | Mỗi clip giọng có 1 cặp dương (mặt cùng người, khác video) và 1 cặp âm | 18.19 / 26.37 |
| **MV** | 1.110 | 15 trong 55 người MAV | Chưa từng được dùng bởi bất kỳ nhánh thử nghiệm nào của mình | 17.75 / 24.95 |
| **EN** | 6.000 | 300 người VoxCeleb2 và VoxBlink2 nói English | Người giữ lại, không dùng để train | 10.23 / 13.95 (*) |
| **XL** | 6.000 | 250 người VoxBlink2 nói tiếng Nam Á (không phải Bengali) | Người giữ lại, không dùng để train | 21.08 / 22.55 (*) |

(*) Với EN và XL, cột sub90c là **hệ rút gọn** LR[NF, farlD, r2] fit trên OOF của FLAG. Các cột ImageBind, giới và tuổi không có cho những hàng này. Giống cách `outputs/ultra/desc/multi.py` tính.

Tham chiếu tới dữ liệu gốc, không kèm media:

| Tiền tố | Ví dụ | Nghĩa |
|---|---|---|
| `flag:` | `flag:faces/English/id041/01229.jpg` | Đường dẫn tương đối trong dữ liệu train chính thức của FLAG |
| `mav1:` | `mav1:voices/id0006/English/3YdCvOC-kI0/00002.wav` | Đường dẫn tương đối trong MAV-Celeb v1 |
| `vox2:` | `vox2:id05668/XWKC1JUM_ow/00052_1.jpg` | VoxCeleb2 dev: `<id>/<video>/<clip>`. Giọng là `.wav` 16 kHz mono tách từ clip. Mặt `_1` là frame thứ 20, `_2` là frame thứ 60 của mp4 (`kaggle/build_t3_vox2_notebook.py`) |
| `vb:` | `vb:vb_id19167/sBD6vrgOd9o/00001.wav` | VoxBlink2: `<id>/<video>/<clip>`. Mặt `<clip>_<k>.jpg` là frame do mình trích (`scripts/pack_vb2.py`) |

---

## 2. Cách so hệ của bạn với sub90c

1. Lấy các trial cần chấm từ `trials.csv.gz`. Các cột: `trial_id, protocol, set, fold, hard, y, face_ref, voice_ref, face_person, voice_person, sub90c, mav_split`.
2. Chấm từng trial theo đúng luật dưới đây. Nếu không, con số **không so được** với sub90c.

   | Tập | Luật |
   |---|---|
   | CV | Out-of-fold: với fold k, train chỉ bằng người của 4 fold còn lại (`folds.json`), rồi chấm fold k. 10 người reserve không được nằm trong dữ liệu train |
   | RES | Train bằng người của cả 5 fold, không có 10 người reserve, rồi chấm RES |
   | MAV, MV, EN, XL | Dùng model cuối. Những người trong `heldout_people.csv` không được có trong dữ liệu train |

   Bộ trộn điểm (LR, trọng số ensemble…) cũng phải fit theo đúng luật trên. Ví dụ, với CV thì bộ trộn cũng fit không có fold k.
3. Ghi file CSV có 2 cột `trial_id,score`, điểm càng cao càng giống cùng người. Có thể chỉ chấm một phần, ví dụ chỉ MAV và EN.
4. Chạy:
   ```bash
   python score_local.py my_scores.csv               # thêm --sets MAV,MV,EN nếu muốn chỉ xem vài tập
   ```
   Kết quả mỗi dòng: EER của bạn, EER của sub90c, delta (âm là bạn tốt hơn), SE bootstrap theo người, P(better). Tập nào chỉ chấm một phần sẽ ghi `PARTIAL`, và được so trên đúng các trial bạn đã chấm.

Để kiểm tra: đưa chính cột `sub90c` vào làm file điểm thì mọi delta phải bằng 0 (đã chạy thử ngày 13/11).

---

## 3. Lưu ý quan trọng

- **Hệ XJ (train trên MAV v1 + v2 + VoxCeleb1):**
  - MAV và MV **không hợp lệ** với XJ nếu chưa bỏ 55 người MAV v1 trong `heldout_people.csv` khỏi dữ liệu train. Lý do: XJ đã train trên chính những người này.
  - MAV v1, v2 và v3 dùng chung tên `id0001…` nhưng là **người khác nhau**. Phải loại theo người của **v1**, không loại theo tên trên mọi bản.
  - Với CV và RES, XJ phải train lại theo fold (mục 2). EN và XL dùng được nếu những người đó không có trong dữ liệu train. Theo tài liệu chính thức thì người của VoxCeleb1 và VoxCeleb2 không trùng nhau.
- **Proxy local không dự đoán được English trên eval** (LEV-cal, 13/11, `outputs/ultra/desc/lev_cal.txt`):
  - Với 5 cặp bản của mình (88b→89b, 90a→90b, 90b→90c, 90c→91c, 90c→92a), mọi proxy đều có tương quan âm với delta English trên eval (r từ −0.14 đến −0.74).
  - sub91c và sub92a tốt hơn sub90c ở gần như mọi proxy (CV −0.3 đến −0.9, MV −0.4 đến −1.7), nhưng trên eval English lại tệ hơn từ +0.5 đến +2.4.
  - Nhiễu ghép cặp của eval khoảng 1.0–1.2 điểm mỗi ô English.
  - Nên dùng bộ này để bắt lỗi và để thấy khác biệt lớn (vài điểm, ổn định trên nhiều tập). Đừng coi nó là dự báo điểm eval.
- **Mức điểm:** sub90c đạt 16.84 / 30.29 trên eval English (no_gender / gender). Gần nhất là HARD (20.34 / 29.95) và MAV (18.19 / 26.37) ở protocol gender. Ở no_gender thì không tập nào khớp hẳn.
- Bộ này không chứa media hay feature của VoxCeleb, VoxBlink hay MAV-Celeb. Ai cần thì tải lại bằng tài khoản của mình. Dữ liệu FLAG chỉ chia sẻ trong team.

---

## 4. Chạy lại đánh giá local cho hệ của mình (khung `t3_final.py`)

Chạy từ `Track 3/scripts`, dùng feature và model có sẵn trong gói. Một spec là JSON `{"nf": ..., "r2": ..., "fuser": ...}`; sub90c là `{"nf": "D", "r2": "E1", "fuser": "two"}`.

```bash
python ../outputs/ultra/desc/multi.py '<spec>'            # CV, HARD, RES, MAV-ALL (+ tập con 15 người), MV, EN/XL rút gọn, so với sub90c
python t3_final.py gate_all '<spec>'                      # luật gate v2 đăng ký trước (PREREG_gate_all.txt), cho spec không train trên MAV
python ../outputs/ultra/desc/lev_cal.py                   # tái lập 88b…92a giống hệt từng byte, đối chiếu proxy với eval (khoảng 3 phút)
python ../release/sub90c/local_eval_kit/export_kit.py     # tạo lại trials.csv.gz và tự kiểm tra các số sub90c ở trên
```

Ngày 13/11 đã chạy cả 4 lệnh trong gói. `lev_cal.py` cho kết quả giống hệt bản ngày 13/11, và `export_kit.py` cho đúng các số sub90c ở mục 1.
