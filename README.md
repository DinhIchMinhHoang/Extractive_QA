# Extractive_QA — Trích xuất câu trả lời tiếng Việt

Cho câu hỏi tiếng Việt và một đoạn văn ngữ cảnh, mô hình trả về đoạn trích trong ngữ cảnh làm câu trả lời,
hoặc "không có câu trả lời" (kiểu SQuAD 2.0). Đây là một phần của [RAG_official](https://github.com/DinhIchMinhHoang/RAG_official).

## Giao thức dữ liệu

- Dữ liệu: UIT-ViQuAD 2.0. Chỉ dùng phần **train** gốc của UIT-ViQuAD để huấn luyện.
- Phần train gốc được chia lại thành **train' / dev / test**. Dev dùng để chọn siêu tham số và ngưỡng
  "không có câu trả lời"; test chỉ dùng để báo cáo kết quả cuối.
- Số mẫu test: 3.814.
- Chỉ số: EM, F1 (kiểu SQuAD 2.0), NoAns (độ chính xác trên câu không có đáp án).

## Mô hình

- `xlmr_base`, `xlmr_large`: fine-tune XLM-R (base / large).
- `phobert`: fine-tune PhoBERT base.
- Kết quả "ens" là ensemble của các lượt chạy trong sweep.
- **Model phát hành** (huấn luyện lại công thức đóng băng, có lưu trọng số): xem mục "Model phát hành" bên dưới.
  Bảng kết quả TEST cuối vẫn là kết quả gốc của stage 3; bảng model phát hành là một bản tái lập riêng.
- Tham chiếu `vi-mrc-large`, `vi-mrc-base`: mô hình có sẵn, đánh giá zero-shot, không do dự án này huấn luyện.

## Cấu trúc repo

- `src/` — mã nguồn: `data_clean.py`, `data_split.py`, `preprocess.py` (cửa sổ trượt), `postprocess.py`
  (N-best + ngưỡng null), `metrics.py`, `evalkit.py`, `train_qa.py`, `sweep.py`, `phobert.py`, `segment.py`,
  `zero_shot.py`, `qa_api.py` (hàm `answer`).
- `notebooks/` — `01_data_eda`, `02_finetune_xlmr`, `03_finetune_xlmr_large`, `04_finetune_phobert`, `05_compare_models`,
  `06_error_analysis`, `07_release_models` (đánh giá model phát hành; hiện có 7 notebook, đánh số 01-07).
- `configs/` — `sweep_stage1.json`, `sweep_stage2.json`, `sweep_stage3.json`, `sweep_release.json`, `sweep_release_rerun.json`.
- `tests/` — test chạy bằng pytest (`python -m pytest tests -q`): 62 passed, 0 skipped (chạy toàn bộ ngày 2026-11-14, 262 giây; sau khi xoá 11 test của công cụ đã xoá).
- `docs/ARCHIVE.md` — bố cục lưu trữ trên Hugging Face; `docs/hf_cards/` — thẻ model/dataset.
- `docs/BAO_CAO.md` — báo cáo đầy đủ (quy trình, kết quả, đối chiếu số liệu).
- `outputs/` — kết quả chạy (logit, tóm tắt, hình, các file `*.tar` của mô hình cũ). `outputs/release/` chứa trọng số
  phát hành (fp16 safetensors) và `best_models.json`; logit/summary của lượt release ở `outputs/sweep/release/`.

## Cách chạy

```bash
# môi trường (Windows)
.venv\Scripts\python -m pip install -r requirements.txt

# test
.venv\Scripts\python -m pytest tests -q

# sweep: xem src/sweep.py và configs/sweep_stage*.json
```

Các notebook `02`-`04` được dùng để fine-tune (trên Colab GPU); `05` so sánh các mô hình; `07` đánh giá model phát hành.

## Kết quả TEST cuối (3.814 mẫu)

| Mô hình | F1 | EM | NoAns | F1 95% CI |
|---|---|---|---|---|
| xlmr_base (ens) | 62.40 | 53.99 | 58.4 | 60.89-63.78 |
| phobert (ens) | 68.97 | 59.49 | 63.9 | 67.63-70.33 |
| xlmr_large (ens) | **77.89** | 68.35 | **76.7** | 76.68-79.15 |
| vi-mrc-large (tham chiếu) | 80.48 | 68.90 | 66.8 | 79.33-81.63 |
| vi-mrc-base (tham chiếu) | 64.72 | 57.26 | 35.7 | - |

Nguồn: `notebooks/05_compare_models.ipynb`, cell 19.

### Lưu ý về tính công bằng

`vi-mrc` thuộc nhóm B: được huấn luyện với dữ liệu bổ sung và dev của ta bị nhiễm (contaminated):
F1 trên dev là 89.74 so với 80.48 trên test. Vì vậy so sánh `vi-mrc` với các mô hình của dự án
(chỉ dùng train của UIT-ViQuAD) là không công bằng; chỉ coi là mốc tham chiếu.

### Giấy phép

PhoBERT dùng giấy phép AGPL-3.0; cần xem xét khi dùng lại hoặc phân phối. Trọng số PhoBERT đã fine-tune cũng theo AGPL-3.0.
XLM-R: xem thẻ model gốc (chưa kiểm chứng trong repo).

## Model phát hành

Release = huấn luyện lại đúng công thức đóng băng của stage 3 (seed 42/43/44) nhưng **có lưu trọng số** (fp16 safetensors).
Cùng dữ liệu, cùng metric, cùng hậu xử lý `min/64/20`; ngưỡng no-answer tune trên dev. Mô hình đơn tốt nhất chọn theo F1 dev
trong các thành viên ensemble; test chỉ chấm một lần (2026-11-14). Bảng dưới là **bản tái lập riêng**, không thay thế bảng TEST cuối ở trên.

| Mô hình | Mô hình đơn (run) | Dev F1 đơn | Test F1 đơn [CI 95%] | Ensemble (thành viên) | Dev F1 ens [CI 95%] | Test F1 ens [CI 95%] | Dung lượng mỗi run (fp16) |
|---|---|---|---|---|---|---|---|
| xlmr_base | `xlmr_base__release_s43` | 63.96 | 63.14 [61.64, 64.66] | s42, s43, s44 | 63.33 [61.75, 64.92] | 63.59 [62.11, 65.00] | 529 MiB |
| phobert | `phobert__release_s42` | 68.33 | 68.10 [66.67, 69.52] | s42, s43 | 68.19 [66.64, 69.70] | 68.62 [67.22, 70.01] | 256 MiB |
| xlmr_large | `xlmr_large__release_s42` | 77.76 | 76.38 [75.14, 77.70] | s42, s43, s44 | 78.59 [77.21, 79.90] | 77.68 [76.47, 78.99] | 1066 MiB |

Nguồn: `notebooks/07_release_models.ipynb` (cell 10, 11, 13, 17, 21) và `outputs/release/best_models.json`.

- So với ensemble gốc của stage 3 (62.40 / 68.97 / 77.89), hiệu F1 test release là +1.19 [0.44, 1.89], -0.35 [-0.88, 0.17],
  -0.21 [-0.86, 0.46]; chỉ `xlmr_base` khác biệt có ý nghĩa (notebook 07, cell 19).
- Vị trí: `outputs/release/<mô hình>/<run>/` (`model.safetensors`, `config.json`, tokenizer) và `outputs/release/best_models.json`
  (mô hình đơn, thành viên ensemble, ngưỡng dev). Thư mục có thêm các run bị loại (PhoBERT s44, s45) làm bằng chứng.
- Trọng số trên Hugging Face:
  [`viquad2-xlmr-base`](https://huggingface.co/DinhIchMinhHoang/viquad2-xlmr-base),
  [`viquad2-phobert`](https://huggingface.co/DinhIchMinhHoang/viquad2-phobert) (4 seed; s44, s45 phân kỳ, có ghi nhãn),
  [`viquad2-xlmr-large`](https://huggingface.co/DinhIchMinhHoang/viquad2-xlmr-large).
  Để demo tìm thấy, tải mỗi repo vào `outputs/release/`, ví dụ
  `hf download DinhIchMinhHoang/viquad2-xlmr-base --local-dir outputs/release`, hoặc trong Python
  `huggingface_hub.snapshot_download("DinhIchMinhHoang/viquad2-xlmr-base", local_dir="outputs/release")`.
  Theo thẻ model (`docs/hf_cards/model_*.md`, mục "Tệp"), mỗi repo chứa thư mục `<họ>/<họ>__release_sNN/` và `best_models.json`,
  nên kết quả là `outputs/release/<họ>/<họ>__release_sNN/`. 

## Lưu trữ nghiên cứu

Toàn bộ `data/processed` và `outputs/` (logit, tóm tắt sweep, mô hình giải nén từ notebook, hình, `outputs/phobert_base_out.tar`)
được lưu ở dataset [`viquad2-extractive-qa-archive`](https://huggingface.co/datasets/DinhIchMinhHoang/viquad2-extractive-qa-archive)
(322 tệp, đã công khai ngày 2026-11-14). Bố cục, kích thước, lệnh và các tệp cố ý không đưa lên: xem [`docs/ARCHIVE.md`](docs/ARCHIVE.md).
- **PhoBERT dễ vỡ ổn định.** Trong 4 lượt release, seed 44 và 45 phân kỳ ở epoch 3 (`eval_loss` ~1,2 lên ~2,7; dev F1 62.78 / 60.86,
  checkpoint khôi phục từ epoch 2); seed 43 có đợt tăng vọt muộn (grad_norm 254.8 ở bước 4400; F1 66.28); chỉ seed 42 sạch (68.33).
  Ở stage 3 cùng công thức cho 68.31 / 66.45 / 68.41 không phân kỳ. **Chưa xác định nguyên nhân.** Ensemble PhoBERT phát hành
  chỉ gồm seed 42 và 43 (loại 44, 45 theo dev và log huấn luyện, không theo test).
- Trọng số lưu fp16: argmax khớp 99,0-100% với logit đã lưu trên 100-300 ví dụ dev (chưa kiểm cho PhoBERT), nên điểm có thể lệch nhẹ so với `summary.json`.

## Phân tích lỗi

Notebook `06_error_analysis.ipynb` phân tích lỗi trên test của XLM-R large ensemble: 1207/3814 lỗi (31.65%).

- Không trả lời được nhưng model vẫn trả lời: 22.4% lỗi (143/270 trùng plausible answer).
- Có đáp án nhưng model trả rỗng: 26.2% lỗi.
- Lệch ranh giới span: 39.6% lỗi (super-span 256, sub-span 213, lệch 9).
- Sai vị trí: 11.9% lỗi.
- Nhóm quyết định có/không đáp án chiếm 48.5% lỗi.

## Chi phí tính toán (Colab)

GPU A100 tốn khoảng 6.73 đơn vị tính toán/giờ. Cộng `est_units` các lượt thành công trong `outputs/sweep/*/*/results.jsonl`:
stage 1 = 12.59, stage 2 = 12.19, stage 3 = 9.18 (tổng 3 giai đoạn 33.95), release (10 lượt) = 9.71; tổng 43.66
(chi tiết ở `docs/BAO_CAO.md`). Đây là ước tính, chưa gồm lượt mất của stage 3 và các phiên Colab khác.

## Dùng thử: `answer()`

Mặc định `load_qa()` nạp **ensemble 3 seed** của XLM-R large phát hành (s42/43/44, ngưỡng ensemble, đọc từ
`outputs/release/best_models.json`); `load_qa(ensemble=False)` nạp mô hình đơn tốt nhất (`xlmr_large__release_s42`, nhanh
hơn khoảng 3 lần). Mô hình cũ (XLM-R large của notebook 03, `outputs/xlmr_large_out.tar`) vẫn được giữ làm
bằng chứng khoa học và dùng bằng `load_qa(legacy=True)`; với bản cũ cần giải nén một lần:

```bash
tar -xf outputs/xlmr_large_out.tar -C outputs --one-top-level=xlmr_large
```

```python
from src.qa_api import load_qa, answer

qa = load_qa()  # lazy; mặc định = ensemble xlmr_large release (3 seed); load_qa(ensemble=False) = mô hình đơn s42;
                # load_qa(legacy=True) = mô hình notebook cũ (bỏ qua ensemble)
# fp16 nếu có CUDA, ngược lại CPU fp32
r = answer("Thủ đô của Việt Nam là gì?", "Hà Nội là thủ đô của Việt Nam.", qa=qa)
# {'text': 'Hà Nội', 'score': ..., 'null_diff': ..., 'has_answer': True, 'n_models': 3}  # n_models chỉ có ở ensemble
```

- Ngưỡng mặc định đọc từ `dev_threshold.json` (khóa `threshold`, chọn trên dev); có thể truyền `threshold=`.
- Ngữ cảnh dài hơn một cửa sổ (384 token, stride 128) được xử lý bằng cửa sổ trượt; điểm null là
  min trên các cửa sổ của `start[0]+end[0]`.
- `has_answer = null_diff <= threshold`; nếu không có đáp án thì `text` là `""`.
- Ensemble chạy lần lượt từng seed trên GPU (giải phóng giữa các seed, vừa 6 GB VRAM), trung bình logit start/end trên
  cùng cửa sổ rồi chấm một lần. Trọng số các seed được cache ở RAM CPU (fp16, LRU; `src.qa_api.ModelCache`, dùng chung với
  demo): mỗi lần chỉ chuyển một seed lên GPU rồi trả về CPU. Ngân sách `QA_CACHE_GB` (mặc định `min(6, 40% RAM)`); đặt
  `QA_CACHE=0` để tắt (nạp lại từ đĩa mỗi lần gọi, ~12-16 s/lần). Cần nhanh nhất thì dùng `ensemble=False`.

## Demo: so sánh 5 mô hình trên terminal

```bash
python -m src.demo                 # chế độ tương tác, gõ /help
python -m src.demo --list          # trạng thái các mô hình
python -m src.demo --single        # nhóm A dùng mô hình đơn tốt nhất (nhanh hơn; mặc định là ensemble)
python -m src.demo --legacy        # nhóm A dùng mô hình notebook cũ
python -m src.demo --models xlmr_large,vi_mrc_large --context-file ctx.txt --question "Thủ đô của Việt Nam là gì?"
```

- Mô hình: `xlmr_base`, `xlmr_large`, `phobert` (nhóm A, fine-tune của nhóm) và `vi_mrc_base`, `vi_mrc_large`
  (tham chiếu nhóm B, pretrained thêm dữ liệu ngoài, đánh dấu `*`).
- Mặc định nhóm A dùng **ensemble của bản phát hành** (nhãn `tên (ens×N)`): trung bình logit các run thành viên
  (xlmr_base s42/43/44, phobert s42/43, xlmr_large s42/43/44) với ngưỡng ensemble tune trên dev, đọc từ
  `outputs/release/best_models.json`; chậm hơn khoảng N lần. `--single` (hoặc `/single`, `/ensemble off` trong REPL) dùng
  mô hình đơn tốt nhất (xlmr_base s43, phobert s42, xlmr_large s42; chọn theo F1 dev); `--ensemble` giữ để tương thích
  (không còn tác dụng). `--legacy` dùng các mô hình notebook cũ
  (file `.tar` được giải nén vào `outputs/` ở lần dùng đầu; giữ làm bằng chứng, không bị xóa).
- Lệnh: `/models`, `/use <tên,...|all>`, `/context [@file.txt]`, `/sample`, `/threshold <mô hình> <giá trị>`, `/ensemble on|off`, `/single`, `/legacy on|off`, `/help`, `/quit`.
  Dòng khác là câu hỏi; kết quả là bảng đáp án / `null_diff` / thời gian, mô hình đồng thuận được tô xanh.
- PhoBERT cần Java + `data/vncorenlp`; thiếu thì được báo là không khả dụng, các mô hình còn lại vẫn chạy.
- Mỗi lần chỉ một mô hình trên GPU (fp16), nạp rồi giải phóng sau mỗi câu hỏi (vừa GPU 6GB; đổi lại tốn thời gian nạp).
- Màu tắt khi không phải TTY hoặc có biến `NO_COLOR`.
