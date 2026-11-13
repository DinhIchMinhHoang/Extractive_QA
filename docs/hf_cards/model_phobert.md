---
license: agpl-3.0
language: [vi]
base_model: vinai/phobert-base-v2
datasets: [taidng/UIT-ViQuAD2.0]
pipeline_tag: question-answering
tags: [extractive-qa, vietnamese, unanswerable, phobert]
---
# viquad2-phobert

**EN summary.** Four fine-tuned seeds (s42-s45, fp16 safetensors) of `vinai/phobert-base-v2` for Vietnamese extractive QA with unanswerable questions. **Seeds 44 and 45 diverged during training** (eval loss jumped to ~2.7); they are published as evidence only and are **excluded from the ensemble** (ensemble = s42 + s43). Because PhoBERT-base-v2 is AGPL-3.0, these weights are released under **AGPL-3.0**. Input must be word-segmented (VnCoreNLP, not redistributed here).

## Bài toán
Hỏi đáp trích xuất tiếng Việt trên UIT-ViQuAD 2.0, gồm cả câu hỏi không có đáp án.

## Mô hình nền và giấy phép
`vinai/phobert-base-v2` (135.0M tham số), **AGPL-3.0** => các trọng số này cũng AGPL-3.0.

## Công thức huấn luyện (configs/sweep_release.json)
lr 1.5e-5, 5 epoch, seeds 42/43/44 (+45 chạy lại, xem configs/sweep_release_rerun.json); max_length 256, doc_stride 128, batch 32, warmup 0.1, weight decay 0.01, n_best 20, max_answer_length 64. Ngữ cảnh và câu hỏi được tách từ (VnCoreNLP) trước khi đưa vào mô hình; span được ánh xạ về ngữ cảnh gốc.

## Kết quả (notebooks/07_release_models.ipynb)
| | Dev F1 | Test EM | Test F1 | Test HasAns F1 | Test NoAns acc | Ghi chú |
|---|---|---|---|---|---|---|
| s42 (ngưỡng -2.00) | 68.33 | 58.18 | 68.10 | 69.00 | 66.06 | trong ensemble |
| s43 (ngưỡng -1.50) | 66.28 | 57.29 | 66.82 | 69.56 | 60.55 | trong ensemble |
| s44 (ngưỡng -1.00) | 62.78 | - | - | - | - | **phân kỳ**, loại |
| s45 (ngưỡng 0.00) | 60.86 | - | - | - | - | **phân kỳ**, loại |
| **Ensemble s42+s43 (ngưỡng -1.50)** | 68.19 | 59.07 | **68.62** | 71.58 | 61.84 | |

Ensemble gốc giai đoạn 3 (kết quả chính của bài báo): test F1 **68.97**; release thấp hơn 0.35 (CI [-0.88, 0.17], không có ý nghĩa). Test chỉ chấm một lần. Eval loss theo epoch: s42 [1.353, 1.159, 1.122, 1.112, 1.123]; s44 [1.431, 1.182, 2.722, 2.648, 2.649]; s45 [1.354, 1.206, 2.696, 2.655, 2.679].

## Cách dùng
`null_diff > threshold` => rỗng. Cần tách từ đầu vào: dùng `src.demo.Loaded` / `src.segment` của repo nghiên cứu (cần VnCoreNLP, tải riêng từ nguồn gốc).
Ensemble: `src.demo.answer_ensemble` với s42+s43, ngưỡng -1.5.

## Lưu ý (caveats)
- PhoBERT rất dễ vỡ theo seed: 2/4 seed phân kỳ ở epoch 3 (lr 1.5e-5); nên chọn checkpoint theo dev, không tin seed cố định.
- fp16 (~256 MiB/run); phương sai seed lớn; test chấm một lần.
- Kết quả chính của bài báo là ensemble gốc giai đoạn 3.

## Tệp
`phobert/phobert__release_s{42,43,44,45}/{config.json, model.safetensors, tokenizer.json, tokenizer_config.json}`, `best_models.json`.

## Trích dẫn
TODO: bài báo / UIT-ViQuAD 2.0 / PhoBERT (Nguyen & Nguyen, 2020).
