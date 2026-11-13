---
license: mit
language: [vi]
base_model: FacebookAI/xlm-roberta-base
datasets: [taidng/UIT-ViQuAD2.0]
pipeline_tag: question-answering
tags: [extractive-qa, vietnamese, unanswerable, xlm-roberta]
---
# viquad2-xlmr-base

**EN summary.** Three fine-tuned seeds (s42/s43/s44, fp16 safetensors) of `FacebookAI/xlm-roberta-base` for Vietnamese extractive QA with unanswerable questions (UIT-ViQuAD 2.0). Ensemble = mean of start/end logits of the 3 seeds. Weights are research artifacts; read the caveats below.
Licence: XLM-R is MIT per its original model card (maintainer: please double-check before publishing).

## Bài toán
Hỏi đáp trích xuất (extractive QA) tiếng Việt trên UIT-ViQuAD 2.0, gồm cả câu hỏi **không có đáp án**. Mô hình chọn một đoạn trong ngữ cảnh hoặc trả về rỗng.

## Mô hình nền và giấy phép
`FacebookAI/xlm-roberta-base` (277.5M tham số), giấy phép MIT theo model card gốc (cần kiểm tra lại trước khi công bố).

## Công thức huấn luyện (configs/sweep_release.json)
lr 2e-5, 5 epoch, plausible_lambda 0.3, seeds 42/43/44; max_length 384, doc_stride 128, batch 32, warmup 0.1, weight decay 0.01, n_best 20, max_answer_length 64. Chọn epoch/ngưỡng trên dev.

## Kết quả (notebooks/07_release_models.ipynb)
| | Dev F1 | Test EM | Test F1 | Test HasAns F1 | Test NoAns acc |
|---|---|---|---|---|---|
| s42 (ngưỡng 0.50) | 61.53 | 52.99 | 61.28 | 61.78 | 60.12 |
| s43 (ngưỡng 0.50) | 63.96 | 54.33 | 63.14 | 63.48 | 62.36 |
| s44 (ngưỡng 0.75) | 60.32 | 51.05 | 61.10 | 66.80 | 48.06 |
| **Ensemble 3 seed (ngưỡng 1.00)** | 63.33 | 54.46 | **63.59** | 67.40 | 54.87 |

Ensemble gốc của giai đoạn 3 (kết quả chính của bài báo): test F1 **62.40**; ensemble release hơn +1.19 (bootstrap ghép cặp, 95% CI [0.44, 1.89]). Tập test chỉ được chấm **một lần**. Mô hình đơn tốt nhất theo dev: s43 (xem `best_models.json`).

## Cách dùng
`null_diff > threshold` => không có đáp án (trả về rỗng). `null_diff` = điểm null (min qua các cửa sổ của start[0]+end[0]) trừ điểm span tốt nhất.
```python
from src.qa_api import QA   # mã nguồn: repo nghiên cứu
qa = QA("xlmr_base/xlmr_base__release_s43", threshold=0.5); qa.release = True
print(qa.answer(question, context))   # {"text", "score", "null_diff", "has_answer"}
```
Ensemble: `src.demo.answer_ensemble` (trung bình logit rồi chấm điểm một lần, ngưỡng 1.0).

## Lưu ý (caveats)
- Trọng số đã chuyển sang fp16 (~529 MiB/run); điểm có thể lệch rất nhỏ so với fp32 lúc huấn luyện.
- Phương sai theo seed lớn: dev F1 60.3-64.0 giữa các seed.
- Ngưỡng được chỉnh trên dev, không phải trên test.
- Kết quả chính của bài báo là ensemble gốc giai đoạn 3, không phải bảng này.

## Tệp
`xlmr_base/xlmr_base__release_s{42,43,44}/{config.json, model.safetensors, tokenizer.json, tokenizer_config.json}`, `best_models.json` (đầy đủ cho cả 3 họ mô hình; ngưỡng ensemble nằm ở `ensembles.ensemble_thresholds`).

## Trích dẫn
TODO: bài báo / UIT-ViQuAD 2.0 (Nguyen et al.) / XLM-R (Conneau et al., 2020).
