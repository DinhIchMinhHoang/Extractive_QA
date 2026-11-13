---
license: mit
language: [vi]
base_model: FacebookAI/xlm-roberta-large
datasets: [taidng/UIT-ViQuAD2.0]
pipeline_tag: question-answering
tags: [extractive-qa, vietnamese, unanswerable, xlm-roberta]
---
# viquad2-xlmr-large

**EN summary.** Three fine-tuned seeds (s42/s43/s44, fp16 safetensors) of `FacebookAI/xlm-roberta-large` for Vietnamese extractive QA with unanswerable questions (UIT-ViQuAD 2.0). Ensemble = mean of start/end logits of the 3 seeds. Best of the three released families.
Licence: XLM-R is MIT per its original model card (maintainer: please double-check before publishing).

## Bài toán
Hỏi đáp trích xuất tiếng Việt trên UIT-ViQuAD 2.0, gồm cả câu hỏi **không có đáp án**.

## Mô hình nền và giấy phép
`FacebookAI/xlm-roberta-large` (559.9M tham số), MIT theo model card gốc (cần kiểm tra lại).

## Công thức huấn luyện (configs/sweep_release.json)
lr 1.5e-5, 3 epoch, plausible_lambda 0.3, seeds 42/43/44; max_length 384, doc_stride 128, batch 16 x grad_accum 2, warmup 0.1, weight decay 0.01, n_best 20, max_answer_length 64.

## Kết quả (notebooks/07_release_models.ipynb)
| | Dev F1 | Test EM | Test F1 | Test HasAns F1 | Test NoAns acc |
|---|---|---|---|---|---|
| s42 (ngưỡng -1.50) | 77.76 | 66.60 | 76.38 | 75.92 | 77.43 |
| s43 (ngưỡng -0.75) | 77.20 | 66.73 | 76.78 | 76.27 | 77.95 |
| s44 (ngưỡng 1.75) | 77.51 | 67.59 | 77.59 | 78.68 | 75.11 |
| **Ensemble 3 seed (ngưỡng -0.25)** | 78.59 | 68.35 | **77.68** | 77.15 | 78.90 |

Ensemble gốc giai đoạn 3 (kết quả chính của bài báo): test F1 **77.89**; ensemble release thấp hơn 0.21, khác biệt không có ý nghĩa (CI [-0.86, 0.46]). Test chỉ chấm **một lần**.

## Cách dùng
`null_diff > threshold` => không có đáp án (rỗng).
```python
from src.qa_api import QA
qa = QA("xlmr_large/xlmr_large__release_s42", threshold=-1.5); qa.release = True
print(qa.answer(question, context))   # {"text", "score", "null_diff", "has_answer"}
```
Ensemble (mặc định của repo): `src.qa_api.load_qa()` (hoặc `src.demo.answer_ensemble`), ngưỡng -0.25; `load_qa(ensemble=False)` = mô hình đơn.

## Lưu ý (caveats)
- Trọng số fp16 (~1066 MiB/run); có thể lệch nhẹ so với fp32.
- Phương sai seed nhỏ hơn xlmr-base nhưng vẫn có (dev F1 77.2-77.8).
- Ngưỡng chỉnh trên dev; test chấm một lần.
- Kết quả chính của bài báo là ensemble gốc giai đoạn 3.

## Tệp
`xlmr_large/xlmr_large__release_s{42,43,44}/{config.json, model.safetensors, tokenizer.json, tokenizer_config.json}`, `best_models.json`.

## Trích dẫn
TODO: bài báo / UIT-ViQuAD 2.0 / XLM-R (Conneau et al., 2020).
