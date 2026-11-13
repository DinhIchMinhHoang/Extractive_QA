---
license: other
language: [vi]
task_categories: [question-answering]
pretty_name: ViQuAD2 extractive QA research archive
---
# viquad2-extractive-qa-archive

**EN summary.** Research archive for Vietnamese extractive QA (UIT-ViQuAD 2.0, incl. unanswerable questions): processed data, all sweep results/logits, and the notebook-generated models, kept as scientific evidence. The release weights (fp16) live in the model repos `<user>/viquad2-xlmr-base`, `<user>/viquad2-phobert`, `<user>/viquad2-xlmr-large` (not duplicated here).

## Nội dung và bố cục
- `data/processed/**`: dữ liệu đã làm sạch/chia tập (`uit_viquad2_clean`, `uit_viquad2_split`, `seg` = bản tách từ cho PhoBERT, `split_ids.json`, `stats.json`).
- `outputs/sweep/{stage1,stage2,stage3,zero_shot,release}`: kết quả quét siêu tham số, log, logit.
- `outputs/{xlmr_base,xlmr_large,phobert_base}` và `outputs/phobert_base_out.tar` (lượt lr 3e-5 phân kỳ, không có thư mục giải nén): mô hình sinh ra từ notebook 02/03/04 (giữ làm bằng chứng).
- `outputs/figures`, `outputs/release_small.tar`, `outputs/release_rerun_small.tar`: hình và gói nhỏ.
- `outputs/MANIFEST.json`, `data/MANIFEST.json`: sha256 + kích thước từng tệp.
- Không có ở đây: `outputs/release/**` (nằm ở 3 model repo), `outputs/release_dl` (bản trùng), `outputs/xlmr_base_out.tar`, `outputs/xlmr_large_out.tar`, `outputs/phobert_base_lr2e5_out.tar` (không thuộc kho; cùng `model.safetensors`, tokenizer và tệp summary với các thư mục giải nén `outputs/xlmr_base`, `outputs/xlmr_large`, `outputs/phobert_base`; vẫn còn trong repo cục bộ; `outputs/MANIFEST.json` vẫn liệt kê chúng nên bước kiểm tra sẽ báo thiếu 3 tệp này), `data/hf_cache`, `data/vncorenlp` (tệp bên thứ ba, tải từ nguồn gốc).

## Tái hiện notebook 05/06/07
Clone mã nguồn (repo nghiên cứu), tải kho này vào cùng thư mục gốc (`hf download <user>/viquad2-extractive-qa-archive --repo-type dataset --local-dir .`), tải trọng số từ 3 model repo vào `outputs/release/<model>/` (kèm `best_models.json`), rồi chạy `notebooks/05_compare_models.ipynb`, `06_error_analysis.ipynb`, `07_release_models.ipynb`. Notebook 07 cần `outputs/release/**`; 05/06 chỉ cần `outputs/sweep` và `data/processed`.

## Kiểm tra toàn vẹn
Mỗi `MANIFEST.json` có `files`: đường dẫn tương đối (so với thư mục chứa manifest) → `{"size", "sha256"}`. Kiểm tra thủ công bằng `sha256sum <tệp>` hoặc `Get-FileHash <tệp> -Algorithm SHA256` rồi so với manifest (bỏ qua `outputs/release/**` nếu chưa tải; ba tar đã gỡ sẽ báo thiếu). Script kiểm tra cũ đã xoá khỏi repo nguồn ngày 2026-11-14.

## Giấy phép dữ liệu
Dữ liệu dẫn xuất từ UIT-ViQuAD 2.0. Chủ dự án xác nhận đã được cấp phép đầy đủ để công bố; người dùng cần tuân thủ điều khoản gốc của UIT-ViQuAD 2.0. Điền giấy phép chính xác vào front matter trước khi công bố.

## Trích dẫn
TODO: bài báo; UIT-ViQuAD 2.0 (Nguyen et al.); XLM-R; PhoBERT.
