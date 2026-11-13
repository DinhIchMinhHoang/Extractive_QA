# Notebook figures index

Every image output (image/png) saved in `notebooks/01..06*.ipynb`, extracted unchanged (notebooks were not re-run) as `nb<NN>_c<cell>_<k>.png` (cell = 0-based index in the .ipynb cells list, k = order within the cell). Notebook 05 contains no image outputs (tables/text only). Notebook 06 was re-executed once on 2026-11-13 and now holds one PNG in cell 7 (`nb06_c7_1.png`).

Notebook 02 = xlmr_base, 03 = xlmr_large, 04 = phobert (single run, seed 42, per the notebook titles). Section = nearest markdown heading above the cell.

| file | notebook | cell | section | what it shows |
|---|---|---|---|---|
| nb01_c4_1.png | 01_data_eda | 4 | 1. Tổng quan dataset | Left: % is_impossible per split (train ~32%, validation ~30%, test 0 = unlabelled). Right: distribution of answers per question (0-3) per split; test is 100% with 0 answers. |
| nb01_c12_1.png | 01_data_eda | 12 | 3. Thống kê độ dài TRƯỚC khi làm sạch | 2x2 histograms: context length in XLM-R tokens (lines at 256/384, mostly <500), question length (peak 12-20 tokens), answer length in words (long tail, mode <10), relative answer start position (fairly flat, spike at 0). |
| nb01_c20_1.png | 01_data_eda | 20 | 5. SAU khi làm sạch: audit lại và so sánh | Per split (train/validation/test), log-scale count of rows per data-quality issue before vs after cleaning (offset_mismatch, double space, nfc, zero-width...). Fixed issues drop to 0; multiple_answers (validation) and answerable_without_answer (test) remain unchanged. |
| nb01_c23_1.png | 01_data_eda | 23 | 6. Phân tích bổ sung (train sau làm sạch) | Left: top-10 first words of questions (những, các, ai, vì, tại...). Right: top-10 first two-word phrases (vì sao, ai là, tại sao...). |
| nb01_c24_1.png | 01_data_eda | 24 | 6. Phân tích bổ sung (train sau làm sạch) | Boxplot of answer length (words): HasAns answers vs "plausible" answers of NoAns questions; both median ~6, similar spread. |
| nb01_c33_1.png | 01_data_eda | 33 | 9. Tập test không có nhãn và chia lại train/dev/test | After re-split: left NoAns % per split (train/dev/test all ~30-33%); right row counts per split (train ~25.5k, dev ~2.9k, test ~3.8k). |
| nb06_c7_1.png | 06_error_analysis | 7 | (error-category chart cell) | xlmr_large error categories on test (n_err=1207): (a) NoAns but answered 22.4%, (b) HasAns but empty 26.2%, (c) span overlap/boundary 39.6%, (d) wrong place 11.8%. Horizontal bars, 4 top-level categories only (no sub-categories). |
| nb02_c11_1.png | 02_finetune_xlmr | 11 | 5. Đường cong học | Training loss (every 50 steps) falling from ~5.9 to ~1.1, and per-epoch eval loss (3 points, ~1.4 to ~1.25), over ~2560 steps. xlmr_base. |
| nb02_c15_1.png | 02_finetune_xlmr | 15 | 7. Tài nguyên | Threshold sweep on dev: F1, HasAns F1, NoAns acc vs no-answer threshold (-10..10) with the chosen best-threshold line (~-1). Heading is nearest markdown above, but content belongs to the dev-evaluation/threshold step. xlmr_base. |
| nb03_c11_1.png | 03_finetune_xlmr_large | 11 | 5. Đường cong học | Training loss (~5.6 to ~0.7) and per-epoch eval loss (~1.0 to ~0.9) over ~2560 steps. xlmr_large. |
| nb03_c13_1.png | 03_finetune_xlmr_large | 13 | 6. Đánh giá dev và chọn ngưỡng | Dev threshold sweep (-11..15): F1 peaks near ~78 around threshold ~-0.6 (red line); HasAns F1 rises to ~84, NoAns acc falls to ~15. |
| nb04_c11_1.png | 04_finetune_phobert | 11 | 5. Đường cong học | Training loss (~5.4 to ~1.0) and per-epoch eval loss (~1.3 to ~1.1) over ~2730 steps. phobert. |
| nb04_c13_1.png | 04_finetune_phobert | 13 | 6. Đánh giá dev và chọn ngưỡng | Dev threshold sweep (-10..10): F1 peaks ~67 near threshold ~-1.5 (red line); HasAns F1 to ~80, NoAns acc falls to ~3. |

## Flags

- All 12 images open and are readable; no duplicates (distinct content and byte sizes).
- nb02_c15_1: section heading ("7. Tài nguyên") does not match the content (threshold sweep); the plot is the same kind as nb03_c13_1 / nb04_c13_1.
- nb01_c4_1 (test bar = 0, 100% zero answers) reflects the original unlabelled test split, before the re-split shown in nb01_c33_1.
- Single-seed plots only (seed 42 style runs of notebooks 02-04). No notebook figure exists for the 3-seed ensembles, test CIs or the error breakdown; the values for those (notebooks 05 and 06) are printed as tables and are parsed by `docs/make_figures.py`.

## Generated figures (docs/make_figures.py) vs notebook figures

| generated | equivalent notebook figure |
|---|---|
| fig01_stage1_hparams | NONE (sweep results.jsonl; nb05 has the table only) |
| fig02_stage2_addons | NONE (nb05 has the ablation table only) |
| fig03_seeds_dev | NONE |
| fig04_test_f1_ci | NONE (nb05 prints the CI table, no plot) |
| fig05_dev_vs_test | NONE (nb05 prints the dev/test gap table, no plot) |
| fig06_hasans_vs_noans | NONE (numbers in the nb05 CI table; threshold trade-off is only visible per model in nb02_c15_1 / nb03_c13_1 / nb04_c13_1) |
| fig07_error_categories | nb06_c7_1: same data (1207 errors); fig07 is re-drawn with sub-category bars (super-/sub-span, shifted, plausible match, same/other sentence) |
| fig08_cost_vs_f1 | NONE |
| fig09_training_curve | PARTIAL: nb02_c11_1 (xlmr_base), nb03_c11_1 (xlmr_large), nb04_c11_1 (phobert) are the same curves per model, as separate x=step plots; fig09 overlays the 3 models by epoch from the final stage-3 seed-42 runs, so not identical runs. |
