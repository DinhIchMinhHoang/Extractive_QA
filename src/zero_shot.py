"""Zero-shot inference of nguyenvulebinh/vi-mrc-{base,large}: logits in the usual run format.

python -m src.zero_shot --model vi_mrc_large [--smoke]   (dev scored; test only stored, never scored)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import src.evalkit as ek
from src.preprocess import prepare_eval_features
from src.train_qa import ROOT, get_data, save_logits, score_from_logits

MODELS = {"vi_mrc_base": "nguyenvulebinh/vi-mrc-base", "vi_mrc_large": "nguyenvulebinh/vi-mrc-large"}
NOTES = ("pretrained on SQuAD2/mailong25/VLSP-MRC-2021/MLQA; dev may overlap training "
         "(ViQuAD2.0 = VLSP 2021); subword path, not author's sum-features")


def load_model(repo, device="cuda"):
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo)
    model, info = AutoModelForQuestionAnswering.from_pretrained(repo, output_loading_info=True)
    bad = [k for k in info["missing_keys"] if "position_ids" not in k]
    assert not bad, f"head/encoder not loaded (random init): {bad[:5]}"
    assert tok.padding_side == "right" and tok.cls_token_id == 0, (tok.padding_side, tok.cls_token_id)
    model = model.half().to(device).eval() if device == "cuda" else model.to(device).eval()
    return tok, model


def build_split_features(ds_split, tok, max_length=384, doc_stride=128):
    return ds_split.map(lambda b: prepare_eval_features(b, tokenizer=tok, max_length=max_length,
                                                        doc_stride=doc_stride),
                        batched=True, remove_columns=ds_split.column_names)


def predict_logits_hf(model, feats, bs=16, device="cuda"):
    """(start, end) float32 numpy [n_windows, max_length], window order kept."""
    import torch

    ids = np.asarray(feats["input_ids"]); mask = np.asarray(feats["attention_mask"])
    s_all, e_all, i = [], [], 0
    while i < len(ids):
        try:
            with torch.inference_mode():
                out = model(input_ids=torch.as_tensor(ids[i:i + bs]).to(device),
                            attention_mask=torch.as_tensor(mask[i:i + bs]).to(device))
            s_all.append(out.start_logits.float().cpu().numpy()); e_all.append(out.end_logits.float().cpu().numpy())
            i += bs
        except torch.cuda.OutOfMemoryError:
            if bs == 1:
                raise
            torch.cuda.empty_cache(); bs = max(1, bs // 2)
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(s_all).astype(np.float32), np.concatenate(e_all).astype(np.float32)


def _xlmr_hash():
    p = next(iter(sorted((ROOT / "outputs" / "sweep" / "stage3" / "xlmr_base").glob("*/dev_logits.npz"))), None)
    return ek._load_npz(p)["feature_hash"] if p else None


def run(name, out_root, max_n=None, save_test=True, bs=16, device="cuda", ds=None):
    import torch

    repo = MODELS.get(name, name)
    out = Path(out_root) / name / f"{name}__zs"
    out.mkdir(parents=True, exist_ok=True)
    tok, model = load_model(repo, device)
    cuda = device == "cuda"
    cfg = dict(model_name=repo, max_length=384, doc_stride=128, n_best=20, max_answer_length=64, eval_bs=bs,
               seed=None, run_name=f"{name}__zs", zero_shot=True, dtype="float16" if cuda else "float32",
               max_dev=max_n, lr=None, epochs=None, lr_decay=None, ema_decay=None, plausible_lambda=0,
               train_bs=None)
    gpu = ({"name": torch.cuda.get_device_name(0),
            "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9}
           if cuda else {"name": "cpu", "total_memory_gb": None})
    ds = ds or get_data()
    pick = lambda sp: ds[sp].select(range(max_n)) if max_n else ds[sp]  # noqa: E731
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    dev_ex = pick("dev")
    dev_f = build_split_features(dev_ex, tok, cfg["max_length"], cfg["doc_stride"])
    ds_, de_ = predict_logits_hf(model, dev_f, bs, device)
    dev = score_from_logits(dev_ex, dev_f, ds_, de_, cfg)
    save_logits(out / "dev_logits.npz", dev_f, ds_, de_)
    print("dev @0:", {k: round(v, 2) for k, v in dev["metrics_at_0"].items()})
    print(f"dev @{dev['best_threshold']:.3f}:", {k: round(v, 2) for k, v in dev["metrics_at_best"].items()})
    n_test = None
    test_hash = None
    if save_test:  # test is only stored, never scored
        test_f = build_split_features(pick("test"), tok, cfg["max_length"], cfg["doc_stride"])
        ts, te = predict_logits_hf(model, test_f, bs, device)
        save_logits(out / "test_logits.npz", test_f, ts, te)
        n_test, test_hash = len(test_f), ek.feature_hash(test_f["example_id"])
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if cuda else None
    (out / "dev_threshold.json").write_text(json.dumps(
        {"threshold": dev["best_threshold"], "table": dev["table"], "metrics_at_0": dev["metrics_at_0"],
         "metrics_at_best": dev["metrics_at_best"]}, indent=1))
    h_dev = ek.feature_hash(dev_f["example_id"])
    ref = _xlmr_hash()
    if ref is not None and ref != h_dev:
        print("WARNING: dev features differ from xlmr_base stage3 features")
    summary = {"cfg": cfg, "gpu": gpu, "wall_s": wall, "peak_mem_gb": peak,
               "windows": {"dev": len(dev_f), "test": n_test, "dev_examples": len(dev_ex)}, "dev": dev,
               "feature_hash_dev": h_dev, "feature_hash_test": test_hash, "notes": NOTES,
               "same_features_as_xlmr_base": None if ref is None else ref == h_dev}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=[*MODELS, "all"])
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    root = a.out_root or str(ROOT / "outputs" / "sweep" / ("zero_shot_smoke" if a.smoke else "zero_shot"))
    import torch
    for n in (MODELS if a.model == "all" else [a.model]):
        print("==", n)
        run(n, root, max_n=32 if a.smoke else None, save_test=not a.no_test, bs=a.bs,
            device="cuda" if torch.cuda.is_available() else "cpu")
