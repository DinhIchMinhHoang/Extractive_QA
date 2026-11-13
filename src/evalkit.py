"""CPU-only evaluation from saved logits: re-scoring variants, seed ensembles, bootstrap stats, report tables.

Run dirs: outputs/sweep/<stage>/<model>/<run_name>/{summary.json, dev_logits.npz, [test_logits.npz]}.
"""
import hashlib
import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import exact_match, f1_score
from src.postprocess import apply_threshold, score_examples, tune_threshold
from src.metrics import squad_v2_metrics

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("evalkit")

NULL_VARIANTS = ("min", "mean", "first")
GRID = dict(null_variant=NULL_VARIANTS, max_answer_length=(30, 64, 96), n_best=(20, 50))
DEFAULT_VARIANT = ("min", 64, 20)
# approximate parameter counts (QA head included); not stored in summary.json
PARAMS_M = {"xlmr_base": 277.5, "phobert": 135.0, "xlmr_large": 559.9,
            "vi_mrc_base": 277.5, "vi_mrc_large": 559.9}


def _warn_test():
    log.warning("TEST SPLIT SCORED")
    print("WARNING: TEST SPLIT SCORED")


def _check_split(split, allow_test):
    if split not in ("dev", "test"):
        raise ValueError(f"split must be 'dev' or 'test', got {split!r}")
    if split == "test":
        if not allow_test:
            raise PermissionError("split='test' requires allow_test=True")
        _warn_test()


def feature_hash(example_ids):
    return hashlib.sha1("\n".join(map(str, example_ids)).encode()).hexdigest()


def _load_npz(path):
    z = np.load(path)
    return {"start": z["start_logits"].astype(np.float32), "end": z["end_logits"].astype(np.float32),
            "example_id": [str(x) for x in z["example_id"]], "feature_hash": str(z["feature_hash"])}


def load_run(run_dir, allow_test=False):
    """dict(cfg, summary, dev={start,end,example_id,feature_hash}, test=... only if allow_test and file exists)."""
    run_dir = Path(run_dir)
    summ = json.loads((run_dir / "summary.json").read_text())
    out = {"run_dir": run_dir, "cfg": summ["cfg"], "summary": summ, "dev": _load_npz(run_dir / "dev_logits.npz")}
    if allow_test and (run_dir / "test_logits.npz").exists():
        _warn_test()
        out["test"] = _load_npz(run_dir / "test_logits.npz")
    return out


def load_runs(root=None):
    """DataFrame, one row per run under <root>/sweep/<stage>/<model>/<run_name>/ (root default: outputs/)."""
    root = Path(root) if root else ROOT / "outputs"
    rows = []
    for p in sorted(root.glob("sweep/*/*/*/summary.json")):
        s = json.loads(p.read_text())
        cfg, dev = s.get("cfg", {}), s.get("dev", {})
        m, m0 = dev.get("metrics_at_best", {}), dev.get("metrics_at_0", {})
        w = s.get("windows", {})
        rows.append({
            "stage": p.parents[2].name, "model": p.parents[1].name, "run_name": p.parent.name,
            "run_dir": str(p.parent), "seed": cfg.get("seed"), "lr": cfg.get("lr"), "epochs": cfg.get("epochs"),
            "train_bs": cfg.get("train_bs"), "lr_decay": cfg.get("lr_decay"), "ema_decay": cfg.get("ema_decay"),
            "plausible_lambda": cfg.get("plausible_lambda") or 0.0, "max_answer_length": cfg.get("max_answer_length"),
            "n_best": cfg.get("n_best"), "threshold": dev.get("best_threshold"),
            "dev_f1": m.get("f1"), "dev_em": m.get("exact"), "dev_hasans_f1": m.get("HasAns_f1"),
            "dev_noans_acc": m.get("NoAns_exact"), "dev_f1_at0": m0.get("f1"),
            "train_runtime_s": s.get("train_runtime_s"), "est_units": s.get("est_units"),
            "train_windows": w.get("train"), "dev_windows": w.get("dev"),
            "has_test_logits": (p.parent / "test_logits.npz").exists()})
    return pd.DataFrame(rows)


# ---------- features for rescoring ----------

def build_eval_context(model_key, split="dev", max_n=None, allow_test=False, expected_hash=None):
    """(examples, eval_features, extract_fn_or_None) with the training-time pipeline.

    model_key: 'xlmr_base' | 'xlmr_large' | 'phobert' | 'vi_mrc_base' | 'vi_mrc_large'. max_n: first N examples (smoke runs, cfg max_dev).
    """
    _check_split(split, allow_test)
    if model_key == "phobert":
        from src.phobert import get_seg_data, load_phobert_tokenizer, make_extract
        tok, ds, max_length = load_phobert_tokenizer(), get_seg_data(), 256
    elif model_key in ("xlmr_base", "xlmr_large"):
        from transformers import AutoTokenizer
        from src.train_qa import get_data
        tok = AutoTokenizer.from_pretrained("xlm-roberta-" + model_key.split("_")[1])
        ds, max_length = get_data(), 384
    elif model_key in ("vi_mrc_base", "vi_mrc_large"):
        from transformers import AutoTokenizer
        from src.train_qa import get_data
        tok = AutoTokenizer.from_pretrained("nguyenvulebinh/vi-mrc-" + model_key.split("_")[2])
        ds, max_length = get_data(), 384
    else:
        raise ValueError(f"unknown model_key {model_key!r}")
    from src.preprocess import prepare_eval_features
    ex = ds[split].select(range(max_n)) if max_n else ds[split]
    feats = ex.map(lambda b: prepare_eval_features(b, tokenizer=tok, max_length=max_length, doc_stride=128),
                   batched=True, remove_columns=ex.column_names)
    if expected_hash is not None and feature_hash(feats["example_id"]) != expected_hash:
        raise ValueError("feature_hash mismatch: rebuilt features differ from the saved logits' features")
    return ex, feats, (make_extract(ex) if model_key == "phobert" else None)


def refs_of(examples):
    if isinstance(examples, dict):  # dict-of-lists -> list of rows
        examples = [dict(zip(examples, v)) for v in zip(*examples.values())]
    return {r["id"]: list(r["ref_texts"] if "ref_texts" in r else r["answers"]["text"]) for r in examples}


# ---------- post-processing variants ----------

def null_scores(features, start, end, null_variant="min"):
    """example_id -> null score from window-level start[0]+end[0]: min | mean | first window."""
    if null_variant not in NULL_VARIANTS:
        raise ValueError(f"null_variant must be in {NULL_VARIANTS}")
    per = {}
    for fi, eid in enumerate(features["example_id"]):
        per.setdefault(eid, []).append(float(start[fi][0]) + float(end[fi][0]))
    f = {"min": min, "mean": lambda v: sum(v) / len(v), "first": lambda v: v[0]}[null_variant]
    return {eid: f(v) for eid, v in per.items()}


def rescore(examples, feats, start, end, extract=None, null_variant="min", max_answer_length=64, n_best=20,
            _cache=None):
    """Same dict as score_examples, with the chosen null variant. _cache reuses the span search across variants."""
    key = (max_answer_length, n_best)
    if _cache is not None and key in _cache:
        base = _cache[key]
    else:
        base = score_examples(examples, feats, start, end, n_best, max_answer_length, extract=extract)
        if _cache is not None:
            _cache[key] = base
    nulls = null_scores(feats, start, end, null_variant)
    return {eid: dict(v, null_score=nulls.get(eid, np.inf), diff=nulls.get(eid, np.inf) - v["span_score"])
            for eid, v in base.items()}


def tune_variant_grid(examples, feats, start, end, extract=None, grid=GRID, tie=0.3):
    """Grid on DEV. Best by dev F1 at the tuned threshold; within `tie` F1 of the best -> default variant.

    Returns (choice dict, table DataFrame). Same grid for every model.
    """
    refs, cache, rows = refs_of(examples), {}, []
    for nv, mal, nb in itertools.product(grid["null_variant"], grid["max_answer_length"], grid["n_best"]):
        sc = rescore(examples, feats, start, end, extract, nv, mal, nb, _cache=cache)
        thr, _ = tune_threshold(sc, refs)
        m = squad_v2_metrics(apply_threshold(sc, thr), refs)
        rows.append({"null_variant": nv, "max_answer_length": mal, "n_best": nb, "threshold": thr,
                     "f1": m["f1"], "em": m["exact"], "HasAns_f1": m["HasAns_f1"], "NoAns_acc": m["NoAns_exact"]})
    table = pd.DataFrame(rows)
    best = table.loc[table["f1"].idxmax()]
    d = table[(table.null_variant == DEFAULT_VARIANT[0]) & (table.max_answer_length == DEFAULT_VARIANT[1])
              & (table.n_best == DEFAULT_VARIANT[2])]
    if len(d) and best["f1"] - d.iloc[0]["f1"] <= tie:
        best = d.iloc[0]
    choice = {k: best[k].item() if hasattr(best[k], "item") else best[k]
              for k in ("null_variant", "max_answer_length", "n_best", "threshold", "f1")}
    return choice, table


# ---------- ensembling ----------

def ensemble_logits(runs, split="dev", allow_test=False):
    """Mean start/end logits over runs of the SAME model; asserts same example order + feature_hash."""
    _check_split(split, allow_test)
    parts = [r[split] for r in runs]
    if not parts:
        raise ValueError("no runs")
    ref = parts[0]
    for p in parts[1:]:
        if p["feature_hash"] != ref["feature_hash"] or p["example_id"] != ref["example_id"]:
            raise ValueError("runs differ in feature_hash / example_id order; cannot ensemble")
    return (np.mean([p["start"] for p in parts], axis=0), np.mean([p["end"] for p in parts], axis=0),
            ref["example_id"], ref["feature_hash"])


# ---------- statistics ----------

def per_example(preds, refs):
    """DataFrame indexed by example id (sorted): em, f1 in [0,1], has_ans. Same ids -> comparable across models."""
    if set(preds) != set(refs):
        raise ValueError("prediction and reference id sets differ")
    ids = sorted(refs)
    return pd.DataFrame({"em": [exact_match(preds[i], refs[i]) for i in ids],
                         "f1": [f1_score(preds[i], refs[i]) for i in ids],
                         "has_ans": [bool(refs[i]) for i in ids]}, index=ids)


def bootstrap_ci(scores, n=10000, seed=0, alpha=0.05):
    """(mean, lo, hi): percentile CI of the mean over examples."""
    x = np.asarray(scores, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap(a, b, n=10000, seed=0, alpha=0.05):
    """dict(diff=mean(a-b), lo, hi, p): p = share of resamples with diff <= 0 (one-sided, H1: a > b).

    Identical vectors -> every resample diff is exactly 0 -> p = 1.0.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return {"diff": float(d.mean()), "lo": float(np.quantile(means, alpha / 2)),
            "hi": float(np.quantile(means, 1 - alpha / 2)), "p": float((means <= 0).mean())}


def ci_table(pe, n=10000, seed=0):
    """Rows overall/HasAns/NoAns x EM/F1 (x100) with 95% CI, from a per_example frame."""
    rows = []
    for name, sub in (("overall", pe), ("HasAns", pe[pe.has_ans]), ("NoAns", pe[~pe.has_ans])):
        for met in ("em", "f1"):
            if len(sub):
                m, lo, hi = bootstrap_ci(sub[met].values * 100, n, seed)
                rows.append({"subset": name, "metric": met, "n": len(sub), "mean": m, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


# ---------- report tables ----------

def _best_group(df, cols):
    """Mean over seeds per config, then the config with the best mean dev F1."""
    if df.empty:
        return None
    g = df.groupby(cols, dropna=False).agg(dev_f1=("dev_f1", "mean"), dev_em=("dev_em", "mean"),
                                           n_seeds=("seed", "nunique")).reset_index()
    return g.loc[g["dev_f1"].idxmax()]


def layer_table(runs_df, model, default_lr=3e-5, default_epochs=3, extra=None):
    """'Peeling layers' ablation for one model. Rows are cumulative add-ons identified from run cfg:

    baseline (default lr/epochs, no add-ons) -> +hyperparameters (other lr/epochs) -> +LLRD -> +EMA ->
    +plausible loss. `extra` = {label: dev_f1} appended (e.g. '+post-processing', '+ensemble').
    """
    df = runs_df[runs_df.model == model] if len(runs_df) else runs_df
    rows = []
    if len(df):
        none = lambda c: df[c].isna() | (df[c] == 0)  # noqa: E731
        plain = none("lr_decay") & none("ema_decay") & none("plausible_lambda")
        dflt = (df.lr == default_lr) & (df.epochs == default_epochs)
        steps = [("baseline", plain & dflt), ("+hyperparameters", plain & ~dflt),
                 ("+LLRD", ~none("lr_decay") & none("ema_decay") & none("plausible_lambda")),
                 ("+EMA", ~none("lr_decay") & ~none("ema_decay") & none("plausible_lambda")),
                 ("+plausible loss", ~none("lr_decay") & ~none("ema_decay") & ~none("plausible_lambda"))]
        for label, mask in steps:
            b = _best_group(df[mask], ["lr", "epochs", "lr_decay", "ema_decay", "plausible_lambda"])
            if b is not None:
                rows.append({"step": label, "dev_f1": b["dev_f1"], "dev_em": b["dev_em"], "n_seeds": b["n_seeds"]})
    for label, f1 in (extra or {}).items():
        rows.append({"step": label, "dev_f1": f1, "dev_em": np.nan, "n_seeds": np.nan})
    t = pd.DataFrame(rows, columns=["step", "dev_f1", "dev_em", "n_seeds"])
    t["delta_f1"] = t["dev_f1"].diff()
    return t


def cost_table(runs_df):
    """Per model: params (M, approx), mean train_runtime_s, est_units, training windows/sec (windows*epochs/s)."""
    if runs_df.empty:
        return pd.DataFrame(columns=["model", "params_M", "n_runs", "train_runtime_s", "est_units", "windows_per_s"])
    d = runs_df.assign(wps=runs_df.train_windows * runs_df.epochs / runs_df.train_runtime_s)
    g = d.groupby("model").agg(n_runs=("run_name", "count"), train_runtime_s=("train_runtime_s", "mean"),
                               est_units=("est_units", "mean"), windows_per_s=("wps", "mean")).reset_index()
    g.insert(1, "params_M", g["model"].map(PARAMS_M))
    return g
