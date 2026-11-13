"""N-best span extraction + null-threshold decision for SQuAD-2.0 style QA."""
from collections import defaultdict

import numpy as np

from src.metrics import squad_v2_metrics


def score_examples(examples, features, start_logits, end_logits, n_best=20, max_answer_length=64, extract=None):
    """Per example: best non-null span text/score, min-over-windows null score, diff = null - span.

    max_answer_length is in tokens: answers p95 = 31 words ~ 55 XLM-R tokens, so 64 covers them.
    examples/features: dict-likes (datasets or dict-of-lists) with id/context and example_id/offset_mapping.
    """
    ctx = dict(zip(examples["id"], examples["context"]))
    by_ex = defaultdict(list)
    for fi, eid in enumerate(features["example_id"]):
        by_ex[eid].append(fi)
    out = {}
    for eid, context in ctx.items():
        null, best, text = np.inf, -np.inf, ""
        for fi in by_ex.get(eid, []):
            sl, el = np.asarray(start_logits[fi]), np.asarray(end_logits[fi])
            offs = features["offset_mapping"][fi]
            null = min(null, float(sl[0] + el[0]))
            for s in np.argsort(sl)[-1:-n_best - 1:-1]:
                for e in np.argsort(el)[-1:-n_best - 1:-1]:
                    if offs[s] is None or offs[e] is None or e < s or e - s + 1 > max_answer_length:
                        continue
                    sc = float(sl[s] + el[e])
                    if sc > best:
                        best = sc
                        text = (extract(eid, offs[s][0], offs[e][1]) if extract
                                else context[offs[s][0]:offs[e][1]])
        out[eid] = {"text": text, "span_score": best, "null_score": null, "diff": null - best}
    return out


def apply_threshold(scores, threshold=0.0):
    return {i: ("" if v["diff"] > threshold else v["text"]) for i, v in scores.items()}


def tune_threshold(scores, references, grid=None):
    """Pick threshold maximizing overall F1 over a grid; returns (best_threshold, table).

    Default grid: linspace(-10, 10, 81) plus the finite diffs' 5%..95% quantiles (simple, not exhaustive).
    """
    if grid is None:
        d = np.array([v["diff"] for v in scores.values() if np.isfinite(v["diff"])])
        q = np.quantile(d, np.linspace(0.05, 0.95, 19)) if len(d) else []
        grid = np.unique(np.concatenate([np.linspace(-10, 10, 81), q]))
    table = []
    for t in grid:
        m = squad_v2_metrics(apply_threshold(scores, float(t)), references)
        table.append({"threshold": float(t), "f1": m["f1"], "em": m["exact"],
                      "HasAns_f1": m["HasAns_f1"], "NoAns_acc": m["NoAns_exact"]})
    best = max(table, key=lambda r: r["f1"])  # first max wins on ties
    return best["threshold"], table
