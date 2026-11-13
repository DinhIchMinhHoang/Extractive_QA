"""SQuAD-v2 style EM/F1 for Vietnamese (no hub downloads, no `evaluate`)."""
import unicodedata
from collections import Counter


def normalize_answer(s):
    # Deviation from official SQuAD: no English-article removal (a/an/the are meaningless in Vietnamese).
    s = unicodedata.normalize("NFC", s or "").lower()
    # Only Unicode P* punctuation is stripped (official also strips ASCII symbols like $+<=>^`|~); deviation accepted.
    s = "".join(c for c in s if not unicodedata.category(c).startswith("P"))
    return " ".join(s.split())


def _em1(pred, ref):
    return float(normalize_answer(pred) == normalize_answer(ref))


def _f1_1(pred, ref):
    p, r = normalize_answer(pred).split(), normalize_answer(ref).split()
    if not p or not r:
        return float(p == r)
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def _score(fn, pred, refs):
    if not refs:  # NoAns: correct only if the model abstains
        return float(not normalize_answer(pred))
    return max(fn(pred, r) for r in refs)


def exact_match(pred, refs):
    return _score(_em1, pred, refs)


def f1_score(pred, refs):
    return _score(_f1_1, pred, refs)


def squad_v2_metrics(predictions, references):
    """predictions: id -> str; references: id -> list[str] ([] = NoAns). Scores are x100."""
    if set(predictions) != set(references):
        raise ValueError("prediction and reference id sets differ")
    out = {"exact": 0.0, "f1": 0.0, "HasAns_exact": 0.0, "HasAns_f1": 0.0, "NoAns_exact": 0.0}
    n_has = sum(bool(r) for r in references.values())
    n_no = len(references) - n_has
    for i, refs in references.items():
        em, f1 = exact_match(predictions[i], refs), f1_score(predictions[i], refs)
        out["exact"] += em
        out["f1"] += f1
        if refs:
            out["HasAns_exact"] += em
            out["HasAns_f1"] += f1
        else:
            out["NoAns_exact"] += em
    for k, d in (("exact", len(references)), ("f1", len(references)), ("HasAns_exact", n_has),
                 ("HasAns_f1", n_has), ("NoAns_exact", n_no)):
        out[k] = 100.0 * out[k] / d if d else 0.0
    out.update(has_ans=n_has, no_ans=n_no)
    return out
