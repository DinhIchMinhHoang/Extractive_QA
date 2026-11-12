"""Audit and cleaning helpers for UIT-ViQuAD2.0 (SQuAD-style rows, plain dicts)."""
import re
import unicodedata
from collections import Counter

ISSUES = (
    "offset_mismatch",
    "empty_answer_text",
    "not_nfc",
    "has_double_space",
    "has_nbsp_or_zerowidth",
    "multiple_answers",
    "answerable_without_answer",
    "impossible_with_answer",
    "empty_question",
    "empty_context",
)

_BAD_CHARS = " ​‌‍﻿"
_ZW = {ord(c): None for c in "​‌‍﻿"}
_WS = re.compile(r"\s+")
_MULTI_WS = re.compile(r"\s{2,}|[\t\n\r]")


def normalize_text(s):
    """NFC, drop zero-width chars, nbsp -> space, collapse whitespace, strip."""
    s = (s or "").translate(_ZW).replace(" ", " ")
    s = unicodedata.normalize("NFC", s)
    return _WS.sub(" ", s).strip()


def _pairs(d):
    """Return (texts, starts) from an answers-like dict, tolerant of None/missing."""
    d = d or {}
    texts = list(d.get("text") or [])
    starts = list(d.get("answer_start") or [])
    starts += [0] * (len(texts) - len(starts))
    return texts, starts[: len(texts)]


def audit_row(row):
    """Return dict of boolean issue flags (keys = ISSUES) for one row."""
    ctx = row.get("context") or ""
    q = row.get("question") or ""
    texts, starts = _pairs(row.get("answers"))
    imp = bool(row.get("is_impossible"))
    parts = [ctx, q] + texts
    return {
        "offset_mismatch": any(
            int(s) < 0 or ctx[int(s): int(s) + len(t)] != t for t, s in zip(texts, starts)
        ),
        "empty_answer_text": (not imp) and any(not t.strip() for t in texts),
        "not_nfc": any(unicodedata.normalize("NFC", x) != x for x in parts),
        "has_double_space": any(_MULTI_WS.search(x) is not None for x in parts),
        "has_nbsp_or_zerowidth": any(c in x for x in parts for c in _BAD_CHARS),
        "multiple_answers": len(texts) > 1,
        "answerable_without_answer": (not imp) and not texts,
        "impossible_with_answer": imp and bool(texts),
        "empty_question": not q.strip(),
        "empty_context": not ctx.strip(),
    }


def _locate(old_ctx, new_ctx, new_text, old_start):
    """Find new_text in new_ctx; pick occurrence nearest to the remapped old_start."""
    if not new_text:
        return None
    old_start = max(int(old_start or 0), 0)
    # estimate only: length of normalized prefix ("x" guard keeps trailing space)
    est = len(normalize_text(old_ctx[:old_start] + "x")) - 1
    best = None
    i = new_ctx.find(new_text)
    while i != -1:
        if best is None or abs(i - est) < abs(best - est):
            best = i
        i = new_ctx.find(new_text, i + 1)
    return best


def clean_row(row, is_train):
    """Return (new_row or None, actions). Eval splits (is_train False) are never dropped."""
    acts = []
    old_ctx = row.get("context") or ""
    ctx = normalize_text(old_ctx)
    q = normalize_text(row.get("question"))
    title = normalize_text(row.get("title"))
    for name, a, b in (
        ("context", old_ctx, ctx),
        ("question", row.get("question") or "", q),
        ("title", row.get("title") or "", title),
    ):
        if a != b:
            acts.append("normalized_" + name)

    imp = bool(row.get("is_impossible"))
    texts, starts = _pairs(row.get("answers"))
    new_t, new_s, ok = [], [], True
    for t, s in zip(texts, starts):
        nt = normalize_text(t)
        p = _locate(old_ctx, ctx, nt, s)
        if p is None:
            ok = False
            break
        new_t.append(nt)
        new_s.append(p)
        if nt != t:
            acts.append("normalized_answer")
        if p != s:
            acts.append("offset_fixed")

    problems = []
    if not q:
        problems.append("empty_question")
    if not ctx:
        problems.append("empty_context")
    if not imp and not texts:
        problems.append("no_answer")
    if not imp and texts and not ok:
        problems.append("unfixable")

    acts = list(dict.fromkeys(acts))
    if is_train and problems:
        return None, acts + ["dropped_" + problems[0]]
    acts += [p + "_kept" for p in problems]

    new = dict(row)
    new.update(question=q, title=title, context=ctx)
    if ok:
        new["answers"] = {"text": new_t, "answer_start": new_s}
    # else: leave answers as-is so eval labels stay untouched

    if "plausible_answers" in row:
        pt, ps = _pairs(row.get("plausible_answers"))
        keep_t, keep_s = [], []
        for t, s in zip(pt, ps):
            nt = normalize_text(t)
            p = _locate(old_ctx, ctx, nt, s)
            if p is not None:  # best effort: silently drop if not found
                keep_t.append(nt)
                keep_s.append(p)
        new["plausible_answers"] = {"text": keep_t, "answer_start": keep_s}
    return new, acts


def clean_split(rows, is_train):
    """Clean all rows; return (list_of_rows, Counter of actions)."""
    out, cnt = [], Counter()
    for r in rows:
        new, acts = clean_row(r, is_train)
        cnt.update(acts)
        if new is not None:
            out.append(new)
    return out, cnt


def _selftest():
    def mk(ctx, text, start, imp=False):
        return {"id": "x", "title": "t", "context": ctx, "question": "q?",
                "is_impossible": imp,
                "answers": {"text": [text] if text is not None else [],
                            "answer_start": [start] if text is not None else []},
                "plausible_answers": {"text": [], "answer_start": []}}

    # 1. zero-width + nbsp + double space shift offsets; must be remapped
    ctx = "Xin​ chào  bạn Hà Nội"
    row = mk(ctx, "Hà Nội", ctx.index("Hà Nội"))
    assert not audit_row(row)["offset_mismatch"]
    assert audit_row(row)["has_nbsp_or_zerowidth"]
    new, acts = clean_row(row, True)
    s = new["answers"]["answer_start"][0]
    assert new["context"] == "Xin chào bạn Hà Nội"
    assert new["context"][s: s + 6] == "Hà Nội" and "offset_fixed" in acts

    # 2. wrong offset repaired
    new, acts = clean_row(mk("abc def ghi", "def", 0), True)
    assert audit_row(new)["offset_mismatch"] is False
    assert new["answers"]["answer_start"] == [4] and "offset_fixed" in acts

    # 3. unfixable: dropped in train, kept untouched in validation
    bad = mk("abc def ghi", "xyz", 0)
    new, acts = clean_row(bad, True)
    assert new is None and "dropped_unfixable" in acts
    new, acts = clean_row(bad, False)
    assert new is not None and "unfixable_kept" in acts
    assert new["answers"] == bad["answers"]

    # 4. impossible row passes through
    new, _ = clean_row(mk("abc", None, 0, imp=True), True)
    assert new is not None and new["answers"]["text"] == []
    print("selftest OK")


if __name__ == "__main__":
    _selftest()
