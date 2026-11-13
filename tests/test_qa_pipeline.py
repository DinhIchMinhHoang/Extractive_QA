"""Run: .venv\\Scripts\\python tests\\test_qa_pipeline.py"""
import sys

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.metrics import exact_match, f1_score, normalize_answer, squad_v2_metrics  # noqa: E402
from src.postprocess import apply_threshold, score_examples, tune_threshold  # noqa: E402
from src.preprocess import prepare_eval_features, prepare_train_features  # noqa: E402

DATA = ROOT / "data" / "processed" / "uit_viquad2_split"
_cache = {}


def _tok():
    if "tok" not in _cache:
        from transformers import AutoTokenizer
        _cache["tok"] = AutoTokenizer.from_pretrained("xlm-roberta-base")
    return _cache["tok"]


def _data():
    if not DATA.exists():
        pytest.skip(f"{DATA} missing")
    if "ds" not in _cache:
        from datasets import load_from_disk
        _cache["ds"] = load_from_disk(str(DATA))
    return _cache["ds"]


def _batch(rows):
    return {k: [r[k] for r in rows] for k in ("id", "question", "context", "answers")}


def _midword(ctx, s, e):
    """True if gold span [s, e) starts or ends inside a word (known source noise)."""
    return (s > 0 and ctx[s - 1].isalnum() and ctx[s].isalnum()) or \
           (e < len(ctx) and ctx[e - 1].isalnum() and ctx[e].isalnum())


def test_metrics():
    assert exact_match("Hà Nội", ["Hà Nội"]) == 1 and f1_score("Hà Nội", ["Hà Nội"]) == 1
    assert exact_match("  HÀ NỘI. ", ["hà nội"]) == 1
    # pred 3 tokens, ref 2, overlap 2 -> P=2/3 R=1 -> F1=0.8
    assert abs(f1_score("a b c", ["a b"]) - 0.8) < 1e-9
    assert exact_match("a b c", ["a b"]) == 0
    assert exact_match("", []) == 1 and f1_score("", []) == 1
    assert exact_match("x", []) == 0 and f1_score("x", []) == 0
    assert exact_match("", ["x"]) == 0 and f1_score("", ["x"]) == 0
    # both normalize to empty (official SQuAD: empty == empty -> 1)
    assert exact_match("!!", ["!!"]) == 1 and f1_score("!!", ["!!"]) == 1
    assert exact_match("", ["!!"]) == 1 and f1_score("!!", [""]) == 1
    assert exact_match("!!", ["x"]) == 0 and f1_score("!!", ["x"]) == 0
    assert f1_score("a b", ["c d", "a b"]) == 1 and exact_match("a b", ["c d", "a b"]) == 1
    assert normalize_answer("Các, the  Đà-Nẵng!") == "các the đànẵng"
    m = squad_v2_metrics({"1": "a", "2": ""}, {"1": ["a"], "2": []})
    assert m["exact"] == 100 and m["has_ans"] == 1 and m["no_ans"] == 1
    try:
        squad_v2_metrics({"1": "a"}, {"2": ["a"]})
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_labels():
    tok, ds = _tok(), _data()
    rows = [r for r in ds["train"].select(range(3000)) if r["answers"]["text"]][:300]
    f = prepare_train_features(_batch(rows), tok)
    assert "offset_mapping" not in f
    ev = prepare_eval_features(_batch(rows), tok)  # same tokenization, offsets give char spans
    by_id = {r["id"]: r for r in rows}
    n_checked = n_exact = n_mid = 0
    for i, eid in enumerate(ev["example_id"]):
        s, e = f["start_positions"][i], f["end_positions"][i]
        if s > 0:
            r = by_id[eid]
            o = ev["offset_mapping"][i]
            ctx = r["context"]
            s_char = r["answers"]["answer_start"][0]
            e_char = s_char + len(r["answers"]["text"][0])  # exclusive
            got = ctx[o[s][0]:o[e][1]]
            g, a = normalize_answer(got), normalize_answer(r["answers"]["text"][0])
            n_checked += 1
            if _midword(ctx, s_char, e_char):
                n_mid += 1
                assert a in g, (got, r["answers"])
                continue
            n_exact += 1
            assert g == a, (got, r["answers"])
            # a lone "▁" token can share its start with the word token after it: s must be the later one
            assert o[s][0] <= s_char and (o[s - 1] is None or o[s - 1][1] <= s_char
                                          or o[s - 1][0] == o[s][0]), (s, o[s - 1:s + 1], s_char)
            assert o[e][1] >= e_char and (o[e + 1] is None or o[e + 1][0] > e_char - 1), (e, o[e:e + 2], e_char)
    assert n_checked >= 250, n_checked
    print(f"  labels: {n_exact} exact / {n_mid} mid-word / {n_checked} total")
    no = [r for r in ds["train"].select(range(3000)) if not r["answers"]["text"]][:100]
    fn = prepare_train_features(_batch(no), tok)
    assert all(s == 0 for s in fn["start_positions"]) and all(e == 0 for e in fn["end_positions"])
    # long contexts: some window labeled 0 with the answer outside it, another labeled non-zero
    found = False
    for r in ds["train"].select(range(3000)):
        if not r["answers"]["text"] or len(tok(r["context"])["input_ids"]) <= 384:
            continue
        b = _batch([r])
        ft, fe = prepare_train_features(b, tok), prepare_eval_features(b, tok)
        s_char = r["answers"]["answer_start"][0]
        e_char = s_char + len(r["answers"]["text"][0])
        outside = [i for i, o in enumerate(fe["offset_mapping"])
                   if ft["start_positions"][i] == 0 and
                   (max(x[1] for x in o if x) <= s_char or min(x[0] for x in o if x) >= e_char)]
        if outside and any(s > 0 for s in ft["start_positions"]):
            found = True
            break
    assert found, "no long-context example with a null-labeled window found"


def test_window_boundaries():
    tok = _tok()
    ctx = " ".join(f"từ{i}" for i in range(100))
    ans = " ".join(f"từ{i}" for i in range(48, 53))
    s_char = ctx.index(ans)
    e_char = s_char + len(ans)
    b = {"id": ["x"], "question": ["Câu hỏi?"], "context": [ctx],
         "answers": [{"text": [ans], "answer_start": [s_char]}]}
    f = prepare_train_features(b, tok, max_length=32, doc_stride=8)
    ev = prepare_eval_features(b, tok, max_length=32, doc_stride=8)
    n_in = n_part = n_out = 0
    for i, offs in enumerate(ev["offset_mapping"]):
        c = [o for o in offs if o is not None]
        lo, hi = c[0][0], c[-1][1]
        s, e = f["start_positions"][i], f["end_positions"][i]
        if lo <= s_char and hi >= e_char:
            n_in += 1
            assert s > 0 and e >= s and offs[s] and offs[e], (i, s, e)
            assert ctx[offs[s][0]:offs[e][1]] == ans, (i, ctx[offs[s][0]:offs[e][1]])
        else:
            assert s == 0 and e == 0, (i, s, e)
            if lo < e_char and hi > s_char:
                n_part += 1
            else:
                n_out += 1
    print(f"  windows: {len(ev['offset_mapping'])} total, {n_in} contain, {n_part} straddle, {n_out} disjoint")
    assert n_in >= 1 and n_part >= 1 and n_out >= 1, (n_in, n_part, n_out)


def test_oracle_pipeline():
    """Tests postprocess + threshold only (oracle logits on token-aligned dev rows)."""
    tok, ds = _tok(), _data()
    rows = [r for r in ds["dev"].select(range(300))
            if not r["answers"]["text"]
            or not _midword(r["context"], r["answers"]["answer_start"][0],
                            r["answers"]["answer_start"][0] + len(r["answers"]["text"][0]))][:200]
    refs = {r["id"]: list(r["answers"]["text"]) for r in rows}
    batch = _batch(rows)
    feats = prepare_eval_features(batch, tok)
    by_id = {r["id"]: r for r in rows}
    n = len(feats["input_ids"])
    L = len(feats["input_ids"][0])
    sl, el = np.full((n, L), -1000.0), np.full((n, L), -1000.0)
    for i, eid in enumerate(feats["example_id"]):
        r, offs = by_id[eid], feats["offset_mapping"][i]
        placed = False
        if r["answers"]["text"]:
            s_char = r["answers"]["answer_start"][0]
            e_char = s_char + len(r["answers"]["text"][0])
            idx = [k for k, o in enumerate(offs) if o is not None]
            if offs[idx[0]][0] <= s_char and offs[idx[-1]][1] >= e_char:
                s = max(k for k in idx if offs[k][0] <= s_char)
                e = min(k for k in idx if offs[k][1] >= e_char)
                sl[i, s] = el[i, e] = 10.0
                placed = True
        if not placed:
            sl[i, 0] = el[i, 0] = 10.0
    scores = score_examples(batch, feats, sl, el)
    t, table = tune_threshold(scores, refs)
    m = squad_v2_metrics(apply_threshold(scores, t), refs)
    assert m["exact"] == 100 and m["f1"] == 100 and m["NoAns_exact"] == 100, (t, m)
    assert m["has_ans"] > 0 and m["no_ans"] > 0, m
    # "" iff diff > threshold, so a very LOW threshold abstains everywhere, a high one never abstains
    m2 = squad_v2_metrics(apply_threshold(scores, -1e9), refs)
    assert m2["HasAns_f1"] == 0 and m2["NoAns_exact"] == 100, m2
    m3 = squad_v2_metrics(apply_threshold(scores, 1e9), refs)
    assert m3["HasAns_f1"] == 100 and m3["NoAns_exact"] < 100, m3
