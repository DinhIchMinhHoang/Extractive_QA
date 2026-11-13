"""Run: .venv\\Scripts\\python tests\\test_phobert.py"""
import sys

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.metrics import squad_v2_metrics  # noqa: E402
from src.phobert import load_phobert_tokenizer, make_extract  # noqa: E402
from src.postprocess import apply_threshold, score_examples, tune_threshold  # noqa: E402
from src.preprocess import prepare_eval_features, prepare_train_features  # noqa: E402
from src.segment import (AlignError, Segmenter, _midword, _norm, align, orig_span_to_seg,  # noqa: E402
                         segment_dataset, seg_span_to_orig)

DATA = ROOT / "data" / "processed" / "uit_viquad2_split"
_cache = {}


def _seg():
    return _cache.setdefault("seg", Segmenter())


def _tok():
    if "tok" not in _cache:
        _cache["tok"] = load_phobert_tokenizer()
    return _cache["tok"]


def _data():
    if not DATA.exists():
        pytest.skip(f"{DATA} missing")
    if "ds" not in _cache:
        from datasets import load_from_disk
        _cache["ds"] = load_from_disk(str(DATA))
    return _cache["ds"]


def _seg_rows(split, n_scan, n, has_ans):
    key = (split, n_scan, n, has_ans)
    if key not in _cache:
        rows = [r for r in _data()[split].select(range(n_scan)) if bool(r["answers"]["text"]) or not has_ans][:n]
        out, st = segment_dataset(rows, _seg())
        _cache[key] = (rows, out, st)
    return _cache[key]


def _batch(rows):
    return {k: [r[k] for r in rows] for k in ("id", "question", "context", "answers")}


def _roundtrip(orig, seg):
    o2s, s2o = align(orig, seg)
    assert len(o2s) == len(orig) + 1 and len(s2o) == len(seg) + 1
    n = 0
    for a in range(len(orig)):
        if orig[a].isspace():
            continue
        for b in range(a + 1, len(orig) + 1):
            if orig[b - 1].isspace():
                continue
            assert seg_span_to_orig(s2o, *orig_span_to_seg(o2s, a, b)) == (a, b), (a, b)
            n += 1
    return n


def test_align_units():
    orig = "Hà Nội là thủ đô của Việt Nam, nằm ở đồng bằng."
    seg = _seg().segment(orig)
    assert "Hà_Nội" in seg and " , " in seg, seg
    assert _roundtrip(orig, seg) > 100
    # hand-written: dropped double space + newline, inserted spaces around punctuation
    orig2 = "a  b_c,d\ne (f)"
    seg2 = "a b_c , d e ( f )"
    assert _roundtrip(orig2, seg2) > 10
    o2s, s2o = align(orig2, seg2)
    assert o2s[-1] == len(seg2) and s2o[-1] == len(orig2)
    s, e = orig_span_to_seg(o2s, 0, 4)  # "a  b"
    assert seg2[s:e] == "a b"
    for a, b in [("abc", "abd"), ("keo", "kêo"), ("đa", "da"), ("hoa", "hoă")]:
        try:
            align(a, b)
            raise AssertionError(f"expected AlignError {a!r} {b!r}")
        except AlignError:
            pass
    align("hòa", "hoà")  # tone-mark move only: still aligns


def test_span_snapping():
    # inserted '_' glued to punctuation: seg span (0,3) 'Mỹ_' must give orig 'Mỹ', not 'Mỹ”'
    o2s, s2o = align("Mỹ”", "Mỹ_”")
    assert s2o[2] == -1 and s2o[-1] == 3
    assert seg_span_to_orig(s2o, 0, 3) == (0, 2)
    assert seg_span_to_orig(s2o, 2, 4) == (2, 3)  # starts on an inserted char
    assert seg_span_to_orig(s2o, 2, 3) is None  # only inserted chars
    assert seg_span_to_orig(s2o, 1, 1) is None  # empty
    # dropped orig whitespace
    o2s, s2o = align("x  y", "x y")
    assert o2s[2] == -1
    assert orig_span_to_seg(o2s, 0, 3) == (0, 2)  # trailing dropped space snapped off
    assert orig_span_to_seg(o2s, 2, 4) == (2, 3)  # leading dropped space snapped off
    assert orig_span_to_seg(o2s, 2, 3) is None
    assert orig_span_to_seg(o2s, 0, 0) is None
    # '_' replacing an orig space counts as matched
    o2s, s2o = align("Hà Nội", "Hà_Nội")
    assert orig_span_to_seg(o2s, 0, 6) == (0, 6) and orig_span_to_seg(o2s, 2, 3) == (2, 3)


def test_segment_real_data():
    rows, out, st = _seg_rows("train", 3000, 300, True)
    assert st["align_errors"] == 0 and len(out) == 300, st
    assert not any(o.get("align_error") for o in out)
    for r, o in zip(rows, out):
        # the segmenter also moves tone marks (hòa -> hoà): compare with tone marks (only) removed
        assert _norm(o["answers"]["text"][0]) == _norm(r["answers"]["text"][0]), (o["answers"], r["answers"])
        assert o["ref_texts"] == list(r["answers"]["text"]) and o["orig_context"] == r["context"]
    # 300 more questions-only check: question segmentation is non-empty
    assert all(o["question"] for o in out)


def _pl_rows():
    # 300 NoAns train rows that have a plausible answer
    if "pl" not in _cache:
        rows = [r for r in _data()["train"].select(range(6000)) if r["plausible_answers"]["text"]][:300]
        _cache["pl"] = (rows, segment_dataset(rows, _seg())[0])
    return _cache["pl"]


def test_plausible_real_data():
    rows, out = _pl_rows()
    assert len(rows) == 300 and len(out) == 300
    n = 0
    for r, o in zip(rows, out):
        p = o["plausible_answers"]
        if not p["text"]:
            continue
        s = p["answer_start"][0]
        assert p["text"][0] == o["context"][s:s + len(p["text"][0])]
        got = o["context"][s:s + len(p["text"][0])].replace("_", "").replace(" ", "")
        assert _norm(got) == _norm(r["plausible_answers"]["text"][0].replace(" ", "")), (p, r["plausible_answers"])
        assert not o["answers"]["text"]
        n += 1
    assert n >= 290, n


def test_plausible_token_window():
    tok = _tok()
    _, rows = _pl_rows()
    ev = prepare_eval_features(_batch(rows), tok, max_length=256, doc_stride=128)
    by_id = {r["id"]: r for r in rows}
    seen, n = set(), 0
    for i, eid in enumerate(ev["example_id"]):
        p = by_id[eid]["plausible_answers"]
        if eid in seen or not p["text"]:
            continue
        sc, s = by_id[eid]["context"], p["answer_start"][0]
        e = s + len(p["text"][0])
        offs = ev["offset_mapping"][i]
        c = [k for k, o in enumerate(offs) if o is not None]
        if offs[c[0]][0] > s or offs[c[-1]][1] < e or _midword(sc, s, e):
            continue
        seen.add(eid)
        a = max(k for k in c if offs[k][0] <= s)
        b = min(k for k in c if offs[k][1] >= e)
        assert sc[offs[a][0]:offs[b][1]] == p["text"][0], (sc[offs[a][0]:offs[b][1]], p["text"])
        n += 1
    assert n >= 100, n


def test_seg_data_plausible_schema():
    from src.phobert import get_seg_data
    try:
        d = get_seg_data()
    except FileNotFoundError:
        pytest.skip("seg data missing")
    for k in ("train", "dev", "test"):
        f = d[k].features["plausible_answers"]
        assert set(f) == {"text", "answer_start"}, (k, f)
        assert f["text"].feature.dtype == "string" and f["answer_start"].feature.dtype == "int64", (k, f)
    tr = d["train"]
    ex = next(r for r in tr.select(range(3000)) if not r["answers"]["text"])
    assert len(ex["plausible_answers"]["text"]) == 1 and len(ex["plausible_answers"]["answer_start"]) == 1
    ha = next(r for r in tr.select(range(50)) if r["answers"]["text"])
    assert ha["plausible_answers"] == {"text": [], "answer_start": []}


def test_labels():
    tok = _tok()
    _, rows, _ = _seg_rows("train", 3000, 300, True)
    f = prepare_train_features(_batch(rows), tok, max_length=256, doc_stride=128)
    ev = prepare_eval_features(_batch(rows), tok, max_length=256, doc_stride=128)
    by_id = {r["id"]: r for r in rows}
    n_exact = n_mid = n_tot = 0
    for i, eid in enumerate(ev["example_id"]):
        s, e = f["start_positions"][i], f["end_positions"][i]
        if s == 0:
            continue
        r, o = by_id[eid], ev["offset_mapping"][i]
        a = r["answers"]["answer_start"][0]
        b = a + len(r["answers"]["text"][0])
        got = r["context"][o[s][0]:o[e][1]]
        n_tot += 1
        if _midword(r["context"], a, b):
            n_mid += 1
            assert r["answers"]["text"][0] in got, (got, r["answers"])
        else:
            n_exact += 1
            assert got == r["answers"]["text"][0], (got, r["answers"])
    print(f"  labels: {n_exact} exact / {n_mid} mid-word / {n_tot} total")
    assert n_tot >= 250, n_tot


def test_window_boundaries():
    tok = _tok()
    ctx = " ".join(f"từ{i}_ngữ{i}" for i in range(100))
    ans = " ".join(f"từ{i}_ngữ{i}" for i in range(48, 50))  # 2 words: PhoBERT BPE makes ~4 tokens/word
    s_char = ctx.index(ans)
    e_char = s_char + len(ans)
    b = {"id": ["x"], "question": ["Câu_hỏi ?"], "context": [ctx],
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


def test_midword_oracle_contains_gold():
    rows, out, st = _seg_rows("dev", 1500, 1500, True)
    ex = make_extract(out)
    n_mid = 0
    for o in out:
        if o.get("align_error") or not o["answers"]["text"]:
            continue
        sc, s = o["context"], o["answers"]["answer_start"][0]
        e = s + len(o["answers"]["text"][0])
        if not _midword(sc, s, e):
            continue
        n_mid += 1
        while s > 0 and sc[s - 1] != " ":  # labels expand to whole seg words
            s -= 1
        while e < len(sc) and sc[e] != " ":
            e += 1
        assert _norm(o["ref_texts"][0]) in _norm(ex(o["id"], s, e)), (o["ref_texts"], ex(o["id"], s, e))
    assert n_mid > 0 and not ex.failures
    print(f"  mid-word rows: {n_mid}; EM ceiling ~ {1 - st['midword'] / max(1, st['has_ans']):.4f}")


def test_oracle_pipeline():
    tok = _tok()
    rows = [r for r in _seg_rows("dev", 300, 300, False)[1]
            if not r["answers"]["text"]
            or not _midword(r["context"], r["answers"]["answer_start"][0],
                            r["answers"]["answer_start"][0] + len(r["answers"]["text"][0]))][:200]
    refs = {r["id"]: list(r["ref_texts"]) for r in rows}
    batch = _batch(rows)
    feats = prepare_eval_features(batch, tok, max_length=256, doc_stride=128)
    by_id = {r["id"]: r for r in rows}
    n, L = len(feats["input_ids"]), len(feats["input_ids"][0])
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
    scores = score_examples(batch, feats, sl, el, extract=make_extract(rows))
    t, _ = tune_threshold(scores, refs)
    preds = apply_threshold(scores, t)
    m = squad_v2_metrics(preds, refs)
    assert m["exact"] == 100 and m["f1"] == 100 and m["NoAns_exact"] == 100, (t, m)
    assert m["has_ans"] > 0 and m["no_ans"] > 0, m
    for eid, p in preds.items():
        if p:
            assert p in by_id[eid]["orig_context"], p
            assert p in refs[eid], (p, refs[eid])


def test_model_forward_inputs():
    # RoBERTa type_vocab_size=1 / 258 positions: a stray token_type_ids=1 or id >= 64001 crashes (CUDA assert on GPU)
    import torch
    from transformers import AutoModelForQuestionAnswering
    from src.phobert import PHOBERT
    tok = _tok()
    _, rows, _ = _seg_rows("dev", 300, 16, True)
    model = AutoModelForQuestionAnswering.from_pretrained(PHOBERT).eval()
    for fn in (prepare_train_features, prepare_eval_features):
        f = fn(_batch(rows), tok, max_length=256, doc_stride=128)
        assert "token_type_ids" not in f, list(f.keys())
        assert max(max(x) for x in f["input_ids"]) < 64001
        assert max(len(x) for x in f["attention_mask"]) <= 256
        ids = torch.tensor(f["input_ids"][:4])
        mask = torch.tensor(f["attention_mask"][:4])
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
        assert out.start_logits.shape == (4, 256) and out.end_logits.shape == (4, 256)


def test_no_out_of_vocab_ids():
    tok = _tok()
    # 'tuso' is a rare piece: raw fast tokenizer gives id 65972 (>= 64001), slow tokenizer gives <unk>
    assert max(tok("tuso", add_special_tokens=False)["input_ids"]) >= 64001
    f = prepare_eval_features({"id": ["x"], "question": ["Câu_hỏi ?"], "context": ["tuso"]}, tok, max_length=32, doc_stride=8)
    assert 3 in f["input_ids"][0] and max(f["input_ids"][0]) <= 64000, f["input_ids"][0]
    from src.phobert import get_seg_data
    try:
        tr = get_seg_data()["train"].select(range(2000))
    except FileNotFoundError:
        pytest.skip("seg data missing")
    b = {k: list(tr[k]) for k in ("id", "question", "context", "answers")}
    for fn in (prepare_train_features, prepare_eval_features):
        ids = np.array(fn(b, tok, max_length=256, doc_stride=128)["input_ids"])
        assert ids.min() >= 0 and ids.max() <= 64000, (fn.__name__, ids.min(), ids.max())
