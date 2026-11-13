"""Run: .venv\\Scripts\\python tests\\test_evalkit.py"""
import json
import sys

import pytest
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src import evalkit as ek  # noqa: E402
from src.postprocess import apply_threshold, tune_threshold  # noqa: E402
from src.metrics import squad_v2_metrics  # noqa: E402
from src.preprocess import prepare_eval_features  # noqa: E402
from src.train_qa import save_logits  # noqa: E402

DATA = ROOT / "data" / "processed" / "uit_viquad2_split"
_cache = {}


def _oracle():
    """~100 dev rows, XLM-R tokenizer, oracle logits (same recipe as test_qa_pipeline)."""
    if "o" in _cache:
        return _cache["o"]
    if not DATA.exists():
        pytest.skip(f"{DATA} missing")
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
    ds = load_from_disk(str(DATA))

    def midword(c, s, e):
        return (s > 0 and c[s - 1].isalnum() and c[s].isalnum()) or (e < len(c) and c[e - 1].isalnum() and c[e].isalnum())

    rows = [r for r in ds["dev"].select(range(300)) if not r["answers"]["text"]
            or not midword(r["context"], r["answers"]["answer_start"][0],
                           r["answers"]["answer_start"][0] + len(r["answers"]["text"][0]))][:100]
    batch = {k: [r[k] for r in rows] for k in ("id", "question", "context", "answers")}
    feats = prepare_eval_features(batch, tok)
    by_id = {r["id"]: r for r in rows}
    n, L = len(feats["input_ids"]), len(feats["input_ids"][0])
    sl, el = np.full((n, L), -1000.0, np.float32), np.full((n, L), -1000.0, np.float32)
    for i, eid in enumerate(feats["example_id"]):
        r, offs, placed = by_id[eid], feats["offset_mapping"][i], False
        if r["answers"]["text"]:
            s_c = r["answers"]["answer_start"][0]
            e_c = s_c + len(r["answers"]["text"][0])
            idx = [k for k, o in enumerate(offs) if o is not None]
            if offs[idx[0]][0] <= s_c and offs[idx[-1]][1] >= e_c:
                sl[i, max(k for k in idx if offs[k][0] <= s_c)] = 10.0
                el[i, min(k for k in idx if offs[k][1] >= e_c)] = 10.0
                placed = True
        if not placed:
            sl[i, 0] = el[i, 0] = 10.0
    _cache["o"] = (batch, feats, sl, el, ek.refs_of(rows))
    return _cache["o"]


def test_rescore_oracle_all_variants():
    batch, feats, sl, el, refs = _oracle()
    cache = {}
    for nv in ek.NULL_VARIANTS:
        for mal in (30, 64, 96):
            for nb in (20, 50):
                sc = ek.rescore(batch, feats, sl, el, None, nv, mal, nb, _cache=cache)
                t, _ = tune_threshold(sc, refs)
                m = squad_v2_metrics(apply_threshold(sc, t), refs)
                # max_answer_length=30 tokens legitimately truncates a few long gold answers
                assert (m["f1"] == 100 if mal >= 64 else m["f1"] > 95), (nv, mal, nb, m)
    choice, table = ek.tune_variant_grid(batch, feats, sl, el)
    assert len(table) == 18 and (choice["null_variant"], choice["max_answer_length"], choice["n_best"]) == ek.DEFAULT_VARIANT


def test_null_variants_differ():
    # one example, 3 windows with null logits sum 4, 10, 1 -> min 1, mean 5, first 4
    sl, el = np.zeros((3, 4), np.float32), np.zeros((3, 4), np.float32)
    for i, v in enumerate((4, 10, 1)):
        sl[i, 0] = v
    feats = {"example_id": ["a"] * 3, "offset_mapping": [[None, (0, 1), (1, 2), None]] * 3}
    got = {v: ek.null_scores(feats, sl, el, v)["a"] for v in ek.NULL_VARIANTS}
    assert got == {"min": 1.0, "mean": 5.0, "first": 4.0}, got
    ex = {"id": ["a"], "context": ["xy"]}
    d = {v: ek.rescore(ex, feats, sl, el, None, v)["a"]["diff"] for v in ek.NULL_VARIANTS}
    assert len({round(x, 6) for x in d.values()}) == 3, d


def test_ensemble_and_hash():
    batch, feats, sl, el, refs = _oracle()
    ids = list(feats["example_id"])

    def run(s, e, h=None):
        d = {"start": s, "end": e, "example_id": ids, "feature_hash": h or ek.feature_hash(ids)}
        return {"dev": d}
    s, e, _, _ = ek.ensemble_logits([run(sl, el), run(sl, el)])
    assert np.allclose(s, sl) and np.allclose(e, el)
    sc = ek.rescore(batch, feats, s, e)
    assert squad_v2_metrics(apply_threshold(sc, tune_threshold(sc, refs)[0]), refs)["f1"] == 100
    for bad in (run(sl, el, "other"), {"dev": dict(run(sl, el)["dev"], example_id=ids[::-1])}):
        try:
            ek.ensemble_logits([run(sl, el), bad])
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_stats():
    rng = np.random.default_rng(1)
    x = rng.random(200)
    m, lo, hi = ek.bootstrap_ci(x, n=2000)
    assert lo <= m <= hi and hi - lo > 0, (lo, m, hi)
    r = ek.paired_bootstrap(x, x, n=2000)
    # identical vectors: every resample diff == 0 -> "diff <= 0" share is exactly 1.0 (p is conservative, not 0.5)
    assert r["diff"] == 0 and r["lo"] == r["hi"] == 0 and r["p"] == 1.0, r
    r2 = ek.paired_bootstrap(x + 0.2 + rng.normal(0, 0.05, 200), x, n=2000)
    assert r2["diff"] > 0.1 and r2["p"] < 0.01 and r2["lo"] > 0, r2
    pe = ek.per_example({"1": "a", "2": ""}, {"1": ["a"], "2": ["b"]})
    assert list(pe.f1) == [1.0, 0.0] and list(pe.has_ans) == [True, True]
    t = ek.ci_table(pe, n=200)
    assert set(t.subset) == {"overall", "HasAns"}


def test_test_guard():
    for call in (lambda: ek.build_eval_context("xlmr_base", "test"),
                 lambda: ek.ensemble_logits([{"test": {}}], split="test"),
                 lambda: ek.build_eval_context("xlmr_base", "oops")):
        try:
            call()
            raise AssertionError("expected guard error")
        except (PermissionError, ValueError):
            pass


def test_load_run_synthetic():
    ids = np.array(["a", "a", "b"])
    st = np.random.default_rng(0).normal(size=(3, 8))
    with tempfile.TemporaryDirectory() as d:
        run = Path(d) / "sweep" / "s1" / "xlmr_base" / "r1"
        run.mkdir(parents=True)
        save_logits(run / "dev_logits.npz", {"example_id": ids}, st, st)
        save_logits(run / "test_logits.npz", {"example_id": ids}, st, st)
        s = {"cfg": {"seed": 1, "lr": 3e-5, "epochs": 3}, "train_runtime_s": 10.0, "windows": {"train": 100},
             "dev": {"best_threshold": 0.5, "metrics_at_best": {"f1": 50.0, "exact": 40.0}}}
        (run / "summary.json").write_text(json.dumps(s))
        r = ek.load_run(run)
        assert "test" not in r and r["dev"]["start"].dtype == np.float32
        assert r["dev"]["feature_hash"] == ek.feature_hash(ids)
        assert "test" in ek.load_run(run, allow_test=True)
        df = ek.load_runs(Path(d))
        assert len(df) == 1 and df.iloc[0].dev_f1 == 50.0 and df.iloc[0].model == "xlmr_base"
        assert len(ek.layer_table(df, "xlmr_base")) == 1 and len(ek.cost_table(df)) == 1
    assert ek.load_runs(Path(d) / "nope").empty and ek.layer_table(ek.load_runs(Path(d) / "nope"), "x").empty


def test_smoke_plumbing():
    run = ROOT / "outputs" / "smoke_x"
    if not (run / "dev_logits.npz").exists() or not DATA.exists():
        pytest.skip("smoke logits or data missing")
    r = ek.load_run(run)
    n = r["summary"]["windows"]["dev_examples"]
    ex, feats, extract = ek.build_eval_context("xlmr_base", "dev", max_n=n, expected_hash=r["dev"]["feature_hash"])
    assert extract is None and len(feats["example_id"]) == len(r["dev"]["example_id"])
    sc = ek.rescore(ex, feats, r["dev"]["start"], r["dev"]["end"])
    refs = ek.refs_of(ex)
    t, _ = tune_threshold(sc, refs)
    f1 = squad_v2_metrics(apply_threshold(sc, t), refs)["f1"]
    assert abs(f1 - r["summary"]["dev"]["metrics_at_best"]["f1"]) < 2.0, f1  # float16 round-trip only
    try:
        ek.build_eval_context("xlmr_base", "dev", max_n=n, expected_hash="bad")
        raise AssertionError("expected hash error")
    except ValueError:
        pass


def test_vi_mrc_branch():
    assert ek.PARAMS_M["vi_mrc_base"] == 277.5 and ek.PARAMS_M["vi_mrc_large"] == 559.9
    try:
        ek.build_eval_context("vi_mrc_base", "test")
        raise AssertionError("expected PermissionError")
    except PermissionError:
        pass  # guard fires before any tokenizer load
    if not DATA.exists():
        pytest.skip(f"{DATA} missing")
    try:
        ex, feats, extract = ek.build_eval_context("vi_mrc_base", "dev", max_n=5)
    except OSError as e:  # tokenizer not cached / offline
        pytest.skip(f"vi-mrc tokenizer unavailable: {type(e).__name__}")
    assert extract is None and len(feats["example_id"]) >= 5
