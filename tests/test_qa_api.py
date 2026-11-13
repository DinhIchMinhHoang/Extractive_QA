"""Run: .venv\\Scripts\\python tests\\test_qa_api.py"""
import sys

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from transformers import AutoTokenizer, RobertaConfig, RobertaForQuestionAnswering  # noqa: E402

from src.qa_api import QA, answer, load_qa  # noqa: E402


try:
    TOK = AutoTokenizer.from_pretrained("xlm-roberta-base", local_files_only=True)
except Exception as e:  # no cached tokenizer / no network
    TOK = None
    WHY = f"xlm-roberta-base tokenizer not cached offline: {type(e).__name__}"


def _qa(threshold=0.0):
    if TOK is None:
        pytest.skip(WHY)
    torch.manual_seed(0)
    m = RobertaForQuestionAnswering(RobertaConfig(
        vocab_size=250002, hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
        max_position_embeddings=514, type_vocab_size=1, pad_token_id=1)).eval()
    return QA(threshold=threshold, device="cpu", tok=TOK, model=m)


def _check(r, ctx, thr):
    assert set(r) == {"text", "score", "null_diff", "has_answer"}, r
    assert r["text"] == "" or r["text"] in ctx, r
    assert r["has_answer"] == (r["null_diff"] <= thr), r
    assert r["has_answer"] or r["text"] == "", r


def test_keys_and_substring():
    ctx = "Ha Noi la thu do cua Viet Nam."
    for thr in (-1e9, 0.0, 1e9):
        _check(answer("Thu do la gi?", ctx, qa=_qa(thr)), ctx, thr)


def test_long_context():
    ctx = " ".join(f"word{i}" for i in range(900))
    _check(answer("Thu do la gi?", ctx, qa=_qa(1e9)), ctx, 1e9)


def _release(root, with_files=True):
    import json
    rel = root / "outputs" / "release"
    rel.mkdir(parents=True)
    (rel / "best_models.json").write_text(json.dumps({
        "xlmr_large": {"run": "r2", "dir": "outputs/release/xlmr_large/r2", "threshold": -1.5, "dev_f1": 77.7},
        "ensembles": {"xlmr_large": ["r1", "r2", "r3"], "ensemble_thresholds": {"xlmr_large": -0.25}}}))
    for r in ("r1", "r2", "r3") if with_files else ():
        (rel / "xlmr_large" / r).mkdir(parents=True)
        (rel / "xlmr_large" / r / "model.safetensors").write_text("")
    return root


def test_release_plan_ensemble_default_and_single(tmp_path):
    from src.qa_api import release_plan
    root = _release(tmp_path)
    dirs, thr = release_plan(root=root)
    assert dirs == [root / "outputs/release/xlmr_large" / r for r in ("r1", "r2", "r3")] and thr == -0.25
    assert release_plan(ensemble=False, root=root) == ([root / "outputs/release/xlmr_large/r2"], -1.5)


def test_release_plan_missing_member_weights_falls_to_single(tmp_path):
    from src.qa_api import release_plan
    assert release_plan(root=_release(tmp_path, with_files=False))[1] == -1.5


def test_load_qa_ensemble_default_single_and_legacy():
    import json
    from src import qa_api
    rel = json.loads((qa_api.ROOT / "outputs" / "release" / "best_models.json").read_text())
    b, runs = rel["xlmr_large"], rel["ensembles"]["xlmr_large"]
    qa = load_qa()  # lazy: no weights loaded
    assert qa.members and len(qa.members) == len(runs) and qa.threshold == rel["ensembles"]["ensemble_thresholds"]["xlmr_large"]
    single = load_qa(ensemble=False)
    assert single.release and not single.members and single.dir == qa_api.ROOT / b["dir"] and single.threshold == b["threshold"]
    for leg in (load_qa(legacy=True), load_qa(legacy=True, ensemble=True)):
        assert not leg.release and not leg.members and leg.dir == qa_api.DEFAULT_DIR and leg.threshold is None


def test_mean_logits_tiny():
    import numpy as np
    from src.qa_api import mean_logits
    s, e = mean_logits([(np.array([[0., 2.]]), np.array([[4., 4.]])), (np.array([[2., 4.]]), np.array([[0., 2.]]))])
    assert s.tolist() == [[1., 3.]] and e.tolist() == [[2., 3.]]


class _Fake(torch.nn.Linear):  # 4x4 fp32 Linear = 80 bytes; records .to() targets
    def __init__(self):
        super().__init__(4, 4)
        self.moves = []

    def to(self, *a, **k):
        self.moves.append(a[0])
        return self


def _use(c, key, device="cpu", free=None):
    with c.use(key, lambda: (key, _Fake()), device, free) as (obj, m):
        assert obj == key
    return m


def test_cache_lru_budget_and_eviction_order():
    from src.qa_api import ModelCache
    c = ModelCache(budget_gb=170 / 2**30)  # fits 2 models of 80 bytes
    a = _use(c, "a")
    _use(c, "b")
    assert _use(c, "a") is a and list(c.items) == ["b", "a"]  # hit -> most recent
    _use(c, "c")  # evicts LRU "b"
    assert list(c.items) == ["a", "c"] and c.used() == 160
    assert _use(c, "a") is a  # still cached, no reload


def test_cache_over_budget_not_kept_and_free_called():
    from src.qa_api import ModelCache
    freed = []
    c = ModelCache(budget_gb=40 / 2**30)
    _use(c, "a", free=freed.append)
    assert not c.items and freed == ["a"]


def test_cache_moves_to_device_and_back():
    from src.qa_api import ModelCache
    c = ModelCache(budget_gb=1)
    m = _use(c, "a", "cuda")  # miss: loader puts it on device; cache sends it back to cpu
    assert m.moves == ["cpu"]
    assert _use(c, "a", "cuda") is m and m.moves == ["cpu", "cuda", "cpu"]
    m2 = _use(c, "b", "cpu")  # cpu mode: no moves at all
    assert _use(c, "b", "cpu") is m2 and m2.moves == []


def test_cache_disabled_by_env():
    from src.qa_api import ModelCache
    assert not ModelCache.from_env({"QA_CACHE": "0"}).enabled
    c = ModelCache.from_env({"QA_CACHE": "0", "QA_CACHE_GB": "1"})
    assert _use(c, "a") is not _use(c, "a") and not c.items
    on = ModelCache.from_env({"QA_CACHE_GB": "2"})
    assert on.enabled and on.budget == 2 * 2**30


def test_default_budget_capped():
    from src.qa_api import default_budget_gb, total_ram_gb
    assert default_budget_gb() == min(6.0, 0.4 * total_ram_gb())


def test_real_model_smoke():
    for qa in (load_qa(), load_qa(ensemble=False)):
        dirs = qa.members or [qa.dir]
        if not all((d / ("model.safetensors" if qa.release or qa.members else "model")).exists() for d in dirs):
            pytest.skip("release xlmr_large not present")
    ctx = "Hà Nội là thủ đô của Việt Nam. Thành phố Hồ Chí Minh là thành phố đông dân nhất cả nước."
    qa = load_qa()
    for q in ["Thủ đô của Việt Nam là gì?", "Ai là tổng thống Pháp?"]:
        r = answer(q, ctx, qa=qa)
        assert r["n_models"] == len(qa.members)
        print("   ", q, "->", r)
