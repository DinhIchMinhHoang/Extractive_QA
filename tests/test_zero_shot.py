"""Run: .venv\\Scripts\\python tests\\test_zero_shot.py"""
import json
import sys

import pytest
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import Dataset, DatasetDict  # noqa: E402
from transformers import AutoTokenizer, RobertaConfig, RobertaForQuestionAnswering, RobertaModel  # noqa: E402

import src.evalkit as ek  # noqa: E402
import src.zero_shot as zs  # noqa: E402
from src.train_qa import save_logits  # noqa: E402


try:
    TOK = AutoTokenizer.from_pretrained("xlm-roberta-base", local_files_only=True)
except Exception as e:  # no cached tokenizer / no network
    TOK = None
    WHY = f"xlm-roberta-base tokenizer not cached offline: {type(e).__name__}"


def _need_tok():
    if TOK is None:
        pytest.skip(WHY)


def _rows(n_ex=3):
    long_ctx = " ".join(f"word{i}" for i in range(600))
    ctxs = ["Ha Noi la thu do cua Viet Nam.", long_ctx, "Con meo ngoi tren tham."]
    return [{"id": f"e{i}", "question": "Thu do la gi?", "context": ctxs[i],
             "answers": {"text": [], "answer_start": []}} for i in range(n_ex)]


def _model():
    torch.manual_seed(0)
    return RobertaForQuestionAnswering(RobertaConfig(
        vocab_size=250002, hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
        max_position_embeddings=514, type_vocab_size=1, pad_token_id=1)).eval()


def test_predict_bs_invariant():
    _need_tok()
    feats = zs.build_split_features(Dataset.from_list(_rows()), TOK)
    assert len(feats) >= 4
    s1, e1 = zs.predict_logits_hf(_model(), feats, bs=1, device="cpu")
    s3, e3 = zs.predict_logits_hf(_model(), feats, bs=3, device="cpu")
    assert s1.shape == (len(feats), 384) and s1.dtype == np.float32
    np.testing.assert_allclose(s1, s3, atol=1e-4)
    np.testing.assert_allclose(e1, e3, atol=1e-4)


def test_save_logits_roundtrip():
    _need_tok()
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        feats = zs.build_split_features(Dataset.from_list(_rows()), TOK)
        s, e = zs.predict_logits_hf(_model(), feats, bs=2, device="cpu")
        save_logits(tmp_path / "d.npz", feats, s, e)
        z = ek._load_npz(tmp_path / "d.npz")
        assert z["feature_hash"] == ek.feature_hash(feats["example_id"])
        assert z["example_id"] == list(feats["example_id"])
        np.testing.assert_allclose(z["start"], s, atol=0.05, rtol=1e-2)


def test_load_guard():
    _need_tok()
    with tempfile.TemporaryDirectory() as td:
        cfg = RobertaConfig(vocab_size=250002, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                            intermediate_size=64, max_position_embeddings=514, type_vocab_size=1, pad_token_id=1)
        RobertaModel(cfg).save_pretrained(td)
        TOK.save_pretrained(td)
        try:
            zs.load_model(td, device="cpu")
        except AssertionError:
            return
        raise AssertionError("load_model should reject a model without a QA head")


def test_run_never_scores_test():
    _need_tok()
    dev_ids = []

    def fake_score(dev_ex, dev_f, s, e, cfg):
        dev_ids.append(list(dev_ex["id"]))
        m = {"f1": 1.0, "exact": 1.0}
        return {"best_threshold": 0.0, "table": [], "metrics_at_0": m, "metrics_at_best": m}

    orig_score, orig_load = zs.score_from_logits, zs.load_model
    zs.score_from_logits = fake_score
    zs.load_model = lambda repo, device: (TOK, _model())
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            ds = DatasetDict({"dev": Dataset.from_list(_rows()[:2]),
                              "test": Dataset.from_list([dict(r, id="t" + r["id"]) for r in _rows()])})
            summ = zs.run("vi_mrc_base", tmp_path, save_test=True, bs=4, device="cpu", ds=ds)
            assert dev_ids == [["e0", "e1"]]
            assert not any("test" in k and k not in ("feature_hash_test",) for k in summ if k != "windows")
            assert summ["windows"]["test"] >= 3 and "metrics" not in json.dumps(summ["windows"])
            out = tmp_path / "vi_mrc_base" / "vi_mrc_base__zs"
            assert (out / "test_logits.npz").exists() and (out / "dev_logits.npz").exists()
            assert summ["cfg"]["zero_shot"] is True
    finally:
        zs.score_from_logits, zs.load_model = orig_score, orig_load
