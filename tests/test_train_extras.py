"""Run: .venv\\Scripts\\python tests\\test_train_extras.py  (LLRD, EMA, plausible loss, plausible positions, seeds, smoke)"""
import json
import subprocess
import sys

import pytest
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.metrics import normalize_answer  # noqa: E402
from src.preprocess import prepare_eval_features, prepare_train_features  # noqa: E402
from src.train_qa import EMA, _qa_trainer_cls, llrd_param_groups, plausible_term  # noqa: E402

DATA = ROOT / "data" / "processed" / "uit_viquad2_split"
_cache = {}


def _tiny(seed=0):
    from transformers import RobertaConfig, RobertaForQuestionAnswering
    torch.manual_seed(seed)
    cfg = RobertaConfig(vocab_size=100, hidden_size=32, num_hidden_layers=3, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=40, type_vocab_size=1)
    return RobertaForQuestionAnswering(cfg)


def _args(tmp, **kw):
    from transformers import TrainingArguments
    base = dict(output_dir=tmp, use_cpu=True, report_to="none", save_strategy="no", disable_tqdm=True,
                remove_unused_columns=False, per_device_train_batch_size=4, per_device_eval_batch_size=8,
                learning_rate=1e-3, logging_steps=1, seed=7, data_seed=7)
    base.update(kw)
    return TrainingArguments(**base)


def _tiny_ds(n=8, L=16, seed=0):
    from datasets import Dataset
    g = np.random.default_rng(seed)
    return Dataset.from_dict({
        "input_ids": g.integers(3, 100, (n, L)).tolist(), "attention_mask": [[1] * L] * n,
        "start_positions": g.integers(0, L, n).tolist(), "end_positions": g.integers(0, L, n).tolist(),
        "plausible_start": g.integers(0, L, n).tolist(), "plausible_end": g.integers(0, L, n).tolist()})


def _batch(ds):
    return {k: torch.tensor(ds[k]) for k in ds.column_names}


def test_llrd_groups():
    m, lr, dec, wd = _tiny(), 1e-3, 0.9, 0.01
    groups = llrd_param_groups(m, lr, dec, wd)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(m.parameters())), "param missing or duplicated"
    L = m.config.num_hidden_layers
    by_param = {id(p): g for g in groups for p in g["params"]}
    for n, p in m.named_parameters():
        g = by_param[id(p)]
        assert g["weight_decay"] == (0.0 if ("bias" in n or "LayerNorm" in n) else wd), n
        if "embeddings" in n:
            assert abs(g["lr"] - lr * dec ** (L + 1)) < 1e-15, n
        elif "encoder.layer." in n:
            i = int(n.split("encoder.layer.")[1].split(".")[0])
            assert abs(g["lr"] - lr * dec ** (L - i)) < 1e-15, n
        else:
            assert n.startswith("qa_outputs") and g["lr"] == lr, n
    lrs = [lr * dec ** (L + 1)] + [lr * dec ** (L - i) for i in range(L)] + [lr]
    assert all(b > a for a, b in zip(lrs, lrs[1:])) and abs(lrs[1] / lrs[0] - 1 / dec) < 1e-9
    with tempfile.TemporaryDirectory() as tmp:  # same groups through the Trainer hook
        t = _qa_trainer_cls()(model=m, args=_args(tmp, weight_decay=wd), lr_decay=dec)
        opt = t.create_optimizer()
        assert sorted(g["lr"] for g in opt.param_groups)[-1] == lr and len(opt.param_groups) == len(groups)


def test_ema_formula_and_eval_restore():
    lin = torch.nn.Linear(1, 1)
    with torch.no_grad():
        lin.weight.fill_(1.0); lin.bias.fill_(0.0)
    ema, d, w, exp = EMA(lin, 0.99), 0.99, 1.0, 1.0
    for k in range(1, 6):
        with torch.no_grad():
            lin.weight.fill_(1.0 + k)
        ema.update(lin)
        dk = min(d, (1 + k) / (10 + k))
        exp = dk * exp + (1 - dk) * (1.0 + k)
    assert abs(ema.shadow["weight"].item() - exp) < 1e-6, (ema.shadow["weight"].item(), exp)
    with ema.applied(lin):
        assert abs(lin.weight.item() - exp) < 1e-6
    assert lin.weight.item() == 6.0, "raw weights not restored"
    # Trainer.evaluate must use the EMA weights and leave the raw ones in place
    m, ds = _tiny(), _tiny_ds()
    ev = ds.remove_columns(["plausible_start", "plausible_end"])
    with tempfile.TemporaryDirectory() as tmp:
        t = _qa_trainer_cls()(model=m, args=_args(tmp), eval_dataset=ev)
        t.ema = EMA(m, 0.9)
        with torch.no_grad():
            for p in m.parameters():
                p.add_(0.05)  # raw drifts away from the shadow
        raw = {n: p.clone() for n, p in m.named_parameters()}
        m.eval()
        with t.ema.applied(m), torch.no_grad():
            want = m(**_batch(ev)).loss.item()
        with torch.no_grad():
            raw_l = m(**_batch(ev)).loss.item()
        got = t.evaluate()["eval_loss"]
        assert abs(got - want) < 1e-4 and abs(got - raw_l) > 1e-6, (got, want, raw_l)
        assert all(torch.equal(raw[n], p) for n, p in m.named_parameters()), "raw weights not restored"


def test_plausible_loss():
    m, ds = _tiny(), _tiny_ds()
    ds = ds.map(lambda r: {"plausible_start": max(r["plausible_start"], 1), "plausible_end": max(r["plausible_end"], 1)})
    with tempfile.TemporaryDirectory() as tmp:
        Q = _qa_trainer_cls()
        t0 = Q(model=m, args=_args(tmp), plausible_lambda=0.0)
        t1 = Q(model=m, args=_args(tmp), plausible_lambda=0.5)
        m.train()
        b = _batch(ds)
        base = t0.compute_loss(m, dict(b)).item()
        torch.manual_seed(1); l0 = t0.compute_loss(m, dict(b))  # same dropout masks for compared losses
        torch.manual_seed(1); l1 = t1.compute_loss(m, dict(b))
        assert l1.item() > l0.item(), (l0.item(), l1.item())
        l1.backward()
        assert m.qa_outputs.weight.grad.abs().sum() > 0
        # zero positions -> exactly the plain loss
        z = dict(b, plausible_start=torch.zeros_like(b["plausible_start"]), plausible_end=torch.zeros_like(b["plausible_end"]))
        torch.manual_seed(1); lz = t1.compute_loss(m, dict(z))
        assert lz.item() == l0.item(), (lz.item(), l0.item())
        # eval mode: aux term not added
        m.eval()
        with torch.no_grad():
            e0, e1 = t0.compute_loss(m, dict(b)).item(), t1.compute_loss(m, dict(b)).item()
        assert e0 == e1, (e0, e1)
        assert base > 0
    sl = torch.randn(2, 5); el = torch.randn(2, 5)
    assert plausible_term(sl, el, torch.tensor([0, 0]), torch.tensor([0, 0])).item() == 0.0
    assert plausible_term(sl, el, torch.tensor([2, 0]), torch.tensor([3, 0])).item() > 0


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


def _b(rows, keys=("id", "question", "context", "answers", "plausible_answers")):
    return {k: [r[k] for r in rows] for k in keys}


def test_plausible_positions():
    tok, ds = _tok(), _data()
    no = [r for r in ds["train"].select(range(3000)) if not r["answers"]["text"]
          and r["plausible_answers"] and r["plausible_answers"]["text"]][:100]
    assert len(no) == 100
    f = prepare_train_features(_b(no), tok, with_plausible=True)
    ev = prepare_eval_features(_b(no), tok)
    by_id = {r["id"]: r for r in no}
    n_in = 0
    for i, eid in enumerate(ev["example_id"]):
        ps, pe = f["plausible_start"][i], f["plausible_end"][i]
        assert f["start_positions"][i] == 0, "NoAns row must keep the null label"
        if ps > 0:
            r, o = by_id[eid], ev["offset_mapping"][i]
            got = normalize_answer(r["context"][o[ps][0]:o[pe][1]])
            want = normalize_answer(r["plausible_answers"]["text"][0])
            assert got == want or want in got, (got, want)  # contain: source spans can start mid-word
            n_in += 1
        else:
            assert pe == 0
    assert n_in >= 80, n_in
    has = [r for r in ds["train"].select(range(300)) if r["answers"]["text"]][:50]
    fh = prepare_train_features(_b(has), tok, with_plausible=True)
    assert set(fh["plausible_start"]) == {0} and set(fh["plausible_end"]) == {0}
    # missing / None plausible_answers -> zeros, no crash
    f2 = prepare_train_features(_b(no, ("id", "question", "context", "answers")), tok, with_plausible=True)
    assert set(f2["plausible_start"]) == {0}
    b3 = _b(no); b3["plausible_answers"] = [None] * len(no)
    assert set(prepare_train_features(b3, tok, with_plausible=True)["plausible_end"]) == {0}


def _old_train_features(examples, tokenizer, max_length=384, doc_stride=128):
    """Verbatim copy of the pre-W1 prepare_train_features, kept as the byte-identical reference."""
    from src.preprocess import _tokenize
    tok, right = _tokenize(examples, tokenizer, max_length, doc_stride)
    sample_map = tok.pop("overflow_to_sample_mapping")
    offsets_all = tok.pop("offset_mapping")
    tok["start_positions"], tok["end_positions"] = [], []
    for i, offsets in enumerate(offsets_all):
        input_ids = tok["input_ids"][i]
        cls = input_ids.index(tokenizer.cls_token_id)
        seq = tok.sequence_ids(i)
        ans = examples["answers"][sample_map[i]]
        if len(ans["answer_start"]) == 0:
            tok["start_positions"].append(cls); tok["end_positions"].append(cls)
            continue
        s_char = ans["answer_start"][0]
        e_char = s_char + len(ans["text"][0])
        ctx_id = 1 if right else 0
        ts = 0
        while seq[ts] != ctx_id:
            ts += 1
        te = len(input_ids) - 1
        while seq[te] != ctx_id:
            te -= 1
        if not (offsets[ts][0] <= s_char and offsets[te][1] >= e_char):
            tok["start_positions"].append(cls); tok["end_positions"].append(cls)
            continue
        while ts < len(offsets) and offsets[ts][0] <= s_char:
            ts += 1
        tok["start_positions"].append(ts - 1)
        while offsets[te][1] >= e_char:
            te -= 1
        tok["end_positions"].append(te + 1)
    return tok


def test_default_features_unchanged():
    tok, ds = _tok(), _data()
    rows = list(ds["train"].select(range(50)))
    for kw in (dict(), dict(max_length=64, doc_stride=16)):  # short windows exercise the outside-window branch
        new = prepare_train_features(_b(rows), tok, **kw)
        old = _old_train_features(_b(rows), tok, **kw)
        assert set(new.keys()) == set(old.keys()) and "plausible_start" not in new
        assert all(list(new[k]) == list(old[k]) for k in old), "default features changed"
        wp = prepare_train_features(_b(rows), tok, with_plausible=True, **kw)
        assert all(list(wp[k]) == list(old[k]) for k in old), "with_plausible changed the labels"


def test_seed_reproducible():
    from transformers import set_seed
    losses = []
    for _ in range(2):
        set_seed(3)
        m = _tiny(seed=3)
        with tempfile.TemporaryDirectory() as tmp:
            t = _qa_trainer_cls()(model=m, args=_args(tmp, max_steps=4), train_dataset=_tiny_ds(16),
                                  lr_decay=0.9, ema_decay=0.9, plausible_lambda=0.1)
            res = t.train()
            assert t.ema is None
        losses.append(res.training_loss)
    assert losses[0] == losses[1] and np.isfinite(losses[0]), losses


def _smoke(args, phobert=False):
    name = "smoke_phobert_x" if phobert else "smoke_x"
    cmd = [sys.executable, "-m", "src.train_qa", "--smoke", "--run-name", name, "--lr-decay", "0.9", "--ema", "0.99",
           "--plausible-lambda", "0.1"] + args
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr[-2000:]
    out = ROOT / "outputs" / name
    s = json.loads((out / "summary.json").read_text())
    z = np.load(out / "dev_logits.npz")
    n = s["windows"]["dev"]
    L = s["cfg"]["max_length"]
    assert z["start_logits"].shape == z["end_logits"].shape == (n, L) and z["start_logits"].dtype == np.float16
    assert len(z["example_id"]) == n and str(z["feature_hash"])
    assert (out / "dev_threshold.json").exists() and (out / "model").exists()
    print("  smoke summary:", {"windows": s["windows"], "best_eval_loss": s.get("best_eval_loss"),
                                "f1@0": round(s["dev"]["metrics_at_0"]["f1"], 2), "wall_s": round(s["wall_s"], 1)})


def test_smoke_xlmr():
    if not DATA.exists():
        pytest.skip("data missing")
    _smoke([])


def test_smoke_phobert():
    seg = ROOT / "data" / "processed" / "seg" / "train.jsonl"
    if not seg.exists():
        pytest.skip("seg/train.jsonl missing")
    first = json.loads(seg.open(encoding="utf-8").readline())
    if "plausible_answers" not in first:
        pytest.skip("seg data has no plausible_answers yet")
    _smoke(["--phobert"], phobert=True)
