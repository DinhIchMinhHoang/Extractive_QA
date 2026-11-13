"""Fine-tune xlm-roberta-base for Vietnamese extractive QA (SQuAD-2.0 style) and tune the null threshold on dev.

CLI: python -m src.train_qa [--smoke] [--out DIR]
"""
import argparse
import contextlib
import hashlib
import inspect
import json
import random
import re
import shutil
import time
from pathlib import Path

import numpy as np

from src.data_clean import clean_split
from src.postprocess import apply_threshold, score_examples, tune_threshold
from src.metrics import squad_v2_metrics
from src.phobert import PHOBERT, get_seg_data, load_phobert_tokenizer, make_extract
from src.preprocess import prepare_eval_features, prepare_train_features

ROOT = Path(__file__).resolve().parent.parent
LOCAL_SPLIT = ROOT / "data" / "processed" / "uit_viquad2_split"
EXPECTED = {"train": 25542, "dev": 2912, "test": 3814}

CFG = dict(
    model_name="xlm-roberta-base", max_length=384, doc_stride=128, lr=3e-5, epochs=3,
    train_bs=32, eval_bs=64, grad_accum=1, warmup_ratio=0.1, weight_decay=0.01, seed=42,
    n_best=20, max_answer_length=64, num_workers=2, max_steps=-1, max_train=None, max_dev=None,
    out_dir=str(ROOT / "outputs" / "xlmr_base"),
    # model-agnostic add-ons (defaults = old behaviour)
    lr_decay=None, ema_decay=None, plausible_lambda=0.0, data_seed=None, run_name=None,
    save_model=True, save_logits=True, save_test_logits=False,
)
CFG_PHOBERT = dict(CFG, model_name=PHOBERT, max_length=256, doc_stride=128, lr=3e-5, epochs=3,
                   train_bs=32, eval_bs=64, grad_accum=1, out_dir=str(ROOT / "outputs" / "phobert_base"),
                   phobert=True)
SMOKE = dict(max_train=64, max_dev=32, epochs=1, train_bs=4, eval_bs=8, max_steps=5,
             num_workers=0, out_dir=str(ROOT / "outputs" / "smoke"))


def get_data(root=None):
    """DatasetDict(train/dev/test): from disk if present, else rebuilt from the hub + split_ids.json."""
    from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

    root = Path(root) if root else ROOT
    local = root / "data" / "processed" / "uit_viquad2_split"
    if local.exists():
        ds = load_from_disk(str(local))
        return DatasetDict({k: ds[k] for k in EXPECTED})

    ids_path = next((p for p in (root / "split_ids.json", root / "data" / "processed" / "split_ids.json")
                     if p.exists()), None)
    if ids_path is None:
        raise FileNotFoundError(f"split_ids.json not found under {root}")
    ids = json.loads(ids_path.read_text())
    raw = load_dataset("taidng/UIT-ViQuAD2.0")
    train_rows, _ = clean_split(raw["train"], True)
    test_rows, _ = clean_split(raw["validation"], False)
    dev_ids, tr_ids = set(ids["dev"]), set(ids["train"])
    assert dev_ids.isdisjoint(tr_ids), "train/dev ids overlap"
    dev_rows = [r for r in train_rows if r["id"] in dev_ids]
    train_rows = [r for r in train_rows if r["id"] in tr_ids]
    ds = DatasetDict({"train": Dataset.from_list(train_rows), "dev": Dataset.from_list(dev_rows),
                      "test": Dataset.from_list(test_rows)})
    got = {k: len(v) for k, v in ds.items()}
    assert got == EXPECTED, got
    return ds


def build_features(ds, tokenizer, cfg, max_train=None, max_dev=None):
    """Return (train_feats, dev_loss_feats, dev_eval_feats, dev_examples); prints window counts."""
    train = ds["train"].select(range(max_train)) if max_train else ds["train"]
    dev = ds["dev"].select(range(max_dev)) if max_dev else ds["dev"]
    kw = dict(tokenizer=tokenizer, max_length=cfg["max_length"], doc_stride=cfg["doc_stride"])
    pl = bool(cfg.get("plausible_lambda"))
    tr = train.map(lambda b: prepare_train_features(b, with_plausible=pl, **kw), batched=True,
                   remove_columns=train.column_names)
    dl = dev.map(lambda b: prepare_train_features(b, **kw), batched=True, remove_columns=dev.column_names)
    de = dev.map(lambda b: prepare_eval_features(b, **kw), batched=True, remove_columns=dev.column_names)
    print(f"windows: train {len(tr)} from {len(train)} examples ({len(tr) / len(train):.3f}/ex), "
          f"dev {len(de)} from {len(dev)} examples ({len(de) / len(dev):.3f}/ex)")
    return tr, dl, de, dev


def llrd_param_groups(model, lr, decay, weight_decay):
    """Layer-wise lr decay groups for RoBERTa-style QA models (XLM-R, PhoBERT).

    embeddings lr*decay^(L+1), encoder layer i lr*decay^(L-i), anything else (qa_outputs) lr.
    bias / LayerNorm params get no weight decay. Every trainable parameter is in exactly one group.
    """
    n_layers = model.config.num_hidden_layers
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        m = re.search(r"encoder\.layer\.(\d+)\.", n)
        if m:
            depth = n_layers - int(m.group(1))
        elif ".embeddings." in "." + n:
            depth = n_layers + 1
        else:
            depth = 0
        wd = 0.0 if ("bias" in n or "LayerNorm" in n) else weight_decay
        g = groups.setdefault((depth, wd), {"params": [], "lr": lr * decay ** depth, "weight_decay": wd})
        g["params"].append(p)
    return list(groups.values())


class EMA:
    """Exponential moving average of the trainable parameters (fp32, same device as the params)."""

    def __init__(self, model, decay):
        self.decay, self.n = decay, 0
        self.shadow = {n: p.detach().clone().float() for n, p in model.named_parameters() if p.requires_grad}

    @property
    def current_decay(self):
        return min(self.decay, (1 + self.n) / (10 + self.n))  # usual warm-up

    def update(self, model):
        import torch
        self.n += 1
        d = self.current_decay
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.shadow:
                    self.shadow[n].mul_(d).add_(p.detach().float(), alpha=1 - d)

    def copy_to(self, model):
        import torch
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.shadow:
                    p.copy_(self.shadow[n].to(p.dtype))

    @contextlib.contextmanager
    def applied(self, model):
        """Temporarily load the EMA weights into the model, restore the raw ones on exit."""
        import torch
        raw = {n: p.detach().clone() for n, p in model.named_parameters() if n in self.shadow}
        self.copy_to(model)
        try:
            yield
        finally:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in raw:
                        p.copy_(raw[n])


def _qa_trainer_cls():
    from transformers import Trainer, TrainerCallback

    class _EMACallback(TrainerCallback):
        def __init__(self, trainer):
            self.t = trainer

        def on_train_begin(self, args, state, control, model=None, **kw):
            if self.t.ema_decay:
                self.t.ema = EMA(self.t.model, self.t.ema_decay)

        def on_step_end(self, args, state, control, model=None, **kw):  # after the optimizer step
            if self.t.ema is not None:
                self.t.ema.update(self.t.model)

    class QATrainer(Trainer):
        """Trainer + LLRD, EMA weights (used for eval/saves/final model) and the plausible-answer aux loss."""

        def __init__(self, *a, lr_decay=None, ema_decay=None, plausible_lambda=0.0, **kw):
            super().__init__(*a, **kw)
            self.lr_decay, self.ema_decay, self.plausible_lambda = lr_decay, ema_decay, plausible_lambda
            self.ema = None
            self.add_callback(_EMACallback(self))

        def create_optimizer(self, model=None):
            if not self.lr_decay:
                return super().create_optimizer(model)
            if self.optimizer is None:
                opt_model = self.model if model is None else model
                groups = llrd_param_groups(opt_model, self.args.learning_rate, self.lr_decay, self.args.weight_decay)
                cls, kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
                self.optimizer = cls(groups, **kwargs)
            return self.optimizer

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            inputs = dict(inputs)
            ps, pe = inputs.pop("plausible_start", None), inputs.pop("plausible_end", None)
            loss, out = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
            if self.plausible_lambda > 0 and model.training and ps is not None:
                loss = loss + self.plausible_lambda * plausible_term(out.start_logits, out.end_logits, ps, pe)
            return (loss, out) if return_outputs else loss

        # EMA weights are swapped in for every eval/predict and every checkpoint write, so best-checkpoint
        # selection (eval_loss) and load_best_model_at_end both see the EMA weights.
        def _ema_ctx(self):
            return self.ema.applied(self.model) if self.ema is not None else contextlib.nullcontext()

        def evaluation_loop(self, *a, **kw):
            with self._ema_ctx():
                return super().evaluation_loop(*a, **kw)

        def _save(self, *a, **kw):
            with self._ema_ctx():
                return super()._save(*a, **kw)

        def train(self, *a, **kw):
            res = super().train(*a, **kw)
            if self.ema is not None:
                if not self.state.best_model_checkpoint:  # best (EMA) checkpoint not reloaded -> use EMA directly
                    self.ema.copy_to(self.model)
                # shortcut: after train() the model holds EMA weights and EMA is switched off (no resume), upgrade if resuming is needed
                self.ema = None
            return res

    return QATrainer


def plausible_term(start_logits, end_logits, ps, pe):
    """Unlikelihood on the plausible span: batch mean of -log(1-p_start[ps]) - log(1-p_end[pe]), 0 where ps/pe == 0."""
    import torch
    mask = ((ps > 0) & (pe > 0)).float()
    idx = torch.arange(len(ps), device=ps.device)
    p_s = torch.softmax(start_logits.float(), -1)[idx, ps]
    p_e = torch.softmax(end_logits.float(), -1)[idx, pe]
    term = -torch.log(1 - p_s + 1e-6) - torch.log(1 - p_e + 1e-6)
    return (term * mask).mean()


MODEL_COLS = {"input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions",
              "plausible_start", "plausible_end"}


def make_trainer(cfg, tokenizer, train_feats, dev_loss_feats):
    import torch
    from transformers import AutoModelForQuestionAnswering, TrainingArguments, default_data_collator

    cuda = torch.cuda.is_available()
    seed = cfg["seed"]
    data_seed = cfg.get("data_seed") if cfg.get("data_seed") is not None else seed
    drop = lambda f: f.remove_columns([c for c in f.column_names if c not in MODEL_COLS])  # noqa: E731
    params = inspect.signature(TrainingArguments.__init__).parameters
    ev_key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    # transformers 5 dropped warmup_ratio; a float < 1 in warmup_steps means the same thing
    wu_key = "warmup_ratio" if "warmup_ratio" in params else "warmup_steps"
    args = TrainingArguments(
        output_dir=str(Path(cfg["out_dir"]) / "ckpt"),
        learning_rate=cfg["lr"], num_train_epochs=cfg["epochs"], max_steps=cfg["max_steps"],
        per_device_train_batch_size=cfg["train_bs"], per_device_eval_batch_size=cfg["eval_bs"],
        gradient_accumulation_steps=cfg["grad_accum"], weight_decay=cfg["weight_decay"], seed=seed,
        data_seed=data_seed, remove_unused_columns=False,
        bf16=cuda and torch.cuda.is_bf16_supported(),
        fp16=cuda and not torch.cuda.is_bf16_supported(),
        save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="eval_loss",
        save_total_limit=2, logging_steps=50, disable_tqdm=True, report_to="none",
        dataloader_num_workers=cfg["num_workers"], **{ev_key: "epoch", wu_key: cfg["warmup_ratio"]},
    )
    model = AutoModelForQuestionAnswering.from_pretrained(cfg["model_name"])
    return _qa_trainer_cls()(
        model=model, args=args, train_dataset=drop(train_feats), eval_dataset=drop(dev_loss_feats),
        data_collator=default_data_collator, lr_decay=cfg.get("lr_decay"), ema_decay=cfg.get("ema_decay"),
        plausible_lambda=cfg.get("plausible_lambda") or 0.0)


def predict_logits(trainer, eval_feats):
    """(start, end) logits [n_windows, max_length] for eval features."""
    # offsets/ids stay aside: the model forward cannot take them
    pred = trainer.predict(eval_feats.remove_columns(["example_id", "offset_mapping"]))
    return pred.predictions[:2]


def score_dev(trainer, dev_examples, dev_eval_feats, cfg):
    """Predict dev windows, tune the null threshold on dev refs, report metrics at 0 and at best."""
    start, end = predict_logits(trainer, dev_eval_feats)
    return score_from_logits(dev_examples, dev_eval_feats, start, end, cfg)


def score_from_logits(dev_examples, dev_eval_feats, start, end, cfg, extract=None):
    """CPU-only part of score_dev: n-best + threshold tuning + metrics from (saved) logits."""
    extract = extract or (make_extract(dev_examples) if cfg.get("phobert") else None)
    scores = score_examples(dev_examples, dev_eval_feats, start, end, cfg["n_best"], cfg["max_answer_length"],
                            extract=extract)
    refs = {r["id"]: list(r["ref_texts"] if "ref_texts" in r else r["answers"]["text"]) for r in dev_examples}
    best, table = tune_threshold(scores, refs)
    return {"best_threshold": best, "table": table,
            "metrics_at_0": squad_v2_metrics(apply_threshold(scores, 0.0), refs),
            "metrics_at_best": squad_v2_metrics(apply_threshold(scores, best), refs)}


def save_logits(path, eval_feats, start, end):
    """float16 logits + example_id order + hash of that order (to check alignment when re-scoring)."""
    ids = np.array(eval_feats["example_id"])
    np.savez_compressed(path, start_logits=np.asarray(start).astype(np.float16),
                        end_logits=np.asarray(end).astype(np.float16), example_id=ids,
                        feature_hash=np.array(hashlib.sha1("\n".join(ids).encode()).hexdigest()))


def run(cfg, smoke=False, ds=None, feats=None):
    """Train, score dev, save model + json into cfg['out_dir']; returns the summary dict.

    ds / feats (the build_features tuple) can be passed to skip rebuilding them (notebook).
    """
    import torch
    from transformers import AutoTokenizer, set_seed

    cfg = dict(cfg, **(SMOKE if smoke else {}))
    if cfg.get("run_name"):  # <out_root>/<run_name>, out_root = parent of the configured out_dir
        cfg["out_dir"] = str(Path(cfg["out_dir"]).parent / cfg["run_name"])
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); set_seed(cfg["seed"])
    cuda = torch.cuda.is_available()
    gpu = ({"name": torch.cuda.get_device_name(0),
            "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9}
           if cuda else {"name": "cpu", "total_memory_gb": None})
    print("device:", gpu)

    tok = load_phobert_tokenizer() if cfg.get("phobert") else AutoTokenizer.from_pretrained(cfg["model_name"])
    if feats is None:
        ds = ds or (get_seg_data() if cfg.get("phobert") else get_data())
        feats = build_features(ds, tok, cfg, cfg["max_train"], cfg["max_dev"])
    train_f, dev_loss_f, dev_eval_f, dev_ex = feats
    trainer = make_trainer(cfg, tok, train_f, dev_loss_f)

    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    res = trainer.train()
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if cuda else None
    print(f"train wall {wall:.0f}s, {res.metrics.get('train_samples_per_second')} samples/s, peak GPU {peak} GB")

    d_start, d_end = predict_logits(trainer, dev_eval_f)  # model holds EMA weights here if ema_decay
    dev = score_from_logits(dev_ex, dev_eval_f, d_start, d_end, cfg)
    print("dev @0:", {k: round(v, 2) for k, v in dev["metrics_at_0"].items()})
    print(f"dev @{dev['best_threshold']:.3f}:", {k: round(v, 2) for k, v in dev["metrics_at_best"].items()})

    if cfg.get("save_logits"):
        save_logits(out / "dev_logits.npz", dev_eval_f, d_start, d_end)
    if cfg.get("save_test_logits"):  # test is never used for a decision here, only stored
        ds = ds or (get_seg_data() if cfg.get("phobert") else get_data())
        test = ds["test"].select(range(cfg["max_dev"])) if cfg["max_dev"] else ds["test"]
        kw = dict(tokenizer=tok, max_length=cfg["max_length"], doc_stride=cfg["doc_stride"])
        test_f = test.map(lambda b: prepare_eval_features(b, **kw), batched=True, remove_columns=test.column_names)
        t_start, t_end = predict_logits(trainer, test_f)
        save_logits(out / "test_logits.npz", test_f, t_start, t_end)
    if cfg.get("save_model", True):
        trainer.model.save_pretrained(out / "model")  # fp32 weights (autocast does not change them)
        tok.save_pretrained(out / "model")
    (out / "dev_threshold.json").write_text(json.dumps(
        {"threshold": dev["best_threshold"], "table": dev["table"], "metrics_at_0": dev["metrics_at_0"],
         "metrics_at_best": dev["metrics_at_best"]}, indent=1))
    summary = {
        "cfg": cfg, "gpu": gpu, "train_runtime_s": res.metrics.get("train_runtime"), "wall_s": wall,
        "samples_per_sec": res.metrics.get("train_samples_per_second"), "peak_mem_gb": peak,
        "windows": {"train": len(train_f), "dev": len(dev_eval_f), "dev_examples": len(dev_ex)},
        "log_history": trainer.state.log_history, "dev": dev,
    }
    evals = [h for h in trainer.state.log_history if "eval_loss" in h]
    if evals:
        b = min(evals, key=lambda h: h["eval_loss"])
        summary["best_eval_loss"], summary["best_epoch"] = b["eval_loss"], b.get("epoch")
    text = json.dumps(summary, indent=1, default=str)
    (out / "summary.json").write_text(text)
    (out / "train_summary.json").write_text(text)  # name the notebooks read
    shutil.rmtree(out / "ckpt", ignore_errors=True)  # best model already reloaded and saved above
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--phobert", action="store_true")
    ap.add_argument("--lr-decay", type=float, default=None)
    ap.add_argument("--ema", type=float, default=None)
    ap.add_argument("--plausible-lambda", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--run-name", default=None)
    a = ap.parse_args()
    cfg = dict(CFG_PHOBERT if a.phobert else CFG)
    for k, v in (("lr_decay", a.lr_decay), ("ema_decay", a.ema), ("plausible_lambda", a.plausible_lambda),
                 ("seed", a.seed), ("run_name", a.run_name)):
        if v is not None:
            cfg[k] = v
    if a.out:
        cfg["out_dir"] = a.out
    cfg = {**cfg, **(SMOKE if a.smoke else {}), **({"out_dir": a.out} if a.out else {})}
    run(cfg)
