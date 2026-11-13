"""Reusable QA entry point: answer(question, context) -> {text, score, null_diff, has_answer[, n_models]}."""
import gc
import json
import logging
import os
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

from src.demo import mean_logits  # shared with the demo ensemble path
from src.postprocess import apply_threshold, score_examples
from src.preprocess import prepare_eval_features
from src.zero_shot import load_model, predict_logits_hf

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "outputs" / "xlmr_large"  # extracted from outputs/xlmr_large_out.tar: model/ + dev_threshold.json
log = logging.getLogger(__name__)


def total_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().total / 2**30
    except ImportError:  # Windows fallback; elsewhere sysconf
        if os.name != "nt":
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in
                                                                              ("total", "a", "tp", "ap", "tv", "av", "ae")]
        m = MS()
        m.l = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.total / 2**30


def default_budget_gb():
    return min(6.0, 0.4 * total_ram_gb())


class ModelCache:
    """LRU cache of member models kept in CPU RAM (fp16 when the device is cuda) between calls.
    use(): one model on the device at a time, moved back to CPU afterwards. QA_CACHE=0 disables, QA_CACHE_GB = budget."""

    def __init__(self, budget_gb=None, enabled=True):
        self.enabled = enabled
        self.budget = int((default_budget_gb() if budget_gb is None else budget_gb) * 2**30)
        self.items = OrderedDict()  # key -> (obj, model, nbytes); last = most recently used
        self.last_load = 0.0  # seconds spent loading/moving in the latest use()

    @classmethod
    def from_env(cls, env=os.environ):
        gb = env.get("QA_CACHE_GB")
        return cls(float(gb) if gb else None, env.get("QA_CACHE", "1").strip() != "0")

    @staticmethod
    def nbytes(model):
        return sum(t.numel() * t.element_size() for t in [*model.parameters(), *model.buffers()])

    def used(self):
        return sum(v[2] for v in self.items.values())

    def _store(self, key, obj, model):
        n = self.nbytes(model)
        if n > self.budget:
            log.debug("cache skip %s (%d MB > budget)", key, n >> 20)
            return False
        while self.items and self.used() + n > self.budget:
            old, _ = self.items.popitem(last=False)
            log.debug("cache evict %s", old)
        self.items[key] = (obj, model, n)
        return True

    @contextmanager
    def use(self, key, loader, device, free=None):
        """loader() -> (obj, model) with the model already on `device`. Yields (obj, model). `free(obj)` is called
        when the model is not kept (disabled / over budget); callers must also drop their own references."""
        t0 = time.time()
        if self.enabled and key in self.items:
            self.items.move_to_end(key)
            obj, model, _ = self.items[key]
            if device != "cpu":
                model.to(device)
            kept = True
            log.debug("cache hit %s", key)
        else:
            obj, model = loader()
            kept = self.enabled and self._store(key, obj, model)
            log.debug("cache miss %s (kept=%s, used %d MB)", key, kept, self.used() >> 20)
        self.last_load = time.time() - t0
        try:
            yield obj, model
        finally:
            if kept and device != "cpu":
                model.to("cpu")
            elif not kept and free:
                free(obj)
            gc.collect()
            if device == "cuda":
                import torch
                torch.cuda.empty_cache()


_CACHE = None


def get_cache():
    global _CACHE
    _CACHE = _CACHE or ModelCache.from_env()
    return _CACHE


class QA:
    """Lazy: weights load on first answer(). Pass tok+model to inject (tests).

    members (release ensemble): list of run dirs. answer() loads them ONE AT A TIME (fp16 on cuda, fp32 on cpu),
    averages start/end logits over the same windows, scores once with `threshold` (the ensemble threshold).
    Members are cached in CPU RAM (ModelCache, fp16, LRU; QA_CACHE=0 disables) and moved to the GPU one at a time.
    Use load_qa(ensemble=False) for the fast single model."""

    def __init__(self, model_dir=None, threshold=None, device=None, tok=None, model=None, members=None):
        self.dir = Path(model_dir) if model_dir else DEFAULT_DIR
        self.release = False  # release run dir: weights directly in dir (no model/ subdir)
        self.members = [Path(m) for m in members] if members else None
        self.threshold, self.device, self.tok, self.model = threshold, device, tok, model

    def _device(self):
        import torch

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        return self.device

    def _load(self):
        if self.model is not None:
            return
        self._device()
        self.tok, self.model = load_model(str(self.dir if self.release else self.dir / "model"), device=self.device)
        if self.threshold is None:  # threshold tuned on dev for this run
            self.threshold = json.loads((self.dir / "dev_threshold.json").read_text())["threshold"]

    def _logits_ensemble(self, ex):
        """(feats, mean start, mean end); one member on the device at a time."""
        parts, feats = [], None
        for d in self.members:
            with get_cache().use((str(d), self._device()), lambda d=d: load_model(str(d), device=self.device),
                                 self.device) as (tok, model):
                if feats is None:  # same tokenizer family -> same windows for every member
                    feats = prepare_eval_features(ex, tok)
                parts.append(predict_logits_hf(model, feats, bs=8, device=self.device))
            del tok, model  # an uncached model must not stay on the GPU while the next one loads
        return (feats, *mean_logits(parts))

    def answer(self, question, context):
        ex = {"id": ["q"], "question": [question], "context": [context]}
        if self.members:
            feats, s, e = self._logits_ensemble(ex)
        else:
            self._load()
            feats = prepare_eval_features(ex, self.tok)  # sliding windows 384/128
            s, e = predict_logits_hf(self.model, feats, bs=8, device=self.device or "cpu")
        sc = score_examples(ex, feats, s, e)["q"]  # null = min over windows of start[0]+end[0]
        thr = self.threshold if self.threshold is not None else 0.0
        out = {"text": apply_threshold({"q": sc}, thr)["q"], "score": sc["span_score"],
               "null_diff": sc["diff"], "has_answer": sc["diff"] <= thr}
        if self.members:
            out["n_models"] = len(self.members)
        return out


def release_plan(model="xlmr_large", ensemble=True, root=ROOT):
    """Pure: (dirs, threshold) from best_models.json: ensemble members + ensemble threshold, else the best single.
    Raises OSError/ValueError/KeyError when release files are missing (caller falls back to legacy)."""
    root = Path(root)
    rel = json.loads((root / "outputs" / "release" / "best_models.json").read_text(encoding="utf-8"))
    if ensemble:
        runs = rel["ensembles"][model]
        dirs = [root / "outputs" / "release" / model / r for r in runs]
        if runs and all((d / "model.safetensors").exists() for d in dirs):
            return dirs, rel["ensembles"]["ensemble_thresholds"][model]
    b = rel[model]  # ensemble=False, or member weights missing -> best single
    return [root / b["dir"]], b["threshold"]


def load_qa(model_dir=None, threshold=None, device=None, legacy=False, *, model="xlmr_large", ensemble=True):
    """Default: release ENSEMBLE of `model` (members + ensemble threshold from best_models.json), ~3x latency.
    ensemble=False: release best single; legacy=True (or missing release files): notebook model, ignores ensemble."""
    if model_dir is None and not legacy:
        try:
            dirs, thr = release_plan(model, ensemble)
            thr = thr if threshold is None else threshold
            if len(dirs) > 1:
                return QA(dirs[0], thr, device, members=dirs)
            qa = QA(dirs[0], thr, device)
            qa.release = True
            return qa
        except (OSError, ValueError, KeyError):
            pass  # no release files -> legacy
    return QA(model_dir, threshold, device)


_DEFAULT = None


def answer(question, context, qa=None):
    global _DEFAULT
    if qa is None:
        _DEFAULT = _DEFAULT or load_qa()
        qa = _DEFAULT
    return qa.answer(question, context)
