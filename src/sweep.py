"""Resumable fine-tuning sweep for one model in one process (data + features built once).

CLI: python -m src.sweep --model {xlmr_base,phobert,xlmr_large} --configs configs/sweep_stage1.json
         --out outputs/sweep/stage1 [--only NAME[,NAME]] [--smoke] [--detach]
     python -m src.sweep --tail <out>/<model>
Layout: <out>/<model>/{results.jsonl,status.json,sweep.log} and <out>/<model>/<run_name>/ per run.
"""
import argparse
import gc
import json
import re
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from src.train_qa import CFG, CFG_PHOBERT, ROOT, SMOKE, build_features, get_data, run
from src.phobert import get_seg_data, load_phobert_tokenizer

A100_UNITS_PER_HOUR = 6.73
MODELS = ("xlmr_base", "phobert", "xlmr_large")
# OOM is recoverable (next run goes on); these mean the CUDA context may be poisoned
CUDA_FATAL = re.compile(r"CUDA error|CUBLAS_STATUS|device-side assert|cudaError|CUDNN_STATUS", re.I)


def base_cfg(model):
    if model == "xlmr_base":
        return dict(CFG)
    if model == "xlmr_large":
        return dict(CFG, model_name="xlm-roberta-large", train_bs=16, grad_accum=2)
    return dict(CFG_PHOBERT)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def make_cfg(model, entry, out_dir, smoke):
    """Base cfg + entry overrides + per-run fixed keys (+ smoke settings, minus their out_dir)."""
    cfg = base_cfg(model)
    bad = [k for k in entry.get("overrides", {}) if k not in cfg]
    if bad:
        raise KeyError(f"unknown override keys {bad}")
    cfg.update(entry.get("overrides", {}))
    name = f"{model}__{entry['name']}"
    cfg.update(run_name=name, seed=cfg["seed"], save_model=bool(entry.get("save_model", False)),
               save_logits=True, save_test_logits=bool(entry.get("save_test_logits", False)),
               out_dir=str(Path(out_dir) / model / name))
    if smoke:
        cfg.update({k: v for k, v in SMOKE.items() if k != "out_dir"})
    return cfg


def result_row(model, entry, cfg, summary, wall):
    d = summary["dev"]
    m0, mb = d["metrics_at_0"], d["metrics_at_best"]
    return {
        "run_name": cfg["run_name"], "model": model, "overrides": entry.get("overrides", {}), "seed": cfg["seed"],
        "status": "ok", "best_eval_loss": summary.get("best_eval_loss"), "best_epoch": summary.get("best_epoch"),
        "f1_at_0": m0["f1"], "em_at_0": m0["exact"], "hasans_f1_at_0": m0["HasAns_f1"],
        "noans_acc_at_0": m0["NoAns_exact"],
        "f1_tuned": mb["f1"], "em_tuned": mb["exact"], "hasans_f1_tuned": mb["HasAns_f1"],
        "noans_acc_tuned": mb["NoAns_exact"], "threshold": d["best_threshold"],
        "train_runtime_s": summary.get("train_runtime_s"), "wall_s": wall,
        "peak_mem_gb": summary.get("peak_mem_gb"), "est_units": wall / 3600 * A100_UNITS_PER_HOUR,
        "finished_at": _now(),
    }


def sweep(model, configs_path, out, only=None, smoke=False):
    """Run every (selected) config of `model`; returns 0, or 2 if stopped by a CUDA error."""
    entries = json.loads(Path(configs_path).read_text())[model]
    if only:
        entries = [e for e in entries if e["name"] in only]
        missing = set(only) - {e["name"] for e in entries}
        assert not missing, f"--only names not in config: {missing}"
    mdir = Path(out) / model
    mdir.mkdir(parents=True, exist_ok=True)
    results, t_start = mdir / "results.jsonl", time.time()
    state = {"current": None, "index": 0, "total": len(entries), "run_started": None}

    def write_status():
        now = time.time()
        _write_json(mdir / "status.json", {
            "model": model, "current_run": state["current"], "index": state["index"], "total": state["total"],
            "elapsed_s": round(now - t_start), "run_elapsed_s": round(now - state["run_started"]) if state["run_started"] else None,
            "updated_at": _now()})

    stop = threading.Event()

    def beat():
        while not stop.wait(30):
            write_status()

    threading.Thread(target=beat, daemon=True).start()
    write_status()

    # features once per model; plausible columns only if some config in the file asks for them
    import torch
    plausible = any((e.get("overrides") or {}).get("plausible_lambda", 0) > 0 for e in entries)
    bcfg = dict(base_cfg(model), plausible_lambda=1.0 if plausible else 0.0,
                **({k: v for k, v in SMOKE.items() if k != "out_dir"} if smoke else {}))
    ds = get_seg_data(ROOT) if model == "phobert" else get_data(ROOT)
    if model == "phobert":
        tok = load_phobert_tokenizer()
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(bcfg["model_name"])
    feats = build_features(ds, tok, bcfg, bcfg["max_train"], bcfg["max_dev"])
    del tok

    code = 0
    for i, entry in enumerate(entries, 1):
        name = f"{model}__{entry['name']}"
        state.update(current=name, index=i, run_started=None)
        if (mdir / name / "summary.json").exists():
            print(f"[{i}/{len(entries)}] {name}: summary.json exists, skipped", flush=True)
            continue
        state["run_started"] = time.time()
        write_status()
        t0, row, fatal = time.time(), None, False
        try:
            cfg = make_cfg(model, entry, out, smoke)
            summary = run(cfg, ds=ds, feats=feats)
            row = result_row(model, entry, cfg, summary, time.time() - t0)
            print(f"[{i}/{len(entries)}] {name}: ok F1@0 {row['f1_at_0']:.2f} F1@tuned {row['f1_tuned']:.2f} "
                  f"(thr {row['threshold']:.3f}) best_eval_loss {row['best_eval_loss']} "
                  f"wall {row['wall_s']:.0f}s est_units {row['est_units']:.2f}", flush=True)
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            wall = time.time() - t0
            fatal = bool(CUDA_FATAL.search(tb))
            row = {"run_name": name, "model": model, "overrides": entry.get("overrides", {}),
                   "seed": (entry.get("overrides") or {}).get("seed", 42), "status": "failed",
                   "error": tb[-1500:], "wall_s": wall, "est_units": wall / 3600 * A100_UNITS_PER_HOUR,
                   "finished_at": _now()}
            print(f"[{i}/{len(entries)}] {name}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
        with results.open("a") as f:
            f.write(json.dumps(row) + "\n")
        gc.collect()  # exception frames hold the trainer; they are gone after the except block
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if fatal:
            print("CUDA error: stopping the whole sweep (context may be poisoned); restart the process to resume.",
                  flush=True)
            code = 2
            break
    state.update(current=None, run_started=None)
    stop.set()
    write_status()
    return code


def _cmd(model, configs, out, only=None, smoke=False):
    cmd = [sys.executable, "-u", "-m", "src.sweep", "--model", model, "--configs", str(configs), "--out", str(out)]
    return cmd + (["--only", only] if only else []) + (["--smoke"] if smoke else [])


def launch(model, configs, out, only=None, smoke=False):
    """Start the sweep as a detached background process, log to <out>/<model>/sweep.log; returns the Popen."""
    mdir = Path(out) / model
    mdir.mkdir(parents=True, exist_ok=True)
    log = open(mdir / "sweep.log", "ab")
    return subprocess.Popen(_cmd(model, configs, out, only, smoke), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)


def tail(mdir):
    mdir = Path(mdir)
    for name, n in (("status.json", None), ("results.jsonl", 5), ("sweep.log", 15)):
        p = mdir / name
        print(f"--- {name}")
        if not p.exists():
            print("(missing)")
            continue
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        print("\n".join(lines if n is None else lines[-n:]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--configs", default="configs/sweep_stage1.json")
    ap.add_argument("--out", default="outputs/sweep/stage1")
    ap.add_argument("--only", default=None, help="comma-separated run names")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--detach", action="store_true", help="run in the background, log to <out>/<model>/sweep.log")
    ap.add_argument("--tail", default=None, metavar="OUT/MODEL")
    a = ap.parse_args(argv)
    if a.tail:
        return tail(a.tail)
    assert a.model, "--model required"
    if a.detach:
        p = launch(a.model, a.configs, a.out, a.only, a.smoke)
        print(f"started pid {p.pid}; poll with: python -m src.sweep --tail {Path(a.out) / a.model}")
        return 0
    return sweep(a.model, a.configs, a.out, a.only.split(",") if a.only else None, a.smoke)


if __name__ == "__main__":
    sys.exit(main())
