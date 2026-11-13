"""Run: .venv\\Scripts\\python tests\\test_sweep.py  (config file, smoke sweeps, resumability, failure isolation)"""
import json
import subprocess
import sys

import pytest
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

CONFIGS = ROOT / "configs" / "sweep_stage1.json"


def _sweep(model, configs, out, *extra):
    cmd = [sys.executable, "-m", "src.sweep", "--model", model, "--configs", str(configs), "--out", str(out),
           "--smoke", *extra]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")


def _rows(mdir):
    return [json.loads(x) for x in (mdir / "results.jsonl").read_text().splitlines()]


def test_config_file():
    cfgs = json.loads(CONFIGS.read_text())
    assert set(cfgs) == {"xlmr_base", "phobert", "xlmr_large"}
    for m, entries in cfgs.items():
        names = [e["name"] for e in entries]
        assert len(names) == 4 and len(set(names)) == 4, (m, names)
        assert all(not e.get("save_model") for e in entries)


def test_stage2_config():
    from src.sweep import make_cfg
    cfgs = json.loads((ROOT / "configs" / "sweep_stage2.json").read_text())
    assert set(cfgs) == {"xlmr_base", "phobert", "xlmr_large"}
    for m, entries in cfgs.items():
        names = [e["name"] for e in entries]
        assert len(names) == 4 and len(set(names)) == 4, (m, names)
        for e in entries:
            make_cfg(m, e, "out", False)  # raises KeyError on unknown override keys


def _check_ok_run(mdir, model, name):
    run = mdir / f"{model}__{name}"
    s = json.loads((run / "summary.json").read_text())
    z = np.load(run / "dev_logits.npz")
    assert len(z["example_id"]) == s["windows"]["dev"] and not (run / "model").exists()
    assert s["cfg"]["run_name"] == f"{model}__{name}" and s["cfg"]["seed"] == 42


def _smoke_twice(model, names, data_path):
    if not data_path.exists():
        pytest.skip(f"{data_path} missing")
    with tempfile.TemporaryDirectory() as tmp:
        r = _sweep(model, CONFIGS, tmp, "--only", ",".join(names))
        assert r.returncode == 0, r.stderr[-2000:]
        mdir = Path(tmp) / model
        rows = _rows(mdir)
        assert [x["status"] for x in rows] == ["ok"] * len(names), rows
        for n in names:
            _check_ok_run(mdir, model, n)
        assert json.loads((mdir / "status.json").read_text())["total"] == len(names)
        r2 = _sweep(model, CONFIGS, tmp, "--only", ",".join(names))
        assert r2.returncode == 0 and r2.stdout.count("skipped") == len(names), r2.stdout[-1000:]
        assert len(_rows(mdir)) == len(names), "resume re-ran a finished run"
        print("  results line:", json.dumps(rows[0])[:300])


def test_smoke_xlmr_resume():
    first_two = [e["name"] for e in json.loads(CONFIGS.read_text())["xlmr_base"][:2]]
    _smoke_twice("xlmr_base", first_two, ROOT / "data" / "processed" / "uit_viquad2_split")


def test_smoke_phobert():
    name = json.loads(CONFIGS.read_text())["phobert"][0]["name"]
    _smoke_twice("phobert", [name], ROOT / "data" / "processed" / "seg" / "train.jsonl")


def test_failure_does_not_stop_sweep():
    if not (ROOT / "data" / "processed" / "uit_viquad2_split").exists():
        pytest.skip("data missing")
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "c.json"
        cfg.write_text(json.dumps({"xlmr_base": [
            {"name": "bad", "overrides": {"no_such_key": 1}},
            {"name": "good", "overrides": {"lr": 3e-5, "epochs": 3}}]}))
        r = _sweep("xlmr_base", cfg, Path(tmp) / "out")
        assert r.returncode == 0, r.stderr[-2000:]
        rows = _rows(Path(tmp) / "out" / "xlmr_base")
        assert [x["status"] for x in rows] == ["failed", "ok"], rows
        assert "no_such_key" in rows[0]["error"]
