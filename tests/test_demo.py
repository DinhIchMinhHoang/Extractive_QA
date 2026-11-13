"""Pure-function tests for src/demo.py (no model loading)."""
import sys
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import demo  # noqa: E402


def test_parse_command():
    assert demo.parse_command("/use xlmr_base,phobert") == ("use", "xlmr_base,phobert")
    assert demo.parse_command("/HELP") == ("help", "")
    assert demo.parse_command("  Thủ đô là gì?  ") == ("ask", "Thủ đô là gì?")


def test_context_file_and_use(tmp_path):
    f = tmp_path / "c.txt"
    f.write_text("Hà Nội là thủ đô.\n", encoding="utf-8")
    assert demo.read_context_arg(f"@{f}") == "Hà Nội là thủ đô."
    assert demo.read_context_arg("") is None
    assert demo.parse_use("all", ["xlmr_base"]) == ["xlmr_base"]
    assert demo.parse_use("xlmr_base, phobert", []) == ["xlmr_base", "phobert"]
    with pytest.raises(ValueError):
        demo.parse_use("nope", [])


def test_decide():
    assert demo.decide("Hà Nội", -2.0, -1.0) == "Hà Nội"
    assert demo.decide("Hà Nội", -1.0, -1.0) == "Hà Nội"  # null_diff == thr vẫn có đáp án
    assert demo.decide("Hà Nội", 0.5, -1.0) == ""


def test_format_table_alignment():
    res = [dict(name="xlmr_large", group="A", text="Hà Nội", null_diff=-3.1, time=0.5),
           dict(name="vi_mrc_base", group="B", text="Hà Nội", null_diff=-2.0, time=1.25),
           dict(name="phobert", group="A", text="", null_diff=4.0, time=0.3),
           dict(name="xlmr_base", group="A", error="boom")]
    t = demo.format_table(res)
    assert len(t) == 6
    # cột đáp án thẳng hàng theo độ rộng hiển thị (cả NFC lẫn NFD)
    assert len({demo._w(x.split(" | ")[0]) for x in t if " | " in x}) == 1
    assert "(không có đáp án)" in t[4] and "vi_mrc_base*" in t[3] and "lỗi" in t[5]
    assert demo.agreeing(res) == {"xlmr_large", "vi_mrc_base"}
    assert "\x1b[1;32m" in demo.format_table(res, color=True)[2]
    assert demo._w(unicodedata.normalize("NFD", "Hà Nội")) == demo._w("Hà Nội") == 6


def _release(tmp_path, with_files=True):
    import json
    rel = tmp_path / "outputs" / "release"
    rel.mkdir(parents=True)
    (rel / "best_models.json").write_text(json.dumps({
        "xlmr_large": {"run": "r2", "dir": "outputs/release/xlmr_large/r2", "threshold": -1.5, "dev_f1": 77.7},
        "ensembles": {"xlmr_large": ["r1", "r2"], "ensemble_thresholds": {"xlmr_large": -0.25}}}))
    for r in ("r1", "r2") if with_files else ():
        (rel / "xlmr_large" / r).mkdir(parents=True)
        (rel / "xlmr_large" / r / "model.safetensors").write_text("")
    return tmp_path


def test_read_release_and_registry(tmp_path):
    assert demo.read_release(tmp_path) is None
    root = _release(tmp_path)
    assert demo.read_release(root)["xlmr_large"]["threshold"] == -1.5
    reg, notes = demo.build_registry(ensemble=False, root=root)
    sp = reg["xlmr_large"]
    assert sp["source"] == "release single" and sp["threshold"] == -1.5 and sp["dev_f1"] == 77.7 and not notes
    assert sp["dirs"] == [root / "outputs/release/xlmr_large/r2"]
    ens = demo.build_registry(root=root)[0]["xlmr_large"]  # default = ensemble
    assert ens["label"] == "xlmr_large (ens×2)" and ens["threshold"] == -0.25 and len(ens["dirs"]) == 2
    assert demo.build_registry(legacy=True, root=root)[0]["xlmr_large"]["source"] == "legacy notebook"
    assert demo.build_registry(root=root)[0]["vi_mrc_base"]["source"] == "legacy notebook"


def test_registry_fallback_legacy(tmp_path):
    reg, notes = demo.build_registry(root=tmp_path)  # no best_models.json
    assert reg["xlmr_large"]["source"] == "legacy notebook" and "dirs" not in reg["xlmr_large"] and notes
    reg, notes = demo.build_registry(root=_release(tmp_path, with_files=False))  # json but no weights
    assert reg["xlmr_large"]["source"] == "legacy notebook" and any("xlmr_large" in n for n in notes)


def test_mean_logits():
    import numpy as np
    s, e = demo.mean_logits([(np.array([[0., 2.]]), np.array([[4., 4.]])), (np.array([[2., 4.]]), np.array([[0., 2.]]))])
    assert s.tolist() == [[1., 3.]] and e.tolist() == [[2., 3.]]


def test_parse_toggle_commands():
    assert demo.parse_command("/ensemble on") == ("ensemble", "on")
    assert demo.parse_command("/single") == ("single", "")
    assert demo.parse_command("/Legacy off") == ("legacy", "off")
    assert demo.parse_toggle("on") is True and demo.parse_toggle(" OFF ") is False
    with pytest.raises(ValueError):
        demo.parse_toggle("maybe")


def test_availability_tmp(tmp_path):
    (tmp_path / "x" / "model").mkdir(parents=True)
    reg = {"a": dict(group="A", kind="hf", dir=tmp_path / "x", tar=tmp_path / "no.tar"),
           "b": dict(group="A", kind="hf", dir=tmp_path / "y", tar=tmp_path / "no.tar"),
           "p": dict(group="A", kind="phobert", dir=tmp_path / "x", tar=tmp_path / "no.tar"),
           "z": dict(group="B", kind="zs", repo="r/z", dir=tmp_path / "z")}
    (tmp_path / "z").mkdir()
    (tmp_path / "z" / "dev_threshold.json").write_text('{"threshold": -1}')
    av = demo.availability(reg, vncorenlp_dir=tmp_path / "vn", java=True, hf_cached=lambda r: True)
    assert av["a"][0] and not av["b"][0] and "thiếu" in av["b"][1]
    assert not av["p"][0] and "VnCoreNLP" in av["p"][1]
    assert av["z"][0]
    (tmp_path / "vn").mkdir()
    (tmp_path / "vn" / "VnCoreNLP-1.2.jar").write_text("")
    assert demo.availability(reg, tmp_path / "vn", java=True, hf_cached=lambda r: True)["p"][0]
    assert "Java" in demo.availability(reg, tmp_path / "vn", java=False, hf_cached=lambda r: False)["p"][1]
    assert "HF cache" in demo.availability(reg, tmp_path / "vn", java=True, hf_cached=lambda r: False)["z"][1]
    assert demo.read_threshold(tmp_path / "z") == -1
