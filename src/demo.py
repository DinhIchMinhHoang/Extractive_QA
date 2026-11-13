"""Demo terminal: so sánh 5 mô hình QA trên cùng một câu hỏi + ngữ cảnh.

python -m src.demo                      # tương tác (gõ /help)
python -m src.demo --list
python -m src.demo --models xlmr_large,phobert --context-file ctx.txt --question "..."   # mặc định: ensemble release
python -m src.demo --single ...         # mô hình đơn tốt nhất (nhanh hơn); --legacy: mô hình notebook cũ
python -m src.demo --models xlmr_large,vi_mrc_large --context-file ctx.txt --question "..."
"""
import argparse
import json
import os
import shutil
import sys
import tarfile
import time
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
ZS = OUT / "sweep" / "zero_shot"
NO_ANS = "(không có đáp án)"

# name -> source. dir: nơi chứa model/ + dev_threshold.json; tar: giải nén lần đầu vào dir.
REGISTRY = {
    "xlmr_base": dict(group="A", kind="hf", dir=OUT / "xlmr_base", tar=OUT / "xlmr_base_out.tar"),
    "xlmr_large": dict(group="A", kind="hf", dir=OUT / "xlmr_large", tar=OUT / "xlmr_large_out.tar"),
    "phobert": dict(group="A", kind="phobert", dir=OUT / "phobert_base", tar=OUT / "phobert_base_lr2e5_out.tar"),
    "vi_mrc_base": dict(group="B", kind="zs", repo="nguyenvulebinh/vi-mrc-base",
                        dir=ZS / "vi_mrc_base" / "vi_mrc_base__zs"),
    "vi_mrc_large": dict(group="B", kind="zs", repo="nguyenvulebinh/vi-mrc-large",
                         dir=ZS / "vi_mrc_large" / "vi_mrc_large__zs"),
}
OWN = ("xlmr_base", "xlmr_large", "phobert")
GROUP_LABEL = {"A": "nhóm A (fine-tune của nhóm)", "B": "tham chiếu nhóm B (pretrained thêm dữ liệu ngoài)"}

SAMPLE_CONTEXT = (
    "Hà Nội là thủ đô của nước Cộng hòa Xã hội Chủ nghĩa Việt Nam và là thành phố đứng đầu cả nước về diện tích "
    "tự nhiên cũng như một trong hai đô thị lớn nhất về dân số. Năm 1010, vua Lý Thái Tổ dời đô từ Hoa Lư ra "
    "thành Đại La và đổi tên thành Thăng Long. Thành phố nằm bên bờ sông Hồng, có Hồ Gươm là danh thắng nổi tiếng "
    "ở trung tâm. Văn Miếu - Quốc Tử Giám, xây dựng năm 1070, được xem là trường đại học đầu tiên của Việt Nam.")
SAMPLE_QUESTIONS = ["Thủ đô của Việt Nam là gì?", "Vua Lý Thái Tổ dời đô ra Thăng Long vào năm nào?",
                    "Ai là tổng thống đầu tiên của Hoa Kỳ?"]  # câu cuối không có đáp án

HELP = """Lệnh:
  /models                  liệt kê mô hình + trạng thái
  /use <tên,...|all>       chọn mô hình (mặc định: tất cả mô hình khả dụng)
  /context                 nhập ngữ cảnh nhiều dòng, kết thúc bằng một dòng trống
  /context @file.txt       đọc ngữ cảnh từ file utf-8
  /sample                  nạp ngữ cảnh + 3 câu hỏi mẫu (câu cuối không có đáp án)
  /threshold <mô hình> <giá trị>   ghi đè ngưỡng null_diff
  /legacy on|off           mô hình notebook cũ thay cho bản release
  /ensemble on|off         ensemble các seed release (mặc định: on; chậm ~N lần)
  /single                  mô hình đơn tốt nhất của release (tương đương /ensemble off)
  /help, /quit
Dòng khác = câu hỏi."""

# ---------- màu ----------
_COLOR = False


def init_console():
    """utf-8 cho stdout/stdin; bật ANSI nếu là TTY và không có NO_COLOR."""
    global _COLOR
    for s in (sys.stdout, sys.stdin):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    _COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    if _COLOR and os.name == "nt":
        os.system("")  # bật xử lý ANSI (VT) trên console Windows


def paint(s, code):
    return f"\x1b[{code}m{s}\x1b[0m" if _COLOR else s


# ---------- hàm thuần ----------
def parse_command(line):
    """'/use a,b' -> ('use', 'a,b'); dòng không bắt đầu bằng '/' -> ('ask', line)."""
    line = line.strip()
    if not line.startswith("/"):
        return "ask", line
    cmd, _, arg = line[1:].partition(" ")
    return cmd.lower(), arg.strip()


def parse_toggle(arg):
    """'on'/'off' -> True/False cho /legacy, /ensemble."""
    if arg.strip().lower() not in ("on", "off"):
        raise ValueError("dùng: on hoặc off")
    return arg.strip().lower() == "on"


def read_release(root=ROOT):
    """outputs/release/best_models.json -> dict, hoặc None nếu thiếu/hỏng."""
    try:
        return json.loads((Path(root) / "outputs" / "release" / "best_models.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def build_registry(legacy=False, ensemble=True, root=ROOT):
    """(registry, ghi chú). 3 mô hình của nhóm: ensemble/single release; thiếu file -> legacy kèm ghi chú."""
    reg, notes, rel = {}, [], None if legacy else read_release(root)
    if not legacy and rel is None:
        notes.append("thiếu/hỏng outputs/release/best_models.json -> dùng mô hình legacy (notebook)")
    for n, sp in REGISTRY.items():
        sp = dict(sp, source="legacy notebook")
        if n in OWN and rel and n in rel:
            b, root_p = rel[n], Path(root)
            single = dict(sp, source="release single", dirs=[root_p / b["dir"]], threshold=b["threshold"],
                          dev_f1=b["dev_f1"], label=n)
            runs = (rel.get("ensembles") or {}).get(n) or []
            thr = ((rel.get("ensembles") or {}).get("ensemble_thresholds") or {}).get(n)
            ens_dirs = [root_p / "outputs" / "release" / n / r for r in runs]
            if ensemble and runs and thr is not None and all((d / "model.safetensors").exists() for d in ens_dirs):
                sp = dict(single, source=f"release ensemble ×{len(runs)}", dirs=ens_dirs, threshold=thr,
                          dev_f1=None, label=f"{n} (ens×{len(runs)})")
            elif (single["dirs"][0] / "model.safetensors").exists():
                sp = single
                if ensemble:
                    notes.append(f"{n}: thiếu file ensemble -> dùng release single")
            else:
                notes.append(f"{n}: thiếu {single['dirs'][0]} -> dùng legacy (notebook)")
        reg[n] = sp
    return reg, notes


def mean_logits(parts):
    """parts: list of (start, end) numpy [n_windows, L] cùng cửa sổ -> trung bình theo thành viên."""
    import numpy as np
    return np.mean([p[0] for p in parts], axis=0), np.mean([p[1] for p in parts], axis=0)


def read_context_arg(arg):
    """'@path' -> nội dung file (utf-8); ngược lại None (nghĩa là nhập tay)."""
    if arg.startswith("@"):
        return Path(arg[1:].strip().strip('"')).read_text(encoding="utf-8").strip()
    return None


def parse_use(arg, available):
    names = list(available) if arg.strip() == "all" else [n.strip() for n in arg.split(",") if n.strip()]
    bad = [n for n in names if n not in REGISTRY]
    if bad or not names:
        raise ValueError(f"mô hình không hợp lệ: {bad or arg!r}; có: {', '.join(REGISTRY)}")
    return names


def decide(text, null_diff, threshold):
    """Đáp án rỗng nếu null_diff > threshold (cùng quy tắc với postprocess.apply_threshold)."""
    return "" if null_diff > threshold else text


def _w(s):  # độ rộng hiển thị: bỏ dấu kết hợp
    return sum(1 for c in unicodedata.normalize("NFC", s) if not unicodedata.combining(c))


def _pad(s, n):
    return s + " " * (n - _w(s))


def _norm(s):
    return " ".join(unicodedata.normalize("NFC", s).lower().split())


def agreeing(results):
    """Tên các mô hình có cùng đáp án với nhóm đồng thuận lớn nhất (>=2 mô hình); rỗng nếu không có."""
    cnt = Counter(_norm(r["text"]) for r in results if "error" not in r)
    if not cnt:
        return set()
    key, n = cnt.most_common(1)[0]
    return {r["name"] for r in results if "error" not in r and _norm(r["text"]) == key} if n >= 2 else set()


def format_table(results, color=False):
    """results: list of dict(name, group, text, null_diff, time) hoặc (name, error). Trả về list dòng."""
    head = ["mô hình", "đáp án", "null_diff", "thời gian (s)"]
    agree, rows = agreeing(results), []
    for r in results:
        name = r["name"] + ("*" if r["group"] == "B" else "")
        if "error" in r:
            rows.append([name, f"(lỗi: {r['error']})", "-", "-"])
        else:
            rows.append([name, r["text"] or NO_ANS, f"{r['null_diff']:.2f}", f"{r['time']:.2f}"])
    widths = [max(_w(x[i]) for x in [head] + rows) for i in range(4)]
    line = lambda cells: " | ".join(_pad(c, w) for c, w in zip(cells, widths))  # noqa: E731
    out = [line(head), "-+-".join("-" * w for w in widths)]
    for r, cells in zip(results, rows):
        s = line(cells)
        out.append(f"\x1b[1;32m{s}\x1b[0m" if color and r["name"] in agree else s)
    return out


# ---------- khả dụng / giải nén ----------
def java_ok():
    from src.segment import _ensure_java_home
    _ensure_java_home()
    return bool(os.environ.get("JAVA_HOME") or shutil.which("java"))


def _hf_cached(repo):
    from huggingface_hub import try_to_load_from_cache
    return isinstance(try_to_load_from_cache(repo, "config.json"), str)


def availability(registry=None, vncorenlp_dir=None, java=None, hf_cached=None):
    """{name: (ok, lý do)}. Không tải mô hình. Các tham số cho phép tiêm giá trị khi test."""
    registry = REGISTRY if registry is None else registry
    out = {}
    for name, sp in registry.items():
        d = Path(sp["dir"])
        miss = [x for x in sp.get("dirs", []) if not (x / "model.safetensors").exists()]
        if miss:
            out[name] = (False, f"thiếu {miss[0] / 'model.safetensors'}")
            continue
        if sp["kind"] == "zs":
            if not (d / "dev_threshold.json").exists():
                out[name] = (False, f"thiếu {d / 'dev_threshold.json'}")
            elif not (hf_cached or _hf_cached)(sp["repo"]):
                out[name] = (False, f"{sp['repo']} chưa có trong HF cache")
            else:
                out[name] = (True, "")
            continue
        if "dirs" not in sp and not (d / "model").is_dir() and not Path(sp["tar"]).exists():  # release: đã kiểm tra ở trên
            out[name] = (False, f"thiếu {d / 'model'} và {sp['tar']}")
        elif sp["kind"] == "phobert":
            vn = Path(vncorenlp_dir) if vncorenlp_dir else None
            if vn is None:
                from src.segment import VNCORENLP_DIR
                vn = VNCORENLP_DIR
            if not (vn / "VnCoreNLP-1.2.jar").exists():
                out[name] = (False, f"thiếu VnCoreNLP ({vn})")
            elif not (java if java is not None else java_ok()):
                out[name] = (False, "không tìm thấy Java (cần cho VnCoreNLP)")
            else:
                out[name] = (True, "")
        else:
            out[name] = (True, "")
    return out


def ensure_extracted(sp):
    """Giải nén tar vào sp['dir'] nếu chưa có model/. Từ chối đường dẫn thoát khỏi thư mục đích."""
    d = Path(sp["dir"])
    if (d / "model").is_dir() or "tar" not in sp:
        return d
    d.mkdir(parents=True, exist_ok=True)
    with tarfile.open(sp["tar"]) as t:
        for m in t.getmembers():
            if os.path.isabs(m.name) or ".." in Path(m.name).parts or not (m.isfile() or m.isdir()):
                raise RuntimeError(f"tar không an toàn: {m.name}")
        t.extractall(d, filter="data")
    return d


def read_threshold(d):
    return json.loads((Path(d) / "dev_threshold.json").read_text())["threshold"]


# ---------- suy luận ----------
class Loaded:
    """Một mô hình đang nạp. Mỗi lần chỉ giữ một mô hình trên GPU (fp16); free() trả bộ nhớ."""

    def __init__(self, name, sp=None, run_dir=None):
        import torch
        from src.zero_shot import load_model
        self.name, sp = name, sp or REGISTRY[name]
        self.phobert = sp["kind"] == "phobert"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if run_dir is not None or "dirs" in sp:  # release: config.json + model.safetensors ngay trong thư mục run
            d = md = Path(run_dir or sp["dirs"][0])
            self.threshold = sp["threshold"]
        else:
            d = ensure_extracted(sp)
            md = d / "model"
            self.threshold = read_threshold(d)
        if self.phobert:
            from src.phobert import load_phobert_tokenizer
            from src.segment import Segmenter
            from transformers import AutoModelForQuestionAnswering
            self.tok, self.seg = load_phobert_tokenizer(), Segmenter()
            self.model = AutoModelForQuestionAnswering.from_pretrained(str(md))
            self.model = (self.model.half() if self.device == "cuda" else self.model).to(self.device).eval()
        else:
            self.tok, self.model = load_model(sp["repo"] if sp["kind"] == "zs" else str(md), self.device)

    def features(self, question, context):
        """(ex, feats, extract): cửa sổ trượt; phobert: tách từ trước, ánh xạ span về ngữ cảnh gốc."""
        from src.phobert import make_extract
        from src.preprocess import prepare_eval_features
        extract, ml = None, 384
        if self.phobert:
            sc = self.seg.segment(context)
            ex = {"id": ["q"], "question": [self.seg.segment(question)], "context": [sc]}
            extract, ml = make_extract([{"id": "q", "context": sc, "orig_context": context}]), 256
        else:
            ex = {"id": ["q"], "question": [question], "context": [context]}
        return ex, prepare_eval_features(ex, self.tok, max_length=ml, doc_stride=128), extract

    def logits(self, feats):
        from src.zero_shot import predict_logits_hf
        return predict_logits_hf(self.model, feats, bs=8, device=self.device)

    def answer(self, question, context, threshold=None):
        """Cùng pipeline qa_api.QA.answer."""
        ex, feats, extract = self.features(question, context)
        s, e = self.logits(feats)
        return score_once(ex, feats, s, e, extract, self.threshold if threshold is None else threshold)

    def free(self):
        import gc
        self.model = None
        gc.collect()
        if self.device == "cuda":
            import torch
            torch.cuda.empty_cache()


def score_once(ex, feats, s, e, extract, thr):
    from src.postprocess import score_examples
    r = score_examples(ex, feats, s, e, n_best=20, max_answer_length=64, extract=extract)["q"]
    return {"text": decide(r["text"], r["diff"], thr), "null_diff": r["diff"], "threshold": thr}


def cached_loaded(name, sp, run_dir=None):
    """Context manager (qa_api.ModelCache.use) yielding (Loaded, model); weights stay in CPU RAM between calls."""
    import torch
    from src.qa_api import get_cache

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    key = (name, str(run_dir or (sp["dirs"][0] if "dirs" in sp else sp["dir"])), dev)

    def load():
        m = Loaded(name, sp, run_dir)
        return m, m.model
    return get_cache().use(key, load, dev, free=lambda m: m.free())


def answer_ensemble(name, sp, question, context, threshold=None):
    """Chạy lần lượt từng run (một mô hình trên GPU mỗi lúc, trọng số cache ở RAM CPU), trung bình logit, chấm điểm một lần."""
    from src.qa_api import get_cache
    parts, ex, feats, extract, load = [], None, None, None, 0.0
    for d in sp["dirs"]:
        with cached_loaded(name, sp, d) as (m, _):
            load += get_cache().last_load
            if ex is None:  # cùng tokenizer -> cùng cửa sổ cho mọi thành viên
                ex, feats, extract = m.features(question, context)
            parts.append(m.logits(feats))
        del m
    s, e = mean_logits(parts)
    return score_once(ex, feats, s, e, extract, sp["threshold"] if threshold is None else threshold), load


def run_models(names, question, context, overrides=None, registry=None):
    """Nạp lần lượt từng mô hình, chạy, giải phóng GPU (trọng số cache ở RAM CPU, QA_CACHE=0 để tắt)."""
    registry, res = registry or REGISTRY, []
    for n in names:
        sp = registry[n]
        r = dict(name=sp.get("label", n), group=sp["group"])
        try:
            if len(sp.get("dirs", [])) > 1:
                t0 = time.time()
                out, r["load"] = answer_ensemble(n, sp, question, context, (overrides or {}).get(n))
                r.update(out)
                r["time"] = time.time() - t0 - r["load"]
            else:
                from src.qa_api import get_cache
                with cached_loaded(n, sp) as (m, _):
                    r["load"] = get_cache().last_load
                    t1 = time.time()
                    r.update(m.answer(question, context, (overrides or {}).get(n)))
                    r["time"] = time.time() - t1
                del m
        except Exception as ex:  # noqa: BLE001
            r["error"] = f"{type(ex).__name__}: {ex}"[:150]
        res.append(r)
    return res


def show_context(context, results):
    """In ngữ cảnh, tô đậm đáp án của mô hình có null_diff thấp nhất (tìm theo chuỗi)."""
    ok = [r for r in results if "error" not in r and r["text"]]
    if not ok:
        return
    best = min(ok, key=lambda r: r["null_diff"])
    i = context.find(best["text"])
    if i < 0:
        return
    j = i + len(best["text"])
    hl = paint(best["text"], "1;33") if _COLOR else f"[[{best['text']}]]"
    print(f"Ngữ cảnh ({best['name']}): " + context[max(0, i - 80):i] + hl + context[j:j + 80])


def ask(names, question, context, overrides=None, registry=None):
    res = run_models(names, question, context, overrides, registry)
    print("\n".join(format_table(res, _COLOR)))
    print("  (* = tham chiếu nhóm B; dòng xanh = các mô hình đồng thuận; thời gian = suy luận, "
          "chưa tính nạp: " + ", ".join(f"{r['name']} {r['load']:.1f}s" for r in res if "load" in r) + ")")
    show_context(context, res)
    return res


def print_models(av, registry=None):
    registry = registry or REGISTRY
    for g in "AB":
        print(GROUP_LABEL[g])
        for n, sp in registry.items():
            if sp["group"] == g:
                ok, why = av[n]
                src = sp.get("source", "")
                f1 = f", dev F1 {sp['dev_f1']:.2f}" if sp.get("dev_f1") is not None else ""
                info = f" [{src}{f1}]" if src and g == "A" else ""
                print(f"  {n:<13} {paint('khả dụng', '32') if ok else paint('không khả dụng', '31') + ' - ' + why}{info}")


# ---------- vòng lặp tương tác ----------
def read_multiline():
    print("Dán ngữ cảnh, kết thúc bằng một dòng trống:")
    lines = []
    while True:
        ln = input()
        if not ln.strip():
            return "\n".join(lines).strip()
        lines.append(ln)


def repl(legacy=False, ensemble=True):
    reg, notes = build_registry(legacy, ensemble)
    av = availability(reg)
    names = [n for n in reg if av[n][0]]
    context, over = "", {}
    for x in notes:
        print(x)
    print(paint("=== Demo so sánh 5 mô hình hỏi đáp trích xuất tiếng Việt ===", "1;36"))
    print("Gõ /help để xem lệnh. Mô hình khả dụng: " + (", ".join(names) or "(không có)"))
    while True:
        try:
            line = input(paint("> ", "36"))
        except (EOFError, KeyboardInterrupt):
            print("\nTạm biệt!")
            return 0
        try:
            cmd, arg = parse_command(line)
            if cmd == "ask":
                if not arg:
                    continue
                if not context:
                    print("Chưa có ngữ cảnh: dùng /context hoặc /sample.")
                elif not names:
                    print("Chưa chọn mô hình khả dụng nào.")
                else:
                    ask(names, arg, context, over, reg)
            elif cmd in ("quit", "exit", "q"):
                return 0
            elif cmd == "help":
                print(HELP)
            elif cmd == "models":
                print_models(av, reg)
                print("Đang chọn:", ", ".join(names) or "(không có)")
            elif cmd == "use":
                new = parse_use(arg, [n for n in reg if av[n][0]])
                for n in new:
                    if not av[n][0]:
                        print(f"  bỏ qua {n}: không khả dụng ({av[n][1]})")
                names = [n for n in new if av[n][0]]
                print("Đang chọn:", ", ".join(names))
            elif cmd in ("legacy", "ensemble", "single"):
                on = True if cmd == "single" else parse_toggle(arg)
                legacy, ensemble = {"legacy": (on, ensemble), "ensemble": (legacy, on), "single": (legacy, False)}[cmd]
                reg, notes = build_registry(legacy, ensemble)
                av = availability(reg)
                names = [n for n in names if av[n][0]]
                for x in notes:
                    print(x)
                print_models(av, reg)
            elif cmd == "context":
                context = (read_context_arg(arg) if arg else None) or read_multiline()
                print(f"Đã nhận ngữ cảnh ({len(context)} ký tự).")
            elif cmd == "sample":
                context = SAMPLE_CONTEXT
                print(SAMPLE_CONTEXT, "\nCâu hỏi mẫu:", *[f"\n  - {q}" for q in SAMPLE_QUESTIONS], sep="")
            elif cmd == "threshold":
                n, _, v = arg.partition(" ")
                if n not in REGISTRY:
                    raise ValueError(f"mô hình không hợp lệ: {n!r}")
                over[n] = float(v)
                print(f"Ngưỡng {n} = {over[n]}")
            else:
                print(f"Lệnh không rõ: /{cmd}. Gõ /help.")
        except KeyboardInterrupt:
            print("\n(đã hủy)")
        except EOFError:
            return 0
        except (ValueError, OSError) as ex:
            print(f"Lỗi: {ex}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="all", help="a,b hoặc all")
    ap.add_argument("--context-file")
    ap.add_argument("--question")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--legacy", action="store_true", help="mô hình notebook cũ thay cho bản release")
    ap.add_argument("--single", action="store_true", help="mô hình đơn tốt nhất của release thay cho ensemble (nhanh hơn ~N lần)")
    ap.add_argument("--ensemble", action="store_true", help="(mặc định; giữ để tương thích) ensemble các seed release")
    a = ap.parse_args(argv)
    init_console()
    ens = not a.single
    reg, notes = build_registry(a.legacy, ens)
    av = availability(reg)
    for x in notes:
        print(x)
    if a.list:
        print_models(av, reg)
        return 0
    avail = [n for n in reg if av[n][0]]
    if a.question:
        if not a.context_file:
            ap.error("--question cần --context-file")
        names = parse_use(a.models, avail)
        context = Path(a.context_file).read_text(encoding="utf-8").strip()
        for n in names:
            if not av[n][0]:
                print(f"{n}: không khả dụng ({av[n][1]})")
        res = ask([n for n in names if av[n][0]], a.question, context, registry=reg)
        return 0 if res else 1
    return repl(a.legacy, ens)


if __name__ == "__main__":
    sys.exit(main())
