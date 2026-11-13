"""VnCoreNLP word segmentation + char-level offset alignment between original and segmented text.

CLI: python -m src.segment [--out data/processed/seg]
"""
import argparse
import glob
import json
import os
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VNCORENLP_DIR = ROOT / "data" / "vncorenlp"
LOCAL_SPLIT = ROOT / "data" / "processed" / "uit_viquad2_split"
MAX_ALIGN_FAIL_RATE = 0.005


class AlignError(Exception):
    def __init__(self, orig_snippet, seg_snippet, i, j):
        super().__init__(f"align failed at orig[{i}] seg[{j}]: {orig_snippet!r} vs {seg_snippet!r}")
        self.orig_snippet, self.seg_snippet, self.i, self.j = orig_snippet, seg_snippet, i, j


def _ensure_java_home():
    # shortcut: only globs Program Files on Windows; elsewhere set JAVA_HOME yourself
    import sys
    if os.environ.get("JAVA_HOME") or sys.platform != "win32":
        return
    hits = sorted(glob.glob("C:/Program Files/Java/*/bin/server/jvm.dll"))
    if hits:
        os.environ["JAVA_HOME"] = str(Path(hits[-1]).parents[2])


class Segmenter:
    """Lazy VnCoreNLP wseg. The JVM starts once per process, so the model is shared at class level."""
    _model = None

    def __init__(self, save_dir=VNCORENLP_DIR):
        self.save_dir = str(Path(save_dir).resolve())

    def _load(self):
        if Segmenter._model is None:
            _ensure_java_home()
            import py_vncorenlp
            cwd = os.getcwd()
            try:
                Segmenter._model = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=self.save_dir)
            finally:
                os.chdir(cwd)  # the constructor chdirs into save_dir
        return Segmenter._model

    def segment(self, text):
        if not text or not text.strip():
            return ""
        return " ".join(self._load().word_segment(text))


_TONES = {"̀", "́", "̃", "̉", "̣"}


def _key(c):
    # the segmenter only moves tone marks ("hòa" -> "hoà"): drop those, keep base letter + breve/circumflex/horn; đ stays distinct
    return "".join(x for x in unicodedata.normalize("NFD", c) if x not in _TONES).lower()


def _norm(t):
    """Comparison form: no spaces/'_', no tone marks, lowercase."""
    return "".join(_key(c) for c in t if c != "_" and not c.isspace())


def _err(orig, seg, i, j):
    return AlignError(orig[max(0, i - 15):i + 15], seg[max(0, j - 15):j + 15], i, j)


def align(orig, seg):
    """Return (o2s, s2o): char index maps, lengths len(orig)+1 / len(seg)+1 (last entry = other length).
    Chars with no counterpart (dropped orig whitespace, inserted seg space/'_') map to -1.
    A '_' that replaced an orig space is matched."""
    n, m = len(orig), len(seg)
    o2s, s2o = [-1] * n + [m], [-1] * m + [n]
    i = j = 0
    while i < n and j < m:
        o, s = orig[i], seg[j]
        if o == s or (s == "_" and o.isspace()) or _key(o) == _key(s):
            o2s[i], s2o[j] = j, i
            i += 1
            j += 1
        elif s in " _":  # inserted space, or '_' gluing a word to adjacent punctuation (Mỹ_”)
            j += 1
        elif o.isspace():  # dropped whitespace
            i += 1
        else:
            raise _err(orig, seg, i, j)
    while j < m:
        if seg[j] != " ":
            raise _err(orig, seg, i, j)
        j += 1
    while i < n:
        if not orig[i].isspace():
            raise _err(orig, seg, i, j)
        i += 1
    return o2s, s2o


def _snap(m, s, e):
    """Snap [s, e) inward past unmatched (-1) entries; None if empty/invalid."""
    s = max(s, 0)
    while s < e and m[s] == -1:
        s += 1
    while e > s and m[e - 1] == -1:
        e -= 1
    return (m[s], m[e - 1] + 1) if s < e else None


def orig_span_to_seg(o2s, s, e):
    """Original [s, e) -> segmented span, snapped inward past dropped chars. None if no matched char remains."""
    return _snap(o2s, s, e)


def seg_span_to_orig(s2o, s, e):
    """Segmented [s, e) -> original span, snapped inward past inserted chars. None if no matched char remains."""
    return _snap(s2o, s, e)


def _midword(seg, s, e):
    """True if [s, e) in the segmented text starts or ends inside a word."""
    return (s > 0 and seg[s - 1] != " ") or (e < len(seg) and seg[e] != " ")


def segment_dataset(ds, segmenter, is_train=False):
    """Segment cleaned rows; returns (list of dict rows, stats). Rows with an alignment error or an answer
    mismatch are dropped for train, kept with align_error=True, empty answers and original ref_texts otherwise."""
    seg_cache, q_cache, rows, stats = {}, {}, [], dict(
        rows=0, align_errors=0, answer_mismatch=0, dropped=0, has_ans=0, midword=0,
        plausible_total=0, plausible_dropped=0, plausible_midword=0, align_error_ids=[])
    for r in ds:
        stats["rows"] += 1
        ctx = r["context"]
        if ctx not in seg_cache:
            sc = segmenter.segment(ctx)
            try:
                seg_cache[ctx] = (sc,) + align(ctx, sc)
            except AlignError:
                seg_cache[ctx] = (sc, None, None)
        sc, o2s, _ = seg_cache[ctx]
        q = r["question"]
        if q not in q_cache:
            q_cache[q] = segmenter.segment(q)
        ans = r["answers"]
        out = {"id": r["id"], "title": r["title"], "context": sc, "question": q_cache[q],
               "answers": {"text": [], "answer_start": []}, "is_impossible": r["is_impossible"],
               "orig_context": ctx, "ref_texts": list(ans["text"])}
        out["plausible_answers"] = {"text": [], "answer_start": []}
        pl = r.get("plausible_answers") or {"text": [], "answer_start": []}
        if pl["text"]:  # failures here never count toward the align-error guard nor drop the row
            stats["plausible_total"] += 1
            psp = None if o2s is None else orig_span_to_seg(o2s, pl["answer_start"][0], pl["answer_start"][0] + len(pl["text"][0]))
            if psp is None or _norm(sc[psp[0]:psp[1]]) != _norm(pl["text"][0]):
                stats["plausible_dropped"] += 1
            else:
                out["plausible_answers"] = {"text": [sc[psp[0]:psp[1]]], "answer_start": [psp[0]]}
                stats["plausible_midword"] += _midword(sc, *psp)
        bad = o2s is None
        if bad:
            stats["align_errors"] += 1
        elif ans["text"]:
            a0 = ans["answer_start"][0]
            sp = orig_span_to_seg(o2s, a0, a0 + len(ans["text"][0]))
            if sp is None or _norm(sc[sp[0]:sp[1]]) != _norm(ans["text"][0]):
                bad = True
                stats["answer_mismatch"] += 1
            else:
                s, e = sp
                out["answers"] = {"text": [sc[s:e]], "answer_start": [s]}
                stats["has_ans"] += 1
                stats["midword"] += _midword(sc, s, e)
        if bad:
            if len(stats["align_error_ids"]) < 50:
                stats["align_error_ids"].append(r["id"])
            if is_train:
                stats["dropped"] += 1
                continue
            out["align_error"] = True
        rows.append(out)
    if stats["align_errors"] + stats["answer_mismatch"] > MAX_ALIGN_FAIL_RATE * stats["rows"]:
        raise RuntimeError(f"align failure rate too high: {stats}")
    stats["midword_pct"] = 100 * stats["midword"] / max(1, stats["has_ans"])
    return rows, stats


def save_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    import sys
    from datasets import load_from_disk
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "seg"))
    a = ap.parse_args()
    ds, seg = load_from_disk(str(LOCAL_SPLIT)), Segmenter()
    for name in ("train", "dev", "test"):
        rows, st = segment_dataset(ds[name], seg, is_train=(name == "train"))
        save_jsonl(rows, Path(a.out) / f"{name}.jsonl")
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in st.items()})
