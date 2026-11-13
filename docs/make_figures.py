"""Generate report figures into docs/figures. Run from repo root: python docs/make_figures.py"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP = os.path.join(ROOT, "outputs", "sweep")
OUT = os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})

A = ["xlmr_base", "phobert", "xlmr_large"]
ALL = A + ["vi_mrc_base", "vi_mrc_large"]
COL = {"xlmr_base": "tab:blue", "phobert": "tab:orange", "xlmr_large": "tab:green",
       "vi_mrc_base": "grey", "vi_mrc_large": "darkred"}

def nb_texts(num):
    """Saved text outputs (one string per code cell) of notebooks/<num>_*.ipynb. Notebooks are never re-run."""
    p = glob.glob(os.path.join(ROOT, "notebooks", f"{num}_*.ipynb"))[0]
    with open(p, encoding="utf-8") as f:
        cells = json.load(f)["cells"]
    out = []
    for c in cells:
        if c["cell_type"] != "code":
            continue
        t = ""
        for o in c.get("outputs", []):
            t += "".join(o["text"]) if "text" in o else "".join(o.get("data", {}).get("text/plain", []))
        out.append(t)
    return out


def ci_table(texts, n):
    """Parse the 'subset metric n mean lo hi' bootstrap table with sample size n -> {(model, subset, metric): (mean, lo, hi)}."""
    row = re.compile(r"^(?:(\w+)\s+)?\d+\s+(overall|HasAns|NoAns)\s+(em|f1)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")
    for t in texts:
        if "subset metric" not in t:
            continue
        res, model = {}, None
        for line in t.splitlines():
            m = row.match(line.strip())
            if m:
                model = m.group(1) or model
                res[(model, m.group(2), m.group(3))] = (int(m.group(4)),) + tuple(float(m.group(i)) for i in (5, 6, 7))
        # table identified by the overall-subset sample size (HasAns/NoAns rows have smaller n)
        if res and all(v[0] == n for k, v in res.items() if k[1] == "overall"):
            return {k: v[1:] for k, v in res.items()}
    raise RuntimeError(f"no CI table with n={n}")


def first(pattern, text):
    m = re.search(pattern, text, re.M)
    if not m:
        raise RuntimeError(f"pattern not found: {pattern}")
    return m


NB05, NB06 = nb_texts("05"), nb_texts("06")
_t, _d = ci_table(NB05, 3814), ci_table(NB05, 2912)
TEST = {  # f1, ci_lo, ci_hi, em, hasans_f1, noans_acc  (nb05 test CI table; scored once)
    m: (_t[m, "overall", "f1"][0], _t[m, "overall", "f1"][1], _t[m, "overall", "f1"][2], _t[m, "overall", "em"][0],
        _t[m, "HasAns", "f1"][0], _t[m, "NoAns", "em"][0]) for m in ALL}
DEV = {m: _d[m, "overall", "f1"][0] for m in ALL}  # nb05 dev CI table
_all05 = "\n".join(NB05)
SEED_TEST = {m: (float(a), float(b)) for m, a, b in
             re.findall(r"^(\w+): ensemble F1 [\d.]+ \| seeds F1 ([\d.]+) ± ([\d.]+)", _all05, re.M)}  # nb05 test cell
PARAMS = {m: f"{float(p):g}M" for m, p in re.findall(r"^\d+\s+(\w+)\s+([\d.]+)\s+\d+\s", "\n".join(NB05), re.M)
          if m in A}  # nb05 resource table (params_M)

_e = next(t for t in NB06 if "TỔNG lỗi" in t)
ERR_TOTAL = int(first(r"TỔNG lỗi \(EM=0\)\s+(\d+)", _e).group(1))
N_TEST = int(first(r"^n=(\d+) \|", "\n".join(NB06)).group(1))
_c = lambda pat: int(first(pat + r"\s+(\d+)\s+[\d.]+\s+[\d.]+\s*$", _e).group(1))  # noqa: E731
ERRORS = [  # label, count (xlmr_large ens, test) from nb06 category table
    ("Lệch ranh giới (boundary)", _c(r"^\(c\) .*?"), ), ("↳ super-span", _c(r"super-span")),
    ("↳ sub-span", _c(r"sub-span")), ("↳ shifted", _c(r"shifted")),
    ("Có đáp án nhưng dự đoán rỗng", _c(r"^\(b\) .*?")),
    ("Không đáp án nhưng trả lời", _c(r"^\(a\) .*?")), ("↳ trùng đáp án plausible", _c(r"= plausible")),
    ("↳ không trùng", _c(r"khác plausible")),
    ("Sai vị trí (wrong place)", _c(r"^\(d\) .*?")), ("↳ câu khác", _c(r"câu khác")), ("↳ cùng câu", _c(r"cùng câu")),
]

# Previously hard-coded values, kept only to prove the parse is right (mismatch > 0.01 -> exit 1).
EXPECTED = {
    "TEST": {
        "xlmr_base": (62.40, 60.89, 63.78, 53.99, 64.15, 58.40),
        "phobert": (68.97, 67.63, 70.33, 59.49, 71.18, 63.91),
        "xlmr_large": (77.89, 76.68, 79.15, 68.35, 78.39, 76.74),
        "vi_mrc_base": (64.72, 63.25, 66.18, 57.26, 77.43, 35.66),
        "vi_mrc_large": (80.48, 79.33, 81.63, 68.90, 86.45, 66.84),
    },
    "DEV": {"xlmr_base": 62.87, "phobert": 68.75, "xlmr_large": 78.86, "vi_mrc_base": 64.45, "vi_mrc_large": 89.74},
    "SEED_TEST": {"xlmr_base": (61.95, 1.25), "phobert": (67.38, 0.78), "xlmr_large": (76.40, 1.25)},
    "ERR_TOTAL": 1207, "N_TEST": 3814,
    "ERRORS": [478, 256, 213, 9, 316, 270, 143, 127, 143, 105, 38],
    "PARAMS": {"xlmr_base": "277.5M", "phobert": "135M", "xlmr_large": "559.9M"},
}


def _flat(x):
    return [float(v) for v in x] if isinstance(x, (tuple, list)) else [float(x)]


def check_against_expected():
    got = {"TEST": TEST, "DEV": DEV, "SEED_TEST": SEED_TEST, "ERR_TOTAL": ERR_TOTAL, "N_TEST": N_TEST,
           "ERRORS": [c for _, c in ERRORS], "PARAMS": PARAMS}
    bad = []
    for k, exp in EXPECTED.items():
        g = got[k]
        if k == "PARAMS":
            bad += [f"PARAMS[{m}]: {g.get(m)} != {v}" for m, v in exp.items() if g.get(m) != v]
        elif isinstance(exp, dict):
            for m, v in exp.items():
                if m not in g or any(abs(a - b) > 0.01 for a, b in zip(_flat(g[m]), _flat(v))) or len(_flat(g[m])) != len(_flat(v)):
                    bad.append(f"{k}[{m}]: parsed {g.get(m)} != expected {v}")
        elif len(_flat(g)) != len(_flat(exp)) or any(abs(a - b) > 0.01 for a, b in zip(_flat(g), _flat(exp))):
            bad.append(f"{k}: parsed {g} != expected {exp}")
    if bad:
        print("MISMATCH:\n  " + "\n  ".join(bad))
        sys.exit(1)
    print("parse check OK: all notebook-parsed values match EXPECTED")


check_against_expected()
# Parsed from notebook outputs: TEST, DEV, SEED_TEST, PARAMS (nb05); ERRORS, ERR_TOTAL, N_TEST (nb06).
# NOT IN NOTEBOOK OUTPUT: nothing hard-coded; est_units (fig08) and curves (fig09) come from sweep results/summary files.


def jl(stage, m):
    with open(os.path.join(SWEEP, f"stage{stage}", m, "results.jsonl"), encoding="utf-8") as f:
        return [d for d in map(json.loads, f) if d.get("status") == "ok"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("wrote", name)


def short(run):
    return run.split("__", 1)[1]


# fig01
s1 = {m: jl(1, m) for m in A}
sel = {m: max(s1[m], key=lambda d: d["f1_tuned"]) for m in A}
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for ax, m in zip(axs, A):
    rs = s1[m]
    bars = ax.bar(range(len(rs)), [d["f1_tuned"] for d in rs], color=COL[m], alpha=0.5)
    for i, (b, d) in enumerate(zip(bars, rs)):
        if d is sel[m]:
            b.set_alpha(1); b.set_edgecolor("black"); b.set_linewidth(2)
            ax.text(i, d["f1_tuned"] + 0.15, "★ chọn", ha="center", fontsize=9, weight="bold")
        ax.text(i, min(x["f1_tuned"] for x in rs) - 2.5, f'{d["f1_tuned"]:.2f}', ha="center", color="black", weight="bold")
    ax.set_xticks(range(len(rs)))
    ax.set_xticklabels([short(d["run_name"]).replace("base_", "").replace("_", "\n") for d in rs], fontsize=8)
    lo = min(d["f1_tuned"] for d in rs)
    ax.set_ylim(lo - 3, max(d["f1_tuned"] for d in rs) + 1)
    ax.set_title(m); ax.set_ylabel("F1 dev (đã tinh chỉnh ngưỡng)")
fig.suptitle("Giai đoạn 1: so sánh siêu tham số (lr / epochs) theo mô hình")
save(fig, "fig01_stage1_hparams.png")

# fig02
fig, axs = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
labels = [("llrd0.9", "LLRD 0.9"), ("ema0.999", "EMA 0.999"), ("pl0.1", "λ=0.1"), ("pl0.3", "λ=0.3")]
for ax, m in zip(axs, A):
    base = sel[m]["f1_tuned"]
    s2 = {short(d["run_name"]): d["f1_tuned"] - base for d in jl(2, m)}
    vals = [s2[k] for k, _ in labels]
    bars = ax.bar([l for _, l in labels], vals, color=COL[m])
    ax.axhline(0, color="black", lw=1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.1 if v >= 0 else -0.1), f"{v:+.2f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax.set_title(f"{m} (cơ sở {base:.2f})"); ax.tick_params(axis="x", labelsize=8)
    ax.margins(y=0.15)
axs[0].set_ylabel("ΔF1 dev so với cấu hình giai đoạn 1")
fig.suptitle("Giai đoạn 2: hiệu quả của các kỹ thuật bổ sung")
save(fig, "fig02_stage2_addons.png")

# fig03
s3 = {m: jl(3, m) for m in A}
fig, ax = plt.subplots(figsize=(7, 5))
for i, m in enumerate(A):
    v = [d["f1_tuned"] for d in s3[m]]
    ax.scatter(np.full(len(v), i) + np.linspace(-0.08, 0.08, len(v)), v, color=COL[m], s=60, zorder=3)
    ax.hlines(np.mean(v), i - 0.25, i + 0.25, color="black", lw=2, zorder=4)
    ax.text(i + 0.28, np.mean(v), f"TB {np.mean(v):.2f}", va="center", fontsize=9)
ax.set_xticks(range(3)); ax.set_xticklabels(A); ax.set_xlim(-0.5, 2.9)
ax.set_ylabel("F1 dev (tinh chỉnh ngưỡng)")
ax.set_title("Giai đoạn 3: F1 dev theo từng seed (42, 43, 44) và trung bình")
save(fig, "fig03_seeds_dev.png")

# fig04
fig, ax = plt.subplots(figsize=(9, 5))
for i, m in enumerate(ALL):
    f1, lo, hi = TEST[m][:3]
    grp_b = m not in A
    ax.bar(i, f1, color=COL[m], hatch="//" if grp_b else None, edgecolor="white" if not grp_b else "black",
           alpha=0.75 if grp_b else 1)
    ax.errorbar(i, f1, yerr=[[f1 - lo], [hi - f1]], color="black", capsize=5)
    ax.text(i, hi + 0.5, f"{f1:.2f}\n[{lo:.2f}, {hi:.2f}]", ha="center", fontsize=8)
ax.set_xticks(range(5))
ax.set_xticklabels([m if m in A else m + "\n(tham chiếu)" for m in ALL], fontsize=8)
ax.set_ylim(50, 90); ax.set_ylabel("F1 test (%)")
ax.set_title(f"F1 trên tập test ({N_TEST} mẫu) với khoảng tin cậy 95%")
save(fig, "fig04_test_f1_ci.png")

# fig05
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(5); w = 0.38
for i, m in enumerate(ALL):
    ax.bar(i - w / 2, DEV[m], w, color=COL[m], alpha=0.45, edgecolor="black")
    ax.bar(i + w / 2, TEST[m][0], w, color=COL[m])
    ax.text(i - w / 2, DEV[m] + 0.5, f"{DEV[m]:.2f}", ha="center", fontsize=8)
    ax.text(i + w / 2, TEST[m][0] + 0.5, f"{TEST[m][0]:.2f}", ha="center", fontsize=8)
ax.annotate(f"chênh {DEV['vi_mrc_large'] - TEST['vi_mrc_large'][0]:.2f}", xy=(4, 90), xytext=(2.6, 92),
            arrowprops={"arrowstyle": "->"}, fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(ALL, fontsize=8); ax.set_ylim(50, 96)
ax.set_ylabel("F1 (%)")
ax.set_title("F1 dev (nhạt) so với F1 test (đậm)")
save(fig, "fig05_dev_vs_test.png")

# fig06
fig, ax = plt.subplots(figsize=(9, 5))
for i, m in enumerate(ALL):
    ax.bar(i - w / 2, TEST[m][4], w, color=COL[m])
    ax.bar(i + w / 2, TEST[m][5], w, color=COL[m], hatch="//", edgecolor="white")
    ax.text(i - w / 2, TEST[m][4] + 0.8, f"{TEST[m][4]:.1f}", ha="center", fontsize=8)
    ax.text(i + w / 2, TEST[m][5] + 0.8, f"{TEST[m][5]:.1f}", ha="center", fontsize=8)
ax.set_xticks(range(5)); ax.set_xticklabels(ALL, fontsize=8); ax.set_ylim(0, 100)
ax.set_ylabel("%")
ax.set_title("Test: F1 câu có đáp án (đặc) và độ chính xác câu không đáp án (sọc)")
save(fig, "fig06_hasans_vs_noans.png")

# fig07
fig, ax = plt.subplots(figsize=(9, 5.5))
labs = [l for l, _ in ERRORS][::-1]; cnt = [c for _, c in ERRORS][::-1]
cols = ["tab:green" if l.startswith("↳") else "darkgreen" for l in labs]
ax.barh(range(len(cnt)), [c / ERR_TOTAL * 100 for c in cnt], color=cols)
for i, c in enumerate(cnt):
    ax.text(c / ERR_TOTAL * 100 + 0.5, i, f"{c} ({c / ERR_TOTAL * 100:.1f}%)", va="center", fontsize=9)
ax.set_yticks(range(len(cnt))); ax.set_yticklabels(labs)
ax.set_xlim(0, 50); ax.set_xlabel(f"% trên tổng {ERR_TOTAL} lỗi")
ax.set_title(f"Phân loại lỗi của xlmr_large (ensemble, test; {ERR_TOTAL}/{N_TEST} lỗi)")
save(fig, "fig07_error_categories.png")

# fig08
fig, ax = plt.subplots(figsize=(8, 5))
for m in A:
    u = np.mean([d["est_units"] for d in s3[m]])
    ax.scatter(u, TEST[m][0], s=140, color=COL[m], zorder=3)
    ax.annotate(f"{m}\n{PARAMS[m]} tham số", (u, TEST[m][0]), textcoords="offset points", xytext=(10, -22), fontsize=9)
ax.set_xlim(0.4, 2.0); ax.set_ylim(58, 82)
ax.set_xlabel("Chi phí ước tính (Colab units / lần chạy, trung bình giai đoạn 3)")
ax.set_ylabel("F1 test (ensemble, %)")
ax.set_title("Chi phí huấn luyện so với F1 test")
save(fig, "fig08_cost_vs_f1.png")

# fig09
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
done = 0
for m in A:
    p = os.path.join(SWEEP, "stage3", m, f"{m}__final_s42", "summary.json")
    if not os.path.exists(p):
        print("skip curve", m, "(no summary.json)"); continue
    with open(p, encoding="utf-8") as f:
        lh = json.load(f).get("log_history", [])
    tr = [(d["epoch"], d["loss"]) for d in lh if "loss" in d]
    ev = [(d["epoch"], d["eval_loss"]) for d in lh if "eval_loss" in d]
    if tr:
        axs[0].plot(*zip(*tr), color=COL[m], label=m)
    if ev:
        axs[1].plot(*zip(*ev), color=COL[m], marker="o", label=m)
    done += 1
axs[0].set_title("Loss huấn luyện"); axs[1].set_title("Loss đánh giá (dev)")
for ax in axs:
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend()
fig.suptitle("Đường cong huấn luyện (seed 42, cấu hình cuối)")
if done:
    save(fig, "fig09_training_curve.png")
