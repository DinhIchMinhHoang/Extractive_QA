"""Group-aware train/dev split for UIT-ViQuAD2.0 rows (plain dicts, pure python)."""
import hashlib
import random
from collections import OrderedDict


def _group_key(r):
    # Same title => same article => same/near contexts, so keep them on one side.
    t = r.get("title")
    if t:
        return "t:" + t
    return "c:" + hashlib.md5((r.get("context") or "").encode("utf-8")).hexdigest()


def split_train_dev(rows, dev_frac=0.10, seed=42, tol=0.03):
    """Split rows into (train_rows, dev_rows) by title group; rows are returned unchanged.

    Groups are shuffled with random.Random(seed) and added to dev while the total stays
    <= 1.1 * target and the dev NoAns ratio stays within `tol` of the global ratio (or does
    not move further away from it).
    shortcut: single greedy pass, so dev can end slightly below target and the NoAns match is
    approximate; upgrade to iterative swapping if a tighter stratification is needed.
    """
    rows = list(rows)
    groups = OrderedDict()
    for r in rows:
        groups.setdefault(_group_key(r), []).append(r)
    keys = sorted(groups)  # fixed base order so the shuffle depends only on seed
    random.Random(seed).shuffle(keys)

    total = len(rows)
    target = dev_frac * total
    g_ratio = sum(bool(r.get("is_impossible")) for r in rows) / max(total, 1)
    dev_keys, dev_n, dev_no = set(), 0, 0
    for k in keys:
        if dev_n >= target:
            break
        n = len(groups[k])
        no = sum(bool(r.get("is_impossible")) for r in groups[k])
        if dev_n + n > 1.1 * target:
            continue
        cur = abs(dev_no / dev_n - g_ratio) if dev_n else 0.0
        new = abs((dev_no + no) / (dev_n + n) - g_ratio)
        if new <= max(cur, tol):
            dev_keys.add(k)
            dev_n += n
            dev_no += no

    train_rows = [r for r in rows if _group_key(r) not in dev_keys]
    dev_rows = [r for r in rows if _group_key(r) in dev_keys]
    return train_rows, dev_rows


def _selftest():
    rng = random.Random(0)
    rows = []
    for t in range(100):
        for i in range(rng.randint(5, 15)):
            rows.append({"id": f"{t}-{i}", "title": f"T{t}", "context": f"ctx{t}",
                         "is_impossible": rng.random() < 0.32})
    tr, dv = split_train_dev(rows, 0.10, 42)
    assert len(tr) + len(dv) == len(rows)
    assert {r["title"] for r in tr}.isdisjoint({r["title"] for r in dv})
    frac = len(dv) / len(rows)
    assert 0.07 <= frac <= 0.13, frac
    ratio = lambda xs: sum(r["is_impossible"] for r in xs) / len(xs)
    assert abs(ratio(dv) - ratio(rows)) <= 0.06, (ratio(dv), ratio(rows))
    tr2, dv2 = split_train_dev(rows, 0.10, 42)
    assert [r["id"] for r in dv] == [r["id"] for r in dv2]
    assert [r["id"] for r in tr] == [r["id"] for r in tr2]
    # title missing -> falls back to context hash
    nt = [{"id": str(i), "title": None, "context": f"c{i % 20}", "is_impossible": False} for i in range(200)]
    a, b = split_train_dev(nt, 0.10, 1)
    assert {r["context"] for r in a}.isdisjoint({r["context"] for r in b})
    print(f"OK: dev {len(dv)}/{len(rows)} ({frac:.3f}), NoAns dev {ratio(dv):.3f} vs all {ratio(rows):.3f}")


if __name__ == "__main__":
    _selftest()
