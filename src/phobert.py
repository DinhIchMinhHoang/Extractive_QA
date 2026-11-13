"""PhoBERT glue: fast tokenizer, seg->original span extraction, segmented-data loader."""
from pathlib import Path

from src.segment import align, load_jsonl, seg_span_to_orig

ROOT = Path(__file__).resolve().parent.parent
PHOBERT = "vinai/phobert-base-v2"


def load_phobert_tokenizer():
    # AutoTokenizer gives the slow tokenizer (no offsets), so build the fast one explicitly
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(
        PHOBERT, bos_token="<s>", eos_token="</s>", sep_token="</s>", cls_token="<s>",
        unk_token="<unk>", pad_token="<pad>", mask_token="<mask>")
    # RoBERTa has type_vocab_size=1, so never feed token_type_ids
    tok.model_input_names = ["input_ids", "attention_mask"]
    # tokenizer.json has 66119 ids but the model embeds 64001: ids beyond it are rare BPE merges the slow tokenizer maps to <unk>
    tok.model_vocab_size = 64001
    return tok


def make_extract(rows):
    """extract(example_id, start_char, end_char): span in the SEGMENTED context -> ORIGINAL context text."""
    by_id = {r["id"]: r for r in rows}
    cache = {}

    def extract(example_id, s, e):
        r = by_id[example_id]
        if example_id not in cache:
            try:
                cache[example_id] = align(r["orig_context"], r["context"])[1]
            except Exception:  # noqa: BLE001
                cache[example_id] = None
                extract.failures.append(example_id)
                if len(extract.failures) == 1:
                    print(f"make_extract: align failed for {example_id!r}; best-effort text (see extract.failures)")
        s2o = cache[example_id]
        if s2o is None:  # unaligned row: best effort
            return r["context"][s:e].replace("_", " ")
        sp = seg_span_to_orig(s2o, s, e)
        return r["orig_context"][sp[0]:sp[1]] if sp else ""

    extract.failures = []
    return extract


def get_seg_data(root=None):
    """DatasetDict train/dev/test from <root>/data/processed/seg/*.jsonl or flat <root>/seg_*.jsonl (Colab)."""
    from datasets import Dataset, DatasetDict, Sequence, Value
    root = Path(root) if root else ROOT
    out = {}
    for k in ("train", "dev", "test"):
        p = root / "data" / "processed" / "seg" / f"{k}.jsonl"
        if not p.exists():
            p = root / f"seg_{k}.jsonl"
        ds = Dataset.from_list(load_jsonl(p))
        if "plausible_answers" in ds.column_names:  # force struct-of-lists even if the inferred type is odd
            ds = ds.cast_column("plausible_answers", Sequence({"text": Value("string"), "answer_start": Value("int64")}))
        out[k] = ds
    return DatasetDict(out)
