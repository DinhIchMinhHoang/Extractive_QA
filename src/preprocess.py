"""Sliding-window feature builders for extractive QA (HF fast tokenizers, run_qa style)."""


def _tokenize(examples, tokenizer, max_length, doc_stride):
    questions = [q.lstrip() for q in examples["question"]]
    right = tokenizer.padding_side == "right"
    tok = tokenizer(
        questions if right else examples["context"],
        examples["context"] if right else questions,
        truncation="only_second" if right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    limit = getattr(tokenizer, "model_vocab_size", None)
    if limit is not None:  # ids beyond the model embedding -> <unk> (as the slow tokenizer does)
        unk = tokenizer.unk_token_id
        tok["input_ids"] = [[unk if i >= limit else i for i in ids] for ids in tok["input_ids"]]
    return tok, right


def _span_tokens(offsets, input_ids, seq, ctx_id, s_char, e_char):
    """(start_tok, end_tok) of the char span if it lies fully inside this window's context, else None."""
    ts = 0
    while seq[ts] != ctx_id:
        ts += 1
    te = len(input_ids) - 1
    while seq[te] != ctx_id:
        te -= 1
    if not (offsets[ts][0] <= s_char and offsets[te][1] >= e_char):
        return None
    while ts < len(offsets) and offsets[ts][0] <= s_char:
        ts += 1
    while offsets[te][1] >= e_char:
        te -= 1
    return ts - 1, te + 1


def prepare_train_features(examples, tokenizer, max_length=384, doc_stride=128, with_plausible=False):
    """with_plausible adds plausible_start/plausible_end (same window logic as the labels, 0 if absent/outside)."""
    tok, right = _tokenize(examples, tokenizer, max_length, doc_stride)
    sample_map = tok.pop("overflow_to_sample_mapping")
    offsets_all = tok.pop("offset_mapping")
    tok["start_positions"], tok["end_positions"] = [], []
    if with_plausible:
        tok["plausible_start"], tok["plausible_end"] = [], []
        pl_all = examples.get("plausible_answers") or [None] * len(examples["question"])
    ctx_id = 1 if right else 0
    for i, offsets in enumerate(offsets_all):
        input_ids = tok["input_ids"][i]
        cls = input_ids.index(tokenizer.cls_token_id)
        seq = tok.sequence_ids(i)
        ans = examples["answers"][sample_map[i]]
        pos = None
        if len(ans["answer_start"]) > 0:
            s_char = ans["answer_start"][0]
            pos = _span_tokens(offsets, input_ids, seq, ctx_id, s_char, s_char + len(ans["text"][0]))
        tok["start_positions"].append(cls if pos is None else pos[0])
        tok["end_positions"].append(cls if pos is None else pos[1])
        if with_plausible:
            pl, pp = pl_all[sample_map[i]], None
            if len(ans["answer_start"]) == 0 and pl and len(pl.get("answer_start") or []) > 0:
                ps = pl["answer_start"][0]
                pp = _span_tokens(offsets, input_ids, seq, ctx_id, ps, ps + len(pl["text"][0]))
            tok["plausible_start"].append(0 if pp is None else pp[0])
            tok["plausible_end"].append(0 if pp is None else pp[1])
    return tok


def prepare_eval_features(examples, tokenizer, max_length=384, doc_stride=128):
    tok, right = _tokenize(examples, tokenizer, max_length, doc_stride)
    sample_map = tok.pop("overflow_to_sample_mapping")
    tok["example_id"] = []
    ctx_id = 1 if right else 0
    for i in range(len(tok["input_ids"])):
        tok["example_id"].append(examples["id"][sample_map[i]])
        seq = tok.sequence_ids(i)
        tok["offset_mapping"][i] = [o if seq[k] == ctx_id else None
                                    for k, o in enumerate(tok["offset_mapping"][i])]
    return tok
