# -*- coding: utf-8 -*-
"""看星见雅页面前 170 token 里到底有什么。"""
import json
import os

DS = "/mnt/c/Users/user/Desktop/ds harness/data/zzz/dataset"
MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL)
docs = {d["doc_id"]: d for d in (json.loads(l) for l in open(os.path.join(DS, "corpus.jsonl"), encoding="utf-8"))}
for did in ("st-071", "st-026", "st-007"):
    t = docs[did]["text"]
    ids = tok(t, add_special_tokens=False)["input_ids"][:170]
    s = tok.decode(ids)
    print("=" * 20, docs[did]["title"], "chars", len(t), "-> clipped chars", len(s))
    print(s[:700].replace("\n", " | "))
    print()
