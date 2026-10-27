# -*- coding: utf-8 -*-
"""位置间隙扫描 v2：更干净的提示格式 + 无文档对照。

提示：system + 记录 + "Question: ..." + "Answer:"，只有 position_ids 不同。
记录 ~200 token，问题 ~20 token，KV 恒定 ~300，不吃显存。
"""
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"
DS = "/mnt/c/Users/user/Desktop/ds harness/data/zzz/dataset"
GAPS = [0, 8000, 32000, 80000, 160000, 320000]
N_FACTS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
SYS = "You read a record and answer one question about it."
CUT_CHARS = 900


@torch.no_grad()
def greedy(model, tok, ids, pos, n_new=16):
    cur_ids, cur_pos, out = list(ids), list(pos), []
    for _ in range(n_new):
        logits = model(input_ids=torch.tensor([cur_ids], device="cuda"),
                       position_ids=torch.tensor([cur_pos], device="cuda")).logits[0, -1]
        nxt = int(torch.argmax(logits))
        if nxt == tok.eos_token_id:
            break
        out.append(nxt)
        cur_ids.append(nxt)
        cur_pos.append(cur_pos[-1] + 1)
    return tok.decode(out, skip_special_tokens=True)


def norm(s):
    return "".join(c for c in s.lower() if c.isalnum())


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda").eval()

    docs = {}
    for line in open(os.path.join(DS, "corpus.jsonl"), encoding="utf-8"):
        d = json.loads(line)
        docs[d["doc_id"]] = d
    qs = [json.loads(l) for l in open(os.path.join(DS, "queries_base.jsonl"), encoding="utf-8")]
    facts = [q for q in qs if q["fact_key"] in ("Height", "Birthday", "Real Name")][:N_FACTS]

    def enc(t):
        return tok(t, add_special_tokens=False)["input_ids"]

    rows = []
    for q in facts:
        rec = docs[q["source_doc"]]["text"][:CUT_CHARS]
        ctx_ids = enc("%s\n\nRecord:\n%s\n\n" % (SYS, rec))
        tail = enc("Question: %s\nAnswer:" % q["question"])
        res = {}
        for gap in GAPS:
            pos = list(range(len(ctx_ids))) + [len(ctx_ids) + gap + i for i in range(len(tail))]
            gen = greedy(model, tok, ctx_ids + tail, pos)
            res[gap] = {"hit": norm(q["gold"]) in norm(gen), "gen": gen[:50]}
            print("  %-30s gap=%-6d hit=%d %r" % (q["question"][:30], gap, res[gap]["hit"],
                                                  gen[:34]), flush=True)
        # 无文档对照
        gen0 = greedy(model, tok, tail, list(range(len(tail))))
        res["nodoc"] = {"hit": norm(q["gold"]) in norm(gen0), "gen": gen0[:50]}
        rows.append({"q": q["question"], "gold": q["gold"], **res})

    print("\n%-8s %-10s" % ("间隙", "命中"))
    for gap in GAPS:
        print("%-8d %d/%d" % (gap, sum(r[gap]["hit"] for r in rows), len(rows)))
    print("%-8s %d/%d" % ("无文档", sum(r["nodoc"]["hit"] for r in rows), len(rows)))
    json.dump(rows, open("/home/user/_ab_scratch/position_gap2.json", "w"),
              ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
