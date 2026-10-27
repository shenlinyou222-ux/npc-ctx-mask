# -*- coding: utf-8 -*-
"""位置拉伸模拟：保持单调递增，把位置整体乘 s。

s=1 -> 正常；s=2/4/8 -> 7.5K token 撑到 15K/30K/60K 的位置跨度（超训练范围）。
证据固定在物理槽位 ~4000，填充用中性文本（排除内容竞争）。
"""
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"
DS = "/mnt/c/Users/user/Desktop/ds harness/data/zzz/dataset"
N_FACTS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
FILLER_TOK = 3800
STRETCH = [1, 2, 4, 8]
SYS = "You read a record and answer one question about it."


@torch.no_grad()
def greedy(model, tok, ids, pos, n_new=24):
    cur_ids, cur_pos, out = list(ids), list(pos), []
    for _ in range(n_new):
        logits = model(input_ids=torch.tensor([cur_ids], device="cuda"),
                       position_ids=torch.tensor([cur_pos], device="cuda")).logits[0, -1]
        if torch.isnan(logits).any():
            return "<NaN>"
        nxt = int(torch.argmax(logits))
        if nxt == tok.eos_token_id:
            break
        out.append(nxt)
        cur_ids.append(nxt)
        cur_pos.append(cur_pos[-1] + 1)
    return tok.decode(out, skip_special_tokens=True)


def norm(s):
    return "".join(c for c in s.lower() if c.isalnum())


def hit(gold, ans):
    g, a = norm(gold), norm(ans)
    for L in (len(g), 8, 6, 4):
        if 0 < L <= len(g) and g[:L] in a:
            return True
    return False


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    print("max_position_embeddings =", model.config.max_position_embeddings)

    docs = {}
    for line in open(os.path.join(DS, "corpus.jsonl"), encoding="utf-8"):
        d = json.loads(line)
        docs[d["doc_id"]] = d
    qs = [json.loads(l) for l in open(os.path.join(DS, "queries_base.jsonl"), encoding="utf-8")]
    facts = [q for q in qs if q["fact_key"] in ("Height", "Birthday")][:N_FACTS]

    def enc(t):
        return tok(t, add_special_tokens=False)["input_ids"]

    filler = enc("The city hums with activity as trams pass by the old station. " * 400)[:FILLER_TOK]

    rows = []
    for q in facts:
        rec = docs[q["source_doc"]]["text"][:900]
        ids = enc("%s\n\nRecord:\n" % SYS) + filler + enc(rec + "\n\n") \
            + enc("Question: %s\nAnswer:" % q["question"])
        res = {}
        for s in STRETCH:
            pos = [i * s for i in range(len(ids))]
            gen = greedy(model, tok, ids, pos)
            res[s] = {"hit": hit(q["gold"], gen), "gen": gen[:50]}
            print("  %-28s s=%-3d maxpos=%-6d hit=%d %r"
                  % (q["question"][:28], s, pos[-1], res[s]["hit"], gen[:32]), flush=True)
        rows.append({"q": q["question"], "gold": q["gold"], **{str(k): v for k, v in res.items()}})

    print("\n%-6s %-10s %-12s" % ("拉伸", "命中", "最大位置"))
    for s in STRETCH:
        h = sum(r[str(s)]["hit"] for r in rows)
        print("%-6d %-10s %-12d" % (s, "%d/%d" % (h, len(rows)), (len(ids) - 1) * s))
    json.dump(rows, open("/home/user/_ab_scratch/pos_stretch.json", "w"),
              ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
