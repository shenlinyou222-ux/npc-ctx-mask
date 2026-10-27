# -*- coding: utf-8 -*-
"""位置压缩能不能抵抗衰减：间隙不变，把位置轴压缩。

gap=80000 时：
  scale=1.0  有效距离 80K（基线，应该开始崩）
  scale=0.5  有效距离 40K（压缩 2 倍）
  scale=0.25 有效距离 20K（压缩 4 倍）
如果压缩能救回命中率，说明"额外位置编码计算"这条路可行。
"""
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"
if not os.path.exists(MODEL):
    MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"
DS = "/mnt/c/Users/user/Desktop/ds harness/data/zzz/dataset"
GAPS = [80000, 160000]
SCALES = [1.0, 0.5, 0.25]
N_FACTS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
SYS = "You read a record and answer one question about it."


@torch.no_grad()
def greedy(model, tok, ids, pos, n_new=20):
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

    docs = {}
    for line in open(os.path.join(DS, "corpus.jsonl"), encoding="utf-8"):
        d = json.loads(line)
        docs[d["doc_id"]] = d
    qs = [json.loads(l) for l in open(os.path.join(DS, "queries_base.jsonl"), encoding="utf-8")]
    facts = [q for q in qs if q["fact_key"] in ("Height", "Birthday")][:N_FACTS]

    def enc(t):
        return tok(t, add_special_tokens=False)["input_ids"]

    rows = []
    for q in facts:
        rec = docs[q["source_doc"]]["text"][:900]
        h = enc("%s\n\nRecord:\n" % SYS)
        e = enc(rec + "\n\n")
        qt = enc("Question: %s\nAnswer:" % q["question"])
        ids = h + e + qt
        base = len(h) + len(e)
        res = {}
        for gap in GAPS:
            for s in SCALES:
                pos = list(range(base)) + [base + int(gap * s) + i for i in range(len(qt))]
                gen = greedy(model, tok, ids, pos)
                ok = hit(q["gold"], gen)
                res["%d_%s" % (gap, s)] = {"hit": ok, "gen": gen[:40],
                                           "eff_gap": int(gap * s)}
                print("  %-22s gap=%-7d s=%-5s eff=%-6d hit=%d %r"
                      % (q["question"][:22], gap, s, int(gap * s), ok, gen[:26]), flush=True)
        rows.append({"q": q["question"], "gold": q["gold"], **res})

    print("\n%-10s %-8s %-8s %-8s" % ("原始间隙", "s=1.0", "s=0.5", "s=0.25"))
    for gap in GAPS:
        cells = []
        for s in SCALES:
            k = "%d_%s" % (gap, s)
            cells.append("%d/%d" % (sum(r[k]["hit"] for r in rows), len(rows)))
        print("%-10d %-8s %-8s %-8s" % (gap, cells[0], cells[1], cells[2]))
    json.dump(rows, open("/home/user/_ab_scratch/pos_compress.json", "w"),
              ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
