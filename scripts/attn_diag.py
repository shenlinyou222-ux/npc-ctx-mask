# -*- coding: utf-8 -*-
"""诊断：位置间隙变大时，注意力在文档上的分布怎么变。

对每个 gap，取最后一个 token 的注意力（全层全头平均），统计：
  mass   文档 token 上的注意力总质量
  ent    文档内部注意力的熵（归一化后）
  top1   文档内最大单点注意力占比
"""
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"
DS = "/mnt/c/Users/user/Desktop/ds harness/data/zzz/dataset"
GAPS = [0, 8000, 32000, 80000, 160000]
N_FACTS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
SYS = "You read a record and answer one question about it."


@torch.no_grad()
def analyse(model, tok, ids, pos, doc_lo, doc_hi):
    out = model(input_ids=torch.tensor([ids], device="cuda"),
                position_ids=torch.tensor([pos], device="cuda"),
                output_attentions=True)
    # [layer][1, heads, q, k] -> 取最后一个 query，全层全头平均
    att = torch.stack([a[0, :, -1, :].float().mean(dim=0) for a in out.attentions])  # [L, K]
    att = att.mean(dim=0)                      # [K]
    doc = att[doc_lo:doc_hi]
    mass = float(doc.sum())
    if mass <= 1e-9:
        return {"mass": 0.0, "ent": 0.0, "top1": 0.0}
    p = doc / mass
    ent = float(-(p * (p + 1e-12).log()).sum() / torch.log(torch.tensor(float(len(p)))))
    return {"mass": mass, "ent": ent, "top1": float(p.max())}


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager").eval()

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
        head_ids = enc("%s\n\nRecord:\n" % SYS)
        ev_ids = enc(rec + "\n\n")
        q_ids = enc("Question: %s\nAnswer:" % q["question"])
        ids = head_ids + ev_ids + q_ids
        lo, hi = len(head_ids), len(head_ids) + len(ev_ids)
        res = {}
        for gap in GAPS:
            pos = list(range(len(head_ids) + len(ev_ids))) + \
                  [len(head_ids) + len(ev_ids) + gap + i for i in range(len(q_ids))]
            r = analyse(model, tok, ids, pos, lo, hi)
            res[gap] = r
            print("  %-26s gap=%-7d mass=%.6f ent=%.3f top1=%.4f"
                  % (q["question"][:26], gap, r["mass"], r["ent"], r["top1"]), flush=True)
        rows.append({"q": q["question"], **{str(k): v for k, v in res.items()}})

    print("\n%-8s %-12s %-10s %-10s" % ("间隙", "文档注意力", "归一化熵", "top1占比"))
    for gap in GAPS:
        m = sum(r[str(gap)]["mass"] for r in rows) / len(rows)
        e = sum(r[str(gap)]["ent"] for r in rows) / len(rows)
        t = sum(r[str(gap)]["top1"] for r in rows) / len(rows)
        print("%-8d %-12.6f %-10.3f %-10.4f" % (gap, m, e, t))
    json.dump(rows, open("/home/user/_ab_scratch/attn_diag.json", "w"),
              ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
