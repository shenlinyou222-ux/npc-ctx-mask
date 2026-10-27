# -*- coding: utf-8 -*-
"""注意力去向拆解：无掩码，只看位置间隙变大时注意力分给了谁。

分段：HEAD(系统提示) / DOC(证据) / Q(问题) / SELF(最后一个 token 自己)
"""
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/mnt/c/Users/npc-models-qwen3-1.7b"
if not os.path.exists(MODEL):
    MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"
DS = "/mnt/c/Users/user/Desktop/ds harness/data/zzz/dataset"
GAPS = [0, 32000, 80000]
N_FACTS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
SYS = "You read a record and answer one question about it."


@torch.no_grad()
def seg_mass(model, ids, pos, bounds):
    out = model(input_ids=torch.tensor([ids], device="cuda"),
                position_ids=torch.tensor([pos], device="cuda"),
                output_attentions=True)
    att = torch.stack([a[0, :, -1, :].float().mean(dim=0) for a in out.attentions]).mean(dim=0)
    res = {}
    for name, (lo, hi) in bounds.items():
        res[name] = float(att[lo:hi].sum())
    res["self"] = float(att[-1])
    return res


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
        h = enc("%s\n\nRecord:\n" % SYS)
        e = enc(rec + "\n\n")
        qt = enc("Question: %s\nAnswer:" % q["question"])
        ids = h + e + qt
        bounds = {"HEAD": (0, len(h)), "DOC": (len(h), len(h) + len(e)),
                  "Q": (len(h) + len(e), len(ids))}
        res = {}
        for gap in GAPS:
            pos = list(range(len(h) + len(e))) + \
                  [len(h) + len(e) + gap + i for i in range(len(qt))]
            r = seg_mass(model, ids, pos, bounds)
            res[gap] = r
            print("  %-24s gap=%-6d HEAD=%.3f DOC=%.3f Q=%.3f self=%.3f"
                  % (q["question"][:24], gap, r["HEAD"], r["DOC"], r["Q"], r["self"]),
                  flush=True)
        rows.append({"q": q["question"], **{str(k): v for k, v in res.items()}})

    print("\n%-8s %-8s %-8s %-8s %-8s" % ("间隙", "HEAD", "DOC", "Q", "self"))
    for gap in GAPS:
        f = lambda k: sum(r[str(gap)][k] for r in rows) / len(rows)  # noqa: E731
        print("%-8d %-8.3f %-8.3f %-8.3f %-8.3f" % (gap, f("HEAD"), f("DOC"), f("Q"), f("self")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
