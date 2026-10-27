# -*- coding: utf-8 -*-
"""填充类型对照：同样的位置（证据在 7.5K 上下文的正中间），只换填充的信息量。

条件（每条约 7.5K token，证据固定在第 4000 位附近）：
  real_visible   填充=真实角色页，可见
  real_masked    填充=真实角色页，被 KV 块掩码遮掉
  neutral_visible 填充=中性重复文本，可见
  neutral_masked  填充=中性重复文本，被遮
  short           只给证据 + 问题
"""
import json
import os
import re
import statistics as st
import sys
import time
import urllib.request

VIS = "/home/user/_ab_scratch/npc_vis_blocks.json"
URL = "http://127.0.0.1:8001/v1/chat/completions"
DS = "/mnt/c/Users/user/Desktop/ds harness/data/zzz/dataset"
B, PAD = 16, "\u200c"
TARGET = 7500
EVID_AT = 4000
N_FACTS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
NEUTRAL = ("The city hums with activity as trams pass by the old station. "
           "Shops open their shutters and the morning crowd moves slowly. ")
SYS = ("You answer using only the provided records. Reply with the shortest exact answer. "
       "If the records do not contain the answer, reply exactly: I don't know.")


def post(system, q, rid=None, max_tokens=20):
    body = {"model": "q17b-flex",
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": q}],
            "temperature": 0.0, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    if rid:
        body["request_id"] = rid
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, text, ptok = None, "", 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p == "[DONE]":
                break
            try:
                d = json.loads(p)
            except Exception:  # noqa: BLE001
                continue
            if d.get("usage"):
                ptok = d["usage"].get("prompt_tokens", ptok)
            ch = d.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                text += ch[0]["delta"]["content"]
    return text.strip(), ptok, ttft or 0.0


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/mnt/c/Users/user/npc-models-qwen3-1.7b")

    def ntok(s):
        return len(tok(s, add_special_tokens=False)["input_ids"])

    def clip(s, n):
        return tok.decode(tok(s, add_special_tokens=False)["input_ids"][:n])

    docs = [json.loads(l) for l in open(os.path.join(DS, "corpus.jsonl"), encoding="utf-8")]
    by_id = {d["doc_id"]: d for d in docs}
    qs = [json.loads(l) for l in open(os.path.join(DS, "queries_base.jsonl"), encoding="utf-8")]
    facts = [q for q in qs if q["fact_key"] in ("Height", "Birthday")][:N_FACTS]

    real_pool = [clip(d["text"], 180) for d in docs if d["tenant"] == "agent"]

    rows = []
    for fi, q in enumerate(facts):
        src = by_id[q["source_doc"]]
        ev = "[%s] %s" % (src["title"], clip(src["text"], 220))
        rec = {}

        for ftype in ("real", "neutral"):
            for masked in (False, True):
                head = SYS + "\n\n"
                body, spans = "", []
                cur, idx, placed = 0, 0, False
                while cur < TARGET:
                    if not placed and cur >= EVID_AT:
                        a = -(-ntok(head + body) // B)
                        body += ev + "\n"
                        body += PAD * ((B - ntok(head + body) % B) % B)
                        b = ntok(head + body) // B
                        spans.append(("EV", (a, b)))
                        placed = True
                        cur = ntok(head + body)
                        continue
                    a = -(-ntok(head + body) // B)
                    chunk = (real_pool[idx % len(real_pool)] if ftype == "real"
                             else NEUTRAL)
                    body += "[%s] %s\n" % ("filler", chunk)
                    body += PAD * ((B - ntok(head + body) % B) % B)
                    b = ntok(head + body) // B
                    spans.append(("FILL", (a, b)))
                    cur = ntok(head + body)
                    idx += 1
                text_all = head + body
                deny = [sp for tag, sp in spans if tag == "FILL"] if masked else []
                arm = "%s_%s" % (ftype, "masked" if masked else "visible")
                json.dump({arm: deny}, open(VIS, "w", encoding="utf-8"))
                ans, pt, ttft = post(text_all, q["question"],
                                     "scope=%s|%d" % (arm, fi))
                rec[arm] = {"hit": norm(q["gold"]) in norm(ans), "pt": pt,
                            "ttft": ttft, "ans": ans[:30]}

        ans, pt, ttft = post(SYS + "\n\n" + ev, q["question"])
        rec["short"] = {"hit": norm(q["gold"]) in norm(ans), "pt": pt,
                        "ttft": ttft, "ans": ans[:30]}
        rows.append({"q": q["question"], "gold": q["gold"], **rec})
        print("  %02d real_vis=%d real_mask=%d neutral_vis=%d neutral_mask=%d short=%d"
              % (fi, rec["real_visible"]["hit"], rec["real_masked"]["hit"],
                 rec["neutral_visible"]["hit"], rec["neutral_masked"]["hit"],
                 rec["short"]["hit"]), flush=True)

    print("\n%-14s %-8s %-10s" % ("条件", "命中", "TTFT"))
    for k in ("real_visible", "real_masked", "neutral_visible", "neutral_masked", "short"):
        h = sum(r[k]["hit"] for r in rows)
        tt = st.median([r[k]["ttft"] for r in rows])
        print("%-14s %-8s %-10.3f" % (k, "%d/%d" % (h, len(rows)), tt))
    json.dump(rows, open("/home/user/_ab_scratch/filler_ablation.json", "w"),
              ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
