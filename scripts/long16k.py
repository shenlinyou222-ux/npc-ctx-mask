# -*- coding: utf-8 -*-
"""真实长上下文（16K token，位置自然递增）：证据放在开头 / 中间 / 末尾。

与"位置拉伸"不同：这里序列真的长 16K，位置 0..N-1，不做任何重标定。
"""
import json
import os
import re
import statistics as st
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8001/v1/chat/completions"
DS = "/mnt/c/Users/user/Desktop/ds harness/data/zzz/dataset"
TARGET = 15800
POSITIONS = [0, 7900, 15500]
N_FACTS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
NEUTRAL = ("The city hums with activity as trams pass by the old station. "
           "Shops open their shutters and the morning crowd moves slowly. ")
SYS = ("You answer using only the provided records. Reply with the shortest exact answer. "
       "If the records do not contain the answer, reply exactly: I don't know.")


def post(system, q, rid=None, max_tokens=24):
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
    text, ptok = "", 0
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
                text += ch[0]["delta"]["content"]
    return text.strip(), ptok, time.perf_counter() - t0


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def hit(gold, ans):
    g, a = norm(gold), norm(ans)
    for L in (len(g), 8, 6, 4):
        if 0 < L <= len(g) and g[:L] in a:
            return True
    return False


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/mnt/c/Users/user/npc-models-qwen3-1.7b")

    def ntok(s):
        return len(tok(s, add_special_tokens=False)["input_ids"])

    def clip(s, n):
        return tok.decode(tok(s, add_special_tokens=False)["input_ids"][:n])

    docs = {}
    for line in open(os.path.join(DS, "corpus.jsonl"), encoding="utf-8"):
        d = json.loads(line)
        docs[d["doc_id"]] = d
    qs = [json.loads(l) for l in open(os.path.join(DS, "queries_base.jsonl"), encoding="utf-8")]
    facts = [q for q in qs if q["fact_key"] in ("Height", "Birthday")][:N_FACTS]

    rows = []
    for q in facts:
        ev = "[%s] %s" % (q["source_doc"], clip(docs[q["source_doc"]]["text"], 240))
        res = {}
        for pos in POSITIONS:
            head = SYS + "\n\n"
            body = ""
            # 先填到目标位置
            while ntok(head + body) < pos:
                body += "[filler] " + NEUTRAL + "\n"
            body += ev + "\n"
            while ntok(head + body) < TARGET:
                body += "[filler] " + NEUTRAL + "\n"
            text = head + body
            ans, pt, sec = post(text, q["question"])
            res[pos] = {"hit": hit(q["gold"], ans), "pt": pt, "sec": round(sec, 2),
                        "ans": ans[:40]}
            print("  %-26s pos=%-6d tok=%-6d hit=%d %.1fs %r"
                  % (q["question"][:26], pos, pt, res[pos]["hit"], sec, ans[:26]), flush=True)
        rows.append({"q": q["question"], "gold": q["gold"], **{str(k): v for k, v in res.items()}})

    print("\n%-8s %-10s" % ("位置", "命中"))
    for pos in POSITIONS:
        print("%-8d %d/%d" % (pos, sum(r[str(pos)]["hit"] for r in rows), len(rows)))
    json.dump(rows, open("/home/user/_ab_scratch/long16k.json", "w"),
              ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
