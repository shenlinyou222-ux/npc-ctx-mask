# -*- coding: utf-8 -*-
"""验证「每页独立 prefill」v2：增量建库（前缀逐页变长），正序 vs 倒序。

建库第 k 步：序列只有 system + 前 k 页；把第 1..k-1 页挡掉。
于是第 k 页的 K/V 只可能看过 system——它出现的那一刻，前面的页对它不可见。
最后逐页单独提问，比较正序/倒序两次建库的答案。
"""
import json
import os
import sys
import time
import urllib.request

VIS = "/home/user/_ab_scratch/npc_vis_blocks.json"
URL = "http://127.0.0.1:8001/v1/chat/completions"
DS = "/mnt/c/Users/user/Desktop/ds harness/data/zzz/dataset"
MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"
B, PAD = 16, "\u200c"
CLIP = 170
ORDER = sys.argv[1] if len(sys.argv) > 1 else "forward"
SYS = ("你只能根据下面的资料回答问题。每条资料是一个角色的页面，开头用方括号标出角色名。"
       "如果资料里没有答案，就回答：我不知道。")
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL)


def ntok(s):
    return len(tok(s, add_special_tokens=False)["input_ids"])


probe = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": ""}],
                                tokenize=True, add_generation_prompt=True, return_dict=False)
sid = tok(SYS, add_special_tokens=False)["input_ids"]
OFF = next(i for i in range(len(probe)) if probe[i:i + len(sid)] == sid)

docs = {d["doc_id"]: d for d in (json.loads(l) for l in open(os.path.join(DS, "corpus.jsonl"), encoding="utf-8"))}
IDS = ["st-007", "st-038", "st-103", "st-181", "st-071", "st-026"]
pages = [docs[x] for x in IDS]
if ORDER == "reverse":
    pages = list(reversed(pages))

head = SYS + "\n\n"
body, spans = "", []
for d in pages:
    a = (OFF + ntok(head + body)) // B
    body += "[%s] %s\n" % (d["title"], tok.decode(tok(d["text"], add_special_tokens=False)["input_ids"][:CLIP]))
    body += PAD * ((B - (OFF + ntok(head + body)) % B) % B)
    spans.append({"doc_id": d["doc_id"], "title": d["title"], "blk": [a, (OFF + ntok(head + body)) // B]})
full = head + body

Q = {"st-007": "安比·德玛拉的生日是哪天？", "st-038": "艾莲·乔是什么种族？",
     "st-103": "妮可·德玛拉的生日是哪天？", "st-181": "冯·莱卡恩的生日是哪天？",
     "st-071": "星见雅的身高是多少？", "st-026": "克拉蕾特·弗林特的生日是哪天？"}


def set_deny(rs):
    tmp = VIS + ".tmp"
    json.dump({"n": rs}, open(tmp, "w"))
    os.replace(tmp, VIS)
    time.sleep(0.45)


def post(corpus, user, rid, max_tokens=40):
    b = {"model": "q17b-flex",
         "messages": [{"role": "system", "content": corpus}, {"role": "user", "content": user}],
         "temperature": 0.0, "max_tokens": max_tokens, "stream": True,
         "stream_options": {"include_usage": True},
         "chat_template_kwargs": {"enable_thinking": False}, "request_id": rid}
    req = urllib.request.Request(URL, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    txt = ""
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
            except Exception:
                continue
            ch = d.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content"):
                txt += ch[0]["delta"]["content"]
    return txt.strip().replace("\n", " ")


print("[%s] %d 页，建库顺序：%s" % (ORDER, len(spans), " → ".join(s["title"] for s in spans)))

# 增量建库：第 k 步的序列 = system + 前 k 页，前 k-1 页对当前页不可见
for k in range(1, len(spans) + 1):
    cut = spans[k - 1]["blk"][1] * B
    prefix_ids = tok(full, add_special_tokens=False)["input_ids"][:cut]
    prefix = tok.decode(prefix_ids)
    deny = [s["blk"] for s in spans[:k - 1]]
    set_deny(deny)
    post(prefix, "请回答：好的。", "scope=build%d|%s" % (k, ORDER), 4)
    print("  第 %d 步：序列 %d 块，挡掉前 %d 页 → 建【%s】" % (k, cut // B, k - 1, spans[k - 1]["title"]))
set_deny([])

# 逐页单独提问（只留 system + 该页）
res = {}
for s in spans:
    set_deny([x["blk"] for x in spans if x["doc_id"] != s["doc_id"]])
    res[s["title"]] = post(full, Q[s["doc_id"]], "scope=ask|%s" % s["doc_id"], 40)
    print("   问【%s】：%s" % (s["title"], res[s["title"]]))
set_deny([])

json.dump({"order": ORDER, "seq": [s["title"] for s in spans], "answers": res},
          open("/home/user/_ab_scratch/order2_%s.json" % ORDER, "w"), ensure_ascii=False, indent=1)
