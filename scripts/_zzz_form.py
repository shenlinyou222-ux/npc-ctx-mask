# -*- coding: utf-8 -*-
"""测试两种问法哪种能让模型准确读出资料里的字段（不屏蔽）。"""
import json
import os
import time
import urllib.request

VIS = "/home/user/_ab_scratch/npc_vis_blocks.json"
URL = "http://127.0.0.1:8001/v1/chat/completions"
DS = "/mnt/c/Users/user/Desktop/ds harness/data/zzz/dataset"
MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"
B, PAD = 16, "\u200c"
CLIP = 170
SYS = ("你只能根据下面的资料回答问题。每条资料是一个角色的页面，用方括号标出名字。"
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
WANT = ["st-007", "st-038", "st-071", "st-011", "st-026", "st-103", "st-181", "st-191", "st-169", "st-021"]
pages = [docs[x] for x in WANT]
pages.sort(key=lambda d: d["title"])
head = SYS + "\n\n"
body = ""
for d in pages:
    body += "[%s] %s\n" % (d["title"], tok.decode(tok(d["text"], add_special_tokens=False)["input_ids"][:CLIP]))
    body += PAD * ((B - (OFF + ntok(head + body)) % B) % B)
corpus = head + body


def post(user, max_tokens=40):
    b = {"model": "q17b-flex",
         "messages": [{"role": "system", "content": corpus}, {"role": "user", "content": user}],
         "temperature": 0.0, "max_tokens": max_tokens, "stream": True,
         "stream_options": {"include_usage": True},
         "chat_template_kwargs": {"enable_thinking": False},
         "request_id": "scope=all|t%d" % time.time_ns()}
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


json.dump({}, open(VIS, "w"))
time.sleep(0.3)

CASES = [
    ("Anby Demara", "Birthday", "11/11"),
    ("Ellen Joe", "Birthday", "1/4"),
    ("Hoshimi Miyabi", "Height", "165"),
    ("Asaba Harumasa", "Birthday", "?"),
    ("Claret Flint", "Birthday", "7/23"),
    ("Nicole Demara", "Birthday", "11/11"),
]
for name, field, gold in CASES:
    a = post("资料【%s】里的 %s 是什么？只回答值。" % (name, field))
    b = post("【%s】的%s是多少？" % (name, "生日" if field == "Birthday" else "身高"))
    print("%-16s %-8s 期望~%-6s | 甲: %-30s | 乙: %s" % (name, field, gold, a[:30], b[:30]))
