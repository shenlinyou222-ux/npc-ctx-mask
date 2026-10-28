# -*- coding: utf-8 -*-
"""小语料 + 中文问法：找能让模型稳定答对的问法。

语料固定 6 页（真实绝区零角色），每题只问其中一页。
"""
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
IDS = ["st-007", "st-038", "st-103", "st-071", "st-181", "st-026"]
pages = [docs[x] for x in IDS]
pages.sort(key=lambda d: d["title"])
head = SYS + "\n\n"
body = ""
for d in pages:
    body += "[%s] %s\n" % (d["title"], tok.decode(tok(d["text"], add_special_tokens=False)["input_ids"][:CLIP]))
    body += PAD * ((B - (OFF + ntok(head + body)) % B) % B)
corpus = head + body
print("语料 %d token / %d 页" % (ntok(corpus), len(pages)))


def post(user, max_tokens=32):
    b = {"model": "q17b-flex",
         "messages": [{"role": "system", "content": corpus}, {"role": "user", "content": user}],
         "temperature": 0.0, "max_tokens": max_tokens, "stream": True,
         "stream_options": {"include_usage": True},
         "chat_template_kwargs": {"enable_thinking": False},
         "request_id": "scope=all|x%d" % time.time_ns()}
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
post("资料读完请回答：好的。", 8)

CASES = [
    ("Anby Demara", "生日", "February 20th", "安比·德玛拉的生日是哪天？"),
    ("Anby Demara", "生日", "February 20th", "资料【Anby Demara】里写的生日是哪天？"),
    ("Ellen Joe", "种族", "Thiren", "艾莲·乔是什么种族？"),
    ("Hoshimi Miyabi", "身高", "170", "星见雅有多高？"),
    ("Hoshimi Miyabi", "身高", "170", "资料【Hoshimi Miyabi】里写的身高是多少？"),
    ("Claret Flint", "生日", "July 23rd", "克拉蕾特·弗林特的生日是哪天？"),
    ("Von Lycaon", "生日", "October 4th", "冯·莱卡恩的生日是哪天？"),
]
for name, field, gold, q in CASES:
    outs = [post(q) for _ in range(2)]
    ok = [gold.lower().split()[0] in o.lower() for o in outs]
    print("%-16s %-4s 期望~%-14s %s | %s || %s" % (name, field, gold, "OK " if all(ok) else "x  ", outs[0][:38], outs[1][:38]))
