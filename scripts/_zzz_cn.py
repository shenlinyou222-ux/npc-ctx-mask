# -*- coding: utf-8 -*-
"""用绝区零真实角色页，中文提问，跑 不屏蔽 / 屏蔽 两组，直接打印输入输出。"""
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

# 真实绝区零角色页（英文 wiki 原文）
WANT = ["st-007",   # Anby Demara 安比·德玛拉
        "st-038",   # Ellen Joe 艾莲·乔
        "st-071",   # Hoshimi Miyabi 星见雅
        "st-011",   # Asaba Harumasa 浅羽悠真
        "st-026",   # Claret Flint 克拉蕾特·弗林特
        "st-103",   # Nicole Demara 妮可·德玛拉
        "st-181",   # Von Lycaon 冯·莱卡恩
        "st-191",   # Zhu Yuan 朱鸢
        "st-169",   # Trigger 扳机
        "st-021"]   # Caesar King 凯撒·金
pages = [docs[x] for x in WANT]
pages.sort(key=lambda d: d["title"])

head = SYS + "\n\n"
body, spans = "", []
for d in pages:
    a = (OFF + ntok(head + body)) // B
    body += "[%s] %s\n" % (d["title"], tok.decode(tok(d["text"], add_special_tokens=False)["input_ids"][:CLIP]))
    body += PAD * ((B - (OFF + ntok(head + body)) % B) % B)
    spans.append({"doc_id": d["doc_id"], "title": d["title"], "version": d.get("version") or "",
                  "blk": [a, (OFF + ntok(head + body)) // B]})
corpus = head + body
print("语料 %d token / %d 块 / %d 页" % (ntok(corpus), (OFF + ntok(corpus)) // B, len(pages)))
for s in spans:
    print("   %-6s v%-4s %s" % (s["doc_id"], s["version"], s["title"]))


def V(v):
    s = str(v)
    return int(s.split(".")[0]) * 100 + int(s.split(".")[1]) if "." in s else -1


def post(system, user, rid=None, max_tokens=48):
    b = {"model": "q17b-flex",
         "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
         "temperature": 0.0, "max_tokens": max_tokens, "stream": True,
         "stream_options": {"include_usage": True},
         "chat_template_kwargs": {"enable_thinking": False}}
    if rid:
        b["request_id"] = rid
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
    return txt.strip()


def set_deny(rs):
    tmp = VIS + ".tmp"
    json.dump({"q": rs}, open(tmp, "w"))
    os.replace(tmp, VIS)
    time.sleep(0.4)


QS = [
    ("安比·德玛拉的生日是哪天？", "st-007"),
    ("艾莲·乔的生日是哪天？", "st-038"),
    ("星见雅的身高是多少？", "st-071"),
    ("浅羽悠真的生日是哪天？", "st-011"),
    ("克拉蕾特·弗林特的生日是哪天？", "st-026"),
    ("妮可·德玛拉的生日是哪天？", "st-103"),
]

# 提问者：妮可（剧情进度 1.0）——1.0 之后的角色页不该知道
CUT = "1.0"
deny = [[s["blk"][0], s["blk"][1]] for s in spans if V(s["version"]) > V(CUT)]
print("\n提问者：妮可·德玛拉（进度 1.0）；被挡的页：",
      [s["title"] for s in spans if V(s["version"]) > V(CUT)])

set_deny([])
print("[预热]", post(corpus, "资料已读完，请回答：好的。", "scope=all|w", 8))

for q, src in QS:
    user = "（妮可·德玛拉的提问）%s\n请用一句中文回答。" % q
    set_deny([])
    a = post(corpus, user, "scope=all|q", 40)
    set_deny(deny)
    m = post(corpus, user, "scope=q|q", 40)
    print("\n问：%s" % q)
    print("  不屏蔽：%s" % a)
    print("  屏蔽　：%s" % m)
set_deny([])
