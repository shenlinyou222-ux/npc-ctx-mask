# -*- coding: utf-8 -*-
"""端到端：①增量建库（每页 prefill 时挡住前面的页，只能看到 system）
          ②按人物挡掉不该看的页，问 4 个问题，记录不挡/挡 两次输出。
用于生成展示文档，输出写到 /home/user/_ab_scratch/io_zzz.json。
"""
import json
import os
import time
import urllib.request

VIS = "/home/user/_ab_scratch/npc_vis_blocks.json"
URL = "http://127.0.0.1:8001/v1/chat/completions"
DS = "/mnt/c/Users/user/Desktop/ds harness/data/zzz/dataset"
MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"
OUT = "/home/user/_ab_scratch/io_zzz.json"
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
IDS = ["st-007", "st-038", "st-103", "st-181", "st-071", "st-026"]
pages = [docs[x] for x in IDS]

head = SYS + "\n\n"
body, spans = "", []
for d in pages:
    a = (OFF + ntok(head + body)) // B
    body += "[%s] %s\n" % (d["title"], tok.decode(tok(d["text"], add_special_tokens=False)["input_ids"][:CLIP]))
    body += PAD * ((B - (OFF + ntok(head + body)) % B) % B)
    spans.append({"doc_id": d["doc_id"], "title": d["title"], "version": d.get("version") or "",
                  "blk": [a, (OFF + ntok(head + body)) // B]})
full = head + body
full_ids = tok(full, add_special_tokens=False)["input_ids"]


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


print("=== 第 1 步：逐页建库（每页只能看到 system）===")
for k in range(1, len(spans) + 1):
    cut = spans[k - 1]["blk"][1] * B
    set_deny([s["blk"] for s in spans[:k - 1]])
    post(tok.decode(full_ids[:cut]), "请回答：好的。", "scope=build%d|f" % k, 4)
    print("  建【%s】时，挡掉前 %d 页" % (spans[k - 1]["title"], k - 1))
set_deny([])

# 妮可看不到：星见雅(1.4)、克拉蕾特(3.2)
def V(v):
    s = str(v)
    return int(s.split(".")[0]) * 100 + int(s.split(".")[1]) if "." in s else -1


deny = [s["blk"] for s in spans if V(s["version"]) > V("1.0")]
blocked = [s["title"] for s in spans if V(s["version"]) > V("1.0")]
print("\n=== 第 2 步：按人物挡 ===\n妮可看不到：%s" % blocked)

QS = [
    ("安比·德玛拉的生日是哪天？", "2月20日", True),
    ("艾莲·乔是什么种族？", "Thiren（鲨鱼）", True),
    ("星见雅有多高？", "170 cm", False),
    ("克拉蕾特·弗林特的生日是哪天？", "7月23日", False),
]
rows = []
for q, gold, visible in QS:
    set_deny([])
    a = post(full, q, "scope=all|a", 40)
    set_deny(deny)
    m = post(full, q, "scope=n|m", 40)
    rows.append({"q": q, "gold": gold, "visible": visible, "a": a, "m": m})
    print("\n问：%s（正解 %s；%s）" % (q, gold, "该知道" if visible else "不该知道"))
    print("  不挡：%s" % a)
    print("  挡　：%s" % m)
set_deny([])

json.dump({"pages": [s["title"] for s in spans], "blocked": blocked, "rows": rows},
          open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\n写入", OUT)
