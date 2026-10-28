# -*- coding: utf-8 -*-
"""为「前后对比」补一个数：同一份 3.4K 语料，冷 prefill vs 热命中。

冷：在 system 开头插入唯一 nonce，破坏整段前缀（等价于"每次换角色都重拼 prompt"）。
热：不加 nonce，直接问（等价于"序列不变，只换可见性位图"）。
"""
import json
import os
import statistics as st
import time
import urllib.request

VIS = "/home/user/_ab_scratch/npc_vis_blocks.json"
URL = "http://127.0.0.1:8001/v1/chat/completions"
DS = "/mnt/c/Users/user/Desktop/ds harness/data/zzz/dataset"
MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"
B, PAD = 16, "\u200c"
CLIP = 150
SYS = ("You answer questions about the world using only the records below. "
       "Each record is one character page, bracketed by its title. "
       "If the records do not contain the answer, say exactly: I don't know.")
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL)


def ntok(s):
    return len(tok(s, add_special_tokens=False)["input_ids"])


probe = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": ""}],
                                tokenize=True, add_generation_prompt=True, return_dict=False)
sid = tok(SYS, add_special_tokens=False)["input_ids"]
OFF = next(i for i in range(len(probe)) if probe[i:i + len(sid)] == sid)

docs = {d["doc_id"]: d for d in (json.loads(l) for l in open(os.path.join(DS, "corpus.jsonl"), encoding="utf-8"))}
ids = ["st-026", "st-135", "st-143", "st-117", "st-015", "st-010", "st-122"]
fillers = [d for d in docs.values() if d["tenant"] == "agent" and d["doc_id"] not in ids]
fillers.sort(key=lambda d: (-(int(d["version"].split(".")[0]) * 100 + int(d["version"].split(".")[1])
                             if "." in str(d["version"]) else -1), d["title"]))
pages = [docs[x] for x in ids] + fillers[:14]
pages.sort(key=lambda d: d["title"])

head = SYS + "\n\n"
body = ""
for d in pages:
    body += "[%s] %s\n" % (d["title"], tok.decode(tok(d["text"], add_special_tokens=False)["input_ids"][:CLIP]))
    body += PAD * ((B - (OFF + ntok(head + body)) % B) % B)
corpus = head + body
Q = "【Banyue 的提问】When is Banyue's birthday?\n（简短回答。）"
print("语料 %d token / %d 块" % (ntok(corpus), (OFF + ntok(corpus)) // B))


def ask(system, rid, max_tokens=24):
    b = {"model": "q17b-flex",
         "messages": [{"role": "system", "content": system}, {"role": "user", "content": Q}],
         "temperature": 0.0, "max_tokens": max_tokens, "stream": True,
         "stream_options": {"include_usage": True},
         "chat_template_kwargs": {"enable_thinking": False}, "request_id": rid}
    req = urllib.request.Request(URL, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, txt = None, ""
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
                if ttft is None:
                    ttft = time.perf_counter() - t0
                txt += ch[0]["delta"]["content"]
    return (ttft or 0.0), time.perf_counter() - t0, txt.strip()[:40]


json.dump({}, open(VIS, "w"))
time.sleep(0.3)
print("[预热]", ask(corpus, "scope=all|w", 4))

cold, warm = [], []
for i in range(3):
    c = "nonce-%d-%d. " % (int(time.time()), i) + corpus     # 冷：破坏前缀
    cold.append(ask(c, "scope=all|c%d" % i))
    warm.append(ask(corpus, "scope=all|h%d" % i))
    time.sleep(0.1)

print("\n冷 prefill（每次换角色都重拼 prompt）")
for t in cold:
    print("   ttft=%.3f ttot=%.3f  %s" % (t[0], t[1], t[2]))
print("热命中（序列不变，只换位图）")
for t in warm:
    print("   ttft=%.3f ttot=%.3f  %s" % (t[0], t[1], t[2]))
print("\n中位数：冷 %.3f s / 热 %.3f s / %.1f×" %
      (st.median(x[0] for x in cold), st.median(x[0] for x in warm),
       st.median(x[0] for x in cold) / st.median(x[0] for x in warm)))
