#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TTFT 四点测量：可见性变化 vs 文本变化，谁会让前缀缓存失效。

每轮用「段落轮转」构造一份新的全量序列（保证冷启动），然后：
  A_cold   全塞，首次（冷 prefill）
  A_warm   全塞，重复（前缀命中）
  M_warm   同样 token 序列，只换可见性标签（遮中段 15% 的块）
  D_cold   删掉中段那几段文本（token 序列在中途分叉，必须重 prefill）
  D_warm   重复删除版

用法: _ttft_probe.py [reps]
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
SYS = ("You answer using only the provided records. Reply with the shortest exact answer. "
       "If the records do not contain the answer, reply exactly: I don't know.")
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL)


def ntok(s):
    return len(tok(s, add_special_tokens=False)["input_ids"])


def post(system, q, rid=None, max_tokens=16):
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


def tmpl_offset(sys_text):
    probe = tok.apply_chat_template(
        [{"role": "system", "content": sys_text}, {"role": "user", "content": ""}],
        tokenize=True, add_generation_prompt=True, return_dict=False)
    sysids = tok(sys_text, add_special_tokens=False)["input_ids"]
    return next(i for i in range(len(probe)) if probe[i:i + len(sysids)] == sysids)


def pad(s):
    need = (B - (tmpl_offset(SYS) + ntok(s)) % B) % B
    return s + PAD * need


def set_deny(ranges):
    json.dump({"m": ranges}, open(VIS, "w", encoding="utf-8"))
    time.sleep(0.05)


def main():
    docs = [json.loads(l) for l in open(os.path.join(DS, "corpus.jsonl"), encoding="utf-8")]
    qs = [json.loads(l) for l in open(os.path.join(DS, "queries_base.jsonl"), encoding="utf-8")]
    q = next(x for x in qs if x["fact_key"] == "Height")
    src = next(d for d in docs if d["doc_id"] == q["source_doc"])
    head = src["text"][:1200]

    parts, total = [], 0
    for d in docs:
        if d["doc_id"] == q["source_doc"]:
            continue
        parts.append(d["text"][:1200])
        total += ntok(parts[-1])
        if total > 6000:
            break
    k = max(1, len(parts) // 5)

    off = tmpl_offset(SYS)
    for rep in range(REPS):
        rot = parts[rep:] + parts[:rep]                      # 轮转：保证本轮是新序列
        full_txt = pad(head + "\n" + "\n".join(rot))
        deleted = pad(head + "\n" + "\n".join(rot[: len(rot) // 2] + rot[len(rot) // 2 + k:]))
        pre = off + ntok(head) + ntok("\n" + "\n".join(rot[: len(rot) // 2]))
        start_blk = pre // B
        end_blk = (pre + ntok("\n" + "\n".join(rot[len(rot) // 2: len(rot) // 2 + k]))) // B
        nblk = (off + ntok(full_txt)) // B
        print(f"--- rep {rep}: full={ntok(full_txt)}tok/{nblk}blk  del={ntok(deleted)}tok  "
              f"deny=[{start_blk},{end_blk}) = {100*(end_blk-start_blk)/nblk:.0f}% 块")

        set_deny([])
        for name, txt, rid, deny in [
            ("A_cold", full_txt, None, []),
            ("A_warm", full_txt, None, []),
            ("M_warm", full_txt, "scope=m", [[start_blk, end_blk]]),
            ("D_cold", deleted, None, []),
            ("D_warm", deleted, None, []),
        ]:
            set_deny(deny)
            ans, pt, tt = post(txt, q["question"], rid=rid)
            print(f"  {name:7s} pt={pt:5d}  ttft={tt:6.3f}s  ans={ans[:44]!r}")
    set_deny([])


if __name__ == "__main__":
    main()
