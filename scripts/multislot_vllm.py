# -*- coding: utf-8 -*-
"""多槽位 + 分段放回（vLLM 掩码路径）。

上下文里并排放三段"槽位"（各含一个编码），用掩码控制哪几段可见：
  all   一个都不遮          -> 三个问题都应答对
  onlyB 遮 A 和 C           -> 只有 B 答对
  none  三个都遮            -> 三个都应答不出

这是"一次只放回一部分"的直接检验，且复用已验证 8/8 的掩码路径。
"""
import json
import random
import re
import sys
import time
import urllib.request

VIS = "/home/user/_ab_scratch/npc_vis_blocks.json"
URL = "http://127.0.0.1:8001/v1/chat/completions"
B, PAD = 16, "\u200c"
SYS = ("You answer using only the provided records. Reply with the shortest exact answer. "
       "If the records do not contain the answer, reply exactly: I don't know.")
rnd = random.Random(20261028)
SYL = ["Kar", "vel", "Mor", "tha", "Zir", "dan", "Osk", "rya", "Nym", "bra"]
KIND = ["pump", "valve", "turbine"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def code():
    return "".join(rnd.choice(SYL) for _ in range(2)).upper() + "-" + str(rnd.randint(1000, 9999))


def post(system, q, rid, max_tokens=20):
    body = {"model": "q17b-flex",
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": q}],
            "temperature": 0.0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}, "request_id": rid}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"].strip()


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/mnt/c/Users/user/npc-models-qwen3-1.7b")
    enc = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])  # noqa: E731
    sysids = tok(SYS, add_special_tokens=False)["input_ids"]
    probe = tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": ""}],
        tokenize=True, add_generation_prompt=True, return_dict=False)
    OFF = next(i for i in range(len(probe)) if probe[i:i + len(sysids)] == sysids)
    print("template off = %d" % OFF)

    stat = {"all": 0, "onlyB": 0, "none": 0}
    total = 0
    for i in range(N):
        ents = ["%s %s" % (KIND[k], code()) for k in range(3)]
        codes = [code() for _ in range(3)]
        head = SYS + "\n\n"
        body = ""

        def pos():
            return OFF + enc(head + body)

        spans = []
        for k in range(3):
            a = (pos() + B - 1) // B
            body += "Record %d: the access code for the %s is %s.\n" % (k + 1, ents[k], codes[k])
            body += PAD * ((B - pos() % B) % B)
            spans.append((a, pos() // B))
        ctx = head + body

        def ask(tag, deny):
            json.dump({"s": deny}, open(VIS, "w", encoding="utf-8"))
            time.sleep(0.15)
            hits = []
            for k in range(3):
                q = "What is the access code for the %s?" % ents[k]
                ans = post(ctx, q, "scope=s|%s-%d-%d" % (tag, i, k))
                hits.append(norm(codes[k]) in norm(ans))
            return hits

        all_h = ask("all", [])
        onlyb_h = ask("onlyB", [spans[0], spans[2]])
        none_h = ask("none", spans)

        total += 1
        for key, h in (("all", all_h), ("onlyB", onlyb_h), ("none", none_h)):
            stat[key] += sum(h)
        print("  %d) codes=%s spans=%s" % (i + 1, codes, spans))
        print("     all   %s" % all_h)
        print("     onlyB %s   （A/C 被遮）" % onlyb_h)
        print("     none  %s" % none_h, flush=True)

    print("\nn=%d（每种条件 %d 个问题）" % (total, 3 * total))
    for k in ("all", "onlyB", "none"):
        print("  %-6s 命中 %d/%d" % (k, stat[k], 3 * total))
    return 0


if __name__ == "__main__":
    sys.exit(main())
