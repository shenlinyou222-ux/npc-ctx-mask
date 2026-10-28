# -*- coding: utf-8 -*-
"""返回通知点名问题：三个问题先摆出，通知说「先答 Qk」，看它答哪个。

上下文尾部严格是：
    Questions:
    Q1: ... Q2: ... Q3: ...
    [System note] Q{k} has been returned. Answer Q{k} first.
    Answer:
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
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4


def code():
    return "".join(rnd.choice(SYL) for _ in range(2)).upper() + "-" + str(rnd.randint(1000, 9999))


def post(system, q, rid, max_tokens=24):
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

    stat = {"order312": 0, "order123": 0, "n": 0}
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
            body += "Segment %d: the access code for the %s is %s.\n" % (k + 1, ents[k], codes[k])
            body += PAD * ((B - pos() % B) % B)
            spans.append((a, pos() // B))
        qs = "\n\nQuestions:\n" + "".join(
            "Q%d: what is the access code for the %s?\n" % (k + 1, ents[k]) for k in range(3))

        def turn(show, ask_q):
            ctx = head + body + qs + (
                "\n[System note] A sub-agent has returned for Q%d. Answer Q%d first.\nAnswer:"
                % (ask_q, ask_q))
            deny = [spans[k] for k in range(3) if k != show]
            json.dump({"s": deny}, open(VIS, "w", encoding="utf-8"))
            time.sleep(0.15)
            return post(ctx, "Go.", "scope=s|%d-%d-%d" % (i, show, ask_q))

        # 返回顺序 3 -> 1 -> 2（展示段与点名的问题一致）
        ok = []
        for show, ask_q in ((2, 3), (0, 1), (1, 2)):
            ans = turn(show, ask_q)
            hit = norm(codes[show]) in norm(ans)
            ok.append(hit)
            print("  %d) show=段%d ask=Q%d -> %s  %r"
                  % (i + 1, show + 1, ask_q, "✔" if hit else "✘", ans[:42]), flush=True)
        # 顺序 1 -> 2 -> 3 作为对照
        ok2 = []
        for show, ask_q in ((0, 1), (1, 2), (2, 3)):
            ans = turn(show, ask_q)
            ok2.append(norm(codes[show]) in norm(ans))
        stat["order312"] += sum(ok)
        stat["order123"] += sum(ok2)
        stat["n"] += 3

    print("\nn=%d 轮次=%d" % (N, stat["n"]))
    print("  返回顺序 3->1->2 命中 %d/%d" % (stat["order312"], stat["n"]))
    print("  返回顺序 1->2->3 命中 %d/%d" % (stat["order123"], stat["n"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
