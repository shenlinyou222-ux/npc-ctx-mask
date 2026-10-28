# -*- coding: utf-8 -*-
"""4B 的多实体弱点定位：是「一次摆三题」还是「多实体本身」。

三臂（全部可见，不遮任何段）：
  one_q    上下文里只放一个问题，单问单答
  three_q  三个问题一次摆出，要求三行作答
  seq_q    三个问题一次摆出，但只回答其中一个（逐题各发一次请求）

判据：被问的题答对；three_q 看三题各自答对。
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
MODEL_NAME = sys.argv[2] if len(sys.argv) > 2 else "q4b-flex"
TOK_PATH = sys.argv[3] if len(sys.argv) > 3 else "/mnt/c/Users/user/npc-models-qwen3-4b-awq"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def code():
    return "".join(rnd.choice(SYL) for _ in range(2)).upper() + "-" + str(rnd.randint(1000, 9999))


def post(system, q, rid, max_tokens=48):
    body = {"model": MODEL_NAME,
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


def split_answers(text):
    parts = re.split(r"Q\s*([123])\s*[:：]", text)
    out = {}
    if len(parts) > 2:
        for i in range(1, len(parts) - 1, 2):
            out[int(parts[i])] = parts[i + 1].strip()
    return out


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_PATH)
    enc = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])  # noqa: E731
    sysids = tok(SYS, add_special_tokens=False)["input_ids"]
    probe = tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": ""}],
        tokenize=True, add_generation_prompt=True, return_dict=False)
    OFF = next(i for i in range(len(probe)) if probe[i:i + len(sysids)] == sysids)

    stat = {"one_q": 0, "three_q": 0, "seq_q": 0}
    slots = {"one_q": 0, "three_q": 0, "seq_q": 0}
    json.dump({"s": []}, open(VIS, "w", encoding="utf-8"))
    time.sleep(0.15)

    for i in range(N):
        ents = ["%s %s" % (KIND[k], code()) for k in range(3)]
        codes = [code() for _ in range(3)]
        head = SYS + "\n\n"
        body = ""

        def pos():
            return OFF + enc(head + body)

        for k in range(3):
            body += "Segment %d: the access code for the %s is %s.\n" % (k + 1, ents[k], codes[k])
            body += PAD * ((B - pos() % B) % B)
        base_ctx = head + body

        # one_q：每次只放一个问题
        one_ok = 0
        for k in range(3):
            ctx = base_ctx + ("\nQuestion: what is the access code for the %s?\nAnswer:" % ents[k])
            ans = post(ctx, "Go.", "scope=s|%d-one-%d" % (i, k))
            one_ok += int(norm(codes[k]) in norm(ans))
        stat["one_q"] += one_ok
        slots["one_q"] += 3

        # three_q：三题一起，要求三行
        qs = "\n\nQuestions:\n" + "".join(
            "Q%d: what is the access code for the %s?\n" % (k + 1, ents[k]) for k in range(3))
        ans = post(base_ctx + qs + "\nAnswer each as Q1: / Q2: / Q3:\nAnswer:",
                   "Go.", "scope=s|%d-three" % i, max_tokens=64)
        parts = split_answers(ans)
        three_ok = sum(1 for k in range(3) if k + 1 in parts and norm(codes[k]) in norm(parts[k + 1]))
        stat["three_q"] += three_ok
        slots["three_q"] += 3

        # seq_q：三题一起摆，但只答被点名那一题
        seq_ok = 0
        for k in range(3):
            note = ("\n[System note] Answer Q%d only. Then reply exactly \"I don't know\" "
                    "for the others.\nAnswer:" % (k + 1))
            ans = post(base_ctx + qs + note, "Go.", "scope=s|%d-seq-%d" % (i, k))
            parts = split_answers(ans)
            a = parts.get(k + 1, ans)
            seq_ok += int(norm(codes[k]) in norm(a))
        stat["seq_q"] += seq_ok
        slots["seq_q"] += 3

        print("  %d) codes=%s one=%d three=%d seq=%d"
              % (i + 1, codes, one_ok, three_ok, seq_ok), flush=True)

    print("\nn=%d（每臂 %d 题）" % (N, slots["one_q"]))
    for k in ("one_q", "three_q", "seq_q"):
        print("  %-8s %d/%d" % (k, stat[k], slots[k]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
