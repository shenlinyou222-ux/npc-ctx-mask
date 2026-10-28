# -*- coding: utf-8 -*-
"""N 槽位扩展：N 段并排，逐段单独解除掩码，用 "only" 协议点名。

每个 trial 对每一段 k 各发一次请求：只解除段 k，问 Qk。
另外跑一次「全遮」对照。
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
KIND = ["pump", "valve", "turbine", "boiler", "compressor", "condenser", "tank"]
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4
MODEL_NAME = sys.argv[3] if len(sys.argv) > 3 else "q4b-flex"
TOK_PATH = sys.argv[4] if len(sys.argv) > 4 else "/mnt/c/Users/user/npc-models-qwen3-4b-awq"


def code():
    return "".join(rnd.choice(SYL) for _ in range(2)).upper() + "-" + str(rnd.randint(1000, 9999))


def post(system, q, rid, max_tokens=40):
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


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_PATH)
    enc = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])  # noqa: E731
    sysids = tok(SYS, add_special_tokens=False)["input_ids"]
    probe = tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": ""}],
        tokenize=True, add_generation_prompt=True, return_dict=False)
    OFF = next(i for i in range(len(probe)) if probe[i:i + len(sysids)] == sysids)

    hit = 0
    leak = 0
    blank_ok = 0
    checks = 0
    for i in range(N):
        ents = ["%s %s" % (KIND[k], code()) for k in range(NS)]
        codes = [code() for _ in range(NS)]
        head = SYS + "\n\n"
        body = ""

        def pos():
            return OFF + enc(head + body)

        spans = []
        for k in range(NS):
            a = (pos() + B - 1) // B
            body += "Segment %d: the access code for the %s is %s.\n" % (k + 1, ents[k], codes[k])
            body += PAD * ((B - pos() % B) % B)
            spans.append((a, pos() // B))
        qs = "\n\nQuestions:\n" + "".join(
            "Q%d: what is the access code for the %s?\n" % (k + 1, ents[k]) for k in range(NS))
        base_ctx = head + body + qs

        # 逐段单独放回
        t_hit = 0
        per_k = []
        for k in range(NS):
            note = ("\n[System note] The sub-agent for Q%d has returned. Answer Q%d only. "
                    "Then reply exactly \"I don't know\" for the others.\nAnswer:" % (k + 1, k + 1))
            deny = [spans[j] for j in range(NS) if j != k]
            json.dump({"s": deny}, open(VIS, "w", encoding="utf-8"))
            time.sleep(0.12)
            ans = post(base_ctx + note, "Go.", "scope=s|%d-%d" % (i, k))
            ok = norm(codes[k]) in norm(ans)
            others = sum(1 for j in range(NS)
                         if j != k and norm(codes[j]) in norm(ans))
            t_hit += int(ok)
            per_k.append(int(ok))
            leak += others
            checks += 1
        hit += t_hit

        # 全遮对照
        json.dump({"s": spans}, open(VIS, "w", encoding="utf-8"))
        time.sleep(0.12)
        note0 = ("\n[System note] No sub-agent has returned yet. Answer Q1 only.\nAnswer:")
        a0 = post(base_ctx + note0, "Go.", "scope=s|%d-blank" % i)
        b_ok = not any(norm(c) in norm(a0) for c in codes)
        blank_ok += int(b_ok)

        print("  %d) N=%d 逐段放回命中 %d/%d 明细=%s | 全遮时无泄漏 %s"
              % (i + 1, NS, t_hit, NS, per_k, b_ok), flush=True)

    print("\nN=%d  n=%d" % (NS, N))
    print("  逐段放回：命中 %d/%d" % (hit, checks))
    print("  跨段泄漏：%d 次（应为 0）" % leak)
    print("  全遮对照：无泄漏 %d/%d" % (blank_ok, N))
    return 0


if __name__ == "__main__":
    sys.exit(main())
