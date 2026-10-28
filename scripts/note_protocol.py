# -*- coding: utf-8 -*-
"""返回 prefill 里带上协议句：只答点名那题，其余答不知道。

通知：
  [System note] The sub-agent for Q2 has returned. Answer Q2 only.
  For Q1 and Q3, reply exactly: I don't know.

检查：
  Q2 命中      被点名那题答对
  污染         其余题被填上段 2 的编码
  正确弃答     其余题答 "I don't know"
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

NOTE = ("\n[System note] The sub-agent for Q2 has returned. Answer Q2 only. "
        "For Q1 and Q3, reply exactly: I don't know.\nAnswer:")


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

    stat = {"all": [0, 0, 0], "only2": [0, 0, 0]}
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
            body += "Segment %d: the access code for the %s is %s.\n" % (k + 1, ents[k], codes[k])
            body += PAD * ((B - pos() % B) % B)
            spans.append((a, pos() // B))
        qs = "\n\nQuestions:\n" + "".join(
            "Q%d: what is the access code for the %s?\n" % (k + 1, ents[k]) for k in range(3))
        ctx = head + body + qs + NOTE

        for arm, deny in (("all", []), ("only2", [spans[0], spans[2]])):
            json.dump({"s": deny}, open(VIS, "w", encoding="utf-8"))
            time.sleep(0.15)
            ans = post(ctx, "Go.", "scope=s|%d-%s" % (i, arm))
            parts = split_answers(ans)
            q2 = parts.get(2, ans)
            hit = norm(codes[1]) in norm(q2)
            pollute = [k for k in (1, 3)
                       if k in parts and norm(codes[1]) in norm(parts[k])]
            abstain = [k for k in (1, 3)
                       if k in parts and "idontknow" in norm(parts[k])]
            stat[arm][0] += int(hit)
            stat[arm][1] += len(pollute)
            stat[arm][2] += len(abstain)
            print("  %d) %-5s Q2=%r 命中=%s 污染=%s 弃答=%s"
                  % (i + 1, arm, q2[:24], "✔" if hit else "✘", pollute, abstain),
                  flush=True)
        total += 1

    print("\nn=%d" % total)
    for arm in ("all", "only2"):
        print("  %-5s Q2命中 %d/%d | 其余题被污染 %d 次 | 其余题正确弃答 %d 次"
              % (arm, stat[arm][0], total, stat[arm][1], stat[arm][2]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
