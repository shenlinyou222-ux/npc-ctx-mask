# -*- coding: utf-8 -*-
"""两种协议句写法，找能让 Q2 保持正确、同时不乱填的那一种。

只测 only2 臂（只留段 2）：
  A 先答后禁：「Answer Q2 first. Then reply exactly I don't know for Q1 and Q3.」
  B 先禁后答：「Reply exactly I don't know for Q1 and Q3. Then answer Q2.」
  C 不写协议句（基线，会乱填）
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

NOTES = {
    "A": ("\n[System note] The sub-agent for Q2 has returned. Answer Q2 first. "
          "Then reply exactly \"I don't know\" for Q1 and Q3.\nAnswer:"),
    "B": ("\n[System note] The sub-agent for Q2 has returned. For Q1 and Q3 reply exactly "
          "\"I don't know\". Then answer Q2.\nAnswer:"),
    "C": "\n[System note] The sub-agent for Q2 has returned. Answer Q2 first.\nAnswer:",
}


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

    stat = {k: [0, 0, 0] for k in NOTES}
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

        deny = [spans[0], spans[2]]
        json.dump({"s": deny}, open(VIS, "w", encoding="utf-8"))
        time.sleep(0.15)
        for tag, note in NOTES.items():
            ans = post(head + body + qs + note, "Go.", "scope=s|%d-%s" % (i, tag))
            parts = split_answers(ans)
            q2 = parts.get(2, ans)
            hit = norm(codes[1]) in norm(q2)
            pollute = sum(1 for k in (1, 3)
                          if k in parts and norm(codes[1]) in norm(parts[k]))
            abstain = sum(1 for k in (1, 3)
                          if k in parts and "idontknow" in norm(parts[k]))
            stat[tag][0] += int(hit)
            stat[tag][1] += pollute
            stat[tag][2] += abstain
            print("  %d) %s Q2=%r 命中=%s 污染=%d 弃答=%d"
                  % (i + 1, tag, q2[:22], "✔" if hit else "✘", pollute, abstain), flush=True)
        total += 1

    print("\nn=%d（只留段 2，只问 Q2）" % total)
    for tag in NOTES:
        print("  %s  Q2命中 %d/%d | 污染 %d | 正确弃答 %d"
              % (tag, stat[tag][0], total, stat[tag][1], stat[tag][2]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
