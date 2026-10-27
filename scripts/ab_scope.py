# -*- coding: utf-8 -*-
"""同文件双 scope 对照：a 不做限制，b 遮住 B 块。"""
import json
import re
import urllib.request

VIS = "/home/user/_ab_scratch/npc_vis_blocks.json"
URL = "http://127.0.0.1:8001/v1/chat/completions"
B, PAD = 16, "\u200c"
SYS = ("You answer using only the provided records. Reply with the shortest exact answer. "
       "If the records do not contain the answer, reply exactly: I don't know.")
FILL = ("Routine maintenance logs indicate steady output and no anomalies were reported "
        "by the night shift. ")


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


from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("/mnt/c/Users/user/npc-models-qwen3-1.7b")
ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])  # noqa: E731

vis_code, msk_code = "VIS-3355", "MSK-9902"
head = SYS + "\n\n"
body, spans = "", []
a = -(-ntok(head + body) // B)
body += "Record A: the access code is %s.\n" % vis_code
body += PAD * ((B - ntok(head + body) % B) % B)
spans.append(("VIS", (a, ntok(head + body) // B)))
for _ in range(2):
    a = -(-ntok(head + body) // B)
    body += "Log: %s\n" % FILL
    body += PAD * ((B - ntok(head + body) % B) % B)
    spans.append(("FILL", (a, ntok(head + body) // B)))
a = -(-ntok(head + body) // B)
body += "Record B: the access code is %s.\n" % msk_code
body += PAD * ((B - ntok(head + body) % B) % B)
spans.append(("MSK", (a, ntok(head + body) // B)))
ctx = head + body
msk_span = [sp for t, sp in spans if t == "MSK"]
print("块: %s  遮: %s" % (spans, msk_span))

json.dump({"open": [], "deny": msk_span}, open(VIS, "w", encoding="utf-8"))

for sc in ("open", "deny", "open"):
    ans = post(ctx, "What is the access code in Record B?", "scope=%s|x" % sc)
    print("  scope=%-5s B -> %-30r 读到B=%s" % (sc, ans[:30], norm(msk_code) in norm(ans)))
