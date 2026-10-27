# -*- coding: utf-8 -*-
"""验证：不重启服务，改 mask JSON 是否立即生效（mtime 热重载）。

同一 scope 名，分两步：
  1) deny B  -> 问 B 读不到，问 A 读得到
  2) deny [] -> 问 B 立刻读得到（证明热切换）
"""
import json
import re
import sys
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
            "chat_template_kwargs": {"enable_thinking": False},
            "request_id": rid}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read().decode())
    return d["choices"][0]["message"]["content"].strip()


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/mnt/c/Users/user/npc-models-qwen3-1.7b")
    ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])  # noqa: E731

    vis_code, msk_code = "VIS-7391", "MSK-4820"
    head = SYS + "\n\n"
    body, spans = "", []
    a = -(-ntok(head + body) // B)
    body += "Record A: the access code is %s.\n" % vis_code
    body += PAD * ((B - ntok(head + body) % B) % B)
    spans.append(("VIS", (a, ntok(head + body) // B)))
    for _ in range(3):
        a = -(-ntok(head + body) // B)
        body += "Log: %s\n" % FILL
        body += PAD * ((B - ntok(head + body) % B) % B)
        spans.append(("FILL", (a, ntok(head + body) // B)))
    a = -(-ntok(head + body) // B)
    body += "Record B: the access code is %s.\n" % msk_code
    body += PAD * ((B - ntok(head + body) % B) % B)
    spans.append(("MSK", (a, ntok(head + body) // B)))
    ctx = head + body
    print("块区间: %s" % spans)

    def write(deny):
        tmp = VIS + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"live": deny}, f)
        import os
        os.replace(tmp, VIS)

    def probe(tag):
        deny = [sp for t, sp in spans if t in ("MSK", "FILL")]
        return (post(ctx, "What is the access code in Record B?",
                     "scope=live|%s-b" % tag),
                post(ctx, "What is the access code in Record A?",
                     "scope=live|%s-a" % tag))

    write([sp for t, sp in spans if t in ("MSK", "FILL")])
    ans_b1, ans_a1 = probe("deny")
    leak1, hit_a1 = norm(msk_code) in norm(ans_b1), norm(vis_code) in norm(ans_a1)
    print("  [deny ] B->%r  读到B=%s | A->%r 读到A=%s" % (ans_b1[:30], leak1, ans_a1[:30], hit_a1))

    write([])                                  # 放开，不重启
    ans_b2, ans_a2 = probe("open")
    leak2, hit_a2 = norm(msk_code) in norm(ans_b2), norm(vis_code) in norm(ans_a2)
    print("  [open ] B->%r  读到B=%s | A->%r 读到A=%s" % (ans_b2[:30], leak2, ans_a2[:30], hit_a2))

    ok = (not leak1) and hit_a1 and leak2 and hit_a2
    print("\n热重载%s" % ("生效 ✔" if ok else "未生效 ✘"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
