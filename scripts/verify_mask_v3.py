# -*- coding: utf-8 -*-
"""修正版 canary（多轮）：块区间在渲染后的真实序列里计算。

修正点：chat template 会在 system 内容前插 3 个 token，
旧脚本按原始字符串算块号，整体偏一格，会遮错块。
"""
import json
import os
import re
import sys
import time
import urllib.request

VIS = "/home/user/_ab_scratch/npc_vis_blocks.json"
URL = "http://127.0.0.1:8001/v1/chat/completions"
B, PAD = 16, "\u200c"
SYS = ("You answer using only the provided records. Reply with the shortest exact answer. "
       "If the records do not contain the answer, reply exactly: I don't know.")
FILL = ("Routine maintenance logs indicate steady output and no anomalies were reported "
        "by the night shift. ")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8


def post(sys_text, q, rid, max_tokens=24):
    body = {"model": "q17b-flex",
            "messages": [{"role": "system", "content": sys_text},
                         {"role": "user", "content": q}],
            "temperature": 0.0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}, "request_id": rid}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"].strip()


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def write_cfg(cfg):
    tmp = VIS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    os.replace(tmp, VIS)
    time.sleep(0.15)


def locate(full, part):
    m = len(part)
    for i in range(len(full) - m + 1):
        if full[i:i + m] == part:
            return i
    return -1


def one(tok, enc, render, ca, cb, i, offset):
    q = "What is the access code in Record B?"
    qa = "What is the access code in Record A?"
    sys_text, body, spans = SYS + "\n\n", "", []
    a = len(enc(sys_text + body))
    body += "Record A: the access code is %s.\n" % ca
    body += PAD * ((B - len(enc(sys_text + body)) % B) % B)
    spans.append(("VIS", (a, len(enc(sys_text + body)))))
    for _ in range(2):
        a = len(enc(sys_text + body))
        body += "Log: %s\n" % FILL
        body += PAD * ((B - len(enc(sys_text + body)) % B) % B)
        spans.append(("FILL", (a, len(enc(sys_text + body)))))
    a = len(enc(sys_text + body))
    body += "Record B: the access code is %s.\n" % cb
    body += PAD * ((B - len(enc(sys_text + body)) % B) % B)
    spans.append(("MSK", (a, len(enc(sys_text + body)))))
    full = sys_text + body

    ids_full = render(full, q)
    st = locate(ids_full, enc(full))
    assert st >= 0, "定位失败"
    blk = {t: ((st + s) // B, (st + z) // B) for t, (s, z) in spans}
    deny = [list(blk["MSK"])]

    write_cfg({"visible": [], "masked": deny, "live": deny})
    vb = post(full, q, "scope=visible|t%d" % i)
    ma = post(full, qa, "scope=masked|t%d" % i)
    mb = post(full, q, "scope=masked|t%d" % i)

    write_cfg({"visible": [], "masked": deny, "live": []})
    ob = post(full, q, "scope=live|t%d" % i)

    return (norm(cb) in norm(vb), norm(ca) in norm(ma),
            norm(cb) in norm(mb), norm(cb) in norm(ob), st, blk)


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/mnt/c/Users/user/npc-models-qwen3-1.7b")
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    render = lambda s, q: tok.apply_chat_template(  # noqa: E731
        [{"role": "system", "content": s}, {"role": "user", "content": q}],
        tokenize=True, add_generation_prompt=True, return_dict=False)

    hits = [0, 0, 0, 0]
    st = -1
    for i in range(N):
        ca = "VIS-%04d" % (1000 + i * 37)
        cb = "MSK-%04d" % (2000 + i * 53)
        v, a_ok, mk, op, st, blk = one(tok, enc, render, ca, cb, i, st)
        for k, val in enumerate((v, a_ok, mk, op)):
            hits[k] += int(val)
        print("  t%d vis_readB=%d mask_readA=%d mask_leakB=%d open_readB=%d blk=%s"
              % (i, v, a_ok, mk, op, blk), flush=True)

    print("\nn=%d  template offset=%d" % (N, st))
    print("  visible 读到B  %d/%d" % (hits[0], N))
    print("  masked  读到A  %d/%d  (正对照)" % (hits[1], N))
    print("  masked  泄漏B  %d/%d  (应为 0)" % (hits[2], N))
    print("  open    读到B  %d/%d  (热切换)" % (hits[3], N))
    ok = hits[0] == N and hits[1] == N and hits[2] == 0 and hits[3] == N
    print("\n%s" % ("全部成立 ✔" if ok else "存在失败 ✘"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
