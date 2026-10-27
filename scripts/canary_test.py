# -*- coding: utf-8 -*-
"""掩码正确性验证（canary）：被遮块里的唯一编号，模型必须读不出来。

每次试验构造：
  [可见块A: 编号 CANARY_VIS]  [被遮块: 编号 CANARY_MASK]  [问题]
条件：
  masked   遮掉 CANARY_MASK 所在块 -> 问它必须答不出
  visible  不遮                 -> 问它应该答得出（正对照）
  deleted  物理删除             -> 问它必须答不出（等价对照）
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
TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
SYS = ("You answer using only the provided records. Reply with the shortest exact answer. "
       "If the records do not contain the answer, reply exactly: I don't know.")
FILL = ("Routine maintenance logs indicate steady output and no anomalies were reported "
        "by the night shift. ")


def post(system, q, rid=None, max_tokens=24):
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
    ttft, text = None, ""
    with urllib.request.urlopen(req, timeout=300) as r:
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
            ch = d.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                text += ch[0]["delta"]["content"]
    return text.strip(), ttft or 0.0


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/mnt/c/Users/user/npc-models-qwen3-1.7b")

    def ntok(s):
        return len(tok(s, add_special_tokens=False)["input_ids"])

    rnd = random.Random(20261027)
    rows = []
    for t in range(TRIALS):
        vis_code = "VIS-%04d" % rnd.randint(1000, 9999)
        msk_code = "MSK-%04d" % rnd.randint(1000, 9999)
        head = SYS + "\n\n"
        body, spans = "", []
        # 可见块
        a = -(-ntok(head + body) // B)
        body += "Record A: the access code is %s.\n" % vis_code
        body += PAD * ((B - ntok(head + body) % B) % B)
        b = ntok(head + body) // B
        spans.append(("VIS", (a, b)))
        # 被遮块（夹在中间，前面再垫一点中性填充）
        for _ in range(3):
            a = -(-ntok(head + body) // B)
            body += "Log: %s\n" % FILL
            body += PAD * ((B - ntok(head + body) % B) % B)
            b = ntok(head + body) // B
            spans.append(("FILL", (a, b)))
        a = -(-ntok(head + body) // B)
        body += "Record B: the access code is %s.\n" % msk_code
        body += PAD * ((B - ntok(head + body) % B) % B)
        b = ntok(head + body) // B
        spans.append(("MSK", (a, b)))
        text_all = head + body
        text_del = (head
                    + "Record A: the access code is %s.\n" % vis_code
                    + "".join("Log: %s\n" % FILL for _ in range(3)))

        res = {}
        for arm in ("visible", "masked", "deleted"):
            deny = [] if arm != "masked" else [sp for tag, sp in spans if tag in ("MSK", "FILL")]
            json.dump({arm: deny}, open(VIS, "w", encoding="utf-8"))
            sysp = text_all if arm != "deleted" else text_del
            ask_m, _ = post(sysp, "What is the access code in Record B?", "scope=%s|%d" % (arm, t))
            ask_v, _ = post(sysp, "What is the access code in Record A?", "scope=%s|%d" % (arm, t))
            res[arm] = {"masked_leak": norm(msk_code) in norm(ask_m),
                        "visible_hit": norm(vis_code) in norm(ask_v),
                        "ans_m": ask_m[:40], "ans_v": ask_v[:40]}
        rows.append({"vis": vis_code, "msk": msk_code, **res})
        print("  t%d masked: leakB=%d visA=%d | visible: leakB=%d visA=%d | deleted: leakB=%d"
              % (t, res["masked"]["masked_leak"], res["masked"]["visible_hit"],
                 res["visible"]["masked_leak"], res["visible"]["visible_hit"],
                 res["deleted"]["masked_leak"]), flush=True)

    print("\n%-10s %-16s %-16s" % ("条件", "读到被遮编号", "读到可见编号"))
    for arm in ("visible", "masked", "deleted"):
        print("%-10s %-16s %-16s"
              % (arm, "%d/%d" % (sum(r[arm]["masked_leak"] for r in rows), len(rows)),
                 "%d/%d" % (sum(r[arm]["visible_hit"] for r in rows), len(rows))))
    json.dump(rows, open("/home/user/_ab_scratch/canary.json", "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
