# -*- coding: utf-8 -*-
"""四格检验（canary 原结构）：同一轮里跑四次。

  ① B 可见，内容 code1  -> 应答出 code1
  ② B 可见，内容 code2  -> 应答出 code2（且与①不同）
  ③ B 被遮，内容 code1  -> 不应答出 code1
  ④ B 被遮，内容 code2  -> 不应答出 code2，且输出必须与③**逐字相同**

③④ 逐字相同 = 换掉被遮块的内容对输出零影响。
①② 各自答对 = 同一位置上内容确实可读，证明③④不是"两次都失败"。
"""
import json
import random
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
    ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])  # noqa: E731

    rnd = random.Random(20261028)
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 6

    def build(vis_code, msk_code):
        """完全照搬 canary 的构造：按原始 token 数补零宽字符。"""
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
        return head + body, [sp for t, sp in spans if t == "MSK"]

    qb = "What is the access code in Record B?"
    stats = {"v1_ok": 0, "v2_ok": 0, "diff": 0, "m3_leak": 0, "m4_leak": 0, "same": 0}
    rows = []
    used = 0

    for i in range(N):
        vis = "VIS-%04d" % rnd.randint(1000, 9999)
        c1 = "MSK-%04d" % rnd.randint(1000, 9999)
        c2 = "MSK-%04d" % rnd.randint(1000, 9999)

        ctx1, msk_span = build(vis, c1)
        ctx2, msk_span2 = build(vis, c2)
        if msk_span != msk_span2:
            print("  %d) 布局不一致，跳过" % (i + 1))
            continue
        used += 1

        json.dump({"off": [], "on": msk_span}, open(VIS, "w", encoding="utf-8"))

        a1 = post(ctx1, qb, "scope=off|%d-1" % i)     # ① 可见 code1
        a2 = post(ctx2, qb, "scope=off|%d-2" % i)     # ② 可见 code2
        a3 = post(ctx1, qb, "scope=on|%d-3" % i)      # ③ 被遮 code1
        a4 = post(ctx2, qb, "scope=on|%d-4" % i)      # ④ 被遮 code2

        v1 = norm(c1) in norm(a1)
        v2 = norm(c2) in norm(a2)
        leaked3 = norm(c1) in norm(a3)
        leaked4 = norm(c2) in norm(a4)
        same = a3 == a4

        stats["v1_ok"] += int(v1)
        stats["v2_ok"] += int(v2)
        stats["diff"] += int(a1 != a2)
        stats["m3_leak"] += int(leaked3)
        stats["m4_leak"] += int(leaked4)
        stats["same"] += int(same)

        rows.append({"trial": i + 1, "code1": c1, "code2": c2, "block": msk_span,
                     "visible_1": a1, "visible_2": a2, "masked_1": a3, "masked_2": a4,
                     "v1_ok": v1, "v2_ok": v2, "diff": a1 != a2,
                     "m3_leak": leaked3, "m4_leak": leaked4, "same": same})
        print("  %d) %s / %s   可见块=%s" % (i + 1, c1, c2, msk_span))
        print("     ①可见读1: %-20r %s" % (a1[:20], "✔" if v1 else "✘"))
        print("     ②可见读2: %-20r %s" % (a2[:20], "✔" if v2 else "✘"))
        print("     ③遮后问1: %-20r 泄漏=%s" % (a3[:20], leaked3))
        print("     ④遮后问2: %-20r 泄漏=%s   ③==④:%s"
              % (a4[:20], leaked4, same), flush=True)

    print("\nn=%d" % used)
    print("  ① 可见时读到 code1      %d/%d" % (stats["v1_ok"], used))
    print("  ② 可见时读到 code2      %d/%d" % (stats["v2_ok"], used))
    print("  ①≠②（内容确实有用）     %d/%d" % (stats["diff"], used))
    print("  ③ 遮蔽后泄漏 code1      %d/%d（应为 0）" % (stats["m3_leak"], used))
    print("  ④ 遮蔽后泄漏 code2      %d/%d（应为 0）" % (stats["m4_leak"], used))
    print("  ③==④（换内容无影响）    %d/%d" % (stats["same"], used))

    json.dump({"n": used, "summary": stats, "rows": rows},
              open("/home/user/_ab_scratch/four_cell.json", "w"), ensure_ascii=False, indent=1)
    ok = (stats["v1_ok"] == used and stats["v2_ok"] == used and stats["diff"] == used
          and stats["m3_leak"] == 0 and stats["m4_leak"] == 0 and stats["same"] == used)
    print("\n%s" % ("四格检验全部通过 ✔" if ok else "未全部通过"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
