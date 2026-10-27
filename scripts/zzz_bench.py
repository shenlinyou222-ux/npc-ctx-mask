# -*- coding: utf-8 -*-
"""ZZZ 版本可见性评测：asker 的知识截止到某个版本，之后的文档必须被屏蔽。

对每个问题取两个 scope：
  visible  scope = 3.2（源文档可见）      -> 应答对
  hidden   scope < answer_version（源文档受限） -> 应答"不知道"，说出来即泄漏
四臂：CB 闭卷 / A 全塞 / M 块掩码 / D 文本删除
"""
import json
import os
import re
import statistics as st
import sys
import time
import urllib.request

VIS = "/home/user/_ab_scratch/npc_vis_blocks.json"
URL = "http://127.0.0.1:8001/v1/chat/completions"
DS = "/mnt/c/Users/user/Desktop/ds harness/data/zzz/dataset"
B, PAD = 16, "\u200c"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
CUT = 240          # 每篇文档最大 token
SYS = ("You answer using only the provided records. Reply with the shortest exact answer. "
       "If the records do not contain the answer, reply exactly: I don't know.")


def vnum(v):
    try:
        a, b = str(v).split(".")
        return int(a) * 100 + int(b)
    except Exception:  # noqa: BLE001
        return None


def post(system, q, rid=None, max_tokens=20):
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
    ttft, text, ptok = None, "", 0
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
            if d.get("usage"):
                ptok = d["usage"].get("prompt_tokens", ptok)
            ch = d.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                text += ch[0]["delta"]["content"]
    return text.strip(), ptok, ttft or 0.0


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/mnt/c/Users/user/npc-models-qwen3-1.7b")

    def ntok(s):
        return len(tok(s, add_special_tokens=False)["input_ids"])

    def clip(s):
        return tok.decode(tok(s, add_special_tokens=False)["input_ids"][:CUT])

    docs = {}
    for line in open(os.path.join(DS, "corpus.jsonl"), encoding="utf-8"):
        d = json.loads(line)
        docs[d["doc_id"]] = d
    base = {}
    for line in open(os.path.join(DS, "queries_base.jsonl"), encoding="utf-8"):
        q = json.loads(line)
        base[q["qid"]] = q

    agent_docs = [d for d in docs.values() if d["tenant"] == "agent"]
    items = []
    for q in list(base.values()):
        src = docs.get(q["source_doc"])
        if not src:
            continue
        others = [d for d in agent_docs if d["doc_id"] != src["doc_id"]]
        items.append({"q": q, "src": src, "distract": others[:6]})
        if len(items) >= N:
            break

    print("题目 %d 条，上下文 = 源文档 + 6 篇干扰" % len(items))
    rows = []
    for i, it in enumerate(items):
        q, src = it["q"], it["src"]
        head = SYS + "\n\n"
        body, spans = "", []
        for d in [src] + it["distract"]:
            a = -(-ntok(head + body) // B)
            body += "[%s] %s\n" % (d["title"], clip(d["text"]))
            need = (B - ntok(head + body) % B) % B
            body += PAD * need
            b = ntok(head + body) // B
            spans.append((d["doc_id"], d.get("version") or "", (a, b)))
        text_all = head + body
        av = vnum(q["answer_version"])

        for tag, cutoff in (("visible", 999), ("hidden", max(0, av - 1))):
            deny = [sp for did, ver, sp in spans if (vnum(ver) or 0) > cutoff]
            keep = [(did, ver) for did, ver, sp in spans if (vnum(ver) or 0) <= cutoff]
            text_del = head + "\n".join(
                "[%s] %s" % (docs[did]["title"], clip(docs[did]["text"])) for did, _ in keep)
            res = {}
            for arm, rid, sysp in (("CB", None, "Answer questions. Reply with the shortest exact answer."),
                                   ("A", "scope=all|%d%s" % (i, tag), text_all),
                                   ("M", "scope=m|%d%s" % (i, tag), text_all),
                                   ("D", None, text_del)):
                json.dump({"m": deny}, open(VIS, "w", encoding="utf-8"))
                ans, pt, ttft = post(sysp, q["question"], rid)
                res[arm] = {"hit": norm(q["gold"]) in norm(ans),
                            "abstain": "dontknow" in norm(ans),
                            "ans": ans[:40], "pt": pt, "ttft": ttft}
            rows.append({"tag": tag, "q": q["question"], "gold": q["gold"],
                         "av": q["answer_version"], "deny_blocks": sum(b - a for a, b in deny),
                         "total_blocks": ntok(text_all) // B, **res})
            print("  %02d %-8s A=%d M=%d D=%d  %s" % (i, tag, res["A"]["hit"],
                                                      res["M"]["hit"], res["D"]["hit"],
                                                      q["question"][:38]), flush=True)

    for tag in ("visible", "hidden"):
        sub = [r for r in rows if r["tag"] == tag]
        print("\n[%s] n=%d  平均遮块 %.0f%%" % (
            tag, len(sub), 100 * sum(r["deny_blocks"] for r in sub) /
            max(1, sum(r["total_blocks"] for r in sub))))
        for arm in ("CB", "A", "M", "D"):
            h = sum(r[arm]["hit"] for r in sub)
            ab = sum(r[arm]["abstain"] for r in sub)
            tt = st.median([r[arm]["ttft"] for r in sub])
            print("  %-2s 命中 %-7s 拒答 %-7s TTFT %.3fs"
                  % (arm, "%d/%d" % (h, len(sub)), "%d/%d" % (ab, len(sub)), tt))
    json.dump(rows, open("/home/user/_ab_scratch/zzz_bench.json", "w"),
              ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
