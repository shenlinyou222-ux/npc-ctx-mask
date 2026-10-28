# -*- coding: utf-8 -*-
"""MultiHop-RAG：时间可见性三臂对照。

标签：文章 published_at <= CUTOFF 可见，之后受限（= "只知情到 T 之前"）。
问题分两类：
  answerable   所有证据文章都可见 -> 应答对
  unanswerable 有证据文章在受限区 -> 应答"不知道"，说出来即泄漏
另加闭卷过滤：不看任何上下文就能答对的题（模型记忆）单独剔除。
"""
import json
import re
import statistics as st
import sys
import time
import urllib.request

VIS = "/home/user/_ab_scratch/npc_vis_blocks.json"
URL = "http://127.0.0.1:8001/v1/chat/completions"
CORPUS = "/home/user/data/multihop_corpus.json"
QUERIES = "/home/user/data/multihop_queries.json"
B, PAD = 16, "\u200c"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
CUTOFF = "2023-11-15"                     # 只知情到这一天
MAX_ART_TOK = 110

SYS = ("You are an assistant that only knows what is in the provided news passages. "
       "Answer with the shortest exact answer. If the passages do not contain the answer, "
       "reply exactly: I don't know.")


def post(system, rid, q, max_tokens=24):
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
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/mnt/c/Users/user/npc-models-qwen3-1.7b")

    def ntok(s):
        return len(tok(s, add_special_tokens=False)["input_ids"])

    _probe = tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": ""}],
        tokenize=True, add_generation_prompt=True, return_dict=False)
    import re as _re
    _sysids = tok(SYS, add_special_tokens=False)["input_ids"]
    TMPL_OFF = next(i for i in range(len(_probe)) if _probe[i:i + len(_sysids)] == _sysids)
    print("chat template 偏移 = %d token" % TMPL_OFF)

    def blkpos(s):
        return TMPL_OFF + ntok(s)

    corpus = json.load(open(CORPUS, encoding="utf-8"))
    queries = json.load(open(QUERIES, encoding="utf-8"))
    by_title = {c["title"]: c for c in corpus}

    rows, used = [], 0
    for qi, q in enumerate(queries):
        ev = [e["title"] for e in q.get("evidence_list", []) if e["title"] in by_title]
        if not ev or len(ev) > 4:
            continue
        arts = []
        for t in ev[:4]:
            c = by_title[t]
            body_ids = tok(c["body"], add_special_tokens=False)["input_ids"][:MAX_ART_TOK]
            arts.append((c["title"], c["published_at"], tok.decode(body_ids)))
        # 补 4 篇干扰（随机、且不是证据）
        for c in corpus:
            if len(arts) >= 8:
                break
            if c["title"] in ev:
                continue
            body_ids = tok(c["body"], add_special_tokens=False)["input_ids"][:60]
            arts.append((c["title"], c["published_at"], tok.decode(body_ids)))

        head = SYS + "\n\n"
        body, spans = "", []
        for title, pub, txt in arts:
            a = -(-blkpos(head + body) // B)
            body += "%s (%s): %s\n" % (title, pub[:10], txt)
            need = (B - blkpos(head + body) % B) % B
            body += PAD * need
            b = blkpos(head + body) // B
            spans.append((title, pub, (a, b)))
        text_all = head + body
        restricted = [(t, sp) for t, pub, sp in spans if pub[:10] > CUTOFF]
        deny = [sp for _, sp in restricted]
        allowed = [(t, pub, sp) for t, pub, sp in spans if pub[:10] <= CUTOFF]
        text_del = head + "\n".join("%s (%s): %s" % (t, p[:10], by_title[t]["body"][:400])
                                    for t, p, _ in allowed)

        answerable = all(by_title[t]["published_at"][:10] <= CUTOFF for t in ev)
        qtext = q["query"]
        gold = norm(q["answer"])

        # 闭卷
        cb, _, _ = post("Answer with the shortest exact answer.", None, qtext, 16)
        memory = gold in norm(cb)

        res = {}
        for arm, rid, sysp in (("A", "scope=all|A%d" % qi, text_all),
                               ("M", "scope=m|M%d" % qi, text_all),
                               ("D", None, text_del)):
            json.dump({"m": deny, "all": []}, open(VIS, "w", encoding="utf-8"))
            ans, pt, ttft = post(sysp, rid, qtext)
            res[arm] = {"hit": gold in norm(ans), "abstain": "don't know" in ans.lower(),
                        "ans": ans[:50], "ttft": ttft, "pt": pt}
        rows.append({"q": qtext, "gold": q["answer"], "answerable": answerable,
                     "memory": memory, "n_restricted": len(restricted), **res})
        used += 1
        print("  q%02d %-12s mem=%d rest=%d hit(A/M/D)=%s"
              % (qi, "answerable" if answerable else "unanswerable", memory,
                 len(restricted),
                 "".join("1" if res[a]["hit"] else "0" for a in ("A", "M", "D"))),
              flush=True)
        if used >= N:
            break

    fresh = [r for r in rows if not r["memory"]]
    print("\n共 %d 条；闭卷已能答对（记忆项）%d 条 -> 有效 %d 条"
          % (len(rows), len(rows) - len(fresh), len(fresh)))
    for name, subset in (("answerable", [r for r in fresh if r["answerable"]]),
                         ("unanswerable", [r for r in fresh if not r["answerable"]])):
        if not subset:
            continue
        print("%s n=%d" % (name, len(subset)))
        for arm in ("A", "M", "D"):
            if name == "answerable":
                s = sum(r[arm]["hit"] for r in subset)
                lab = "命中"
            else:
                s = sum(r[arm]["hit"] for r in subset)
                lab = "泄漏"
            ab = sum(r[arm]["abstain"] for r in subset)
            tt = st.median([r[arm]["ttft"] for r in subset])
            print("  %-2s %s %-7s 拒答 %-7s TTFT %.3fs"
                  % (arm, lab, "%d/%d" % (s, len(subset)),
                     "%d/%d" % (ab, len(subset)), tt))
    json.dump(rows, open("/home/user/_ab_scratch/multihop_bench.json", "w"),
              ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
