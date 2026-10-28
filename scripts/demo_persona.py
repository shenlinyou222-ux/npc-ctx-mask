# -*- coding: utf-8 -*-
"""ZZZ 多人物演示：一份共享 KV 缓存，四种剧情进度，各自只能看到自己该看的块。

设计
- 语料只 prefill 一次（61 篇角色页，~9.8K token / 614 块），之后**不改 token 序列**；
- 人物身份只出现在 user 轮，因此语料前缀始终命中；
- 每个人物一个知识截止版本（2.4 / 2.6 / 2.8 / 3.2），
  截止之后的角色页被块掩码遮掉（deny list 按人物 id 分键）；
- **人物提示里不写"你只知道 X 版本"**——否则模型是靠提示拒答，不是靠掩码；
- 每人 2–4 问：1 问"他该知道"（answer_version <= 截止），2–3 问"他不该知道"（> 截止）；
- 两臂：**A 全知（不遮）** vs **M 角色视角（遮）**——同一段 token 序列，只换可见性位图。

指标：TTFT（首 token）、TTOT（末 token，端到端）、输出 token、遮块数与占比、
      是否复现正解（越界 = 不该答的题复现了正解）。

用法: demo_persona.py [--max-tokens 64]
输出: /home/user/_ab_scratch/demo_persona.json
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

VIS = "/home/user/_ab_scratch/npc_vis_blocks.json"
URL = "http://127.0.0.1:8001/v1/chat/completions"
DS = "/mnt/c/Users/user/Desktop/ds harness/data/zzz/dataset"
MODEL = "/mnt/c/Users/user/npc-models-qwen3-1.7b"
OUT = "/home/user/_ab_scratch/demo_persona.json"
B, PAD = 16, "\u200c"
CLIP = 150

SYS = ("You answer questions about the world using only the records below. "
       "Each record is one character page, bracketed by its title. "
       "If the records do not contain the answer, say exactly: I don't know.")

CAST = [
    {"pid": "banyue",  "name": "Banyue",       "cutoff": "2.4",
     "line": "你是个头很高、话不多的护卫，回答简短克制。"},
    {"pid": "aria",    "name": "Aria",         "cutoff": "2.6",
     "line": "你是机敏的少女，说话轻快，喜欢用短句。"},
    {"pid": "promeia", "name": "Promeia",      "cutoff": "2.8",
     "line": "你冷静克制，习惯先确认事实再回答。"},
    {"pid": "claret",  "name": "Claret Flint", "cutoff": "3.2",
     "line": "你就是克拉蕾特·弗林特本人，语气从容自信。"},
]

PLAN = {
    "banyue":  [("st-015", "Birthday", True),  ("st-135", "Birthday", False),
                ("st-010", "Birthday", False), ("st-143", "Birthday", False)],
    "aria":    [("st-010", "Birthday", True),  ("st-026", "Birthday", False), ("st-143", "Birthday", False)],
    "promeia": [("st-122", "Birthday", True),  ("st-026", "Birthday", False), ("st-135", "Birthday", False)],
    "claret":  [("st-026", "Birthday", True),  ("st-117", "Birthday", False)],
}


def nrm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def vnum(v):
    try:
        a, b = str(v).split(".")
        return int(a) * 100 + int(b)
    except Exception:  # noqa: BLE001
        return -1


def write_vis(cfg):
    """原子写：先写 .tmp 再 replace，避免服务端读到半份 JSON。"""
    tmp = VIS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    os.replace(tmp, VIS)
    time.sleep(0.35)


def post(system, user, rid=None, max_tokens=64):
    body = {"model": "q17b-flex",
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0.0, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    if rid:
        body["request_id"] = rid
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, text, ptok, cached, out_tok, usage = None, "", 0, 0, 0, {}
    with urllib.request.urlopen(req, timeout=600) as r:
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
                u = d["usage"]
                usage = u
                ptok = u.get("prompt_tokens", ptok) or ptok
                out_tok = u.get("completion_tokens", out_tok) or out_tok
                cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", cached) or cached
            ch = d.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                text += ch[0]["delta"]["content"]
    return text.strip(), (ttft or 0.0), time.perf_counter() - t0, ptok, cached, out_tok, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--pages", type=int, default=7,
                    help="语料页数（题目涉及的角色页必选，其余按版本从高到低补）")
    ap.add_argument("--reps", type=int, default=3, help="每个单元重复次数，聚合取众数")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)

    def ntok(s):
        return len(tok(s, add_special_tokens=False)["input_ids"])

    probe = tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": ""}],
        tokenize=True, add_generation_prompt=True, return_dict=False)
    sysids = tok(SYS, add_special_tokens=False)["input_ids"]
    OFF = next(i for i in range(len(probe)) if probe[i:i + len(sysids)] == sysids)

    docs = {d["doc_id"]: d for d in (json.loads(l) for l in open(os.path.join(DS, "corpus.jsonl"), encoding="utf-8"))}
    qmap = {}
    for line in open(os.path.join(DS, "queries_base.jsonl"), encoding="utf-8"):
        q = json.loads(line)
        qmap.setdefault((q["source_doc"], q["fact_key"]), q)

    def clip(s):
        return tok.decode(tok(s, add_special_tokens=False)["input_ids"][:CLIP])

    required = {src for plan in PLAN.values() for src, _, _ in plan}
    fillers = [d for d in docs.values() if d["tenant"] == "agent" and d["doc_id"] not in required]
    fillers.sort(key=lambda d: (-vnum(d.get("version")), d["title"]))
    pages = [docs[x] for x in required] + fillers[: max(0, args.pages - len(required))]
    # 按标题排序：被遮的页会散布在缓存中段，而不是整齐的尾部——
    # 这样"掩码"和"删文本"的差别才是块级的，而不是"截断前缀"。
    pages.sort(key=lambda d: d["title"])
    head = SYS + "\n\n"
    body, spans = "", []
    for d in pages:
        a = (OFF + ntok(head + body)) // B
        body += "[%s] %s\n" % (d["title"], clip(d["text"]))
        body += PAD * ((B - (OFF + ntok(head + body)) % B) % B)
        spans.append({"doc_id": d["doc_id"], "title": d["title"],
                      "version": d.get("version") or "", "blk": [a, (OFF + ntok(head + body)) // B]})
    corpus = head + body
    total_blk = (OFF + ntok(corpus)) // B
    print("共享语料：%d 篇 / %d token / %d 块（模板偏移 %d）" % (len(pages), ntok(corpus), total_blk, OFF))

    def deny_for(cutoff):
        c = vnum(cutoff)
        rs = [s["blk"] for s in spans if vnum(s["version"]) > c]
        out = []
        for a, z in sorted(rs):
            if out and a <= out[-1][1]:
                out[-1][1] = max(out[-1][1], z)
            else:
                out.append([a, z])
        return out

    # 预热：整份语料进前缀缓存
    write_vis({})
    _, tt, tot, pt, ca, _, _ = post(corpus, "The records are loaded. Reply with OK.", None, 4)
    print("预热：pt=%d ttft=%.3fs ttot=%.3fs\n" % (pt, tt, tot))

    rows = []
    for p in CAST:
        deny = deny_for(p["cutoff"])
        dblk = sum(z - a for a, z in deny)
        print("== %s（截止 %s）遮 %d/%d 块 = %.0f%% ==" % (p["name"], p["cutoff"], dblk, total_blk, 100 * dblk / total_blk))
        for src, fact, should in PLAN[p["pid"]]:
            q = qmap[(src, fact)]
            user = ("【%s 的提问】%s\n（简短回答。）" % (p["name"], q["question"]))
            for arm in ("A", "M"):
                for rep in range(args.reps):
                    if arm == "M":
                        write_vis({p["pid"]: deny})
                        ans, ttft, ttot, pt, ca, ot, usg = post(
                            corpus, user, "scope=%s|%s|r%d" % (p["pid"], q["qid"], rep), args.max_tokens)
                    else:
                        write_vis({})
                        ans, ttft, ttot, pt, ca, ot, usg = post(
                            corpus, user, "scope=all|r%d" % rep, args.max_tokens)
                    a = nrm(ans)
                    abstain = ("dontknow" in a) or ("idontknow" in a) or ("不知道" in ans)
                    rows.append({
                        "persona": p["pid"], "name": p["name"], "cutoff": p["cutoff"], "arm": arm, "rep": rep,
                        "qid": q["qid"], "question": q["question"], "gold": q["gold"],
                        "answer_version": q["answer_version"], "should_answer": should,
                        "deny_blocks": dblk if arm == "M" else 0,
                        "deny_ranges": deny if arm == "M" else [],
                        "total_blocks": total_blk, "pt": pt, "cached": ca, "out_tokens": ot,
                        "ttft": round(ttft, 4), "ttot": round(ttot, 4), "ans": ans,
                        "hit": nrm(q["gold"]) in a, "abstain": abstain, "usage": usg,
                    })
                    flag = "复现正解" if nrm(q["gold"]) in a else "未复现"
                    print("   %-36s %s#%d ttft=%6.3f ttot=%6.3f out=%2d  %-6s %s"
                          % (q["question"][:36], arm, rep, ttft, ttot, ot, flag,
                             ans[:44].replace("\n", " ")), flush=True)

    # 聚合：同一 (人物, 问题, 臂) 的多轮取众数，并记录一致率
    cells = {}
    for r in rows:
        k = (r["persona"], r["qid"], r["arm"])
        cells.setdefault(k, []).append(r)
    agg = []
    for (pid, qid, arm), rs in cells.items():
        cnt = {}
        for r in rs:
            cnt[nrm(r["ans"])] = cnt.get(nrm(r["ans"]), 0) + 1
        top = max(cnt.values())
        mode_norm = [k for k, v in cnt.items() if v == top][0]
        rep0 = next(r for r in rs if nrm(r["ans"]) == mode_norm)
        agg.append({
            "persona": pid, "name": rs[0]["name"], "cutoff": rs[0]["cutoff"], "arm": arm,
            "qid": qid, "question": rs[0]["question"], "gold": rs[0]["gold"],
            "answer_version": rs[0]["answer_version"], "should_answer": rs[0]["should_answer"],
            "deny_blocks": rs[0]["deny_blocks"], "total_blocks": rs[0]["total_blocks"],
            "n": len(rs), "agree": round(top / len(rs), 2),
            "ans": rep0["ans"], "hit": rep0["hit"], "abstain": rep0["abstain"],
            "ttft_med": round(sorted(r["ttft"] for r in rs)[len(rs) // 2], 4),
            "ttot_med": round(sorted(r["ttot"] for r in rs)[len(rs) // 2], 4),
            "out_tokens": rep0["out_tokens"], "pt": rep0["pt"],
        })
    json.dump({"corpus": {"pages": len(pages), "tokens": ntok(corpus), "blocks": total_blk},
               "cast": CAST, "spans": spans, "rows": rows, "cells": agg},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n写入 %s（逐请求 %d 条 / 聚合 %d 条）" % (OUT, len(rows), len(agg)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
