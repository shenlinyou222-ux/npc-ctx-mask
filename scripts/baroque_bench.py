# -*- coding: utf-8 -*-
"""冷门游戏 wiki 上的可见性对照（真实语料，非合成）。

语料：Baroque (1998) 的 Wikipedia 正文。
标签：世界观（Gameplay / Plot）可见；制作、音乐、发行、评价受限（角色不该知道这些）。
问题：10 条世界观（该答） + 5 条制作信息（该拒答）。
四臂：CB 闭卷 / A 全塞 / M 块掩码 / D 文本删除。
"""
import json
import re
import statistics as st
import sys
import time
import urllib.request

VIS = "/home/user/_ab_scratch/npc_vis_blocks.json"
URL = "http://127.0.0.1:8001/v1/chat/completions"
SRC = __import__("os").environ.get("BAROQUE_SRC", "/home/user/data/baroque.txt")
B, PAD = 16, "\u200c"
SYS = ("You answer using only the provided records. Reply with the shortest exact answer. "
       "If the records do not contain the answer, reply exactly: I don't know.")

VISIBLE_SEC = {"Gameplay", "Plot", "Setting and characters", "Story"}
SKIP_SEC = {"Notes", "References", "External links", "Intro", "Reception", "Remake", "Additional media", "Release"}

QS = [
    ("What is the world-altering cataclysm in Baroque called?", "Blaze", "in"),
    ("When did the Blaze take place?", "May 14, 2032", "in"),
    ("What are the once-human creatures the protagonist must purify called?", "Meta-Beings", "in"),
    ("Which order discovered that the Absolute God had returned?", "Malkuth", "in"),
    ("Who can read an Idea Sephirah?", "Baroquemonger", "in"),
    ("Who lost her identity to shield herself and can voice others' thoughts?", "Horned Girl", "in"),
    ("What are the embodiments of pain called?", "Littles", "in"),
    ("Who cultivated the Littles?", "Angelicus", "in"),
    ("What weapon uses the Littles as bullets?", "Angelic Rifle", "in"),
    ("Who lies impaled on a Consciousness Orb at the tower's bottom floor?", "Archangel", "in"),
    ("Who composed the music of Baroque?", "Iwata", "out"),
    ("Who created the concept and directed Baroque?", "Yonemitsu", "out"),
    ("What was the first song created for the soundtrack?", "Sanctuary", "out"),
    ("What platform was the early prototype developed for?", "PC-9800", "out"),
    ("Which studio created the CGI?", "Inertia Pictures", "out"),
]


def post(system, q, rid=None, max_tokens=16):
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
    return re.sub(r"[^a-z0-9]", "", s.lower())


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/mnt/c/Users/user/npc-models-qwen3-1.7b")

    def ntok(s):
        return len(tok(s, add_special_tokens=False)["input_ids"])

    # chat template 在 system 内容前插入的 token 数
    _probe = tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": ""}],
        tokenize=True, add_generation_prompt=True, return_dict=False)
    _sysids = tok(SYS, add_special_tokens=False)["input_ids"]
    TMPL_OFF = next(i for i in range(len(_probe)) if _probe[i:i + len(_sysids)] == _sysids)
    print("chat template 偏移 = %d token" % TMPL_OFF)

    def blkpos(s):
        return TMPL_OFF + ntok(s)

    raw = open(SRC, encoding="utf-8").read()
    cur, secs, buf = "Intro", {}, []
    for line in raw.split("\n"):
        m = re.match(r"^==+\s*(.+?)\s*==+$", line.strip())
        if m:
            secs[cur] = "\n".join(buf).strip()
            cur, buf = m.group(1), []
        else:
            buf.append(line)
    secs[cur] = "\n".join(buf).strip()

    docs = []
    for name, txt in secs.items():
        if name in SKIP_SEC or not txt:
            continue
        vis = name in VISIBLE_SEC
        para, acc = txt.split("\n"), ""
        chunks = []
        for p in para:
            if not p.strip():
                continue
            if ntok(acc + " " + p) > 200 and acc:
                chunks.append(acc)
                acc = p
            else:
                acc = (acc + " " + p).strip()
        if acc:
            chunks.append(acc)
        for c in chunks:
            docs.append({"sec": name, "vis": vis, "text": c})

    head = SYS + "\n\n"
    body, spans = "", []
    for i, d in enumerate(docs):
        a = -(-blkpos(head + body) // B)
        body += "[%02d] %s\n" % (i, d["text"])
        need = (B - blkpos(head + body) % B) % B
        body += PAD * need
        b = blkpos(head + body) // B
        spans.append((i, d["vis"], (a, b)))
    text_all = head + body
    total_blk = blkpos(text_all) // B
    deny = [sp for _, vis, sp in spans if not vis]
    cov = sum(b - a for a, b in deny)
    text_del = head + "\n".join("[%02d] %s" % (i, d["text"]) for i, d in enumerate(docs) if d["vis"])
    print("语料：%d 段 / %d token / %d 块；受限 %d 块（%.0f%%）"
          % (len(docs), ntok(text_all), total_blk, cov, 100 * cov / total_blk))
    print("受限段落来自：", ", ".join(sorted({d["sec"] for d in docs if not d["vis"]})))

    rows = []
    for k, (q, gold, kind) in enumerate(QS):
        res = {}
        for arm, rid, sysp in (("CB", None, "You answer questions. Reply with the shortest exact answer."),
                               ("A", "scope=all|A%d" % k, text_all),
                               ("M", "scope=m|M%d" % k, text_all),
                               ("D", None, text_del)):
            json.dump({"m": deny}, open(VIS, "w", encoding="utf-8"))
            ans, pt, ttft = post(sysp, q, rid)
            res[arm] = {"hit": norm(gold) in norm(ans),
                        "abstain": "dontknow" in norm(ans),
                        "ans": ans[:40], "pt": pt, "ttft": ttft}
        rows.append({"q": q, "gold": gold, "kind": kind, **res})
        print("  %02d %-3s CB=%d A=%d M=%d D=%d  %s"
              % (k, kind, res["CB"]["hit"], res["A"]["hit"], res["M"]["hit"],
                 res["D"]["hit"], q[:44]), flush=True)

    for kind, name in (("in", "世界观（该答）"), ("out", "制作信息（该拒答）")):
        sub = [r for r in rows if r["kind"] == kind]
        print("\n%s n=%d" % (name, len(sub)))
        for arm in ("CB", "A", "M", "D"):
            hits = sum(r[arm]["hit"] for r in sub)
            ab = sum(r[arm]["abstain"] for r in sub)
            tt = st.median([r[arm]["ttft"] for r in sub])
            print("  %-2s 命中/泄漏 %-6s 拒答 %-6s TTFT %.3fs"
                  % (arm, "%d/%d" % (hits, len(sub)), "%d/%d" % (ab, len(sub)), tt))
    json.dump({"rows": rows, "coverage": cov / total_blk, "tokens": blkpos(text_all), "tmpl_offset": TMPL_OFF},
              open("/home/user/_ab_scratch/baroque.json", "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
