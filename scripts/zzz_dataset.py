# -*- coding: utf-8 -*-
"""构建 ZZZ「版本可见性」RAG 数据集。

可见性：asker 的知识截止到版本 V；version > V 的文档受限。
数据：data/zzz/corpus_all.jsonl + corpus_story.jsonl（Fandom, CC BY-SA）
产出：data/zzz/dataset/{corpus,scopes,queries}.jsonl + NOTICE.md
"""
import json
import os
import re
import sys
from collections import Counter

BASE = "/mnt/c/Users/user/Desktop/ds harness/data/zzz"
OUT = os.path.join(BASE, "dataset")
SCOPES = ["2.0", "2.4", "2.7", "3.2"]

# infobox 里「键在上一行、值在下一行」
KEYS = {
    "Faction": "Which faction does {title} belong to?",
    "Birthday": "When is {title}'s birthday?",
    "Gender": "What is {title}'s gender?",
    "Species": "What species is {title}?",
    "Height": "How tall is {title}?",
    "Real Name": "What is {title}'s real name?",
}


def vnum(v):
    try:
        a, b = str(v).split(".")
        return int(a) * 100 + int(b)
    except Exception:  # noqa: BLE001
        return None


def parse_infobox(text):
    lines = [l.strip() for l in text.split("\n")]
    facts = {}
    for i, l in enumerate(lines):
        if l in KEYS and l not in facts:
            for j in range(i + 1, min(i + 3, len(lines))):
                v = lines[j].strip()
                if v and v not in KEYS and len(v) <= 50 and not v.startswith("("):
                    facts[l] = v
                    break
    return facts


def load_all():
    rows, seen = [], set()
    for name in ("corpus_story.jsonl", "corpus_all.jsonl"):
        p = os.path.join(BASE, name)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r["title"] in seen:
                    continue
                seen.add(r["title"])
                rows.append(r)
    return rows


def main():
    rows = load_all()
    print("合并去重后页面 %d" % len(rows))

    keep = []
    for r in rows:
        v = vnum(r.get("version"))
        if v is not None and v >= 200:
            keep.append(r)
        elif r.get("tenant") in ("agent", "lore", "faction") and re.search(
                r"Version\s+[23]\.\d", r["text"]):
            keep.append(r)
    print("2.0+ 页面 %d" % len(keep))
    print("  版本分布：", dict(sorted(Counter(r.get("version") or "-" for r in keep).items(),
                                  key=lambda x: vnum(x[0]) or 0)))

    os.makedirs(OUT, exist_ok=True)
    docs, queries, qid = [], [], 0
    for r in keep:
        docs.append({"doc_id": r["id"], "title": r["title"], "revid": r.get("revid"),
                     "version": r.get("version") or "", "tenant": r.get("tenant"),
                     "chars": r.get("chars"), "text": r["text"]})
        # 只从 2.0+ 的角色页出题（1.x 内容模型记得住，不能用来测上下文）
        if (r.get("tenant") != "agent" or not r.get("version")
                or (vnum(r["version"]) or 0) < 200):
            continue
        for k, v in parse_infobox(r["text"]).items():
            qid += 1
            queries.append({"qid": "Q%04d" % qid,
                            "question": KEYS[k].format(title=r["title"]),
                            "gold": v, "source_doc": r["id"],
                            "answer_version": r["version"], "type": "single",
                            "fact_key": k})

    out_q = []
    for q in queries:
        av = vnum(q["answer_version"])
        for s in SCOPES:
            sv = vnum(s)
            out_q.append({**q, "qid": q["qid"] + "@" + s, "scope_id": s,
                          "expected": "answer" if av <= sv else "abstain"})

    def dump(path, items):
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    dump(os.path.join(OUT, "corpus.jsonl"), docs)
    dump(os.path.join(OUT, "scopes.jsonl"),
         [{"scope_id": s, "cutoff_version": s} for s in SCOPES])
    dump(os.path.join(OUT, "queries.jsonl"), out_q)

    ans = sum(1 for q in out_q if q["expected"] == "answer")
    print("文档 %d / 基础问题 %d / 展开 %d（该答 %d、该拒答 %d）"
          % (len(docs), len(queries), len(out_q), ans, len(out_q) - ans))
    print("问题来源版本：", dict(sorted(Counter(q["answer_version"] for q in queries).items(),
                                   key=lambda x: vnum(x[0]) or 0)))
    with open(os.path.join(BASE, "NOTICE.md"), encoding="utf-8") as f:
        notice = f.read()
    with open(os.path.join(OUT, "NOTICE.md"), "w", encoding="utf-8") as f:
        f.write(notice)
    print("->", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
