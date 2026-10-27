# -*- coding: utf-8 -*-
"""数据集质量过滤：剔除答案在语料里不唯一的题（性别/物种等）。

规则：
  1. 答案值在语料中出现次数 > 3 的题 -> 丢弃
  2. 答案值属于常见词（female/male/human/unknown）- > 丢弃
  3. 答案长度 < 2 -> 丢弃
输出：dataset/queries_filtered.jsonl + 统计
"""
import json
import os
import re
import sys
from collections import Counter

DS = "/mnt/c/Users/user/Desktop/ds harness/data/zzz/dataset"
COMMON = {"female", "male", "human", "unknown", "yes", "no", "none"}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def main():
    docs = [json.loads(l) for l in open(os.path.join(DS, "corpus.jsonl"), encoding="utf-8")]
    blob = "\n".join(d["text"] for d in docs).lower()
    cnt = Counter()
    for d in docs:
        for tok in set(re.findall(r"\b[a-z0-9]+\b", d["text"].lower())):
            cnt[tok] += 1

    base, seen = [], set()
    for line in open(os.path.join(DS, "queries.jsonl"), encoding="utf-8"):
        q = json.loads(line)
        key = q["qid"].split("@")[0]
        if key in seen:
            continue
        seen.add(key)
        base.append(q)

    kept, dropped = [], Counter()
    for q in base:
        g = norm(q["gold"])
        if len(g) < 2:
            dropped["太短"] += 1
            continue
        if g in COMMON:
            dropped["常见值"] += 1
            continue
        # 该值在多少篇文档里出现过
        n_docs = sum(1 for d in docs if g and g in norm(d["text"]))
        if n_docs > 3:
            dropped["多文档共有"] += 1
            continue
        q["n_docs_with_answer"] = n_docs
        kept.append(q)

    print("基础问题 %d -> 保留 %d" % (len(base), len(kept)))
    print("丢弃原因：", dict(dropped))
    print("答案字段分布：", dict(Counter(q["fact_key"] for q in kept)))
    print("来源版本分布：", dict(sorted(Counter(q["answer_version"] for q in kept).items())))

    with open(os.path.join(DS, "queries_base.jsonl"), "w", encoding="utf-8") as f:
        for q in kept:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
