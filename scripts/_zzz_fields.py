# -*- coding: utf-8 -*-
"""打印几个真实角色页的信息框字段，确认正解。"""
import json
import os
import re

DS = "/mnt/c/Users/user/Desktop/ds harness/data/zzz/dataset"
docs = {d["doc_id"]: d for d in (json.loads(l) for l in open(os.path.join(DS, "corpus.jsonl"), encoding="utf-8"))}

for did in ("st-007", "st-038", "st-071", "st-011", "st-026", "st-103", "st-181", "st-191", "st-169", "st-021"):
    d = docs[did]
    t = d["text"]
    out = {"name": d["title"], "ver": d.get("version")}
    for field in ("Height", "Birthday", "Species", "Faction", "Gender"):
        m = re.search(field + r"\s*\n\s*([^\n]{0,40})", t)
        out[field] = (m.group(1).strip() if m else "-")
    print("%-6s v%-4s %-18s | 身高 %-22s | 生日 %-14s | 种族 %-16s | 阵营 %s" % (
        did, out["ver"], out["name"], out["Height"], out["Birthday"], out["Species"], out["Faction"]))
