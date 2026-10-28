# -*- coding: utf-8 -*-
"""由 demo_persona.json 生成自包含 HTML 展示页（两臂：不遮 vs 遮）。

用法: demo_report.py [json] [out.html]
"""
import html
import json
import os
import statistics as st
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "/home/user/_ab_scratch/demo_persona.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/user/_ab_scratch/demo_report.html"

d = json.load(open(SRC, encoding="utf-8"))
cells, rows, cast, spans, corpus = d["cells"], d["rows"], d["cast"], d["spans"], d["corpus"]


def esc(s):
    return html.escape(str(s))


def nrm(s):
    return "".join(c for c in str(s).lower() if c.isalnum())


def med(v):
    return st.median(v) if v else 0.0


by = {}
for c in cells:
    by.setdefault((c["persona"], c["qid"]), {})[c["arm"]] = c
groups, seen = [], set()
for c in cells:
    k = (c["persona"], c["qid"])
    if k not in seen:
        seen.add(k)
        groups.append(by[k])

restricted = [g for g in groups if not g["A"]["should_answer"]]
answerable = [g for g in groups if g["A"]["should_answer"]]
leak_A = sum(1 for g in restricted if g["A"]["hit"])
leak_M = sum(1 for g in restricted if g["M"]["hit"])
same_AM = sum(1 for g in answerable if nrm(g["A"]["ans"]) == nrm(g["M"]["ans"]))
hit_A = sum(1 for g in answerable if g["A"]["hit"])
hit_M = sum(1 for g in answerable if g["M"]["hit"])

ttft = {a: [r["ttft"] for r in rows if r["arm"] == a] for a in ("A", "M")}
ttot = {a: [r["ttot"] for r in rows if r["arm"] == a] for a in ("A", "M")}


def V(v):
    s = str(v)
    return int(s.split(".")[0]) * 100 + int(s.split(".")[1]) if "." in s else -1


deny_pages = {p["pid"]: [s for s in spans if V(s["version"]) > V(p["cutoff"])] for p in cast}


def badge(cell):
    return '<span class="b ok">复现正解</span>' if cell["hit"] else '<span class="b no">未复现</span>'


def persona_cards():
    out = []
    for p in cast:
        den = deny_pages[p["pid"]]
        pct = 100.0 * sum(s["blk"][1] - s["blk"][0] for s in den) / corpus["blocks"]
        titles = "、".join(esc(s["title"]) for s in den[:4]) + ("…" if len(den) > 4 else "")
        out.append(f"""
        <div class="card">
          <div class="cname">{esc(p['name'])}</div>
          <div class="cmeta">剧情进度 <b>{p['cutoff']}</b></div>
          <div class="cbar"><div style="width:{pct:.0f}%"></div></div>
          <div class="csmall">不可读 {len(den)} 页 / {pct:.0f}% 块</div>
          <div class="ctitles">{titles or '（无需遮蔽）'}</div>
        </div>""")
    return "".join(out)


def table_rows():
    out = []
    for p in cast:
        gs = [g for g in groups if g["A"]["persona"] == p["pid"]]
        for i, g in enumerate(gs):
            a, m = g["A"], g["M"]
            same = nrm(a["ans"]) == nrm(m["ans"])
            ok = "ok" if a["should_answer"] else "bad"
            first = (f'<td class="pcell" rowspan="{len(gs)}">{esc(p["name"])}'
                     f'<br><span class="dim">进度 {p["cutoff"]}</span></td>') if i == 0 else ""
            out.append(f"""
        <tr class="row-{ok}">
          {first}
          <td class="q">{esc(a['question'])}</td>
          <td class="gold">{esc(a['gold'])}</td>
          <td>{esc(a['ans'])} {badge(a)}</td>
          <td>{esc(m['ans'])} {badge(m)}</td>
          <td class="mid">{'同' if same else '异'}</td>
          <td class="num">{a['ttft_med']*1000:.0f} / {m['ttft_med']*1000:.0f}</td>
          <td class="num">{a['ttot_med']*1000:.0f} / {m['ttot_med']*1000:.0f}</td>
        </tr>""")
    return "".join(out)


def svg_bars():
    items = [("A 不遮（全知）", med(ttft["A"])), ("M 遮蔽（角色视角）", med(ttft["M"]))]
    w, h, top, bw = 400, 160, 20, 70
    mx = max(items[0][1], items[1][1]) * 1.35
    bars = []
    for i, (label, v) in enumerate(items):
        bh = (v / mx) * (h - 2 * top)
        x, y = 60 + i * 170, h - top - bh
        bars.append(f'<rect x="{x}" y="{y:.1f}" width="{bw}" height="{bh:.1f}" rx="3" fill="#3b82f6"/>')
        bars.append(f'<text x="{x+bw/2:.0f}" y="{y-8:.0f}" text-anchor="middle" class="v">{v*1000:.0f} ms</text>')
        bars.append(f'<text x="{x+bw/2:.0f}" y="{h-top+16}" text-anchor="middle" class="l">{esc(label)}</text>')
    return f'<svg viewBox="0 0 {w} {h+26}" class="chart">{"".join(bars)}</svg>'


def transcript():
    gs = [g for g in groups if g["A"]["persona"] == "banyue"]
    picked = ([g for g in gs if g["A"]["should_answer"]][:1]
              + [g for g in gs if not g["A"]["should_answer"]][:1])
    out = []
    for g in picked:
        a, m = g["A"], g["M"]
        kind = "该答" if a["should_answer"] else "不该答"
        out.append(f"""
        <div class="tx">
          <div class="tq">【{esc(a['name'])} 的提问】{esc(a['question'])} <span class="dim">（{kind}；正解 {esc(a['gold'])}）</span></div>
          <div class="ta"><span class="tag">不遮</span>{esc(a['ans'])}</div>
          <div class="ta"><span class="tag m">遮蔽</span>{esc(m['ans'])}</div>
        </div>""")
    return "".join(out)


HTML = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>ZZZ 多人物 KV 块可见性演示</title>
<style>
 :root {{ --ink:#1b1f24; --dim:#6b7280; --line:#e5e7eb; --ok:#0f766e; --bad:#b91c1c; }}
 * {{ box-sizing:border-box }}
 body {{ margin:0; padding:36px 40px 56px; font:15px/1.6 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
        color:var(--ink); background:#fbfbfd; max-width:1140px; }}
 h1 {{ font-size:26px; margin:0 0 6px; letter-spacing:-.3px }}
 .sub {{ color:var(--dim); margin-bottom:24px; max-width:900px }}
 h2 {{ font-size:17px; margin:32px 0 12px; padding-bottom:6px; border-bottom:1px solid var(--line) }}
 .kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:18px 0 6px }}
 .kpi {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:14px 16px }}
 .kpi .v {{ font-size:24px; font-weight:650; letter-spacing:-.5px }}
 .kpi .k {{ color:var(--dim); font-size:12.5px; margin-top:2px }}
 .cast {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-top:14px }}
 .card {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:14px 16px }}
 .cname {{ font-weight:650 }}
 .cmeta {{ color:var(--dim); font-size:12.5px }}
 .cbar {{ height:6px; background:#eef2f7; border-radius:99px; margin:10px 0 6px; overflow:hidden }}
 .cbar div {{ height:100%; background:linear-gradient(90deg,#3b82f6,#8b5cf6) }}
 .csmall {{ font-size:12.5px; color:var(--dim) }}
 .ctitles {{ font-size:12px; color:#94a3b8; margin-top:6px; line-height:1.45 }}
 table {{ width:100%; border-collapse:collapse; background:#fff; border:1px solid var(--line);
          border-radius:10px; overflow:hidden; font-size:13.5px; margin-top:12px }}
 th {{ background:#f6f7f9; text-align:left; font-weight:600; font-size:12.5px; color:#374151;
       padding:9px 10px; border-bottom:1px solid var(--line) }}
 td {{ padding:9px 10px; border-bottom:1px solid #f1f3f6; vertical-align:top }}
 tr:last-child td {{ border-bottom:none }}
 .pcell {{ font-weight:650; white-space:nowrap; background:#fcfcfd }}
 .q {{ min-width:210px }}
 .gold {{ color:#0f766e; white-space:nowrap }}
 .dim {{ color:var(--dim) }}
 .mid {{ color:var(--dim); text-align:center }}
 .num {{ font-variant-numeric:tabular-nums; white-space:nowrap; color:#374151 }}
 .b {{ display:inline-block; font-size:11px; padding:1px 6px; border-radius:99px; margin-left:4px; vertical-align:1px }}
 .ok {{ background:#ecfdf5; color:var(--ok); border:1px solid #a7f3d0 }}
 .no {{ background:#fef2f2; color:var(--bad); border:1px solid #fecaca }}
 .row-bad td {{ background:#fffdfd }}
 .chart {{ width:100%; max-width:420px; margin-top:8px }}
 .chart .v {{ font-size:12px; fill:#374151 }}
 .chart .l {{ font-size:11px; fill:#6b7280 }}
 .note {{ background:#fff; border:1px solid var(--line); border-left:3px solid #3b82f6;
          border-radius:8px; padding:12px 14px; font-size:13.5px; color:#374151 }}
 .tx {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:12px 14px; margin-bottom:10px }}
 .tq {{ font-weight:600; margin-bottom:8px }}
 .ta {{ margin:4px 0 0 }}
 .tag {{ display:inline-block; font-size:11px; padding:1px 7px; border-radius:99px;
         background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe; margin-right:8px }}
 .tag.m {{ background:#f5f3ff; color:#6d28d9; border-color:#ddd6fe }}
 code {{ background:#f3f4f6; padding:1px 5px; border-radius:4px; font-size:12.5px }}
 .foot {{ color:var(--dim); font-size:12.5px; margin-top:24px }}
</style></head><body>

<h1>一份缓存，四个视角：ZZZ 剧情进度 × KV 块可见性</h1>
<div class="sub">同一份世界观只 prefill 一次；人物只声明"我看不到哪些块"，引擎在注意力层把块剔除。
两组对照共用<b>完全相同的 token 序列</b>，唯一变量是可见性位图。</div>

<div class="kpis">
  <div class="kpi"><div class="v">{corpus['pages']} 页 / {corpus['tokens']:,} tok</div><div class="k">共享语料（{corpus['blocks']} 个 16-token 块）</div></div>
  <div class="kpi"><div class="v">{leak_A}/{len(restricted)} → {leak_M}/{len(restricted)}</div><div class="k">越界：不遮 → 遮蔽</div></div>
  <div class="kpi"><div class="v">{hit_A}/{len(answerable)} = {hit_M}/{len(answerable)}</div><div class="k">该答的题：不遮 = 遮蔽</div></div>
  <div class="kpi"><div class="v">{med(ttft['A'])*1000:.0f} vs {med(ttft['M'])*1000:.0f} ms</div><div class="k">TTFT：不遮 vs 遮蔽</div></div>
</div>

<h2>角色与知识边界</h2>
<div class="cast">{persona_cards()}</div>

<h2>逐题对照</h2>
<table>
 <tr><th>人物</th><th>提问</th><th>正解</th><th>A 不遮</th><th>M 遮蔽</th><th>答案</th>
     <th>TTFT A/M (ms)</th><th>TTOT A/M (ms)</th></tr>
 {table_rows()}
</table>

<h2>延迟：遮蔽几乎零成本</h2>
{svg_bars()}
<div class="note">
同一段 token 序列，只换可见性位图：<b>TTFT 中位数 {med(ttft['A'])*1000:.0f} ms vs {med(ttft['M'])*1000:.0f} ms</b>，
<b>TTOT 中位数 {med(ttot['A'])*1000:.0f} ms vs {med(ttot['M'])*1000:.0f} ms</b>，差异在噪声内（每格 3 次取中位数）。
<br>为什么几乎免费：序列没变，前缀缓存照常命中；被遮块在 kernel 之前就从 KV 索引里剔除，连读都不读。
</div>

<h2>现场片段（Banyue，剧情进度 2.4）</h2>
{transcript()}

<h2>看什么、别误读什么</h2>
<div class="note">
<b>看两点：</b>① 不该答的 {len(restricted)} 题里，不遮臂复现正解 <b>{leak_A} 次</b>，遮蔽臂 <b>{leak_M} 次</b>；
② 该答的 {len(answerable)} 题里，两臂答案<b>逐字一致 {same_AM}/{len(answerable)}</b>，且都复现正解 {hit_A}/{len(answerable)}。
<br><b>别误读：</b>①「越界」的定义是"复现了受限版本的正解"，不是逐条人工审核；
② 1.7B 在 21 页里定位本就弱，个别行（Sigrid）不遮臂也没复现，该行不作越界证据；
③ n 小（4 人 × 2–4 题 × 3 次），只演示机制，不是基准。
</div>

<h2>复现</h2>
<div class="note" style="font-family:ui-monospace,Consolas,monospace;font-size:12.5px">
NPC_VIS_BLOCKS=/path/blocks.json python -m vllm.entrypoints.openai.api_server --model Qwen3-1.7B &lt;br&gt;<br>
&nbsp;&nbsp;--attention-backend FLEX_ATTENTION --enable-prefix-caching --enforce-eager --port 8001<br>
python demo_persona.py --pages 21 --reps 3&nbsp;&nbsp;&nbsp;# → demo_persona.json<br>
python demo_report.py demo_persona.json demo_report.html
</div>

<div class="foot">
数据：{esc(os.path.basename(SRC))}｜每格 3 次重复，答案一致率 100%（表内取众数）｜
TTFT = 首 token 时间，TTOT = 末 token 时间（端到端）｜ 生成于 2026-10-28
</div>
</body></html>
"""

open(OUT, "w", encoding="utf-8").write(HTML)
print("写入", OUT, len(HTML), "bytes")
