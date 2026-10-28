# 原始结果

> **2026-10-28 复跑**：`zzz_bench.json`、`baroque_v2.json`、`multihop_bench_v2.json` 三份是**掩码未生效**时跑出来的
> （M 臂的越界数与 A 臂相同：60/60、5/5），已由带 `_20261028` 后缀的文件取代。旧文件保留仅供追溯。
> 对账与复现说明见 `../docs/RERUN_20261028.md`。
>
> 新文件同时修正了两个测量问题：
> ① 逐臂 TTFT 有顺序混淆（每题固定 A→M→D，A 恒冷、M/D 恒热），TTFT 改用独立四点测量（`scripts/_ttft_probe.py`）；
> ② 掩码位图按 mtime 重载，脚本"写 JSON → 立刻发请求"可能沿用上一份位图，写入后需 `sleep 0.05`。

| 文件 | 对应论文 | 说明 |
|---|---|---|
| `canary_20261028.json` | §5.1 | **复跑**：可见 8/8 · 被遮泄漏 0/8 · 正对照 8/8 · 热切换 8/8 |
| `zzz_bench_20261028.json` | §5.3 | **复跑**，180 行 = visible / hidden / gated 三臂 × 60；泄漏 58/60 → 0/60，gated 遮蔽 59/60 |
| `baroque_20261028.json` | §5.3b | **复跑**：该答 A 10/10 · M 9/10 · D 10/10；该拒答泄漏 5/5 → 0/5 |
| `multihop_20261028.json` | §7 | **复跑**：该答 6/7 = 6/7；越界臂 A 5/9 · M 4/9 · D 5/9（n=9） |
| `slot_wait_20261028.log` | §5.4 | **首次落盘**：reprefill 20/20 · wait_swap 20/20 · wait_noswap 0/20 |
| `*_20261028.log` | — | 上述三组实验的完整终端输出 |
| `canary_v3.json` | §5.1 | 遮蔽精确性，n=8（修正块偏移后的版本） |
| `canary.json` | — | **旧版**，n=4，块偏移未修正，仅存档 |
| `zzz_bench.json` | — | ⚠️ **superseded**：掩码未生效（M 泄漏 60/60） |
| `baroque_v2.json` | — | ⚠️ **superseded**：掩码未生效（M 泄漏 5/5） |
| `multihop_bench_v2.json` | — | ⚠️ **superseded**：掩码未生效（M 12/17 > A 11/17） |
| `position_gap2.json` | §5.2 | 位置间隙。0-gap = 5/6（非 6/6） |
| `pos_compress.json` | §5.2 | 位置压缩。**宽松判分**（4 字符前缀），严格口径见 ERRATA |
| `pos_stretch.json` | §5.2 | 位置拉伸（反面教材） |
| `attn_diag.json` | §5.2 | 注意力质量/熵，n=3 |
| `long16k.json` | §5.1 | 真实 16K 上下文，24/24 |
| `filler_ablation.json` | §5.3 讨论 | **严格判分**：真实填充 22/24 → 15/24；中性 24/24 → 24/24 |
| `baroque.json` | §5.3 | 51% 遮蔽；语料未随仓库发布，不可复现 |
| `hotpot_v3.json` | — | 39 条，未纳入论文 |
| `multihop_bench.json` | §7 边界 | **A 23/30 vs M 19/30**：块偏移 bug 造成的假象，见 ERRATA 末节 |
| `noise_synth.json` | — | 未纳入论文 |
| `counterfactual.json` | — | 未纳入论文（含乱码） |

## 未落盘的实验

以下实验只有终端输出，**没有原始 JSON**，复算需重跑脚本：

- `slot_subagent.py`（§5.8 子智能体真实生成 5/5·4/5·0/5）
- `audit_deferred_zh2.py`（§5.10 延迟回答，n=1）
- `audit_provenance2.py`（§5.9 来源消融）
- `audit_swap.py`（§5.4 负对照）

见 `docs/ERRATA.md` 第 8 条。
