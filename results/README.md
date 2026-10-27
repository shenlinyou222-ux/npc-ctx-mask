# 原始结果

| 文件 | 对应论文 | 说明 |
|---|---|---|
| `canary_v3.json` | §5.1 | 遮蔽精确性，n=8，逐轮记录（修正块偏移后的版本） |
| `canary.json` | — | **旧版**，n=4，块偏移未修正，仅存档 |
| `zzz_bench.json` | §5.3 | 60 题版本可见性。**注意**：visible 条件遮蔽比例为 0% |
| `position_gap2.json` | §5.2 | 位置间隙。0-gap = 5/6（非 6/6） |
| `pos_compress.json` | §5.2 | 位置压缩。**宽松判分**（4 字符前缀），严格口径见 ERRATA |
| `pos_stretch.json` | §5.2 | 位置拉伸（反面教材） |
| `attn_diag.json` | §5.2 | 注意力质量/熵，n=3 |
| `long16k.json` | §5.1 | 真实 16K 上下文，24/24 |
| `filler_ablation.json` | §5.3 讨论 | **严格判分**：真实填充 22/24 → 15/24；中性 24/24 → 24/24 |
| `baroque.json` | §5.3 | 51% 遮蔽；语料未随仓库发布，不可复现 |
| `hotpot_v3.json` | — | 39 条，未纳入论文 |
| `multihop_bench.json` | §7 边界 | **A 23/30 vs M 19/30**：被遮内容与问题相关时遮蔽会掉点，是重要的反例 |
| `noise_synth.json` | — | 未纳入论文 |
| `counterfactual.json` | — | 未纳入论文（含乱码） |

## 未落盘的实验

以下实验只有终端输出，**没有原始 JSON**，复算需重跑脚本：

- `slot_wait.py`（§5.4 槽位热替换 20/20）
- `slot_subagent.py`（§5.8 子智能体真实生成 5/5·4/5·0/5）
- `audit_deferred_zh2.py`（§5.10 延迟回答，n=1）
- `audit_provenance2.py`（§5.9 来源消融）
- `audit_swap.py`（§5.4 负对照）

见 `docs/ERRATA.md` 第 8 条。
