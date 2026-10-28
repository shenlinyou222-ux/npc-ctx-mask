# 演示：一份缓存，四个视角（ZZZ 剧情进度 × KV 块可见性）

> 打开 `demo_report.html` 即可展示（自包含，无外链）。预览：`preview.png`。
> 原始数据：`demo_persona_20261028.json`（逐请求 72 条 + 聚合 24 条）。

## 设定

- 语料：**21 篇 ZZZ 角色页，3,405 token / 213 个 16-token 块**，只 prefill 一次；
- 四个角色，各自一个剧情进度（知识截止）：Banyue `2.4` / Aria `2.6` / Promeia `2.8` / Claret Flint `3.2`；
- 人物只出现在 user 轮的标签里（`【Banyue 的提问】`），**没有写"你只知道 2.4 之前的事"**——
  否则模型是靠提示拒答，不是靠掩码；
- 每人 2–4 题，每题 3 次重复（答案一致率 100%），**只有两组对照**：

| 臂 | 含义 | token 序列 |
|---|---|---|
| A | 不遮，全知 | 全文 |
| M | 遮蔽，按截止版本遮块 | **与 A 完全相同**，只有可见性位图不同 |

## 结果

| 指标 | 数值 |
|---|---|
| 越界（复现受限版本的正解） | A **7/8** → M **0/8** |
| 该答的题（4 题） | A 4/4 · M 4/4，**答案逐字一致 4/4** |
| TTFT 中位数 | A **58 ms** · M **58 ms** |
| TTOT 中位数 | A **176 ms** · M **174 ms** |
| 遮蔽比例 | 76% / 56% / 38% / 5%（随剧情进度递增而递减） |

典型对照（Banyue，截止 2.4）：

```
【Banyue 的提问】When is Roxy Ifrita Pryce's birthday?（不该答；正解 December 29th）
  不遮：December 29th     ← 会说出来
  遮蔽：July 24th         ← 读不到，答成了自己的生日
```

## 讲的时候注意

1. 只对比 **不遮 vs 遮**；遮蔽臂答错/编造是 1.7B 模型自身的问题，不是机制问题——机制保证的是"读不到"；
2. 「越界」= 复现了受限版本的正解，不做逐条人工审核；
3. 个别行（Sigrid）不遮臂也没复现正解，该行不作越界证据；
4. n 小（4 人 × 2–4 题 × 3 次），演示机制，不是基准。

## 复现

```bash
# 服务端（NPC_VIS_BLOCKS 指向位图文件；按 mtime 热加载）
NPC_VIS_BLOCKS=/home/user/_ab_scratch/npc_vis_blocks.json \
python -m vllm.entrypoints.openai.api_server --model /path/Qwen3-1.7B \
  --served-model-name q17b-flex --port 8001 --max-model-len 16384 \
  --gpu-memory-utilization 0.84 --max-num-seqs 1 --dtype bfloat16 \
  --enforce-eager --enable-prefix-caching --attention-backend FLEX_ATTENTION

python scripts/demo_persona.py --pages 21 --reps 3   # → demo_persona.json
python scripts/demo_report.py demo_persona.json demo_report.html
```

## 两个坑（已经踩过）

1. **位图 key 必须与 `request_id` 里的 `scope=` 一致**。第一次写成了固定 key `"m"`，
   而请求带的是 `scope=banyue` → 取不到黑名单 → 全可见，看起来像"掩码完全没生效"。
2. **位图要原子写**（写 `.tmp` 再 `os.replace`），写入后 sleep ~0.3 s：
   服务端按 mtime 热加载，快速重写可能读到上一份或用半份 JSON。

脚本：`scripts/demo_persona.py`（跑）、`scripts/demo_report.py`（出图）。
