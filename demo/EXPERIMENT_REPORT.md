# 实验报告：一份缓存，四种可见性

> ZZZ 剧情进度 × KV 块级可见性 ｜ 2026-10-28 ｜ Qwen3-1.7B + vLLM 0.30 + FlexAttention ｜ 单卡 8 GB
> 原始数据 `demo_persona_20261028.json`，复现脚本 `scripts/demo_persona.py`

---

## 1. 一句话

**改动前**：每个角色一份 prompt，可见集合一变，整段上下文就要重新 prefill。
**改动后**：世界观只 prefill 一次，每个请求只声明"哪些 KV 块我不可读"，引擎在注意力层把块剔除。

token 序列一个字不改 → 前缀缓存照常命中；被遮的块**在 kernel 之前就从 KV 索引里剔掉**，模型物理上读不到。

---

## 2. 实验设置

| 项 | 值 |
|---|---|
| 模型 / 引擎 | Qwen3-1.7B；vLLM 0.30，`FLEX_ATTENTION`，`--enable-prefix-caching --enforce-eager` |
| 语料 | 21 篇 ZZZ 角色页 = **3,405 token = 213 个 16-token 块**，只 prefill 一次 |
| 人物 | Banyue(进度 2.4)、Aria(2.6)、Promeia(2.8)、Claret Flint(3.2) |
| 可见性 | 按剧情进度遮掉"晚于该进度"的角色页：**76% / 56% / 38% / 5%** 的块 |
| 对照 | 同一问题、同一段 token 序列，**唯一变量是可见性位图** |
| 重复 | 每格 3 次，答案一致率 100%（下表取众数） |

---

## 3. 输入到底长什么样

**system（21 页语料，3,405 token，只算一次）**

```
You answer questions about the world using only the records below.
Each record is one character page, bracketed by its title.
If the records do not contain the answer, say exactly: I don't know.

[Aria] Overview Lore Dialogue ... Height 150cm (4'11") Birthday June 7th ...
[Banyue] ... Height 200cm (6'7") Birthday July 24th ...
[Claret Flint] ... Height 166cm (5'5") Birthday July 23rd ...
...（共 21 页，每页对齐到 16-token 块边界）
```

**user（Banyue，进度 2.4）**

```
【Banyue 的提问】When is Roxy Ifrita Pryce's birthday?
（简短回答。）
```

**这一请求附带的两样东西**

```
request_id : scope=banyue
位图        : deny = [(23,33),(33,43),(53,63),(63,73), ...]   # 161 / 213 块不可读
```

不遮的那一组，只有 `request_id = scope=all` 和 `deny = []` 不同——**prompt 逐 token 相同**。

---

## 4. 四条对比（输入相同，只换位图）

| # | 提问 | 该不该答 | 正解 | 不遮的输出 | 遮蔽的输出 |
|---|---|---|---|---|---|
| 1 | Banyue 问自己生日 | 该答 | July 24th | **July 24th** | **July 24th** |
| 2 | Banyue 问 Roxy 生日 | 不该答 | December 29th | **December 29th** ← 越界 | July 24th（读不到，答成了自己的生日） |
| 3 | Aria 问 Claret 生日 | 不该答 | July 23rd | **July 23rd** ← 越界 | July 24th（读不到） |
| 4 | Claret 问 Phoenix 生日 | 不该答 | October 31st | **October 31st** ← 越界 | March 17th（读不到） |

第 1 行是"该知道的照样知道"，第 2–4 行是"不该知道的说不出来"——**同一个模型、同一段上下文、同一个 prompt**。

---

## 5. 汇总指标

| 指标 | 不遮 | 遮蔽 |
|---|---|---|
| 越界（复现受限版本正解） | **7 / 8** | **0 / 8** |
| 该答的题（4 题） | 4/4 | 4/4（**答案逐字一致 4/4**） |
| TTFT 中位数（首 token） | **58 ms** | **58 ms** |
| TTOT 中位数（末 token，端到端） | **176 ms** | **174 ms** |
| prompt token | 3,441–3,447 | 3,441–3,447（完全相同） |

---

## 6. 前后对比：快在哪

| | 做法 | TTFT（3,405 token 语料） |
|---|---|---|
| **改动前** | 换角色 = 重拼 prompt = 整段重新 prefill | **2.04 s**（n=3：1.16 / 2.04 / 4.35） |
| **改动后** | 序列不变，只换位图 = 前缀缓存命中 | **0.20 s**（n=3：0.16 / 0.20 / 0.22） |

约 **10×**，而且**换可见性的那一下是 0 成本**（遮蔽 58 ms vs 不遮 58 ms），
因为"能读什么"只改位图，不改 token 序列。

---

## 7. 这套东西的边界（面试时主动说，反而加分）

1. **遮蔽只保证"读不到"，不保证"说不知道"**。第 2–4 行里遮蔽臂照样开口，只是编了个别的日期。
   要它承认不知道，是另一个（提示层/后训练层）的问题。
2. **1.7B 模型本身定位弱**：21 页里有个别行不遮也答错（Sigrid），那几行不作越界证据。
3. **样本小**（4 人 × 2–4 题 × 3 次），这是机制演示，不是 benchmark。
4. 16-token 块粒度以下管不了；块边界要按渲染后的序列算（踩过 3-token 偏移的坑）。

---

## 8. 复现

```bash
# 引擎：打了请求级块可见性补丁的 vLLM（5 文件 / +211 行）
NPC_VIS_BLOCKS=/path/blocks.json python -m vllm.entrypoints.openai.api_server \
  --model Qwen3-1.7B --served-model-name q17b-flex --port 8001 --max-model-len 16384 \
  --gpu-memory-utilization 0.84 --max-num-seqs 1 --dtype bfloat16 \
  --enforce-eager --enable-prefix-caching --attention-backend FLEX_ATTENTION

python scripts/demo_persona.py --pages 21 --reps 3    # → demo_persona.json
python scripts/demo_report.py demo_persona.json demo_report.html
```

可视化版本：`demo_report.html`（角色卡 + 逐题对照 + 延迟图 + 现场片段）。
