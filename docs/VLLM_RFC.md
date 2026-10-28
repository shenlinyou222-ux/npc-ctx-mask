# vLLM RFC 讨论稿：请求是否可以自行申请"掩码上下文块"

> 用途：在 `vllm-project/vllm` 的 Discussions / Issue 发帖。
> 姿态：**先问必要性，再给实现**。不要一上来就贴 patch 求合并——先说清问题，让维护者判断这个能力该不该在引擎层。

---

## 标题（二选一）

- `[RFC] Should a request be able to mask KV blocks it is not allowed to read?`
- `[Feature] Per-request KV block visibility (deny list) for multi-tenant / role-scoped RAG`

---

## 正文

### 问题

在多租户或角色扮演 RAG 里，同一份语料要按"谁在问"给出不同可见集合。现在有三种做法，都有代价：

| 做法 | 代价 |
|---|---|
| 每次重拼可见文档 | token 序列变 → 前缀缓存失效，每次重 prefill |
| 提示词写"不要提及 X" | 软约束，模型知道内容，可被诱导 |
| 每个租户一份缓存 | 显存/磁盘线性膨胀 |

我们试了第四条：**token 序列完全不动，只让请求声明"哪些 KV 块我不可读"**。

### 我们验证过什么（单卡 8GB，Qwen3-1.7B，vLLM 0.30 + FlexAttention）

- **屏蔽是硬的**：同一上下文里，可见块的 canary 读得到 8/8，被屏蔽块 0/8；与物理删除行为一致（正负对照都有）。
- **质量不变**：屏蔽 24%–97% 的块，准确率与不屏蔽、与删除文本三者一致。
- **快 18–20×**：7.5K 上下文冷 1.27 s → 热 0.062 s（token 序列不变，前缀缓存命中）；**换可见性零惩罚**（0.064 s），换文本才有代价（中段删 17% 块 → 0.515 s）。
- 另外测到一条独立的东西：**位置跨度有预算**。把问题推到 80K 外（KV 只有 300 token），检索精度掉到 50%，160K 完全失效；对位置轴做压缩（160K→40K 有效）可完全恢复。这条与屏蔽无关，但影响"能装多少"。

### 实现（很小）

我们做了一个 demo 级 patch（5 文件 / +211 行）：

1. `AttentionMetadata` 加一个 `npc_req_ids: list[str] | None`；
2. worker 侧从 `input_batch.req_ids` 填充；
3. FlexAttention 构建 BlockMask 前，把该请求声明为不可见的块从 `kv_indices` 剔除；
4. 可见性来自环境变量指向的 JSON（`{scope: [[start, end), ...]}`，deny list）。

patch、脚本、原始结果：<repo>

### 想请教的问题

1. **该不该在引擎层？** 还是说这属于应用层（比如 KV connector / 自定义 attention backend）的职责？
2. **是否已有等价能力？** 我可能漏了现有机制：prefix-cache salting、KV cache groups、`kv_transfer`、或者 per-request block table 的现有接口。如果有，能否指个路？
3. **接口该怎么长**？现在用 `request_id` 里的魔法字符串 + 环境变量，显然不是 upstream 能接受的形态。更合理的是 `SamplingParams` 上的一个字段，还是 `Request` 层？
4. **块粒度够用吗**？FlexAttention 强制 16-token 块，所以"隐藏"的最小单位是 16 token。实际场景里是否必须 token 级？
5. **有没有人需要相反语义**（allow list，只允许读这些块）？我们试过 allow list，会把模型自己新生成的块也挡掉，输出退化——所以强烈建议 deny list 语义。

### 我们不主张什么

- 屏蔽**不**提升准确率（随机干扰下我们没测出差异）；
- 屏蔽**不**让模型拒答（模板删文本时拒答 5/6，屏蔽时 0/6）；
- 位置预算只在一个模型上测过。

---

## 发帖注意事项

- 一个帖子里**只问 1–2 个核心问题**（建议：该不该在引擎层 + 是否已有等价能力）。
- patch 作为附件/链接，不要贴全文。
- 如果维护者说"这属于应用层"，那也是有用结论：**记录下来**，说明我们尝试过引擎路线、被明确指到应用层。
- 不要在标题里写 "novel" / "first" —— 我们没做完整的相关工作检索。
