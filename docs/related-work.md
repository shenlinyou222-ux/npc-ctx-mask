# 相关工作对照 v2（逐篇核对）

> **核实方式**：逐篇打开 arXiv 摘要页（或正文）核对标题、作者、年份、机制。
> **未做系统检索**：arXiv 搜索接口限流、代理不稳定，因此**不保证覆盖完整**，但以下每条的机制描述都来自原文。

## 一、按"淘汰哪些 token"的动机分类

现有工作几乎都在回答同一个问题：**哪些 KV 值得留**。区别在判据：

| 工作 | 判据 | 机制 | 与我们的关系 |
|---|---|---|---|
| **StreamingLLM**（2309.17453） | 位置：开头 sink + 最近窗口 | 丢弃中间 | 按**位置**保留 |
| **H2O**（2306.14048） | 注意力累积量 | 保留 heavy hitters | 按**内容重要性** |
| **SnapKV**（2404.14469） | 头部在 prompt 末尾的注意力模式 | 每个 head 选聚类重要位置，16K 下内存省 8.2× | 按**学出来的重要性** |
| **DuoAttention**（2410.10819） | 区分 Retrieval heads 与 Streaming heads | 前者全量 KV，后者常数长度 | 按**head 类型** |
| **Star Attention**（2411.17116） | 分块：块内局部 + 全局 | 两阶段 block-sparse，省 11× 内存，保 97–100% 精度 | **机制上最像我们**：都是块级稀疏。但他们的阶段二是"query 看全部"，我们的阶段是"query 按标签看子集" |
| **CacheGen**（2310.07240） | 带宽：KV 压缩 + 流式加载 | 自定义张量编码，KV 体积降 3.5–4.3× | 支撑我们"知识库可放 NVMe"的论证 |
| **CacheBlend**（2405.16444） | 复用非前缀块 | 选择性重算约 18% token | 他们处理"块之间必须互相看见"；我们的消息**不需要** |
| **我们** | **可见性**：业务标签 | 请求级 deny-list，从 kv_indices 剔除 | 与前七者**正交**：他们问"哪些重要"，我们问"哪些被允许" |

**关键区分**：前七种的判据都是**内容相关**的（位置、注意力、head 类型、带宽），会随输入变化；我们的判据是**业务策略**（谁能看），与内容无关，且必须稳定可审计。

## 二、按"谁管理上下文"分类

| 工作 | 谁在管 | 载体 | 要不要训练 |
|---|---|---|---|
| **MemGPT**（2310.08560） | agent 自己（借操作系统分层内存） | 文本，在快慢存储间搬运 | 不要 |
| **Memento**（MSR 2026） | 模型自己（memento + 遮块） | 文本摘要 + 推理时遮原块 | **要**（三阶段 SFT） |
| **Memento-Skills**（2603.18743） | agent 自己（技能库） | markdown 文件 | 要（RL 路由） |
| **我们** | **引擎** | KV 块 + 标签 | 不要 |

## 三、Memento 给我们的两条硬信息

1. **侧信道真实存在**：memento 在块可见时生成，其 K/V 携带块内容；改成"重启重算"后 AIME'24 从 **66.1% → 50.8%**（15 个百分点）；线性探针能从下游 memento 恢复被遮块里的随机 5 位数，隔 7 块仍可检出。
2. **训练能改变结论**：他们的模型被训练成"靠 memento 承载信息"，所以 KV 通道是主干道。我们的通用模型没有这个压力——实测纯 KV 泄漏 p≈4e-5，而"看得见"是 0.63（见 `docs/` 侧信道测量）。

## 四、我们相对这些工作的定位

```
             谁决定"读什么"
        内容相关                业务策略
   ┌──────────────────┬──────────────────┐
压缩 │ SnapKV/H2O/      │                  │
     │ DuoAttention/    │                  │
     │ Star Attention   │                  │
   ──┼──────────────────┼──────────────────┤
复用 │ CacheBlend/      │      ★ 我们       │
     │ CacheGen         │                  │
   └──────────────────┴──────────────────┘
```

- 左上：**哪些重要**（模型/数据决定）
- 右上：**哪些被允许**（策略决定）——**空白**
- 左下：**怎么搬更快**
- 右下：我们

**一句话**：现有工作优化"哪些 token 值得留"，我们引入"哪些 token 被允许读"，后者是策略问题，不能由重要性替代——因为**不重要的内容也可能是不该看的**。

## 五、引用清单（已核对）

```
[1] Packer et al. MemGPT: Towards LLMs as Operating Systems. arXiv:2310.08560, 2023.
[2] Xiao et al. Efficient Streaming Language Models with Attention Sinks. arXiv:2309.17453, 2023.
[3] Zhang et al. H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs. arXiv:2306.14048, 2023.
[4] Li et al. SnapKV: LLM Knows What You are Looking for Before Generation. arXiv:2404.14469, 2024.
[5] Xiao et al. DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads. arXiv:2410.10819, 2024.
[6] Acharya et al. Star Attention: Efficient LLM Inference over Long Sequences. arXiv:2411.17116, 2025 (ICML).
[7] Liu et al. CacheGen: KV Cache Compression and Streaming for Fast LLM Serving. arXiv:2310.07240, 2024 (SIGCOMM).
[8] Yao et al. CacheBlend: Fast LLM Serving for RAG with Cached Knowledge Fusion. arXiv:2405.16444, 2024.
[9] Hsieh et al. RULER: What's the Real Context Size of Your Long-Context LMs? arXiv:2404.06654, 2024.
[10] Liu et al. Lost in the Middle: How Language Models Use Long Contexts. arXiv:2307.03172, 2023.
[11] Zhou et al. Memento-Skills: Let Agents Design Agents. arXiv:2603.18743, 2026.
[12] Microsoft Research. Memento: Teaching LLMs to Manage Their Own Context. 2026.（文章，无 arXiv 编号）
```
