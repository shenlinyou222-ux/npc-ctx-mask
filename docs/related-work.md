# 相关工作对照（仅限已核实条目）

> **声明**：这不是系统检索。arXiv 搜索接口当时限流、代理时断时续，以下每一条都逐篇打开核对了标题与摘要（或正文），但**不保证覆盖完整**。
> 核实程度：`正文` = 读过正文/摘要全文；`摘要` = 读过 arXiv 摘要页；`引用` = 仅从他人论文的引用中得知。

## 一、上下文/记忆管理（agent 层）

| 工作 | 机制 | 与我们的关系 | 核实 |
|---|---|---|---|
| **MemGPT**（arXiv 2310.08560） | 借用操作系统分层内存：在快/慢存储间搬运上下文，用中断控制流程，让模型"以为"有更大窗口 | 同样把上下文当资源管理，但**在文本层**、由 agent 驱动；我们在 **KV 块层**、由引擎驱动，且不改变 token 序列 | 正文 |
| **Memento**（Microsoft Research, 2026） | 把推理链切块，每块生成 *memento*，推理时遮住原块（连 KV 一起删）；用三阶段 SFT 训练模型适应 | 与我们最接近的**机制**：都在推理时遮块。差别是**他们训练模型、我们不训练**；且他们遮的是"自己刚生成的推理块"，我们遮的是语料块 | 正文 |
| **Memento-Skills**（arXiv 2603.18743） | 技能存成 markdown 文件作为持久记忆，用 RL 训练的 router 选择，不更新参数 | 记忆载体是文本文件，选择靠训练；我们靠标签 | 摘要 |

**Memento 给我们的最重要一条**：他们报告，去掉"memento 从其原块继承的 KV 信息"（改为重启重算）后，AIME'24 从 66.1% 掉到 50.8%（**15 个百分点**）；线性探针还能从下游 memento 的 KV 中恢复被遮块里的随机 5 位数，隔 7 个块仍可检出。

这说明：**遮蔽之后，信息仍可能经由"遮蔽前生成的 token"的 K/V 存活**。我们自己的测量见 §三。

## 二、KV 复用与稀疏（引擎层）

| 工作 | 机制 | 与我们的关系 | 核实 |
|---|---|---|---|
| **CacheBlend**（arXiv 2405.16444） | 复用非前缀文本块的 KV，**选择性重算约 18% 的 token** 以恢复跨块注意力；TTFT 降 2.2–3.3× | 他们处理"块之间必须互相看见"的情形（RAG 文档）；我们的分支消息**不需要**互相看见，因此不需要重算 | 正文 |
| **StreamingLLM**（arXiv 2309.17453） | 保留注意力汇（sink）+ 最近窗口，实现流式长上下文 | 按**注意力位置**保留；我们按**业务标签**保留 | 摘要 |
| **H2O**（arXiv 2306.14048） | 按注意力重要性保留 heavy hitters | 同上，按**内容重要性**淘汰；我们按**可见性** | 摘要 |
| **RULER**（arXiv 2404.06654） | 定义并测量"有效上下文长度" | 我们的"位置预算"是它的低成本标定变体（§5.2） | 摘要 |
| **Lost in the Middle**（arXiv 2307.03172） | 报告长上下文中"中间位置"性能下降（U 形） | 我们测的是"证据与问题的距离"，单调衰减；**不是同一个变量** | 摘要 |

## 三、我们自己的侧信道测量（与 Memento 同一问题的不同结论）

| 条件 | p(答案首 token) | 平均排名 |
|---|---|---|
| 文档可见（上限） | 0.633 | 1 |
| 挖洞，无过洞 token | 4.4e-5 | 871 |
| 挖洞，1–4 个过洞 token（纯 KV 样本） | 3.6–4.1e-5 | 468–534 |
| 挖洞，≥16 个过洞 token | **模型自己把答案写进了文本**（5/5） | — |

结论：

1. **纯 KV 侧信道在我们的设置下与基线同量级**（4e-5），远低于"看得见"的 0.63——差四个数量级；
2. 一旦模型"想起来"，它倾向于**直接写出来**，于是变成**文本泄漏**，而不是隐蔽的 KV 通道；
3. 与 Memento 的 15 个百分点不同，他们的模型**被训练成靠 memento 承载信息**，所以 KV 通道是主干道；我们的通用模型没有这个压力。

**限制**：本测量是 Qwen3-1.7B、单跳事实、过洞 token ≤128。长链、训练过的模型、或更大的模型可能不同。

## 四、我们还缺的对照

- **块级检索**：MemGPT / Memento 都有"该取什么"的策略；我们目前是硬编码标签，没有检索；
- **多槽位**：所有实验只有一个槽位；
- **训练过的小模型 vs 通用模型**：Memento 的对照提示，"是否被训练来使用这种机制"会显著改变结论。

## 引用清单（供正文使用）

```
[1] Packer et al. MemGPT: Towards LLMs as Operating Systems. arXiv:2310.08560, 2023.
[2] Yao et al. CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion. arXiv:2405.16444, 2024.
[3] Xiao et al. Efficient Streaming Language Models with Attention Sinks. arXiv:2309.17453, 2023.
[4] Zhang et al. H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models. arXiv:2306.14048, 2023.
[5] Hsieh et al. RULER: What's the Real Context Size of Your Long-Context Language Models? arXiv:2404.06654, 2024.
[6] Liu et al. Lost in the Middle: How Language Models Use Long Contexts. arXiv:2307.03172, 2023.
[7] Zhou et al. Memento-Skills: Let Agents Design Agents. arXiv:2603.18743, 2026.
[8] Microsoft Research. Memento: Teaching LLMs to Manage Their Own Context. 2026. (文章，无 arXiv 编号)
```
