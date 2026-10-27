# 外联清单与渠道

> 原则：**优先公开渠道**（GitHub issue / discussion），其次邮件。冷邮件给知名研究者的回复率很低，但公开 issue 至少能给项目留痕迹、也能被搜索引擎索引。

---

## 第一梯队：引擎侧（决定这个 API 该不该存在）

| 对象 | 为什么找他 | 渠道 |
|---|---|---|
| **vLLM 团队**（Woosuk Kwon / Simon Mo / Lianmin Zheng） | 我们的 patch 就是给他们引擎提的接口需求；他们决定"块级可见性"是应用层还是引擎层 | **GitHub issue**（`vllm-project/vllm`，标签 `feature request` / `discussion`）——最推荐 |
| **Lianmin Zheng**（同时是 H2O 作者） | 既懂引擎又懂 KV 压缩，最可能看懂"块选择"的两面性 | GitHub / 论文邮箱 |

**建议动作**：开一个 issue，标题类似
`[Feature] Per-request KV block visibility (allow/deny list) for multi-tenant RAG`
正文用 `docs/CALL_FOR_FEEDBACK.md` 的前两节 + 五个问题。

---

## 第二梯队：方法论（位置预算）

| 对象 | 相关工作 | 我们的关联 |
|---|---|---|
| **Cheng-Ping Hsieh / NVIDIA RULER 团队** | RULER: What's the Real Context Size（arXiv 2404.06654） | 我们的"位置预算 1 分钟标定法"是他们指标的极低成本变体 |
| **Simeng Sun / Samuel Kriman**（同组） | 同上 | 同上 |

**问的问题**：用 ~300 token 的 KV、只推远问题位置，是否足以代理"有效上下文长度"？

---

## 第三梯队：KV 压缩 / 稀疏（技术近邻）

| 对象 | 相关工作 | 我们的关联 |
|---|---|---|
| **Guangxuan Xiao / Song Han** | StreamingLLM（2309.17453） | 我们的注意力诊断复现了 sink 现象（开头 token 吸走 50% 质量） |
| **Zhenyu Zhang / Ying Sheng** | H2O（2306.14048） | 他们是"按重要性淘汰块"，我们是"按标签隐藏块"，可组合 |
| **Yuhong Li**（NVIDIA） | SnapKV（2404.14469） | 同样是块/位置的取舍 |
| **DeepSeek-AI** | DeepSeek-V2/V4 的 MLA + CSA2 | 他们的 `candidate_block_size=8`、`kv_source_layer_ids` 说明块级控制已是原生设计 |

**问的问题**：按业务标签隐藏块，与按注意力重要性淘汰块，是否应该共用一套块元数据？

---

## 邮件模板（短，控制在一屏内）

```
主题：[feedback] request-level KV block visibility for RAG (patch + measurements)

您好 <name>，

我读了你关于 <paper> 的工作，想请教一个方向性的问题。

我做了一个 vLLM 0.30 + FlexAttention 的小 patch：让每个请求声明"哪些 KV 块不可见"，
这样同一份 prefill 缓存可以被不同角色/租户复用。实测：
- 屏蔽是硬的（被遮块的 canary 0/8 读不到，可见块 8/8）；
- 质量不变（屏蔽 24–97% 块，准确率与删除文本、与不遮都一致）；
- TTFT 0.98s → 0.06s（token 序列不变，前缀缓存命中）。

代码与全部实验：<repo>

我最不确定的两点：
1) 这个能力应该放在引擎（块级 allow/deny list）还是应用层？
2) 我用"把问题推到 80K 外、KV 只有 300 token"来测位置预算，这个方法可靠吗？

如果方向对，能否给一点建议？另外我在找 cs.CL 的 arXiv 背书（首次投稿），
如果你觉得这份报告值得发，endorsement code 可以在 arxiv.org/auth/endorse 输入。

谢谢！
<your name>
```

---

## 实操建议

1. **先开 GitHub issue 再发邮件**——邮件里可以直接附 issue 链接，回复率更高。
2. **一次只问两个问题**：引擎归属 + 标定方法。问题多了没人回。
3. **不要在邮件里说"首次提出"**——相关工作没核实过。
4. **准备好被拒绝**：多数人不会回。目标是拿到 1–2 条实质性反馈，而不是"被认可"。
