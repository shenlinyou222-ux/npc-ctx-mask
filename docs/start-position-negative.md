# 负结果：给请求加「起始位置偏移」在 vLLM 里会崩

## 想做什么

子智能体要把总结写在主序列预留的槽位 `[P, P+L)`。最直觉的做法是给请求一个 `start_position`，让它的 prefill 从 P 开始，这样 K/V 天然落在槽位位置上，不需要任何重旋转。

## 怎么做的

在 `model_runner.py` 的 `prepare_pos_seq_lens()` 之后加了一个钩子（`NPC_START_POS=1` 开启）：

```python
offs = _npc_vis.start_offsets(list(req_ids[:num_reqs]))   # "scope=name@P" -> [P, ...]
for r, off in enumerate(offs):
    if off:
        s, e = int(query_start_loc_np[r]), int(query_start_loc_np[r + 1])
        self.input_buffers.positions[s:e] += off
```

## 结果

| 请求 | 结果 |
|---|---|
| `scope=plain` | `MSK-9902` ✔ 正常 |
| `scope=off1@1000` | 输出乱码 `1.\n\n\n\nMS\n...` |
| `scope=off2@20000` | HTTP 500，**CUDA illegal memory access** |
| 之后任何请求 | 500（引擎已死） |

引擎日志：

```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

## 为什么

vLLM 的 KV cache 是**分页**的，块表按 `position // block_size` 索引：

```
position 1000  ->  block 62
```

但这条请求只分配了 `ceil(len/16)` 个块（比如 8 个）。偏移之后，写 KV 时去访问第 62 块——那块根本不属于它，于是越界。

**位置不是空闲空间**。把位置往前推，不会自动把中间的块补出来。

## 正确的做法（现在没有，但方向明确）

要做的是**往已存在的块里写 K/V**，而不是改位置：

| 方案 | 说明 | 现状 |
|---|---|---|
| **A. 槽位实体化** | 主序列里放 L 个占位 token，块真实分配；子智能体产出总结后，把总结的 K/V 写进这些**已分配**的块 | 需要「块级 KV 注入」接口，vLLM 没有 |
| **B. 文本重 prefill** | 总结作为文本拼回，整段重 prefill | ✅ 可行（我们的 fork/splice 实验证明与 KV 拼接等价） |
| **C. 预分配位置区间** | 生成时就给每个分支分配不重叠的位置，且**同时分配对应块** | 需要引擎同时支持位置与块分配，改动大 |

## 结论

1. 「指定起始位置」这个想法**单独不成立**——它必须和「预分配块」绑定，否则越界。
2. 在现有引擎上，方案 B（文本重 prefill）是唯一稳的路径，而且我们的实验已经证明它与 KV 拼接**行为等价**（20/20 = 20/20）。
3. 方案 A 是真正想要的，但需要一个新原语：**把一个已完成请求的 K/V 写进另一个请求的指定块**。这比掩码大得多，属于「块级 KV 注入」。

## 附带发现（有用的）

掩码本身**支持运行时切换且不需要重启**：同一个进程里，不同 `scope` 可以有不同可见集，改 JSON 后下一条请求立即生效（`_load_cfg` 按 mtime 重读）。

实测（同一份 JSON 里两个 scope）：

```
scope=open  B -> 'MSK-9902'   读到 B=True
scope=deny  B -> 'VIS-3355'   读到 B=False
scope=open  B -> 'MSK-9902'   读到 B=True
```

所以「子智能体返回后把槽位置为可见」这一步，在引擎侧**已经能做**；缺的只是"把 K/V 写进槽位"。
