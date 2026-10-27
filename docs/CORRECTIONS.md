# 修正记录：块区间必须按「渲染后的序列」计算

## 错误

之前所有实验都用**原始字符串**算 token 位置，再除以 16 得到块号：

```python
a = ceil(ntok(head + body) / 16)
```

但实际请求经过 chat template，会在 system 内容**前面插入 3 个 token**：

```
raw(system)     = 47 token
rendered(chat)  = 69 token（含 <|im_start|>system\n 3 个 + 尾部模板）
偏移            = 3 token
```

于是所有块号都偏移一格。某个块被算成 `(6,7)` 实际对应 `(7,8)`，遮错块。

## 影响范围

| 实验 | 是否受影响 | 说明 |
|---|---|---|
| `verify_mask_v3.py`（新） | — | 按渲染后序列定位，已修正 |
| 热切换验证（旧 `reload_test.py`） | **是，结论错误** | 当时报告“未生效”，实际是遮错了块 |
| `ab_scope.py` | 是（侥幸正确） | 偏移后恰好仍覆盖目标块 |
| `canary_test.py` | 轻微 | deny 覆盖连续 4 块，偏一格仍覆盖 MSK，结论不变 |
| ZZZ / Baroque / HotpotQA 三臂 | 轻微 | deny 覆盖整篇文档的多个块，偏一格只影响边界块；且结果是不泄漏（0/60），掩盖更强而非更弱 |
| 位置/注意力实验 | 否 | 不涉及块区间 |

## 修正后的验证（8/8 全对）

```
chat template 前缀偏移 = 3 token
块区间: VIS(2,3) FILL(5,7) MSK(7,8)

visible 读到B   8/8
masked  读到A   8/8   (正对照)
masked  泄漏B   0/8
open    读到B   8/8   (热切换)
```

复现：`python scripts/verify_mask_v3.py 8`

## 关键结论在修正后仍然成立

**ZZZ 版本可见性（全量 60 题，修正块偏移后）**：

| 条件 | CB（闭卷） | A（全塞） | M（块遮蔽） | D（删文本） |
|---|---|---|---|---|
| visible（遮 0%） | 0/60 | 57/60 | 57/60 | 58/60 |
| hidden（遮 23%） | 0/60 | **57/60（泄漏）** | **0/60** | 0/60 |

- 遮蔽把泄漏从 57/60 压到 **0/60**；
- 可见条件下遮蔽不损质量（A 57/60 = M 57/60）；
- 遮蔽**不**诱导拒答（M 拒答 0/60，D 拒答 60/60），与之前一致。

修正前后的差别只有 1 条（原 58/60），说明旧脚本的块偏移恰好没有影响到这几条结论——但证据链现在是干净的。

复现：`python scripts/zzz_bench.py 60`

## 正确做法

不要用原始字符串算位置，先在**渲染后的完整序列**里定位 system 内容：

```python
ids_full = tokenizer.apply_chat_template(msgs, tokenize=True,
                                         add_generation_prompt=True,
                                         return_dict=False)
start = locate(ids_full, tokenizer(sys_text)["input_ids"])   # 真实起点
block = (start + local_offset) // 16
```

否则块号整体偏移，遮罩会落到相邻块上——**看起来"掩码没生效"，其实是遮错了地方**。
