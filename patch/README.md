# vLLM 补丁说明（npc_ctx_mask.patch）

> 这是本工作**实际使用**的引擎补丁。5 个文件，211 行新增。

## 基线

| 项 | 值 |
|---|---|
| vLLM | **0.30.0**（pip wheel，非 git 仓库） |
| 注意力后端 | `FLEX_ATTENTION`（必须，块掩码依赖其 BlockMask） |
| 关键开关 | `--enforce-eager`、`--enable-prefix-caching` |
| 环境变量 | `VLLM_USE_FLASHINFER_SAMPLER=0`（RTX 5060 是 sm_120） |
| 已验证 | `patch -p1 --dry-run` 干净；应用后与实验所用构建**逐字节一致** |

## 改动逐文件说明

| 文件 | 行数 | 改了什么 |
|---|---|---|
| `vllm/_npc_vis.py` | 新增 ~140 | 解析 `request_id` 里的 `scope=<名>`；读黑名单 JSON（**按 mtime 重读**）；构造 `[请求数, 块数]` 布尔位图；展开成 per-token 行交给 `BlockSparsityHint` |
| `v1/attention/backend.py` | +5 | `AttentionMetadata` 增加 `npc_req_ids: list[str] | None` 字段 |
| `v1/worker/gpu/attn_utils.py` | +2 | 透传该字段 |
| `v1/worker/gpu/model_states/default.py` | +8 | 当 `NPC_VIS_BLOCKS` 存在时，从 `input_batch.req_ids` 填充 |
| `v1/attention/backends/flex_attention.py` | +18 | 构建 BlockMask 之前，把被遮块从 `kv_indices` 中剔除 |

## 接口

```bash
export NPC_VIS_BLOCKS=/path/to/blocks.json
# blocks.json: {"<scope名>": [[块起, 块止), ...], ...}   —— 黑名单
# 请求: request_id = "scope=<scope名>|<任意后缀>"
```

语义：**列出的块不可见，其余（含新生成 token）可见。**

## 为什么是黑名单

我们最初实现白名单（只列可见块），结果模型自己新生成的块也被遮掉，输出退化。黑名单是必须的。

## 与论文实验的对应

| 实验 | 用法 |
|---|---|
| canary（§5.1） | 只遮一个块，问该块内的编号 |
| 版本可见性（§5.3） | 遮住版本号高于进度的所有文档块 |
| 运行时切换（§5.5） | 同一 scope 改 JSON，不重启 |
| 槽位（§5.4） | 本补丁负责"遮盖"；写回 K/V 在研究环境中完成（引擎缺块表别名） |

## 已知限制

- **不能**给请求指定起始位置（试过，CUDA 越界，见 `docs/start-position-negative.md`）；
- **不能**把一个请求的 K/V 写进另一个请求的块（需块表别名，见 `docs/block-alias-design.md`）；
- 每个进程只记前 12 条单请求日志（`NPC_VIS_LOG_LIMIT` 可调）；
- 块大小固定 16，由 FlexAttention 决定，不可配置。

## 如何自检

```bash
# 1) 补丁能否干净应用
patch -p1 --dry-run -d <site-packages> < npc_ctx_mask.patch

# 2) 掩码是否真的生效（8 轮）
python scripts/verify_mask_v3.py 8
# 期望：visible 8/8、masked 泄漏 0/8、正对照 8/8、热切换 8/8
```
