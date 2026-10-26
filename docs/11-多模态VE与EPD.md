# 第 11 课：多模态 VE 与 EPD

> V4.1-Flash 原生支持视觉。图像不是"预处理成文本"，而是走独立的 Vision Encoder（VE），
> 再把 embedding 注入 LLM 的输入。本课讲这条支线。

## 本课目标

1. 说清 EPD（Encoder-Prefill-Decode）分离的动机与拓扑。
2. 理解 `MultimodalProcessor` 的共享内存池、worker 进程、异步 fetch。
3. 知道 image token 如何编码、如何与文本 token 对齐。
4. 了解 VE 的负载均衡策略与失败处理。

## 为什么 VE 要单独部署

| 维度 | LLM（P/D） | VE |
| --- | --- | --- |
| 计算特征 | 自回归、KV 增长 | 单次前向、固定形状 |
| 显存 | 权重 + KV cache | 权重 + 激活 |
| 批处理 | 连续批处理、变长 | 按图像 patch 数批处理 |
| 硬件 | SM100 / Ascend 950 | 官方建议"单 GPU、便宜卡即可" |

Q&A：

> VE：单实例一般一张 GPU 就够，建议用更便宜的 GPU。

混在一起的问题：图像的预处理（解码、resize、patch）是 CPU 密集且耗时不稳，会污染 LLM 的 step 时间线。拆开后，VE 可以独立扩缩容，LLM 的 P/D 不受影响。

## 拓扑

```
用户请求（含图）
   │
   ▼
Entrypoint ──► Vision Encoder LB ──► VE Service（每 GPU 一个 encoder）
   │                                      │
   │  ◄──── image embedding ──────────────┘
   ▼
Prefill LB ──► Prefill Service ──► Decode LB ──► Decode Service
```

Q&A 的路由细节：
- 图像按**机器粒度**路由：prefill 机器随机选一台，再选该机器上"当前处理 patch 数最少"的 GPU；
- patch 数直接由尺寸算：`(width/patch_size) × (height/patch_size)`；
- 这样不需要维护复杂的负载表。

## 代码：MultimodalProcessor

`deepinfer/ve/multimodal_processor.py:29`：

```python
class MultimodalProcessor:
    def __init__(self, model_config, tokenizer, vocab_size, dp_rank, image_token=None): ...
    def overwrite_prompt(self, prompt_token_ids, multi_modal_data): ...
    def add(self, id, multi_modal_data, num_cached_tokens=0): ...
    def fetch(self): ...
    def fetch_to_abort_requests(self): ...
    def abort(self, ids): ...
    def stop(self): ...
```

关键方法：

### overwrite_prompt

在 `add_request` 阶段同步执行：
- 把 prompt 里的 image 占位符替换成 `image_token` 序列；
- 记录每张图在序列中的位置（供后续 embedding 注入）；
- 校验 token 范围（`vocab_size + ImageTokenOffset`）。

```python
max_valid_token_id = vocab_size + (ImageTokenOffset.NUM_SPECIAL if multi_modal_data is not None else 0)
```

`ImageTokenOffset`（`multimodal_tokens.py`）定义图像相关特殊 token 的编号偏移，避免和文本词表冲突。

### add / fetch

- `add`：把 `multi_modal_data`（图像字节）放进待处理队列；
- `fetch`：在每步调度前调用，把已经处理完的 embedding 取回。

处理发生在**独立 worker 进程**里（`_start_workers`），通过共享内存池传递结果：

```python
def _processor(self, multi_modal_pool: SharedTorchMemoryPool): ...
```

`SharedTorchMemoryPool`（`shared_memory.py`）用 `posix_ipc` / 共享内存传张量，避免 pickle 大对象。

### 失败处理

```python
abort_requests = self.multi_modal_processor.fetch_to_abort_requests()
if abort_requests:
    logger.error("多模态处理失败（通常是损坏的图片/视频），即将 abort ...")
    self.abort_request(abort_requests)
```

- 图像解码失败、worker 超时（`DEEPINFER_MULTI_MODAL_WORKER_TIMEOUT`）、共享池不足，都会导致请求被 abort；
- 被 abort 的请求标记 `FINISHED_VE_FAILED`，在调度器里进 `ignored_seqs`，带 finish_reason 回传。

## 数据流：图像怎么变成 embedding

```
image bytes
  │
  ├─ processor/image_processor.py：解码、resize、normalize、切 patch
  ├─ processor/mm_processor.py：组织多图 batch、pad
  ├─ processor/text_processor.py：文本部分（与图像交错）
  │
  ▼
ViT（models/vision_encoder/vit.py）
  │   patch embed → transformer blocks → downsample
  ▼
mlp_projector.py / vl_aligner.py
  │   投影到 LLM hidden size
  ▼
embedding（写入共享内存池）
  │
  ▼
LLM 输入：文本 token embed + image embedding 按位置拼接
```

相关 kernel：`tilelang_ops/*/image_embed_ops.py`（图像 patch 的 embedding 组织）。

## 图像与 KV cache 的交互

两个容易忽略的点：

1. **图像不能跨越 prefix cache 边界**：benchmark 明确要求"整图放在 cached prefix 或 non-cached suffix，不能跨边界"（`examples/README.md`）。原因是图像 token 的 embedding 依赖整张图，切一半没法算。
2. **image pad 区域不能切 chunk**：`_adjust_chunk_size` 里会检查 `_is_inside_image_pad_region`，chunked prefill 不能切在 pad 中间。

## VE 的独立测量

`examples/sweep/ve_sweep.py` 只加载 VE 权重，计时窗口覆盖：
```
GPU 侧 pad → ViT → downsample → aligner
```
不包含 CPU 预处理（异步 worker）和 H2D（随 InputMetadata 传输）。

吞吐按"每张图对应的 LLM token 数"折算，单位与 prefill 曲线一致（tokens/s/GPU）。

## 配置

| 变量 | 作用 |
| --- | --- |
| `DEEPINFER_USE_MULTIMODAL_POOL` | 是否用共享内存池 |
| `DEEPINFER_MULTIMODAL_POOL_SIZE_GB` | 池大小 |
| `DEEPINFER_MULTI_MODAL_BUFFER_SIZE` | 单请求缓冲 |
| `DEEPINFER_MULTI_MODAL_WORKER_TIMEOUT` | worker 超时（默认较长） |

## 图解：一个多模态请求

```
add_request(prompt, image_bytes)
  │
  ├─ overwrite_prompt：占位符 → image token 序列
  ├─ processor.add(image_bytes)          # 异步
  │
  ▼
scheduler（先按 token 调度，不等图像）
  │
  ▼
每步 fetch() ──► embedding 到了？
  │                 │
  │ 否              │ 是
  ▼                 ▼
继续调度文本      注入 embedding → forward
  │
  └─ 失败 → abort → FINISHED_VE_FAILED
```

**注意**：引擎可以先调度 token，图像 embedding 晚到再注入。这要求 metadata 支持"图像位置占位"，也是为什么 `prepare_tensor_input_vl.py` 单独存在。

## 设计取舍

| 选择 | 收益 | 代价 |
| --- | --- | --- |
| VE 独立部署 | 资源解耦、独立扩缩容 | 多一跳网络、需要 LB |
| 异步预处理 | 不阻塞 LLM step | 状态机复杂、失败要 abort |
| 共享内存池 | 零拷贝传大张量 | 池大小要预估，容易 OOM |
| 机器粒度路由 | 实现简单 | 不是最优负载均衡 |

## 易错点

1. **以为图像会被转成文本 token**：它走 embedding，不占文本词表。
2. **在 chunked prefill 里切断图像**：会被 `_adjust_chunk_size` 拦住。
3. **忘了 `free_multi_modal_pool_handles`**：abort 的请求要归还池块，否则泄漏。
4. **以为 VE 失败会重试**：当前实现是直接 abort 请求。
5. **在多图请求里混用尺寸**：patch 数不同，batch 组织会 pad，吞吐会降。

## 自测题

1. EPD 分离解决了什么问题？不分离会怎样？
2. `overwrite_prompt` 在哪个阶段执行？为什么必须同步？
3. 图像 embedding 晚到，引擎怎么继续调度？
4. 为什么图像不能跨 prefix cache 边界？
5. VE 的负载均衡为什么按"patch 数"而不是"请求数"？

## 下一课预告

**第 12 课《双硬件与 Kernel 工程》** 会讲 `Platform` 抽象、`tilelang_ops/{cuda,asc}` 的双份实现、vendor 库的构建，以及 CUDA Graph 与 profiling 钩子。
