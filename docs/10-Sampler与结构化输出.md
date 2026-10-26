# 第 10 课：Sampler 与结构化输出

> 采样看起来简单：softmax 然后抽样。但在 26k tok/s/GPU 的 decode 里，logits 的读写本身就是瓶颈。
> 本课讲三件事：融合、确定性、结构化约束。

## 本课目标

1. 说清 `fused_sample` 一个 kernel 里做了哪几步，省了什么。
2. 解释 batch-invariant 与 deterministic sampling 的区别和前置条件。
3. 画出结构化输出从 schema 到 mask 的完整链路。
4. 知道 SO 的失败模式与回滚逻辑。

## 为什么需要 fused sampler

vocab 很大（V4.1 约 128k+），每步 logits 是 `[num_rows, vocab]`。如果按"temperature → top-k → top-p → softmax → sample"分成 5 个 kernel：

- 每个 kernel 都要把 logits 读一遍、写一遍；
- `[384 行, 128k]` 的 FP32 logits 约 196 MB，5 遍就是 1 GB 的 HBM 流量；
- decode 一步才几毫秒，这个流量占比很高。

`fused_sample`（`tilelang_ops/fused_sample.py`）把这些步骤放进一个 kernel：

```python
def fused_sample(logits, sampling_inputs, *, tag="default", is_verify=None,
                 logprobs_out=None, last_nextn_token_logprobs=None,
                 last_next_n_token_ids=None, mtp_info_read_rows=None):
    """Apply temperature, top-p/top-k, log-softmax, and optional sampling."""
```

融合的步骤：
1. 按 `temperature` 缩放；
2. top-k / top-p 过滤（用 `DEEPINFER_SAMPLER_FORCE_TOPK` 限制候选集大小）；
3. log-softmax；
4. 采样（或验证）；
5. 可选输出 logprobs / top-k token ids；
6. 可选应用 SO mask。

`force_topk` 的意义：不是所有 token 都需要参与采样，先取 top-4096 再算，能把 softmax 的规模从 128k 降到 4k。代价是极小概率的 token 被截断（对生成质量影响可忽略）。

## 随机数：为什么不用全局 RNG

`random_sampling.py` 用整数 hash 生成均匀分布：

```python
def _hash_to_uniform(...): ...
```

好处：
- **无状态**：同一 `(seq_seed, token_index, step)` 永远得到同一个随机数；
- **可重现**：不依赖调用顺序，batch 怎么组都不影响；
- **可并行**：GPU 上每个线程独立算，不需要维护 RNG state。

这是 deterministic sampling 的基础。

## batch-invariant

目标：**不同 batch 组成下，同一个请求的输出逐 bit 一致**。

为什么难：
- GEMM 的 split-K、reduction 顺序会随 batch 大小改变；
- softmax 的 max/sum 归约顺序也会变；
- 浮点加法不满足结合律，顺序一变结果就变。

做法（`DEEPINFER_BATCH_INVARIANT=1`，仅 NVIDIA）：
- 固定 reduction 顺序（不随 batch 动态选 kernel）；
- 避免 batch 相关的 split；
- sampler 用固定的归一化路径。

**注意**：这不是"数学上等价"，而是"实现上固定顺序"。代价是可能损失一点性能（不能用最优的 split）。

## deterministic sampling

目标：**相同输入 + 相同采样参数 → 完全相同输出**。

前置条件（README）：

```
DEEPINFER_BATCH_INVARIANT=1
DEEPINFER_ENABLE_MTP_VARLEN=0
SamplingParams(seq_seed=<固定值>)
```

为什么要关 MTP varlen：变长验证会让每步的行数依赖运行时接受率，而接受率又依赖浮点结果——只要有一处不 deterministic，整条链就崩。关掉后每步长度固定，链路可重现。

两者区别：

| | batch-invariant | deterministic |
| --- | --- | --- |
| 保证 | 不同 batch 组成结果一致 | 相同输入结果一致 |
| 范围 | 模型 forward + sampler | 端到端（含投机） |
| 前置 | `BATCH_INVARIANT=1` | 还要关 varlen + 固定 seed |
| 用途 | 评测可复现、PPL 金标准 | 回归测试、审计 |

## 结构化输出：三段式

```
[编译]  schema/regex/EBNF → XGrammarCompiler → CompiledGrammar
   │        （独立线程池，异步）
   ▼
[匹配]  XGrammarMatcher：每步 accept_token / fill_vocab_mask
   │        （每请求一个，带 rollback）
   ▼
[应用]  fused sampler：mask 直接作用在 logits 上
```

### 编译

`structure_output/compiler.py:14`：

```python
class XGrammarCompiler:
    def dispatch_and_compile(self, schema_type: str, schema: str) -> CompiledGrammar:
```

`schema_type` 支持 `json` / `regex` / `ebnf`。编译产物是 grammar automaton，可以缓存（`DEEPINFER_XGRAMMAR_CACHE_LIMIT_BYTES`）。

### 异步化

`so_matcher_builder.py:19`：

```python
class SOMatcherBuilder:
    def __init__(...):
        self._pool = ThreadPoolExecutor(max_workers=max_workers, initializer=_bind_cpu_affinity)
        self._compiler = self._pool.submit(XGrammarCompiler, ...).result()
        self._pending = []

    def submit(self, seq, so_args, on_ready, *, replay_tokens=None):
        future = self.build(so_args, replay_tokens=replay_tokens)
        self._pending.append((seq, future, on_ready))

    def pump(self):
        for seq, future, on_ready in self._pending:
            matcher = future.try_result()   # 非阻塞
            if matcher is None: continue
            seq.so_matcher = matcher
            on_ready(seq)
```

三个设计点：
1. **独立线程池 + CPU 亲和**：编译是 CPU 密集，绑到 idle CPU，不抢主线程；
2. **`pump()` 非阻塞**：引擎在 `pre_sampler_sync` 调，编完的请求入调度，没编完的下次再看；
3. **失败隔离**：编译失败/超时返回 `aborted_matcher`，请求标记 `FINISHED_SO_FAILED`，不影响其他请求。

`add_request` 里的分支：

```python
if so_pending:
    self.so_matcher_builder.submit(seq, sampling_params.so_args, _enqueue)
else:
    _enqueue(seq)
```

注意：**SO 请求在编译完成前不会进入 scheduler**。所以 `has_unfinished_requests()` 要把 pending 数算进去，否则引擎会提前退出。

### 匹配与掩码

`matcher.py:XGrammarMatcher`：

| 方法 | 作用 |
| --- | --- |
| `accept_token(token)` | 推进自动机；非法 token 返回 False |
| `fill_vocab_mask(mask, idx)` | 产出该请求的合法 token 掩码 |
| `rollback(steps)` | 投机被拒时回退 |
| `finished()` | 是否到达接受状态 |
| `copy_without_status()` | 克隆（fork 请求用） |

掩码的应用在 `sampler._get_so_mask()` 和 `apply_token_bitmask_by_step_inplace()`：按 `step` 把 mask 应用到对应的 logits 行（投机验证的每一行都有自己的 mask）。

### trigger / bound

SO 支持两个高级参数：
- `trigger_token_ids`：只有出现这些 token 后才开始约束（比如 `{"` 之后才要求 JSON）；
- `bound_token_id`：遇到它就结束约束（比如 `}`）。

`validate_trigger_bound()` 会校验二者不冲突。这些是"部分结构化"的常用手段。

## 失败模式

| 失败 | 表现 | 处理 |
| --- | --- | --- |
| schema 非法 | 编译抛异常 | 请求 `FINISHED_SO_FAILED` |
| 编译超时 | `try_result` 返回 aborted | 同上 |
| 生成被拒 | matcher 长时间无合法 token | 引擎终止请求 |
| 投机回滚越界 | rollback 步数 > 已接受步数 | 断言/降级 |
| 抢占 | SO 不能重算 | 直接结束（`_preempt` 里有专门分支） |

## 配置

| 变量 | 作用 |
| --- | --- |
| `DEEPINFER_SO_BUILDER_CPUS` | 编译线程绑定的 CPU 列表 |
| `DEEPINFER_XGRAMMAR_CACHE_LIMIT_BYTES` | grammar 缓存上限 |
| `DEEPINFER_XGRAMMAR_TIMEOUT` | 单次编译超时 |
| `DEEPINFER_XGRAMMAR_DISABLE_CACHE` | 关缓存（调试） |
| `DEEPINFER_XGRAMMAR_DEBUG` | 打印编译细节 |
| `DEEPINFER_SAMPLER_FORCE_TOPK` | 采样候选集大小（影响 SO mask 的规模） |

## 图解：一个 SO 请求的时间线

```
add_request
  │  so_args 非空
  ├─► SOMatcherBuilder.submit（线程池编译）
  │        │
  │        └─ 编译完成 → pending
  │
  ▼（未入调度）
pre_sampler_sync → pump() → 入 scheduler
  │
  ▼
每步：matcher.accept_token(上一步 token)
      fill_vocab_mask → fused_sample
      │
      └─ 投机被拒 → rollback
```

## 易错点

1. **以为 SO 是"生成后校验"**：它是**生成时约束**（每步 mask）。
2. **以为编译是同步的**：默认异步，必须 pump。
3. **自己写循环忘了 pump**：请求永远不入调度。
4. **以为 SO 请求可以被抢占重算**：不能，直接结束。
5. **以为 `force_topk` 不影响 SO**：候选集变小，mask 也只在候选集上生效。

## 自测题

1. `fused_sample` 融合了哪几步？不融合会多多少 HBM 流量（估算）？
2. 为什么随机数要用 hash 而不是全局 RNG？
3. batch-invariant 和 deterministic 的前置条件各是什么？
4. SO 的编译为什么必须异步？同步会阻塞什么？
5. 投机被拒时，SO 状态为什么要 rollback？

## 下一课预告

**第 11 课《多模态 VE 与 EPD》** 会解释图像如何进入引擎、为什么视觉编码器要单独部署、以及 `MultimodalProcessor` 的共享内存池与异步 fetch。
