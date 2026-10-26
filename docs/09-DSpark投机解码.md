# 第 09 课：DSpark——置信度变长投机

> 主流投机解码验证固定 5 个 draft token。DeepSeek 说：为什么不全验证，也不只验证 1 个？
> 答案是按"期望接受率"排序，选一个前缀子集，让单位验证算力产出最大。

## 本课目标

1. 说清 DFlash（draft head）与 DSpark（调度策略）的分工。
2. 理解 confidence head 输出什么、怎么用。
3. 能手动推演 `MTPVarlen::schedule()` 的两队列算法。
4. 解释"用 N-2 轮接受率选 batch、用 N-1 轮分配每请求长度"的原因。
5. 知道 ZOS 下为什么要在 GPU 上重算长度。

## 组件：draft 模型 + 置信度 + 调度

| 组件 | 位置 | 作用 |
| --- | --- | --- |
| DFlash | `models/mtp/dflash.py` | MTP draft head，一次生成多个候选 token |
| ConfidenceHead | 同上，`class ConfidenceHead` | 预测每个 draft token 的接受概率 |
| Sampler 验证 | `layers/sampler.py:_verification` | 用 target 模型 logits 做拒绝采样 |
| MTPVarlen | `csrc/engine/mtp_varlen.hpp` | 决定这一步验证多少、每个请求验证几个 |
| GPU 重排 kernel | `tilelang_ops/*/schedule_mtp_varlen.py` | ZOS 下在 GPU 上重算 per-request 长度 |

DFlash 的 forward 路径（`dflash.py`）：
`forward_drafts` → 生成 draft token 的 hidden；
`sample_drafts` → 采样 draft token + confidence；
`run_drafts` → 组织成下一步的输入；
`decode_forward` → 与主模型共享 KV 的 decode 路径。

`dflash_cfg.dspark_target_layer_ids` 指定主模型哪几层的 hidden 喂给 draft head（多层特征拼接，`dflash_input_size = hidden_size * len(target_layer_ids)`）。

## 验证：拒绝采样

`Sampler._verification()`（`sampler.py:411`）的核心步骤：

```python
ftp = decode_inputs.mtp_first_token_positions
seqlen_q = decode_inputs.mtp_seqlens_q
row_to_req, local_step, is_verify = prepare_verification_metadata(
    ftp, seqlen_q, total_q, self.max_MTP_value
)
...
logprobs, token_index_cur, candidate_token_ids, accept_result_verify, \
    draft_selected_logprobs, accept_probs = process_sampling(
        logits, driver_inputs.kernel, self.force_topk,
        row_to_req=row_to_req, local_step=local_step, is_verify=is_verify,
        ...
    )
...
sample_results_cur, accept_result, is_final = get_accept_result(
    accept_result_verify, driver_inputs.golden_output, ...
)
```

逐行解释：
1. `ftp`（first token positions）+ `seqlens_q` 把扁平的行映射到"请求 × 第几个 draft"；
2. `prepare_verification_metadata` 生成 `row_to_req` / `local_step` / `is_verify` 三个索引张量；
3. `process_sampling` 对验证行做拒绝采样（target 概率 vs draft 概率），得到 `accept_result_verify`；
4. `get_accept_result` 找出每个请求的**接受前缀**和**最终行**（第一个被拒的位置要采样一个新 token）；
5. `finalize_verification` 把结果打包成 `SamplerOutput`。

被拒绝的 draft 之后，SO 的自动机要 `rollback(steps)`（`matcher.py`），否则 grammar 状态会错。

## 调度：两个队列，两个时间尺度

`mtp_varlen.hpp:schedule()` 的注释（建议逐字读）：

```
需要保证 confidence_{i+1..nextn} 不影响 draft_token_i 是否被验证。
假设此时在第 N 轮，期望使用 N-2 轮的 accept_rate 计算 best_bsz (total_q)；
并使用 N-1 轮的 accept_rate 分配每个请求的 seqlen_q（non-ZOS 在这里做，ZOS 在 GPU 上修正）。
```

为什么是 N-2 和 N-1：
- confidence 是**滞后**的：第 N 步调度时，第 N 步的 forward 还没跑，拿不到第 N 步的接受率；
- 用 N-1 轮分配 per-request 长度（相对新）；
- 用 N-2 轮决定总预算 `best_bsz`（更保守，因为总量错误代价更大）。

代码维护两个队列：

```cpp
std::vector<std::pair<float,int>> acc_queue_for_bsz;     // accept_rate, seq_idx
std::vector<std::pair<float,int>> acc_queue_for_seqs_q;
...
push_rates(acc_queue_for_bsz, post_process_paired ? mtp.prev_accept_rates : mtp.accept_rates, ...);
push_rates(acc_queue_for_seqs_q, mtp.accept_rates, ...);
std::sort(..., by_rate_desc);
```

`post_process_paired` 判断"上一步的 post-process 是否已完成"：
- 配对（非 ZOS）：`accept_rates` 是 N-1 轮；
- 未配对（ZOS）：`accept_rates` 是 N-2 轮，要再往前取一轮。

## 选总预算：扫描 + 吞吐模型

```cpp
float cur_expected = num_seqs;   // 每个请求至少 1 个 target token
int cur_bsz = num_seqs;
update_best();                   // 全不接受的情况

for (auto &acc : acc_queue_for_bsz) {
    cur_expected += acc.first;   // 期望多接受一个 token
    cur_bsz += 1;
    update_best();
}
```

`update_best()`：

```cpp
const auto &[graph_size, baseline_ms] = _decode_perf_curve[cur_bsz_idx];
const double time_ms = baseline_ms + slope_ms * graph_size + sampler_ms;
const double cur_throughput = cur_expected / time_ms;
if (cur_throughput > best_throughput) { ... }
```

- 分子：期望接受的 token 数（`num_seqs` + 每个 draft 的接受率之和）；
- 分母：forward 时间（启动曲线 + 斜率 × graph_size）+ sampler 时间；
- 扫所有候选 `graph_size`，选吞吐最高的。

**这是"按边际收益分配验证算力"的具体实现**：每多验证一个 draft，分子加它的接受率，分母加一点时间；接受率低于边际成本时就停。

## 分配每请求长度

```cpp
seqlens_q.assign(num_seqs, 1);                      // 每个请求至少 1 个 target token
for (int i = 0; i < best_bsz - num_seqs; ++i) {
    seqlens_q[acc_queue_for_seqs_q[i].second] += 1; // 给接受率最高的 draft 加一格
}
```

注意：按**单个 draft token** 分配，不是按请求。所以一个请求可能拿到 0、1、2…个 draft，总数是 `best_bsz`。

`seqlens_q` 的含义：该请求这一步要验证多少行（1 = 只算 target token；1+k = 再验证 k 个 draft）。

## 运行时估计：滑动中位数

`mtp_varlen.hpp` 里两个估计器：

| 估计器 | 估计什么 | 方法 |
| --- | --- | --- |
| `SamplerRuntimeEstimator` | sampler 耗时 | 最近 9 个样本的中位数 |
| `ForwardRuntimeEstimator` | forward 启动曲线 + 斜率 | 按 graph_size 分组，先组内中位数，再对中位数取中位数；窗口 5000 |

为什么用中位数而不是均值：GPU 上偶发的长尾（别的进程、时钟抖动）会污染均值，中位数更稳。

## ZOS 下的冲突与解法

冲突：
- ZOS 让 `schedule(N+1)` 在第 N 步 sample 之前执行；
- 此时第 N 步的 confidence 还没回来；
- 但 per-request 长度需要它。

解法（Q&A）：
> CPU 只定总预算；per-request 长度由 GPU 上的 TileLang kernel 重算（cumulative product、global sort for top-k、fused remap scatter），结果写回 pinned buffer。

kernel 实现（`schedule_mtp_varlen.py`）：
1. **Phase 1**：按 `remap` 读上一步的 confidence，算累积乘积 `cum_rates`；
2. **Phase 2**：每个请求并行计算"自己的每个 draft 在全局的排名"（higher rate wins, tie-break by flat index）；
3. 按排名分配 `total_q`，写回 `seqlens_q`；
4. 同时产出 `remap`，供 CPU 侧 metadata 重排。

这就是为什么 `worker.py` 里要有 `_correct_seqlens_qk()` 和 `MetadataZOS.remap_cpu`。

## 收益（Q&A 数据）

配置：EP64/DP64，nextn=5，生产工作点约 60 并发/rank。

| 并发 | fixed-5 平均验证长度 | dynamic 平均验证长度 | fixed 接受长度 | dynamic 接受长度 | 吞吐增益 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 5 | 4.44 | 3.32 | 3.24 | −1.5% |
| 32 | 5 | 3.65 | 3.25 | 3.17 | +1.6% |
| 48 | 5 | 3.94 | 3.3 | 3.26 | +3.6% |
| 60 | 5 | 3.58 | 3.3 | 3.21 | +5.4% |
| 72 | 5 | 3.24 | 3.29 | 3.12 | +6.7% |
| 96 | 5 | 3.24 | 3.3 | 3.15 | +11.0% |

两个洞察：
1. **低并发时动态验证反而更差**（−1.5%）：因为 batch 小，验证的固定开销占比高，少验证省不了多少时间；
2. **高并发时收益递增**：验证算力是稀缺资源，按边际收益分配的价值才体现出来。

## 配置与开关

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `DEEPINFER_USE_NEXTN_DECODE` | 5 | 最大 draft 长度；0 = 关投机 |
| `DEEPINFER_ENABLE_MTP_VARLEN` | 0/1 | 开变长验证（DSpark） |
| `DEEPINFER_DFLASH_CONFIDENCE_HEAD_TEMPERATURE` | — | confidence 温度，影响排序 |
| `DEEPINFER_MAX_BACKTRACKING_TOKENS` | — | 最大回溯长度（和 YoCo 的 6 有关） |
| `DEEPINFER_DFLASH_ENFORCE_EAGER` | 0 | 关 DFlash 的 CUDA Graph（调试） |
| `DEEPINFER_DECODE_PERF_CACHE_DIR` | — | decode 性能曲线缓存 |

固定长度 vs 变长：
- `ENABLE_MTP_VARLEN=0`：每条请求都验证 `nextn` 个，实现简单、可预测；
- `=1`：走上面的调度算法，需要 decode 性能曲线（`decode_perf_cache`）。

## 图解：一步投机的生命周期

```
step N:
  target forward ──► logits
       │
       ├─ draft head ──► draft tokens + confidence
       │
       └─ verify ──► 接受前缀 k, 新 token t
                        │
step N+1:               ▼
  schedule: 用 N-1 的接受率分配长度（ZOS 下 GPU 重算）
       │
       └─ 只 forward 1+k 行
```

## 易错点

1. **以为 confidence 是"概率"**：它是接受率的估计，用于排序，不参与拒绝采样。
2. **以为验证长度固定**：开 varlen 后每请求不同。
3. **在 ZOS 下用 CPU 的 `seqlens_q` 做逻辑判断**：它是估计值，真实值在 GPU 修正后。
4. **低并发开 varlen**：可能负收益（见上表）。
5. **忘了 SO rollback**：被拒的 draft 会让 grammar 状态错位。

## 自测题

1. 为什么用 N-2 轮接受率选总预算、N-1 轮分配每请求长度？
2. `best_bsz` 的吞吐模型里，分子和分母各是什么？
3. 为什么按单个 draft token 分配，而不是按请求分配？
4. ZOS 下 GPU kernel 做哪三件事？
5. 为什么低并发时动态验证会亏？

## 下一课预告

**第 10 课《Sampler 与结构化输出》** 会解释：fused sampler 怎么把温度/top-p/top-k/log-softmax 融成一个 kernel、batch-invariant 怎么保证、xgrammar 的编译如何异步化。
