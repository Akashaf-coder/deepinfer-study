# 第 04 课：ZOS——把 CPU 调度藏进 GPU 计算

> ZOS = Zero-Overhead Scheduling。它不是"调度更快"，而是"调度时 GPU 不空转"。
> 本课会拆开 `worker.execute_model_step()` 的四段流水线，并解释它带来的三个新问题。

## 本课目标

1. 画出非 ZOS 与 ZOS 的 CPU/GPU 时间线，指出空转发生在哪。
2. 说清四段流水线的顺序，以及为什么顺序不能换。
3. 解释 `DeferredRelease`、`placeholder step`、`seqlens 修正` 各自防的问题。
4. 知道 ZOS 在什么情况下关闭，以及调试时会遇到什么怪现象。

## 问题：decode step 里的两段"纯 CPU"

一个 decode step 的逻辑：

```
schedule(N) → forward(N) → sample(N) → post_process(N)
```

- `schedule` 是纯 CPU（C++，但仍然要等 GPU 空闲才能发下一步）；
- `post_process` 是纯 CPU，而且**必须先拿到 sample 的结果**（GPU→CPU 同步）；
- 所以 GPU 在 `post_process(N)` 和 `schedule(N+1)` 期间是空闲的。

decode 的 step 时间是毫秒级，CPU 这两段可能占 5%–20%（取决于 batch 大小和 CPU 主频）。batch 越小，占比越高。

## 解法：把 CPU 提前一步

ZOS 后的稳态：

```
GPU:  forward(N) ──────────► sample(N) ──────────► forward(N+1)
CPU:  schedule(N+1) → launch_forward(N+1) → post_process(N-1) → launch_sample(N+1)
```

CPU 在第 N 步的 GPU 计算期间，已经把第 N+1 步的调度做完、并且把第 N-1 步的后处理补完。GPU 从 `forward(N)` 到 `forward(N+1)` 之间没有空隙。

## 源码：四段流水线

`deepinfer/worker/worker.py:89`，函数注释就是设计文档：

```python
def execute_model_step(self, is_prompt, is_placeholder,
                       seq_metadata_list=None, scheduler_outputs=None):
    """The unified entry for model execution: every step ... goes through
    this Forward-Post_process-Sample round.
      1. Forward current step (main model)
      2. Post-process previous step
      3. Sample current step
      4. Post-process current step: in place for prompt / non-ZOS steps;
         ZOS decode steps defer it to the next step to overlap with forward
    """
```

对应代码：

```python
# 1. forward curr step
main_model_output = self.model_runner.execute_main_model(...)

# 2. post process prev step
request_outputs = self._post_process_step(self.prev_step_state)

# 3. sample curr step
sampler_output = self.model_runner.execute_sample(...)

# 4. prompt 步或非 ZOS：当场 post process；ZOS decode 留到下一步
if is_prompt or not envs.DEEPINFER_ZERO_OVERHEAD_SCHEDULING:
    request_outputs = np.concatenate([request_outputs, self._post_process_step(curr_step_state)])
    self.prev_step_state = None
else:
    self.prev_step_state = curr_step_state
```

**为什么顺序是 1→2→3→4，而不是 1→3→2？**
因为 `post_process(N-1)` 要用第 N-1 步的 sampler 输出，而那时 GPU 正在跑 forward(N)。放在 forward 之后、sample 之前，可以让 CPU 的工作和 GPU 的 forward 重叠；如果放在 sample 之后，就要等 sample 的同步，白等。

## 三个新问题

### 问题 1：资源什么时候释放

第 N 步产出的 `SequenceMetadata`、`RequestOutput` 在返回给调用方后，不能立刻回收——调用方可能还在读。ZOS 又让"下一步"提前开始，资源生命周期更乱。

解法：`DeferredRelease`（`llm_engine.py:67`）。

```python
class DeferredRelease:
    """延后一轮释放 step 产出的资源，保证调用方读完。
    回收（collect）要释放的资源；在下一次的 pre_sampler_sync 时释放（release），
    与 GPU 计算 overlap。
    """
```

流程：
```
step(N):   产出 → collect_*() 挂起
step(N+1): forward 开始后 → pre_sampler_sync() → release() 释放 N 的资源
```
`pre_sampler_sync` 的时机很讲究：它在 sample 之前、forward 之后，此时调用方早已读完 N 的输出。

### 问题 2：没有请求时 GPU 也要动

DP 组要求所有 rank 同型同步。如果某个 rank 这一步没请求，它不能"跳过"——否则其他 rank 的集合通信会挂起。

解法：**placeholder step**。构造一个假的 `SequenceMetadata`，跑一次 forward（结果丢弃），保持步型一致、NCCL 不超时。

```python
if is_placeholder:
    seq_metadata_list = [self.model_runner._placeholder_seq_metadata(is_prompt)]
    main_model_output = self.model_runner.execute_main_model(...)
```

代价：空转的 forward 也要花时间。所以 placeholder 只用于"必须保持同步"的场景，不是越多越好。

### 问题 3：调度时还不知道真实长度

ZOS 下 `schedule(N+1)` 在第 N 步的 sample 结果回来之前执行。对于普通 decode，每序列长度是 1（或 1+nextn），可以确定；但**投机解码的验证长度依赖上一步的接受情况**，CPU 不知道。

解法分两层：
1. **CPU 侧乐观估计**：`SchedulerOutputs.seqlens_q` 是"乐观估计"，按历史接受率给一个长度；
2. **GPU 侧修正**：`worker._correct_seqlens_qk()` 用 D2H 回来的真实 `seqlens_q` 修正 metadata，再进入 sample。

```python
if correction is not None:
    if correction.d2h_evt is not None:
        correction.d2h_evt.synchronize()
    scheduler_outputs.update_metadata_for_sampling_zos(
        seq_metadata_list, correction.seqlens_q_cpu, correction.metadata_zos.remap_cpu, self.nextn
    )
```

`MetadataZOS` 里还有 `remap`：因为 GPU 侧重排了请求顺序，CPU 侧的 metadata 要按 remap 重新对应。

## 时序细节：一个稳态 step 的完整顺序

```
CPU:
  schedule(N) ──► launch_forward(N) ──► post_process(N-1) ──► launch_sample(N)
                                             │
GPU:                                     forward(N) ─────────► sample(N)
                                             ▲
                                     pre_sampler_sync
                                     （释放上一步资源、SO pump、保存 KV）
```

`pre_sampler_sync` 里做的事（`llm_engine.py:830`）：
1. `deferred_release.release()`；
2. `do_save_kvcache(...)`（decode 时保存 SWA/state 到 host/3FS）；
3. `so_matcher_builder.pump()`（收割编译好的 grammar）；
4. 对 COMPLETION/NORMAL 引擎，`finalize_seq` 回收已结束的序列。

## 什么时候不用 ZOS

| 场景 | 原因 |
| --- | --- |
| prompt step | prefill 本身长，CPU 那点开销可忽略；且 prefill 的输出要立刻用于后续调度 |
| `DEEPINFER_ENFORCE_EAGER=1` | 调试时关 graph，也常关 ZOS |
| Ascend 某些配置 | 平台差异（代码里默认值按 engine_type 给，实际可被环境变量覆盖） |
| 单请求延迟测试 | ZOS 会让"输出晚一步"，测 TTFT 时要小心 |

`EngineType.default_zero_overhead_scheduling`：

```python
return self in (EngineType.NORMAL, EngineType.COMPLETION)
```

PROMPT 引擎默认不开。

## 调试 ZOS 的三个坑

1. **输出晚一步**：`post_process(N)` 处理的是第 N 步的 sample 输出，但它在第 N+1 步的 forward 之后才执行。日志时间戳会看起来"错位"。
2. **同步守卫**：`deepinfer/utils.py` 有 `forbid_cuda_sync()` / `allow_cuda_sync()`，`tests/test_cuda_sync_guards.py` 会检查关键路径上有没有非法的 `.item()` / `.cpu()`。加调试代码时很容易触发。
3. **CUDA Graph + ZOS**：`CudaGraphRunner(input_on_device=True)` 是 ZOS 专用路径，metadata 要在 device 上准备好。关掉 ZOS 时某些 graph 路径不走。

## 如何验证 ZOS 真的生效

- 看 `cpu_timer` 的 `bubble` 项（`post_process` 里记录），ZOS 开启后应显著下降；
- `PROFILE=1` 导出 trace，看 GPU 上 `forward → sample → forward` 之间有没有 gap；
- 对比 `DEEPINFER_ZERO_OVERHEAD_SCHEDULING=0/1` 的 `avg_SPS`（benchmark 会打印）。

**注意**：官方 README 提醒，profiler 本身会让 step 时间膨胀（bsz=1 时可达 50%），所以只能看"有没有 gap"，不能用 trace 里的 step 时间算吞吐。

## 图解：ZOS 的收益与代价

```
非 ZOS:
  GPU  ████████░░░░████████░░░░████████
  CPU      ░░░░      ░░░░      ░░░░
           ↑空转     ↑空转     ↑空转

ZOS:
  GPU  ████████████████████████████████
  CPU    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
         （全部藏在 GPU 计算下面）

代价：
  1. 输出晚一步 → 调试复杂
  2. 调度时信息不全 → 需要乐观估计 + 修正
  3. 资源生命周期变长 → 需要 DeferredRelease
  4. 空 rank 也要跑 → placeholder step
```

## 易错点

1. **以为 ZOS 是"多线程调度"**：它是单线程的顺序调整，不是并行。
2. **以为 `prev_step_state` 是缓存**：它是"上一步还没 post-process 的状态"，只用一次。
3. **以为 placeholder 不花钱**：它跑完整 forward，只是结果丢弃。
4. **以为 `seqlens_q` 是真实值**：ZOS 下是估计值，GPU 侧会修正。
5. **在 ZOS 路径里加 `.item()`**：会触发同步守卫测试失败，也会破坏重叠。

## 自测题

1. 画出非 ZOS 和 ZOS 的两条时间线，标出空转段。
2. 为什么 `post_process(N-1)` 要放在 forward(N) 之后、sample(N) 之前？
3. `DeferredRelease` 如果改成"本步立即释放"，会出现什么 bug？
4. placeholder step 解决了什么问题？不用它会怎样？
5. ZOS 下投机解码的长度信息为什么必须由 GPU 修正？

## 下一课预告

**第 05 课《KV 缓存：block 与 state》** 会解释：为什么 paged block 和 state slot 是两套管理器、FP4 的两种 scale 粒度、fork 请求的 copy-on-write、以及"前缀命中"到底发生在哪一层。
