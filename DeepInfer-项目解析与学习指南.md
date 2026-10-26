# DeepInfer 项目解析与学习指南

> 分析对象：`https://github.com/deepseek-ai/DeepInfer`
> 本地路径：`~/DeepInfer`（浅克隆，commit `9bd9cf2`，2026-10-23）
> 说明：本文所有"代码事实"都能在仓库里对应到文件；标注"外部资料"的内容来自官方博客/Q&A/第三方解读，未在代码中直接验证。

---

## 0. 十句话看懂这个项目

1. **它是 DeepSeek 生产推理框架的开源版**，支撑 DeepSeek-V4.1-Flash 的大规模在线服务；开源部分只包含**离线推理**能力。
2. **血统是 vLLM**：大量文件头写着 `Derived from vLLM; modified by DeepSeek.`，异步调度思路来自 SGLang，图优化借鉴 TensorRT-LLM。
3. **规模**：270 个 Python 文件 / 54,929 行；33 个 C++ 文件 / 5,908 行。其中 **23,331 行（约 42%）是 TileLang kernel**，真正的"框架逻辑"只有约 3 万行。
4. **只支持一个模型**：DeepSeek-V4.1-Flash。模型加载、权重名映射、注意力结构全部为该架构特化，不是通用框架。
5. **只支持两类硬件**：NVIDIA **SM100**（B200/GB200 级）与华为 **Ascend 950**。H100/A100/4090 跑不起来（`get_bf16_tflops_limit()` 会直接抛 `unknown GPU type`）。
6. **没有 TP**：全仓库 `tensor_parallel` / `tp_size` 出现次数为 **0**。生产形态是 **EP64–128 + DP +（prefill 侧）CP**。
7. **调度器用 C++ 写**（`csrc/engine/scheduler.hpp`，1100+ 行），Python 只做编排；这跟 vLLM V1 的 Python 调度器是相反的设计取向。
8. **三大招牌特性**：ZOS 零开销异步调度、YoCo bounded replay（只重算 128 token）、DSpark 置信度变长投机验证。
9. **刻意不开源的部分**：KV Cache 存储（3FS）、Prefill/Decode/VE/EP 四类负载均衡器、扩缩容、在线 API server、DeepSeek Recipe 编解码。这些恰恰是"大规模"三个字的核心。
10. **学习价值**：它不是"又一个 vLLM"，而是**"通用框架如何为一个特定模型做极致外科手术"**的样本。读它是为了学特化手法，不是为了找个通用推理引擎。

---

## 1. 项目定位：它是什么，不是什么

### 1.1 官方三条设计目标（来自 `doc/inference_deployment_qa.md`）

1. 支持多厂商加速器，并吃满各平台性能；
2. 用** commodity 硬件**组织 prefill/decode：8 卡机器做 prefill，64–128 卡超节点做 decode；
3. KV cache 保留 **72 小时以上**，用长窗口换前缀命中率。

这三条决定了后面所有设计：为什么要 EPD 分离、为什么要 Wide-EP、为什么要有独立的 KV 存储集群、为什么引擎内不做自动前缀缓存。

### 1.2 是什么

| 维度 | 事实 |
| --- | --- |
| 定位 |  DeepSeek 内部生产框架的开源子集（离线推理） |
| 模型 | DeepSeek-V4.1-Flash（`deepseek-ai/DeepSeek-V4.1-Flash-1017`） |
| 硬件 | NVIDIA SM100 / Ascend 950 |
| 入口 | `deepinfer.entrypoints.llm.LLM`（离线批量） |
| 部署形态 | NORMAL（P+D 合一）/ PROMPT / COMPLETION 三种引擎 |
| 并行 | EP + DP，prefill 可开 CP，**无 TP/PP** |
| 核心 kernel | FlashMLA、DeepGEMM、DeepEP、DeepSelect、TileKernels、TileLang 自研算子 |
| License | Apache-2.0 |

### 1.3 不是什么（很重要，别踩坑）

- **不是通用推理框架**：没有 OpenAI 兼容 server、没有 `AsyncLLMEngine`、没有 adapter/LoRA、没有多模型热加载。
- **不是即插即用的 vLLM 替代品**：只认 V4.1 的 HF checkpoint 格式，换模型要改模型代码 + kernel。
- **不是完整生产系统**：KV store、四类 LB、autoscaling、3FS 集成都不在仓库里（Q&A 明确说明）。
- **不是"消费级 GPU 也能玩"的项目**：需要 CUDA 13 + SM100，或 CANN 9.2 + Ascend 950。
- **不是稳定 API**：4 个 commit、v0.1.0、154 个 PR，接口随时会变，学习时请 pin 住 commit。

---

## 2. 目录地图（按阅读优先级排序）

### 2.1 顶层结构

| 目录 | 行数 | 作用 | 优先级 |
| --- | ---: | --- | :---: |
| `deepinfer/engine/` | 1,957 | 引擎编排：`LLMEngine`、`EngineArgs`、统计 | ★★★★★ |
| `deepinfer/worker/` | 2,805 | Worker、ModelRunner、CacheEngine、CUDA Graph | ★★★★★ |
| `csrc/engine/` | ~5,000 | **C++ 调度器**、BlockManager、Sequence、ZOS、FlopsEstimator | ★★★★★ |
| `deepinfer/model_executor/models/` | 7,051 | V4 模型结构（attn/moe/engram/hc/yoco/vision/MTP） | ★★★★★ |
| `deepinfer/model_executor/layers/` | 24,996 | 层实现 + **23,331 行 TileLang 算子** | ★★★★ |
| `deepinfer/model_executor/eplb/` | 792 | 专家负载均衡（策略 + 权重重排） | ★★★★ |
| `deepinfer/structure_output/` | 595 | xgrammar 结构化输出（编译/匹配/掩码） | ★★★ |
| `deepinfer/ve/` | 979 | 多模态视觉编码器侧：图像预处理、metadata | ★★★ |
| `deep_cluster/` | 656 | 硬件发现（NVIDIA/Ascend）、NUMA 绑定、拓扑 | ★★ |
| `deepinfer/distributed/` | ~500 | 共享内存广播（SHM broadcast） | ★★ |
| `examples/` | 3,018 | 离线推理、benchmark、sweep、YoCo 等价性验证 | ★★★★★ |
| `tests/` | 3,117 | 单元/端到端测试，含 PPL 金标准 | ★★★ |
| `eval/` | 1,694 | AIME/HLE/GPQA 离线评测 | ★★ |
| `vendor/` | — | 6 个 git submodule（DeepGEMM/FlashMLA/DeepEP/DeepSelect + Ascend 版） | ★ |

### 2.2 必读的 15 个文件（建议按顺序）

| # | 文件 | 为什么读 |
| --: | --- | --- |
| 1 | `README.md` | 特性清单 + 环境要求（10 分钟） |
| 2 | `doc/inference_deployment_qa.md` | **全文最值钱**：生产系统的设计取舍、数字、坑 |
| 3 | `examples/offline_inference.py` | 最小可运行入口，看懂参数与调用链 |
| 4 | `deepinfer/entrypoints/llm.py` | `LLM` 如何包 `LLMEngine`，`generate` 的批量循环 |
| 5 | `deepinfer/engine/llm_engine.py` | 引擎生命周期 + `step()` 主循环（860 行起） |
| 6 | `deepinfer/engine/arg_utils.py` | 所有可调参数的真实含义 |
| 7 | `csrc/engine/scheduler.hpp` | 调度器全貌：prefill/decode 投票、chunk、抢占 |
| 8 | `csrc/engine/block_manager.hpp` | 极简 paged KV 分配器（free list + refcount） |
| 9 | `csrc/engine/sequence.hpp` | `Sequence` / `KvCacheState` / `SequenceMetadata` 字段 |
| 10 | `deepinfer/worker/worker.py` | `execute_model_step` 四段流水线（89 行起） |
| 11 | `deepinfer/worker/model_runner.py` | forward/sample 分离、CUDA Graph、DFlash 分支 |
| 12 | `deepinfer/worker/cache_engine.py` | paged block + state slot + FP4 布局（978 行） |
| 13 | `deepinfer/model_executor/models/deepseek.py` | V4 主干：embed → layers → head，YoCo 边界切换 |
| 14 | `deepinfer/model_executor/models/v4_modules/attn.py` | MLA + 压缩 KV + 稀疏索引 + SWA + 跨层共享 |
| 15 | `deepinfer/model_executor/layers/sampler.py` | 采样、投机验证、SO 掩码、batch-invariant |

---

## 3. 一次请求的完整生命周期

### 3.1 启动阶段

```
examples/offline_inference.py
  └─ launch_dp(main, dp_size)                 # 每个 DP rank 一个进程
      └─ create_llm(...)                      # examples/_config.py
          └─ LLM(...)                         # entrypoints/llm.py
              └─ LLMEngine.__init__           # engine/llm_engine.py:155
                  ├─ _init_workers()          # 1 个 driver + (cp_size-1) 个 spawn worker
                  ├─ load_model               # model_loader.py（meta device 建模型再灌权重）
                  ├─ _init_cache()            # 先 profile 显存，再算 num_gpu_blocks
                  ├─ BlockManager(num_gpu_blocks)
                  ├─ Scheduler(...)           # C++，传入 block_manager / DP group / FlopsEstimator
                  └─ warm_up_model            # CUDA Graph 捕获
```

关键点：
- **每个 DP rank 是一个独立进程**，进程内 driver worker 直接跑模型，CP rank 用 `SpawnProcessWithCrashHeader` 起子进程。
- `_init_cache()` 先跑 `profile_num_available_blocks` 量显存，再把剩余显存切成 `num_gpu_blocks × block_size`，同时按 `max_num_seqs` 切 state slots。
- 模型构建走 `build_model_on_meta` + `load_hf_checkpoint`，说明它对权重加载做了**零拷贝/meta 初始化**优化。

### 3.2 请求入队

```
LLM.generate(prompts, sampling_params)
  └─ tokenizer.encode → prompt_token_ids (np.int32)
      └─ engine.add_request(user_id, request_id, sampling_params, prompt_token_ids, ...)
          ├─ SamplingParams 校验（eos_id、so_args、seq_seed）
          ├─ CompressedTokenizer.encode()      # Engram 词表压缩（启用时）
          ├─ Sequence(...)                     # C++ 对象，含 tokens/cache/mtp/so 状态
          ├─ so_args != None → SOMatcherBuilder.submit()   # 异步编译 grammar
          └─ scheduler.add_sequence(seq)
```

注意：**prompt 必须由调用方先 tokenize**，`add_request` 接收的是 `np.ndarray[np.int32]`，不是字符串。这是"离线批量"定位的体现。

### 3.3 每个 step 的执行

```
LLMEngine.step()                                   # llm_engine.py:860
├─ scheduler.schedule(sequence_metadata_pool)       # C++：返回 (seq_metadata_list, SchedulerOutputs)
├─ clone_and_zero_states_and_blocks_on_default_stream()   # 分支请求的 COW
├─ post_schedule(outputs)                           # 子类钩子（PD 分离用）
├─ [decode] do_copy_state()                         # 需要落盘的 state 拷贝
├─ multi_modal_processor.fetch()                    # 异步取图像 embedding
└─ _execute_model_async_step()
    └─ worker.execute_model_step(is_prompt, ...)    # worker/worker.py:89
        ├─ 1. forward 当前 step（主模型，可能走 CUDA Graph）
        ├─ 2. post-process 上一步（ZOS 重叠的关键）
        ├─ 3. sample 当前 step（fused_sample kernel）
        ├─ 4. prompt 步 / 非 ZOS：立刻 post-process 当前步
        │     ZOS decode：把 curr_step_state 留到下一步（与 forward 重叠）
        └─ 返回 request_outputs（uint64 指针数组）
```

`post_process` 回调（`llm_engine.py:984`）做四件事：处理 sampler 输出 → 统计 → `pre_sampler_sync` 释放资源 → 收集要延后释放的 metadata。
其中 `DeferredRelease` 很典型：**本步产出的 CPU 资源不在本步释放，而是挂到下一步 `pre_sampler_sync` 再释放**，保证调用方读完之前不会被回收，同时把释放开销藏进 GPU 计算里。

### 3.4 输出回传

- Sampler 输出不是 Python 对象，而是 `np.ndarray[np.uint64]`（指针数组），指向 `RequestOutputPool` 里的对象；下一步自动回收，需要保留就 `pool.clone()`。
- 这种"指针 + 对象池 + 延后释放"的设计，是为了在 decode 高频小 step 下**消掉 Python 对象分配与 GC 抖动**。读 vLLM 时你看到的是 `RequestOutput` 列表，这里是显式的内存管理。

---

## 4. 核心机制逐个拆

### 4.1 三种引擎与 EPD 分离

`deepinfer/engine/engine_type.py` 与 `csrc/engine/scheduler.hpp` 的 `enum class EngineType` 严格对应：

| 类型 | 行为 | 典型部署 |
| --- | --- | --- |
| `NORMAL` | 既做 prefill 又做 decode | 单机小规模、开发调试 |
| `PROMPT` | 只做 prefill，算完把 prompt 发给 completion | prefill 集群（EP2–8 / DP2–8） |
| `COMPLETION` | 只做 decode，不接收新 prompt | decode 集群（EP64–128） |

- `COMPLETION` 引擎在构造时被强制要求 `stop_preempted_request=true`，因为它没有 prefill 通道，无法重算被抢占的请求（`scheduler.hpp` 里有 `TORCH_CHECK`）。
- 这套拆分在**同一个 `LLMEngine` 类**里，通过虚函数钩子（`init_pd_split`、`do_save_kvcache`、`send_processed_prompts`…）实现，而不是三套代码。这是很值得学的工程手法：**用钩子做形态分化，而不是继承爆炸**。
- 多模态还有第四种角色：Vision Encoder（VE），单独一个进程/GPU，`deepinfer/ve/` 里做图像预处理，`MultimodalProcessor` 异步 fetch。

生产拓扑（`doc/images/architecture_overview.png`）：

```
用户 JSON → DeepSeek Recipe → Entrypoint ×3
                                 ├→ Vision Encoder LB → VE Service
                                 ├→ Prefill LB → Prefill Service ×3 ──EPLB──┐
                                 └→ Decode LB  → Decode Service ×3 ──EPLB──┤
                                                                            └→ 3FS KV Cache
```

### 4.2 C++ 调度器：它到底在调度什么

入口：`Scheduler::schedule(SequenceMetadataPool&)`，核心状态是 `waiting` / `running` / `prompted` 三个队列。

设计要点（都能在 `scheduler.hpp` 里找到）：

1. **prompt_limit 两态**
   `prompt_limit = chunked_prefill ? max_model_len : min(max_model_len, max_num_batched_tokens)`。
   不开 chunked prefill 时，单条 prompt 不能超过 batch token 预算。

2. **prefill 与 decode 的"投票"**
   NORMAL 引擎里，prefill 不是每步都做，而是在 decode 的第 N 步尝试：
   - `prefill_attempt_interval`：**ALL 点**，所有 DP rank 都有活才一起 prefill；
   - `prefill_any_attempt_interval`：**ANY 点**，任一 rank 有活就集体 prefill，防止饿死。
   用 `dp_group` 做 `all_reduce(MIN/MAX)`（CPU group 可用 gloo）。这是**多 DP rank 同步进入 prefill** 的关键，否则一个 rank 做 prefill 时其他 rank 空转。

3. **chunked prefill 的切分**
   先按 `max_num_batched_tokens`、`prefill_max_sum_seqlen_k`（所有序列 KV 长度之和上限）、`max_num_seqs`、block 容量逐层裁剪，再调 `_adjust_chunk_size()` 对齐：
   - 不能切在 image pad 区域中间（多模态）；
   - 对齐 `cp_align = cp_size*2`；
   - 支持 `chunk_checkpoints` 强制在指定位置断开（benchmark/测试用）。

4. **抢占**
   `running` 队列超限或 block 不够时，按策略抢占：`stop_preempted_request=true` 直接终止，否则重置 `cache` 回 waiting 重算。`reset_for_recompute()` 会把 `num_tokens/previous_num_tokens/swa_valid_start` 一起清掉。

5. **成本模型**
   `FlopsEstimator`（`csrc/engine/flops_estimator.hpp`）按 FP8 等价 FLOPs 估算 attention + 非 attention + logits，用于 MFU 统计与调度决策。
   **注意**：Q&A 里说的"按 compute_quota（纳秒预算）+ 二次 cost 模型解 chunk 大小"属于 **Prefill Load Balancer**，不在本仓库；本仓库只有单引擎内的预算裁剪。这是开源版与生产版最容易被误读的差别之一。

6. **BlockManager 极简**
   `csrc/engine/block_manager.hpp`：一个双向链表 free list + refcount，`0` 号块保留不用。没有 radix tree、没有 hash、没有 LRU。
   **引擎内没有自动前缀缓存**：`KvCacheState.hit_tokens` 由外部（Entrypoint/KV store）注入；引擎只处理"已经有多少 token 的 KV 在显存里"。分支请求（`branch_id` 以 `_` 开头）会 pin 父请求的 block，做 copy-on-write。
   → 对比 vLLM 的 automatic prefix caching / SGLang 的 RadixAttention，这是理解 DeepSeek 架构的关键差异：**跨请求前缀复用在引擎外，靠独立 KV 存储集群解决**。

### 4.3 ZOS：零开销异步调度

问题（Q&A 原话）：decode 一步是 `schedule → forward → sample → post-process`，其中 schedule 和 post-process 是纯 CPU，且 post-process 前必须等 sample 的 GPU→CPU 同步。GPU 在这两段空转。

解法（`worker.py:89` 的四段注释就是实现说明）：

```
GPU:  forward(N) ─────────────► sample(N) ──────────► forward(N+1)
CPU:  schedule(N+1) → launch_forward(N+1) → post_process(N-1) → launch_sample(N+1)
```

配套机制：
- `WorkerStepState` 保存上一步的 `seq_metadata_list` / `scheduler_outputs` / `sampler_output` / event；
- 下一步用 `prev_step_state` 做 post-process，`sample_end_evt` 保证只在真正需要时才同步；
- `DeferredRelease` 把资源释放也挪到下一步；
- `placeholder step`：没活干时也发一个占位 forward，保持 DP rank 锁步、避免 NCCL 超时；
- `MetadataZOS` / `prepare_zos_inputs`：C++ 侧预先把 metadata 准备到 pinned buffer，GPU 侧 kernel 直接读；
- `DEEPINFER_ZERO_OVERHEAD_SCHEDULING` 默认开（NORMAL/COMPLETION）。

代价/约束：
- decode 的 `seqlens_q` 在调度时只能"乐观估计"，要靠 `_correct_seqlens_qk()` 用 D2H 回来的真实值修正；
- 因此 `DEEPINFER_ENABLE_MTP_VARLEN` 下的变长验证需要 GPU 侧重排（见 4.8）；
- prompt step 不走 ZOS（`is_prompt` 时当场 post-process）。

### 4.4 YoCo / CED 与 bounded replay（这个项目最"邪门"的优化）

背景（代码 + 外部资料）：V4.1-Flash 是 **Causal Encoder-Decoder（CED）** 架构，前 `n_prefill_layer` 层是 encoder，后面的层是 decoder/consumer。代码里只认 `n_prefill_layer` 这个参数；官方/第三方资料给出的 V4.1 取值是 40 层 = 20 encoder + 20 decoder。decoder 的 global KV 由 encoder 最后一层 hidden state 投影而来，所以 **prefill 只需要跑 encoder 层**。

代码里的对应：
- `cfg.n_prefill_layer < cfg.num_hidden_layers` 即启用 YoCo（`v4_modules/yoco.py:yoco_enabled`）；
- `DeepseekModel.forward` 在 `i == yoco_boundary_layer_idx` 时把 `input_metadata` 切成 consumer view（compact 模式）或 mask 视图；
- `CrossLayerRuntime` 负责跨层传递 `shared_hidden` / `pre_rope_kv` / `topk`。

**Decoder bounded replay**（Q&A 的数字）：
- 想精确重建 decoder 的 SWA KV，需要把最后 2,560 token 跑过 20 个 decoder 层；
- 实际只重算**最后 128 token**，并且把 query 的可见范围截断到 `max(s, i-127)`，得到一个近似 SWA KV；
- 结果只用于 decode，不进入 prefix cache；
- 默认配置：`DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact`、`DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140`（128 窗口 + 6 回溯 + 余量）；
- 收益：prefill 计算量接近减半（Q&A 原文 "nearly halving total prefill computation"）。

**命中策略**（Q&A，值得背下来）：
1. 输入长度 L 的命中上限是 `L - 134`；
2. 先查 block KV，得到 L1（128 的倍数）；
3. 再查 SWA KV：若 `[L1-128, L1]` 有 SWA 则命中 L1，否则回退一个 block 到 L1-128；
4. SWA 只存 `20 层 × 134 token × 528 B ≈ 1.35 MB/序列`，放在 host memory 的 `StateBlock` 里，不进 3FS。

代码入口：`deepinfer/worker/cache_engine.py` 的 `StateBlock` / `get_inference_swa_block_size` / `select_state_slice_to_store`；kernel 在 `tilelang_ops/{cuda,asc}/state_cache.py`、`swa_index.py`。
验证工具：`examples/yoco/yoco_equiv.py`（L1 单卡秒级，不需要权重；L2 需要 8 卡 + checkpoint）。

### 4.5 CSA2：压缩 KV + 稀疏索引 + SWA + 跨层共享

`v4_modules/attn.py` 的 `V4Attention` 把多条路径揉在一个类里：

| 机制 | 代码线索 | 作用 |
| --- | --- | --- |
| MLA | `mla_attn.py`、`q_lora_rank` / `o_lora_rank` | 低秩 Q/KV 投影，压 KV |
| 压缩 KV | `compress.py` 的 `Compressor` / `KProject` | `kv_compress_ratio` ∈ {0,1,2}（C1/C2），把 KV 压到更少 token |
| 稀疏索引 | `index_attn.py` 的 `MQA` + `index_topk` | 对压缩 KV 打分 `relu(q@k)*w`，top-k 选块，只算选中的块 |
| SWA | `use_swa`、`window_size` | 局部窗口注意力，和全局稀疏结果拼接 |
| 跨层共享 | `cross_layer.py`、`shared_topology` | Full/Reindex/Reuse 三种层模式，共享 main KV / indexer K / topk 索引 |
| 分层索引器 | `candidate_blocks.py`、`index_attn.py` | decoder 后段索引层只在 first-Full 层给出的候选池里选，限制深层索引成本 |

几个可验证的配置约束（`model_metadata_utils.py`）：
- 只支持 C1/C2 压缩，C4/C128 已删除；
- 不支持 compression overlap / compression APE；
- 所有 indexer 头数必须全局唯一（`assert len(heads) <= 1`）；
- Engram 层必须在 encoder 半区（YoCo compaction 下）。

**为什么值得研究**：这是"稀疏注意力 + 压缩 KV + 跨层复用"三者叠加的工程实现。大多数框架只做 MLA + paged KV；这里的 `index_topk`、`n_packed_heads`、`kv_compress_ratio` 是模型架构直接长进推理引擎的典型例子。

### 4.6 CacheEngine：两套缓存，不是一个 pool

`deepinfer/worker/cache_engine.py`（978 行）里有两类完全不同的缓存：

1. **Paged KV block**（`KVCache`）：按 `block_size` 切块，`block_table` 映射；主 KV 与 indexer KV 分开布局；支持 FP4/FP8 多种 format。
2. **State slot**（`StateBlock`）：每个序列一个固定大小 slot，存 SWA KV + 压缩状态（`ca_state` / `ica_state`）。按 `[stored region: SWA][dropped region: YOCO consumer SWA]` 切分。
   - 只有前 20 层的 SWA 需要持久化；
   - 只在 128 token 边界存；
   - decode 时在 device 上留一个 replica region，每跨 128 边界拷一次，结束时从 replica 取。

低精度格式（Q&A + 代码）：
- 主 KV：per-16 e4m3（类似 NVFP4，但没有 global scale）；
- Indexer KV：per-32 ue8m0（MXFP4）；
- MoE routed expert 权重：per-32 ue8m0（MXFP4）；
- `derive_main_kv_cache_num_per_channels` 里能看到 16/32 的分界。

**工程启示**：SWA 和 paged KV 的生命周期、存储介质、命中策略完全不同，硬塞进一个 block pool 会把逻辑搅烂。DeepSeek 选择"两套管理器 + 两套 ID 空间（`block_id` / `state_id`）"，这是很实用的设计。

### 4.7 EPLB：冗余专家 + 统计窗口 + 原地 shuffle

`deepinfer/model_executor/eplb/`：

- **冗余专家**：除逻辑专家外，每卡多放几个 hot expert 的副本；`resolve_num_extra_experts` 会把总数向上取整到能被 `ep_size` 整除。
- **统计**：`expert_count` 每 step 累加，`_expert_count_window` 保留最近 N step。
- **重排**：每 M step 调 `rebalance_experts_global` 重算 `physical_to_logical_map` / `logical_to_physical_map`，再用 `symm_mem` 在 NVLink 域内原地 shuffle 权重（`shuffle.py`），避免停机。
- **映射表**：dispatch / 路由 / sampler 都直接读这两张表，所以重排对上层透明。
- 生产参数（Q&A）：prefill 4 个冗余专家 / 100 step 窗口 / 1000 step 重排，`moe_lb≈1.045–1.074`；decode 1 个冗余 / 1000 step 窗口 / 10000 step 重排，`moe_lb≈1.09`。

**注意**：`moe_lb` 是"最慢 rank / 平均"，越接近 1 越好。这个指标比"平均负载"更能反映 EP 的真实瓶颈。

### 4.8 DSpark / DFlash：投机解码的"按边际收益分配算力"

代码位置：
- MTP head：`models/mtp/dflash.py`；
- 主模型侧接收 draft 的 target 层：`dflash_cfg.dspark_target_layer_ids`；
- 投机相关 kernel：`tilelang_ops/*/dflash.py`、`dflash_copy.py`、`store_mtp_info.py`、`schedule_mtp_varlen.py`；
- 采样验证：`layers/sampler.py:_verification()`；
- 运行时参数估计：`csrc/engine/mtp_varlen.hpp`（sampler/forward 耗时的滑动中位数估计）。

机制（Q&A）：
- draft 模型一次出 5 个 token，并给出每个 draft token 的 **acceptance confidence**；
- 按"期望接受率"排序，选一个前缀子集做验证，而不是固定验证 5 个；
- 生产数据：平均验证长度 5.00 → 3.2–4.4，接受长度基本不变（3.29 → 3.20），吞吐 +4–7%（60 并发/rank 时 +5.4%）；
- 与 ZOS 的冲突：`schedule(N+1)` 时第 N 步的 confidence 还没回来 → CPU 只定总预算，**per-request 长度由 GPU 上的 TileLang kernel 重算**（cumulative product + global sort top-k + fused remap scatter），写回 pinned buffer。

**这是全仓库最值得细读的"系统协同"案例**：算法（confidence head）→ 调度（变长预算）→ kernel（GPU 侧重排）→ 异步流水线（拿不到未来信息）四方妥协的结果。

### 4.9 Sampler：三个"确定性"目标

`layers/sampler.py` + `tilelang_ops/fused_sample.py`（832 行）+ `random_sampling.py`：

- **fused**：temperature、top-p/top-k、log-softmax、采样、可选 logprob 输出在一个 kernel 里完成，避免多次读写 logits（vocab 很大时这是主要开销）。
- **batch-invariant**（`DEEPINFER_BATCH_INVARIANT=1`，仅 NVIDIA）：不同 batch 组成下结果 bit 级一致。做法是固定 reduction 顺序、避免 batch 相关的 split-K。
- **deterministic sampling**：固定 `seq_seed` + 关 MTP varlen，同一输入同一参数输出完全一致。`_hash_to_uniform` 用整数 hash 而不是全局 RNG 状态，避免调用顺序影响结果。
- **SO mask**：`_get_so_mask` 把结构化输出的 vocab mask 直接喂进 fused sampler，避免单独一次全词表 mask 写回。

这几个特性对**评测复现**极其重要：`tests/models/offline_decode_ppl.json` 里存了每个 DP rank 的 PPL 金标准，测试逐样本精确比对。没有 batch-invariant，这种测试根本做不了。

### 4.10 结构化输出：编译与推理解耦

`deepinfer/structure_output/`：

```
so_args={"schema_type": "json"|"regex"|"ebnf", "schema": ...}
  → SOMatcherBuilder.submit()        # 独立线程池（绑 idle CPU）
      → XGrammarCompiler（xgrammar 0.2.2）编译成 grammar automaton
          → XGrammarMatcher            # 每个请求一个
              → fill_vocab_mask()      # 每 step 推进自动机，产出 mask
                  → fused sampler 应用 mask
```

关键设计：
- 编译是**异步**的，请求在 `add_request` 阶段就能提交，引擎在 `pre_sampler_sync` 里 `pump()` 收割；
- 编译失败/超时只杀该请求（`so_status=-1` → `FINISHED_SO_FAILED`），不影响引擎；
- `DEEPINFER_XGRAMMAR_*` 一组环境变量控制 cache、超时、日志；
- 支持 rollback（投机解码被拒时回退自动机状态）。

### 4.11 双硬件后端：一套代码，两套 kernel

- `deepinfer/platform.py`：`Platform` 抽象（`tilelang_ops_backend = "cuda"|"asc"`、`distributed_backend = "nccl"|"hccl"`、`fp8_sf_type`）。
- `tilelang_ops/{cuda,asc}/` 各 30 个算子文件，**同名同签名**，用 TileLang 写两遍；上层只 `from .ops import ...`。
- `deep_cluster/hardware/{nvidia,ascend}.py`：只用 procfs/sysfs，不 import torch，避免初始化 CUDA。
- `npu_ops/`：DeepSelect 等 NPU 侧的 patch。

**启示**：跨厂商不是"写个 if/else"，而是**把算子层做成可替换后端 + 统一签名**。代价是每个算子要维护两份（约 1.1 万行重复实现）。

---

## 5. 性能数据：可以引用的锚点

来自仓库里的 Pareto 图（`doc/images/*.png`），配置写在图标题里，可直接引用：

### 5.1 Prefill（`prefill_pareto.png`）

配置：EP4DP4，384+32 experts，context 131,072，Decoder Bounded Replay (140 tokens)。

| batch (tokens/DP rank) | 4,096 | 8,192 | 16,384 | 32,768 | 65,536 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 吞吐 (tokens/s/GPU) | ~60k | ~72k | ~81k | ~86k | ~88k |
| 延迟 (ms/step) | ~70 | ~115 | ~200 | ~370 | ~730 |

解读：**2 万 token 以后曲线就平了**——prefill 不是靠堆 batch 提升，而是靠 YoCo 截断把"每 token 的计算量"降下来。这是 CED 架构的收益，不是调度器的收益。

### 5.2 Decode（`decode_pareto.png`）

配置：EP64DP64，384+64 experts，context 131,072，nextn=3 fixed，接受率 75% → τ=3.25。

| 并发 (req/DP rank) | 8 | 16 | 32 | 64 | 128 | 256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 吞吐 (tokens/s/GPU) | ~2.5k | ~5k | ~9.5k | ~15.5k | ~22.5k | ~26k |
| 单用户速度 (tokens/s/user) | ~325 | ~315 | ~298 | ~249 | ~180 | ~104 |

解读：**吞吐与交互性是硬 trade-off**；254 req/rank 时每 GPU 26k tok/s，但单用户只有 104 tok/s（约 50ms/token，接近可接受上限）。生产上按 SLA 选点，不是无脑堆并发。

### 5.3 Wide-EP 的收益（`wide_ep_decode_pareto.png`）

同 context 131k、nextn=3、75% 接受率下：

| 单用户速度 | EP8DP8 | EP64DP64 | 倍数 |
| --- | ---: | ---: | ---: |
| ~175 tok/s | ~15k | ~22.5k | 1.5× |
| ~250 tok/s | ~8k | ~16k | 2.0× |

结论：**EP 越大，单步 MoE 的每卡访存量越小**，在低交互性区间收益更明显。这就是 Q&A 里"decode 用能拿到的最大的 EP"的量化依据。

> 注意：这些数字是 **SM100 + 该模型 + 该配置**下的结果，不能直接拿去和 H100 上的 vLLM 比。跨硬件/跨模型的吞吐对比基本没有意义，除非你能控制变量。

---

## 6. 与 vLLM / SGLang / TensorRT-LLM / LMDeploy 对比

### 6.1 血统关系

```
vLLM ──(fork/改造)──► DeepInfer ──(借鉴异步调度)──► SGLang 的 overlap scheduler
  │                        ▲
  └──(图优化/编译思路)──────┘ TensorRT-LLM
```

代码证据：`llm_engine.py`、`model_runner.py`、`worker/base.py` 等文件头都是 `Derived from vLLM; modified by DeepSeek.`；Q&A 明确写异步调度"Following existing implementations (SGLang, NanoFlow)"。

### 6.2 一张大表

| 维度 | **DeepInfer** | **vLLM (V1)** | **SGLang** | **TensorRT-LLM** | **LMDeploy** |
| --- | --- | --- | --- | --- | --- |
| 定位 | DeepSeek 生产框架开源子集 | 通用高吞吐推理引擎 | 通用高性能 serving | NVIDIA 官方推理栈 | 轻量部署工具链 |
| 支持模型 | **仅 V4.1-Flash** | 极广（HF 生态） | 很广 | 广（需转换/编译） | 较广 |
| 硬件 | SM100 / Ascend 950 | NVIDIA/AMD/TPU/CPU/XPU | NVIDIA/AMD | NVIDIA | NVIDIA 为主，含昇腾等后端 |
| 并行 | EP + DP + CP，**无 TP** | TP/PP/DP/EP | TP/DP/EP/PP | TP/PP/EP | TP/DP |
| 调度器 | **C++**（csrc） | Python（V1 统一 token 预算） | Python + overlap scheduler | C++ runtime | Python + TurboMind |
| Prefill/Decode | **原生 EPD 三引擎** | 统一调度；PD 分离靠 connector | 统一调度；PD 分离可选 | in-flight batching；PD 可选 | persistent batching |
| 前缀缓存 | **引擎外**（KV store + 注入 hit_tokens）；引擎内只支持 fork/COW | 自动前缀缓存（block hash） | RadixAttention（基数树） | 需显式配置/自建 | 有 prefix cache |
| KV 精度 | FP4 主 KV / MXFP4 indexer / MXFP8 SWA | FP8/FP16，社区 FP4 支持 | FP8/FP16 | FP8/FP4（Blackwell） | INT4/INT8 KV（部分） |
| 投机解码 | **DSpark + confidence 变长验证** | 多种 draft 方法，固定长度为主 | EAGLE/草稿模型 | Medusa/EAGLE/draft | 有 |
| 结构化输出 | xgrammar，**编译异步化 + mask 融进 sampler** | xgrammar/outlines | xgrammar/outlines | guided decoding | 有 |
| 在线服务 | **无**（离线 `LLM`） | OpenAI server + async engine | OpenAI server + gRPC | Triton backend / server | api_server |
| 可扩展性 | 低（模型特化） | 高（插件/自定义模型） | 高 | 中（需重编译） | 中 |
| 上手成本 | 极高（SM100 + 专用 ckpt） | 低 | 低 | 中 | 低 |
| 代码成熟度 | v0.1.0，4 commits | 成熟 | 成熟 | 成熟 | 成熟 |

> 表中竞品信息基于 2026 年公开文档的共性描述，具体版本差异请以各自最新文档为准；DeepInfer 列为代码实证。

### 6.3 五个关键差异，展开说

**① 调度器放 C++，是性能选择也是组织选择**
vLLM V1 把调度器放 Python，靠"unified scheduler + 极简 per-step 开销"达到 near-zero CPU overhead；DeepInfer 直接把 scheduler/block manager/sequence 全搬进 C++，Python 只传 numpy 数组和指针。代价是调试难、迭代慢（改调度要重编译），收益是 decode 高频小 step 下没有 Python 对象开销。
**判断**：只有"模型固定、流量巨大、值得为每微秒优化"的场景才值得这么做。通用框架学不来。

**② 前缀缓存的位置相反**
vLLM/SGLang 在引擎内做自动前缀复用（block hash / radix tree），因为它们的部署假设是"单实例、显存即缓存"。
DeepInfer 把前缀缓存推到**独立 3FS 存储集群**，引擎内只处理"已经命中多少 token"，原因是：跨机器任意前缀命中 + 存储与计算解耦 + 计算节点故障不影响缓存。代价是引入了 Entrypoint、KV store、网络拉取一整套组件（都没开源）。
**判断**：这是"数据中心级"设计，不是"单机引擎"设计。读的时候要分清哪些代码是引擎、哪些是系统。

**③ 并行策略：EP 优先，放弃 TP**
vLLM/SGLang 的默认心智是 TP（单机 8 卡切模型）；DeepInfer 全仓库没有 TP，decode 直接上 EP64–128。原因是 MoE 模型里 TP 会把 expert 切碎、通信放大，而 EP 只传 token 不传权重，配合 EPLB 能把 `moe_lb` 压到 1.09。
**判断**：MoE + 超大 EP 是 DeepSeek 的路线，不是所有模型的路线。Dense 模型上这套不成立。

**④ 投机解码：从"固定长度"到"按边际收益分配"**
主流框架的投机解码基本是固定 draft 长度（如 5），验证成本固定。DSpark 用 confidence head 预测每个 draft token 的接受率，按期望收益排序后选前缀子集，把验证算力花在"最可能被接受"的 token 上。
**判断**：这是"算法-系统协同"的典型，单看推理框架代码看不懂，必须结合模型结构（`dflash_cfg`、target layer ids）一起读。

**⑤ 工程形态：一个模型，两套硬件，三种引擎，零在线服务**
vLLM/SGLang 是"一个引擎适配 N 个模型"；DeepInfer 是"一个模型榨干 N 种硬件"。前者的复杂度在抽象层，后者的复杂度在 kernel 和调度细节。开源版还刻意切掉了在线服务层，所以你看到的是一个**批处理引擎**，不是 serving 系统。

### 6.4 什么时候选谁（务实版）

- 你要**部署 DeepSeek-V4.1-Flash 且手里有 SM100/Ascend 950**：DeepInfer 是唯一能吃到 YoCo/CSA2/FP4 KV 全部收益的选择。
- 你要**部署任意 HF 模型、要 OpenAI API、要 LoRA/量化/多租户**：vLLM 或 SGLang。
- 你要**NVIDIA 上极致低延迟、能接受模型编译**：TensorRT-LLM。
- 你要**快速起一个中文社区友好的 serving**：LMDeploy。
- 你要**学习推理系统设计**：DeepInfer 的 Q&A + C++ scheduler + ZOS + YoCo 是最佳案例库，但别拿它当通用框架用。

### 6.5 不要做的对比

- ❌ 拿 README 的 26k tok/s/GPU 和 H100 上 vLLM 的数字比 —— 硬件、模型、精度全不同。
- ❌ 拿 `moe_lb=1.09` 和 vLLM 的 EP 负载比 —— 指标定义、EP 规模、EPLB 策略都不同。
- ❌ 说"DeepInfer 比 vLLM 快 N 倍" —— 官方从未给出跨框架对比；它的基线是"自己内部的完美负载均衡离线 benchmark 的 80%"。
- ✅ 可以比的是**设计选择**：调度器语言、前缀缓存位置、并行策略、投机验证策略、KV 精度。

---

## 7. 四周学习路线（每周 8–10 小时）

> 前提：你已经懂 Transformer 推理基础（prefill/decode、KV cache、PagedAttention、连续批处理）。不懂的话先补 vLLM 的 PagedAttention 论文和 vLLM V1 文档，否则读这个项目会卡在术语上。

### Week 1：骨架与数据流（目标：能画出端到端时序图）

| 天 | 任务 | 产出 |
| --- | --- | --- |
| D1 | 读 README + Q&A 全文，标出所有数字和"没开源"的组件 | 一页系统架构笔记 |
| D2 | 读 `examples/offline_inference.py` + `entrypoints/llm.py` + `arg_utils.py` | 参数表：每个参数影响什么 |
| D3 | 读 `llm_engine.py` 的 `__init__` / `add_request` / `step` / `post_process` | 画出对象关系图（Engine/Worker/Scheduler/BlockManager） |
| D4 | 读 `worker.py:execute_model_step` + `worker/base.py` | 画出 CPU/GPU 四段流水线时序图 |
| D5 | 读 `model_runner.py` 的 `execute_main_model` / `execute_sample` | 标出 CUDA Graph 的捕获与回放点 |
| D6 | 用 grep 追一遍 `DEEPINFER_*`（64 个），分类：调度/缓存/kernel/调试 | 环境变量速查表 |
| D7 | 复盘：合上代码，手画"一个请求从 add_request 到输出"的完整路径 | 自测：能说出每步在哪个文件 |

**Week 1 关键问题**：
- 为什么 `add_request` 要收 `np.int32` 而不是字符串？
- `DeferredRelease` 到底在防什么？
- `placeholder step` 不产生 token，为什么还要跑 forward？

### Week 2：调度与缓存（目标：能预测调度行为）

| 天 | 任务 | 产出 |
| --- | --- | --- |
| D1 | `csrc/engine/sequence.hpp`：`Sequence` / `KvCacheState` / `SequenceScheduleState` 字段 | 字段表 + 生命周期 |
| D2 | `block_manager.hpp` 全文（很短）；理解 free list、refcount、COW | 手写一个 block 分配模拟 |
| D3 | `scheduler.hpp` 的 prefill 路径（355–520 行） | 画 chunk 裁剪流程图 |
| D4 | `scheduler.hpp` 的 decode 路径 + 抢占 + DP 投票 | 解释 ALL/ANY 两个 interval |
| D5 | `flops_estimator.hpp` + `stats_logger.py` | 搞懂 MFU/attn_tflops 怎么算 |
| D6 | `cache_engine.py`：`KVCache` vs `StateBlock`，FP4 布局，`select_state_slice_to_store` | 两套缓存的对比表 |
| D7 | 读 `eplb/` + `policy.py` + `shuffle.py` | 解释 `moe_lb` 与冗余专家的关系 |

**Week 2 关键问题**：
- 引擎内没有 hash 前缀缓存，那 `hit_tokens` 从哪来？fork 请求的 COW 怎么做？
- `prefill_max_sum_seqlen_k` 防的是什么 OOM？
- 为什么 COMPLETION 引擎必须 `stop_preempted_request=true`？
- SWA 为什么不能进 paged block？

### Week 3：模型与 kernel（目标：看懂 V4 的"特化"在哪）

| 天 | 任务 | 产出 |
| --- | --- | --- |
| D1 | `models/deepseek.py` 全文（525 行）：embed → layers → head，YoCo 边界 | 画模型 forward 的数据流 |
| D2 | `v4_modules/attn.py`：MLA、压缩 KV、稀疏、SWA、跨层共享 | 一张"注意力分支决策树" |
| D3 | `v4_modules/index_attn.py` + `compress.py` + `candidate_blocks.py` | 解释 top-k 选块与候选池 |
| D4 | `v4_modules/yoco.py` + `input_metadata/yoco.py` + `examples/yoco/README.md` | 画 mask vs compact 两种视图 |
| D5 | `v4_modules/{engram,hc,mhc_ops,cross_layer}.py`（先看 docstring） | 每个模块一句话 + 参数量位置 |
| D6 | `layers/tilelang_ops/fused_transform.py`（1330 行）：norm+rope+hadamard+cast+store 融合 | 列出融合了哪几步、省了哪些 HBM 读写 |
| D7 | `layers/tilelang_ops/fused_sample.py` + `sampler.py` | 解释 batch-invariant 和 SO mask 怎么融进去 |

**Week 3 关键问题**：
- 为什么 prefill 只跑 encoder 层？（CED 的 KV 从哪来？）
- `n_prefill_layer` 在 forward 里怎么改变 `input_metadata` 的布局？
- Engram 的 n-gram hash 表为什么必须放在 encoder 半区？
- `fused_transform` 一个 kernel 干了 5 件事，收益来自减少 HBM 往返还是减少 launch？

### Week 4：专题 + 对比 + 实验（目标：形成自己的判断）

| 天 | 任务 | 产出 |
| --- | --- | --- |
| D1 | `models/mtp/dflash.py` + `sampler._verification` + `schedule_mtp_varlen.py` | 画 DSpark 的 draft→verify→accept 流程 |
| D2 | `structure_output/` 全部（595 行，很少） | 画 SO 的异步编译时序 |
| D3 | `ve/` + `vision_encoder/` + 多模态 metadata | EPD 里 VE 的角色 |
| D4 | `examples/benchmark.py` + `sweep/` | 搞懂 prefill/decode 解耦测量、为什么要"烧 step" |
| D5 | 写一页对比：DeepInfer vs vLLM（只比设计，不比数字） | 对比笔记 |
| D6 | 选一个专题深挖（建议 ZOS 或 YoCo），写成 2000 字技术博客 | 博客草稿 |
| D7 | 自测 20 题（见第 9 节），不会的回查代码 | 错题清单 |

---

## 8. 动手实验清单（按硬件分三档）

### A 档：没有 GPU（macOS 也能做，占学习价值 60%）

1. **画时序图**：纯读代码，产出 `step()` 的 CPU/GPU 甘特图。
2. **静态追踪脚本**：写 Python 解析 `csrc/engine/scheduler.hpp`，提取所有 `TORCH_CHECK`，整理成"调度器不变量清单"。
3. **环境变量普查**：`grep -rho 'DEEPINFER_[A-Z0-9_]*' deepinfer csrc | sort -u`，给每个变量写一句"作用 + 默认值 + 风险"。
4. **依赖图**：`python -c` 解析 import，画出 `engine → worker → model_executor → layers` 的层次，找出反向依赖（有没有下层 import 上层）。
5. **读测试当文档**：`tests/engine/test_sequence.py`、`tests/models/test_batch_invariant.py`、`tests/worker/test_state_cache_ops.py`，测试往往比注释更准确。
6. **对照读 vLLM**：把 `llm_engine.py` 和 vLLM V1 的 `engine/core.py` 并排读，列出"改了哪些、为什么"。
7. **YoCo L1 等价性**：`python examples/yoco/yoco_equiv.py mask-compact --keep-tokens 140 --window 128`。理论上单卡、不加载权重，但需要能 import `deepinfer._C`（即需要先 build），macOS 上不行；可以退化为**读这个脚本**，它把 mask/compact 两种 metadata 的差异写得很清楚。

### B 档：单机 H100/A100（能跑通一部分，但会撞墙）

- 可以：读代码、跑纯 Python 单测（不依赖 `_C` 的部分）、用 vLLM 跑同规模模型做"设计对照"。
- 会失败：`get_bf16_tflops_limit()` 对非 SM100 直接 `RuntimeError("unknown GPU type")`；DeepGEMM/FlashMLA 的 FP4 路径、TileLang SM100 算子、CUDA 13 依赖都不匹配。
- 不建议在这档上硬改，投入产出比很低。

### C 档：B200/GB200 或 Ascend 950（完整体验，占学习价值 40%）

```bash
# 0) 补全 submodule（普通 clone 不会拉）
cd ~/DeepInfer
git submodule update --init --recursive

# 1) 环境：Linux + GCC>=13 + Python 3.12 + CUDA 13（或 CANN 9.2）
python3.12 -m venv .venv && source .venv/bin/activate
export DEEPINFER_ACCELERATOR=nvidia   # 或 ascend
./build.sh                             # 装依赖 + 编 vendor + 编 _C

# 2) 跑通离线推理（会下载 ~几百 GB 权重，注意磁盘）
python examples/offline_inference.py --ep 8
python examples/offline_inference.py --ep 2 --dp 8
python examples/offline_inference.py --ep 2 --cp 8

# 3) 关掉投机，看 baseline
DEEPINFER_USE_NEXTN_DECODE=0 python examples/offline_inference.py --ep 8

# 4) 开 batch-invariant + 确定性采样
DEEPINFER_BATCH_INVARIANT=1 DEEPINFER_ENABLE_MTP_VARLEN=0 python examples/offline_inference.py --ep 8

# 5) 结构化输出自检
python examples/offline_inference.py --ep 8 --check_structured_output

# 6) benchmark + profile
export WORLD_SIZE=1 RANK=0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python examples/benchmark.py
PROFILE=1 python examples/benchmark.py --decode_num_requests 64 --context_length 32768

# 7) 扫 Pareto
bash examples/sweep/run_sweep.sh decode ep64-decode-128k 131072 3 0.75

# 8) 回归测试（会重跑 PPL 金标准）
python -m pytest -v -s tests/models/test_offline_inference.py
```

**建议的对照实验**（每个都只改一个变量）：
| 实验 | 变量 | 观察 |
| --- | --- | --- |
| E1 | `DEEPINFER_USE_NEXTN_DECODE=0/3/5` | accept_rate、decode tok/s、SPS |
| E2 | `DEEPINFER_ENABLE_MTP_VARLEN=0/1` | 验证长度、吞吐增益（对照 Q&A 表格） |
| E3 | `--disable_eplb` | `moe_lb`、decode tok/s |
| E4 | `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=off/mask/compact` | prefill 延迟、输出是否等价 |
| E5 | `--cp 1/2/8` | prefill 吞吐、通信开销 |
| E6 | `DEEPINFER_ZERO_OVERHEAD_SCHEDULING=0/1` | SPS、CPU 时间占比（trace 里看 bubble） |
| E7 | `DEEPINFER_BATCH_INVARIANT=0/1` | 输出是否逐 bit 一致（用固定 seed） |

> 注意：E4 的 mask/compact 等价性必须用 `examples/yoco/yoco_equiv.py` 验证，不能靠肉眼比文本。

---

## 9. 自测 20 题（能答上来就算读懂了）

1. `NORMAL` / `PROMPT` / `COMPLETION` 三种引擎的差别？为什么 COMPLETION 不能重算被抢占的请求？
2. `prompt_limit` 在 chunked prefill 开/关时分别是什么？
3. `prefill_attempt_interval` 与 `prefill_any_attempt_interval` 分别解决什么问题？
4. 为什么 DP rank 之间要用 `all_reduce(MIN/MAX)` 决定这一步做不做 prefill？
5. `DeferredRelease` 延后释放哪些资源？为什么不在本步释放？
6. ZOS 的四段流水线具体是什么？`prev_step_state` 起什么作用？
7. ZOS 下 `seqlens_q` 为什么要"乐观估计 + D2H 修正"？
8. 引擎内为什么没有 radix tree / block hash？`hit_tokens` 从哪来？
9. 分支请求（`branch_id` 以 `_` 开头）的 COW 是怎么做的？
10. `BlockManager` 的 0 号块为什么不用？
11. Paged KV 与 State slot 的生命周期、存储介质、命中策略有何不同？
12. SWA KV 为什么只在 128 token 边界存？decode 时的 replica region 干什么用？
13. YoCo 的 `n_prefill_layer` 如何改变 `DeepseekModel.forward` 的 metadata 布局？
14. Decoder bounded replay 为什么只重算 128 token？近似误差如何被限制？
15. 命中策略里 `L - 134` 的 134 是怎么来的？
16. CSA2 的 Full/Reindex/Reuse 三种层模式分别共享什么？
17. 分层稀疏索引器如何把深层索引成本与 context 长度解耦？
18. EPLB 的冗余专家数量为什么要向上取整到 `ep_size` 的倍数？
19. DSpark 的 confidence 与 ZOS 为什么冲突？GPU 侧 kernel 具体做什么？
20. batch-invariant 和 deterministic sampling 分别保证什么？各自的前置条件？

---

## 10. 术语表

| 术语 | 含义 |
| --- | --- |
| **EP** | Expert Parallel，专家并行；MoE 专家分散到多卡 |
| **DP** | Data Parallel，数据并行；每个 rank 跑完整模型的不同请求 |
| **CP** | Context Parallel，上下文并行；prefill 时切序列 |
| **Wide-EP** | 大规模 EP（64–128），decode 用 |
| **EPLB** | Expert Parallel Load Balancing，专家负载均衡 |
| **PD / EPD** | Prefill-Decode / Encoder-Prefill-Decode 分离部署 |
| **VE** | Vision Encoder，视觉编码器 |
| **ZOS** | Zero-Overhead Scheduling，零开销异步调度 |
| **MTP** | Multi-Token Prediction，多 token 预测（投机解码） |
| **DSpark / DFlash** | DeepSeek 的投机解码方案 / draft head |
| **YoCo** | You Only Cache Once；本项目中指 CED 的 decoder 半区与截断重放 |
| **CED** | Causal Encoder-Decoder，V4.1 的 40 层架构 |
| **CSA2** | Compressed Sparse Attention 2，压缩稀疏注意力 |
| **SWA** | Sliding Window Attention，滑窗注意力 |
| **Engram** | 基于 n-gram hash 的条件记忆模块（196B 参数，外部资料） |
| **mHC** | 多路残差流混合（`hc_mult` 路 trunk，子层前后各做一次 mix）；代码见 `v4_modules/hc.py` |
| **SO** | Structured Output，结构化输出 |
| **state slot** | 每序列固定大小的状态缓存（SWA KV + 压缩状态） |
| **bounded replay** | 只重算有界后缀来近似重建 SWA KV |

---

## 11. 坑与注意事项

### 环境/构建
1. **submodule 不会自动拉**：`vendor/` 下 6 个目录是空的，必须 `git submodule update --init --recursive`。
2. **SM100 only**：`get_bf16_tflops_limit()` 对非 SM10 直接抛错；H100/A100 没有 fallback 路径。
3. **CUDA 13 + torch 2.13 + flash-attn-4 beta**：版本很新，和常见 CUDA 12.x 环境不兼容。
4. **Python 3.12**：`pyproject.toml` 写死 `requires-python >=3.12`。
5. **Ascend 需要 CANN 9.2**，且部分算子（DeepSelect）走 `npu_ops/` 的 patch。

### 运行
6. **prompt 必须预 tokenize**：`add_request` 只收 `np.int32` 数组。
7. **默认会下几百 GB 权重**：用 `DEEPINFER_HF_MODEL_DIR` 指向本地副本。
8. **cudagraph OOM**：README 明确提示 `illegal memory access` 可能是 graph OOM，调低 `gpu_memory_utilization`。
9. **调试时关 graph**：`DEEPINFER_ENFORCE_EAGER=1`。
10. **SO 必须 pump**：自己写调度循环时，要在每步调 `so_matcher_builder.pump()`，否则编译结果不会被收割。
11. **benchmark 的 prefill/decode 是解耦的**：`--prefill_batch_size` 不影响 decode 吞吐，别把两者混着解读。
12. **profiler 有残余减速**：decode 测量后仍有 7–11% slowdown，官方要求 profiler 放最后。
13. **`DEEPINFER_BATCH_INVARIANT` 仅 NVIDIA**；确定性采样还要求 `DEEPINFER_ENABLE_MTP_VARLEN=0` + 固定 `seq_seed`。

### 代码
14. **中文注释密集**：`llm_engine.py`、`scheduler.hpp` 里大量中文设计说明，是宝贵的一手资料，别只看英文 docstring。
15. **接口会变**：只有 4 个 commit，函数签名/环境变量随时可能改，学习时 pin commit。
16. **别把 Q&A 当代码文档**：Q&A 描述的是**生产系统**（含未开源的 LB / KV store），代码只是其中"引擎"部分。比如"compute quota 解二次方程"在仓库里找不到。
17. **测试依赖 GPU 和权重**：`tests/models/*` 大多是端到端测试，没有 SM100 跑不了；`tests/engine/*` 也 import `_C`。
18. **不要试图加 TP**：模型代码假设 `model_parallel_size == context_parallel_size`，没有 TP 的切分逻辑；硬加要重写 attention/MLP 的 shard 逻辑。

---

## 12. 延伸阅读

**官方/一手**
- `doc/inference_deployment_qa.md`（本仓库，最值钱）
- DeepSeek 知乎博客（Q&A 里引用）：V4.1 Flash 推理系统架构与关键实现
- 官方发布：`https://www.deepseek.com/en/news/deepseek-v4-1-flash/`
- 关联仓库：FlashMLA、DeepGEMM、DeepEP、DeepSelect、TileKernels、TileLang、deepseek-recipe

**对比阅读**
- vLLM V1 设计文档：`https://docs.vllm.ai/en/latest/usage/v1_guide/`
- SGLang 论文/文档（RadixAttention、overlap scheduler）
- TensorRT-LLM 文档（in-flight batching、CUDA Graph）
- PagedAttention 论文（vLLM 的起点）

**背景知识**
- DeepSeek-V3/R1 推理系统博客（EP/DP、PD 分离的前作）
- YOCO 论文（You Only Cache Once）
- Mamba/线性注意力（理解 SWA 与 state cache 的另一条线）
- xgrammar 文档（结构化输出编译器）

**第三方架构解读（非官方，注意甄别）**
- zartbot：DeepSeek-V4.1 Flash 架构解析
- DeepInfra 模型卡：CED / CSA2 / Engram / DSpark 概述

---

## 附录 A：常用 grep 速查

```bash
# 主循环
grep -n "def step" -A40 deepinfer/engine/llm_engine.py

# 四段流水线
grep -n "def execute_model_step" -A100 deepinfer/worker/worker.py

# 调度器不变量
grep -n "TORCH_CHECK" csrc/engine/scheduler.hpp

# 所有环境变量及默认值
grep -n "EnvVar" deepinfer/environ.py | head -80

# 所有 TileLang 算子
ls deepinfer/model_executor/layers/tilelang_ops/cuda/

# 模型结构开关
grep -rn "n_prefill_layer\|index_topk\|kv_compress_ratio\|use_swa\|hc_mult" deepinfer/model_executor/model_metadata_utils.py

# 找所有 "Derived from vLLM"（理解哪些是继承、哪些是自研）
grep -rl "Derived from vLLM" deepinfer | sort

# 找所有 TODO/FIXME（作者自己承认的短板）
grep -rn "TODO\|FIXME" deepinfer csrc | head -50
```

## 附录 B：一分钟版总结（给同事讲）

> DeepInfer 是 DeepSeek 把 vLLM 拆了重装的产物：调度器换成 C++，引擎按 P/D 拆成三种形态，KV 缓存分成 paged block 和 state slot 两套，前缀缓存推到 3FS 外部存储，并行只做 EP/DP 不做 TP，投机解码用 confidence head 做变长验证，prefill 用 YoCo 只算 encoder 层 + 重放 128 token。它只支持 V4.1-Flash 和 SM100/Ascend 950，没有在线 server，开源的是"引擎"不是"系统"。学习价值在于看它如何为单一模型做极致特化，而不是拿它替代 vLLM。
