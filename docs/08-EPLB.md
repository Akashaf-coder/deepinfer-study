# 第 08 课：EPLB——冗余专家与权重重排

> MoE 的 EP 有一个天然问题：热门专家会被所有 token 挤爆，冷门专家闲着。
> EPLB 用"多放几个副本 + 定期搬家"解决它。本课拆开 `deepinfer/model_executor/eplb/`。

## 本课目标

1. 区分 **logical expert** 与 **physical expert**，理解两张映射表。
2. 说清冗余副本的分配算法（最小化每副本峰值负载）。
3. 说清物理放置算法（最小化多 step 的 rank 峰值之和）。
4. 知道重排如何在不中断推理的情况下完成，以及相关环境变量。
5. 能解释 `moe_lb` 指标与生产参数。

## 概念：两套编号

`eplb/policy.py` 开头就写清楚了：

```
logical expert：模型定义的 expert，简写 log
physical expert：GPU 上实际放置的 expert 副本，简写 phy
                 数量 = num_logical + num_redundant
```

例：384 个逻辑专家，EP64，每卡 6 个物理槽 → 384 个物理专家；如果加 64 个冗余副本，物理专家 448 个，每卡 7 个。

两张映射表（每层一份）：

| 表 | 形状 | 用途 |
| --- | --- | --- |
| `physical_to_logical_map` | `[num_layers, num_physical]` | 这个物理槽放的是哪个逻辑专家 |
| `logical_to_physical_map` | `[num_layers, num_logical, num_redundant+1]` | 这个逻辑专家有哪些物理副本（-1 表示无效位） |

dispatch / 路由 / sampler 都直接读这两张表，所以重排对上层透明。

## 冗余数量：为什么要向上取整

`eplb/__init__.py`：

```python
def resolve_num_extra_experts(num_routed_experts, min_num_extra_experts, ep_size):
    """取不小于下限、且使 num_physical_experts 能被 ep_size 整除的冗余 expert 数。"""
    padding = -(num_routed_experts + min_num_extra_experts) % ep_size
    return min_num_extra_experts + padding
```

原因：`num_physical_experts % ep_size == 0` 是硬约束（每卡槽位必须一样多）。所以实际冗余数可能比配置的下限大。

生产参数（Q&A）：

| 阶段 | 冗余专家/卡 | 统计窗口 | 重排间隔 | `moe_lb` |
| --- | ---: | ---: | ---: | ---: |
| prefill | 4 | 100 step | 1,000 step | 1.074 (EP4DP4) / 1.045 (EP2DP2) |
| decode | 1 | 1,000 step | 10,000 step | 1.090 |

decode 的重排更保守（1 万步一次），因为大 EP 下搬权重的通信量更大，且 decode 负载变化更平滑。

## 算法一：冗余副本给谁

`policy.py:rebalance_experts_global()`：

```python
for i in range(num_redundant_experts):
    score = tokens_per_expert_all_diff / logical_count   # 每副本负载
    score_added = tokens_per_expert / (logical_count + 1) # 加一个副本后的负载
    values, indices = score.max(-1, keepdim=True)
    values = values.expand_as(score).contiguous()
    score.scatter_(-1, indices, score_added.gather(-1, indices))
    values.scatter_(-1, indices, score.max(-1, keepdim=True).values)
    redundancy_indices = values.sum(0).argmin(-1)         # 选"加副本后峰值最小"的专家
```

直观解释：
- 当前瓶颈是"负载最高的那个专家"（`score.max`）；
- 给专家 j 加副本后，它的每副本负载变成 `tokens/(count+1)`；
- 重新算一遍最大值，看哪个 j 能把最大值压得最低；
- 选它。

这是**贪心**：每轮加一个副本，不保证全局最优，但 O(冗余数 × 层数) 很快，且效果够好。

`tokens_per_expert_all_diff = tokens_per_expert + arange * 1e-4` 是为了**打破平局**：如果两个专家负载完全相同，`argmax` 会总是选同一个，导致副本分配不均。加一个微小的层内偏移，让平局时有确定性的不同选择。

## 算法二：物理槽怎么排到各 rank

`_batch_greedy_indices()`：

```python
# 优化目标（对每层独立）：
#   sum_{step i} max_{rank j} ( sum_{physical expert e 分到 j} e 在 step i 的负载 )  尽量小
for job_index in range(num_physical):
    vector = vectors[layers, expert_order[:, job_index]]
    torch.add(loads, vector.unsqueeze(1), out=candidates)
    torch.maximum(candidates, peaks.unsqueeze(1), out=candidates)
    torch.sum(candidates, dim=-1, out=scores)
    scores.masked_fill_(occupancy == capacity, inf)
    rank = scores.argmin(dim=-1)
    indices[layers, rank * capacity + occupied] = physical_order[:, job_index]
```

关键点：
- 不是按"平均负载"排，而是按**每个历史 step 的 rank 峰值之和**排——MoE 的瓶颈是"最慢的 rank"，平均值会掩盖尖峰；
- `vectors` 是 `[num_layers, num_experts, num_steps]`，即保留了窗口内每一步的分布，不是只保留均值；
- 每个 rank 容量固定 `capacity = num_local_physical_experts`，满了就 `inf` 排除；
- 逐层独立优化（不同层的专家热度不同）。

## 重排：怎么搬权重

`eplb/shuffle.py`：

- `expert_model_parallel_shuffle_inplace`：跨节点，走 DeepEP/通信；
- `expert_model_parallel_shuffle_inplace_intranode`：节点内，用 `torch.distributed._symmetric_memory`（NVLink `symm_mem`）做原地交换。

流程（`ExpertLoadBalancer`）：
1. 每 step 累加 `expert_count`；
2. 每 `REARRANGEMENT_STEP_INTERVAL` 步，用窗口统计调 `rebalance_experts_global` 算新映射；
3. 把权重按新映射 shuffle 到目标 rank/slot；
4. 原子更新两张映射表 + `logical_count`。

**为什么能"不中断"**：映射表更新是原子的，且新旧布局在权重搬完之前不会切换；推理用旧表，搬完后一次性切新表。代码里 `_shuffle_initialized` / `symmetric_weight_buffers` 管理这套状态。

生产上还有一个细节（Q&A）：rearrangement 可以安排在 prefill 之后（`DEEPINFER_REARRANGE_AFTER_PREFILL_IN_NORMAL_ENGINE`），避免 decode 期间搬权重造成长尾。

## 统计窗口与环境变量

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS` | 32（example） | 冗余专家下限，会被向上取整 |
| `DEEPINFER_EXPERT_COUNT_WINDOW_SIZE` | 3（example）/ 1000（生产 decode） | 统计窗口步数 |
| `DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL` | 1000 / 10000 | 重排间隔 |
| `DEEPINFER_MOE_LB_STAT_INTERVAL` | 10（benchmark） | 打印/聚合 `moe_lb` 的间隔 |
| `DEEPINFER_FORCE_LOAD_BALANCE` | 0 | 强制均匀（测试用） |
| `DEEPINFER_FORCE_IMBALANCE_RATIO` | 0 | 人为制造倾斜（测试用） |
| `DEEPINFER_SHOW_EPLB_DETAIL` | 0 | 打印每层映射 |

**硬约束**：`0 < window <= interval`，否则启动报错。窗口比间隔大没有意义（会包含上一轮重排后的数据）。

## 指标：moe_lb

```
moe_lb = 最慢 rank 的 token 数 / 所有 rank 的平均 token 数
```

- 1.0 = 完美均衡；
- 1.09 = 最慢 rank 比平均多 9% 的活；
- 这个指标比"平均负载"更能反映 EP 的真实瓶颈，因为一步的时间由最慢的 rank 决定。

Q&A 还给了 EPLB 的整体收益：负载均衡后能达到"完美均衡离线 benchmark"的约 80%。

## 图解：EPLB 的三层结构

```
逻辑层    expert 0   expert 1   ...   expert 383
             │          │                  │
映射层    physical_to_logical / logical_to_physical（每层一张）
             │          │                  │
物理层   rank0: [e0, e17, e99, ...]  rank1: [...]  ...  rank63: [...]
             │
通信层   DeepEP dispatch/combine（token 走网络）
         symm_mem shuffle（权重走 NVLink）
```

## 设计取舍

| 选择 | 收益 | 代价 |
| --- | --- | --- |
| 冗余副本 | 热点专家并行处理 | 显存 + 通信 |
| 贪心分配 | O(n) 快、可在线跑 | 非全局最优 |
| 按 step 峰值优化 | 抗突发 | 需要保存窗口内的分布 |
| NVLink 原地 shuffle | 不占 HBM 带宽、快 | 只能节点内 |
| 低频重排（decode） | 减少长尾 | 对突变响应慢 |

## 易错点

1. **以为冗余专家是"多训练几个专家"**：它们只是已有专家的副本。
2. **忘了向上取整**：配置 32 个冗余，实际可能是 64 个（取决于 ep_size）。
3. **窗口 > 间隔**：启动直接报错。
4. **在 decode 期间重排**：会造成长尾；生产建议放到 prefill 后。
5. **只看平均负载**：会误判 EPLB 效果，必须看 `moe_lb`。

## 自测题

1. 画出 logical / physical / rank 三层的映射关系。
2. 冗余副本的贪心准则是什么？为什么不是"给负载最高的专家"？
3. 物理放置为什么按"多 step 峰值之和"而不是"平均负载"？
4. `resolve_num_extra_experts` 为什么要向上取整？
5. decode 的重排间隔为什么比 prefill 大一个数量级？

## 下一课预告

**第 09 课《DSpark：置信度变长投机》** 会讲清 draft 模型、confidence head、变长验证的调度算法，以及它和 ZOS 的冲突如何解决。
