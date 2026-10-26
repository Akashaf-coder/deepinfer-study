# 第 06 课：YoCo / CED 与 bounded replay

> 这是全项目最"反常识"的优化：prefill 只跑一半层；decoder 的 KV 不精确重建，只重算最后 128 个 token。
> 本课解释它为什么成立、代码怎么实现、误差怎么控制。

## 本课目标

1. 说清 CED（Causal Encoder-Decoder）架构与 YoCo 的关系。
2. 解释 `n_prefill_layer` 如何改变 `DeepseekModel.forward` 的执行路径。
3. 区分 **mask** 与 **compact** 两种截断模式，知道各自改了什么。
4. 能复述 bounded replay 的命中策略，并说明 134 这个数字的来源。
5. 知道如何验证两种模式等价（`examples/yoco/`）。

## 背景：CED 是什么

外部资料（官方模型卡/第三方解读）给出的 V4.1-Flash 架构：

- 40 层 Transformer，前 20 层是 **causal encoder**，后 20 层是 **decoder**；
- decoder 的 global KV 不是每层自己算的，而是**从 encoder 最后一层的 hidden state 投影**得到；
- 因此 prefill 只需要跑 encoder 的 20 层，激活参数约 8B；decode 才需要 16B。

代码里不写死 20，而是用 `cfg.n_prefill_layer`：

```python
def yoco_enabled(cfg):
    """Whether the model enables YOCO (layers after n_prefill_layer are consumer layers)."""
    return cfg.get("n_prefill_layer", cfg.num_hidden_layers) < cfg.num_hidden_layers
```

`n_prefill_layer` 之后的层叫 **consumer layer**。YoCo（You Only Cache Once）指的是：全模型只保留一份"全局 KV"，consumer 层不再各自维护一套完整 KV。

## 代码：forward 在边界处切换

`models/deepseek.py:190`：

```python
for i in range(self.num_layers):
    if input_metadata.is_prompt and i == self.yoco_boundary_layer_idx:
        if envs.DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE == "compact":
            consumer_metadata = input_metadata.get_consumer_view()
            cross_layer_runtime.consumer_full_metadata = input_metadata
            input_metadata = consumer_metadata
            if self.dflash_targets_in_consumer:
                ...
```

三件事：
1. 前 `n_prefill_layer` 层用 full metadata（完整 prompt）；
2. 到边界层，切到 consumer view（compact 或 mask）；
3. `CrossLayerRuntime` 保留 full metadata，供"还需要完整序列"的路径使用（比如某些 KV source 层）。

`yoco_boundary_layer_idx = cfg.n_prefill_layer`，只在 `is_prompt` 且开启截断时生效。decode 阶段不切。

## 两种截断模式

`DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE` 三档：`off` / `mask` / `compact`。

### mask 模式：不真截断，只改可见域

`input_metadata/yoco.py` 的字段说明：

```
yoco_mask_swa_index_pivots: mask 模式的 SWA 可见域 pivots
  （滑窗 ∩ keep_tokens 矩形，full 布局；矩形之外的 token 可见域为空，展开后整行 -1）
```

做法：仍然算完整的 token 行，但把超出 `keep_tokens` 的 query 的 SWA 可见域设成空。好处是实现简单（布局不变），坏处是**白算**——被 mask 掉的行仍然占 forward 的计算。

### compact 模式：真的只保留后缀

```
yoco_consumer_query_indices: full varlen 下标的 gather 列表（full → compact 的正向映射）
yoco_consumer_cu_seqlens_q:  compact varlen 边界（cumsum(keep_len)）
yoco_consumer_positions:     截断后 token 的 positions
yoco_consumer_ks / ke:       每个 sparse 压缩率的 compact 版
yoco_consumer_slot_mapping_store_swa: compact 版；占位 token 置 -1，不写 ring
```

做法：用 `gather` 把"需要算的行"抽出来，重新组成一个更短的 varlen batch。边界层还要同步搬：
- `hidden_states`；
- `residual`（如果开 mHC，是 `HCLazyPostData` 的四个字段）；
- `cross_layer_runtime.topk_indices` / `block_candidate_idxs`；
- 并清空派生的候选缓存（`candidate_mqa_lens`、`candidate_mqa_schedule_metadata`），因为 query 行布局变了。

两种模式共用同一个 SWA 可见域公式（C++ 的 `yoco_swa_visibility`），所以**在保留区间内数值一致**。这就是 `examples/yoco/yoco_equiv.py` 要验证的东西。

### 选哪个

| 模式 | 计算量 | 实现复杂度 | 适用 |
| --- | --- | --- | --- |
| off | 100% | 低 | 调试、对照 |
| mask | 接近 100%（只是可见域变小） | 中 | 验证等价性、保守上线 |
| compact | 只算后缀（约 50%） | 高 | 生产默认 |

Q&A 的生产配置：`DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact` + `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140`。

## bounded replay：为什么是 128 + 6

问题：decoder 的 SWA KV 在 prefill 时只算了一部分（consumer 层被截断）。如果 prefix cache 命中，但没有对应的 SWA KV，怎么办？

精确做法：把最后 2,560 个 token 重跑 20 层 decoder —— 成本太高。

DeepSeek 的做法（Q&A）：

> 重放从位置 s 开始，位置 i 的 query 只与 `max(s, i-127)` 到 `i` 的 key 计算，得到一个近似 SWA KV。重算成本因此变成常数，与上下文长度和层数无关。

具体参数：
- 重放窗口 128 token；
- 保留 134 token（128 + 6 个投机回溯余量）；
- 默认 `KEEP_TOKENS=140`（再留一点余量）。

为什么可以近似：
- SWA 只看最近窗口，更早的 key 本来就不参与；
- 重放段内的 key 是精确的，只有"段边界之前"的 key 缺失，而它们本来就在窗口外；
- 误差被限制在窗口边界附近，且只影响 decode 的第一批 token。

**注意**：Q&A 说 decoder replay 的结果"只用于 decode，不进入 prefix cache"。这是关键——近似值不能被后续请求复用，否则误差会累积。

## 命中策略（背下来）

对长度 L 的输入：

```
1. 命中上限 = L - 134
2. 先查 block KV → 得到 L1（128 的倍数）
3. 再查 SWA KV：
     若 [L1-128, L1] 有 SWA → 命中 L1
     否则 → 回退一个 block，命中 L1 - 128
4. 缺失部分重算（bounded replay）
```

为什么上限是 `L - 134`：要给 replay 留出 128 窗口 + 6 回溯，不能把整条 prompt 都算成命中。

为什么先查 block 再查 SWA：block KV 决定"能复用多少"，SWA 决定"最后一个 block 是否需要重算"。如果 SWA 缺失，退一个 block 保证 replay 段完整。

## state slot 的存法

Q&A 的数字：

```
只存 state_size_to_disk = 20 层 × 134 token × 528 B = 1,415,040 B / 序列
```

存哪儿：
- 不进 paged block，进 state slot；
- slot 分两区：`[stored: SWA][dropped: consumer SWA + ca_state/ica_state]`；
- 只在 128 边界存；
- decode 时 device 上留 replica，每跨边界拷一次，结束时取最后一次完整拷贝；
- 跨机命中靠 Entrypoint 的内存目录 + 400G VLAN 拉取。

## 验证：mask vs compact 等价性

`examples/yoco/README.md`：

| 级别 | 命令 | 成本 | 验证什么 |
| --- | --- | --- | --- |
| L1 | `yoco_equiv.py mask-compact` | 单卡秒级，不加载权重 | SWA pivots、可见 key、store slot、压缩 key 范围 |
| L2 | `run_yoco_equiv_l2.sh` | 8 卡 + checkpoint | 固定 teacher-forcing 轨迹，逐元素比 logprob / top-K |

L1 需要 `deepinfer._C`（要构建），所以 macOS 上跑不了；但脚本本身值得读，它把两种模式的 metadata 差异列得很清楚。

L2 的注意事项：
- 必须用当前仓库编出的 `.so`；
- 除 `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE` 外，两次运行的所有 `DEEPINFER_*` 必须一致；
- 模型必须是 `n_prefill_layer < n_layer` 的兼容 checkpoint。

## 图解：prefill 的两种路径

```
full（off）:
  layer 0 ─────────────── layer 19 ─────────────── layer 39
  [======================== 40 层全算 ========================]

compact:
  layer 0 ─────────────── layer 19 │ layer 20 ─────── layer 39
  [==== encoder 20 层全算 ====]    │ [只算 keep_tokens 后缀]
                                   │
                          边界层 gather：
                          hidden / residual / topk / candidates
                          切 compact 布局

decode:
  不分 encoder/decoder，全部层都跑，但 KV 只有一份 global + SWA
```

## 设计取舍

| 选择 | 收益 | 代价 |
| --- | --- | --- |
| 只算 encoder 层 | prefill 计算量约减半 | 模型结构必须支持 CED |
| 近似 SWA KV | 重算成本常数级 | 误差、不能进 prefix cache |
| 128 边界对齐 | 存储/拷贝简单、可批量 | 尾部浪费、实现约束多 |
| compact 而非 mask | 真省算力 | 布局切换复杂、CP 交互难 |

## 易错点

1. **以为 `n_prefill_layer` 是层数配置**：它是"encoder 层数"，之后的是 consumer。
2. **以为 mask 模式省算力**：它只改可见域，行还是算的。
3. **以为 replay 结果可以缓存**：Q&A 明确说不进 prefix cache。
4. **以为 134 是随意取的**：128 窗口 + 6 回溯。
5. **在 CP > 1 时手动 gather**：compact 与 CP 的交互有专门的 `yoco_cp_*` 字段，顺序是"先 compact 再 CP"。

## 自测题

1. CED 架构下，decoder 的 global KV 从哪来？为什么 prefill 不用跑 decoder 层？
2. mask 和 compact 的实现差异是什么？为什么两者在保留区间内数值一致？
3. 重放窗口为什么是 128 而不是 256 或 64？
4. 命中策略里，为什么要"先 block 后 SWA"，且 SWA 缺失时退一个 block？
5. 近似 SWA KV 如果不进 prefix cache，会不会影响多轮对话的一致性？为什么？

## 下一课预告

**第 07 课《CSA2：压缩、稀疏、SWA、跨层共享》** 会拆开 `V4Attention`，解释一个 attention 类里到底塞了几种机制，以及它们的组合约束。
