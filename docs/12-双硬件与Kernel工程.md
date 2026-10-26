# 第 12 课：双硬件与 Kernel 工程

> 同一套 Python 代码，跑在 NVIDIA SM100 和华为 Ascend 950 上。
> 本课讲 `Platform` 抽象、双份 TileLang 算子、vendor 库构建、CUDA Graph，以及"加一个 kernel"的完整流程。

## 本课目标

1. 说清 `Platform` 抽象暴露了哪四类差异。
2. 理解 `tilelang_ops/{cuda,asc}` 的组织方式与代价。
3. 知道 `build.sh` 的三个阶段各自产出什么、为什么这么设计。
4. 理解 CUDA Graph 在这个项目里的使用边界。
5. 能按步骤加一个新算子（至少知道要改哪些文件）。

## Platform 抽象

`deepinfer/platform.py`：

```python
class Platform:
    enum: PlatformEnum                 # GPU / NPU
    tilelang_ops_backend: str          # "cuda" / "asc"
    distributed_backend: str           # "nccl" / "hccl"
    fp8_sf_type: torch.dtype           # scale factor 的类型

    def pin_host_memory(self, tensor): ...   # 固定页内存，异步 H2D 用
    def unpin_host_memory(self, ptr): ...
```

两个实现：
- `PlatformGPU`：`tilelang_ops_backend="cuda"`、`distributed_backend="nccl"`、`fp8_sf_type=torch.int32`；
- `PlatformNPU`：对应 `asc` / `hccl` /（Ascend 的 scale 类型）。

上层代码只依赖 `PLATFORM` 单例，不写 `if is_nvidia`。

## 双份算子

```
deepinfer/model_executor/layers/tilelang_ops/
├── gather.py                 # 平台无关的调度/封装
├── fused_transform.py        # 平台无关的入口
├── cuda/                     # 30 个 CUDA 实现
│   ├── fused_transform.py
│   ├── compress.py
│   ├── swa_index.py
│   └── ...
└── asc/                      # 30 个 Ascend 实现（同名同签名）
    ├── fused_transform.py
    └── ...
```

代价：约 1.1 万行重复实现（`asc/` 9k + `cuda/` 9k，部分共享）。
收益：每个平台都能用最适合的指令/内存层次，而不是靠编译器"猜"。

`ops` 模块负责按 `PLATFORM.tilelang_ops_backend` 选择实现。看 `tilelang_ops/__init__.py` 或各文件的 import 逻辑。

## 构建流水线

### 阶段 1：Python 依赖

`requirements-nvidia.txt`：

```
torch==2.13.0+cu132
flash-attn-4==4.0.0b34
```

`requirements.txt` 里还有 `tilelang==0.1.15`、`tile-kernels==2.1.0`、`xgrammar==0.2.2`、`transformers==4.50.0`。

### 阶段 2：vendor wheel

`scripts/build_vendor.sh` 的逻辑：

```bash
PACKAGES=(DeepSelect FlashMLA DeepEP DeepGEMM)
INSTALL=(deep_select flash_mla deep_ep deep_gemm)
for pkg in "${PACKAGES[@]}"; do
    cd vendor/$pkg
    git submodule update --init --recursive
    python setup.py bdist_wheel --dist-dir "$WHEEL_DIR"
done
pip install --no-deps --no-index --find-links "$WHEEL_DIR" "${INSTALL[@]}"
```

为什么先打 wheel 再装：每个包有自己的编译 flags（`DEEP_SELECT_BUILD_TARGET_PLATFORM=CUDA`、`DG_FORCE_BUILD=1`），wheel 是隔离构建产物的最简单方式；`--no-deps` 避免它们把 torch 降级。

### 阶段 3：`_C` 扩展

`scripts/build.sh`：

```bash
python setup.py build --build-base "$BUILD_BASE"
# 把 build/.../deepinfer/*.so 软链到 deepinfer/_so/$PLATFORM_TAG/
python -m compileall deepinfer -j "$MAX_JOBS" -q
```

- `setup.py` 编译 `csrc/`（pybind11 + nanobind）；
- `.so` 按平台分目录，支持同一 checkout 下多平台构建；
- `compileall` 预编译 Python，减少首次 import 开销。

### 运行时：TileLang JIT

TileLang 算子在首次调用时编译，缓存到 `DEEPINFER_JIT_CACHE_DIR`。所以：
- 第一次跑：慢（编译几十个 kernel）；
- 第二次跑：快（读缓存）；
- 换 GPU 架构/驱动：缓存失效，重编。

## CUDA Graph

`deepinfer/worker/cuda_graph_runner.py`：

- 按 batch size 分桶捕获（`DEEPINFER_CUDAGRAPH_BATCH_SIZE_INTERVAL`）；
- decode 用 graph 回放，避免每步的 kernel launch 开销；
- ZOS 下 `input_on_device=True`：metadata 直接在 device 上准备，不需要 H2D；
- prompt 不用 graph（形状太变）；
- DFlash 有独立的 `DFlashCudaGraphRunner`。

开关：
- `DEEPINFER_ENFORCE_EAGER=1`：完全关 graph（调试）；
- `DEEPINFER_DFLASH_ENFORCE_EAGER=1`：只关 DFlash 的 graph。

**坑**：README 明确说，开 graph 时如果遇到 `illegal memory access`，可能是 graph OOM，调低 `gpu_memory_utilization`。

## Profiling 钩子

| 工具 | 位置 | 用途 |
| --- | --- | --- |
| `cpu_timer` | `deepinfer/timer.py` | CPU 段计时（schedule / bubble / run_workers） |
| `cuda_timer` | 同上 | GPU event 计时（device_forward / device_sample） |
| `ProfileManager` | `engine/profile_server.py` | HTTP 触发 torch profiler |
| `graph_timer` | `csrc/graph_timer/` | CUDA Graph 内的分段计时 |
| `PROFILE=1` | `examples/benchmark.py` | 导出 trace |

`ProfileManager` 由 `DEEPINFER_PROFILE_HTTP_ENABLE` 开启，端口从 `DEEPINFER_PROFILE_HTTP_PORT_BASE` 起。它只在 driver worker 上启动，避免多 rank 抢端口。

## 加一个新算子：步骤清单

以"给 decode 加一个 custom top-k 过滤"为例：

1. **写 kernel**：`tilelang_ops/cuda/my_op.py`，用 `@tilelang.jit` 定义 prim_func；
2. **写 Ascend 版**：`tilelang_ops/asc/my_op.py`，同名同签名（可以先写 reference 实现）；
3. **写平台无关入口**：`tilelang_ops/my_op.py`，按 `PLATFORM` 分发；
4. **接入调用点**：在 `layers/` 或 `models/` 里 import 并调用；
5. **加测试**：`tests/` 里对 reference 与 kernel 做数值比对（很多 tilelang_ops 文件自带 `_ref` 函数和 pytest）；
6. **加开关**（可选）：`environ.py` 里加 `EnvVar.bool()`；
7. **文档**：如果是用户可见的行为，更新 README/Q&A。

注意：`tilelang_ops` 里的文件本身可能就是 pytest 测试（有 `import pytest`、`_ref` 函数），这是这个项目的特殊风格——**kernel 与测试同文件**。

## Ascend 的特殊处理

- `deepinfer/npu_ops/`：`deep_select.py`（top-k 的 NPU 实现）、`vit_attn.py`、`patch_imports.py`（替换 import）；
- `deepinfer/platform.py` 的 NPU 分支：`hccl`、不同的 `fp8_sf_type`；
- `deep_cluster/hardware/ascend.py`：通过 `/dev/davinci*`、`npu-smi` 发现设备；
- `csrc/asc_utils.cpp`：Ascend 侧的 C++ 工具；
- 调试环境变量：`ASCEND_LAUNCH_BLOCKING=1`（同步执行，栈更清晰）、`ASCEND_SLOG_PRINT_TO_STDOUT=1`。

## 图解：一次 kernel 调用的路径

```
Python: layers.fused_transform.fused_norm_rope_...
   │
   ├─ PLATFORM.tilelang_ops_backend == "cuda" ──► tilelang_ops/cuda/fused_transform.py
   └─ == "asc" ────────────────────────────────► tilelang_ops/asc/fused_transform.py
                                                        │
                                                        ▼
                                              TileLang JIT → 设备 kernel
                                                        │
                                                        ▼
                                              CUDA / Ascend runtime
```

## 设计取舍

| 选择 | 收益 | 代价 |
| --- | --- | --- |
| 双份 kernel | 每个平台都能榨性能 | 1.1 万行重复、双倍维护 |
| 运行时 JIT | 不用预编译所有形状 | 首次慢、缓存管理复杂 |
| CUDA Graph 分桶 | 降 launch 开销 | 形状受限、OOM 风险 |
| pybind11 + nanobind | 低开销跨语言 | 调试难、stub 要生成 |
| 平台单例 | 上层无 if/else | 测试要 mock 平台 |

## 易错点

1. **只写 CUDA 版**：Ascend 构建会失败（同名文件缺失）。
2. **改了 kernel 不清 JIT 缓存**：跑的还是旧 kernel。
3. **在 graph 捕获区里做动态分配**：会捕获失败或结果错。
4. **忘了 `compileall`**：首次 import 慢，影响 benchmark 启动时间。
5. **在 NPU 上用 CUDA 的 dtype 假设**：`fp8_sf_type` 不同。

## 自测题

1. `Platform` 抽象了哪四类差异？为什么不用 `if is_nvidia`？
2. 双份 TileLang 算子的维护成本体现在哪？
3. `build_vendor.sh` 为什么要 `--no-deps`？
4. CUDA Graph 为什么只用于 decode？
5. 加一个新算子要改哪些文件？

## 下一课预告

**第 13 课《性能基准与实验方法》** 会解释 benchmark 的三个指标怎么定义、为什么要"烧 step"、Pareto 图怎么读，以及常见的测量错误。
