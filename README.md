# DeepInfer 学习资料索引

> 面向已有推理基础的读者：15 节课 + 46 页 slides，把 DeepSeek 生产推理框架拆到"能讲、能改、能验证"。
> 分析对象：`deepseek-ai/DeepInfer`，commit `9bd9cf2`（2026-10-21）。

## 目录

```
deepinfer-study/
├── README.md                          # 本文件
├── DeepInfer-项目解析与学习指南.md      # 初版单文件指南（775 行，速查用）
├── docs/                              # 细化版：15 节课
│   ├── 00-导读-学习地图.md
│   ├── 01-项目全景.md
│   ├── 02-请求生命周期.md
│   ├── 03-C++调度器.md
│   ├── 04-ZOS异步调度.md
│   ├── 05-KV缓存.md
│   ├── 06-YoCo与bounded-replay.md
│   ├── 07-CSA2稀疏注意力.md
│   ├── 08-EPLB.md
│   ├── 09-DSpark投机解码.md
│   ├── 10-Sampler与结构化输出.md
│   ├── 11-多模态VE与EPD.md
│   ├── 12-双硬件与Kernel工程.md
│   ├── 13-性能基准与实验方法.md
│   └── 14-对比与学习计划.md
└── slides/
    ├── DeepInfer-深度解析.html         # 自包含 slides（双击打开，无外部依赖）
    ├── DeepInfer-深度解析.pdf          # 同内容 PDF（16:9，46 页）
    ├── deck_template.html             # 模板（可改）
    └── build_deck.py                  # 构建脚本：嵌图 + 生成 HTML
```

## 怎么用

### 只想快速了解（30 分钟）

读初版 `DeepInfer-项目解析与学习指南.md` 的第 0/1/4/6 节，然后翻一遍 slides。

### 系统学习（约 40 小时）

按 `docs/00` 的三条路线选一条：
- 路线 A（源码读者）：00 → 01 → … → 14；
- 路线 B（实验派）：00 → 01 → 02 → 13 → 04/06/09 → 08 → 14；
- 路线 C（面试/汇报）：00 → 01 → 02 → 03 → 04 → 06 → 09 → 14。

### 做分享（45–60 分钟）

直接用 `slides/DeepInfer-深度解析.html`：

```bash
open slides/DeepInfer-深度解析.html
```

- `→` / `空格` 下一页，`←` 上一页，`Home` / `End` 首尾；
- `F` 全屏，`P` 打印（Chrome 里"另存为 PDF"）；
- URL 带 `#/12` 可直接跳到第 12 页。

已生成 PDF，可直接发人：

```bash
open slides/DeepInfer-深度解析.pdf
```

### 改 slides

```bash
# 改 deck_template.html 里的 __IMG_*__ 占位符以外任意内容
python3 slides/build_deck.py
# 重新导出 PDF（需要 Chrome）
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="slides/DeepInfer-深度解析.pdf" \
  "file://$PWD/slides/DeepInfer-深度解析.html"
```

## 15 节课速览

| 课 | 一句话 |
| --- | --- |
| 00 | 三条学习路线 + 三遍读法 |
| 01 | 血统、代码地图、构建流水线、五个边界 |
| 02 | 请求的七个阶段与可观测点 |
| 03 | 三个队列、两轮投票、chunk 裁剪、LCFS 抢占 |
| 04 | 四段流水线与三个新问题 |
| 05 | paged block vs state slot、FP4、COW |
| 06 | CED、mask/compact、bounded replay 128+6 |
| 07 | MLA + 压缩 + 稀疏 + SWA + 跨层共享 |
| 08 | 逻辑/物理专家、贪心分配、symm_mem shuffle |
| 09 | confidence、两队列调度、ZOS 下 GPU 重排 |
| 10 | fused sampler、batch-invariant、xgrammar 异步编译 |
| 11 | VE 独立部署、共享内存池、EPD |
| 12 | Platform 抽象、双份 kernel、CUDA Graph |
| 13 | 三个指标、Pareto 读法、实验矩阵 |
| 14 | 15 维对比、选型、四周计划、20 题 |

## 核心结论（一页纸）

1. **定位**：DeepSeek 生产框架的开源子集，只做离线推理，只支持 V4.1-Flash + SM100/Ascend 950。
2. **血统**：vLLM 的骨架 + SGLang 的异步调度思想 + TRT-LLM 的图优化。
3. **结构**：C++ 调度器、Python 编排、TileLang kernel、EP/DP 并行（无 TP）。
4. **三个招牌**：ZOS（CPU 藏进 GPU）、YoCo bounded replay（重算 128 token）、DSpark（变长验证）。
5. **最大差异**：前缀缓存在引擎外（3FS + Entrypoint），引擎内只有 fork/COW。
6. **性能**：prefill 60–88k tok/s/GPU；decode 2.5–26k tok/s/GPU（交互性 325–104 tok/s/user）。
7. **不能做的**：换模型、上 H100、开箱即用的 OpenAI API。
8. **学习价值**：看"通用框架如何为单一模型做外科手术"，不是找通用引擎。

## 事实与标注

- 代码引用格式 `文件:行号`，基于 commit `9bd9cf2`。
- "外部资料"（官方 Q&A 之外的架构数字、生产参数）会在文中标注。
- 未在本机运行：macOS 无 SM100，`deepinfer._C` 无法构建；所有结论来自静态分析 + 仓库自带图表。

## 原始仓库

```bash
cd ~/DeepInfer
git submodule update --init --recursive   # vendor 需要单独拉
export DEEPINFER_ACCELERATOR=nvidia
./build.sh
python examples/offline_inference.py --ep 8
```
