# 算子缺口与跨架构实施矩阵

## 结论与证据边界

本文中的 FlagOS 指 FlagGems 代码库。结论按证据强度划分，不能将源码
检查或工程目标解释为设备性能结论。

| 标记 | 含义 |
| --- | --- |
| 已测量 | 在 Tesla V100-SXM2-16GB 上运行了完整的正确性门禁和 30 个原始延迟样本。 |
| 已实现 | 代码、单元测试和语法/静态检查已进入两个开发 fork，但尚未在目标设备运行。 |
| 源码缺口 | 依据锁定的 FlagGems 源码目录得出；通用算子存在不等于目标设备上已有调优实现。 |
| 目标 | 后续租赁设备上的验收阈值，不是预测或性能承诺。 |

当前 P0 的 SToFM 路径已经完成：高斯 pair bias、保留 pair state 的注意力，
以及不需要最终 pair state 时的生命周期消除。V100 的选定路径 O4 为已测量
结果：端到端 p50 从 23.4533 ms 降至 10.8458 ms（B0 的 2.162x；B1 的
2.122x），峰值已分配显存降低 33.6%。原始样本和完整环境记录见
[`v100_optimization_report.md`](v100_optimization_report.md)。

Ascend 310 和 MTT S4000 目前只有正确性优先的静态适配器，**没有**租赁
设备上的运行或性能结论。它们存在的目的，是在租赁前固定 API、语义、测试
矩阵和依赖边界，缩短实际设备上的调试时间。

## 工作负载来源与算子清单

| 模型 | 源码/论文证据 | 热点或特殊语义 | 对应原子算子 |
| --- | --- | --- | --- |
| SToFM | `SToFM.pdf` 第 4、13--14 页；`model/se2transformer.py` | 距离矩阵经过可学习 Gaussian 得到 `[B,N,N,H]` pair bias；每层将 QK score 加入并更新 pair state。论文的典型子切片约为 1,000 个细胞。 | 广播/仿射、`exp`、两层线性投影、`baddbmm`、mask、softmax、`bmm`、pair state 生命周期。 |
| Uni2-h | `../../../20260814-uni2/UNI/uni/get_encoder/get_encoder.py` 第 129--145 行 | `timm` 的 `vit_giant_patch14_224`：24 层、24 heads、1536 hidden、动态图像尺寸、`SwiGLUPacked` 和 SiLU。 | patch `conv2d`、动态位置编码插值/重排、LayerNorm+residual、QKV/SDPA、SwiGLU、token `cat`/gather。 |
| KRONOS | `A Foundation Model for Spatial Proteomics.pdf` 第 5--6、44 页 | DINOv2 ViT 上的逐 marker token embedding；每张图可有 3--58 个 marker，且需叠加 token/position/marker embedding。 | 每通道 patch 投影、embedding/gather、逐通道 token 拼接、可变通道 mask/reduction、ViT block。 |

SToFM 的 pair state 是标准 FlashAttention/SDPA 不能自动替代的关键：模型必须
保留 `score + pair_bias` 作为下一层 pair representation，而普通 SDPA 的输出
只定义 context。因此既不能只依赖通用注意力替换，也不能在 `return_pair_rep`
的训练或 pair-distance-recovery 路径上删除中间 pair state。

## 按计算框架的缺口

| 算子族 | PyTorch eager / 生态缺口 | `torch.compile` / 编译器缺口 | 当前 FlagGems 状态 | 下一步 |
| --- | --- | --- | --- | --- |
| SToFM Gaussian pair bias | 原公式会形成大规模 `[B,N,N,K]` 广播中间量；框架不知道这是可重排的 RBF 语义。 | Inductor 能融合当前静态 CUDA 图，但动态 `N` 会产生编译缓存/重编译决策，且没有 SToFM 专用调度。 | 已实现公共 `stofm_gaussian_pair_bias`：CUDA 自动使用编译版，其他路径使用可控 tile 的参考实现。 | V100 决定是否需要 Volta 专用调度；国产设备在基线后实现 vendor fused kernel。 |
| SToFM pair-state attention | `bmm + add + mask + softmax + bmm` 分散，且原路径为输出 pair state 有额外 clone/物化。标准 SDPA 不返回更新后的 pair state。 | 编译器不会从一般 SDPA 调用推导出 pair-state 更新语义或其反向边界。 | 已实现 `stofm_pair_attention`，语义为 `baddbmm -> bias/mask -> softmax -> bmm`，可返回 context、pair state、weights。CUDA 当前是正确性参考组合，不把它称为 fused kernel。 | 分别实现 score+bias+mask+softmax+context 的前向/反向 fused boundary。 |
| 最终 pair state 生命周期 | 原 SToFM 推理即使只取 embedding 也会保留最终 `[B,H,N,N]`。 | 编译器无法从 Python 返回字典的下游使用自动证明该值无用。 | 已实现 `return_pair_rep=False`，仅可删最终层的输出，层间 pair state 保持。 | 保持 API 回归测试；训练/pair recovery 永远不走该消除。 |
| Uni2 ViT block | 原子算子齐全，但 patch、QKV、attention、residual、MLP 的边界由 `timm`/PyTorch 决定，动态图像尺寸会增加图形分裂风险。 | 需先记录 shape bucket 和重编译次数；不能假设每个动态尺寸均得到同一融合图。 | 通用库包含多个基础算子和 fused SwiGLU，但不存在针对 Uni2 的完整 ViT block 公共复合 API。 | 增加独立 `vision` 实验 API，先以单 block 基准确认收益。 |
| KRONOS marker-aware tokenization | 普通 RGB ViT 不表达可变 marker 数和 marker identity；逐通道循环会增加小 kernel 和拼接开销。 | 可变通道数、ragged batch 和 mask 需要显式 bucket 策略，不能依赖通用 trace。 | 没有 marker-aware patch/token 复合算子。 | 实现 channel patch projection + marker gather + token assembly，随后再连接 ViT block。 |

这里的 "通用库包含" 只说明源码中存在某个同名原子实现或可退回 vendor
PyTorch；它不表示在每个目标芯片上已经启用、支持完整反向，或具有可接受的
性能。

## 按计算架构的缺口与实施列表

### NVIDIA V100（已测量）

FlagGems 的 NVIDIA 架构映射只列出 compute capability 8（Ampere）和 9
（Hopper），不含 Volta 的 capability 7。因而 V100 使用的是通用 CUDA /
Inductor 路径，而不是错误地套用 Ampere 配置。已经修复设备发现，使
`Tesla V100-SXM2-16GB` 这类不含 `NVIDIA` 前缀的名称能被识别。

| 项目 | 实现状态 | V100 基线与结果 | 后续门槛 |
| --- | --- | --- | --- |
| O1 Gaussian | 已实现且已测量 | B0 17.5996 ms -> O1 5.8197 ms，3.024x；峰值已分配显存 4382.8 -> 2759.7 MiB。 | 若写 Volta 专用 kernel，必须再快于 O1，且不增加 O1 的峰值显存；否则维持 Inductor。 |
| O2 pair attention | 已实现且已测量 | 单算子 B0 0.9282 ms -> O2 0.9212 ms，1.008x。 | 不应因微基准而单独推广；以完整 O4 与 O3 的端到端 p50 决策。 |
| O4 end-to-end | 已实现且已测量 | B1 23.0191 ms -> O4 10.8458 ms，2.122x；比 O3 再低 4.8%。 | 作为 V100 默认部署路径；新实现需通过全部输出/梯度门禁并不回退该 p50。 |
| Uni2/KRONOS vision | 未实现专用复合算子 | 无 V100 测量，不从 SToFM 数字外推。 | 先做 block 级 B0/O1；仅当端到端 p50 的 bootstrap 95% 下界超过 1.05x 时推广。 |

### Huawei Ascend 310（源码已检查，运行待验证）

`runtime/backend/_ascend/ops` 中已有 `attention.py`、`baddbmm.py`、
`bmm.py`、`embedding.py`、`masked_fill.py` 和 `softmax.py`，因此 SToFM
参考适配器能以清晰的基础算子契约开始。该目录没有目标专用 `conv2d.py` 或
`layernorm.py`；而 backend 配置还把 `cat` 列为禁用项。这使 Uni2/KRONOS
视觉路径的缺口比 SToFM 路径更靠前。`fp64_enabled=False` 也意味着设备上的
梯度验证应使用 FP32 对照，数值 `gradcheck` 留在 CPU/支持 FP64 的环境执行。

| 优先级 | 可补充实现 | 当前状态 | 目标性能门槛（非结果） |
| --- | --- | --- | --- |
| ASC-S1 | Gaussian 仿射 + RBF + 双投影的 CANN/vector 融合，保留 tile fallback。 | `ascend310` 静态适配器已实现正确性路径。 | Gaussian p50 至少为该设备 B0 的 1.30x。 |
| ASC-S2 | `baddbmm + pair bias + mask + softmax + bmm` 复合前向；通过后再做反向。 | 参考组合已实现；无 vendor runtime 测试。 | 至少一个 O3/O4 相对本机 B1 端到端 p50 >= 1.10x。 |
| ASC-V1 | patch `conv2d`、LayerNorm+residual 与 QKV/attention block。 | 目标专用覆盖缺口；不能以通用回退视为优化。 | 各 block 先达到 >= 1.05x，再测 Uni2/KRONOS 端到端。 |
| ASC-V2 | KRONOS channel projection、marker embedding/gather、token assembly；规避/替代禁用 `cat` 的路径。 | 未实现。 | 先以数值/内存正确性为准；有稳定 bucket 后再设延迟阈值。 |

Ascend 310 的 target 名称仅是本项目的部署标签，不构成对特定 CANN、
torch_npu 版本或算子能力的兼容性声明。租赁后必须先按
[`target_device_acceptance.md`](target_device_acceptance.md) 记录实际硬件与
runtime，再决定 kernel API。

### Moore Threads MTT S4000（源码已检查，运行待验证）

`runtime/backend/_mthreads/ops` 包含 `conv2d.py`、`baddbmm.py`、`bmm.py`
及部分 GEMM/索引算子，但没有目标专用 `attention.py`、`softmax.py`、
`layernorm.py`、`embedding.py` 或 `masked_fill.py`。所以 MTT 的第一工作不是
直接调大 tile，而是确认这些必要操作走什么路径、布局和反向是否正确。该 backend
同样声明 `fp64_enabled=False`，且当前 SToFM `mtt_s4000` 适配器只保证无
`torch_musa` 的 import-time 依赖和可语法检查。

| 优先级 | 可补充实现 | 当前状态 | 目标性能门槛（非结果） |
| --- | --- | --- | --- |
| MTT-S1 | 先验证/补齐 softmax、masked fill、attention score 复合及其反向。 | SToFM tiled 参考适配器已实现；目标原子覆盖有缺口。 | 通过全量正确性矩阵后，O4 相对本机 B1 p50 >= 1.10x 才可推广。 |
| MTT-S2 | MUSA/TLE Gaussian tile 或 fused RBF 投影。 | 未调优。 | Gaussian p50 相对本机 B0 >= 1.30x，且峰值显存不增加。 |
| MTT-V1 | LayerNorm、embedding、masked fill、softmax 与 ViT attention block。 | 目标专用实现缺失。 | 先完成 block 级正确性，再要求 >= 1.05x 的 block p50。 |
| MTT-V2 | KRONOS 变长 marker token assembly 与 bucketed batch。 | 未实现。 | 不丢 marker identity，且与 Python reference 完全对齐后再做性能优化。 |

## 已落地代码与后续扩展边界

| 仓库 | 已落地内容 | 扩展入口 |
| --- | --- | --- |
| FlagGems fork `integration/stofm` | `experimental_ops/stofm.py` 的版本化公共 API；高斯 dense/tiled 实现；pair-state attention；`stofm_backends/ascend.py` 和 `mthreads.py` 静态适配器；V100 发现与 Triton 兼容修复。 | 新 SToFM 或 vision 复合算子放在 `experimental_ops`，目标代码放在各自 backend 子目录，绝不让 SToFM 直接 import vendor extension。 |
| SToFM fork `integration/flagos` | 可选 FlagGems bridge、`return_pair_rep=False`、命令行 backend 选择、严格 V100 harness、锁文件和原始报告。 | 新模型算子通过稳定公共 API 接入；每次移动 FlagGems commit 都更新 `deps/flagos.lock.json` 和测试证据。 |

Fork 与依赖管理的完整流程见 [`flagos_integration.md`](flagos_integration.md)：两个
仓库独立 fork、各自跟踪 upstream；SToFM 锁定 FlagGems 的完整 commit SHA，
本地开发可 editable install，但发布/基准只接受已推送的不可变 SHA。不要使用
Git submodule 或仅以分支名作为依赖版本。

建议的 vision 扩展不应混入现有 SToFM API。新增下列独立边界，才能让 Uni2 和
KRONOS 的测量不污染 pair-state 基准：

```text
FlagGems/src/flag_gems/experimental_ops/vision.py
FlagGems/src/flag_gems/experimental_ops/vision_backends/{cuda,ascend,mthreads}.py
FlagGems/tests/test_vision_experimental.py
SToFM-flagos/benchmarks/uni2_v100.py
SToFM-flagos/benchmarks/kronos_v100.py
```

第一个 vision API 应只暴露可单独验证的 `vit_block` 和
`marker_token_embed`；不要把完整 `timm` 模型或数据加载逻辑放入 FlagGems。

## 严格测试与报告规范

### 当前已执行的门禁

| 层级 | 已执行内容 | 结论 |
| --- | --- | --- |
| FlagGems 算子 | Gaussian 输出和全部可学习参数梯度对 dense FP32 reference；pair attention 的 context/pair state/weights/pair-bias 梯度与 padding。 | 4 项测试通过。 |
| SToFM bridge | backend 继承/覆盖、Gaussian、attention、最终 pair state 生命周期、端到端输入梯度。 | 5 项测试通过。 |
| 静态目标 | AST 检查 Ascend/MTT 适配器元数据，且无 `torch_npu`/`torch_musa` import-time 依赖。 | 通过；不代表设备运行通过。 |
| V100 性能 | 输出比较后使用 CUDA events，10 warm-up、30 样本、每样本 5 次调用；保存全部原始样本。 | O4 为选定结果，见 V100 报告。 |

已有数值比较的容差为 FP32 `rtol=3e-4`、`atol=3e-5`。所有基准在
`torch.inference_mode()`、dropout 为零、固定 seed 下进行，编译只发生在
warm-up，不计入延迟。

### 租赁设备前后必须补齐的测试矩阵

1. 在 CPU/支持 FP64 的环境对最小 Gaussian 和 pair-attention reference 做
   `gradcheck`；在目标设备以相同随机输入做 FP32 前向/反向对照。
2. 对 `B={1,2}`、`N={7,33,65,256,1050}`、奇数维度、非 contiguous tensor、
   零距离、无 padding 和尾部 padding 分别运行 Gaussian/attention。完整 mask
   行必须明确记录为支持或受限，不能悄然吞掉 NaN 语义。
3. 验证 `return_pair_rep=False` 仅影响最终输出；训练、pair-distance-recovery
   和任何中间 Transformer layer 必须仍保留 pair state。
4. 在 Ascend/MTT 上先运行目标适配器的 forward、所有可训练输入梯度和全模型
   `last_hidden_state`；失败例保存输入 seed、shape、dtype、最大绝对/相对误差。
5. 为 Uni2/KRONOS 增加 image size、patch size、marker count、marker ID、
   padded/ragged batch 的 reference test。KRONOS 必须验证 marker permutation
   对应地只置换 token，而不丢失 marker embedding。
6. 每个新 kernel 先 benchmark 单算子/单 block，再 benchmark 完整模型。记录
   `p20/p50/p80/p95/mean`、峰值 allocated/reserved memory、每个样本 CSV、驱动、
   runtime、编译日志和精确两个 Git SHA。

### 性能判定与抗噪声规则

* 基线和候选必须在同一设备、dtype、shape、batch、驱动和热状态下比较；不得把
  V100 的绝对毫秒与 Ascend/MTT 直接相除。
* 每个候选至少三次独立进程运行；保存所有样本，并用 p50 及 bootstrap 95% 置信
  下界判断是否超过前述门槛。仅最小值或一次运行不能作为结论。
* 记录时钟/功耗策略、温度、可用显存、并发进程和 compiler cache 状态；若无法
  固定时钟，交替执行 B0/B1/Ox 并在报告中标记。
* 任何 fused 实现若没有同时通过输出、梯度、mask、非连续布局与显存门禁，就保留
  正确的 tiled/reference adapter，不得默认启用。
* 结果目录沿用 V100 的 `result.json`、`samples.csv`、`report.md`、`report.html`
  格式，使以后增加 Ascend/MTT/vision 的形状扫描和训练吞吐测试时可直接汇总。

详细 B0/B1/O1/O2/O3/O4 定义、计时方式和提升计算规则见
[`benchmark_protocol.md`](benchmark_protocol.md)；实际租赁时的设备验收顺序和
推广规则见 [`target_device_acceptance.md`](target_device_acceptance.md)。
