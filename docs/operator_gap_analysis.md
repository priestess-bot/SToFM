# 算子缺口与跨架构实施矩阵

本文中 FlagOS 指 FlagGems。结论按证据强度划分，避免将源码检查、静态 adapter
或目标阈值误写成真实设备性能。

| 标记 | 含义 |
| --- | --- |
| 已测量 | 在 V100 上完成正确性门禁、三独立进程和原始样本/统计汇总。 |
| 已实现未测量 | 公共 API、回退、测试、语法/AST 检查已存在，但目标硬件尚未运行。 |
| 源码缺口 | 当前框架/API 没有覆盖该模型语义或目标专用优化。 |
| 目标 | 未来租赁设备上的工程接受阈值，不是性能预测。 |

## 工作负载和不可替代语义

| 模型 | 来源 | 算子/语义边界 | 为什么通用替换不够 |
| --- | --- | --- | --- |
| SToFM | `SToFM.pdf`；`model/se2transformer.py` | 距离 `[B,N,N]` 经可学习 Gaussian 得到 `[B,H,N,N]` pair bias；每层必须输出下一层 pair state。 | SDPA/FlashAttention 通常只返回 context，不能替代可学习的下一层 pair state。 |
| Uni2-h | `20260814-uni2/UNI/.../get_encoder.py` | 动态图像尺寸、patch projection、ViT residual/LayerNorm、QKV/attention、SwiGLU。 | `timm`/PyTorch 原子算子存在，但动态 shape 会导致图分裂，且没有可独立测量的 Uni2 block 复合契约。 |
| KRONOS | `A Foundation Model for Spatial Proteomics.pdf` | 每 marker 通道 token、marker identity embedding、token/position add、可变 marker 数、padding/CLS。 | 普通 RGB ViT 不表达 marker identity 和 ragged marker batch；Python 循环/拼接会产生小 kernel 和额外物化。 |

SToFM 的 `return_pair_rep=False` 仅可移除调用方确认无用的最后一层输出，不能移除
层间 pair state，也不能用于训练或 pair-distance-recovery。

## 按计算框架的缺口

| 算子族 | PyTorch eager / 生态缺口 | 编译器缺口 | FlagGems 现状 | 可补充实现 |
| --- | --- | --- | --- | --- |
| SToFM Gaussian pair bias | 原公式会物化大 `[B,N,N,K]` RBF 中间量；框架不知道 RBF 和两层投影可联合调度。 | Inductor 可融合当前静态 CUDA 图，但动态 `N` 会重编译/缓存，且没有模型专用 tile policy。 | 公共 `stofm_gaussian_pair_bias`；CUDA auto=Inductor，其他路径是 autograd-safe tile reference；显式 NVIDIA Triton O1n 已实现。 | 动态 shape bucket、Volta/目标设备 profile 驱动的 tile/fused RBF-projection；只有胜过 O1 才默认。 |
| SToFM pair-state attention | `bmm + add + mask + softmax + bmm` 分散，且 pair state 需要可选物化。 | 编译器不能从一般 attention 调用推导 pair-state 语义与安全 backward 边界。 | 公共 API；NVIDIA O2n 融合 score/mask/pair/softmax，QK/PV 仍为 cuBLAS；训练/不支持布局回退。 | 目标设备前向复合，随后在完整梯度矩阵通过后才增加 backward。 |
| 最终 pair 生命周期 | Python 返回字典隐藏下游是否读取 pair state。 | 编译器不能可靠证明其无用。 | SToFM 已提供显式 `return_pair_rep=False`。 | 保持生命周期回归测试；不扩展到中间层或训练。 |
| KRONOS marker token | 可变 marker/CLS/padding 导致循环、gather、add、reshape、cat 边界。 | Ragged batch 和动态 marker 数需要显式 bucket，而非假设单一 trace。 | 版本化 `marker_token_embed`；NVIDIA Triton inference kernel；Ascend/MTT reference adapter。 | vendor gather/add/flatten kernel，marker-count bucket 和真实图像 patch 投影连接。 |
| Uni2 ViT residual/MLP | 基础 LayerNorm/SwiGLU 存在，但融合是否实际更快依赖形状与架构。 | 动态 image size、ViT block 边界和编译缓存需要独立测量。 | `vit_residual_layer_norm` 和 `vit_swiglu` 公共 API；V100 当前保留 reference。 | 只在 block/端到端证明收益后增加 residual-LN、SwiGLU、QKV/attention 专用实现。 |

“已有基础算子”不表示每个国产设备已有高性能 kernel、支持完整 backward，或适合
本模型的布局/动态形状。

## NVIDIA V100：已测量结果

设备是 Tesla V100-SXM2-16GB，PyTorch 2.5.1+cu121、CUDA 12.1、FP32
`B=1,N=1050,L=4,D=256,H=8,K=128`。每次运行 30 样本、5 调用/样本，三个独立
进程由聚合器验证环境与提交一致性。完整统计和 raw CSV 见
[`v100_operator_optimization_report.md`](v100_operator_optimization_report.md)。

| 项目 | 实现状态 | 三运行 p50-of-p50 | 决策 |
| --- | --- | ---: | --- |
| O1 Gaussian Inductor | 已测量编译器优化 | 17.5860 -> 5.8339 ms，`3.014x` | 默认 Gaussian |
| O1n Gaussian Triton | 已测量真实 native kernel | 相对 O1 `0.543x` | 正确但拒绝默认 |
| O2 pair reference | 已测量 API/reference 组合 | 相对 B0 attention `0.983x` | 比较基线，不推广 |
| O2n pair epilogue Triton | 已测量真实 native kernel | 0.9181 -> 0.6603 ms，`1.391x` | 默认 CUDA inference attention |
| B1 lifecycle | 已测量生命周期变化 | B0/B1 `1.019x` | 保留 |
| O3 Gaussian + lifecycle | 已测量比较路径 | 相对 B1 `2.031x` | 比较基线 |
| O4 direct pair reference | 已测量比较路径 | 相对 B1 `2.134x` | O5 的直接基线 |
| O5 default | 已测量 O1+B1+O2n | B0 23.6024 -> 8.8105 ms，`2.679x`；B0 显存 -33.24% | 选定默认 |
| KRONOS marker assembly | 已测量真实 native inference kernel | pooled raw sample `1.349x`，95% CI `1.327--1.382x` | 接受为独立 vision API |
| Uni2 SwiGLU / residual-LN | 已测量候选 | SwiGLU `0.591x`；skip-LN 0.1053 vs torch 0.0351 ms | 拒绝默认 |

O5/O4 的 10,000-resample bootstrap 95% CI 为 `1.2307--1.2324x`。当前默认
`flagos_backend=flaggems` 加 `flagos_attention_backend=inherit` 真实选择 O1+O2n；
explicit `nvidia` 仍是 O1n 研究路径，不是推荐默认。

## Huawei Ascend 310：已实现未测量

现有 FlagGems adapter 位于 `stofm_backends/ascend.py` 和
`vision_backends/ascend.py`。它们没有 import-time `torch_npu` 依赖，使用
reference contract 保证安装 CANN 前可检查/测试；这不是 CANN fused kernel。

| 优先级 | 缺失或待实现的目标专用算子 | 当前代码状态 | 租赁后的接受目标 |
| --- | --- | --- | --- |
| ASC-S1 | Gaussian affine + RBF + 双投影的 CANN/vector 融合与 tile policy | reference adapter、SToFM bridge、正确性/基准 harness 已实现 | Gaussian p50 >= `1.30x` vs 本机 B0，且峰值显存不增 |
| ASC-S2 | QK 后 score+bias+mask+pair+softmax 复合，之后才是 backward | reference pair adapter 和完整梯度验证器已实现 | 至少一个 full SToFM candidate >= `1.10x` vs 本机 B1 |
| ASC-V1 | marker embedding/gather + token/position add + flatten，替代变长 `cat` 流 | reference vision adapter、marker correctness/benchmark harness 已实现 | marker/block p50 >= `1.05x` vs 本机 reference |
| ASC-V2 | patch conv、residual-LN、SwiGLU、QKV/attention 的 shape-bucketed ViT block | API/reference 存在；没有目标专用 kernel | 逐 block 验证后 >= `1.05x`，再测真实 Uni2/KRONOS |

Ascend 310 的具体 CANN、`torch_npu` 版本、FP64/混合精度能力在租赁前未知，不能
从静态代码推断。FP32 设备对照和 CPU/支持 FP64 环境的 `gradcheck` 应分开报告。

## Moore Threads MTT S4000：已实现未测量

`stofm_backends/mthreads.py` 和 `vision_backends/mthreads.py` 同样是无
`torch_musa` import-time 依赖的 reference adapter。MUSA runtime 安装后可由
同一 SToFM bridge 选择 `mthreads`。

| 优先级 | 缺失或待实现的目标专用算子 | 当前代码状态 | 租赁后的接受目标 |
| --- | --- | --- | --- |
| MTT-S1 | score softmax/masked fill/pair state 复合及 backward 的布局验证 | reference adapter、运行时梯度验证器、SToFM harness 已实现 | full SToFM candidate >= `1.10x` vs 本机 B1 |
| MTT-S2 | MUSA/TLE Gaussian tile 或 fused RBF projection | reference adapter 已实现；无 profile/kernel | Gaussian >= `1.30x` vs 本机 B0，显存不增 |
| MTT-V1 | marker gather、LayerNorm、SwiGLU、QKV/attention 的 MUSA block 实现 | vision adapter 与 benchmark harness 已实现 | 每个接受的 operator/block >= `1.05x` vs reference |
| MTT-V2 | KRONOS 变长 marker-count bucket 与实际 patch projection | API 已有；没有目标 kernel/端到端图像 workload | identity/padding 精确对齐后再设端到端阈值 |

MTT 的 host-clock benchmark 会在每个样本两端同步设备；若 vendor event API 被
验证可用，可以增加单独标识的 event-timer，不能与 host timer 混合。

## 已落地代码和扩展入口

| 仓库 | 已落地内容 | 后续入口 |
| --- | --- | --- |
| FlagGems fork `integration/stofm` | SToFM/vision v1 公共 API；NVIDIA Gaussian 与 pair epilogue；NVIDIA marker kernel；Ascend/MTT reference adapter；target runtime validator；AST checker。 | 新模型复合算子放 `experimental_ops`，目标代码放相应 backend，保持 source import 无 vendor runtime。 |
| SToFM fork `integration/flagos` | versioned bridge、O5 默认调度、生命周期选择、V100/target/vision benchmark、跨进程聚合器、锁文件和报告。 | 新调用点只使用公共 API；移动 FlagGems SHA 时同步更新 lock、requirements、测试和报告。 |

Fork 和不可变依赖升级流程见 [`flagos_integration.md`](flagos_integration.md)。目标设备的
执行命令、错误保存和三运行聚合规则见
[`target_device_acceptance.md`](target_device_acceptance.md)。

## 当前测试边界

已执行：FlagGems SToFM/vision/native-fallback/target-validator CPU fallback，
SToFM bridge/default O5 dispatch/aggregation，Python `compileall`，以及五项
target source AST checks。未执行：Ascend 310 或 MTT S4000 上的任何 forward、
backward、性能或 vendor kernel。外部设备结果必须新增独立 raw artifact，不能覆盖
V100 目录或从 V100 数字外推。
