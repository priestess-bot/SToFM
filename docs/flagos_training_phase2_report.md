## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-31 (Asia/Shanghai)
- Verification Status: VERIFIED
- Version Label: stofm_flagos_training_phase2_v1
- Primary Evidence: `artifacts/fake-training/v100-phase2-formal-20260830/`

# SToFM × FlagOS PHASE 2：NVIDIA V100 训练性能报告

## 1. 结论

PHASE 2 已在 Tesla V100-SXM2-16GB 上完成 FP32 单卡训练优化。正式生产形状为
`B=1, N=1050, L=4, D=256, H=8, K=128`，目标为假数据 MCM + PDR。

优化后 FlagOS 完整训练步中位数由初始 FlagOS 的 **411.775 ms** 降至
**335.333 ms**，加速 **1.228x**、延迟下降 **18.56%**；3 个独立进程、每进程
30 个 CUDA event 样本的分层 bootstrap 95% 区间为 **[1.218x, 1.311x]**。

逐算子锁定复验使用 10 个独立进程轮次、每实现 300 个样本：

- Gaussian pair bias：`236.029 -> 141.202 ms`，**1.672x**，95% CI
  **[1.667x, 1.673x]**。
- Pair-score attention：`6.385 -> 4.269 ms`，**1.496x**，95% CI
  **[1.442x, 1.543x]**。

FlagOS 仍未追上 PyTorch。优化后 FlagOS 对 PyTorch fused 基线的速度比只有
**0.245x**，即约慢 **4.08 倍**。Profile 显示通用 `mm_kernel_general` 在优化后
单步仍占约 219.7 ms，是下一阶段的首要瓶颈。

## 2. 实验边界

| 项目 | 固定值 |
|---|---|
| GPU | Tesla V100-SXM2-16GB，compute capability 7.0 |
| 软件 | Python 3.11.15；PyTorch 2.6.0+cu124；CUDA 12.4；Triton 3.2.0 |
| 精度 | FP32；TF32 关闭；CuDNN benchmark 关闭 |
| 数据 | 固定 seed `20260830` 的 CPU 合成张量，随后搬到 GPU |
| 模型 | `B=1,N=1050,L=4,D=256,H=8,K=128` |
| 目标 | MCM cosine objective + PDR pair MSE |
| 计时 | CUDA event；计时区为 forward + backward + optimizer，排除 zero_grad |
| 预热 | 独立 disposable 模型覆盖全部 30 个待测 optimizer step index |
| 整步样本 | 6 路 × 3 独立进程 × 30 样本 |
| 算子样本 | 2 算子 × 3 实现 × 10 独立进程 × 30 样本 |
| 统计 | median、mean、std、p90、p95、trial CV、10,000 次分层 bootstrap |

整步原始 worker 固定在 SToFM `3a200a8727f553745e02aa3196f6653f503aa1dc`；
算子锁定复验固定在 SToFM `2154e0d82fae98c5a3e0b7dd65028300d73f3962`；
两者均使用 FlagGems `a4bb672191bcdccdbc974f640a5e799fdd2ee9ae`。后一个
SToFM 提交只增加审计、通用训练入口和单位范数算子梯度，不改变整步 worker 的
计算路线。

## 3. 六路整步对照

| 训练路线 | forward median | backward median | optimizer median | 完整步 median | p95 | 峰值 allocated | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| 纯 PyTorch 原始算子 + 单张量 AdamW | 25.081 ms | 56.006 ms | 1.385 ms | 82.497 ms | 85.261 ms | 4.304 GiB | 稳定 Torch 基线 |
| 纯 PyTorch 原始算子 + CUDA fused AdamW | 25.145 ms | 56.876 ms | 0.200 ms | 82.205 ms | 153.813 ms | 4.304 GiB | 竞争性 Torch 基线；一轮有系统抖动 |
| 初始 FlagOS 可微参考算子 + 单张量 AdamW | 73.336 ms | 337.453 ms | 1.341 ms | 411.775 ms | 702.896 ms | 3.365 GiB | 优化前 FlagOS 基线 |
| **优化后 FlagOS 原生训练算子 + 单张量 AdamW** | **41.535 ms** | **292.690 ms** | **1.332 ms** | **335.333 ms** | **344.552 ms** | **5.419 GiB** | **正式接受** |
| FlagOS 可微参考算子 + 逐参数 fused AdamW | 72.995 ms | 334.545 ms | 6.815 ms | 414.542 ms | 423.471 ms | 3.365 GiB | 拒绝：未优于初始 FlagOS |
| FlagOS 原生训练算子 + 逐参数 fused AdamW | 42.035 ms | 288.861 ms | 6.989 ms | 338.617 ms | 354.221 ms | 5.419 GiB | 拒绝：慢于 native + scalar |

优化后 FlagOS 的 trial median 为 `333.358 / 335.607 / 336.972 ms`，CV 为
**0.44%**。完整步吞吐由初始 FlagOS 的约 **2,554 nodes/s** 提升到
**3,129 nodes/s**。原生 Gaussian backward 会重计算稠密中间量，因此峰值显存比
初始 FlagOS 高约 2.05 GiB；V100 16GB 仍有足够余量，但这是明确的时间/空间权衡。

## 4. Gaussian：在算什么，如何优化

对于距离 `d_ij`，SToFM 先做标量仿射变换，再映射到 K 个 Gaussian basis，最后
经过两层投影得到每个 attention head 的初始 pair bias：

```text
a_ij = w d_ij + b
u_ijk = (a_ij - mu_k) / (|sigma_k| + eps)
r_ijk = exp(-u_ijk^2 / 2) / (sqrt(2 pi) (|sigma_k| + eps))
h_ij = ReLU(r_ij W1^T + b1)
P_ij = h_ij W2^T + b2
```

初始 FlagOS 训练 ABI 使用可微参考表达式。它需要分别启动 affine、broadcast、
sub/div/square/exp、两次投影、ReLU、permute 和 mask，并让通用 Autograd 保存大量
中间张量。生产 shape 下 forward 为 **57.487 ms**、backward 为 **178.472 ms**。

新增的 `flagos_stofm::gaussian_pair_bias_training` 使用独立训练 ABI：

1. 一个 Triton forward kernel 在 pair tile 内融合 affine、Gaussian basis、两层投影、
   ReLU、head layout 和 zero mask。
2. backward 不保存完整 forward 图，而是重算 `u/r/pre-activation`。
3. 参数梯度用 GEMM 和 reduction 计算；Gaussian 部分使用解析导数：

```text
dr/da     = -r u / sigma
dr/dmu    =  r u / sigma
dr/dsigma =  r (u^2 - 1) / sigma * sign(raw_sigma)
```

4. zero-distance mask 在输出梯度入口置零，确保 masked pair 不贡献任一参数梯度。

结果如下：

| Gaussian 实现 | forward | backward | 总耗时 | 相对初始 FlagOS | 输出最大误差 | 梯度最大误差 |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch 原始实现 | 17.611 ms | 38.116 ms | 55.730 ms | 4.235x | 0 | 0 |
| 初始 FlagOS 可微参考 | 57.487 ms | 178.472 ms | 236.029 ms | 1.000x | 2.68e-7 | 1.40e-4 |
| FlagOS 原生训练实现 | **10.895 ms** | **130.243 ms** | **141.202 ms** | **1.672x** | 3.87e-7 | 1.40e-4 |

原生 forward 相对初始 FlagOS 提升约 **5.28x**，且已快于本轮 PyTorch 原始 forward；
但解析 backward 仍依赖 FlagGems 通用 GEMM/reduction，导致总耗时仍约为 PyTorch 的
2.53 倍。

## 5. Pair-score：在算什么，如何优化

每层 attention 把上一层 pair state 加到 QK score，再施加 padding mask，输出新的
pair state、softmax probability 和 value context：

```text
S = P + alpha Q K^T
S_masked = mask(S, -inf)
P_next = replace_minus_inf_with_zero(S_masked)
A = softmax(S_masked)
C = A V
```

初始 FlagOS 路线由 `bmm -> add -> masked_fill -> clone -> softmax -> bmm` 构成，
backward 由通用 Autograd 展开。新增
`flagos_stofm::pair_score_epilogue_training` 做了两部分融合：

1. forward 使用 `baddbmm` 生成 score，一个 Triton row kernel 同时执行 padding mask、
   `P_next` copy 和 softmax，再用 BMM 生成 context。
2. backward 先用 BMM 得到 `dA` 与 `dV`；一个 Triton row kernel 融合 softmax
   backward、可选 attention-weight 梯度、`P_next` 梯度和 padding zero；最后两次 BMM
   得到 `dQ/dK`。

| Pair-score 实现 | forward | backward | 总耗时 | 相对初始 FlagOS | 输出最大误差 | 梯度最大误差 |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch 原始实现 | 1.160 ms | 1.252 ms | 2.413 ms | 2.647x | 0 | 0 |
| 初始 FlagOS 可微参考 | 2.369 ms | 3.974 ms | 6.385 ms | 1.000x | 1.15e-7 | 7.45e-9 |
| FlagOS 原生训练实现 | **1.174 ms** | **3.140 ms** | **4.269 ms** | **1.496x** | 1.15e-7 | 7.45e-9 |

原生 Pair 总耗时下降 **33.14%**。forward 已接近 PyTorch，但 backward 仍约慢 2.51 倍，
说明后续重点应是 BMM 路径和 backward launch/布局，而不是再次改写 softmax 公式。

## 6. AdamW：实现正确，但候选被拒绝

FlagGems 原有 fused Adam kernel 已注册到标准 `aten::_fused_adamw_` schema。PHASE 2
还修复了一个严重生命周期问题：`bias_correction1/2` 原先是 Triton compile-time
constant，step 改变会触发新 kernel 编译；现在改为运行时标量。

该实现把单个参数的 moment、bias correction、decoupled weight decay 和 parameter
update 融合为一次 Triton launch，但仍是 **每个参数一次 kernel**，不是跨参数 foreach。
参数、一阶矩、二阶矩和 step 与 PyTorch fused AdamW 逐项一致；strict profile 也确认
一次 `_fused_adamw_` 调用映射到 35 个 `_fused_adam_kernel`，无 native fallback。

生产 workload 上模型参数较大而数量有限，单张量 AdamW 只需 **1.332 ms**，逐参数
fused 路线却需 **6.989 ms**。native + fused 相对 native + scalar 的速度比为
**0.990x**，95% CI **[0.978x, 0.999x]**，统计上更慢，因此最终路线保留 scalar
AdamW。实现保留给小参数密集模型继续测试，但本报告不把它计入优化收益。

## 7. Profile 归因

三条主路线各保存一份完整 Chrome trace。下面是同一生产 step 的 profiler 汇总：

| 路线 | CUDA kernel events | unique kernels | 主要热点 |
|---|---:|---:|---|
| PyTorch fused | 424 | 71 | native elementwise、cuBLAS Volta SGEMM、reduce |
| 初始 FlagOS | 1,235 | 64 | `mm_kernel_general` 265.833 ms；broadcast 44.291 ms |
| 优化后 FlagOS | 1,102 | 69 | `mm_kernel_general` 219.656 ms；Gaussian fused forward 10.719 ms |

优化使 kernel event 数减少 **133 个（10.77%）**。`mm_kernel_general` 从 72 次/
265.833 ms 降到 62 次/219.656 ms，但仍约占优化后完整步中位数的 **65.5%**。
`broadcast_to_kernel` 从 44.291 ms 降到 5.268 ms，说明 Gaussian fusion 消除了大块
broadcast/materialization；这也是整步 forward 从 73.336 ms 降至 41.535 ms 的主要来源。

Profile 是单步诊断而非统计计时。性能结论只使用 profiler 外的 CUDA event 样本；
trace 用于解释“时间去了哪里”，不用于替代 3/10-trial bootstrap。

## 8. 严格正确性与恢复测试

完整模型第一步以纯 PyTorch scalar 路线为参考，六路均从相同参数 hash 和 batch hash
启动：

| 最大绝对误差 | 观测值 | 门槛 | 结论 |
|---|---:|---:|---|
| total/MCM/PDR loss | 1.19e-7 | 2e-5 | 通过 |
| 所有参数梯度 | 5.44e-9 | 2e-4 | 通过 |
| 更新后参数 | 3.20e-5 | 5e-5 | 通过 |
| AdamW state | 5.45e-10 | 2e-5 | 通过 |

更新后参数的误差大于梯度误差，是因为 AdamW 第一步会用 `sqrt(v)+eps` 归一化接近零的
梯度；报告同时保留更严格的 loss、gradient 和 optimizer-state 独立门禁，没有只靠
放宽参数门槛验收。

native checkpoint 恢复对照同 seed 的连续 3-step 训练：loss/MCM/PDR 差异为 0，
最大梯度摘要差 `5.96e-8`，模型参数最大差 `3.10e-7`，优化器状态最大差
`1.49e-8`，均低于 `1e-5`。

## 9. 测试过程与结果

| 测试 | 结果 | 覆盖内容 |
|---|---:|---|
| FlagGems 聚焦回归 | 36 passed, 25 skipped | registered ABI、Gaussian/Pair forward+all gradients、AdamW；MUSA runtime tests 因本机无 MUSA 跳过 |
| SToFM 全测试 | 52 passed, 3 skipped | bridge、模型语义、训练器、optimizer、benchmark protocol、已有目标适配 |
| Ascend/MUSA 离线 gate | 2 passed | deferred project source contract 与 CPU target harness |
| native scalar 通用训练器 | passed | strict profile；46/46 计算算子映射 FlagGems；无 native/partial/unmapped fallback |
| fused optimizer 通用训练器 | passed | `_fused_adamw_ -> 35 × _fused_adam_kernel`；无 native fallback |
| checkpoint resume | passed | step、loss、参数和 optimizer state，`atol=1e-5` |
| 整步性能 | passed | 6 路、3 trial、90 raw samples/route、3 Chrome traces |
| 逐算子性能 | passed | 2 算子、3 实现、10 trial、300 raw samples/implementation |

正式结果、snapshot、trace、执行日志和 checksum 位于：

- `artifacts/fake-training/v100-phase2-formal-20260830/model/`
- `artifacts/fake-training/v100-phase2-formal-20260830/operators-locked/`
- `artifacts/fake-training/v100-phase2-native-resume-validation-20260831/`

## 10. 代码索引

- [FlagGems NVIDIA 训练 kernel](https://github.com/priestess-bot/FlagGems/blob/a4bb672191bcdccdbc974f640a5e799fdd2ee9ae/src/flag_gems/experimental_ops/stofm_backends/nvidia.py)
- [FlagGems registered training ABI](https://github.com/priestess-bot/FlagGems/blob/a4bb672191bcdccdbc974f640a5e799fdd2ee9ae/src/flag_gems/experimental_ops/_flagos_stofm_ops.py)
- [FlagGems public dispatch](https://github.com/priestess-bot/FlagGems/blob/a4bb672191bcdccdbc974f640a5e799fdd2ee9ae/src/flag_gems/experimental_ops/stofm.py)
- [FlagGems fused AdamW](https://github.com/priestess-bot/FlagGems/blob/a4bb672191bcdccdbc974f640a5e799fdd2ee9ae/src/flag_gems/ops/_fused_adam.py)
- [SToFM FlagOS optimizer adapter](https://github.com/priestess-bot/SToFM/blob/2154e0d82fae98c5a3e0b7dd65028300d73f3962/model/flagos_optimizer.py)
- [SToFM 整步 worker](https://github.com/priestess-bot/SToFM/blob/2154e0d82fae98c5a3e0b7dd65028300d73f3962/benchmarks/stofm_phase2_v100_worker.py)
- [SToFM 逐算子 worker](https://github.com/priestess-bot/SToFM/blob/2154e0d82fae98c5a3e0b7dd65028300d73f3962/benchmarks/stofm_phase2_v100_operator_worker.py)

## 11. 未完成项与下一步

1. FlagGems 在 V100 上没有 architecture-specific profile；`mm_kernel_general` 仍是通用
   Triton 路径。需要针对 `[M,K]` Gaussian backward GEMM 和 Transformer linear 的
   Volta tile/autotune，并与 cuBLAS 做选择门禁。
2. Gaussian native backward 用重计算换速度，峰值显存高于 reference。下一步应分块
   recompute，目标是不牺牲当前 1.67x 收益并把峰值降到 PyTorch 附近。
3. Pair backward 仍慢于 PyTorch；应合并 dA 计算与 softmax backward，评估
   cuBLAS BMM 或更合适的 FlagGems BMM backend。
4. 真正跨参数 multi-tensor/foreach AdamW 仍未实现。当前逐参数 fused kernel 不能包装成
   foreach，且已在本 workload 被拒绝。
5. 本轮只覆盖 V100 FP32 单卡。AMP/GradScaler、统一 RNG、DDP/FSDP、真实数据和动态
   shape 尚未验收。
6. Ascend 310 与 MTT S4000 当前只保留离线正确性/语法 gate；训练 native ABI、目标
   编译器和实机性能必须在租机后单独测试，不能从 V100 外推。
