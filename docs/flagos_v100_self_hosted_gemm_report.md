# SToFM FlagOS V100 自研 GEMM/BMM 训练优化报告

日期：2026-09-01

正式实测源码：SToFM `b2f36fa80ad5f1bbe7575ece3a3e4d64f4ce56a6`；FlagGems
`4534b88689da2630145c75d7612284f3fecb61cc`

正式二进制 SHA-256：
`83a0e569fe38f3ea5d3365268eb3d403bd9274757814ea26a65bec5c6d07f392`

## 1. 结论

Stage B 已完成不调用 Torch 原生 GEMM、且 self-hosted extension 不链接或调用
cuBLAS、cuBLASLt、CUTLASS 的 SToFM V100 FP32 训练路线。四类 ATen 矩阵算子
`mm/addmm/bmm/baddbmm` 及四个 out variant 由 FlagOS C++/CUDA Dispatcher 接管；
Gaussian 与 Pair 使用 NVIDIA 原生训练 ABI，AdamW 使用自研多张量 CUDA kernel。

| 工作负载 | PyTorch eager + fused AdamW | FlagOS 自研路线 | 加速 | P95（Torch → FlagOS） | 峰值显存比 |
|---|---:|---:|---:|---:|---:|
| 代表形状 `N=384,D=64,L=2` | 16.6236 ms | 14.4942 ms | 1.1469x | 18.1925 → 17.8012 ms | 0.643x |
| 生产形状 `N=1050,D=256,L=4` | 80.2381 ms | 71.0057 ms | 1.1300x | 82.3944 → 71.3161 ms | 0.648x |

两形状等权几何平均加速为 **1.1384x**，20,000 次配对分层 bootstrap 的 95% CI
为 **[1.0410x, 1.1505x]**。预注册的 9 项 acceptance gate 全部通过。

边界必须与结果同时陈述：

- 主要验收基线是 PyTorch eager + CUDA fused AdamW，与 Stage A 协议一致。
- 单进程辅助探针中，`torch.compile + fused AdamW` 为代表 `16.8765 ms`、生产
  `42.8682 ms`。自研路线在代表形状更快，但尚未超过生产形状的 Torch 编译图。
- Stage A Vendor 路线使用 cuBLAS，生产正式结果 `58.9778 ms`，仍快于本阶段的
  `71.0057 ms`；Stage B 的目标是移除外部 GEMM 依赖后仍超过 eager，不是伪造一个比
  Vendor 更快的结论。
- tuned surface 的非 GEMM 通用点算子可以使用明确批准的 PyTorch CUDA device kernel；
  “不依赖 Torch 原生 GEMM”不等于“所有 ATen 算子均已重写”。

## 2. 算子在计算什么

### 2.1 GEMM、bias epilogue 与 BMM

四个矩阵接口对应：

\[
C=\alpha AB,
\qquad
C=\beta X+\alpha AB
\]

\[
C_b=\alpha A_bB_b,
\qquad
C_b=\beta X_b+\alpha A_bB_b
\]

`addmm/baddbmm` 支持广播 bias 和 `alpha/beta`；BMM 使用 `grid.z` 表示 batch。
输入最后两维允许 row-major 或 transpose 形成的 column-major stride，输出为连续布局。

### 2.2 V100 tiled kernel

主 tile 为 `128×128`，每个 block 使用 256 threads；线程持有 `8×8` register
micro-tile。一次 K tile 的计算为：

\[
A_s=A[m_0:m_0+B_M,\ k_0:k_0+B_K]
\]

\[
B_s=B[k_0:k_0+B_K,\ n_0:n_0+B_N]
\]

\[
r_{ij}\mathrel{+}=\sum_{q=0}^{B_K-1}A_s[i,q]B_s[q,j]
\]

`A_s[B_M][B_K+1]` 和 `B_s[B_K][B_N+1]` 的 padding 降低 shared-memory bank
冲突。按 `M/N` 选择 `8×32`、`8×128`、`128×8`、`32×32`、`128×32`、
`32×128`、`128×64`、`64×128` 或 `128×128` tile；`B_K` 在 8/16 间按形状选择。

### 2.3 布局感知的合并访存

原始实现始终按 row-major 线程映射装载 shared tile。对 `A^T` 或 `W^T` view，这会让
相邻线程跨大 stride 读取。优化后读取轴由实际 stride 决定：

- row-major A：相邻线程沿 K 读取；column-major A：相邻线程沿 M 读取；
- row-major B：相邻线程沿 N 读取；column-major B：相邻线程沿 K 读取。

该变化不改变乘法顺序或公式，只改变线程到全局地址的映射。它同时覆盖 Gaussian
投影、线性层反向和 attention 的转置 BMM。

### 2.4 split-K

Gaussian 权重梯度包含 `128×1,102,500 @ 1,102,500×128`。输出 tile 只有一个，普通
二维网格无法填满 80 个 SM。实现把 K 分成 `S` 片：

\[
P_s=\sum_{k=k_s}^{k_{s+1}-1}A_{:,k}B_{k,:}
\]

\[
C=\alpha\sum_{s=0}^{S-1}P_s
\]

第一阶段写 FP32 workspace，第二个 kernel 做确定顺序归约。`S` 由输出 block 数、
80-SM occupancy 和每片约 4096 个 K 元素共同决定，上限 128。

### 2.5 Gaussian 与 Pair 训练边界

Gaussian 对距离构造 RBF 后执行两层投影：

\[
a=wd+b,\quad \sigma_k=|s_k|+10^{-5}
\]

\[
r_k=\frac{\exp[-\frac12((a-\mu_k)/\sigma_k)^2]}
{\sqrt{2\pi}\sigma_k}
\]

\[
h=\operatorname{ReLU}(rW_1^T+b_1),\qquad y=hW_2^T+b_2
\]

前向用 Triton fused tile；反向 recompute RBF/activation，把逐点导数与 reduction 编译
融合，其中全部矩阵乘再次进入 self-hosted CUDA key。

Pair attention 为：

\[
S=\gamma QK^T+B_{pair},\qquad P=\operatorname{softmax}(S),\qquad O=PV
\]

Stage A 正式路线曾选择 reference Pair backward。本阶段重新启用原生 Pair forward 与
analytical backward：score/mask/softmax 和 softmax backward 使用 Triton epilogue，
前向 2 个及反向 4 个 BMM 全部由 self-hosted kernel 执行。

### 2.6 block-index 多张量 AdamW

AdamW 更新为：

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
\]

\[
\theta_t=\theta_{t-1}-\eta\left(
\frac{m_t/(1-\beta_1^t)}{\sqrt{v_t/(1-\beta_2^t)}+\epsilon}
+\lambda\theta_{t-1}\right)
\]

初版每个元素线性扫描最多 64 个 tensor offset。最终版预先构造每个 tensor 的 block
offset，一个 block 只由 thread 0 查找一次 tensor，索引通过 shared memory 广播；查找
复杂度从逐元素变为逐 block。生产 83 个参数分成两个 launch。

## 3. 为什么 PyTorch 快，以及自研路线为什么仍能超过 eager

PyTorch 的单次大矩阵乘使用成熟 cuBLAS，个别 GEMM 明显快于本阶段 SIMT kernel。
仅比较 `128×1,102,500 @ 1,102,500×128` 或生产 Gaussian forward，本阶段并未宣称
超过 cuBLAS。

端到端 eager 训练还包含更多边界成本：独立 Q/K/V 投影、Pair score 后多个 device
kernel、Gaussian 反向临时张量、out variant 临时结果复制，以及优化器的多次更新。
本阶段通过以下组合超过 eager：

1. QKV 一次 GEMM 后 `chunk(3)`，保持 checkpoint 参数名不变；
2. Gaussian forward 和 derivative/reduction 融合；
3. Pair score/softmax epilogue 与 analytical backward；
4. out variant 直接写目标张量，删除 GEMM 临时输出和 D2D copy；
5. 多张量 AdamW 只需 1-2 个 launch；
6. 形状、layout、split-K 和寄存器策略按真实 SToFM 矩阵族选择。

所以结论是“自研算子与训练融合的整体路线超过 PyTorch eager”，不是“每个自研 GEMM
都比 cuBLAS 快”。生产 `torch.compile` 把更大范围的 eager 图融合后仍明显领先，正好
验证了这一区分。

## 4. 每次优化的变化与收益

以下为同一生产形状的开发探针；所有数值保留，不以正式 5×50 替换过程数据：

| 版本 | 训练步 | 相对上一版 | 修改 |
|---|---:|---:|---|
| 初始 self-hosted tiled kernel | 87.1383 ms | - | 共享内存 tile、split-K、BMM、packed AdamW |
| out variant 原位写入 | 82.7530 ms | 1.0530x | 删除临时输出及 28 次主要 D2D copy 路径 |
| stride-aware 装载 | 80.2836 ms | 1.0308x | 转置 A/B 改为合并访存 |
| 自适应低寄存器主 tile | 76.2906 ms | 1.0523x | 超大网格/split-K 使用 128-register 双 block/SM 版本 |
| Pair native forward/backward | 70.9560 ms | 1.0752x | 融合 score/softmax epilogue，显式四 BMM 反向 |
| `-O3/-lineinfo` + block-index AdamW | 70.9263 ms | 1.0004x | 可复现编译与逐 block tensor 定位 |

从初始 self-hosted 到最终开发探针降低 **18.6%**；正式生产 5×50 中位数为
`71.0057 ms`。相对 Stage A 初始 FlagOS `404.3295 ms`，最终路线约快 **5.69x**。

## 5. Nsight Compute：寄存器策略的硬件证据

同一生产 Gaussian forward `1,102,500×128 @ 128×128`：

| 指标 | 优化前 | 最终 | 变化 |
|---|---:|---:|---:|
| Registers / thread | 221 | 128 | -42.1% |
| Achieved occupancy | 12.36% | 24.69% | 2.00x |
| Issue slots busy | 23.88% | 31.15% | +7.27 pct |
| Compute throughput | 32.91% | 40.44% | +7.53 pct |
| Memory throughput | 181.02 GB/s | 301.12 GB/s | 1.66x |
| No eligible warp | 76.11% | 69.08% | -7.03 pct |
| Profile duration | 9.79 ms | 8.22 ms | -16.0% |

低寄存器版本会产生有限 stack spill，因此不适合所有矩阵。实际策略只在
`128×128` tile 且输出网格至少 1024 blocks，或 split-K 启用时选择它；小线性层和
attention 保留高寄存器无 spill 版本。

## 6. 严格实验协议

- GPU：Tesla V100-SXM2-16GB，SM70；Torch 2.6.0+cu124；CUDA 12.4。
- FP32，TF32 关闭，dropout=0，固定 synthetic MCM+PDR batch。
- 每形状 5 个独立进程 trial；每路线每 trial 50 warmup + 50 个完整训练步样本。
- 路线顺序按 trial 轮换；forward/backward/optimizer/step 使用 CUDA events。
- `zero_grad`、约 3.3-3.6 s 编译准备和每样本 50,000,000 cycle 时钟预热均在计时区外。
- 不删除离群样本。代表 self-hosted trial 中位数包含一个 17.6220 ms block，导致该
  形状单独 CI 下界为 0.9604；双形状预注册联合 CI 仍通过，异常被完整保留。
- CI 对 trial block 和 block 内样本做配对分层 bootstrap；两工作负载等权几何平均。

## 7. 正确性、恢复与依赖审计

| 项目 | 代表形状最大绝对误差 | 生产形状最大绝对误差 | 门槛 |
|---|---:|---:|---:|
| Total/MCM/PDR loss | 0 | 0 | 2e-5 |
| 全部梯度 | 1.86e-8 | 5.42e-9 | 2e-4 |
| 更新后参数 | 1.05e-6 | 4.44e-5 | 5e-5 |
| optimizer state | 2.79e-9 | 5.42e-10 | 2e-5 |

- 21 个真实矩阵族逐进程创建 Torch CUDA oracle，再注册 self-hosted Dispatcher；
  FP32 全张量比较全部通过。百万 K 随机归约最大差 `0.01109`，门槛按
  `max(5e-3, 2e-5*sqrt(K))=0.021`；模型级门槛独立且更严。
- 7 项 self-hosted 专项测试覆盖 FP32/FP16、三种布局、广播、default/out、双 stream、
  一阶 Autograd、二阶 fail-closed 与多张量 AdamW。
- 连续 2 步与 `1 + checkpoint reload + 1` 比较 140 个模型/优化器 tensor，逐位一致，
  最大差 0，最终参数 SHA-256 相同。
- `readelf -d` 的 NEEDED 仅包含 Torch 核心、CUDA Runtime 和系统库；`nm -D` 无外部
  GEMM 符号；两形状 profile 中原生 Torch GEMM 事件和禁止 kernel 事件均为 0。

Torch 二进制发行版本在 `import torch` 时可能自行预加载外部 BLAS 动态库。本报告的
可证明边界是：self-hosted extension 不链接这些库，active source 不引用它们，且正式
profile 的矩阵计算不调用它们；不能把“进程 maps 中完全没有预加载库”作为虚假结论。

## 8. 国产芯片离线门禁

AscendC 与 MUSA 项目继续使用既有 vendor 目录，NVIDIA self-hosted `.cu` 只在 CUDA/IX
构建分支加入。无需真实芯片的门禁结果：

- Python lazy-dispatch AST：通过；模块 import 时不加载 `torch_npu/torch_musa`；
- AscendC host/kernel 三文件 `g++ -fsyntax-only`：通过；
- MUSA registration 与两个 `.mu` 文件 host syntax：通过；
- 两项目 manifest、FP32/FP16/BF16 symbol matrix：通过；
- deferred CMake configure/build：通过。

这些结果只证明源代码结构和 host-visible 语法正确，不声称 CANN/MUSA 目标编译器生成
的二进制可运行。真实 Ascend 310 与 MTT S4000 测试仍必须租机执行。

## 9. 代码与证据

- [self-hosted CUDA GEMM/BMM/AdamW](https://github.com/priestess-bot/FlagGems/blob/4534b88689da2630145c75d7612284f3fecb61cc/cpp/lib/stofm_self_hosted_gemm.cu)
- [Python loader 与进程级 Dispatcher](https://github.com/priestess-bot/FlagGems/blob/4534b88689da2630145c75d7612284f3fecb61cc/src/flag_gems/experimental_ops/self_hosted_gemm.py)
- [SM70 独立构建入口](https://github.com/priestess-bot/FlagGems/blob/4534b88689da2630145c75d7612284f3fecb61cc/tools/build_stofm_self_hosted_gemm.py)
- [21 矩阵族验证器](https://github.com/priestess-bot/FlagGems/blob/4534b88689da2630145c75d7612284f3fecb61cc/tools/validate_stofm_self_hosted_shapes.py)
- [Gaussian/Pair NVIDIA 训练实现](https://github.com/priestess-bot/FlagGems/blob/4534b88689da2630145c75d7612284f3fecb61cc/src/flag_gems/experimental_ops/stofm_backends/nvidia.py)
- [SToFM self-hosted 训练路线](https://github.com/priestess-bot/SToFM/blob/b2f36fa80ad5f1bbe7575ece3a3e4d64f4ce56a6/benchmarks/stofm_phase2_v100_worker.py)
- [FlagOS 多张量 optimizer adapter](https://github.com/priestess-bot/SToFM/blob/b2f36fa80ad5f1bbe7575ece3a3e4d64f4ce56a6/model/flagos_optimizer.py)
- [Stage B 联合验收器](https://github.com/priestess-bot/SToFM/blob/b2f36fa80ad5f1bbe7575ece3a3e4d64f4ce56a6/benchmarks/validate_stofm_stageB_v100.py)

本地完整证据目录为
`artifacts/fake-training/v100-stageB-self-hosted-formal-20260901/`，包含 raw samples、
profile trace、correctness、checksum、acceptance、shape/checkpoint/offline gate 与
Nsight Compute report。HTML 只显示人类可读摘要，不把本地绝对路径当成代码链接。

## 10. 后续工作

1. 生产形状仍未超过 `torch.compile`；下一阶段需要更高效的双缓冲/vectorized load、
   warp-level SGEMM/PTX 或更大范围的编译图融合。
2. 当前 FP16 只通过单算子语义测试；端到端 AMP、GradScaler、overflow 与 checkpoint
   尚未进入正式矩阵。
3. 动态 shape、DDP/FSDP、CUDA Graph 完整 step 和多 stream 并发压力尚未验证。
4. Ascend 310 与 MTT S4000 必须在目标工具链重新编译，再执行与 V100 同构的
   Torch/初始 FlagOS/优化 FlagOS 三路性能及正确性协议。
