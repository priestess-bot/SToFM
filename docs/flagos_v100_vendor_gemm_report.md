# SToFM FlagOS V100 Vendor GEMM 训练优化报告

日期：2026-09-01  
性能实现版本：SToFM `e05f4d5c72f480a04c2259225d53ba7b79bb1207`；FlagGems
`8ac4ea5aa3ebdbe793cfda768c8ccee2b89e0c82`

## 1. 结论

Stage A 通过预先锁定的全部主门槛：FlagOS 不调用 Torch 原生 GEMM，直接由 C++
Dispatcher 接管 `mm/addmm/bmm/baddbmm`，在 V100 FP32 的代表形状与生产形状上均超过
PyTorch eager + CUDA fused AdamW。

| 形状 | PyTorch eager + fused AdamW | 初始 FlagOS | FlagOS Vendor tuned | 相对 PyTorch | 95% CI |
|---|---:|---:|---:|---:|---:|
| 代表形状 `N=384,D=64,L=2` | 16.7839 ms | 117.3724 ms | 15.8433 ms | 1.0594x | [1.0458x, 1.1092x] |
| 生产形状 `N=1050,D=256,L=4` | 80.2452 ms | 404.3295 ms | 58.9778 ms | 1.3606x | [1.3364x, 1.3708x] |

两形状等权几何平均加速为 **1.2006x**，20,000 次分层 bootstrap 的 95% CI 为
**[1.1881x, 1.2289x]**。代表/生产峰值 allocated 显存分别为 Torch 的
`0.845x/0.771x`。第一步 loss 完全一致，梯度、参数与优化器状态全部通过阈值。

边界必须同时说明：

- 主验收对象是 PyTorch eager + CUDA fused AdamW，按实验计划锁定。
- `torch.compile` 辅助对照在代表形状为 `14.8168 ms`，仍比 Vendor tuned 快约 6.9%。
- tuned surface 中四类 GEMM、Gaussian 和 AdamW 属于 FlagOS；非 GEMM 点算子使用显式
  批准的成熟 CUDA device kernel。它不是“所有通用 ATen 算子都由 FlagGems kernel
  替换”的结果；全量 FlagGems registrar 对照仍为 `117.37/404.33 ms`。

## 2. 计算内容

### 2.1 GEMM 与 BMM

FlagOS Vendor ABI 覆盖：

\[
C = \alpha AB
\]

\[
C = \beta X + \alpha AB
\]

以及批量形式：

\[
C_b = \alpha A_bB_b,\qquad
C_b = \beta X_b + \alpha A_bB_b
\]

分别对应 `mm`、`addmm`、`bmm`、`baddbmm`，同时覆盖四个 out variant。大矩阵直接
调用 cuBLAS；SM70 上满足阈值的小 FP32 矩阵使用 16×16 shared-memory tile。row-major
输入通过 \(C^T=B^TA^T\) 映射给 column-major cuBLAS，转置 view 与常见 slice stride
不再强制 `contiguous()`。

实现：
[CUDA/C++ Dispatcher](https://github.com/priestess-bot/FlagGems/blob/8ac4ea5aa3ebdbe793cfda768c8ccee2b89e0c82/cpp/lib/stofm_vendor_gemm.cu)，
[Python schema/loader](https://github.com/priestess-bot/FlagGems/blob/8ac4ea5aa3ebdbe793cfda768c8ccee2b89e0c82/src/flag_gems/experimental_ops/vendor_gemm.py)。

### 2.2 Gaussian pair bias

SToFM 对每个距离 \(d\) 先做标量仿射，再构造 \(K\) 个 Gaussian basis：

\[
z=wd+b,\qquad \sigma_k=|s_k|+10^{-5}
\]

\[
r_k=\frac{\exp\left[-\frac12\left(\frac{z-\mu_k}{\sigma_k}\right)^2\right]}
{\sqrt{2\pi}\sigma_k}
\]

随后进行两层投影：

\[
h=\operatorname{ReLU}(rW_1^T+b_1),\qquad y=hW_2^T+b_2
\]

最终输出布局为 `[batch, head, node, node]`，原始距离为零的位置输出零。前向使用
Triton fused kernel；反向选择 recompute，重新生成 RBF/hidden，并由编译图融合导数和
归约，矩阵乘仍由 Vendor Dispatcher 执行。该策略避免保存生产形状下数百 MiB 至 GiB
级中间状态，使生产峰值显存从 Torch 的 4.303 GiB 降至 3.320 GiB。

实现：
[NVIDIA Gaussian forward/backward](https://github.com/priestess-bot/FlagGems/blob/8ac4ea5aa3ebdbe793cfda768c8ccee2b89e0c82/src/flag_gems/experimental_ops/stofm_backends/nvidia.py)。

### 2.3 Pair attention

每层 attention 计算：

\[
S=\gamma QK^T+B_{pair},\qquad P=\operatorname{softmax}(S),\qquad O=PV
\]

Pair state 需要保留 mask 后的 score，而不是标准 SDPA 只返回 context。项目已经实现
原生 fused score/mask/softmax 与解析反向；但消融显示当前两个形状上，参考 Autograd
更快。因此正式 tuned 路线直接运行 SToFM reference attention graph，去掉重复
`torch.ops` 包装边界，同时所有 `QK^T` 与 `PV` BMM 仍由 Vendor C++ 接管。

### 2.4 QKV 与 AdamW

每层原先三次独立 Q/K/V 线性投影改为权重/bias 拼接后一次 GEMM，再 `chunk(3)`；参数
名称、梯度与 checkpoint 格式保持不变。实现位于
[SToFM attention](https://github.com/priestess-bot/SToFM/blob/e05f4d5c72f480a04c2259225d53ba7b79bb1207/model/se2transformer.py)。

AdamW 更新为：

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
\]

\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
\]

\[
\theta_t=\theta_{t-1}-\eta\left(
\frac{m_t/(1-\beta_1^t)}{\sqrt{v_t/(1-\beta_2^t)}+\epsilon}
+\lambda\theta_{t-1}\right)
\]

固定 kernel 参数包一次携带最多 64 组 parameter/gradient/moment/step 指针；4 层生产模型
的 83 个参数自动拆为两个 launch。实现适配器位于
[FlagOS optimizer](https://github.com/priestess-bot/SToFM/blob/e05f4d5c72f480a04c2259225d53ba7b79bb1207/model/flagos_optimizer.py)。

## 3. 优化过程与归因

| 阶段 | 生产 step | 相对上一正式阶段 | 核心变化 |
|---|---:|---:|---|
| 初始 FlagOS reference | 404.3295 ms | - | 通用 Triton GEMM、完整 Python registrar、scalar AdamW |
| PHASE 2 native | 约 335.33 ms | 1.228x | Gaussian/Pair 原生前向与解析反向 |
| Stage A Vendor tuned | 58.9778 ms | 约 5.69x | C++ direct ATen dispatch、cuBLAS、QKV 合并、Gaussian 编译反向、packed AdamW、tuned surface |

Stage A 对初始 FlagOS 的生产端到端加速为 **6.855x**，延迟下降约 **85.4%**。收益并非
“换一个算子名字”：

1. 删除 generic `mm_kernel_general` 的 SM70 路径，直接使用 Volta cuBLAS kernel。
2. 删除每次 GEMM 的 Python callback/custom-op 二次 dispatch。
3. 让转置 view 直接进入 cuBLAS，删除反向中的布局 copy。
4. QKV 由三次投影合并为一次。
5. Gaussian 点算子与归约编译融合。
6. AdamW 从每参数多次 elementwise launch 改为 1–2 次 packed launch。
7. Pair reference graph 去掉无收益的自定义算子包装，但 BMM owner 不变。

## 4. 严格实验协议

- GPU：Tesla V100-SXM2-16GB，SM70；Torch 2.6.0+cu124；Triton 3.2.0。
- 精度：FP32，TF32 关闭，cuDNN benchmark 关闭，dropout=0。
- 数据：固定 seed 的合成 MCM+PDR batch；各路线初始参数与 batch SHA 一致。
- 每个形状 5 个独立进程 trial；每 trial 50 个完整训练步 CUDA event 样本。
- 每个样本记录 forward/backward/optimizer/step；`zero_grad` 不在计时区。
- 服务器无应用时钟锁权限；每个样本前执行固定 50,000,000 cycle device spin，位于
  计时区外，防止短 kernel 路线因 V100 P-state 降频被不公平惩罚。
- 不删除任何离群样本；报告 median、mean、P95、trial CV 和全部 raw samples。
- CI：trial block + trial 内 CUDA-event 样本的配对分层 bootstrap；跨形状采用固定
  workload matrix、等权几何平均，不对两个工作负载本身重采样。

证据目录：`artifacts/fake-training/v100-stageA-vendor-formal-20260901/`。核心入口为
`acceptance.json`，两组原始 suite 为 `representative-5x50/` 与 `production-5x50/`。

## 5. 正确性与恢复

| 项目 | 代表形状最大绝对误差 | 生产形状最大绝对误差 | 门槛 |
|---|---:|---:|---:|
| Total/MCM/PDR loss | 0 | 0 | 2e-5 |
| 梯度 | 1.49e-8 | 5.41e-9 | 2e-4 |
| 更新后参数 | 1.02e-6 | 3.18e-5 | 5e-5 |
| AdamW state | 3.73e-9 | 5.41e-10 | 2e-5 |

checkpoint 使用连续 2 步与 1+1 恢复对照；step 1 指标、最终模型和优化器状态最大差异
均为 `0`。Vendor GEMM 专项测试覆盖 FP32/FP16、transpose/slice、广播、退化维度、
default/out、Autograd、双 CUDA stream 与超过 64 参数的 AdamW 分包。

## 6. Profile 与 provenance

| 形状 | Torch kernel events | 初始 FlagOS | Vendor tuned |
|---|---:|---:|---:|
| 代表形状 | 286 | 710 | 254 |
| 生产形状 | 420 | 1215 | 379 |

Profiler 仍显示高层事件名 `aten::mm/addmm/bmm`，因为 Autograd schema 属于 ATen；严格
审计读取 Dispatcher table 的 CUDA entry，四类 GEMM 均注册于
`stofm_vendor_gemm.cu`，`native_aten_gemm_event_count=0`。AutogradCUDA entry 保留
PyTorch 生成的导数规则，这不等于调用 Torch CUDA GEMM；导数中的 GEMM 会再次下沉到
FlagOS CUDA key。

Nsight Systems 使用 `cudaProfilerStart/Stop` 只捕获一个稳态训练步。代表形状中可见
Volta `sgemm`、`_gaussian_pair_bias_kernel`、融合 Triton derivative/reduction、
`small_gemm_fp32_kernel` 与 packed AdamW；未使用包含时钟预热的全程 capture 作为结论。

## 7. 测试结论

- FlagGems NVIDIA/算子回归：`27 passed`；Vendor 专项最终：`5 passed`。
- SToFM runtime/adapter/protocol/training/optimizer：`26 passed, 1 skipped`，另有 70 参数
  packed AdamW 分包测试通过。
- CANN/MUSA 离线 gate：FlagGems `3 passed`；SToFM `15 passed`。
- Python syntax 与 `git diff --check` 通过。
- CUDA 12.4 / SM70 独立 extension 构建通过；项目全量 CMake 配置需要外部 TritonJIT，
  Vendor 独立构建入口不依赖它。

## 8. 未完成与 Stage B

- `torch.compile` 代表形状 14.8168 ms，当前仍领先 Vendor tuned 15.8433 ms。
- 非 GEMM 通用算子尚未全部换成高性能 FlagGems kernel；强制全量 registrar 会显著变慢。
- AMP/FP16 端到端训练、动态 shape、DDP/FSDP 尚未纳入 Stage A。
- CUDA Graph 完整训练捕获仍受动态 Autograd 临时 buffer 影响；本阶段使用稳定的 eager
  C++ Dispatcher + 编译子图。
- Stage B 将移除 cuBLAS/cuBLASLt/CUTLASS，使用自研 CUDA C++/PTX + Triton，并重新
  评估 Gaussian save/recompute/auto 与国产芯片训练 kernel。
