# FlagOS SToFM 注册算子 V100 验证报告

## 结论

本轮完成的主要工作是**补齐 FlagOS 自定义算子并让真实 SToFM 推理调用它们**，不是把
`torch.compile` 的结果重新命名为 FlagOS 优化。V100 上已经实际注册、调用并由 profiler
确认的 SToFM 算子是：

1. `flagos_stofm::gaussian_pair_bias`
2. `flagos_stofm::pair_score_epilogue`

两者在完整 SToFM 模型中同时启用、且关闭普通 ATen 接管时，相对固定版本的未优化 FlagOS，
端到端 p50 为 FP32 **1.566x**、FP16 **1.690x**。再叠加既有 FlagOS ATen 接管后，
分别为 **1.607x** 和 **1.707x**。后一个数字包含既有 `addmm`、`baddbmm`、`bmm`、
`softmax` 的贡献，不能归因给新算子。

此前的 Inductor 结果保留为历史编译器实验，但不再作为“补齐 FlagOS 算子”的完成证明；
Inductor 现在仅可通过显式 `backend="inductor"` 选择，默认 CUDA 优化路线为注册算子。

## 本轮交付与边界

| 边界 | 本轮实际交付 | V100 证据 | 当前限制 |
| --- | --- | --- | --- |
| Gaussian pair bias | 注册 `torch.ops.flagos_stofm.gaussian_pair_bias`；CUDA 绑定 Triton；CPU 和 Autograd 走参考实现 | 完整 SToFM 端到端 profiler 可见该算子 | 原生 CUDA 路径限连续 FP32/FP16、推理模式、隐藏维度不大于 128、头数不大于 16；其他情况安全回退 |
| Pair score epilogue | 注册 `torch.ops.flagos_stofm.pair_score_epilogue`；包含 score、mask、softmax、可选 pair-state 返回 | 完整 SToFM 端到端 profiler 可见该算子 | 原生 CUDA 路径限连续 FP32/FP16、无 dropout、推理模式、key 长度不大于 4096；其他情况安全回退 |
| Marker-token embed | 注册 `torch.ops.flagos_stofm.marker_token_embed` 并绑定 FlagGems Vision 公共 API | CUDA ABI/profiler 与 FP32/FP16 单元测试通过 | 当前仓库没有可接入的真实 Uni2/KRONOS 模型调用点；不报告其为 Uni2/KRONOS 端到端收益 |
| Ascend 310 / CANN | Gaussian 与 pair-score 使用同一 `flagos_stofm` ABI 的延迟源码项目 | Python/AST/schema/CMake/C++ 语法检查通过 | 未租赁设备，未编译 CANN 二进制，未作性能声明 |
| MTT S4000 / MUSA | Gaussian 与 pair-score 使用同一 `flagos_stofm` ABI 的 `PrivateUse1` 延迟扩展 | Python/AST/schema/CMake/C++ 语法检查通过 | 未租赁设备，未编译 MUSA 扩展，未作性能声明 |

实现入口均在已推送 fork 中：
[FlagGems 注册 ABI](https://github.com/priestess-bot/FlagGems/blob/399d0381ed63a79018f3112ecc43894fd58ba052/src/flag_gems/experimental_ops/_flagos_stofm_ops.py)、
[SToFM 调用桥](https://github.com/priestess-bot/SToFM/blob/fc37f17c25f1c925da451134c979a45d2dbec6bb/model/flagos_backend.py)、
[SToFM 模型接入](https://github.com/priestess-bot/SToFM/blob/fc37f17c25f1c925da451134c979a45d2dbec6bb/model/se2transformer.py)、
[NVIDIA Triton 后端](https://github.com/priestess-bot/FlagGems/blob/399d0381ed63a79018f3112ecc43894fd58ba052/src/flag_gems/experimental_ops/stofm_backends/nvidia.py)。

## 三条基线的定义

- **纯 PyTorch**：不创建 FlagOS ATen 作用域，不调用任何 FlagOS 自定义算子。
- **固定版本的未优化 FlagOS**：独立 Python 进程、固定 FlagGems 提交
  `03bf364ede763d573d5c30124d554283a209ab85`，仅启用原有受限 ATen 作用域。
- **优化后 FlagOS**：独立 Python 进程、FlagGems 提交
  `399d0381ed63a79018f3112ecc43894fd58ba052`，选择注册 SToFM 算子；是否叠加原有
  ATen 作用域在表中明确写出。

每个固定版与优化版工作进程都会记录实际导入的 `flag_gems/__init__.py` 路径；证据验证器
拒绝导入路径越过各自源码根的结果。因此“固定版本的未优化 FlagOS”不是优化版包的别名。

## V100 端到端结果

设备为 Tesla V100-SXM2-16GB（compute capability 7.0），NVIDIA 驱动 550.144.03，
PyTorch 2.6.0+cu124，CUDA 12.4，Triton 3.2.0。工作负载固定为
`B=1, N=1050, L=4, D=256, FFN=256, H=8, K=128`，并设置
`return_pair_rep=False`。

速度比以“固定版本的未优化 FlagOS（作用域保持）”为基线，除非该行明确写为相对纯 PyTorch。
括号内为从 90 个原始样本、10,000 次 bootstrap 重采样得到的 95% 区间。

| 路线 | FP32 p50 (ms) | FP32 速度比 | FP16 p50 (ms) | FP16 速度比 | 归因 |
| --- | ---: | --- | ---: | --- | --- |
| 纯 PyTorch | 23.1114 | 参考 | 21.0362 | 参考 | 无 FlagOS |
| 固定版本的未优化 FlagOS，作用域保持 | 21.6959 | 相对纯 PyTorch 1.065x [1.065x, 1.065x] | 19.3281 | 相对纯 PyTorch 1.088x [1.088x, 1.089x] | 仅既有 ATen 接管 |
| 仅 Gaussian 自定义算子，关闭 ATen 接管 | 16.4589 | 1.321x [1.318x, 1.323x] | 13.4554 | 1.436x [1.436x, 1.438x] | 新 Gaussian 算子 |
| 仅 pair-score 自定义算子，关闭 ATen 接管 | 20.5631 | 1.055x [1.055x, 1.056x] | 19.0429 | 1.015x [1.015x, 1.015x] | 新 pair-score 算子 |
| 两个自定义算子，关闭 ATen 接管 | 13.8535 | 1.566x [1.564x, 1.566x] | 11.4339 | 1.690x [1.690x, 1.691x] | 仅新算子组合 |
| 两个自定义算子 + 既有 ATen 接管，作用域保持 | 13.5012 | 1.607x [1.606x, 1.608x] | 11.3227 | 1.707x [1.706x, 1.708x] | 新算子 + 既有 ATen 接管 |
| 两个自定义算子 + 既有 ATen 接管，每调用创建作用域 | 13.5420 | 1.602x [1.602x, 1.603x] | 11.3478 | 1.703x [1.702x, 1.703x] | 生命周期成本已包含 |

## 逐算子结论

| 算子 | FP32，仅该算子 | FP16，仅该算子 | 是否作为独立默认路线 | 组合路线结论 |
| --- | --- | --- | --- | --- |
| Gaussian pair bias | 1.321x，相对未优化 FlagOS | 1.436x，相对未优化 FlagOS | 是 | 两精度均是主要收益来源 |
| Pair score epilogue | 1.055x，相对未优化 FlagOS | 1.015x，相对未优化 FlagOS | 否 | FP16 独立收益低于 1.05x 预设阈值；保留在已验证的组合路线中，不单独宣传 |
| 两个算子组合 | 1.566x，相对未优化 FlagOS | 1.690x，相对未优化 FlagOS | 是 | 这是不含既有 ATen 接管的真实新算子端到端结果 |
| 两个算子组合 + 既有 ATen 接管 | 1.607x，相对未优化 FlagOS | 1.707x，相对未优化 FlagOS | 是 | 比只用新算子再快约 2.6% / 1.0%；增量属于既有 ATen 接管与交互效应 |

每个优化阶段的 profiler 证据均显式记录了实际事件：

- Gaussian 阶段：`flagos_stofm::gaussian_pair_bias`
- Pair-score 阶段：`flagos_stofm::pair_score_epilogue`
- 双算子阶段：以上两个事件同时出现

“仅新算子”三个阶段的 runtime dispatch 为 inactive，证明其没有被普通
`flag_gems.use_gems()` ATen 覆盖混淆。

## 严格测试过程

1. 两种精度分别运行 3 个独立的固定版/优化版进程对；每个已测阶段有 10 次预热、30 个
   CUDA-event 样本、每个样本 5 次模型调用，即每阶段 90 个原始样本。
2. 禁用 TF32 与 CuDNN benchmark；启用 inference mode；编译不计入计时。本轮注册算子
   套件不请求 Inductor。
3. 每一阶段均先用 `torch.testing.assert_close` 与本进程纯 PyTorch 参考输出比较；FP32
   使用 `rtol=3e-4, atol=3e-5`，FP16 使用 `rtol=3e-2, atol=3e-3`。
4. 聚合器拒绝纯 PyTorch 参考哈希、工作负载、精度或基准套件不一致的进程对。
5. 工作进程用 profiler 断言每个要求的 `torch.ops.flagos_stofm` 事件存在；证据验证器
   再次检查事件、样本数、计时配置、提交一致性、导入根与 checksum manifest。

以下检查已通过：

```text
FlagGems: 注册 ABI + SToFM/Vision API 测试                 29 passed
FlagGems: Ascend/MUSA 延迟项目测试                          2 passed
SToFM: provenance、基准套件、完整模型适配器测试           18 passed
SToFM: 已提交 FP32/FP16 原始证据验证器                      2 passed
```

实际原始证据和 checksum manifest：
[FP32](https://github.com/priestess-bot/SToFM/tree/r2/flagos-inference/benchmark-results/r3-v100-registered-ops-fp32-20260816)、
[FP16](https://github.com/priestess-bot/SToFM/tree/r2/flagos-inference/benchmark-results/r3-v100-registered-ops-fp16-20260816)、
[工作进程](https://github.com/priestess-bot/SToFM/blob/fc37f17c25f1c925da451134c979a45d2dbec6bb/benchmarks/stofm_r2_v100_worker.py)、
[只读证据验证器](https://github.com/priestess-bot/SToFM/blob/r2/flagos-inference/benchmarks/verify_stofm_registered_ops_evidence.py)。

## Ascend 310 与 MTT S4000 后续实验

两个目标的源码均统一使用 `flagos_stofm::gaussian_pair_bias` 和
`flagos_stofm::pair_score_epilogue` ABI。当前只验证延迟导入、Python 接口、schema、
CMake 配置、宿主可见 C++ 语法与清单，不把它们表示为真实芯片性能。

租赁设备后按以下顺序执行：

1. 安装对应的 vendor PyTorch 扩展和 SDK，编译目标扩展，并确认实际 `torch.ops` 注册。
2. 对 FP32、FP16、BF16 逐一执行 Gaussian、pair-score、mask、padding、非连续输入回退
   与梯度回退矩阵。
3. 在目标设备重新采集纯 PyTorch、固定版本未优化 FlagOS、仅新算子、以及新算子叠加
   ATen 接管四条路线，不能复用 V100 数字。
4. 以每个目标自身的未优化 FlagOS 为基线；仅当 Gaussian 达到 1.15x、pair-score 达到
   1.05x、双算子端到端达到 1.10x，且数值与稳定性门槛都通过时，才晋升默认路线。

目标源码入口：
[AscendC](https://github.com/priestess-bot/FlagGems/tree/399d0381ed63a79018f3112ecc43894fd58ba052/src/flag_gems/experimental_ops/vendor/ascendc_stofm)、
[MUSA](https://github.com/priestess-bot/FlagGems/tree/399d0381ed63a79018f3112ecc43894fd58ba052/src/flag_gems/experimental_ops/vendor/musa_stofm)、
[离线检查器](https://github.com/priestess-bot/FlagGems/blob/399d0381ed63a79018f3112ecc43894fd58ba052/tools/check_deferred_native_projects.py)。

## 版本与可复现性

- SToFM 被测提交：`fc37f17c25f1c925da451134c979a45d2dbec6bb`
- 固定版本 FlagGems：`03bf364ede763d573d5c30124d554283a209ab85`
- 优化后 FlagGems：`399d0381ed63a79018f3112ecc43894fd58ba052`
- 本报告中的数据仅适用于声明的 V100、形状、精度和软件栈；不外推到 Ascend、MTT 或
  真实 Uni2/KRONOS 端到端模型。
