# SToFM FlagOS 假数据训练实施清单

更新时间：2026-08-30（Asia/Shanghai）

状态定义：`[ ]` 待完成，`[-]` 进行中，`[x]` 已完成，`[!]` 明确留待后续。

本清单只针对当前 V100、FlagGems ATen 训练路由和假数据 MCM+PDR 训练；不把
Torch-FL PrivateUse1、真实数据或国产芯片训练混入本轮验收。

本轮验收版本：SToFM `28e8794b0ceb93c5e7fa2fb1492bc2a3d3f6a42a`，FlagGems
`c2bee9932aa35730f9eeb919d24cf4e29202e4a1`。

PHASE 2 锁定版本：SToFM `2154e0d82fae98c5a3e0b7dd65028300d73f3962`，
FlagGems `a4bb672191bcdccdbc974f640a5e799fdd2ee9ae`。

## 1. 训练运行时

- [x] 新增 `flagos_training_scope()`，作用域覆盖 forward、backward 和 optimizer step。
- [x] 保留原有 `flagos_inference_scope()` 的推理行为，不改变既有 R2/MUSA 结果。
- [x] 增加显式训练算子白名单，避免全量 FlagGems 首次编译无关算子。
- [x] 增加训练 phase 与 strict 标记，并让模型读取外层 scope 的 provenance。

## 2. SToFM 训练语义

- [x] Gaussian 和 Pair-score 在训练时使用 registered FlagOS composite 的 Autograd-safe reference。
- [x] 保留推理专用 NVIDIA/MUSA native kernel 不被训练误用。
- [x] 补齐训练反向实际用到的 out-of-place `sgn` FlagGems kernel，并修正
  `true_divide/where_self/*_scalar/sum_dim` 的 manifest 函数名。
- [x] 保留 MCM + PDR 两项输出和损失字段。
- [x] 将 cosine loss 改写为等价的 mask 加权基础算子归约。
- [x] 将 pair MSE 改写为 mask 加权基础算子归约。
- [x] 覆盖 CLS、普通节点、padding 节点和部分 pair mask。

## 3. 假数据与训练器

- [x] 生成固定 seed 的 CPU 合成张量，不依赖 Scanpy、Geneformer、网络或真实数据。
- [x] 新增 `benchmarks/train_stofm_fake_flagos.py`。
- [x] 默认配置为 V100、FP32、2 层、N=12、10 步、单张量 AdamW。
- [x] 保存 loss 曲线、环境、参数摘要、checkpoint、算子清单和 Chrome trace。
- [x] 保存 FlagGems 函数日志与原始 CUDA kernel 名称摘要，区分注册证据和执行证据。
- [x] profile step 恢复模型/优化器/RNG 状态，避免 checkpoint 被额外更新一步。

## 4. 严格验证

- [x] Python 语法和 import 检查。
- [x] 纯 Torch 两步训练探索：forward/backward/AdamW 成功，loss 下降。
- [x] 受限 FlagGems 两步训练探索：FlagOS scope active，custom composite provenance 成功。
- [x] V100 strict 模式正式 10 步训练。
  证据：`artifacts/fake-training/v100-formal-20260830-final/`；FP32 总损失
  `2.239270687 -> 1.965290308`，10/10 步通过，最大梯度有限。
- [x] 训练 profile 与算子清单复核。
  证据：`training_profile.json`（95 个高层事件、725 个 CUDA kernel 事件）、
  `flaggems_ops.log`（42 个 FlagGems 函数族）、`fallback_report.json`（计算型
  fallback 为空）。
- [x] checkpoint 恢复后继续训练一致性测试。
  证据：`artifacts/fake-training/v100-resume-validation-20260830/`；使用
  `benchmarks/validate_fake_training.py --atol 1e-5`，step 10/11 指标和状态误差均
  小于声明的 `1e-5`（本次参数最大差异 `8.41e-8`；独立进程历史观测上界
  `3.72e-6`）。
- [x] MCM/PDR 与原始损失逐项数值等价测试。
  证据：`tests/test_fake_flagos_training.py` 的 cosine 与 pair-MSE reduction 对照
  测试；模型使用 mask 加权归约，覆盖 CLS、padding 和 pair mask。

## 5. 明确后续项

- [!] AdamW foreach/multi-tensor FlagGems kernel。
- [!] FP16/BF16 AMP、GradScaler 和统一 RNG。
- [!] Torch-FL PrivateUse1 设备后端训练。
- [!] DDP/FSDP/FlagCX 多卡训练。
- [!] 真实 SToFM 数据预处理和 Geneformer encoder 训练。

说明：FlagGems 全量通用测试在当前 V100 的 BF16（sm70 不支持）和超大 mean
用例上不纳入本轮验收；这不影响本轮 FP32 SToFM strict 训练结果。

## 6. 本轮验收边界

- [x] FlagOS 路由覆盖 forward、backward 和单张量 AdamW 更新；strict 模式不允许
  未批准的计算型 fallback。
- [x] 两个 SToFM 自定义边界在训练时走 registered FlagOS composite 的可微参考实现，
  不误用仅推理的 native Triton kernel。
- [x] V100 实际执行到 FlagGems 生成的 `mm/addmm/bmm/softmax/layer_norm` 等 kernel，
  但设备没有架构专门化调优配置；本轮结果不能外推到其他 GPU 或国产芯片。
- [!] Gaussian/pair-score 原生融合 backward、foreach AdamW、AMP、动态 shape、
  DDP/FSDP 仍是后续优化项，不得把本轮 FP32 参考图称为最终训练性能。

## 7. PHASE 2：V100 训练性能

分支：SToFM `r4/v100-training-performance`；FlagGems
`r4/v100-training-performance`。本阶段只优化 NVIDIA V100 FP32 单卡训练，所有
结论必须同时给出纯 PyTorch、初始 FlagOS 和优化后 FlagOS 三条可复现基线。

### 7.1 接口与算子实现

- [x] 固化 PHASE 2 分支、验收指标和三路对照定义。
- [x] 为 Gaussian pair bias 增加显式训练 ABI，原生前向配套解析反向，不改变
  原有推理 ABI 和参考训练路径。
- [x] 为 Pair-score attention 增加显式训练 ABI，原生 score/mask/softmax/context
  前向配套解析反向，保持 padding、pair-state 和权重返回语义。
- [x] 在 SToFM 配置中显式区分 `reference` 与 `native` 训练实现，并把实际选择写入
  dispatch provenance。
- [x] 增加 FlagOS AdamW 优化路径；实测实现为每个参数一次融合 Triton kernel，
  不是跨参数 foreach；生产 workload 性能门禁拒绝该候选，最终保留 scalar AdamW。

### 7.2 严格正确性

- [x] Gaussian 前向、输入梯度与全部参数梯度逐项对照 PyTorch FP32 参考实现。
- [x] Pair-score 前向、输入梯度逐项对照 PyTorch FP32 参考实现，覆盖 padding、
  `return_pair`、`return_weights` 和非默认 scale。
- [x] AdamW 参数、一阶矩、二阶矩和 step 状态逐项对照 PyTorch。
- [x] 完整 SToFM 第一步 loss、梯度和参数更新六路对照；最大 loss/梯度/参数/
  优化器状态绝对误差分别为 `1.19e-7`、`5.44e-9`、`3.20e-5`、`5.45e-10`。
- [x] 运行静态语法、CPU 单测、V100 集成测试和断点恢复回归。
  FlagGems `36 passed, 25 skipped`；SToFM `52 passed, 3 skipped`；Ascend/MUSA
  离线 gate `2 passed`；native resume 模型/优化器最大误差 `3.10e-7/1.49e-8`。

### 7.3 V100 性能实验

- [x] 固定硬件、软件、seed、模型配置、合成 batch 与初始权重哈希。
- [x] 独立进程预热，至少 30 个 CUDA event 原始样本；分别记录 forward、backward、
  optimizer 和完整 train step。
- [x] 保存纯 PyTorch、初始 FlagOS、优化后 FlagOS 结果，并增加算子优化/优化器优化
  消融，避免把收益错误归因。
- [x] 报告 median、mean、p90、p95、标准差、bootstrap 95% CI、吞吐和相对加速比。
- [x] 报告峰值 allocated/reserved 显存、kernel 启动数和关键 CUDA kernel 时间。
- [x] 保存 Chrome trace、机器可读 JSON、运行命令、环境快照、git revision 与原始样本。
- [x] 对候选 workload 做稳定性与显存探索，最终选择
  `B=1,N=1050,L=4,D=256,H=8,K=128`，优化路线峰值 allocated 显存
  `5.42 GiB`，在 V100 16GB 上有重复运行余量；未把微型 smoke shape 冒充性能结论。

### 7.4 交付与审计

- [x] 形成 PHASE 2 Markdown 技术报告，逐项解释计算公式、实现、收益与限制。
- [x] 将 PHASE 2 数据和可视化补入统一的 `stofm-flagos-training-report.html`，保持
  KaTeX、代码引用 GitHub 化和单文件离线打开。
- [x] Playwright 验收桌面/移动端：6/6 KaTeX、6 路性能图、逐算子图、trace 图，
  无横向溢出、外部资源请求或控制台错误。
- [x] 锁定两个 fork 的提交 SHA、正式实验目录、checksum、phase manifest 和最终结论。

## 8. PHASE A：FlagOS 自有 Vendor GEMM（r5）

目标：在不调用 Torch/ATen 原生 GEMM 的前提下，由 FlagOS C++/CUDA 后端直接调用
cuBLAS/cuBLASLt（CUTLASS 可选），并在 V100 生产形状及代表矩阵上稳定超过
PyTorch eager + fused AdamW。

### 8.1 基础设施

- [-] 建立 `r5/v100-vendor-gemm` 双 fork 分支并锁定基线版本。
- [ ] 建立 CMake 3.25+、CUDA 12.4、SM70 的隔离构建环境。
- [ ] 新增 FlagOS Vendor GEMM/BMM C++ ABI、CPU/meta reference 和 CUDA 实现。
- [ ] 链接 cuBLAS/cuBLASLt，验证当前 stream、handle、workspace 和错误传播。
- [ ] 增加算法选择缓存，禁止首次 heuristic/search 进入正式计时区。

### 8.2 严格 dispatch

- [ ] 增加 `flagos_gemm_backend=vendor` 配置和作用域 provenance。
- [ ] 接管 `mm/addmm/bmm/baddbmm` 及 out 变体，运行时不得调用 Torch native GEMM。
- [ ] 将 SToFM Gaussian、Pair、QKV、FFN、head 的所有 GEMM 统一接入 Vendor ABI。
- [ ] strict profile 证明 native GEMM、partial fallback、unmapped compute 均为 0。

### 8.3 训练反向融合

- [ ] Gaussian backward 改为 Vendor GEMM + FlagOS fused derivative/reduction kernel。
- [ ] Gaussian 实现 save/recompute/auto 三种 workspace 策略并按显存预算选择。
- [ ] Pair backward 改为 Vendor BMM + fused softmax/pair-mask backward。
- [ ] 消除重复 clone、transpose、contiguous 和不必要的 pair probability 写回。
- [ ] 保持一阶梯度语义；二阶梯度不支持时显式 fail-closed。

### 8.4 验收

- [ ] Vendor GEMM/BMM 单算子 correctness、stream、layout、stride、dtype 测试。
- [ ] 完整模型 loss、梯度、参数、optimizer state 和 checkpoint resume 测试。
- [ ] 生产形状 + 代表矩阵：5 trial × 50 CUDA event samples，10,000 bootstrap。
- [ ] aggregate median speedup >= 1.05x，bootstrap 95% 下界 > 1.0x。
- [ ] 每个形状 median 不慢于 Torch，峰值显存不超过 Torch 的 1.25x。
- [ ] 运行 Torch eager、Torch compile 辅助对照和 Nsight/Chrome trace 归因。

### 8.5 交付与阶段转换

- [ ] 生成 shape manifest、algorithm cache、raw samples、trace、checksum 和 phase manifest。
- [ ] 生成 `docs/flagos_v100_vendor_gemm_report.md`，说明实现、原理、收益和失败项。
- [ ] 更新统一 `reporting/stofm-flagos-training-report.html`，所有代码链接固定到 fork SHA。
- [ ] 推送两个 fork、打阶段 A tag，并核对远端 commit 可访问。
- [ ] 阶段 A 完成后创建阶段 B goal：移除 cuBLAS/cuBLASLt/CUTLASS，进入纯自研
  CUDA C++/PTX + Triton kernel 实现。
