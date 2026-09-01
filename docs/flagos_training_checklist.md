# SToFM FlagOS 假数据训练实施清单

更新时间：2026-09-01（Asia/Shanghai）

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

性能验收锁定版本：SToFM `e05f4d5c72f480a04c2259225d53ba7b79bb1207`，
FlagGems `8ac4ea5aa3ebdbe793cfda768c8ccee2b89e0c82`。

### 8.1 基础设施

- [x] 建立 `r5/v100-vendor-gemm` 双 fork 分支并锁定基线版本。
- [x] 建立 CUDA 12.4、SM70 隔离构建入口；`tools/build_stofm_vendor_gemm.py`
  使用 Torch C++ extension/Ninja 生成独立 `.so`。CMake 3.25 配置已验证到项目外部
  `TritonJIT` 依赖边界，Vendor 独立构建不依赖该组件。
- [x] 新增 FlagOS Vendor GEMM/BMM C++ ABI、CPU/meta reference 和 CUDA 实现。
- [x] 链接 cuBLAS/cuBLASLt，使用当前 handle/stream；完成双 CUDA stream、错误传播、
  layout/stride 与 FP16/FP32 测试。
- [x] 锁定确定性算法策略：大矩阵走 cuBLAS，V100 小 FP32 矩阵走 16×16 shared-memory
  tile；正式路径无 heuristic/search，因此无需运行时 algorithm cache。

### 8.2 严格 dispatch

- [x] 增加 `flagos_gemm_backend=vendor` 配置、进程级 C++ Dispatcher 和作用域 provenance。
- [x] 接管 `mm/addmm/bmm/baddbmm` 及四个 out 变体；默认与 out 测试全部通过。
- [x] 将 Gaussian、Pair、QKV、FFN、head 的 GEMM 统一接入 Vendor ABI；Q/K/V 三投影
  合并为一次 GEMM，参数名与 checkpoint 格式不变。
- [x] strict profile 逐项读取 Dispatcher table，四类 ATen GEMM 的 CUDA owner 均为
  `stofm_vendor_gemm.cu`，Torch native GEMM 为 0；非 GEMM CUDA kernel 作为 tuned surface
  的显式批准项单独列出，不冒充全量 FlagGems kernel 覆盖。

### 8.3 训练反向融合

- [x] Gaussian backward 改为 Vendor GEMM + 编译融合 derivative/reduction kernel；编译
  准备成本独立记录，不进入稳态计时。
- [x] 完成 workspace 策略评估并锁定 recompute：生产形状保存 RBF/hidden 会引入数百
  MiB 至 GiB 级常驻状态，当前 recompute 已使峰值显存低于 Torch；save/auto 不进入
  Stage A 正式路径，留给 Stage B 在新 kernel ABI 下重新评估。
- [x] 实现 Pair 原生 fused softmax/pair-mask backward；消融证明当前形状下标准 Autograd
  更快，因此 tuned 路线使用 reference backward，但其中全部 BMM 仍由 Vendor 接管。
- [x] C++ GEMM 直接消费 row-major、transpose view 与常见 slice stride，消除反向中
  不必要的 contiguous copy；QKV 融合减少投影 launch。
- [x] 保持一阶梯度语义；Gaussian/Pair 原生训练 ABI 使用 `once_differentiable`，二阶
  梯度 fail-closed。

### 8.4 验收

- [x] Vendor GEMM/BMM 单算子 correctness、双 stream、layout、stride、FP16/FP32 测试；
  FlagGems 相关回归 `27 passed`，新增 Vendor 专项 `5 passed`。
- [x] 完整模型 loss、梯度、参数、optimizer state 和 checkpoint resume 测试；1+1 恢复
  对 2 步连续训练的 loss/梯度/模型/优化器状态最大差异均为 `0`。
- [x] 生产形状 + 代表矩阵：每形状 5 trial × 50 CUDA event samples，20,000 bootstrap；
  不删除离群样本，逐样本固定周期 CUDA spin 在计时区外控制 V100 P-state。
- [x] 两形状等权几何平均加速 `1.2006x`，bootstrap 95% CI
  `[1.1881x, 1.2289x]`，通过 `>=1.05x` 与下界 `>1.0x` 门槛。
- [x] 代表形状 `16.7839 -> 15.8433 ms`（`1.0594x`）；生产形状
  `80.2452 -> 58.9778 ms`（`1.3606x`）；两形状均不慢。峰值显存比为
  `0.845x/0.771x`，通过 `<=1.25x` 门槛。
- [x] 运行 Torch eager 主对照、Torch compile 辅助对照（代表形状 `14.8168 ms`）以及
  Chrome trace；Nsight Systems 使用 CUDA Profiler range 单独捕获一个稳态训练步。

### 8.5 交付与阶段转换

- [x] 生成 workload/shape manifest、确定性 algorithm policy、两形状各 250 个 raw samples、
  trace、checksum、Nsight 与 `acceptance.json` phase manifest。
- [x] 生成 `docs/flagos_v100_vendor_gemm_report.md`，说明公式、实现、逐阶段收益、严格
  协议、正确性、profile、失败项与 Stage B 边界。
- [x] 更新统一单文件 `reporting/stofm-flagos-training-report.html`；代码链接全部固定到
  两个 fork SHA，8/8 KaTeX 离线渲染。Playwright 验收 1440×1000 与 390×844：
  无横向溢出、无重复 ID、无控制台/请求错误，形状切换与缺口过滤交互通过。
- [x] 推送两个 fork、打阶段 A tag，并核对远端 commit 可访问。
- [x] 阶段 A 完成后创建阶段 B goal；本节 9 继续执行纯自研 CUDA C++/PTX +
  Triton kernel 实现。

## 9. STAGE B：纯自研 V100 GEMM/BMM（r6）

目标：最终训练 route 不链接、不加载、不调用 cuBLAS、cuBLASLt 或 CUTLASS；使用
自研 CUDA C++/PTX 与 Triton，在保持 Stage A 严格正确性的前提下继续超过 PyTorch
eager + fused AdamW。状态定义沿用文件顶部。

### 9.1 分支与实验锁

- [x] 从两个 fork 的 `stofm-v100-stage-a-20260901` tag 创建
  `r6/v100-self-hosted-gemm`。
- [x] 从真实训练 trace 生成 `docs/stage_b_matrix_manifest.json`，覆盖代表/生产形状的
  forward、backward 与 fused QKV 矩阵族。
- [x] 固化 Stage A eager、初始 FlagOS、Vendor tuned 与 Torch compile 辅助基线。
- [x] 增加静态/动态禁用依赖门禁：源码、ELF NEEDED、Dispatcher owner、profile kernel
  均不得出现 cuBLAS/cuBLASLt/CUTLASS。

### 9.2 自研 GEMM/BMM

- [x] 新增独立 `flagos_stofm_self_hosted` C++ ABI 与 build 入口，只链接 CUDA runtime。
- [x] 实现 FP32 tiled GEMM：128×128 主 tile、skinny-M/N tile、transpose/stride 输入。
- [x] 实现 split-K workspace + reduction，覆盖 `M,N <= 128` 且超大 K 的 Gaussian
  weight-gradient 矩阵。
- [x] 实现 strided batched GEMM，覆盖三类 attention BMM 及 alpha/beta epilogue。
- [x] 补 FP16 正确路径；当前保留严格 FP32 accumulate 的 SIMT 路径，不冒险启用
  误差边界尚未证明的 Tensor Core。
- [x] 实现默认/out variant、广播 bias 与多 stream 语义。

### 9.3 训练融合与优化器

- [x] 将 QKV、FFN、head、Gaussian、Pair 的全部矩阵计算切换到 self-hosted backend；
  生产 profile 中 `aten::mm/addmm/bmm` 的 CUDA owner 全部为 self-hosted 实现。
- [x] Gaussian/Pair 反向编译图不得重新引入外部 GEMM；生产 profile 中 Gaussian
  analytical backward 与 Pair 四个 BMM 均由 self-hosted CUDA kernel 执行，外部 GEMM
  kernel 事件为 0。
- [x] packed AdamW 保持纯 CUDA kernel，并消除所有外部 BLAS 链接。
- [x] 保持 checkpoint/state dict 与 Stage A 兼容，一阶梯度一致，二阶梯度 fail-closed；
  连续 2 步与 1+1 resume 的模型/优化器共 140 个 tensor 逐位一致，最大误差为 0。

### 9.4 严格测试与性能

- [x] 单算子：shape manifest 全覆盖，FP32/FP16、layout、stride、broadcast、out、stream；
  7 项语义测试与 21 个真实矩阵族隔离进程 oracle 全部通过。
- [x] 模型：loss、全部梯度、更新后参数、optimizer state、1+1 checkpoint resume；代表/
  生产梯度最大误差 1.86e-8/5.42e-9，resume 的 140 个 tensor 逐位一致。
- [x] 代表形状与生产形状各 5 trial × 50 样本，20,000 次分层 bootstrap。
- [x] 两形状均快于 PyTorch eager；联合 speedup 1.1384x，95% CI
  [1.0410x, 1.1505x]。
- [x] 峰值显存 <= Torch 1.25x；实际为 0.643x/0.648x，并将约 3.3-3.6 s
  Gaussian/Pair 编译准备与稳态 CUDA-event 计时分离。
- [x] Ascend CANN / MTT MUSA 离线 gate 保持通过；host syntax、manifest、deferred
  CMake configure/build 均通过，且明确不声称真实芯片二进制。

### 9.5 交付

- [x] 生成 Stage B raw samples、profile、Nsight Compute、checksum、shape/checkpoint/
  dependency acceptance manifest。
- [x] 更新 Markdown 技术报告与统一单文件 HTML；11/11 KaTeX 离线渲染，Playwright
  1440×1000 / 390×844 无溢出、重复 ID、console/request 错误，形状切换与缺口过滤通过。
- [ ] 推送 r6 双 fork、创建 Stage B tag，核对远端 commit 可访问。
