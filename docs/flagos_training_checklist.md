# SToFM FlagOS 假数据训练实施清单

更新时间：2026-08-30（Asia/Shanghai）

状态定义：`[ ]` 待完成，`[-]` 进行中，`[x]` 已完成，`[!]` 明确留待后续。

本清单只针对当前 V100、FlagGems ATen 训练路由和假数据 MCM+PDR 训练；不把
Torch-FL PrivateUse1、真实数据或国产芯片训练混入本轮验收。

本轮验收版本：SToFM `9c1f29774fcb399aee7ca2ce24be7df9b78f9ef6`，FlagGems
`c2bee9932aa35730f9eeb919d24cf4e29202e4a1`。

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
  小于声明的 `1e-5`（参数最大差异 `3.72e-6`）。
- [x] MCM/PDR 与原始损失逐项数值等价测试。
  证据：`tests/test_fake_flagos_training.py` 的 cosine 与 pair-MSE reduction 对照
  测试；模型使用 mask 加权归约，覆盖 CLS、padding 和 pair mask。

## 5. 明确后续项

- [!] AdamW foreach/multi-tensor FlagGems kernel。
- [!] FP16/BF16 AMP、GradScaler 和统一 RNG。
- [!] Torch-FL PrivateUse1 设备后端训练。
- [!] DDP/FSDP/FlagCX 多卡训练。
- [!] 真实 SToFM 数据预处理和 Geneformer encoder 训练。

## 6. 本轮验收边界

- [x] FlagOS 路由覆盖 forward、backward 和单张量 AdamW 更新；strict 模式不允许
  未批准的计算型 fallback。
- [x] 两个 SToFM 自定义边界在训练时走 registered FlagOS composite 的可微参考实现，
  不误用仅推理的 native Triton kernel。
- [x] V100 实际执行到 FlagGems 生成的 `mm/addmm/bmm/softmax/layer_norm` 等 kernel，
  但设备没有架构专门化调优配置；本轮结果不能外推到其他 GPU 或国产芯片。
- [!] Gaussian/pair-score 原生融合 backward、foreach AdamW、AMP、动态 shape、
  DDP/FSDP 仍是后续优化项，不得把本轮 FP32 参考图称为最终训练性能。
