# SToFM FlagOS 训练算子缺口与解决方案

## 结论

当前 SToFM 代码原本只对推理开放 FlagOS：训练时 `flagos_inference_scope()` 会主动
关闭 dispatch，Gaussian/Pair-score 也会回到 Torch reference。训练图 profile 还显示，
真正需要审计的不只是 `addmm/bmm`，而是前向、反向、损失和优化器四类算子。

本轮的严格目标是：在固定 V100、FP32、固定小 shape 的假数据训练中，计算型事件都能
通过 FlagGems 路由或 FlagOS composite 解释；不能解决的能力必须显式留在“后续项”，
不能静默回退后仍称完全 FlagOS。

## 训练图观测

最小 SToFM MCM+PDR 训练 step（2 层、B=2、N=12、D=16）观测到的主要事件包括：

```text
前向：mm, addmm, bmm, native_layer_norm, gelu, softmax,
      exp, pow, div, mul, masked_fill, where, index
反向：mm, addmm, bmm, native_layer_norm_backward,
      gelu_backward, _softmax_backward_data,
      threshold_backward, embedding_dense_backward,
      mse_loss_backward, index_select_backward
损失：linalg_vector_norm, cosine_embedding_loss, mse_loss, sum, mean
优化器：_foreach_addcdiv_, _foreach_addcmul_, _foreach_mul_,
        _foreach_div_, _foreach_sqrt, _foreach_lerp_, add_, copy_
```

Profile 原始依据：`torch.profiler` CUDA/CPU key averages；训练阶段还保存
`training_trace.json`，可用 Chrome tracing 或 Perfetto 打开。

## 本轮 V100 实测结论

正式命令（固定 seed、FP32、B=2、N=12、2 层、10 步）为：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../FlagGems-stofm/src \
../.venv-flagos-r2/bin/python benchmarks/train_stofm_fake_flagos.py \
  --device cuda:0 --steps 10 --strict --profile \
  --output artifacts/fake-training/v100-formal-20260830-final
```

结果为 `status=passed`、`fallback_compute_ops=[]`：总损失从 `2.239270687` 降至
`1.965290308`，MCM 从 `0.930582464` 降至 `0.658767462`，PDR 从 `1.308688283`
降至 `1.306522846`。首步约 `7099.8 ms`（包含惰性编译），去掉首步后的 steady
step p50 约 `247.5 ms`；相对首步约 `28.7x`，但这是小 shape smoke test，不是完整
模型吞吐。总损失下降约 `12.24%`，MCM 下降约 `29.21%`，PDR 下降约 `0.17%`。

执行证据有三层：

1. `run.json` 的 scope 记录显示训练 phase active，Gaussian 与 pair-score 都选择
   `nvidia` 的 registered composite。
2. `flaggems_ops.log` 记录了 `mm/addmm/bmm/softmax/layernorm/vector_norm` 等
   FlagGems 函数调用。
3. `training_trace.json` 的 CUDA kernel 标签包含 `mm_kernel_general`、`addmm_kernel`、
   `bmm_kernel`、`softmax_kernel_inner`、`layer_norm_backward_kernel` 等；
   `training_profile.json` 同时保留原始标签和 725 个 kernel 事件计数；
   `operator_attribution` 通过 `External id` 逐调用区分 FlagGems 与原生 kernel。

V100 的 `compute capability=(7,0)` 在当前 FlagGems 版本没有架构专门化 profile，运行
时会提示 `Unsupported GPU arch ... specialization`。这不是本次 strict fallback：
通用 Triton kernel 已执行；但它意味着尚未完成 V100 专门 tile/autotune，不能把本轮
steady latency 当作最终优化上限。kernel 名称是原始 profiler 证据，正式发布前仍应
用 Nsight Systems/Compute 做最终归因。

实现对应的两个 fork 提交：

- SToFM 训练桥：
  `https://github.com/priestess-bot/SToFM/tree/b8bd8996e895d61fea206e253177b5ea167c21a0`
- FlagGems 训练算子：
  `https://github.com/priestess-bot/FlagGems/tree/c2bee9932aa35730f9eeb919d24cf4e29202e4a1`

checkpoint 恢复由 `benchmarks/validate_fake_training.py` 验证：step 10 的 loss、MCM、
PDR 与连续 11 步运行一致；本次最终对照的最大梯度差异为 `0`，模型参数最大绝对
差异 `6.24e-8`，优化器状态 `2.09e-7`。此前多次独立进程复核的参数漂移上界为
`3.72e-6`，仍低于声明的 FP32 GPU 容差 `1e-5`。不同进程可能选择不同的 kernel
调优/归约顺序，因此不要求参数字节级 SHA-256 相同；若要做更严格审计，可传
`--atol 1e-6`，失败时应记录为数值漂移而非静默放宽。

## 缺口与处理矩阵

| 训练环节 | 观测算子/能力 | FlagGems 当前状态 | 本轮处理 | 后续完整解决方案 |
|---|---|---|---|---|
| Gaussian forward | `exp/pow/div/mul/addmm` | 已有实现；V100 有通用 kernel、暂无专门化 profile | 纳入训练白名单，registered composite 走 Autograd-safe reference | 实现融合 forward kernel，并补 V100/国产芯片 tile autotune |
| Gaussian backward | `ExpBackward/PowBackward/DivBackward/MmBackward` | 叶子算子可由 Autograd 产生 | 先保证 reference 图中的叶子算子均路由 | 原生 Gaussian backward kernel |
| Pair score forward | `bmm/baddbmm/softmax/where` | 已有实现；训练 composite 可微、native epilogue 仍推理专用 | 纳入白名单；训练不误用 inference-only Triton | 融合 score+softmax+context forward，并实现训练 backward |
| Pair score backward | `BmmBackward/SoftmaxBackward/WhereBackward` | 部分已有 | 通过 registered composite 让 Autograd 展开并验证梯度 | 原生 pair epilogue backward |
| LayerNorm | `native_layer_norm`、`native_layer_norm_backward` | 源码实现存在，manifest 名称是 `layer_norm` | 使用正确 manifest ID 加入白名单 | 完成所有 shape/dtype/动态 shape 契约 |
| 向量归一化 | `linalg_vector_norm` | 源码实现存在，manifest 名称是 `vector_norm` | 用 alias 映射纳入白名单 | 补全高阶梯度与边界 dtype |
| MCM 损失 | `cosine_embedding_loss` | 无直接 FlagGems 注册 | 改为等价 masked cosine decomposition，并做逐项数值对照 | 如 workload 需要再实现专用 loss kernel |
| PDR 损失 | `mse_loss/mse_loss_backward` | 已有实现 | 纳入白名单，同时避免无效 mask 物化，并做逐项数值对照 | 评估融合 reduction |
| Mask/index | `index/nonzero/index_put/index_select` | 多数已有实现 | 保留必要输出索引，损失本体采用 mask 加权 | 动态稀疏 gather/scatter kernel |
| AdamW 更新 | `_foreach_*` | 当前没有可用的 FlagGems Python 路由 | 使用 `foreach=False,fused=False` 的单张量 AdamW；审计实际事件 | FlagOS fused/multi-tensor AdamW，并比较吞吐/显存 |
| AdamW 标量更新 | `addcdiv_`, `addcmul_`, `lerp_`, `sqrt_` | 部分已有实现 | 纳入训练白名单并验证 | 统一 optimizer ABI |
| `abs` 反向符号 | `sgn` | 原分支只有 `sgn_`，功能性 `sgn` 缺失 | 新增可读的 out-of-place `sgn` Triton kernel，并加入 manifest | 补齐复数/低精度/高阶梯度契约 |
| Dropout/RNG | `native_dropout`, generator | 有实现但本轮默认 dropout=0 | 单独 seed/dropout 测试，不作为主结果 | 统一 FlagOS/PyTorch generator |
| AMP | unscale、overflow、loss scale | 尚未形成 SToFM 训练闭环 | 本轮 FP32，明确未完成 | FP16/BF16 + GradScaler 全链路 |
| 动态 shape | shape guard/cache | 仅推理候选有约束 | 主训练固定 shape，额外运行 N=8/12 | 动态编译缓存与退化策略 |
| 数据预处理 | Scanpy、Rapids、Geneformer、Leiden | 不属于当前 FlagGems 训练图 | 合成数据绕过，明确边界 | 独立数据/图构建流水线优化 |
| 分布式 | DDP/FSDP/FlagCX | 本轮单卡未启用 | 不纳入本轮验收 | 多卡通信、梯度同步、容错 |
| Torch-FL | PrivateUse1 plugin | 当前环境未安装，版本不兼容 | 本轮采用 FlagGems ATen 路由 | 单独建立 Python 3.12/PyTorch 2.11 环境 |

## 训练路由的严格规则

1. `strict=True` 必须打开 profile。
2. 计算型 profile 事件若不在 FlagGems 白名单或明确 composite/decomposition 表中，
   训练失败并输出事件名称。
3. `view/reshape/transpose/as_strided/empty` 等纯 metadata 事件不计为缺失计算 kernel，
   但仍写入 profile。
4. `audit` 模式允许回退，但 `fallback_report.json` 必须记录算子、原因和影响。
5. 只有 `fallback_compute_ops=[]` 的运行才可标为 `PASS`；其余只能标为 `PASS_WITH_LIMITS`。

## 当前已知工程限制

- FlagGems 全量 `use_gems()` 在 V100 首次训练会编译大量无关算子，探索过程超过数分钟；
  因此训练必须使用 profile 驱动的显式白名单。
- 当前 SToFM 两个自定义算子在训练阶段是 Autograd-safe reference，不是原生融合反向
  kernel；这保证正确性，但不代表训练性能最优。
- V100 运行的是通用 FlagGems Triton 路径，架构专门化状态为
  `has_specialization=false`；需单独补充 V100 配置和 Nsight 证据后，才能讨论 kernel
  级最终优化量。
- 首轮只验证 FP32。AMP、foreach optimizer 和 Torch-FL PrivateUse1 是独立后续任务。
