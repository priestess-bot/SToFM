## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-30T08:42:28.059464+00:00
- Verification Status: UNVERIFIED
- Version Label: stofm_flagos_training_v1

# SToFM FlagOS 假数据训练报告

## 结论

- 状态：**passed**
- 模式：FlagGems ATen training（strict=True）
- 步数：10
- 总损失：2.239271 -> 1.965290
- 初始参数 SHA-256：`ac2e49b953d7af62e1b8f6854acdb514bdfaa21340d2c1075e4de0e493bf4cdf`
- 最终参数 SHA-256：`b713d253b0d8b2fd624c83329775c3c4297d522c50eef17f66fe2fe2d476f31c`
- V100 架构状态：generic FlagGems path; no architecture-specific profile
- CUDA kernel 事件：725
- FlagGems 函数族：42
- 计算算子 FlagGems kernel 覆盖：45/45
- 原生 kernel fallback：无

## 训练算子缺口状态

- 未批准计算型 fallback：无
- `cosine_embedding_loss`：已改为等价的基础算子归约。
- AdamW foreach：本轮关闭，使用单张量更新；multi-tensor kernel 留作后续优化。
- AMP/GradScaler：本轮未纳入，FP32 训练通过后单独补齐。

## 每步

| Step | Total loss | MCM | PDR | Max grad | Step ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2.239271 | 0.930582 | 1.308688 | 9.7651 | 7038.821 |
| 1 | 2.137816 | 0.829352 | 1.308464 | 7.9018 | 223.470 |
| 2 | 2.086954 | 0.778717 | 1.308236 | 5.8698 | 210.969 |
| 3 | 2.058322 | 0.750314 | 1.308008 | 4.182 | 209.351 |
| 4 | 2.038826 | 0.731052 | 1.307773 | 3.1061 | 208.957 |
| 5 | 2.022963 | 0.715429 | 1.307534 | 2.4359 | 246.273 |
| 6 | 2.007265 | 0.699975 | 1.307289 | 1.9627 | 234.284 |
| 7 | 1.992470 | 0.685430 | 1.307039 | 1.6098 | 224.764 |
| 8 | 1.978986 | 0.672204 | 1.306782 | 1.4491 | 224.238 |
| 9 | 1.965290 | 0.658768 | 1.306523 | 1.7122 | 224.681 |

## 产物

- checkpoint：`checkpoint-step-010.pt`
- `run.json`：运行状态、版本和逐步指标
- `operator_inventory.json`：训练图算子与 fallback 审计
- `training_profile.json` / `training_trace.json`：可在 Chrome tracing/Perfetto 打开的 profile
- `flaggems_ops.log`：FlagGems 实际注册函数的调试日志
