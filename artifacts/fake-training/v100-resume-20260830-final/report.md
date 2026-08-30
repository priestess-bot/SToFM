## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-30T09:08:01.860554+00:00
- Verification Status: UNVERIFIED
- Version Label: stofm_flagos_training_v1

# SToFM FlagOS 假数据训练报告

## 结论

- 状态：**passed**
- 模式：FlagGems ATen training（strict=True）
- 步数：1
- 总损失：1.951187 -> 1.951187
- 初始参数 SHA-256：`fb2c14dd9654e585d1f7005812ff31114b8e68841e5e04f973ba1bdaddd44a9c`
- 最终参数 SHA-256：`8d12fb9ef8b0db6c7906cf8e4d2231195b1c481bd702da5e94883440891315b3`
- V100 架构状态：generic FlagGems path; no architecture-specific profile
- CUDA kernel 事件：776
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
| 10 | 1.951187 | 0.644928 | 1.306259 | 2.1309 | 7129.633 |

## 产物

- checkpoint：`checkpoint-step-011.pt`
- `run.json`：运行状态、版本和逐步指标
- `operator_inventory.json`：训练图算子与 fallback 审计
- `training_profile.json` / `training_trace.json`：可在 Chrome tracing/Perfetto 打开的 profile
- `flaggems_ops.log`：FlagGems 实际注册函数的调试日志
