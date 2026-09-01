# SToFM Stage B V100 自研 GEMM 验收

状态：**passed**

| 形状 | Torch fused | FlagOS 自研 | 加速 | P95（Torch → FlagOS） | 显存比 |
|---|---:|---:|---:|---:|---:|
| representative | 16.6236 ms | 14.4942 ms | 1.1469x | 18.1925 → 17.8012 ms | 0.643x |
| production | 80.2381 ms | 71.0057 ms | 1.1300x | 82.3944 → 71.3161 ms | 0.648x |

联合加速：**1.1384x**，95% bootstrap CI [1.0410x, 1.1505x]。

## 门槛

- [x] aggregate_speedup_at_least_1_05
- [x] bootstrap_lower_above_1
- [x] no_shape_slower_than_torch
- [x] peak_memory_at_most_1_25x_torch
- [x] strict_correctness
- [x] profile_dispatch_and_kernel_provenance
- [x] extension_dependency_audit
- [x] measured_library_hash_matches
- [x] five_trials_fifty_samples
