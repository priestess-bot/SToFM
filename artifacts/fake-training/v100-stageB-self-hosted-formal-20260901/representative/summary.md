# SToFM PHASE 2 V100 训练性能结果

## 三路主结论

| 路线 | 完整训练步 median | forward | backward | optimizer | 峰值显存 |
|---|---:|---:|---:|---:|---:|
| 纯 PyTorch 原始算子 + 单张量 AdamW | 26.1688 ms | 2.4397 ms | 13.9822 ms | 9.3942 ms | 168.2 MiB |
| 纯 PyTorch 原始算子 + CUDA fused AdamW | 16.6236 ms | 2.2820 ms | 13.7953 ms | 0.5478 ms | 168.2 MiB |
| FlagOS Stage B：自研 GEMM/BMM + Gaussian 融合反向 + 多张量 AdamW | 14.4942 ms | 1.6963 ms | 12.1672 ms | 0.5263 ms | 108.2 MiB |

## 优化归因

- `v100_self_hosted_vs_torch_fused`：1.147x，95% bootstrap CI [0.960x, 1.164x]。
- `torch_fused_vs_torch_scalar`：1.574x，95% bootstrap CI [1.427x, 1.632x]。

## 正确性

严格第一步对照：通过。

所有原始 CUDA event 样本、第一步梯度/参数状态、profile trace、运行命令和环境信息均保存在本目录。
