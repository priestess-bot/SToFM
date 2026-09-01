# SToFM PHASE 2 V100 训练性能结果

## 三路主结论

| 路线 | 完整训练步 median | forward | backward | optimizer | 峰值显存 |
|---|---:|---:|---:|---:|---:|
| 纯 PyTorch 原始算子 + 单张量 AdamW | 82.0716 ms | 24.5576 ms | 56.1224 ms | 1.3926 ms | 4406.6 MiB |
| 纯 PyTorch 原始算子 + CUDA fused AdamW | 80.2381 ms | 24.5617 ms | 55.4583 ms | 0.1997 ms | 4406.7 MiB |
| FlagOS Stage B：自研 GEMM/BMM + Gaussian 融合反向 + 多张量 AdamW | 71.0057 ms | 17.6845 ms | 53.2521 ms | 0.0727 ms | 2853.7 MiB |

## 优化归因

- `v100_self_hosted_vs_torch_fused`：1.130x，95% bootstrap CI [1.119x, 1.144x]。
- `torch_fused_vs_torch_scalar`：1.023x，95% bootstrap CI [1.010x, 1.081x]。

## 正确性

严格第一步对照：通过。

所有原始 CUDA event 样本、第一步梯度/参数状态、profile trace、运行命令和环境信息均保存在本目录。
