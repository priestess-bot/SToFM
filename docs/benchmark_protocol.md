# SToFM FlagOS 注册算子基准协议

本协议用于验证真实 FlagOS 自定义算子，不把编译器优化当作算子补齐。历史编译器实验可以
保留，但必须用显式 `--suite legacy` 运行，且不能用于注册算子性能结论。

## 运行路线

| 路线 | FlagOS ATen 接管 | SToFM 注册算子 | 用途 |
| --- | --- | --- | --- |
| 纯 PyTorch | 关闭 | 关闭 | 语义与端到端参考 |
| 固定版本的未优化 FlagOS | 开启 | 关闭 | 独立包环境的真实基线 |
| 仅 Gaussian 自定义算子 | 关闭 | Gaussian | 隔离新 Gaussian 收益 |
| 仅 pair-score 自定义算子 | 关闭 | pair-score | 隔离新 pair-score 收益 |
| 两个自定义算子 | 关闭 | 两者 | 只归因于新算子的端到端结果 |
| 两个自定义算子 + ATen 接管 | 开启 | 两者 | 新算子和既有 FlagOS 的组合结果 |

默认运行器使用 `--suite registered_ops`。该套件在 profiler 中要求看到
`flagos_stofm::gaussian_pair_bias`、`flagos_stofm::pair_score_epilogue`，并在仅新算子
阶段确认普通 ATen 接管处于关闭状态。

## 正确性门槛

1. 解析新增 Python，并运行 FlagGems 目标延迟项目的离线静态检查。
2. Gaussian 对比密集参考公式，包括零距离 mask；FP32 使用 `rtol=3e-4, atol=3e-5`。
3. 直接注册算子必须覆盖 CPU、CUDA、Autograd、FP32、FP16、非连续输入安全回退。
4. Pair-score 比较 context、可选 pair state、attention weights、key padding mask 和梯度
   回退。
5. 完整 SToFM 比较 `last_hidden_state`；只有调用方显式设置 `return_pair_rep=False` 时，
   才允许省略最终 pair state。
6. 每个优化阶段先经 profiler 证明所声明的自定义算子事件出现，才允许记录计时数据。

## V100 测量协议

使用独立的固定版与优化版 Python 进程。每种精度运行 3 个独立进程对，固定
`B=1,N=1050,L=4,D=256,FFN=256,H=8,K=128`，禁用 dropout 和 TF32，启用 inference mode。
每阶段使用 10 次预热、30 个 CUDA-event 样本、每样本 5 次调用；因此每阶段保留 90 个原始
样本。10,000 次 bootstrap 对原始样本计算 95% 区间。

每个工作进程把指定 FlagGems checkout 的 `src` 置于导入路径首位，并写入实际导入的包位置。
聚合前必须验证纯 PyTorch 参考哈希、工作负载、精度、套件名称、源码根和提交一致。输出必须
包含 JSON、CSV、工作日志、聚合结果和 SHA-256 manifest。

执行结束后运行：

```bash
python benchmarks/verify_stofm_registered_ops_evidence.py \
  benchmark-results/r3-v100-registered-ops-fp32-20260816 \
  benchmark-results/r3-v100-registered-ops-fp16-20260816
```

## 目标设备协议

Ascend 310 与 MTT S4000 必须重新建立目标自己的纯 PyTorch、固定版本未优化 FlagOS、仅新
算子和组合路线基线。V100 结果不得外推。租赁设备前只能声称 Python/AST/schema/CMake/C++
静态检查通过；必须在目标 SDK 上成功编译、完成 FP32/FP16/BF16 正确性矩阵并保存同等原始
样本后，才可发表任何目标设备性能结论。
