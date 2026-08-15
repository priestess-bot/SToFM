# V100 算子优化报告（最终）

## 结论

本轮在 Tesla V100-SXM2-16GB 上选定的 SToFM 推理路径是 O5：

```text
flagos_backend=flaggems
flagos_attention_backend=inherit
  Gaussian pair bias: CUDA Inductor O1
  pair-state attention: NVIDIA Triton score/mask/softmax epilogue O2n
  final pair state: caller explicitly requests return_pair_rep=False
```

最终默认配置已实际调度这条组合，而不只是报告中的显式 `nvidia` 实验路径。
在三次独立进程、每次 30 个计时样本的规范工作负载下，O5 的端到端 p50-of-p50
为 `8.8105 ms`，B0 为 `23.6024 ms`，即 `2.679x`；相对 B1 为 `2.630x`。
O5 相对其直接比较对象 O4 的样本级速度比为 `1.2317x`，10,000 次确定性
bootstrap 95% CI 为 `[1.2307, 1.2324]`。峰值 allocated memory 从 B0 的
`4434.95 MiB` 降至 `2960.96 MiB`，节省 `1473.99 MiB`（`33.24%`）。

这是 FP32、V100、推理模式和 `B=1,N=1050` 的结论。它不是训练吞吐、BF16/FP16、
不同 batch 或 Ascend/MTT 的性能声明。

## 实现清单与决策

| ID | 实际实现 | 语义/回退策略 | V100 结果 | 决策 |
| --- | --- | --- | --- | --- |
| B1 | SToFM `return_pair_rep=False` 生命周期消除 | 只删除最后一个未被调用方读取的 pair state；层间和训练路径仍保留 | B0/B1 `1.019x` | 保留 |
| O1 | FlagGems `torch.compile` Gaussian pair-bias 图 | CUDA 自动选 Inductor；其他路径是 tile reference | 17.5860 -> 5.8339 ms，`3.014x` | 默认 |
| O1n | Triton RBF + 双投影 Gaussian kernel | 只在 FP32、连续布局、无梯度、受限 hidden/head 时运行；其余回退 | 相对 O1 `0.543x` | 正确但拒绝默认 |
| O2 | 直接 pair-state reference (`baddbmm`/mask/softmax/`bmm`) | 保留 pair state 和 weights 语义 | 相对 B0 attention `0.986x` | 仅比较基线 |
| O2n | Triton score/mask/optional-pair/row-softmax epilogue | 仅 inference FP32、无 dropout、连续 mask；训练/布局不支持时回退 reference | 0.9181 -> 0.6603 ms，`1.391x` | 默认 CUDA 推理 |
| O3 | O1 + B1，attention 保持 Torch 原路径 | 端到端比较对象 | 相对 B1 `2.031x` | 比较基线 |
| O4 | O1 + B1 + direct pair reference | pair API 的端到端比较对象 | 相对 B1 `2.133x` | 比较基线 |
| O5 | O1 + B1 + O2n | `flaggems`/`inherit` 在 CUDA 推理时选此路径 | 10.8552 -> 8.8105 ms 相对 O4，`1.232x` | 选定 |

O2n 不是完整 FlashAttention 替换：QK 与 PV GEMM 仍交给 cuBLAS，Triton 只融合
`score + mask + optional pair state + row softmax`。这是有意的边界，因为 SToFM
必须保留下一层所需的 pair state，普通 SDPA/FlashAttention 接口并不提供该值。

## 测量设计和原始证据

| 项目 | 固定条件 |
| --- | --- |
| 设备 | Tesla V100-SXM2-16GB，compute capability 7.0，driver 550.144.03 |
| 软件 | PyTorch 2.5.1+cu121，CUDA 12.1，Triton 3.1.0 |
| 代码 | SToFM `ee64e363a06cbc2cdd42ffa540cf0fdcf1f29944`；FlagGems `dde373fe33c71e5819584685781182b0ad2cb144` |
| 模型 | FP32、`B=1,N=1050,L=4,D=256,H=8,K=128`、dropout 0、固定 seed 42 |
| 计时 | `torch.inference_mode()`；CUDA events；10 warm-up；30 samples/process；5 calls/sample；编译不计入延迟 |
| 重复 | 3 个独立进程；每个测量 stage 90 原始样本；O5 共 90 样本 |

三份未经汇总的结果分别保存在
[`run 1`](../benchmark-results/v100-o5-final-20260815-run1/)、
[`run 2`](../benchmark-results/v100-o5-final-20260815-run2/) 和
[`run 3`](../benchmark-results/v100-o5-final-20260815-run3/)；每份都有
`result.json`、`samples.csv`、`report.md` 和 `report.html`。聚合结果
[`three_run_summary.json`](../benchmark-results/v100-o5-final-20260815-run1/three_run_summary.json)
验证了所有运行的工作负载、硬件/runtime、stage contract 和两个 Git SHA 完全一致，
并记录了输入 JSON/CSV 的 SHA-256。

## 三运行结果

下表的 p50 是三次进程 p50 的中位数；区间是三个进程 p50 的 min--max，而不是
把 V100 和其他架构混合后的数字。所有 speedup 都以同一行指定的直接基线计算。

| Stage | p50-of-p50 ms | 三运行 p50 范围 ms | 直接基线速度比 | 峰值 allocated MiB | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| B0 Gaussian | 17.5860 | 17.5847--17.5892 | 1.000x | 4432.89 | 原始公式 |
| O1 Gaussian | 5.8339 | 5.8316--5.8548 | 3.014x vs B0 | 2809.78 | 接受 |
| O1n Gaussian Triton | 10.7400 | 10.7394--10.7858 | 0.543x vs O1 | 2843.42 | 拒绝默认 |
| B0 attention | 0.9020 | 0.8991--0.9040 | 1.000x | 2887.25 | 原始路径 |
| O2 pair reference | 0.9181 | 0.9135--0.9199 | 0.983x vs B0 | 2888.92 | 不推广 |
| O2n Triton epilogue | 0.6603 | 0.6563--0.6663 | 1.391x vs O2 | 2888.92 | 接受 |
| B0 end-to-end | 23.6024 | 23.5998--23.6115 | 1.000x | 4434.95 | 原始模型 |
| B1 end-to-end | 23.1676 | 23.1668--23.1728 | 1.019x vs B0 | 4434.95 | 生命周期基线 |
| O3 end-to-end | 11.4010 | 11.4009--11.4161 | 2.031x vs B1 | 2968.34 | 比较路径 |
| O4 end-to-end | 10.8552 | 10.8529--10.8656 | 2.134x vs B1 | 2960.96 | 比较路径 |
| O5 end-to-end | 8.8105 | 8.8077--8.8360 | 1.232x vs O4 | 2960.96 | 默认 CUDA 推理 |

`O1n`、O2 和已有 `skip_layer_norm` 的拒绝结论同样是结果：它们没有被包装成
“已优化”。在 `N=1050,D=256` 的 residual + LayerNorm 微基准中，PyTorch p50 是
`0.0351 ms`，FlagGems `skip_layer_norm` 是 `0.1053 ms`，且后者没有适用于此路径的
已验证 backward，因此保留 reference。

## 数值和回退门禁

1. Gaussian：zero-distance mask、输出及全部可学习参数梯度与 dense reference
   对比；native 的正向容差为 `rtol=3e-4, atol=3e-5`。
2. Pair attention：无 padding/尾部 padding、context、pair state、weights 和
   `pair_bias` gradient 对比；非连续 tensor、训练/梯度开启、dropout 都必须进入
   reference。
3. SToFM bridge：`flaggems` 默认 CUDA inference 的 Gaussian dispatch 为
   `inductor`，attention dispatch 为 `nvidia`；该默认输出与 Torch 模型一致。
4. 端到端：B1、O3、O4、O5 的 `last_hidden_state` 在计时前全部通过相同容差。
5. 静态目标：Ascend/MTT 的 SToFM 与 vision 四个 adapter 均经 AST 检查，且没有
   import-time `torch_npu`/`torch_musa` 依赖。此项不等价于设备运行通过。

完整命令和最终通过数在 [`implementation_checklist.md`](implementation_checklist.md)
中记录；可复用的跨进程聚合器是
[`aggregate_operator_runs.py`](../benchmarks/aggregate_operator_runs.py)。

## Uni2 和 KRONOS 的 V100 算子证据

这部分是算子级实验，不是完整 Uni2/KRONOS 模型加速声明。工作负载为 FP32
`B=1,M=32,T=256,D=384`、25% marker padding；SwiGLU 为
`[1,264,8192]` packed input，30 samples，5 calls/sample。

| 算子 | 实现 | p50 ms | 相对 reference | 决策 |
| --- | --- | ---: | ---: | --- |
| KRONOS marker token assembly | Triton marker gather + patch/token/position add + flatten | 0.2135 p50-of-p50 | pooled raw samples `1.349x`，95% CI `1.327--1.382x`（reference p50-of-p50 0.2802） | 接受为 NVIDIA inference kernel |
| Uni2 packed SwiGLU | 复用现有 CUDA primitive | 0.0913 p50-of-p50 | pooled raw samples `0.591x`（reference p50-of-p50 0.0556） | 拒绝默认 |
| ViT residual + LayerNorm | reference | 0.0526 p50-of-p50 | 候选已拒绝 | 保留 reference |

三份原始 vision 结果位于
[`run 1`](../benchmark-results/vision-v100-final-20260815-run1/)、
[`run 2`](../benchmark-results/vision-v100-final-20260815-run2/) 和
[`run 3`](../benchmark-results/vision-v100-final-20260815-run3/)，汇总和文件校验和位于
[`three_run_summary.json`](../benchmark-results/vision-v100-final-20260815-run1/three_run_summary.json)。
它覆盖 marker padding、可变 token 数、marker permutation、非连续回退和 gradient 安全回退；
尚缺真实 Uni2/KRONOS 权重与端到端输入管线，因此不能外推模型整体加速。

## 不在本结论内的工作

* Ascend 310 和 MTT S4000 尚未租赁，只有可导入、可语法检查的正确性优先 adapter；
  没有任何性能数字。
* O2n、O1n 和 vision marker kernel 当前只有 FP32 inference native 实现；训练会
  走 reference，直到独立 backward 通过梯度矩阵和目标设备验证。
* 动态形状、batch 大于 1、混合精度、真正 Uni2/KRONOS 端到端数据加载，以及多 GPU
  干扰下的吞吐，均保留给后续 shape sweep 和租赁设备报告。
