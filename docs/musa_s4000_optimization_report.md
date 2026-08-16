# FlagOS SToFM MTT S4000 算子优化报告

## 结论

本轮在单卡 MTT S4000、MUSA 3.1.0 上完成了两个 SToFM 推理算子的真实实现、注册、模型接入和五轮正式测量：

- `flagos_stofm::gaussian_pair_bias`
- `flagos_stofm::pair_score_epilogue`

N=1050、四层、FP32 的完整 SToFM 推理由纯 PyTorch 的 **33.2088 ms** 降至优化后 FlagOS 的 **16.3370 ms**，加速 **2.032x**，延迟下降 **50.79%**。配对分层 bootstrap 的 95% 区间为 **[2.028x, 2.036x]**，五轮 p50 变异系数为 **0.28%**，最大绝对模型误差为 **1.19e-6**。

## 基线定义

报告不使用缩写代号，四种状态分别为：

| 状态 | 含义 | 是否有 S4000 时间 |
| --- | --- | --- |
| 纯 PyTorch | SToFM 原始 PyTorch 表达式，不导入 FlagOS | 有 |
| 固定版本的上游 FlagOS | 固定提交 `03bf364e`，只请求其既有 ATen/Triton 能力 | 无；MUSA 3.1 上报错 `0 active drivers`，不替换版本、不伪造时间 |
| 初始 FlagOS MUSA 注册实现 | 本轮两个新注册算子的第一版串行 MUSA kernel | 有；用作“新增算子但尚未优化”的真实基线 |
| 优化后 FlagOS MUSA 后端 | 协作式 MUSA 原语、vendor GEMM/SDPA 与 Python 调度组合 | 有 |

“固定版本的上游 FlagOS”与“初始 FlagOS MUSA 注册实现”不是同一个概念：前者没有这两个 SToFM 专用算子且在目标运行时不可执行；后者证明新增算子已真实运行，并提供优化前后的可比起点。

## 固定环境与源码

| 项目 | 固定值 |
| --- | --- |
| 设备 | 单卡 MTT S4000，47.91 GiB，架构 `mp_22` |
| Python | 3.10 |
| PyTorch | 2.2.0 |
| torch_musa | 1.3.0+81caf0a |
| MUSA Toolkit | 3.1.0 |
| SToFM 实现提交 | [`e2c6de9ec902bee5d67a4861b4ef6716a58e0cc4`](https://github.com/priestess-bot/SToFM/commit/e2c6de9ec902bee5d67a4861b4ef6716a58e0cc4) |
| FlagGems 实现提交 | [`832c46df4073215d416406181484f9b44594aff2`](https://github.com/priestess-bot/FlagGems/commit/832c46df4073215d416406181484f9b44594aff2) |
| 初始动态库 SHA-256 | `006d5e256060342f1fb188f91e623fed0baaa0746928d710a43de63efb1cf590` |
| 优化动态库 SHA-256 | `a7beac88e8d4b7b999b3620b13234ded848aee22d1a479f66ce9f9744a8e2313` |

两个 fork 通过精确 Git SHA 管理依赖，而不是依赖相邻目录名称。SToFM 的 [`deps/flagos-musa-s4000.lock.json`](https://github.com/priestess-bot/SToFM/blob/r2/musa-s4000/deps/flagos-musa-s4000.lock.json) 和 [`requirements/flagos-musa-s4000.txt`](https://github.com/priestess-bot/SToFM/blob/r2/musa-s4000/requirements/flagos-musa-s4000.txt) 固定 FlagGems 提交；本地协同开发仍可使用 editable checkout。

## 实现内容

### Gaussian pair-bias

初始注册算子把距离仿射、RBF、两层投影、ReLU、head 排布和零距离 mask 放在一个 MUSA 扩展边界中。优化后将其拆为 FP32 累积的 MUSA RBF/layout 原语与 vendor `F.linear` 调度，消除了 C++ PrivateUse1 内嵌 ATen 调用之间的主机调度空隙；FP16/BF16 输入先以 FP32 计算，再转换回输入精度。

- [MUSA kernel](https://github.com/priestess-bot/FlagGems/blob/832c46df4073215d416406181484f9b44594aff2/src/flag_gems/experimental_ops/vendor/musa_stofm/src/stofm_gaussian_pair_bias.mu)
- [PrivateUse1 注册](https://github.com/priestess-bot/FlagGems/blob/832c46df4073215d416406181484f9b44594aff2/src/flag_gems/experimental_ops/vendor/musa_stofm/src/stofm_musa_registration.cpp)
- [MUSA 后端调度](https://github.com/priestess-bot/FlagGems/blob/832c46df4073215d416406181484f9b44594aff2/src/flag_gems/experimental_ops/stofm_backends/mthreads.py)

### Pair-attention score/softmax/context

新注册算子覆盖 score、padding、softmax、context 和可选 pair/weights 输出。优化后对无 padding、无需可选输出的推理路径调用 MUSA vendor SDPA；有 padding或需要可选输出时使用 vendor `baddbmm`/`softmax`/`bmm` 组合；不满足约束时保留自定义 MUSA kernel 或显式失败/参考回退。

- [MUSA kernel](https://github.com/priestess-bot/FlagGems/blob/832c46df4073215d416406181484f9b44594aff2/src/flag_gems/experimental_ops/vendor/musa_stofm/src/stofm_pair_score_epilogue.mu)
- [SToFM 调用桥](https://github.com/priestess-bot/SToFM/blob/e2c6de9ec902bee5d67a4861b4ef6716a58e0cc4/model/flagos_backend.py)
- [模型中的 mask 快速路径](https://github.com/priestess-bot/SToFM/blob/e2c6de9ec902bee5d67a4861b4ef6716a58e0cc4/model/se2transformer.py)

### MUSA 轻量运行时

目标镜像的通用 Triton 没有 MUSA driver。FlagGems 增加显式 opt-in 的 SToFM 轻量入口，只暴露版本化自定义算子 API，不声称通用 ATen 接管可用；请求不可用能力时关闭式失败。

- [轻量 API](https://github.com/priestess-bot/FlagGems/blob/832c46df4073215d416406181484f9b44594aff2/src/flag_gems/_musa_stofm_api.py)
- [包入口隔离](https://github.com/priestess-bot/FlagGems/blob/832c46df4073215d416406181484f9b44594aff2/src/flag_gems/__init__.py)
- [目标构建入口](https://github.com/priestess-bot/FlagGems/blob/832c46df4073215d416406181484f9b44594aff2/tools/build_musa_stofm_extension.py)

## 正式测试协议

1. 从两个 GitHub fork 拉取精确提交，确认干净 HEAD 后重新执行目标正确性测试。
2. 所有 warmup、正确性检查和计时调用均处于 `torch.inference_mode()`；该约束另有无硬件回归测试。
3. 使用 `torch.musa.Event` 记录设备时间，每个样本前后同步；同时保存 `time.perf_counter` 主机交叉计时。
4. 五个独立 Python 进程轮次；三类任务按轮次循环换位，降低热状态和顺序偏差。
5. 优化后矩阵为 5 次预热、10 个设备样本；端到端为 5 次预热、15 个设备样本。
6. 每个候选先通过同输入正确性门，再进入计时；编译时间不计入稳态推理。
7. 以轮次为 block、轮次内原始样本为第二层执行 10,000 次配对分层 bootstrap。
8. 独立验证器重新读取 15 个结果 JSON，校验源码和动态库 SHA、样本向量、p50、聚合值、不可用基线与敏感信息。

实现见 [正式运行器](https://github.com/priestess-bot/SToFM/blob/e2c6de9ec902bee5d67a4861b4ef6716a58e0cc4/benchmarks/run_musa_s4000_formal_suite.py)、[算子矩阵 worker](https://github.com/priestess-bot/SToFM/blob/e2c6de9ec902bee5d67a4861b4ef6716a58e0cc4/benchmarks/stofm_musa_s4000_operator_matrix.py) 和 [模型 worker](https://github.com/priestess-bot/SToFM/blob/e2c6de9ec902bee5d67a4861b4ef6716a58e0cc4/benchmarks/stofm_musa_s4000_worker.py)。

## N=1050、FP32 三方算子对比

| 算子 | 纯 PyTorch p50 | 初始 FlagOS MUSA p50 | 优化后 FlagOS MUSA p50 | 优化后相对 PyTorch | 优化后相对初始实现 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gaussian pair-bias | 22.9559 ms | 2652.8751 ms | 9.5671 ms | 2.399x，95% CI [2.376x, 2.407x] | 277.292x |
| Pair-attention score/softmax/context | 0.7431 ms | 9.6459 ms | 0.6624 ms | 1.123x，95% CI [1.118x, 1.128x] | 14.561x |

初始注册实现证明了新算子的实际执行，但串行工作分配严重低效。优化后的 Gaussian 不仅消除了初始实现的数量级回归，还比 PyTorch 低 58.32%；优化后 pair-attention 比 PyTorch 低 10.97%。

## 完整 SToFM 端到端结果

工作负载：batch=1、N=1050、4 层、embedding=256、8 heads、FP32，不返回最终 pair representation。

| 推理路线 | 五轮 p50 中位数 | 五轮最小/最大 | 相对纯 PyTorch | 95% bootstrap 区间 | p50 变异系数 | 最大绝对误差 |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| 纯 PyTorch | 33.2088 ms | 33.1969 / 33.3399 | 1.000x | - | 0.18% | 0 |
| 仅 FlagOS Gaussian 优化 | 19.9878 ms | 19.9763 / 20.1350 | 1.661x | [1.654x, 1.663x] | 0.33% | 1.19e-6 |
| 仅 FlagOS pair-attention 优化 | 29.5774 ms | 29.5443 / 29.6871 | 1.123x | [1.122x, 1.124x] | 0.19% | 9.54e-7 |
| 两个优化后 FlagOS 算子 | 16.3370 ms | 16.2937 / 16.4112 | **2.032x** | **[2.028x, 2.036x]** | 0.28% | 1.19e-6 |
| 两个算子 + 通用 FlagOS ATen 接管 | 不可用 | - | - | - | - | - |

通用 FlagOS ATen 接管没有 MUSA Triton driver，因此保留为不可用行，不将纯 PyTorch 或其他版本的时间代填。

## 优化后形状与精度矩阵

### Gaussian pair-bias

| 精度 | N | PyTorch p50 | FlagOS p50 | 加速 | 95% CI | 最大绝对误差 |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| FP32 | 256 | 1.9917 | 0.8818 | 2.262x | [2.238x, 2.274x] | 8.94e-8 |
| FP32 | 512 | 6.0762 | 2.6769 | 2.270x | [2.260x, 2.275x] | 2.24e-8 |
| FP32 | 1050 | 22.9559 | 9.5671 | 2.399x | [2.376x, 2.407x] | 2.05e-8 |
| FP32 | 2048 | 85.2320 | 35.6355 | 2.391x | [2.384x, 2.400x] | 3.35e-8 |
| FP16 | 256 | 2.1358 | 1.0460 | 2.043x | [2.036x, 2.052x] | 7.63e-6 |
| FP16 | 512 | 6.2366 | 2.8532 | 2.187x | [2.184x, 2.199x] | 6.10e-5 |
| FP16 | 1050 | 23.1871 | 9.8134 | 2.362x | [2.352x, 2.368x] | 1.53e-5 |
| FP16 | 2048 | 85.5704 | 36.0683 | 2.370x | [2.362x, 2.384x] | 1.53e-5 |
| BF16 | 256 | 2.1357 | 1.0464 | 2.043x | [2.035x, 2.056x] | 0 |
| BF16 | 512 | 6.2322 | 2.8549 | 2.182x | [2.177x, 2.189x] | 7.63e-6 |
| BF16 | 1050 | 23.1921 | 9.8208 | 2.358x | [2.349x, 2.364x] | 6.10e-5 |
| BF16 | 2048 | 85.6379 | 36.0255 | 2.380x | [2.365x, 2.389x] | 1.53e-5 |

Gaussian 在全部 12 个组合上均有统计显著收益，最低 2.043x，最高 2.399x。

### Pair-attention score/softmax/context

| 精度 | N | PyTorch p50 | FlagOS p50 | 加速 | 95% CI | 最大绝对误差 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| FP32 | 256 | 0.2184 | 0.2293 | 0.953x | [0.939x, 1.053x] | 0 | 不晋升 |
| FP32 | 512 | 0.3025 | 0.3000 | 1.006x | [0.995x, 1.069x] | 0 | 区间跨 1 |
| FP32 | 1050 | 0.7431 | 0.6624 | 1.123x | [1.118x, 1.128x] | 0 | 晋升 |
| FP32 | 2048 | 1.7475 | 1.3872 | 1.262x | [1.257x, 1.264x] | 0 | 晋升 |
| FP16 | 256 | 0.1983 | 0.2015 | 0.966x | [0.878x, 1.095x] | 1.95e-3 | 不晋升 |
| FP16 | 512 | 0.2351 | 0.2234 | 1.033x | [0.973x, 1.202x] | 2.93e-3 | 区间跨 1 |
| FP16 | 1050 | 0.5437 | 0.2819 | 1.916x | [1.859x, 2.106x] | 1.95e-3 | 晋升 |
| FP16 | 2048 | 1.0225 | 0.5720 | 1.796x | [1.758x, 1.870x] | 2.44e-3 | 晋升 |
| BF16 | 256 | 0.2002 | 0.2077 | 0.957x | [0.900x, 1.086x] | 3.74e-3 | 不晋升 |
| BF16 | 512 | 0.2336 | 0.2287 | 0.990x | [0.970x, 1.121x] | 2.24e-3 | 区间跨 1 |
| BF16 | 1050 | 0.5461 | 0.2832 | 1.928x | [1.840x, 2.093x] | 3.42e-3 | 晋升 |
| BF16 | 2048 | 1.0189 | 0.5706 | 1.769x | [1.742x, 1.867x] | 2.08e-3 | 晋升 |

Pair-attention 的稳定性能边界是 N≥1050。N=256 出现 3%–5% 回归，N=512 的区间跨 1；部署策略应在小尺寸保留 PyTorch 路线，在 N≥1050 才启用该快速路径。

## 优化过程与每步增量

| 阶段 | Gaussian N=1050 FP32 | Pair-attention N=1050 FP32 | 优化内容 | 证据等级 |
| --- | ---: | ---: | --- | --- |
| 初始串行 MUSA 注册实现 | 2667.10 ms 预检；2652.88 ms 正式 | 9.616 ms 预检；9.646 ms 正式 | 首次补齐 PrivateUse1 注册与正确语义 | 正式基线 |
| 协作式 MUSA kernel | 247.30 ms | 7.405 ms | Gaussian 每 pair 使用 128 线程；pair-score 分层归约与并行 context | 优化过程预检 |
| MUSA 原语 + 内嵌 ATen | 约 24.05 ms | 约 0.75 ms | 拆分 RBF/layout，并复用 vendor GEMM；pair 复用 vendor attention 组件 | profiler 诊断 |
| Python 编排 vendor 算子 | 9.567 ms | 0.662 ms | 避开 PrivateUse1 C++ 内嵌 ATen 的主机调度间隙；无 mask 使用 SDPA | 五轮正式结果 |
| 模型级空 mask 折叠 | 完整组合从 42.7 ms 预检降到 16.27 ms 预检 | 每层避免全 False padding 的通用分支 | encoder 在推理入口只检查一次空 mask | 端到端预检；最终正式 16.337 ms |
| BF16 SDPA 扩展 | Gaussian 保持 FP32 累积 | pair 从 0.600 ms 预检降至 0.290 ms，最终 0.283 ms | 将 BF16 纳入已验证的 vendor SDPA 快速路径 | 五轮正式矩阵 |

初始到最终的正式改善为：Gaussian **277.29x**，pair-attention **14.56x**。约 24 ms 的早期“native”数字曾因计时区域不在 `torch.inference_mode()` 而实际走 Autograd 参考调度，已经作废；表中的约 24.05 ms 是修正协议后、通过 profiler 验证的后续原语组合阶段，不与作废数据混用。

## 严格正确性与测试结论

| 测试层 | 结果 | 覆盖内容 |
| --- | ---: | --- |
| FlagGems S4000 目标测试 | 25 passed | FP32/FP16/BF16、直接注册算子、公共后端、padding、可选输出、非连续回退、错误 mask、inference-only |
| SToFM S4000 模型测试 | 2 passed | 无 padding 与真实 padding；`last_hidden_state`、`pair_rep`、每层 dispatch provenance |
| SToFM 本地完整测试 | 37 passed, 2 skipped | V100/CPU 集成、provenance、基准聚合、目标测试在无 MUSA 主机跳过 |
| FlagGems 本地相关测试 | 23 passed, 25 skipped | 公共 ABI、SToFM 参考语义、延迟项目静态门；MUSA 专用项在 V100 跳过 |
| 基准协议回归 | 3 passed | inference mode 包围 warmup/采样、配对 bootstrap 确定性、三种基线字段不混淆 |
| 正式证据独立验证 | passed | 5 轮、15 个 JSON、130 个算子测量行、20 个模型测量行、全部 SHA 与样本重算 |

最大误差：完整 FP32 模型 1.19e-6；Gaussian FP32 8.94e-8 以内、FP16/BF16 6.10e-5 以内；pair FP32 为 0、FP16 2.93e-3 以内、BF16 3.74e-3 以内。BF16 pair 使用相同 BF16 输入值的 CPU FP32 表达式作为 oracle。

## 原始证据

- [聚合 JSON](../artifacts/musa_s4000/formal-20260816/summary.json)
- [逐算子 CSV](../artifacts/musa_s4000/formal-20260816/operator_summary.csv)
- [独立验证结果](../artifacts/musa_s4000/formal-20260816/verification.json)
- [五轮原始目录](../artifacts/musa_s4000/formal-20260816/)
- [实现 checklist](musa_s4000_checklist.md)

原始目录保留每轮 JSON、设备/主机样本 CSV、worker 日志和 manifest。冻结上游 FlagOS 的不可用 traceback 也单独保留，以证明没有将失败基线替换为别的版本。

## 剩余边界与后续工作

- 为 N<1050 的 pair-attention 加尺寸门控；当前正式结果不支持在小尺寸默认启用。
- 通用 FlagOS ATen/Triton 接管在当前 MUSA 3.1 镜像不可用，不能报告组合增量。
- 当前是推理专用实现；训练和 Autograd 继续走参考路径或关闭式失败。
- Ascend 310 仍只完成高概率正确的 CANN/AscendC 源码与离线语法检查，等待租赁设备做同规格真实编译、正确性和五轮性能测试。
