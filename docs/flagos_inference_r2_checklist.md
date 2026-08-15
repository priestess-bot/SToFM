# FlagOS 推理优化实施清单

本文件是 FlagGems 与 SToFM 推理优化分支的唯一协作记录。只有在这里记录
代码、精确验证命令和证据位置后，任务才可从 `[ ]` 变为 `[x]`；`[!]` 表示
仍依赖外部设备或环境，因此不是已完成任务。

为保证非实验参与者也能阅读，正文使用以下完整名称：

- **纯 PyTorch**：不接管 ATen 算子的规范参考实现。
- **固定版本的未优化 FlagOS**：固定到指定 FlagGems 提交、只启用已存在后端能力的基线。
- **优化后 FlagOS**：在上述基线之上接入经正确性与性能验证的实现。

分支名、文件名、环境变量、命令和代码符号是可复制的技术标识，按原样保留；
说明性文字不使用内部实验阶段代号。

## 0. 仓库与环境

- [x] 仓库-0 从
  `FlagGems:r2/stofm-flagos-inference` 的基准提交
  `03bf364ede763d573d5c30124d554283a209ab85` 创建开发分支，并从
  `SToFM:r2/flagos-inference` 的基准提交
  `2354d5799347867578793752e8c2dd93ae6587b7` 创建配套分支。
  证据：2026-08-15 已创建本地分支头；既有 `integration/*` 分支未改动。
- [x] 环境-1 使用相同的 PyTorch/CUDA 版本，建立纯 PyTorch、固定版本未优化 FlagOS、
  优化后 FlagOS 三套不可变环境清单。
  证据：`requirements/flagos-r2-v100.txt`、`flagos-r2-stock.txt` 与
  `flagos-r2-optimized.txt` 固定 Python 3.11 / PyTorch 2.6.0+cu124 /
  CUDA 12.4 / Triton 3.2.0；`tests/test_r2_provenance.py` 于 2026-08-15
  通过（2 项测试）。
- [x] 环境-2 记录固定版本与优化后 FlagGems 的精确锁定信息、包来源和基准环境。
  证据：`deps/flagos-stock.lock.json` 固定未优化提交
  `03bf364ede763d573d5c30124d554283a209ab85`；优化锁定文件和安装依赖固定已推送的
  FlagGems `a9a96bbcc3d685482c656343e0759b7b4a5c38bc`。
  `tests/test_r2_provenance.py` 在该环境中于 2026-08-15 通过；
  `benchmarks/r2_benchmark_common.py::runtime_capture()` 会为每个计时进程记录
  Python、包清单、CUDA 运行时、驱动、GPU、Torch 后端控制项与相关环境变量。

## 1. 可复现的 FlagOS 推理模式

- [x] 接口-0 在 SToFM 配置与特征提取命令行中加入
  `flagos_mode={torch,stock,optimized}`，同时保持默认的纯 PyTorch 行为。
  证据：`tests/test_flagos_adapter.py::test_config_preserves_torch_default_and_optimized_backend_selection`
  在 V100 环境于 2026-08-15 通过。
- [x] 接口-1 加入受限的 `flag_gems.use_gems()` 推理上下文，记录并测试 ATen
  算子白名单，且不做全局永久注册。
  证据：`model/flagos_runtime.py`；测试范围仅注册
  `addmm,baddbmm,bmm,softmax`（以及 FlagGems 的 `softmax_out` 别名），作用域退出后
  `current_flagos_runtime_dispatch()` 为空。
- [x] 接口-2 为选定后端、精度、回退原因和已注册 ATen 算子建立带版本的
  FlagGems SToFM/Vision 公共接口与调度记录。
  证据：FlagGems 提交 `e69ee3aa04a16d84108bd9ca9a41fd9d6c2d94d7`；
  `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../FlagGems-stofm/src \
  ../.venv-flagos-r2/bin/python -m pytest -q tests/test_flagos_adapter.py \
  tests/test_benchmark_aggregation.py` 于 2026-08-15 通过 14 项测试。
- [x] 接口-3 在计时前证明纯 PyTorch、固定版本未优化 FlagOS、优化后 FlagOS
  保持相同的 SToFM 输出语义。
  证据：固定提交 `03bf364ede763d573d5c30124d554283a209ab85` 已安装到
  `.venv-flagos-stock-r2`；`tests/test_frozen_stock_flagos.py` 在 V100 上
  于 2026-08-15 通过（1 项测试），对照规范的
  `return_pair_rep=False`。优化后实现与前两条基线的一致性由接口-2 记录的
  14 项测试覆盖。固定包需要在受限注册期间传入官方
  `FLAGGEMS_VENDOR=nvidia` 提示，因为 V100 名称不含 `NVIDIA`；注册后该变量会恢复，
  不修改未优化包本身。

## 2. V100 算子与精度矩阵

- [x] V100-0 为 SToFM 与 Vision 热点 ATen 算子建立 FP32/FP16 跟踪和性能分析覆盖，
  将每个算子归类为未优化 FlagOS、复合新内核、参考实现或拒绝路线。
  证据：`benchmark-results/r2-v100-profiles-20260815/` 下有 6 份 SToFM 和
  10 份 Vision 干净跟踪；分析脚本区分三条基线、KRONOS marker 的 Triton 实现、
  被拒绝的 SwiGLU 与保留回退的 LayerNorm，且不将数值断言纳入性能分析。
- [x] V100-1 实现并验证 FP32 Gaussian 编译路线与原生候选实现。
  证据：Gaussian 编译路线的中位延迟（p50）为 10.5686 ms，相对固定版本未优化
  FlagOS 的 21.7006 ms 为 2.054x [2.052x, 2.055x]；见
  `r2-v100-fp32-20260815/`。
- [x] V100-2 实现并验证 FP16 Gaussian 编译路线与原生候选实现。
  证据：Gaussian 编译路线的中位延迟（p50）为 8.1812 ms，相对固定版本未优化
  FlagOS 的 19.3490 ms 为 2.352x [2.319x, 2.365x]；见
  `r2-v100-fp16-20260815/`。
- [x] V100-3 实现并验证 FP32 pair-score 收尾阶段的原生候选实现。
  证据：pair-score 原生路线的中位延迟为 20.6351 ms，相对固定版本未优化 FlagOS
  为 1.052x [1.052x, 1.053x]；公共调度和直接数值检查选择 NVIDIA 推理实现。
- [x] V100-4 实现并验证 FP16 pair-score 收尾阶段的原生候选实现。
  证据：pair-score 原生路线的中位延迟为 18.9990 ms，相对固定版本未优化 FlagOS
  为 1.018x [1.018x, 1.018x]；增益较小但为正，仅保留在组合路线中。
- [x] V100-5 实现并验证 FP32/FP16 KRONOS marker-token 候选实现。
  证据：固定版本未优化 FlagOS 的独立结果为 FP32 0.2785 ms，Triton 为
  0.2512 ms / 1.077x [1.052x, 1.108x]；FP16 分别为 0.2714 ms 与
  0.2459 ms / 1.076x [1.049x, 1.095x]。两种精度每阶段均有 90 个原始样本、
  一致的参考输出哈希和干净的内核跟踪。
- [x] V100-6 在固定版本未优化 FlagOS 基线上重新评估 SwiGLU 与残差 LayerNorm；
  仅在性能分析证明有价值时才增加原生实现。
  证据：已有 SwiGLU 在 FP32 为 0.439x [0.431x, 0.449x]，FP16 为
  0.421x [0.403x, 0.439x]，均慢于固定版本未优化 FlagOS，因此拒绝；
  残差 LayerNorm 因没有已验证的优胜实现而保持参考回退。
- [x] V100-7 分别运行 3 个独立 FP32 计时进程，汇总纯 PyTorch、固定版本未优化
  FlagOS、各候选实现和优化后组合。
  证据：`benchmark-results/r2-v100-fp32-20260815/` 和
  `r2-vision-v100-fp32-20260815-stock-complete/`；Vision 套件具有 3 对
  未优化/优化计时进程、每个已测阶段 90 个原始样本、20 文件校验和清单，及
  10,000 次重采样区间。
- [x] V100-8 运行与 FP32 等价且独立汇总的 FP16 基准套件。
  证据：`benchmark-results/r2-v100-fp16-20260815/` 和
  `r2-vision-v100-fp16-20260815-stock-complete/`，采用相同的三对进程、
  原始证据与校验和规则。
- [x] V100-9 为每个 Vision 边界增加独立的固定版本未优化 FlagOS 计时进程，
  然后重新运行 FP32/FP16 算子套件，使纯 PyTorch、未优化 FlagOS、优化后 FlagOS
  均有独立原始测量。
  证据：`benchmarks/vision_r2_v100_stock_worker.py` 不导入优化后的 Vision 接口，
  重建可移植公共边界，并在固定的 `use_gems()` 作用域中测量。
  `run_vision_r2_v100_suite.py` 固定不同源码根目录，并拒绝跨环境的工作负载、
  参考输出、测量或运行时漂移；两个 `*-stock-complete/` 套件均为 schema v2，
  各有 6 个计时进程产物。

## 3. 目标设备准备

- [x] Ascend-0 为所有公开 SToFM/Vision 接口增加 FP32/FP16/BF16 的按需 Ascend
  后端选择、能力保护和参考回退。
  证据：FlagGems `a9a96bbcc`；目标适配器延迟导入 `torch_npu`，要求推理模式、
  支持的数据类型、连续的原生输入和显式启用变量后才查询 `torch.ops`。
- [x] Ascend-1 为 Gaussian 与 pair-score 收尾阶段加入完整的延迟 CANN/AscendC
  源码项目，包括主机注册、分块元数据、数据类型调度、CMake 预设和部署说明。
  证据：FlagGems `a9a96bbcc`，
  `experimental_ops/vendor/ascendc_stofm`；项目导出独立的 FP32/FP16/BF16
  内核符号，目标构建要求真实的 `ASCEND_NPU_ARCH`，而非从产品名猜测架构。
- [x] Ascend-2 无需导入 `torch_npu` 或安装 CANN，即通过离线 Python、AST、schema、
  CMake 和源码结构检查。
  证据：`PYTHONPATH=src ../.venv-flagos-r2/bin/python \
  tools/check_deferred_native_projects.py` 于 2026-08-15 通过；它执行 Python AST、
  主机可见 C++ 语法、清单和延迟 CMake 配置/构建检查，不宣称已产出 CANN 二进制文件。
- [!] Ascend-3 在租用的 Ascend 310 上：编译、验证 FP32/FP16/BF16 能力矩阵，
  建立三条基线，并测量被纳入的候选实现。
- [x] MTT-0 为 Gaussian 和 pair-score 收尾阶段增加 FP32/FP16/BF16 的按需 MUSA
  后端选择，以及原生 Triton/扩展候选实现。
  证据：FlagGems `a9a96bbcc`；适配器只会在 MUSA 推理能力检查后，才选择延迟
  Triton 候选或 `PrivateUse1` 扩展。
- [x] MTT-1 加入 MUSA 扩展注册、CMake 集成、分块元数据和安全回退，且导入时
  不依赖 `torch_musa`。
  证据：`experimental_ops/vendor/musa_stofm` 提供扩展 schema、`.mu` Gaussian/
  pair 内核、清单以及使用 `mcc` 的 SDK `musa_add_library` 构建路径；非目标环境
  保持参考路线。
- [x] MTT-2 在无 `torch_musa` 或 MUSA SDK 的环境中，通过离线 Python、AST、schema、
  CMake 和源码结构检查。
  证据：同一 `check_deferred_native_projects.py` 检查在不导入 `torch_musa` 或
  MUSA SDK 的前提下通过 MUSA 源码/清单/CMake 检查。
- [!] MTT-3 在租用的 MTT S4000 上：编译、验证 FP32/FP16/BF16 能力矩阵，
  建立三条基线，并测量被纳入的候选实现。

## 4. 验证与证据

- [x] 测试-0 为每种精度、形状桶、mask、padding、pair-state/weights 返回约定、
  非连续输入回退和梯度加入单元测试。
  证据：FlagGems `test_stofm_experimental.py` 和
  `test_vision_experimental.py` 覆盖 FP32/FP16/BF16 路径、动态形状、mask、
  padding/CLS、梯度和回退约定；选定的本轮运行于 2026-08-15 通过 26 项测试及
  目标静态检查。
- [x] 测试-1 加入完整的 SToFM 三基线一致性测试与调度测试。
  证据：优化后 SToFM 套件于 2026-08-15 通过 28 项测试，包括跨环境 Vision 汇总、
  未优化包装器的 CPU 语义、受限调度清理、FP16 优化推理、性能分析分类和校验和验证。
  固定版本环境额外通过 3 项测试，覆盖纯 PyTorch 一致性及不依赖优化 Vision 接口的
  未优化 Vision 计时进程。
- [x] 测试-2 为所有公开接口和延迟原生项目元数据增加仅 CPU 的目标静态验证。
  证据：FlagGems `tests/test_deferred_native_projects.py`、
  `tests/test_stofm_experimental.py` 与 `tests/test_vision_experimental.py`；
  `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src ../.venv-flagos-r2/bin/python -m
  pytest -q tests/test_stofm_experimental.py tests/test_vision_experimental.py
  tests/test_deferred_native_projects.py tests/test_target_runtime_harness.py`
  于 2026-08-15 通过 26 项测试。
- [x] 报告-0 发布原始基准文件、校验和、自助法置信区间、被拒绝候选，以及
  编译器/内核/生命周期增益的独立拆分。
  证据：`docs/flagos_inference_r2_report.md` 与 5 个权威证据目录；
  `benchmarks/write_r2_checksums.py --verify` 分别对 20、20、20、20 和 48 个文件通过。
- [x] 报告-1 发布算子覆盖矩阵和最终纳入结论，且不将过往轮次的加速结果
  复用为本轮结果。
  证据：`docs/flagos_inference_r2_report.md` 包含框架与架构矩阵、V100 的
  纳入/拒绝结论和 CANN/MUSA 无硬件限制；配套
  `docs/flagos_inference_r2_report.html` 增加逐算子的纯 PyTorch/未优化 FlagOS/
  优化后 FlagOS 表格和内联三基线 p50 可视化，所有代码/证据链接均指向已推送的
  SToFM 或 FlagGems 分支。
- [x] 报告-2 将 HTML 重建为人类可读的技术评审报告：区分端到端阶段与隔离算子边界，
  优先展示三条基线和纳入结论，并在同一独立文件中保留完整证据、源码责任和目标设备限制。
  证据：`docs/flagos_inference_r2_report.html` 具有执行结论、命名清楚的三基线栏、
  精度切换器、逐算子 FP32/FP16 三基线行、代码归属、目标成熟度、验证链和租用设备门槛。
  Chromium 在 1440x1000 和 390x844 下确认标题/内容、6 个导航链接、无横向溢出或
  控制台警告，并确认 FP32/FP16 切换正常。所有报告链接仍指向已推送的
  SToFM/FlagGems 分支。
- [x] 报告-3 从展示层移除内部实验阶段缩写和英文状态标签，改用完整的读者语言，
  例如“纯 PyTorch”“未优化 FlagOS”“优化组合”，以及明确的纳入、保留或拒绝结论。
  证据：浏览器在 1440x1000 和 390x844 下确认没有可见内部缩写、没有本地代码链接、
  没有横向溢出或控制台警告，且精度切换支持点击和键盘操作。精度、p50、置信区间、
  预热和目标设备描述均已使用解释性中文。
- [x] 版本控制-0 先提交并推送 FlagGems；仅在匹配的 FlagGems SHA 已测试并推送后，
  才推进 SToFM 的优化锁定。
  证据：FlagGems `r2/stofm-flagos-inference` 已推送至
  `a9a96bbcc3d685482c656343e0759b7b4a5c38bc`；SToFM 已测试该精确 SHA，
  推进优化锁定，并将原始证据提交
  `56a307af4f5bc7e9a79acf9ef486c2202ec4b2c3` 推送至 `r2/flagos-inference`。

## 更新规则

每个完成的原子任务必须在同一提交中更新本文件，并附上一行简洁证据。设备运行时任务
在保存目标设备的精确运行时、驱动、命令、原始样本和 Git SHA 之前，始终保持 `[!]`。
