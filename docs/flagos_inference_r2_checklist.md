# FlagOS Inference R2 Checklist

This file is the single coordination record for the FlagGems and SToFM R2
branches. An item changes to `[x]` only when its code, exact validation command,
and evidence location are recorded here. `[!]` means an external dependency is
required and is not a completed item.

## 0. Repository and Environment

- [x] R2-0 Create `FlagGems:r2/stofm-flagos-inference` from
  `03bf364ede763d573d5c30124d554283a209ab85` and
  `SToFM:r2/flagos-inference` from `2354d5799347867578793752e8c2dd93ae6587b7`.
  Evidence: local branch heads created on 2026-08-15; R1 `integration/*`
  branches remain unchanged.
- [x] R2-1 Add immutable Torch, stock-FlagOS, and optimized-FlagOS environment
  manifests using the same PyTorch/CUDA versions.
  Evidence: `requirements/flagos-r2-v100.txt`, `flagos-r2-stock.txt`, and
  `flagos-r2-optimized.txt` pin Python 3.11 / PyTorch 2.6.0+cu124 / CUDA 12.4
  / Triton 3.2.0; `tests/test_r2_provenance.py` passed (2 tests, 2026-08-15).
- [x] R2-2 Add exact stock and optimized FlagGems locks, package provenance,
  and benchmark environment capture.
  Evidence: `deps/flagos-stock.lock.json` pins frozen stock
  `03bf364ede763d573d5c30124d554283a209ab85`; the optimized lock and install
  requirement pin pushed FlagGems `a9a96bbcc3d685482c656343e0759b7b4a5c38bc`.
  `tests/test_r2_provenance.py` passed in the R2 environment on 2026-08-15;
  `benchmarks/r2_benchmark_common.py::runtime_capture()` records Python,
  package inventory, CUDA runtime, driver, GPU, Torch backend controls, and
  relevant environment variables with every worker result.

## 1. Real FlagOS Inference Modes

- [x] API-0 Add `flagos_mode={torch,stock,optimized}` to SToFM configuration
  and extraction CLI while preserving default Torch behavior.
  Evidence: `tests/test_flagos_adapter.py::test_config_preserves_torch_default_and_optimized_backend_selection`
  passed on 2026-08-15 using the R2 V100 environment.
- [x] API-1 Add a scoped `flag_gems.use_gems()` inference context with a
  recorded, tested ATen allowlist and no global persistent registration.
  Evidence: `model/flagos_runtime.py`; the tested scope registered exactly
  `addmm,baddbmm,bmm,softmax` (plus FlagGems' `softmax_out` alias) and
  `current_flagos_runtime_dispatch()` was empty after scope exit.
- [x] API-2 Add versioned FlagGems SToFM/Vision public APIs and dispatch records
  for selected backend, precision, fallback reason, and registered ATen ops.
  Evidence: FlagGems commit `e69ee3aa04a16d84108bd9ca9a41fd9d6c2d94d7`;
  `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../FlagGems-stofm/src \
  ../.venv-flagos-r2/bin/python -m pytest -q tests/test_flagos_adapter.py \
  tests/test_benchmark_aggregation.py` passed 14 tests on 2026-08-15.
- [x] API-3 Prove P1 Torch, F0 frozen-stock FlagOS, and optimized FlagOS retain
  the same SToFM output semantics before timing.
  Evidence: frozen commit `03bf364ede763d573d5c30124d554283a209ab85` was
  installed in `.venv-flagos-stock-r2`; `tests/test_frozen_stock_flagos.py`
  passed on V100 (1 test, 2026-08-15) against canonical P1
  `return_pair_rep=False`. Optimized P1/F0 equivalence is covered by the
  14-test R2 suite recorded under API-2. The frozen package needed the scoped
  official `FLAGGEMS_VENDOR=nvidia` hint because the V100 name lacks the word
  `NVIDIA`; the hint is restored after registration and does not modify stock.

## 2. V100 Operators and Precision Matrix

- [x] V100-0 Build trace/profile coverage for SToFM and Vision hot ATen ops in
  FP32 and FP16; classify each as stock FlagOS, new composite kernel, reference,
  or rejected.
  Evidence: six SToFM and ten Vision clean traces under
  `benchmark-results/r2-v100-profiles-20260815/`; profile scripts classify P1,
  F0, Ffinal, marker Triton, rejected SwiGLU, and retained LayerNorm without
  profiling numerical assertions.
- [x] V100-1 Implement and validate FP32 Gaussian compiler and native candidates.
  Evidence: C1 FP32 p50 10.5686 ms, 2.054x [2.052x, 2.055x] versus frozen F0
  steady at 21.7006 ms; see `r2-v100-fp32-20260815/`.
- [x] V100-2 Implement and validate FP16 Gaussian compiler and native candidates.
  Evidence: C1 FP16 p50 8.1812 ms, 2.352x [2.319x, 2.365x] versus frozen F0
  steady at 19.3490 ms; see `r2-v100-fp16-20260815/`.
- [x] V100-3 Implement and validate FP32 pair-score epilogue native candidate.
  Evidence: C2 FP32 p50 20.6351 ms, 1.052x [1.052x, 1.053x] versus frozen F0
  steady; public dispatch and direct numerical checks selected NVIDIA inference.
- [x] V100-4 Implement and validate FP16 pair-score epilogue native candidate.
  Evidence: C2 FP16 p50 18.9990 ms, 1.018x [1.018x, 1.018x] versus frozen F0
  steady; the low but positive gain is retained only in the composed route.
- [x] V100-5 Implement and validate FP32/FP16 KRONOS marker-token candidate.
  Evidence: independent frozen F0 is 0.2785 ms and Triton is 0.2512 ms /
  1.077x [1.052x, 1.108x] FP32; frozen F0 is 0.2714 ms and Triton is
  0.2459 ms / 1.076x [1.049x, 1.095x] FP16. Both precisions have 90 raw
  samples per stage, matching reference hashes, and clean kernel traces.
- [x] V100-6 Re-evaluate SwiGLU and residual-LayerNorm under the R2 stock
  baseline; add a native implementation only if profiling justifies it.
  Evidence: existing SwiGLU is rejected at 0.439x [0.431x, 0.449x] FP32
  and 0.421x [0.403x, 0.439x] FP16 versus independent F0; residual LayerNorm
  remains reference fallback because no verified winner exists.
- [x] V100-7 Run three independent FP32 benchmark processes and aggregate P1,
  F0, each candidate, and Ffinal.
  Evidence: `benchmark-results/r2-v100-fp32-20260815/` and
  `r2-vision-v100-fp32-20260815-stock-complete/`; the Vision suite has three
  stock/optimized worker pairs, 90 raw samples per measured stage, 20-file
  checksum manifest, and 10,000-resample intervals.
- [x] V100-8 Run the equivalent independent FP16 benchmark suite and aggregate
  it separately from FP32.
  Evidence: `benchmark-results/r2-v100-fp16-20260815/` and
  `r2-vision-v100-fp16-20260815-stock-complete/`, with the same three-pair,
  raw-evidence, and checksum rules.
- [x] V100-9 Add an independent frozen-FlagOS F0 worker for every Vision
  boundary, then rerun the FP32/FP16 operator suites so Torch, stock FlagOS,
  and optimized FlagOS have separate raw measurements.
  Evidence: `benchmarks/vision_r2_v100_stock_worker.py` imports no R2 Vision
  API, recreates the portable public boundary, and measures it inside frozen
  scoped `use_gems()`. `run_vision_r2_v100_suite.py` pins distinct source roots
  and rejects cross-environment workload/reference/measurement/runtime drift;
  both `*-stock-complete/` suites are schema v2 with six worker artifacts.

## 3. Target-Device Preparation

- [x] ASC-0 Add FP32/FP16/BF16 lazy Ascend dispatch, capability guards, and
  reference fallback for every public SToFM/Vision API.
  Evidence: FlagGems `a9a96bbcc`; target adapters defer `torch_npu` import,
  require inference mode, a supported dtype, contiguous native inputs, and an
  explicit enable variable before consulting `torch.ops`.
- [x] ASC-1 Add the complete deferred CANN/AscendC source project for Gaussian
  and pair-score epilogue, including host registration, tiling metadata,
  dtype dispatch, CMake presets, and deployment instructions.
  Evidence: FlagGems `a9a96bbcc`,
  `experimental_ops/vendor/ascendc_stofm`; the project exposes separate
  FP32/FP16/BF16 kernel symbols and requires an actual `ASCEND_NPU_ARCH` for a
  target build rather than guessing one from the product name.
- [x] ASC-2 Pass offline Python, AST, schema, CMake, and source-structure gates
  without importing `torch_npu` or requiring CANN.
  Evidence: `PYTHONPATH=src ../.venv-flagos-r2/bin/python \
  tools/check_deferred_native_projects.py` passed on 2026-08-15; this runs
  Python AST checks, host-visible C++ syntax checks, manifest checks, and a
  deferred CMake configure/build. It does not claim a CANN binary.
- [!] ASC-3 On rented Ascend 310 hardware: compile, validate FP32/FP16/BF16
  capability matrix, establish P1/F0, and measure promoted candidates.
- [x] MTT-0 Add FP32/FP16/BF16 lazy MUSA dispatch and native Triton/extension
  candidates for Gaussian and pair-score epilogue.
  Evidence: FlagGems `a9a96bbcc`; the adapter selects a lazy Triton candidate
  or a `PrivateUse1` extension only after MUSA inference capability checks.
- [x] MTT-1 Add MUSA extension registration, CMake integration, tile metadata,
  and safe fallback without import-time `torch_musa` dependency.
  Evidence: `experimental_ops/vendor/musa_stofm` provides extension schemas,
  `.mu` Gaussian/pair kernels, manifests, and an SDK `musa_add_library` build
  path using `mcc`; non-target environments remain on the reference path.
- [x] MTT-2 Pass offline Python, AST, schema, CMake, and source-structure gates.
  Evidence: the same `check_deferred_native_projects.py` run passed the MUSA
  source/manifest/CMake gate without importing `torch_musa` or an MUSA SDK.
- [!] MTT-3 On rented MTT S4000 hardware: compile, validate FP32/FP16/BF16
  capability matrix, establish P1/F0, and measure promoted candidates.

## 4. Verification and Evidence

- [x] TEST-0 Add unit tests for every precision, shape bucket, mask, padding,
  pair-state/weights return contract, non-contiguous fallback, and gradients.
  Evidence: FlagGems `test_stofm_experimental.py` and
  `test_vision_experimental.py` exercise FP32/FP16/BF16 paths, dynamic shapes,
  masks, padding/CLS, gradients, and fallback contracts; selected R2 run passed
  26 tests with target static gates on 2026-08-15.
- [x] TEST-1 Add full SToFM P1/F0/optimized consistency tests and dispatch tests.
  Evidence: optimized SToFM R2 suite passed 28 tests on 2026-08-15, including
  cross-environment Vision aggregation, stock wrapper CPU semantics, scoped
  dispatch cleanup, FP16 optimized inference, profile classification, and
  checksum validation. The frozen environment additionally passed 3 tests for
  P1 equivalence and the no-R2-API stock Vision worker.
- [x] TEST-2 Add CPU-only target static validation for all public APIs and
  deferred native-project metadata.
  Evidence: FlagGems `tests/test_deferred_native_projects.py` plus
  `tests/test_stofm_experimental.py` and `tests/test_vision_experimental.py`;
  `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src ../.venv-flagos-r2/bin/python -m
  pytest -q tests/test_stofm_experimental.py tests/test_vision_experimental.py
  tests/test_deferred_native_projects.py tests/test_target_runtime_harness.py`
  passed 26 tests on 2026-08-15.
- [x] REPORT-0 Publish raw benchmark files, checksums, bootstrap confidence
  intervals, rejected candidates, and separate compiler/kernel/lifecycle gains.
  Evidence: `docs/flagos_inference_r2_report.md` and five authoritative R2
  evidence trees; `benchmarks/write_r2_checksums.py --verify` passed for 20,
  20, 20, 20, and 48 files respectively.
- [x] REPORT-1 Publish the R2 operator coverage matrix and final promotion
  decision. Do not reuse R1 speedups as R2 results.
  Evidence: `docs/flagos_inference_r2_report.md` contains the framework and
  architecture matrix, V100 promotion/rejection decisions, and explicit
  CANN/MUSA no-hardware limitations. The companion
  `docs/flagos_inference_r2_report.html` adds per-operator Torch/F0/optimized
  tables and inline three-baseline p50 visualizations; every code/evidence link
  targets the pushed SToFM or FlagGems fork branches.
- [x] REPORT-2 Rebuild the HTML as a human-readable technical-review report:
  separate end-to-end stages from isolated operator boundaries, surface the
  three baselines and promotion decisions first, and retain full evidence,
  source, and target-device limitations in the same standalone file.
  Evidence: docs/flagos_inference_r2_report.html now has an executive
  conclusion, named P1/F0/R2 baseline rail, precision switcher, per-operator
  FP32/FP16 three-baseline rows, code ownership, target maturity, validation
  chain, and rental-device gate in one readable report. Chromium QA at
  1440x1000 and 390x844 verified title/content, six navigation links, no
  horizontal overflow or console warnings, and working FP32/FP16 tab state.
  All report links remain on the pushed SToFM/FlagGems fork branches.
- [x] GIT-0 Commit and push FlagGems first; advance the SToFM optimized lock only
  after the matching FlagGems SHA is tested and pushed.
  Evidence: FlagGems `r2/stofm-flagos-inference` is pushed at
  `a9a96bbcc3d685482c656343e0759b7b4a5c38bc`; SToFM tested that exact SHA,
  advanced its optimized lock, and pushed the raw-evidence commit
  `56a307af4f5bc7e9a79acf9ef486c2202ec4b2c3` on `r2/flagos-inference`.

## Update Rule

Every completed atomic task updates this file in the same commit with a concise
evidence line. Device-runtime items remain `[!]` until a target-device run is
saved with the exact runtime, driver, command, raw samples, and Git SHAs.
