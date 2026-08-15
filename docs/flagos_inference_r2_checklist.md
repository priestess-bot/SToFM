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
- [ ] R2-2 Add exact stock and optimized FlagGems locks, package provenance,
  and benchmark environment capture.

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

- [ ] V100-0 Build trace/profile coverage for SToFM and Vision hot ATen ops in
  FP32 and FP16; classify each as stock FlagOS, new composite kernel, reference,
  or rejected.
- [ ] V100-1 Implement and validate FP32 Gaussian compiler and native candidates.
- [ ] V100-2 Implement and validate FP16 Gaussian compiler and native candidates.
- [ ] V100-3 Implement and validate FP32 pair-score epilogue native candidate.
- [ ] V100-4 Implement and validate FP16 pair-score epilogue native candidate.
- [ ] V100-5 Implement and validate FP32/FP16 KRONOS marker-token candidate.
- [ ] V100-6 Re-evaluate SwiGLU and residual-LayerNorm under the R2 stock
  baseline; add a native implementation only if profiling justifies it.
- [ ] V100-7 Run three independent FP32 benchmark processes and aggregate P1,
  F0, each candidate, and Ffinal.
- [ ] V100-8 Run the equivalent independent FP16 benchmark suite and aggregate
  it separately from FP32.

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

- [ ] TEST-0 Add unit tests for every precision, shape bucket, mask, padding,
  pair-state/weights return contract, non-contiguous fallback, and gradients.
- [ ] TEST-1 Add full SToFM P1/F0/optimized consistency tests and dispatch tests.
- [x] TEST-2 Add CPU-only target static validation for all public APIs and
  deferred native-project metadata.
  Evidence: FlagGems `tests/test_deferred_native_projects.py` plus
  `tests/test_stofm_experimental.py` and `tests/test_vision_experimental.py`;
  `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src ../.venv-flagos-r2/bin/python -m
  pytest -q tests/test_stofm_experimental.py tests/test_vision_experimental.py
  tests/test_deferred_native_projects.py tests/test_target_runtime_harness.py`
  passed 26 tests on 2026-08-15.
- [ ] REPORT-0 Publish raw benchmark files, checksums, bootstrap confidence
  intervals, rejected candidates, and separate compiler/kernel/lifecycle gains.
- [ ] REPORT-1 Publish the R2 operator coverage matrix and final promotion
  decision. Do not reuse R1 speedups as R2 results.
- [ ] GIT-0 Commit and push FlagGems first; advance the SToFM optimized lock only
  after the matching FlagGems SHA is tested and pushed.

## Update Rule

Every completed atomic task updates this file in the same commit with a concise
evidence line. Device-runtime items remain `[!]` until a target-device run is
saved with the exact runtime, driver, command, raw samples, and Git SHAs.
