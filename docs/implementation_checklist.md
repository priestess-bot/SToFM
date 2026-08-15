# FlagOS Operator Optimization Execution Checklist

> Historical R1 checklist. The active completion record is
> `docs/flagos_inference_r2_checklist.md`.

Last updated: 2026-08-15

This is the source of truth for the work requested for SToFM, Uni2, KRONOS,
V100, Ascend 310, and MTT S4000. An item is complete only when its stated
evidence is present. A public API, a routing wrapper, or a static adapter does
not count as a native operator implementation by itself.

Status: `[x]` complete, `[-]` in progress, `[ ]` pending, `[!]` waiting for
external rental hardware.

## 0. Scope and Repository Control

- [x] C0. Define the completion rule: distinguish native kernels, portable
  fallbacks, static target adapters, and measured results.
  Evidence: this checklist and `docs/operator_gap_analysis.md`.
- [x] C1. Maintain independent forks and a versioned public dependency
  boundary.
  Evidence: `priestess-bot/FlagGems:integration/stofm` at
  `dd9abb71aa79a4bbc6428b85cd7eeef5d5b7bb33`; `priestess-bot/SToFM:integration/flagos`.
- [x] C2. Lock the exact FlagGems commit consumed by SToFM.
  Evidence: `deps/flagos.lock.json` and `requirements/flagos-v100.txt`.
- [x] C3. Inventory SToFM, Uni2, KRONOS, PyTorch, and target-backend operator
  gaps without presenting source inspection as device performance.
  Evidence: `docs/operator_gap_analysis.md`.

## 1. V100 SToFM Native Operator Work

- [x] V100-0. Freeze the original SToFM B0/B1 semantics and canonical V100
  baseline (`B=1`, `N=1050`, FP32, four layers).
  Evidence: `benchmark-results/v100-20260815-v100-sxm2-16gb-committed/`.
- [x] V100-1. Implement a real CUDA/Triton Gaussian-pair-bias inference
  kernel or document a measured rejection against the existing Inductor O1.
  Acceptance: exact forward semantics for zero-distance masking; the native
  path is never selected for gradient-enabled calls without a verified backward.
  Evidence: FlagGems `experimental_ops/stofm_backends/nvidia.py`; V100 FP32
  smoke comparison compiled successfully with max absolute error `1.43e-6`.
- [x] V100-2. Add Gaussian shape/layout/mask regression tests and a separate
  native-versus-O1 microbenchmark report.
  Evidence: FlagGems `tests/test_stofm_experimental.py` (7 passed); V100 run
  `benchmark-results/v100-native-gaussian-20260815/`. O1n p50 was
  `10.7886 ms` versus O1 `5.8256 ms` (`0.540x`), so O1n is explicitly
  rejected as the default despite correct output.
- [x] V100-3. Implement a real CUDA/Triton pair-score epilogue that fuses key
  masking, pair-state materialization, and row softmax after the QK GEMM.
  Acceptance: context, pair state, and attention weights match reference;
  gradient-enabled calls retain the verified reference path until backward is
  implemented.
  Evidence: FlagGems `experimental_ops/stofm_backends/nvidia.py`; V100 smoke
  maximum absolute errors were context `2.38e-7`, pair state `0`, and weights
  `5.96e-8`.
- [x] V100-4. Add pair-score epilogue shape/layout/mask tests, including
  non-contiguous inputs and no-pair-output inference.
  Evidence: FlagGems `tests/test_stofm_experimental.py` (10 passed): padded
  and unpadded rows, optional pair output, native inference, and gradient/
  non-contiguous reference fallback.
- [x] V100-5. Extend the V100 benchmark with individual native stages and
  rerun a full end-to-end comparison. Select a default only from measured
  p50 and memory results; record a rejection when a native candidate loses.
  Evidence: `benchmark-results/v100-native-epilogue-20260815/` at SToFM
  `960dda12b98cb2a4fd60c03929ecaa0a6acfc1a7` and FlagGems
  `567297236ddc2db6fee498854fd0e493c6f4ea03`. O2n p50 is `0.6605 ms`
  versus O2 `0.9140 ms` (`1.384x`); O5 p50 is `8.8122 ms` versus O4
  `10.8573 ms` (`1.232x`) at unchanged peak allocation. O5 is selected;
  O1n Gaussian remains rejected.
- [x] V100-6. Profile remaining SToFM LayerNorm/residual/FFN work and record
  whether existing FlagGems fused primitives are sufficient or a new kernel
  is justified. This is a decision task, not permission to count a wrapper as
  a new operator.
  Evidence: V100 post-LayerNorm microbenchmark at `N=1050,D=256` measured
  PyTorch residual+LayerNorm p50 `0.0351 ms` versus FlagGems
  `skip_layer_norm` p50 `0.1053 ms`. The existing primitive is rejected for
  this path; its potential share is too small to justify a new V100 kernel.

## 2. Vision Operator Work for Uni2 and KRONOS

- [x] VIS-0. Add a versioned experimental vision API with narrow, testable
  operator boundaries rather than importing complete `timm` models.
  Evidence: FlagGems `experimental_ops/vision.py`, API version `1`.
- [x] VIS-1. Implement and test marker-aware token assembly for KRONOS:
  marker embedding gather, positional/token addition, optional CLS prepend,
  and padding-safe variable marker counts.
  Evidence: native NVIDIA inference implementation in
  `vision_backends/nvidia.py` plus portable adapters; tests cover padded and
  variable marker counts, optional CLS, and marker permutations.
- [x] VIS-2. Add a ViT residual-LayerNorm inference path that uses an existing
  FlagGems fused primitive only where its semantics and gradient policy match;
  otherwise retain PyTorch reference behavior.
  Evidence: `vit_residual_layer_norm` retains reference behavior because the
  measured V100 `skip_layer_norm` candidate is slower and lacks a verified
  backward; `vit_swiglu` uses the existing CUDA inference primitive.
- [x] VIS-3. Add Uni2/KRONOS operator-level reference tests for dynamic image
  shape metadata, marker permutations, padded batches, and dtype/layout
  behavior.
  Evidence: FlagGems `tests/test_vision_experimental.py` (5 passed), including
  dynamic token count, non-contiguous fallback, gradients, and static adapters.
- [x] VIS-4. Add V100 microbenchmarks for the implemented vision operators.
  Do not claim a full Uni2/KRONOS model speedup without model weights and an
  end-to-end reproducible workload.
  Evidence: `benchmark-results/vision-v100-20260815/` at SToFM
  `5c3662ae55ae59214f3b8ae9899d2cf83c020572` and FlagGems
  `dde373fe33c71e5819584685781182b0ad2cb144`. Marker-token Triton p50 is
  `0.2099 ms` versus reference `0.3058 ms` (`1.456x`) with peak allocation
  `24.0 -> 12.0 MiB`; existing SwiGLU is `0.598x` and is rejected. These are
  operator microbenchmarks, not full Uni2/KRONOS model speedups.

## 3. Ascend 310 Correctness-First Implementation

- [x] ASC-0. Provide static, vendor-import-free adapters for the initial
  SToFM Gaussian and pair-attention APIs.
  Evidence: FlagGems `experimental_ops/stofm_backends/ascend.py` and the AST
  checker.
- [x] ASC-1. Extend static backend contracts for every new SToFM/Vision API
  introduced in sections 1 and 2; use portable reference semantics when CANN
  runtime is absent.
  Evidence: FlagGems `vision_backends/ascend.py`; marker-token assembly,
  residual-LayerNorm, and SwiGLU all use explicit reference contracts.
- [x] ASC-2. Run source syntax/AST/API tests for all Ascend adapters and save
  the deferred device test commands.
  Evidence: expanded `tools/check_stofm_target_backends.py` reports passed
  SToFM and vision Ascend contracts with no `torch_npu` import.
- [!] ASC-3. On rented Ascend 310 hardware: establish B0/B1, run forward and
  gradient matrices, then measure each promoted operator and full model.

## 4. MTT S4000 Correctness-First Implementation

- [x] MTT-0. Provide static, vendor-import-free adapters for the initial
  SToFM Gaussian and pair-attention APIs.
  Evidence: FlagGems `experimental_ops/stofm_backends/mthreads.py` and the
  AST checker.
- [x] MTT-1. Extend static backend contracts for every new SToFM/Vision API
  introduced in sections 1 and 2; use portable reference semantics when MUSA
  runtime is absent.
  Evidence: FlagGems `vision_backends/mthreads.py`; marker-token assembly,
  residual-LayerNorm, and SwiGLU all use explicit reference contracts.
- [x] MTT-2. Run source syntax/AST/API tests for all MTT adapters and save the
  deferred device test commands.
  Evidence: expanded `tools/check_stofm_target_backends.py` reports passed
  SToFM and vision MTT contracts with no `torch_musa` import.
- [!] MTT-3. On rented MTT S4000 hardware: establish B0/B1, run forward and
  gradient matrices, then measure each promoted operator and full model.

## 5. Evidence, Reports, and Promotion

- [x] E0. Run the expanded local correctness suite: forward, gradients,
  zero-distance/mask behavior, non-contiguous layouts, and dispatch fallback.
  Evidence: FlagGems `tests/test_stofm_experimental.py` plus
  `tests/test_vision_experimental.py` (`15 passed`); SToFM
  `tests/test_flagos_adapter.py` (`6 passed`); all four target adapters pass
  `tools/check_stofm_target_backends.py`; related sources pass `compileall`.
- [ ] E1. Publish a V100 optimization report that separates each actual kernel
  from API-only and lifecycle changes, with raw samples for every measured
  stage.
- [ ] E2. Update the architecture gap analysis with implemented/rejected
  operator decisions and no unmeasured performance claims.
- [ ] E3. Commit and push each repository only after its checklist entries,
  tests, locks, and reports agree.

## External-Hardware Completion Rule

`ASC-3` and `MTT-3` cannot truthfully change to `[x]` until the named rental
hardware is available. All other checklist items are local work and must be
completed before pausing for those external prerequisites.
