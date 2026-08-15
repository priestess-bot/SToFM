# Ascend 310 and MTT S4000 Acceptance Procedure

The Ascend and MTT code paths are correctness-first adapters, not measured
accelerators. A static pass means only that source can be imported and routed
without an import-time vendor dependency. Do not change an external-hardware
checklist item to complete until this procedure has run on the named device.

## Pre-rental Host Gate

Run these before spending rental time. They deliberately work without
`torch_npu` or `torch_musa` installed.

```bash
cd /home/ymm/ym/xl/20260815-first-operator/FlagGems-stofm
python -m compileall -q tools/validate_target_operator_runtime.py tools/check_stofm_target_backends.py
python -m pytest -q tests/test_target_runtime_harness.py tests/test_stofm_experimental.py tests/test_vision_experimental.py
python tools/check_stofm_target_backends.py --output /tmp/target-static.json

cd /home/ymm/ym/xl/20260815-first-operator/SToFM-flagos
python -m compileall -q model/flagos_backend.py benchmarks/stofm_target.py benchmarks/vision_target.py benchmarks/aggregate_operator_runs.py
python -m pytest -q tests/test_flagos_adapter.py tests/test_benchmark_aggregation.py
```

The FlagGems checker must report five passes: SToFM/vision for Ascend,
SToFM/vision for MTT, and the device runtime validation harness. The CPU
fallback test exercises the validation harness's tensor/gradient logic, but
does not substitute for target-device execution.

## On-device Correctness Order

1. Install the matching vendor PyTorch extension and runtime, then record the
   device model, driver/runtime versions, memory capacity, supported dtypes,
   exact SToFM/FlagGems commits, and all non-idle processes.
2. Confirm the vendor extension has registered `torch.npu` or `torch.musa`.
   Run the target validator first at `N=7`, then `33`, `65`, `256`, and `1050`.
   If a shape cannot fit, preserve the command, OOM message, and memory state
   as a failed/limited case rather than dropping it from the report.

```bash
cd /home/ymm/ym/xl/20260815-first-operator/FlagGems-stofm
python tools/validate_target_operator_runtime.py \
  --device npu --backend ascend --nodes 33 \
  --output target-results/ascend310-validation-n33.json

python tools/validate_target_operator_runtime.py \
  --device musa --backend mthreads --nodes 33 \
  --output target-results/mtt-s4000-validation-n33.json
```

The validator compares target adapter and Torch reference output plus
gradients for Gaussian pair bias, pair attention with padding, KRONOS marker
token assembly with padding/CLS, ViT residual-LayerNorm, and SwiGLU. It uses
FP32 `rtol=3e-4`, `atol=3e-5` and records maximum absolute output/gradient
errors. Repeat the command for each required shape and both target platforms.

3. Run SToFM `last_hidden_state` and pair-distance-recovery paths. The latter
must retain `pair_rep`; embedding extraction alone may set it to false.
4. Only after all numerical gates pass, establish B0/B1 and candidate timing.
The host-clock timer synchronizes the target before/after each sample, which
is conservative and portable. If the vendor provides a stable event timer,
add it as a separately documented timing backend rather than silently mixing
timer types.

## Three-run Performance Capture

Use a new result directory for every process. The target benchmark is a
correctness-gated B0/B1/O1/O2/O5 harness; current target adapters may perform
like the reference until a vendor fused kernel exists, which is an expected
and useful baseline outcome.

```bash
cd /home/ymm/ym/xl/20260815-first-operator/SToFM-flagos

python benchmarks/stofm_target.py --device npu --backend ascend \
  --output-dir benchmark-results/ascend310-stofm-run1
python benchmarks/stofm_target.py --device npu --backend ascend \
  --output-dir benchmark-results/ascend310-stofm-run2
python benchmarks/stofm_target.py --device npu --backend ascend \
  --output-dir benchmark-results/ascend310-stofm-run3

python benchmarks/aggregate_operator_runs.py \
  benchmark-results/ascend310-stofm-run1 \
  benchmark-results/ascend310-stofm-run2 \
  benchmark-results/ascend310-stofm-run3 \
  --output benchmark-results/ascend310-stofm-run1/three_run_summary.json
```

For MTT replace `npu/ascend/ascend310` with `musa/mthreads/mtt-s4000`. The
same pattern applies to `benchmarks/vision_target.py`, which captures marker
assembly, SwiGLU, and residual-LayerNorm separately from a full Uni2/KRONOS
model. Each run emits `result.json`, `samples.csv`, `report.md`, and
`report.html`; the aggregator checks compatibility, records source-file
checksums and produces deterministic bootstrap CIs.

## Target-specific Promotion Rule

Do not compare target milliseconds directly with V100. Compare each candidate
to its target-local B0/B1 under the same shape, dtype, runtime, thermal state,
and process isolation.

| Candidate family | Required correctness before promotion | Engineering target, not result |
| --- | --- | --- |
| SToFM Gaussian | all forward/gradient/mask/layout cases; no peak-memory regression | p50 >= `1.30x` vs target B0 Gaussian |
| Pair-score epilogue | context, pair state, weights, padding, gradients; preserve fallback | at least one full SToFM candidate >= `1.10x` vs target B1 |
| KRONOS marker assembly | marker identity/permutation/padding/CLS semantics | operator or block p50 >= `1.05x` vs local reference |
| Uni2 residual/MLP/attention | reference forward/gradient matrix and dynamic shape buckets | operator or block p50 >= `1.05x` vs local reference |

For each promoted kernel, retain a reference fallback for unsupported dtype,
layout, dropout/training, or missing vendor feature. Include failed variants,
compiler logs, full raw samples, memory availability/absence, and exact Git
SHAs in the final target report. A correct but slower vendor path remains a
reference adapter and is not enabled by default.
