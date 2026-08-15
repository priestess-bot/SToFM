# SToFM and Vision Operator Benchmark Protocol

This is the evidence contract for the FlagGems integration. A source-level
adapter, a public API, and a measured native kernel are different deliverables;
the reports must identify which one is being discussed.

## SToFM Stages

| ID | Scope | Implementation | Role |
| --- | --- | --- | --- |
| B0 | Gaussian, attention, end-to-end | Original SToFM expressions; final pair state retained | Semantic/performance baseline |
| B1 | End-to-end | B0 with only the unused final `pair_rep` omitted | Allocation-lifecycle baseline |
| B2 | End-to-end | Generic `use_gems` monkey patch | Explicitly excluded; not a direct-op semantic benchmark |
| O1 | Gaussian | FlagGems public Gaussian API, CUDA Inductor implementation | Accepted compiler specialization |
| O1n | Gaussian | Explicit NVIDIA Triton RBF/projection kernel | Native candidate; measured, but currently rejected on V100 |
| O2 | Attention | Direct FlagGems pair-state reference API | Pair-semantic comparison path |
| O2n | Attention | NVIDIA Triton score/mask/optional-pair/softmax epilogue | Accepted native inference candidate |
| O3 | End-to-end | O1 + B1 + original Torch attention | Isolates Gaussian/lifecycle effect |
| O4 | End-to-end | O1 + B1 + direct pair reference | Direct pair API comparison |
| O5 | End-to-end | O1 + B1 + O2n | Selected V100 CUDA inference default |

O2n intentionally does not replace QK/PV GEMMs: cuBLAS performs the GEMMs and
the Triton kernel owns score masking, optional pair-state materialization, and
row softmax. This preserves SToFM's required next-layer pair state rather than
silently changing the model to an SDPA-only interface.

## Correctness Gates

1. Run `compileall` on all new Python sources and FlagGems's static target
   adapter checker.
2. Compare Gaussian output, zero-distance masking, and every learnable-input
   gradient with the dense reference. The FP32 tolerance is `rtol=3e-4`,
   `atol=3e-5`.
3. Compare pair-attention context, attention weights, next pair state, and
   `pair_bias` gradient for unmasked and padded cases.
4. Exercise unsupported native conditions deliberately: gradient-enabled
   execution, non-contiguous layout, dropout/training, and unsupported target
   device must use the verified reference path rather than error or silently
   use an untested backward.
5. Compare equal-weight SToFM `last_hidden_state` before timing. `pair_rep`
   may be omitted only when `return_pair_rep=False`; all intermediate layers
   and pair-recovery/training paths retain it.
6. For vision APIs, compare marker padding, marker permutation, dynamic token
   count, optional CLS, non-contiguous fallback, and gradients against the
   reference. Do not equate an operator result with a Uni2/KRONOS end-to-end
   result without those models and reproducible inputs.
7. Mark absent dtype, runtime, or hardware coverage as `skipped`/`deferred`,
   never as passed.

## V100 Measurement

The canonical SToFM workload is FP32 inference with `batch=1`, `nodes=1050`,
`layers=4`, `embedding_dim=256`, `heads=8`, `gaussian_hidden_dim=128`, and
`input_dim=256`. Use fixed seed 42, disabled dropout, `torch.inference_mode()`,
CUDA events, 10 warm-up iterations, 30 measured samples, and 5 calls/sample.
Compilation happens during warm-up and is excluded from latency.

Promotion requires three independent processes on the same device/runtime and
workload. `benchmarks/aggregate_operator_runs.py` refuses to aggregate runs
with different Git commits, hardware/runtime, workload, or stage contract. It
records raw-file SHA-256 values, p50 min/median/max, peak memory and a
deterministic 10,000-resample bootstrap CI for every direct baseline/candidate
speedup. The accepted V100 data and commands are documented in
[`v100_operator_optimization_report.md`](v100_operator_optimization_report.md).

Each run directory must contain:

- `result.json`: immutable inputs, versions, commits, raw samples, and
  per-run summary statistics.
- `samples.csv`: one latency per row for later statistical analysis.
- `report.md` and `report.html`: readable per-run summaries.

## Promotion and Rejection

For a new architecture, pre-measurement targets are O1 Gaussian at least
`1.30x` versus local B0 and at least one full-model candidate at least `1.10x`
versus local B1, while maintaining or reducing peak allocated memory. They are
engineering thresholds, not predicted results.

On V100, O5 was selected because it passes all gates and its O5/O4 bootstrap
95% lower bound is above `1.23x`. O1n Gaussian, O2 direct reference, existing
SwiGLU, and residual-LayerNorm candidates remain explicitly rejected where
they lose to their own local references. A rejected candidate stays available
for future architectures but is never enabled by default solely because it is
native code.

## Ascend 310 and MTT S4000 Extension

Create fresh B0/B1 baselines on each rented target; never divide their
milliseconds by the V100 result. First run the complete forward/backward/mask/
layout matrix, then use the same raw artifact format and aggregate three
independent runs. Include `B={1,2}`, `N={7,33,65,256,1050}`, odd dimensions,
zero-distance masks, no padding and tail padding. The target-specific ordering
and failure-capture requirements are in
[`target_device_acceptance.md`](target_device_acceptance.md).
