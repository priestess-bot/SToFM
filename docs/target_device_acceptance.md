# Deferred Target-Device Acceptance

The Ascend 310 and MTT S4000 adapters are deliberately limited to a
correctness-first tiled implementation until rental hardware is available.  A
passing static check is not a device performance result.

## Pre-Rental Gate

Run this from the FlagGems fork on any host:

```bash
python tools/check_stofm_target_backends.py --output target-static.json
```

It parses both adapters, verifies their target metadata, confirms that neither
has an import-time `torch_npu` or `torch_musa` dependency, and records the
runtime checks that remain deferred.

## On-Device Test Order

1. Install the vendor PyTorch extension and use the matching optional
   dependency group declared by FlagGems.
2. Record driver/runtime versions, device model, memory capacity, precision
   support, and exact SToFM/FlagGems commits.
3. Run Gaussian forward and gradients against dense FP32 reference for
   `N={33,65,256,1050}`; include zero-distance masks.
4. Run attention forward, pair state, attention weights, and `pair_bias`
   gradients with unmasked and padded inputs.
5. Run full SToFM last-hidden-state and pair-distance-recovery paths.  The
   latter must retain `pair_rep`; embedding extraction may set it to false.
6. Measure B0, B1, O1, O2, O3, and O4 with the same event-timing protocol as
   `benchmarks/stofm_v100.py`, but write a separate target report.
7. Only after correctness and baseline capture, profile tile size and test a
   vendor fused Gaussian or pair-attention kernel.

## Required Result Artifacts

Store `result.json`, per-sample CSV, p20/p50/p80/p95/mean statistics, peak
allocated memory, compiler/runtime logs, and failed-case details.  Do not
compare an Ascend or MTT speedup directly with V100; compare optimized and
baseline paths within the same target environment.

## Promotion Rule

Promote a target fused backend only when it meets all FP32 correctness gates,
has a documented mixed-precision tolerance if enabled, and beats the target's
own B1 end-to-end p50 after warm-up.  Otherwise retain the tested tiled
reference adapter and report the result as correct but not accelerated.
