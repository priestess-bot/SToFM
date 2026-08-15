# SToFM Operator Benchmark Protocol

> Historical R1 protocol. R2 uses P1/F0/C1/C2/Ffinal, frozen-stock process
> isolation, raw checksum manifests, and 10,000-resample bootstrap intervals.
> See `docs/flagos_inference_r2_report.md` for the executable R2 evidence.

This protocol is the evidence contract for the FlagGems integration.  It
separates a measured result from an optimization target so benchmark reports
cannot turn an unverified estimate into a performance claim.

## Scope and Baselines

| ID | Scope | Implementation | Purpose |
| --- | --- | --- | --- |
| B0 | Gaussian / attention / end-to-end | Unmodified SToFM tensor expressions | Semantic and performance baseline |
| B1 | End-to-end | B0 with final unused `pair_rep` omitted | Allocation-elimination baseline |
| B2 | End-to-end | Generic `use_gems` monkey patch | Explicitly excluded; it does not preserve the direct-op contract |
| O1 | Gaussian bias | FlagGems `stofm_gaussian_pair_bias` | Inductor dense fusion on CUDA; tiled reference elsewhere |
| O2 | Pair-state attention | FlagGems `stofm_pair_attention` | `baddbmm` score construction and no clone when pair state is unused |
| O3 | End-to-end | O1 + B1 + native attention | End-to-end comparator reported against B1 |
| O4 | End-to-end | O1 + O2 + B1 | End-to-end comparator reported against B1 |

The V100 report must use `B0` as the baseline for O1 and O2, and `B1` as the
baseline for O3.  B2 is listed only to make its exclusion auditable.

## Correctness Gates

1. Parse every new Python source with `compileall` and run the target-adapter
   static contract checker.
2. Compare Gaussian output to the original formula in FP32, including zero
   distance masking.  Use `rtol=3e-4`, `atol=3e-5`.
3. Compare gradients for all Gaussian learnable tensors against the dense
   reference with the same tolerance.
4. Compare attention output, attention weights, next pair state, and the
   gradient of `pair_bias`, with and without a key-padding mask.
5. Compare the end-to-end `last_hidden_state` for equal model parameters.
   A final `pair_rep` may be omitted only when the caller explicitly sets
   `return_pair_rep=False`; intermediate layers must still propagate it.
6. Record any unsupported dtype or target runtime as `skipped`, never as a
   passing result.

## V100 Measurement

The benchmark uses CUDA events, fixed seed 42, inference mode, disabled
dropout, a ten-iteration warm-up, 30 measured samples, and five calls per
sample.  Compilation occurs during warm-up and is excluded from latency.

The canonical workload is `batch=1`, `nodes=1050`, `layers=4`,
`embedding_dim=256`, `heads=8`, `gaussian_hidden_dim=128`, and
`input_dim=256`.  A smoke workload may be used to validate the harness but
cannot replace the canonical report.

Each result directory contains:

- `result.json`: immutable inputs, runtime versions, commits, raw samples, and
  summary statistics.
- `samples.csv`: one latency per row for future statistical reanalysis.
- `report.md` and `report.html`: a human-readable p50 comparison.

The report must present p20/p50/p80/p95, mean, peak allocated MiB, raw sample
count, and the exact commit IDs.  A speedup is `baseline_p50 / candidate_p50`.

## Acceptance Targets, Not Results

Before measurement, the engineering targets on a V100 are O1 >= 1.30x versus
B0 Gaussian and at least one of O3/O4 >= 1.10x versus B1 end-to-end at p50.
O2 must be assessed both in isolation and as part of O4; the selected default
is the faster end-to-end candidate after correctness validation.  These targets
are decision thresholds, not performance claims.  The generated report is the
only source for measured values.

## Deferred Ascend 310 and MTT S4000 Validation

The target adapters are source-validated before rental but make no performance
claim.  On each rented target, first capture a new B0/B1 baseline under the
vendor PyTorch extension, then rerun all correctness gates and the canonical
benchmark.  Keep the V100 and target reports separate: their compiler,
operator library, memory hierarchy, and precision support are not comparable
as a single speedup number.
