# SToFM Operator Roadmap

> Historical R1 roadmap. The current R2 coverage, measurements, and promotion
> decisions are in `docs/flagos_inference_r2_report.md`; do not compare the
> older O-stage values below with R2 results.

The integration narrows the optimization surface to the two operations that
dominate SToFM's pair-state path.  The stages below distinguish implemented
code from deferred hardware work.

| Operation | Torch / ecosystem limitation | CUDA V100 implementation | Ascend 310 implementation state | MTT S4000 implementation state |
| --- | --- | --- | --- | --- |
| Gaussian pair bias `[B,N,N,K]` | Eager PyTorch materializes a large expanded RBF activation; generic graph rewriting cannot recognize this SToFM-specific formula | O1 uses a compiler-friendly dense formulation through `torch.compile`; the portable fallback tiles pairs | Correctness-first tiled adapter; profile and replace with CANN fused/vector kernel after runtime validation | Correctness-first tiled adapter; profile and replace with MUSA/TLE fused kernel after runtime validation |
| Pair-state attention score and update | Eager path performs BMM, separate bias addition, and a clone solely to expose next `pair_rep` | O2 offers `baddbmm`; it is an explicit opt-in until its target-specific benchmark wins | Correctness-first `baddbmm -> softmax -> bmm`; future fused score/update kernel | Correctness-first `baddbmm -> softmax -> bmm`; future fused score/update kernel |
| Final pair-state materialization | The original model always materializes a final `[B,H,N,N]` state even in embedding extraction | B1 adds explicit `return_pair_rep=False` propagation | Same source path, subject to runtime correctness test | Same source path, subject to runtime correctness test |

## Architecture-Specific Gaps to Close

### NVIDIA V100

- O1 currently relies on Inductor graph fusion rather than a custom CUDA/Triton
  kernel.  Benchmark generated code and peak memory before deciding whether a
  custom kernel is justified.
- O2 is an operator-expression improvement, not a FlashAttention replacement:
  pair-state preservation changes the data dependency and must remain covered
  by a pair-state regression test.

### Huawei Ascend 310

- There is no tested target runtime in this repository yet.  The adapter is
  intentionally vendor-import-free so it can be syntax checked without a
  rental.
- Before fusion work, establish support and accuracy for the exact operations:
  `linear`, `exp`, `relu`, `baddbmm`, masked fill, softmax, and `bmm`.
- Candidate fusion boundary: distance affine transform + RBF + two projection
  layers for Gaussian bias; score BMM + pair bias + mask + softmax for
  attention.  Backward remains on the reference path until gradients are
  validated on device.

### Moore Threads MTT S4000

- As with Ascend, no device results are claimed before the MUSA runtime is
  available.
- First check shape/layout behavior for `baddbmm` and the performance impact of
  contiguous conversions.  Then decide whether a fused MUSA/TLE kernel offers
  enough gain over the correct tiled reference.

## Expansion Contract

Every later backend addition must implement the stable public calls
`stofm_gaussian_pair_bias` and `stofm_pair_attention`, expose a static backend
specification, and add a target result directory containing raw samples.  SToFM
must not import a vendor extension directly; the dependency boundary remains
inside FlagGems.
