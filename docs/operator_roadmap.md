# SToFM Operator Roadmap

The integration narrows the optimization surface to the two operations that
dominate SToFM's pair-state path.  The stages below distinguish implemented
code from deferred hardware work.

| Operation | Torch / ecosystem limitation | CUDA V100 implementation | Ascend 310 implementation state | MTT S4000 implementation state |
| --- | --- | --- | --- | --- |
| Gaussian pair bias `[B,N,N,K]` | Eager PyTorch materializes a large expanded RBF activation; generic graph rewriting cannot recognize this SToFM-specific formula | O1 uses a compiler-friendly dense formulation through `torch.compile`; the portable fallback tiles pairs | Correctness-first tiled adapter; profile and replace with CANN fused/vector kernel after runtime validation | Correctness-first tiled adapter; profile and replace with MUSA/TLE fused kernel after runtime validation |
| Pair-state attention score and update | Eager path performs BMM, separate bias addition, and a clone solely to expose next `pair_rep` | O2 uses `baddbmm`; O4 is the measured V100 winner | Correctness-first `baddbmm -> softmax -> bmm`; future fused score/update kernel | Correctness-first `baddbmm -> softmax -> bmm`; future fused score/update kernel |
| Final pair-state materialization | The original model always materializes a final `[B,H,N,N]` state even in embedding extraction | B1 adds explicit `return_pair_rep=False` propagation | Same source path, subject to runtime correctness test | Same source path, subject to runtime correctness test |

## Architecture-Specific Gaps to Close

### NVIDIA V100

- O1 currently relies on Inductor graph fusion rather than a custom CUDA/Triton
  kernel.  Benchmark generated code and peak memory before deciding whether a
  custom kernel is justified.
- The committed `N=1050` measurement selected O4 (O1 + O2 + B1): it is faster
  than the native-attention O3 comparator while preserving pair-state tests.
- FlagGems has no Volta architecture-specialized registry for this device, so
  the result uses the generic CUDA/Inductor route.  A Volta-specific tuning or
  fused kernel remains a separate optimization opportunity.
- O2 is not a FlashAttention replacement: pair-state preservation changes the
  data dependency and must remain covered by a pair-state regression test.

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
