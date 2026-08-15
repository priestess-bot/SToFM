# FlagOS Inference R2 V100 Optimization Report

## Scope and Provenance

This is the authoritative R2 inference report. It measures a real scoped
`flag_gems.use_gems()` installation, not a generic compiler backend renamed as
FlagOS.

- P1 is canonical pure PyTorch inference.
- F0 is frozen stock FlagGems commit
  `03bf364ede763d573d5c30124d554283a209ab85` in its own virtual environment,
  using the temporary ATen allowlist `addmm,baddbmm,bmm,softmax`.
- Optimized R2 FlagGems is pushed commit
  `a9a96bbcc3d685482c656343e0759b7b4a5c38bc`. SToFM calls its versioned public
  Gaussian and pair-attention composites in addition to the same scoped ATen
  route.
- SToFM timing workers are commit
  `e01b0a9d8348101815fba63224910a25830c29a9`; Vision timing workers are
  `de4278f7307de167dfd21d406198ca2c2881658e`. Later commits only improve
  profile classification and evidence verification.

The test device was GPU 0, Tesla V100-SXM2-16GB (compute capability 7.0), with
NVIDIA driver 550.144.03, PyTorch 2.6.0+cu124, CUDA 12.4, Triton 3.2.0, and
Python 3.11.15. Every worker records `pip freeze`, driver, GPU, Torch backend
settings, and relevant environment variables. TF32 and CuDNN benchmarking were
disabled, and inference mode was enabled.

## Strict Test Process

Each precision used three independent Python processes. Every stage uses a
fresh model state and deterministic generated input, ten warm-up calls, 30
CUDA-event samples, and five calls per sample. Compilation happens in warm-up
and is excluded from latency. Each measured stage therefore has 90 raw samples.
Bootstrap intervals use 10,000 resamples of pooled raw samples.

The canonical SToFM workload is `B=1,N=1050,L=4,D=256,FFN=256,H=8,K=128` with
`return_pair_rep=False`. Vision figures are operator boundaries, not full
Uni2/KRONOS models: KRONOS marker assembly uses `B=1,M=32,T=256,D=384`, and the
Uni2 packed SwiGLU boundary uses sequence 264 and packed hidden size 8192.

Before timing, each candidate is checked against Torch with
`torch.testing.assert_close`; FP32 uses `rtol=3e-4, atol=3e-5`, FP16 uses
`rtol=3e-2, atol=3e-3`, and marker padding masks must be exactly equal. The
pure-Torch end-to-end reference hashes matched across frozen and optimized
environments in all three independent runs:

| Precision | P1 `last_hidden_state` SHA-256 |
| --- | --- |
| FP32 | `6816c64679fdc1af5fb5fe502e53428f0aca070acba8d604c3cc14b247fbe7fc` |
| FP16 | `a2cfaf2864fd99869b1496997b2ef73b517996de83341c23dc6e0f8ac851fa7d` |

The API tests also cover masks, pair state, returned attention weights,
non-contiguous reference fallback, and autograd fallback. Target adapters are
tested separately in CPU-only static mode; no target runtime is represented as
a pass.

## End-to-End SToFM Results

The speedup column names its comparison baseline. It is the bootstrap point
estimate, so it may differ slightly from a ratio of table medians. C1, C2, and
Ffinal use `F0_stock_steady` as their baseline, separating real frozen FlagOS
ATen coverage from temporary scope setup.

| Stage | Route | FP32 p50 ms | FP32 speedup and 95% CI | FP16 p50 ms | FP16 speedup and 95% CI |
| --- | --- | ---: | --- | ---: | --- |
| P0 | Legacy Torch, final pair state materialized | 23.5451 | - | 21.2764 | - |
| P1 | Canonical Torch, final pair state omitted | 23.0998 | - | 21.0268 | - |
| F0 lifecycle | Frozen FlagOS scope per call, vs P1 | 21.7685 | 1.061x [1.061x, 1.061x] | 19.4016 | 1.084x [1.084x, 1.084x] |
| F0 steady | Frozen FlagOS scope held open, vs P1 | 21.7006 | 1.064x [1.063x, 1.065x] | 19.3490 | 1.087x [1.087x, 1.087x] |
| C1 | Gaussian compiler, vs F0 steady | 10.5686 | 2.054x [2.052x, 2.055x] | 8.1812 | 2.352x [2.319x, 2.365x] |
| C2 | Native pair-score epilogue, vs F0 steady | 20.6351 | 1.052x [1.052x, 1.053x] | 18.9990 | 1.018x [1.018x, 1.018x] |
| Ffinal lifecycle | C1 + C2 + FlagOS scope per call, vs F0 steady | 9.5534 | 2.255x [2.159x, 2.273x] | 9.3005 | 2.051x [1.991x, 2.081x] |
| Ffinal steady | C1 + C2 + FlagOS scope held open, vs F0 steady | 9.4850 | 2.277x [2.225x, 2.288x] | 8.8179 | 2.151x [2.098x, 2.181x] |

The aggregate-p50 P1-to-Ffinal ratio is 2.435x in FP32 and 2.385x in FP16.
Removing the unused final pair representation alone changes P0 to P1 by
1.019x FP32 and 1.012x FP16; that is a lifecycle improvement, not a kernel
claim.

Median peak allocation growth within the timed window was 1618.09 MiB FP32 and
1617.07 MiB FP16 for P1/F0, 152.54 MiB and 79.29 MiB for C1, and 145.15 MiB
and 71.40 MiB for Ffinal. This is not total process or model memory.

## Vision Operator Results

| Boundary | FP32 Torch -> candidate p50 ms | FP32 speedup and 95% CI | FP16 Torch -> candidate p50 ms | FP16 speedup and 95% CI | Decision |
| --- | --- | --- | --- | --- | --- |
| KRONOS marker-token assembly | 0.2867 -> 0.2449 | 1.154x [1.137x, 1.174x] | 0.2897 -> 0.2651 | 1.106x [1.075x, 1.138x] | Promote NVIDIA Triton candidate |
| Uni2 packed SwiGLU | 0.0592 -> 0.1227 | 0.484x [0.472x, 0.497x] | 0.0600 -> 0.1373 | 0.438x [0.430x, 0.450x] | Reject existing candidate |
| Uni2 residual LayerNorm | 0.0626 -> no candidate | - | 0.0609 -> no candidate | - | Retain Torch |

Clean profiles show `_marker_token_embed_kernel` in both marker candidate
precisions. They show `swiglu_kernel` for the rejected existing candidate and
`aten::native_layer_norm` for the retained LayerNorm path.

## Operator Coverage and Promotion Matrix

| Boundary | Stock Torch/ecosystem gap | V100 R2 result and decision | Ascend 310 / CANN | MTT S4000 / MUSA |
| --- | --- | --- | --- | --- |
| Gaussian pair bias `[B,N,N,H]` | Eager scalar RBF/projection operations do not expose an SToFM composite boundary; generic ATen replacement does not fuse it. | Versioned public compiler route: C1 2.054x FP32, 2.352x FP16 vs F0. Promote compiler route. The separate NVIDIA Triton Gaussian is available but is not claimed as an R2 winner. | Deferred AscendC source: FP32/FP16/BF16 symbols, tiling metadata, host registration, lazy import, reference fallback. No binary or timing claim. | Deferred MUSA `.mu` extension/Triton route: FP32/FP16/BF16, `musa_add_library`, reference fallback. No binary or timing claim. |
| Pair score, mask, softmax, pair state | `baddbmm`, mask, softmax, and pair materialization are separate generic boundaries. | NVIDIA native inference epilogue: C2 1.052x FP32, 1.018x FP16 vs F0. Keep in optimized route; repeat for shape buckets. | Deferred CANN Gaussian/pair project; Vision remains lazy reference-first. | Deferred extension/Triton and `PrivateUse1` schema. |
| `addmm,baddbmm,bmm,softmax` | No SToFM-aware route selection; global patching hides scope lifecycle. | Real narrow FlagOS scope: F0 steady 1.064x FP32, 1.087x FP16 vs P1. Keep as scoped ATen coverage, not a whole-model backend claim. | Establish target F0 after runtime install. | Establish target F0 after runtime install. |
| KRONOS marker-token assembly | Lookup, broadcast, position/token add, and padding are separate operations. | Triton marker kernel: 1.154x FP32, 1.106x FP16. Promote for contiguous inference input. | Lazy adapter/reference fallback only. | Lazy Triton/extension candidate/reference fallback only. |
| Uni2 packed SwiGLU | `silu` plus multiply is launch-sensitive at this shape. | Existing candidate loses at 0.484x FP32 and 0.438x FP16. Reject on V100. | Lazy reference-first adapter. | Lazy candidate/reference fallback. |
| Residual LayerNorm | No verified composite win. | Retain Torch `native_layer_norm`; no new kernel. | Lazy adapter/reference fallback. | Lazy adapter/reference fallback. |

## Profile and Target Gates

Profiles are qualitative routing evidence, not latency sources. SToFM P1/F0/
Ffinal traces for both precisions are under
`benchmark-results/r2-v100-profiles-20260815/fp{32,16}-{p1,f0,final}/`; they
classify P1 as reference, F0 as stock FlagOS ATen, and final as optimized
FlagOS plus Gaussian compiler. The final public dispatch records pair attention
selected as `nvidia`. Triton does not always preserve a stable pair-epilogue
name in the PyTorch trace, so no ambiguous label is used as proof of execution;
the direct C2 timing, public dispatch, and numerical test support that result.

Ascend and MTT are source-complete only. Offline gates passed Python AST,
schemas, manifests, host-visible C++ syntax, and deferred CMake configure/build
without importing `torch_npu` or `torch_musa`. On rental hardware, build with
the actual SDK, validate FP32/FP16/BF16 shape/mask/padding/non-contiguous/
gradient matrices, save target P1/F0, then measure each promoted boundary and
end-to-end. Planning thresholds, not performance claims, are Gaussian >=1.15x,
pair >=1.05x, marker >=1.05x per operator, and >=1.10x end-to-end versus each
target's F0 baseline.

## Evidence Integrity and Limits

Raw JSON, CSV samples, worker logs, reports, traces, and checksum manifests are
committed under these directories:

- `benchmark-results/r2-v100-fp32-20260815/`
- `benchmark-results/r2-v100-fp16-20260815/`
- `benchmark-results/r2-vision-v100-fp32-20260815/`
- `benchmark-results/r2-vision-v100-fp16-20260815/`
- `benchmark-results/r2-v100-profiles-20260815/`

`benchmarks/write_r2_checksums.py --verify` verified 20, 20, 11, 11, and 48
files respectively. Audit directories named `invalid-*` contain an earlier
concurrent timing attempt or pre-cleanup profiles and are excluded from every
reported number and checksum manifest.

Results apply only to the declared V100 workloads. SToFM is end-to-end synthetic
inference with fixed shapes; Vision results are operator-level boundaries, not
full Uni2/KRONOS throughput claims. Target-device numbers are intentionally
absent until real device execution.
