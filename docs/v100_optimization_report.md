# V100 Optimization Report

## Verdict

The selected V100 path is O4: FlagGems Gaussian pair-bias compilation, direct
pair-state attention, and omission of the final unused pair state. At the
canonical SToFM workload it reduced end-to-end p50 from 23.4533 ms (B0) to
10.8458 ms, a 2.162x speedup. The committed raw evidence is in
[`benchmark-results/v100-20260815-v100-sxm2-16gb-committed`](../benchmark-results/v100-20260815-v100-sxm2-16gb-committed/).

## Reproducibility Record

| Field | Value |
| --- | --- |
| Run ID | `v100-20260814T190640Z` |
| Device | Tesla V100-SXM2-16GB, compute capability 7.0, 16,384 MiB |
| Driver | 550.144.03 |
| Runtime | PyTorch 2.5.1+cu121, CUDA 12.1, Triton 3.1.0 |
| SToFM implementation | `f92db8891139749f812cdbdc8dbb39064f5da406` |
| FlagGems implementation | `dd9abb71aa79a4bbc6428b85cd7eeef5d5b7bb33` |
| Workload | FP32 inference, batch 1, nodes 1050, 4 layers, embedding 256, 8 heads, Gaussian hidden 128 |
| Measurement | CUDA events; 10 warm-up calls; 30 samples; 5 calls per sample; compilation excluded |

The tracked artifacts are [`result.json`](../benchmark-results/v100-20260815-v100-sxm2-16gb-committed/result.json),
[`samples.csv`](../benchmark-results/v100-20260815-v100-sxm2-16gb-committed/samples.csv),
[`report.md`](../benchmark-results/v100-20260815-v100-sxm2-16gb-committed/report.md), and
[`report.html`](../benchmark-results/v100-20260815-v100-sxm2-16gb-committed/report.html).

## Correctness Gates

| Gate | Evidence | Result |
| --- | --- | --- |
| FlagGems Gaussian semantics and gradients | tiled/reference forward and all learnable gradients | passed |
| FlagGems pair attention | context, pair state, weights, and `pair_bias` gradients with padding | passed |
| SToFM bridge | Gaussian, attention, route inheritance/override, and final pair-state omission | passed |
| SToFM end-to-end gradients | cell embedding and distance input gradients | passed |
| Ascend/MTT source readiness | AST/static target contract checker | passed; runtime deferred |
| Pre-benchmark output check | B1, O3, and O4 `last_hidden_state`; `rtol=3e-4`, `atol=3e-5` | passed |

The test commands were `pytest -q tests/test_stofm_experimental.py` in
FlagGems (4 tests) and `pytest -q tests/test_flagos_adapter.py` in SToFM
(5 tests). The latter also passed with `FLAGGEMS_VENDOR` unset, exercising the
V100 CUDA device discovery fix.

## Measured Stages

| Stage | Optimization point | p50 ms | p20-p95 ms | Relative p50 | Peak allocated MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| B0 Gaussian | Original expanded RBF expression | 17.5996 | 17.5894-17.8617 | 1.000x vs B0 Gaussian | 4382.8 |
| O1 Gaussian | Compiler-friendly FlagGems Gaussian expression | 5.8197 | 5.8058-5.8557 | 3.024x vs B0 Gaussian | 2759.7 |
| B0 attention | Original BMM + add score construction | 0.9282 | 0.9073-0.9424 | 1.000x vs B0 attention | 2837.2 |
| O2 attention | FlagGems `baddbmm` pair-state construction | 0.9212 | 0.9187-0.9374 | 1.008x vs B0 attention | 2838.8 |
| B0 end-to-end | Original model with final pair state | 23.4533 | 23.4496-23.4793 | 1.000x | 4384.9 |
| B1 end-to-end | Omit only final unused pair state | 23.0191 | 23.0154-23.0642 | 1.019x vs B0 | 4384.9 |
| O3 end-to-end | O1 + B1 with native attention | 11.3909 | 11.3857-11.4165 | 2.021x vs B1 | 2918.9 |
| O4 end-to-end | O1 + O2 + B1, selected | 10.8458 | 10.8417-10.8586 | 2.122x vs B1; 2.162x vs B0 | 2911.5 |

O1 lowers the Gaussian p50 by 11.7799 ms and its peak allocated memory by
1623.1 MiB (37.0%). B1 lowers end-to-end p50 by 1.9%, while O4 lowers peak
allocated memory by 1473.3 MiB (33.6%) versus B0. Although O2's isolated gain
is small, its end-to-end combination is 4.8% lower latency than O3; this is
why the default `inherit` selection uses the direct pair-attention path when
`flagos_backend=flaggems`.

## Limitations and Next Tests

This result covers FP32 inference on one V100 and the canonical `B=1, N=1050`
shape. It does not claim training throughput, mixed-precision behavior, larger
batches, or performance on Ascend 310 / MTT S4000. FlagGems also logs that no
Volta architecture-specialized registry exists; O1 therefore uses the generic
CUDA Inductor route. The deferred target test order and promotion criteria are
maintained in `target_device_acceptance.md`.
