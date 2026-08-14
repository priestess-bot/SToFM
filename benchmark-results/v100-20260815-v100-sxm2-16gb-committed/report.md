# SToFM V100 Benchmark

Run ID: `v100-20260814T190640Z`

Device: `Tesla V100-SXM2-16GB`; PyTorch `2.5.1+cu121`; CUDA `12.1`

Correctness gate: `passed` for B1/O3/O4 end-to-end last hidden state before timing.

| Stage | Scope | Samples | p20 ms | p50 ms | p80 ms | p95 ms | Mean ms | Speedup | Peak delta MiB | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B0_gaussian | gaussian | 30 | 17.5894 | 17.5996 | 17.6703 | 17.8617 | 17.6465 | 1.000x | 1616.7 | measured |
| O1_gaussian | gaussian | 30 | 5.8058 | 5.8197 | 5.8518 | 5.8557 | 5.8239 | 3.024x | 1.7 | measured |
| B0_attention | attention | 30 | 0.9073 | 0.9282 | 0.9406 | 0.9424 | 0.9238 | 1.000x | 71.0 | measured |
| O2_attention | attention | 30 | 0.9187 | 0.9212 | 0.9261 | 0.9374 | 0.9246 | 1.008x | 72.7 | measured |
| B0_e2e | end_to_end | 30 | 23.4496 | 23.4533 | 23.4609 | 23.4793 | 23.4587 | 1.000x | 1618.7 | measured |
| B1_e2e | end_to_end | 30 | 23.0154 | 23.0191 | 23.0335 | 23.0642 | 23.0296 | 1.019x | 1618.7 | measured |
| B2_e2e | end_to_end | - | - | - | - | - | - | - | - | skipped |
| O3_e2e_native_attention | end_to_end | 30 | 11.3857 | 11.3909 | 11.4076 | 11.4165 | 11.3952 | 2.021x | 152.8 | measured |
| O4_e2e_pair_attention | end_to_end | 30 | 10.8417 | 10.8458 | 10.8490 | 10.8586 | 10.8466 | 2.122x | 145.4 | measured |
