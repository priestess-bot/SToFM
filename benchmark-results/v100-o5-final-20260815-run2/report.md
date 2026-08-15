# SToFM V100 Benchmark

Run ID: `v100-20260815T073938Z`

Device: `Tesla V100-SXM2-16GB`; PyTorch `2.5.1+cu121`; CUDA `12.1`

Correctness gate: `passed` for B1/O3/O4/O5 end-to-end last hidden state before timing.

| Stage | Scope | Samples | p20 ms | p50 ms | p80 ms | p95 ms | Mean ms | Speedup | Peak delta MiB | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B0_gaussian | gaussian | 30 | 17.5804 | 17.5860 | 17.6015 | 17.6912 | 17.6144 | 1.000x | 1616.7 | measured |
| O1_gaussian | gaussian | 30 | 5.8195 | 5.8339 | 5.8368 | 5.8501 | 5.8269 | 3.014x | 1.7 | measured |
| O1n_gaussian_triton | gaussian | 30 | 10.7835 | 10.7858 | 10.7892 | 10.8501 | 10.7927 | 0.541x | 35.3 | measured |
| B0_attention | attention | 30 | 0.8980 | 0.8991 | 0.9001 | 0.9033 | 0.8992 | 1.000x | 71.0 | measured |
| O2_attention | attention | 30 | 0.9111 | 0.9135 | 0.9177 | 0.9195 | 0.9141 | 0.984x | 72.7 | measured |
| O2n_attention_triton_epilogue | attention | 30 | 0.6547 | 0.6563 | 0.6572 | 0.6583 | 0.6562 | 1.392x | 72.7 | measured |
| B0_e2e | end_to_end | 30 | 23.5971 | 23.5998 | 23.6055 | 23.6110 | 23.6039 | 1.000x | 1618.7 | measured |
| B1_e2e | end_to_end | 30 | 23.1610 | 23.1676 | 23.1722 | 23.1932 | 23.1710 | 1.019x | 1618.7 | measured |
| B2_e2e | end_to_end | - | - | - | - | - | - | - | - | skipped |
| O3_e2e_gaussian_lifecycle | end_to_end | 30 | 11.3977 | 11.4010 | 11.4077 | 11.4224 | 11.4032 | 2.032x | 152.1 | measured |
| O4_e2e_pair_attention | end_to_end | 30 | 10.8498 | 10.8552 | 10.8650 | 10.8769 | 10.8581 | 2.134x | 144.7 | measured |
| O5_e2e_triton_pair_epilogue | end_to_end | 30 | 8.8074 | 8.8105 | 8.8164 | 8.8429 | 8.8152 | 1.232x | 144.7 | measured |
