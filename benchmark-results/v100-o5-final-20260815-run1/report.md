# SToFM V100 Benchmark

Run ID: `v100-20260815T073902Z`

Device: `Tesla V100-SXM2-16GB`; PyTorch `2.5.1+cu121`; CUDA `12.1`

Correctness gate: `passed` for B1/O3/O4/O5 end-to-end last hidden state before timing.

| Stage | Scope | Samples | p20 ms | p50 ms | p80 ms | p95 ms | Mean ms | Speedup | Peak delta MiB | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B0_gaussian | gaussian | 30 | 17.5819 | 17.5847 | 17.5918 | 17.8814 | 17.6285 | 1.000x | 1616.7 | measured |
| O1_gaussian | gaussian | 30 | 5.8156 | 5.8316 | 5.8382 | 5.8618 | 5.8285 | 3.015x | 1.7 | measured |
| O1n_gaussian_triton | gaussian | 30 | 10.7384 | 10.7400 | 10.7453 | 10.8106 | 10.7572 | 0.543x | 35.3 | measured |
| B0_attention | attention | 30 | 0.9013 | 0.9040 | 0.9091 | 0.9156 | 0.9054 | 1.000x | 71.0 | measured |
| O2_attention | attention | 30 | 0.9148 | 0.9181 | 0.9288 | 0.9351 | 0.9212 | 0.985x | 72.7 | measured |
| O2n_attention_triton_epilogue | attention | 30 | 0.6580 | 0.6603 | 0.6628 | 0.6658 | 0.6605 | 1.391x | 72.7 | measured |
| B0_e2e | end_to_end | 30 | 23.5987 | 23.6024 | 23.6086 | 23.6128 | 23.6035 | 1.000x | 1618.7 | measured |
| B1_e2e | end_to_end | 30 | 23.1617 | 23.1668 | 23.1746 | 23.1832 | 23.1686 | 1.019x | 1618.7 | measured |
| B2_e2e | end_to_end | - | - | - | - | - | - | - | - | skipped |
| O3_e2e_gaussian_lifecycle | end_to_end | 30 | 11.3959 | 11.4009 | 11.4070 | 11.4097 | 11.4012 | 2.032x | 152.1 | measured |
| O4_e2e_pair_attention | end_to_end | 30 | 10.8466 | 10.8529 | 10.8569 | 10.8724 | 10.8532 | 2.135x | 144.7 | measured |
| O5_e2e_triton_pair_epilogue | end_to_end | 30 | 8.8025 | 8.8077 | 8.8177 | 8.8310 | 8.8106 | 1.232x | 144.7 | measured |
