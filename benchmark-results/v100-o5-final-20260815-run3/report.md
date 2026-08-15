# SToFM V100 Benchmark

Run ID: `v100-20260815T074017Z`

Device: `Tesla V100-SXM2-16GB`; PyTorch `2.5.1+cu121`; CUDA `12.1`

Correctness gate: `passed` for B1/O3/O4/O5 end-to-end last hidden state before timing.

| Stage | Scope | Samples | p20 ms | p50 ms | p80 ms | p95 ms | Mean ms | Speedup | Peak delta MiB | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B0_gaussian | gaussian | 30 | 17.5848 | 17.5892 | 17.6031 | 17.8098 | 17.6238 | 1.000x | 1616.7 | measured |
| O1_gaussian | gaussian | 30 | 5.8406 | 5.8548 | 5.8597 | 5.8645 | 5.8473 | 3.004x | 1.7 | measured |
| O1n_gaussian_triton | gaussian | 30 | 10.7375 | 10.7394 | 10.7572 | 10.8291 | 10.7533 | 0.545x | 35.3 | measured |
| B0_attention | attention | 30 | 0.9013 | 0.9020 | 0.9047 | 0.9081 | 0.9036 | 1.000x | 71.0 | measured |
| O2_attention | attention | 30 | 0.9168 | 0.9199 | 0.9236 | 0.9269 | 0.9207 | 0.981x | 72.7 | measured |
| O2n_attention_triton_epilogue | attention | 30 | 0.6608 | 0.6663 | 0.6968 | 0.7439 | 0.6811 | 1.381x | 72.7 | measured |
| B0_e2e | end_to_end | 30 | 23.6061 | 23.6115 | 23.6181 | 23.6390 | 23.6160 | 1.000x | 1618.7 | measured |
| B1_e2e | end_to_end | 30 | 23.1690 | 23.1728 | 23.1811 | 23.1853 | 23.1745 | 1.019x | 1618.7 | measured |
| B2_e2e | end_to_end | - | - | - | - | - | - | - | - | skipped |
| O3_e2e_gaussian_lifecycle | end_to_end | 30 | 11.4116 | 11.4161 | 11.4226 | 11.4343 | 11.4179 | 2.030x | 152.1 | measured |
| O4_e2e_pair_attention | end_to_end | 30 | 10.8619 | 10.8656 | 10.8725 | 10.8987 | 10.8713 | 2.133x | 144.7 | measured |
| O5_e2e_triton_pair_epilogue | end_to_end | 30 | 8.8245 | 8.8360 | 8.8439 | 8.8493 | 8.8346 | 1.230x | 144.7 | measured |
