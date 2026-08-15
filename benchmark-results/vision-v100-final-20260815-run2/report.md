# Vision Operator V100 Benchmark

Run ID: `vision-v100-20260815T075854Z`

| Stage | Samples | p20 ms | p50 ms | p80 ms | p95 ms | Mean ms | Relative p50 | Peak delta MiB | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B0_marker_token_reference | 30 | 0.2711 | 0.2802 | 0.2895 | 0.2958 | 0.2804 | 1.000x | 24.0 | measured |
| O1_marker_token_triton | 30 | 0.1956 | 0.1987 | 0.2005 | 0.2125 | 0.1998 | 1.410x | 12.0 | measured |
| B0_swiglu_reference | 30 | 0.0500 | 0.0503 | 0.0526 | 0.0539 | 0.0511 | 1.000x | 8.2 | measured |
| O2_swiglu_existing_triton | 30 | 0.0872 | 0.0895 | 0.0923 | 0.0964 | 0.0902 | 0.562x | 4.1 | measured |
| B0_residual_layer_norm | 30 | 0.0504 | 0.0515 | 0.0539 | 0.0570 | 0.0525 | 1.000x | 0.8 | measured |
| O3_residual_layer_norm | - | - | - | - | - | - | - | - | rejected |
