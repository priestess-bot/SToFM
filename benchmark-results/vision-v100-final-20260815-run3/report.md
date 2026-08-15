# Vision Operator V100 Benchmark

Run ID: `vision-v100-20260815T075927Z`

| Stage | Samples | p20 ms | p50 ms | p80 ms | p95 ms | Mean ms | Relative p50 | Peak delta MiB | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B0_marker_token_reference | 30 | 0.2728 | 0.2798 | 0.2881 | 0.3145 | 0.2832 | 1.000x | 24.0 | measured |
| O1_marker_token_triton | 30 | 0.2094 | 0.2135 | 0.2218 | 0.2379 | 0.2170 | 1.310x | 12.0 | measured |
| B0_swiglu_reference | 30 | 0.0528 | 0.0556 | 0.0582 | 0.0593 | 0.0556 | 1.000x | 8.2 | measured |
| O2_swiglu_existing_triton | 30 | 0.0883 | 0.0913 | 0.0936 | 0.0997 | 0.0930 | 0.609x | 4.1 | measured |
| B0_residual_layer_norm | 30 | 0.0532 | 0.0546 | 0.0568 | 0.0589 | 0.0551 | 1.000x | 0.8 | measured |
| O3_residual_layer_norm | - | - | - | - | - | - | - | - | rejected |
