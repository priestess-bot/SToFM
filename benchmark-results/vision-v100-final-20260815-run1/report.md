# Vision Operator V100 Benchmark

Run ID: `vision-v100-20260815T075824Z`

| Stage | Samples | p20 ms | p50 ms | p80 ms | p95 ms | Mean ms | Relative p50 | Peak delta MiB | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B0_marker_token_reference | 30 | 0.2855 | 0.3030 | 0.3344 | 0.3560 | 0.3095 | 1.000x | 24.0 | measured |
| O1_marker_token_triton | 30 | 0.2058 | 0.2154 | 0.2297 | 0.2408 | 0.2182 | 1.406x | 12.0 | measured |
| B0_swiglu_reference | 30 | 0.0521 | 0.0560 | 0.0603 | 0.0698 | 0.0573 | 1.000x | 8.2 | measured |
| O2_swiglu_existing_triton | 30 | 0.0864 | 0.0940 | 0.1036 | 0.1074 | 0.0947 | 0.596x | 4.1 | measured |
| B0_residual_layer_norm | 30 | 0.0487 | 0.0526 | 0.0554 | 0.0576 | 0.0525 | 1.000x | 0.8 | measured |
| O3_residual_layer_norm | - | - | - | - | - | - | - | - | rejected |
