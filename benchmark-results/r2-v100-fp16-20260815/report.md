# SToFM FlagOS Inference R2 V100 Suite

Precision: `fp16`; independent trials: `3`.

| Stage | Gain kind | p50 median ms | p50 min/max ms | Baseline | Speedup | 95% bootstrap CI | Raw samples |
| --- | --- | ---: | ---: | --- | ---: | --- | ---: |
| C1_gaussian_compiler | compiler_routing | 8.1812 | 8.1305/8.3718 | F0_stock_steady | 2.352x | [2.319x, 2.365x] | 90 |
| C2_pair_native_epilogue | custom_kernel | 18.9990 | 18.9954/19.0153 | F0_stock_steady | 1.018x | [1.018x, 1.018x] | 90 |
| F0_stock_lifecycle | flaggems_lifecycle | 19.4016 | 19.3846/19.4038 | P1_canonical_torch | 1.084x | [1.084x, 1.084x] | 90 |
| F0_stock_steady | stock_aten | 19.3490 | 19.3283/19.3501 | P1_canonical_torch | 1.087x | [1.087x, 1.087x] | 90 |
| Ffinal_optimized_lifecycle | flaggems_lifecycle | 9.3005 | 9.1376/9.9806 | F0_stock_steady | 2.051x | [1.991x, 2.081x] | 90 |
| Ffinal_optimized_steady | combined | 8.8179 | 8.7190/9.4030 | F0_stock_steady | 2.151x | [2.098x, 2.181x] | 90 |
| P0_legacy_pair_output | model_lifecycle | 21.2764 | 21.2702/21.2783 | P0_legacy_pair_output | - | - | 90 |
| P1_canonical_torch | model_lifecycle | 21.0268 | 21.0189/21.0306 | P1_canonical_torch | - | - | 90 |

The frozen F0 result is collected in a separate process and package environment. Compiler, custom-kernel, and scope-lifecycle stages are not conflated.
