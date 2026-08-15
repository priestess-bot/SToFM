# SToFM FlagOS Inference R2 V100 Suite

Precision: `fp32`; independent trials: `3`.

| Stage | Gain kind | p50 median ms | p50 min/max ms | Baseline | Speedup | 95% bootstrap CI | Raw samples |
| --- | --- | ---: | ---: | --- | ---: | --- | ---: |
| C1_gaussian_compiler | compiler_routing | 10.5686 | 10.5662/10.5828 | F0_stock_steady | 2.054x | [2.052x, 2.055x] | 90 |
| C2_pair_native_epilogue | custom_kernel | 20.6351 | 20.6280/20.6353 | F0_stock_steady | 1.052x | [1.052x, 1.053x] | 90 |
| F0_stock_lifecycle | flaggems_lifecycle | 21.7685 | 21.7419/21.7895 | P1_canonical_torch | 1.061x | [1.061x, 1.061x] | 90 |
| F0_stock_steady | stock_aten | 21.7006 | 21.6916/21.7284 | P1_canonical_torch | 1.064x | [1.063x, 1.065x] | 90 |
| Ffinal_optimized_lifecycle | flaggems_lifecycle | 9.5534 | 9.5319/10.3354 | F0_stock_steady | 2.255x | [2.159x, 2.273x] | 90 |
| Ffinal_optimized_steady | combined | 9.4850 | 9.4817/9.8195 | F0_stock_steady | 2.277x | [2.225x, 2.288x] | 90 |
| P0_legacy_pair_output | model_lifecycle | 23.5451 | 23.5421/23.5550 | P0_legacy_pair_output | - | - | 90 |
| P1_canonical_torch | model_lifecycle | 23.0998 | 23.0945/23.1067 | P1_canonical_torch | - | - | 90 |

The frozen F0 result is collected in a separate process and package environment. Compiler, custom-kernel, and scope-lifecycle stages are not conflated.
