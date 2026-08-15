# Vision FlagOS Inference R2 V100 Suite

Precision: `fp32`; independent trials: `3`.

| Stage | Gain kind | p50 median ms | p50 min/max ms | Baseline | Speedup | 95% bootstrap CI | Raw samples | Status |
| --- | --- | ---: | ---: | --- | ---: | --- | ---: | --- |
| V0_marker_token_torch | reference | 0.2828 | 0.2765/0.2926 | V0_marker_token_torch | - | - | 90 | measured |
| V0s_marker_token_stock_flagos | stock_aten_reference | 0.2785 | 0.2744/0.2801 | V0s_marker_token_stock_flagos | - | - | 90 | measured |
| V1_marker_token_nvidia | custom_kernel | 0.2512 | 0.2466/0.2821 | V0s_marker_token_stock_flagos | 1.077x | [1.052x, 1.108x] | 90 | measured |
| V2_swiglu_torch | reference | 0.0585 | 0.0578/0.0620 | V2_swiglu_torch | - | - | 90 | measured |
| V2s_swiglu_stock_flagos | stock_aten_reference | 0.0543 | 0.0530/0.0547 | V2s_swiglu_stock_flagos | - | - | 90 | measured |
| V3_swiglu_nvidia | existing_flaggems_kernel | 0.1226 | 0.1194/0.1298 | V2s_swiglu_stock_flagos | 0.439x | [0.431x, 0.449x] | 90 | measured |
| V4_residual_layer_norm_torch | reference | 0.0573 | 0.0573/0.0640 | V4_residual_layer_norm_torch | - | - | 90 | measured |
| V4s_residual_layer_norm_stock_flagos | stock_aten_reference | 0.0526 | 0.0519/0.0546 | V4s_residual_layer_norm_stock_flagos | - | - | 90 | measured |
| V5_residual_layer_norm_rejected | rejected | - | - | V4s_residual_layer_norm_stock_flagos | - | - | 0 | rejected: The existing FlagGems skip-LayerNorm candidate lost on the V100 R1 shape and has no verified backward contract. |
