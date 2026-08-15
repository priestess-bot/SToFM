# Vision FlagOS Inference R2 V100 Suite

Precision: `fp16`; independent trials: `3`.

| Stage | Gain kind | p50 median ms | p50 min/max ms | Baseline | Speedup | 95% bootstrap CI | Raw samples | Status |
| --- | --- | ---: | ---: | --- | ---: | --- | ---: | --- |
| V0_marker_token_torch | reference | 0.2974 | 0.2901/0.3165 | V0_marker_token_torch | - | - | 90 | measured |
| V0s_marker_token_stock_flagos | stock_aten_reference | 0.2714 | 0.2653/0.2734 | V0s_marker_token_stock_flagos | - | - | 90 | measured |
| V1_marker_token_nvidia | custom_kernel | 0.2459 | 0.2455/0.3244 | V0s_marker_token_stock_flagos | 1.076x | [1.049x, 1.095x] | 90 | measured |
| V2_swiglu_torch | reference | 0.0587 | 0.0568/0.0633 | V2_swiglu_torch | - | - | 90 | measured |
| V2s_swiglu_stock_flagos | stock_aten_reference | 0.0553 | 0.0547/0.0567 | V2s_swiglu_stock_flagos | - | - | 90 | measured |
| V3_swiglu_nvidia | existing_flaggems_kernel | 0.1300 | 0.1234/0.1492 | V2s_swiglu_stock_flagos | 0.421x | [0.403x, 0.439x] | 90 | measured |
| V4_residual_layer_norm_torch | reference | 0.0637 | 0.0564/0.0691 | V4_residual_layer_norm_torch | - | - | 90 | measured |
| V4s_residual_layer_norm_stock_flagos | stock_aten_reference | 0.0594 | 0.0548/0.0636 | V4s_residual_layer_norm_stock_flagos | - | - | 90 | measured |
| V5_residual_layer_norm_rejected | rejected | - | - | V4s_residual_layer_norm_stock_flagos | - | - | 0 | rejected: The existing FlagGems skip-LayerNorm candidate lost on the V100 R1 shape and has no verified backward contract. |
