# Vision FlagOS Inference R2 V100 Suite

Precision: `fp32`; independent trials: `3`.

| Stage | Gain kind | p50 median ms | p50 min/max ms | Baseline | Speedup | 95% bootstrap CI | Raw samples | Status |
| --- | --- | ---: | ---: | --- | ---: | --- | ---: | --- |
| V0_marker_token_torch | reference | 0.2867 | 0.2803/0.2972 | V0_marker_token_torch | - | - | 90 | measured |
| V1_marker_token_nvidia | custom_kernel | 0.2449 | 0.2375/0.2587 | V0_marker_token_torch | 1.154x | [1.137x, 1.174x] | 90 | measured |
| V2_swiglu_torch | reference | 0.0592 | 0.0586/0.0630 | V2_swiglu_torch | - | - | 90 | measured |
| V3_swiglu_nvidia | existing_flaggems_kernel | 0.1227 | 0.1187/0.1412 | V2_swiglu_torch | 0.484x | [0.472x, 0.497x] | 90 | measured |
| V4_residual_layer_norm_torch | reference | 0.0626 | 0.0581/0.0710 | V4_residual_layer_norm_torch | - | - | 90 | measured |
| V5_residual_layer_norm_rejected | rejected | - | - | V4_residual_layer_norm_torch | - | - | 0 | rejected: The existing FlagGems skip-LayerNorm candidate lost on the V100 R1 shape and has no verified backward contract. |
