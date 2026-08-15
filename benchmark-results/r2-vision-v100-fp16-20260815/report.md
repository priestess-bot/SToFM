# Vision FlagOS Inference R2 V100 Suite

Precision: `fp16`; independent trials: `3`.

| Stage | Gain kind | p50 median ms | p50 min/max ms | Baseline | Speedup | 95% bootstrap CI | Raw samples | Status |
| --- | --- | ---: | ---: | --- | ---: | --- | ---: | --- |
| V0_marker_token_torch | reference | 0.2897 | 0.2876/0.3130 | V0_marker_token_torch | - | - | 90 | measured |
| V1_marker_token_nvidia | custom_kernel | 0.2651 | 0.2585/0.2850 | V0_marker_token_torch | 1.106x | [1.075x, 1.138x] | 90 | measured |
| V2_swiglu_torch | reference | 0.0600 | 0.0598/0.0628 | V2_swiglu_torch | - | - | 90 | measured |
| V3_swiglu_nvidia | existing_flaggems_kernel | 0.1373 | 0.1349/0.1391 | V2_swiglu_torch | 0.438x | [0.430x, 0.450x] | 90 | measured |
| V4_residual_layer_norm_torch | reference | 0.0609 | 0.0565/0.0650 | V4_residual_layer_norm_torch | - | - | 90 | measured |
| V5_residual_layer_norm_rejected | rejected | - | - | V4_residual_layer_norm_torch | - | - | 0 | rejected: The existing FlagGems skip-LayerNorm candidate lost on the V100 R1 shape and has no verified backward contract. |
