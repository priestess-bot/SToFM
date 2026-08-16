# MTT S4000 MUSA Formal Optimization Evidence

Independent process trials: `5`. All timings use MUSA device events inside `torch.inference_mode()`. 

## Primary operator comparison (N=1050, FP32)

| Operator | Pure PyTorch (ms) | Initial FlagOS MUSA (ms) | Optimized FlagOS MUSA (ms) | Optimized vs PyTorch | Optimized vs initial |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gaussian pair-bias | 22.9559 | 2652.8751 | 9.5671 | 2.399x | 277.292x |
| Pair-attention score/softmax/context | 0.7431 | 9.6459 | 0.6624 | 1.123x | 14.561x |

## End-to-end SToFM (N=1050, four layers, FP32)

| Execution path | p50 (ms) | Speedup over pure PyTorch | Max absolute error |
| --- | ---: | ---: | ---: |
| Pure PyTorch inference | 33.2088 | 1.000x | 0 |
| FlagOS Gaussian optimization only | 19.9878 | 1.661x | 1.19209e-06 |
| FlagOS pair-attention optimization only | 29.5774 | 1.123x | 9.53674e-07 |
| Both optimized FlagOS operators | 16.3370 | 2.032x | 1.19209e-06 |
| Both operators plus generic FlagOS ATen dispatch | unavailable | unavailable | unavailable |

The frozen upstream FlagOS availability probe is preserved separately. An unavailable backend is never replaced with a different implementation or a fabricated timing.
