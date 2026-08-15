# Vision FlagOS Inference R2 ATen Profile

Stage: `swiglu_nvidia`; precision: `fp32`.

This trace is qualitative evidence only; timed latency is reported by the isolated suite.

| Operator | Calls | Self CUDA us | Self CPU us | Classification |
| --- | ---: | ---: | ---: | --- |
| swiglu_kernel | 1 | 24.6 | 0.0 | flaggems_existing_swiglu_kernel_rejected |
| aten::empty | 1 | 0.0 | 2586.7 | unclassified |
| cuLaunchKernel | 1 | 0.0 | 30.1 | unclassified |
| aten::view | 2 | 0.0 | 22.9 | unclassified |
| cudaDeviceSynchronize | 1 | 0.0 | 8.8 | unclassified |
| [memory] | 1 | 0.0 | 0.0 | unclassified |
