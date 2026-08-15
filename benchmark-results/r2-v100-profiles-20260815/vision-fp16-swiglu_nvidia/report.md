# Vision FlagOS Inference R2 ATen Profile

Stage: `swiglu_nvidia`; precision: `fp16`.

This trace is qualitative evidence only; timed latency is reported by the isolated suite.

| Operator | Calls | Self CUDA us | Self CPU us | Classification |
| --- | ---: | ---: | ---: | --- |
| swiglu_kernel | 1 | 16.3 | 0.0 | flaggems_existing_swiglu_kernel_rejected |
| aten::empty | 1 | 0.0 | 3137.4 | unclassified |
| cuLaunchKernel | 1 | 0.0 | 34.0 | unclassified |
| aten::view | 2 | 0.0 | 23.0 | unclassified |
| cudaDeviceSynchronize | 1 | 0.0 | 11.1 | unclassified |
| [memory] | 1 | 0.0 | 0.0 | unclassified |
