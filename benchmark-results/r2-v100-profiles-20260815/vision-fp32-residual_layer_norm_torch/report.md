# Vision FlagOS Inference R2 ATen Profile

Stage: `residual_layer_norm_torch`; precision: `fp32`.

This trace is qualitative evidence only; timed latency is reported by the isolated suite.

| Operator | Calls | Self CUDA us | Self CPU us | Classification |
| --- | ---: | ---: | ---: | --- |
| aten::native_layer_norm | 1 | 8.5 | 75.3 | torch_retained_candidate_rejected |
| void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<float, float>(int, float, float const*, float const*, float const*, float*, float*, float*) | 1 | 8.5 | 0.0 | torch_retained_candidate_rejected |
| aten::add | 1 | 5.6 | 2839.1 | torch_reference_or_input_assembly |
| void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>) | 1 | 5.6 | 0.0 | unclassified |
| cudaLaunchKernel | 2 | 0.0 | 54.5 | unclassified |
| aten::empty | 3 | 0.0 | 28.6 | unclassified |
| aten::layer_norm | 1 | 0.0 | 16.9 | unclassified |
| cudaDeviceSynchronize | 1 | 0.0 | 10.9 | unclassified |
| aten::view | 2 | 0.0 | 6.7 | unclassified |
| [memory] | 2 | 0.0 | 0.0 | unclassified |
