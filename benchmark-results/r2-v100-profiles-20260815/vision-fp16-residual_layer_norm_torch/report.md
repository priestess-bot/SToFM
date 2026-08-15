# Vision FlagOS Inference R2 ATen Profile

Stage: `residual_layer_norm_torch`; precision: `fp16`.

This trace is qualitative evidence only; timed latency is reported by the isolated suite.

| Operator | Calls | Self CUDA us | Self CPU us | Classification |
| --- | ---: | ---: | ---: | --- |
| aten::native_layer_norm | 1 | 8.0 | 62.7 | torch_retained_candidate_rejected |
| void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c10::Half, float>(int, float, c10::Half const*, c10::Half const*, c10::Half const*, float*, float*, c10::Half*) | 1 | 8.0 | 0.0 | torch_retained_candidate_rejected |
| aten::add | 1 | 4.2 | 2984.6 | torch_reference_or_input_assembly |
| void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<c10::Half>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::Half>, std::array<char*, 3ul>) | 1 | 4.2 | 0.0 | unclassified |
| cudaLaunchKernel | 2 | 0.0 | 53.2 | unclassified |
| aten::empty | 3 | 0.0 | 23.8 | unclassified |
| aten::layer_norm | 1 | 0.0 | 17.2 | unclassified |
| cudaDeviceSynchronize | 1 | 0.0 | 8.4 | unclassified |
| aten::view | 2 | 0.0 | 5.9 | unclassified |
| [memory] | 2 | 0.0 | 0.0 | unclassified |
