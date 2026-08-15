# Vision FlagOS Inference R2 ATen Profile

Stage: `marker_nvidia`; precision: `fp16`.

This trace is qualitative evidence only; timed latency is reported by the isolated suite.

| Operator | Calls | Self CUDA us | Self CPU us | Classification |
| --- | ---: | ---: | ---: | --- |
| _marker_token_embed_kernel | 1 | 35.1 | 0.0 | nvidia_custom_marker_token_kernel |
| aten::copy_ | 2 | 6.9 | 50.3 | unclassified |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool)#1} const&)::{lambda(int)#1}) | 1 | 4.7 | 0.0 | unclassified |
| aten::masked_fill_ | 1 | 2.7 | 24.7 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long, bool)#1}, std::array<char*, 3ul> >(int, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long, bool)#1}, std::array<char*, 3ul>) | 1 | 2.7 | 0.0 | unclassified |
| Memcpy DtoD (Device -> Device) | 1 | 2.2 | 0.0 | unclassified |
| aten::empty | 2 | 0.0 | 2708.0 | unclassified |
| cudaLaunchKernel | 2 | 0.0 | 53.7 | unclassified |
| aten::empty_like | 3 | 0.0 | 47.0 | unclassified |
| cudaMemcpyAsync | 1 | 0.0 | 36.8 | unclassified |
| aten::slice | 2 | 0.0 | 28.0 | unclassified |
| aten::clone | 2 | 0.0 | 27.0 | unclassified |
| aten::reshape | 2 | 0.0 | 25.3 | unclassified |
| aten::masked_fill | 1 | 0.0 | 23.9 | torch_reference_padding |
| cuLaunchKernel | 1 | 0.0 | 17.3 | unclassified |
| aten::empty_strided | 1 | 0.0 | 16.6 | unclassified |
| aten::unsqueeze | 1 | 0.0 | 12.0 | unclassified |
| aten::expand | 1 | 0.0 | 9.7 | unclassified |
| aten::as_strided | 4 | 0.0 | 9.6 | unclassified |
| cudaDeviceSynchronize | 1 | 0.0 | 7.6 | unclassified |
| aten::view | 1 | 0.0 | 3.8 | unclassified |
| aten::to | 1 | 0.0 | 2.6 | unclassified |
| aten::_unsafe_view | 1 | 0.0 | 2.3 | unclassified |
| [memory] | 3 | 0.0 | 0.0 | unclassified |
