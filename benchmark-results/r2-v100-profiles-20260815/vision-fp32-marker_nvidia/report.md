# Vision FlagOS Inference R2 ATen Profile

Stage: `marker_nvidia`; precision: `fp32`.

This trace is qualitative evidence only; timed latency is reported by the isolated suite.

| Operator | Calls | Self CUDA us | Self CPU us | Classification |
| --- | ---: | ---: | ---: | --- |
| _marker_token_embed_kernel | 1 | 41.2 | 0.0 | nvidia_custom_marker_token_kernel |
| aten::copy_ | 2 | 6.9 | 41.9 | unclassified |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool)#1} const&)::{lambda(int)#1}) | 1 | 4.7 | 0.0 | unclassified |
| aten::masked_fill_ | 1 | 2.8 | 23.4 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long, bool)#1}, std::array<char*, 3ul> >(int, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long, bool)#1}, std::array<char*, 3ul>) | 1 | 2.8 | 0.0 | unclassified |
| Memcpy DtoD (Device -> Device) | 1 | 2.2 | 0.0 | unclassified |
| aten::empty | 2 | 0.0 | 2709.9 | unclassified |
| aten::empty_like | 3 | 0.0 | 43.1 | unclassified |
| cudaMemcpyAsync | 1 | 0.0 | 35.9 | unclassified |
| cudaLaunchKernel | 2 | 0.0 | 35.5 | unclassified |
| aten::slice | 2 | 0.0 | 25.1 | unclassified |
| aten::clone | 2 | 0.0 | 24.3 | unclassified |
| aten::masked_fill | 1 | 0.0 | 21.9 | torch_reference_padding |
| aten::reshape | 2 | 0.0 | 18.9 | unclassified |
| aten::empty_strided | 1 | 0.0 | 15.5 | unclassified |
| cuLaunchKernel | 1 | 0.0 | 15.3 | unclassified |
| cudaDeviceSynchronize | 1 | 0.0 | 9.6 | unclassified |
| aten::expand | 1 | 0.0 | 8.6 | unclassified |
| aten::unsqueeze | 1 | 0.0 | 7.2 | unclassified |
| aten::as_strided | 4 | 0.0 | 5.5 | unclassified |
| aten::view | 1 | 0.0 | 3.9 | unclassified |
| aten::_unsafe_view | 1 | 0.0 | 3.4 | unclassified |
| aten::to | 1 | 0.0 | 2.5 | unclassified |
| [memory] | 3 | 0.0 | 0.0 | unclassified |
