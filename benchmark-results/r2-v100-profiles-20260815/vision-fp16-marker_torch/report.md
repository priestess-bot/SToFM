# Vision FlagOS Inference R2 ATen Profile

Stage: `marker_torch`; precision: `fp16`.

This trace is qualitative evidence only; timed latency is reported by the isolated suite.

| Operator | Calls | Self CUDA us | Self CPU us | Classification |
| --- | ---: | ---: | ---: | --- |
| aten::add | 3 | 74.6 | 67.2 | torch_reference_or_input_assembly |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10::Half> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<c10::Half> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10::Half> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<c10::Half> const&)::{lambda(int)#1}) | 3 | 74.6 | 0.0 | unclassified |
| aten::masked_fill_ | 2 | 26.0 | 46.9 | unclassified |
| aten::copy_ | 3 | 25.0 | 56.6 | unclassified |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(c10::Half, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(c10::Half, bool)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(c10::Half, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(c10::Half, bool)#1} const&)::{lambda(int)#1}) | 1 | 23.3 | 0.0 | unclassified |
| Memcpy DtoD (Device -> Device) | 2 | 20.4 | 0.0 | unclassified |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool)#1} const&)::{lambda(int)#1}) | 1 | 4.5 | 0.0 | unclassified |
| aten::index_select | 1 | 3.8 | 46.9 | unclassified |
| void at::native::(anonymous namespace)::indexSelectLargeIndex<c10::Half, long, unsigned int, 2, 2, -2, true>(at::cuda::detail::TensorInfo<c10::Half, unsigned int>, at::cuda::detail::TensorInfo<c10::Half const, unsigned int>, at::cuda::detail::TensorInfo<long const, unsigned int>, int, int, unsigned int, unsigned int, long) | 1 | 3.8 | 0.0 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long, bool)#1}, std::array<char*, 3ul> >(int, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long, bool)#1}, std::array<char*, 3ul>) | 1 | 2.7 | 0.0 | unclassified |
| aten::empty | 4 | 0.0 | 2783.9 | unclassified |
| cudaLaunchKernel | 7 | 0.0 | 121.7 | unclassified |
| cudaMemcpyAsync | 2 | 0.0 | 74.2 | unclassified |
| aten::masked_fill | 2 | 0.0 | 60.8 | torch_reference_padding |
| aten::empty_like | 3 | 0.0 | 44.9 | unclassified |
| aten::clone | 3 | 0.0 | 26.4 | unclassified |
| aten::slice | 4 | 0.0 | 25.6 | unclassified |
| aten::view | 5 | 0.0 | 22.9 | unclassified |
| aten::unsqueeze | 4 | 0.0 | 20.4 | unclassified |
| aten::reshape | 3 | 0.0 | 20.0 | unclassified |
| aten::embedding | 1 | 0.0 | 19.1 | torch_reference_marker_lookup |
| aten::as_strided | 12 | 0.0 | 15.7 | unclassified |
| aten::expand | 4 | 0.0 | 15.4 | unclassified |
| aten::resize_ | 1 | 0.0 | 9.8 | unclassified |
| cudaDeviceSynchronize | 1 | 0.0 | 9.6 | unclassified |
| aten::to | 2 | 0.0 | 4.2 | unclassified |
| aten::_unsafe_view | 1 | 0.0 | 3.0 | unclassified |
| [memory] | 7 | 0.0 | 0.0 | unclassified |
