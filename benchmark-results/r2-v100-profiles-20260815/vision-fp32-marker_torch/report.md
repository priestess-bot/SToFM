# Vision FlagOS Inference R2 ATen Profile

Stage: `marker_torch`; precision: `fp32`.

This trace is qualitative evidence only; timed latency is reported by the isolated suite.

| Operator | Calls | Self CUDA us | Self CPU us | Classification |
| --- | ---: | ---: | ---: | --- |
| aten::add | 3 | 106.4 | 71.1 | torch_reference_or_input_assembly |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&)::{lambda(int)#1}) | 3 | 106.4 | 0.0 | unclassified |
| aten::copy_ | 3 | 41.9 | 49.5 | unclassified |
| Memcpy DtoD (Device -> Device) | 2 | 36.9 | 0.0 | unclassified |
| aten::masked_fill_ | 2 | 36.9 | 44.7 | unclassified |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&)::{lambda(int)#1}) | 1 | 34.2 | 0.0 | unclassified |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool)#1} const&)::{lambda(int)#1}) | 1 | 5.0 | 0.0 | unclassified |
| aten::index_select | 1 | 3.9 | 44.3 | unclassified |
| void at::native::(anonymous namespace)::indexSelectLargeIndex<float, long, unsigned int, 2, 2, -2, true>(at::cuda::detail::TensorInfo<float, unsigned int>, at::cuda::detail::TensorInfo<float const, unsigned int>, at::cuda::detail::TensorInfo<long const, unsigned int>, int, int, unsigned int, unsigned int, long) | 1 | 3.9 | 0.0 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long, bool)#1}, std::array<char*, 3ul> >(int, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long, bool)#1}, std::array<char*, 3ul>) | 1 | 2.7 | 0.0 | unclassified |
| aten::empty | 4 | 0.0 | 2609.9 | unclassified |
| cudaLaunchKernel | 7 | 0.0 | 101.8 | unclassified |
| cudaMemcpyAsync | 2 | 0.0 | 71.3 | unclassified |
| aten::masked_fill | 2 | 0.0 | 44.5 | torch_reference_padding |
| aten::empty_like | 3 | 0.0 | 43.6 | unclassified |
| aten::clone | 3 | 0.0 | 27.3 | unclassified |
| aten::slice | 4 | 0.0 | 22.5 | unclassified |
| aten::reshape | 3 | 0.0 | 20.9 | unclassified |
| aten::view | 5 | 0.0 | 20.8 | unclassified |
| aten::embedding | 1 | 0.0 | 19.3 | torch_reference_marker_lookup |
| aten::unsqueeze | 4 | 0.0 | 18.7 | unclassified |
| aten::expand | 4 | 0.0 | 14.7 | unclassified |
| aten::as_strided | 12 | 0.0 | 13.8 | unclassified |
| cudaDeviceSynchronize | 1 | 0.0 | 8.7 | unclassified |
| aten::resize_ | 1 | 0.0 | 8.3 | unclassified |
| aten::to | 2 | 0.0 | 3.5 | unclassified |
| aten::_unsafe_view | 1 | 0.0 | 2.1 | unclassified |
| [memory] | 7 | 0.0 | 0.0 | unclassified |
