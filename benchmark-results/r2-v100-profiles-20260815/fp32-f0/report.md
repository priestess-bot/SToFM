# SToFM R2 ATen Profile

Stage: `f0`; precision: `fp32`.

| Operator | Calls | Self CUDA us | Self CPU us | Classification |
| --- | ---: | ---: | ---: | --- |
| aten::addmm | 28 | 9678.8 | 4413.6 | stock_flagos_aten |
| addmm_kernel | 28 | 9678.8 | 0.0 | unclassified |
| aten::div | 2 | 2807.1 | 38.0 | torch_reference_gaussian |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > const&)::{lambda(int)#1}) | 2 | 2807.1 | 0.0 | unclassified |
| aten::copy_ | 16 | 1441.6 | 165.1 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>) | 6 | 1380.7 | 0.0 | unclassified |
| aten::mul | 2 | 1369.4 | 41.6 | torch_reference_gaussian |
| aten::exp | 1 | 1367.7 | 30.3 | torch_reference_gaussian |
| void at::native::vectorized_elementwise_kernel<4, at::native::exp_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::exp_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>) | 1 | 1367.7 | 0.0 | unclassified |
| aten::clamp_min | 1 | 1367.3 | 29.9 | torch_reference_gaussian |
| void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>) | 1 | 1367.3 | 0.0 | unclassified |
| aten::pow | 1 | 1367.3 | 27.7 | torch_reference_gaussian |
| void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase&, float)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase&, float)::{lambda(float)#1}, std::array<char*, 2ul>) | 1 | 1367.3 | 0.0 | unclassified |
| Memcpy DtoD (Device -> Device) | 11 | 1021.8 | 0.0 | unclassified |
| aten::bmm | 8 | 934.6 | 989.3 | stock_flagos_aten |
| bmm_kernel | 8 | 934.6 | 0.0 | unclassified |
| aten::sub | 1 | 909.1 | 21.3 | torch_reference_gaussian |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&)::{lambda(int)#1}) | 1 | 909.1 | 0.0 | unclassified |
| aten::masked_fill_ | 8 | 740.3 | 111.0 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>) | 13 | 536.6 | 0.0 | unclassified |
| aten::add_ | 5 | 512.8 | 68.1 | unclassified |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&)::{lambda(int)#1}) | 5 | 447.4 | 0.0 | unclassified |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}) | 5 | 419.8 | 0.0 | unclassified |
| aten::_softmax | 4 | 395.0 | 409.7 | stock_flagos_aten |
| softmax_kernel_inner | 4 | 395.0 | 0.0 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}, std::array<char*, 3ul> >(int, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}, std::array<char*, 3ul>) | 3 | 292.9 | 0.0 | unclassified |
| aten::eq | 5 | 185.2 | 139.0 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, float, bool, at::native::(anonymous namespace)::CompareEqFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<float, float, bool, at::native::(anonymous namespace)::CompareEqFunctor<float> >, std::array<char*, 2ul>) | 4 | 182.2 | 0.0 | unclassified |
| aten::native_layer_norm | 8 | 76.7 | 194.2 | torch_retained_candidate_rejected |
| void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<float, float>(int, float, float const*, float const*, float const*, float*, float*, float*) | 8 | 76.7 | 0.0 | torch_retained_candidate_rejected |
| aten::add | 9 | 26.0 | 198.5 | torch_reference_gaussian |
| aten::gelu | 4 | 13.2 | 77.7 | torch_retained |
| void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>) | 4 | 13.2 | 0.0 | unclassified |
| aten::mul_ | 4 | 11.3 | 64.8 | unclassified |
| aten::index_select | 1 | 6.7 | 51.0 | unclassified |
| void at::native::(anonymous namespace)::indexSelectLargeIndex<float, long, unsigned int, 2, 2, -2, true>(at::cuda::detail::TensorInfo<float, unsigned int>, at::cuda::detail::TensorInfo<float const, unsigned int>, at::cuda::detail::TensorInfo<long const, unsigned int>, int, int, unsigned int, unsigned int, long) | 1 | 6.7 | 0.0 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<long, long, bool, at::native::(anonymous namespace)::CompareEqFunctor<long> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<long, long, bool, at::native::(anonymous namespace)::CompareEqFunctor<long> >, std::array<char*, 2ul>) | 1 | 3.0 | 0.0 | unclassified |
| aten::abs | 2 | 2.4 | 41.2 | torch_reference_gaussian |
| void at::native::vectorized_elementwise_kernel<4, at::native::AbsFunctor<float>, std::array<char*, 2ul> >(int, at::native::AbsFunctor<float>, std::array<char*, 2ul>) | 1 | 2.4 | 0.0 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<float>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<float>, std::array<char*, 2ul>) | 1 | 2.2 | 0.0 | unclassified |
| cudaDeviceSynchronize | 1 | 0.0 | 9492.7 | unclassified |
| aten::empty | 75 | 0.0 | 3535.5 | unclassified |
| cudaLaunchKernel | 58 | 0.0 | 842.8 | unclassified |
| cuLaunchKernel | 40 | 0.0 | 625.4 | unclassified |
| aten::linear | 28 | 0.0 | 483.3 | unclassified |
| cudaMemcpyAsync | 11 | 0.0 | 227.9 | unclassified |
| aten::expand | 42 | 0.0 | 223.8 | unclassified |
| aten::view | 107 | 0.0 | 196.0 | unclassified |
| aten::as_strided | 103 | 0.0 | 172.8 | unclassified |
| aten::transpose | 50 | 0.0 | 165.6 | unclassified |
| aten::t | 28 | 0.0 | 104.1 | unclassified |
| aten::clone | 16 | 0.0 | 80.5 | unclassified |
| aten::empty_strided | 7 | 0.0 | 78.5 | unclassified |
| aten::masked_fill | 8 | 0.0 | 78.3 | unclassified |
| aten::broadcast_to | 28 | 0.0 | 72.9 | unclassified |
| aten::reshape | 29 | 0.0 | 72.5 | unclassified |
| aten::empty_like | 17 | 0.0 | 61.6 | unclassified |
| aten::layer_norm | 8 | 0.0 | 55.0 | torch_retained_candidate_rejected |
| aten::unsqueeze | 10 | 0.0 | 40.4 | unclassified |
| aten::embedding | 1 | 0.0 | 19.1 | unclassified |
| aten::softmax | 4 | 0.0 | 16.4 | stock_flagos_aten |
| aten::resize_ | 2 | 0.0 | 16.3 | unclassified |
| aten::dropout | 17 | 0.0 | 15.7 | unclassified |
| aten::permute | 1 | 0.0 | 14.8 | unclassified |
| aten::type_as | 4 | 0.0 | 10.5 | unclassified |
| aten::contiguous | 5 | 0.0 | 8.8 | unclassified |
| aten::to | 19 | 0.0 | 8.2 | unclassified |
| aten::relu | 1 | 0.0 | 7.4 | torch_reference_gaussian |
| aten::expand_as | 1 | 0.0 | 2.0 | unclassified |
| aten::result_type | 1 | 0.0 | 1.2 | unclassified |
| [memory] | 106 | 0.0 | 0.0 | unclassified |
