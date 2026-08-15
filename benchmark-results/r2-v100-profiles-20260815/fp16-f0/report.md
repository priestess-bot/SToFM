# SToFM R2 ATen Profile

Stage: `f0`; precision: `fp16`.

| Operator | Calls | Self CUDA us | Self CPU us | Classification |
| --- | ---: | ---: | ---: | --- |
| aten::addmm | 28 | 8702.6 | 4381.3 | stock_flagos_aten |
| addmm_kernel | 28 | 8702.6 | 0.0 | unclassified |
| aten::div | 2 | 2806.6 | 33.9 | torch_reference_gaussian |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > const&)::{lambda(int)#1}) | 2 | 2806.6 | 0.0 | unclassified |
| aten::mul | 2 | 1370.4 | 36.3 | torch_reference_gaussian |
| void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>) | 2 | 1370.4 | 0.0 | unclassified |
| aten::clamp_min | 1 | 1368.5 | 37.2 | torch_reference_gaussian |
| void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>) | 1 | 1368.5 | 0.0 | unclassified |
| aten::exp | 1 | 1368.3 | 27.1 | torch_reference_gaussian |
| void at::native::vectorized_elementwise_kernel<4, at::native::exp_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::exp_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>) | 1 | 1368.3 | 0.0 | unclassified |
| aten::pow | 1 | 1367.7 | 48.5 | torch_reference_gaussian |
| void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase&, float)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase&, float)::{lambda(float)#1}, std::array<char*, 2ul>) | 1 | 1367.7 | 0.0 | unclassified |
| aten::copy_ | 26 | 1115.3 | 309.6 | unclassified |
| aten::sub | 1 | 908.8 | 21.0 | torch_reference_gaussian |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&)::{lambda(int)#1}) | 1 | 908.8 | 0.0 | unclassified |
| Memcpy DtoD (Device -> Device) | 11 | 596.2 | 0.0 | unclassified |
| aten::masked_fill_ | 8 | 507.2 | 127.3 | unclassified |
| aten::bmm | 8 | 400.0 | 1106.8 | stock_flagos_aten |
| bmm_kernel | 8 | 400.0 | 0.0 | unclassified |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}) | 1 | 395.6 | 0.0 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<c10::Half>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::Half>, std::array<char*, 3ul>) | 13 | 291.8 | 0.0 | unclassified |
| aten::add_ | 5 | 267.6 | 70.3 | unclassified |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(c10::Half, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(c10::Half, bool)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(c10::Half, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(c10::Half, bool)#1} const&)::{lambda(int)#1}) | 4 | 247.6 | 0.0 | unclassified |
| aten::_softmax | 4 | 211.1 | 498.0 | stock_flagos_aten |
| softmax_kernel_inner | 4 | 211.1 | 0.0 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(c10::Half, bool)#1}, std::array<char*, 3ul> >(int, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(c10::Half, bool)#1}, std::array<char*, 3ul>) | 3 | 168.3 | 0.0 | unclassified |
| aten::eq | 5 | 113.6 | 127.8 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<c10::Half, c10::Half, bool, at::native::(anonymous namespace)::CompareEqFunctor<c10::Half> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<c10::Half, c10::Half, bool, at::native::(anonymous namespace)::CompareEqFunctor<c10::Half> >, std::array<char*, 2ul>) | 4 | 110.7 | 0.0 | unclassified |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&)::{lambda(int)#1}) | 1 | 91.3 | 0.0 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::float16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::float16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) | 1 | 65.8 | 0.0 | unclassified |
| aten::native_layer_norm | 8 | 60.5 | 209.4 | torch_retained_candidate_rejected |
| void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c10::Half, float>(int, float, c10::Half const*, c10::Half const*, c10::Half const*, float*, float*, c10::Half*) | 8 | 60.5 | 0.0 | torch_retained_candidate_rejected |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> >(int, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1>) | 9 | 36.8 | 0.0 | unclassified |
| aten::add | 9 | 26.2 | 198.7 | torch_reference_gaussian |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#10}::operator()() const::{lambda(c10::Half)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#10}::operator()() const::{lambda(c10::Half)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#10}::operator()() const::{lambda(c10::Half)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#10}::operator()() const::{lambda(c10::Half)#1} const&)::{lambda(int)#1}) | 4 | 20.9 | 0.0 | unclassified |
| aten::gelu | 4 | 11.9 | 80.4 | torch_retained |
| void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#3}::operator()() const::{lambda(c10::Half)#1}, std::array<char*, 2ul> >(int, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#3}::operator()() const::{lambda(c10::Half)#1}, std::array<char*, 2ul>) | 4 | 11.9 | 0.0 | unclassified |
| aten::mul_ | 4 | 10.4 | 65.8 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<c10::Half, c10::Half, c10::Half, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<c10::Half, c10::Half, c10::Half, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>) | 4 | 10.4 | 0.0 | unclassified |
| aten::index_select | 1 | 7.0 | 48.4 | unclassified |
| void at::native::(anonymous namespace)::indexSelectLargeIndex<c10::Half, long, unsigned int, 2, 2, -2, true>(at::cuda::detail::TensorInfo<c10::Half, unsigned int>, at::cuda::detail::TensorInfo<c10::Half const, unsigned int>, at::cuda::detail::TensorInfo<long const, unsigned int>, int, int, unsigned int, unsigned int, long) | 1 | 7.0 | 0.0 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<long, long, bool, at::native::(anonymous namespace)::CompareEqFunctor<long> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<long, long, bool, at::native::(anonymous namespace)::CompareEqFunctor<long> >, std::array<char*, 2ul>) | 1 | 2.9 | 0.0 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<float>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<float>, std::array<char*, 2ul>) | 1 | 2.0 | 0.0 | unclassified |
| aten::abs | 2 | 1.9 | 22.8 | torch_reference_gaussian |
| void at::native::vectorized_elementwise_kernel<4, at::native::AbsFunctor<float>, std::array<char*, 2ul> >(int, at::native::AbsFunctor<float>, std::array<char*, 2ul>) | 1 | 1.9 | 0.0 | unclassified |
| cudaDeviceSynchronize | 1 | 0.0 | 6088.7 | unclassified |
| aten::empty | 75 | 0.0 | 3614.8 | unclassified |
| cudaLaunchKernel | 68 | 0.0 | 990.4 | unclassified |
| cuLaunchKernel | 40 | 0.0 | 582.4 | unclassified |
| aten::linear | 28 | 0.0 | 524.7 | unclassified |
| cudaMemcpyAsync | 11 | 0.0 | 244.1 | unclassified |
| aten::transpose | 50 | 0.0 | 216.7 | unclassified |
| aten::empty_strided | 17 | 0.0 | 209.8 | unclassified |
| aten::view | 107 | 0.0 | 195.1 | unclassified |
| aten::expand | 42 | 0.0 | 182.3 | unclassified |
| aten::as_strided | 103 | 0.0 | 175.6 | unclassified |
| aten::masked_fill | 8 | 0.0 | 162.8 | unclassified |
| aten::t | 28 | 0.0 | 97.3 | unclassified |
| aten::_to_copy | 10 | 0.0 | 87.4 | unclassified |
| aten::clone | 16 | 0.0 | 86.0 | unclassified |
| aten::empty_like | 17 | 0.0 | 74.5 | unclassified |
| aten::broadcast_to | 28 | 0.0 | 69.5 | unclassified |
| aten::reshape | 29 | 0.0 | 69.0 | unclassified |
| aten::layer_norm | 8 | 0.0 | 58.6 | torch_retained_candidate_rejected |
| aten::unsqueeze | 10 | 0.0 | 31.8 | unclassified |
| aten::to | 19 | 0.0 | 25.3 | unclassified |
| aten::resize_ | 2 | 0.0 | 21.3 | unclassified |
| aten::softmax | 4 | 0.0 | 18.1 | stock_flagos_aten |
| aten::dropout | 17 | 0.0 | 17.3 | unclassified |
| aten::embedding | 1 | 0.0 | 15.1 | unclassified |
| aten::type_as | 4 | 0.0 | 12.3 | unclassified |
| aten::contiguous | 5 | 0.0 | 9.0 | unclassified |
| aten::relu | 1 | 0.0 | 6.7 | torch_reference_gaussian |
| aten::permute | 1 | 0.0 | 4.5 | unclassified |
| aten::result_type | 1 | 0.0 | 2.6 | unclassified |
| aten::expand_as | 1 | 0.0 | 2.5 | unclassified |
| [memory] | 116 | 0.0 | 0.0 | unclassified |
