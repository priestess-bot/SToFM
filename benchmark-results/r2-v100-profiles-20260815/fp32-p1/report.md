# SToFM R2 ATen Profile

Stage: `p1`; precision: `fp32`.

| Operator | Calls | Self CUDA us | Self CPU us | Classification |
| --- | ---: | ---: | ---: | --- |
| aten::addmm | 28 | 9637.2 | 3721.6 | torch_reference_aten |
| volta_sgemm_128x64_tn | 1 | 7514.2 | 0.0 | unclassified |
| aten::div | 2 | 2807.0 | 37.1 | torch_reference_gaussian |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > const&)::{lambda(int)#1}) | 2 | 2807.0 | 0.0 | unclassified |
| aten::_softmax | 4 | 1705.5 | 56.8 | torch_reference_aten |
| void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float, at::native::(anonymous namespace)::SoftMaxForwardEpilogue>(float*, float const*, int) | 4 | 1705.5 | 0.0 | unclassified |
| aten::copy_ | 16 | 1438.3 | 193.1 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>) | 6 | 1385.7 | 0.0 | unclassified |
| aten::mul | 2 | 1369.1 | 29.9 | torch_reference_gaussian |
| aten::clamp_min | 1 | 1368.3 | 22.1 | torch_reference_gaussian |
| void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>) | 1 | 1368.3 | 0.0 | unclassified |
| aten::exp | 1 | 1368.2 | 19.2 | torch_reference_gaussian |
| void at::native::vectorized_elementwise_kernel<4, at::native::exp_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::exp_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>) | 1 | 1368.2 | 0.0 | unclassified |
| aten::pow | 1 | 1367.0 | 29.3 | torch_reference_gaussian |
| void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase&, float)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase&, float)::{lambda(float)#1}, std::array<char*, 2ul>) | 1 | 1367.0 | 0.0 | unclassified |
| volta_sgemm_32x128_tn | 1 | 1325.6 | 0.0 | unclassified |
| aten::bmm | 8 | 1269.9 | 256.0 | torch_reference_aten |
| Memcpy DtoD (Device -> Device) | 11 | 1021.8 | 0.0 | unclassified |
| aten::sub | 1 | 909.5 | 20.6 | torch_reference_gaussian |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&)::{lambda(int)#1}) | 1 | 909.5 | 0.0 | unclassified |
| volta_sgemm_64x64_nn | 4 | 886.5 | 0.0 | unclassified |
| aten::masked_fill_ | 8 | 739.3 | 122.0 | unclassified |
| volta_sgemm_64x32_sliced1x4_tn | 25 | 704.0 | 0.0 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>) | 13 | 536.5 | 0.0 | unclassified |
| aten::add_ | 5 | 512.5 | 63.0 | unclassified |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&)::{lambda(int)#1}) | 5 | 446.7 | 0.0 | unclassified |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}) | 6 | 427.4 | 0.0 | unclassified |
| volta_sgemm_64x64_tn | 4 | 383.5 | 0.0 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}, std::array<char*, 3ul> >(int, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}, std::array<char*, 3ul>) | 3 | 292.6 | 0.0 | unclassified |
| aten::eq | 5 | 185.0 | 147.4 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, float, bool, at::native::(anonymous namespace)::CompareEqFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<float, float, bool, at::native::(anonymous namespace)::CompareEqFunctor<float> >, std::array<char*, 2ul>) | 4 | 182.0 | 0.0 | unclassified |
| void gemmk1_kernel<int, float, 256, 5, true, false, false, false, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float, 0>(cublasGemmk1Params<float, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float, biasType<cublasGemvTensorStridedBatched<float>::value_type, float>::type>) | 1 | 82.5 | 0.0 | unclassified |
| aten::native_layer_norm | 8 | 70.9 | 205.9 | torch_retained_candidate_rejected |
| void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<float, float>(int, float, float const*, float const*, float const*, float*, float*, float*) | 8 | 70.9 | 0.0 | torch_retained_candidate_rejected |
| aten::add | 9 | 30.9 | 176.2 | torch_reference_gaussian |
| aten::mul_ | 4 | 16.6 | 59.8 | unclassified |
| aten::gelu | 4 | 13.8 | 70.9 | torch_retained |
| void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>) | 4 | 13.8 | 0.0 | unclassified |
| aten::index_select | 1 | 7.1 | 45.6 | unclassified |
| void at::native::(anonymous namespace)::indexSelectLargeIndex<float, long, unsigned int, 2, 2, -2, true>(at::cuda::detail::TensorInfo<float, unsigned int>, at::cuda::detail::TensorInfo<float const, unsigned int>, at::cuda::detail::TensorInfo<long const, unsigned int>, int, int, unsigned int, unsigned int, long) | 1 | 7.1 | 0.0 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<float>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<float>, std::array<char*, 2ul>) | 1 | 6.9 | 0.0 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<long, long, bool, at::native::(anonymous namespace)::CompareEqFunctor<long> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<long, long, bool, at::native::(anonymous namespace)::CompareEqFunctor<long> >, std::array<char*, 2ul>) | 1 | 2.9 | 0.0 | unclassified |
| aten::abs | 2 | 2.2 | 30.0 | torch_reference_gaussian |
| void at::native::vectorized_elementwise_kernel<4, at::native::AbsFunctor<float>, std::array<char*, 2ul> >(int, at::native::AbsFunctor<float>, std::array<char*, 2ul>) | 1 | 2.2 | 0.0 | unclassified |
| cudaDeviceSynchronize | 1 | 0.0 | 16063.6 | unclassified |
| cudaLaunchKernel | 99 | 0.0 | 1320.4 | unclassified |
| aten::empty | 66 | 0.0 | 502.5 | unclassified |
| aten::linear | 28 | 0.0 | 429.7 | unclassified |
| cudaMemcpyAsync | 11 | 0.0 | 264.8 | unclassified |
| aten::expand | 15 | 0.0 | 225.2 | unclassified |
| aten::t | 28 | 0.0 | 191.8 | unclassified |
| aten::view | 107 | 0.0 | 170.8 | unclassified |
| aten::transpose | 50 | 0.0 | 165.4 | unclassified |
| aten::as_strided | 76 | 0.0 | 154.0 | unclassified |
| aten::clone | 16 | 0.0 | 101.8 | unclassified |
| aten::masked_fill | 8 | 0.0 | 100.6 | unclassified |
| aten::empty_like | 13 | 0.0 | 64.8 | unclassified |
| aten::reshape | 29 | 0.0 | 59.9 | unclassified |
| aten::layer_norm | 8 | 0.0 | 47.5 | torch_retained_candidate_rejected |
| aten::empty_strided | 3 | 0.0 | 29.3 | unclassified |
| aten::unsqueeze | 10 | 0.0 | 29.2 | unclassified |
| aten::dropout | 17 | 0.0 | 16.3 | unclassified |
| aten::resize_ | 2 | 0.0 | 16.3 | unclassified |
| aten::embedding | 1 | 0.0 | 14.7 | unclassified |
| aten::softmax | 4 | 0.0 | 12.6 | torch_reference_aten |
| cudaOccupancyMaxActiveBlocksPerMultiprocessor | 1 | 0.0 | 11.5 | unclassified |
| aten::type_as | 4 | 0.0 | 11.3 | unclassified |
| aten::contiguous | 5 | 0.0 | 9.5 | unclassified |
| aten::to | 19 | 0.0 | 7.8 | unclassified |
| aten::relu | 1 | 0.0 | 7.3 | torch_reference_gaussian |
| aten::expand_as | 1 | 0.0 | 3.7 | unclassified |
| aten::permute | 1 | 0.0 | 3.3 | unclassified |
| aten::result_type | 1 | 0.0 | 1.0 | unclassified |
| [memory] | 106 | 0.0 | 0.0 | unclassified |
