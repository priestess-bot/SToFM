# SToFM R2 ATen Profile

Stage: `p1`; precision: `fp16`.

| Operator | Calls | Self CUDA us | Self CPU us | Classification |
| --- | ---: | ---: | ---: | --- |
| aten::addmm | 28 | 9269.2 | 3982.0 | torch_reference_aten |
| volta_sgemm_128x64_tn | 1 | 7657.2 | 0.0 | unclassified |
| aten::div | 2 | 2806.8 | 32.5 | torch_reference_gaussian |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > const&)::{lambda(int)#1}) | 2 | 2806.8 | 0.0 | unclassified |
| aten::_softmax | 4 | 1643.5 | 52.8 | torch_reference_aten |
| void at::native::(anonymous namespace)::cunn_SoftMaxForward<8, c10::Half, float, c10::Half, at::native::(anonymous namespace)::SoftMaxForwardEpilogue>(c10::Half*, c10::Half const*, int) | 4 | 1643.5 | 0.0 | unclassified |
| aten::mul | 2 | 1370.2 | 33.4 | torch_reference_gaussian |
| void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>) | 2 | 1370.2 | 0.0 | unclassified |
| aten::exp | 1 | 1369.0 | 20.5 | torch_reference_gaussian |
| void at::native::vectorized_elementwise_kernel<4, at::native::exp_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::exp_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>) | 1 | 1369.0 | 0.0 | unclassified |
| aten::pow | 1 | 1367.4 | 30.5 | torch_reference_gaussian |
| void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase&, float)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase&, float)::{lambda(float)#1}, std::array<char*, 2ul>) | 1 | 1367.4 | 0.0 | unclassified |
| aten::clamp_min | 1 | 1367.2 | 25.1 | torch_reference_gaussian |
| void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>) | 1 | 1367.2 | 0.0 | unclassified |
| volta_sgemm_32x128_tn | 1 | 1275.0 | 0.0 | unclassified |
| aten::copy_ | 26 | 1110.8 | 279.7 | unclassified |
| aten::sub | 1 | 908.6 | 23.9 | torch_reference_gaussian |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&)::{lambda(int)#1}) | 1 | 908.6 | 0.0 | unclassified |
| Memcpy DtoD (Device -> Device) | 11 | 591.5 | 0.0 | unclassified |
| aten::masked_fill_ | 8 | 506.3 | 106.5 | unclassified |
| aten::bmm | 8 | 474.2 | 323.7 | torch_reference_aten |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}) | 2 | 404.8 | 0.0 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<c10::Half>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::Half>, std::array<char*, 3ul>) | 13 | 289.9 | 0.0 | unclassified |
| aten::add_ | 5 | 267.8 | 64.5 | unclassified |
| void cutlass::Kernel2<cutlass_70_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_70_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params) | 4 | 266.7 | 0.0 | unclassified |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(c10::Half, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(c10::Half, bool)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(c10::Half, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(c10::Half, bool)#1} const&)::{lambda(int)#1}) | 4 | 246.8 | 0.0 | unclassified |
| volta_fp16_s884gemm_fp16_64x64_ldg8_relu_f2f_tn | 25 | 244.9 | 0.0 | unclassified |
| void cutlass::Kernel2<cutlass_70_tensorop_f16_s884gemm_f16_128x64_tn_align2>(cutlass_70_tensorop_f16_s884gemm_f16_128x64_tn_align2::Params) | 4 | 207.5 | 0.0 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(c10::Half, bool)#1}, std::array<char*, 3ul> >(int, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(c10::Half, bool)#1}, std::array<char*, 3ul>) | 3 | 168.1 | 0.0 | unclassified |
| aten::eq | 5 | 113.0 | 129.9 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<c10::Half, c10::Half, bool, at::native::(anonymous namespace)::CompareEqFunctor<c10::Half> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<c10::Half, c10::Half, bool, at::native::(anonymous namespace)::CompareEqFunctor<c10::Half> >, std::array<char*, 2ul>) | 4 | 110.0 | 0.0 | unclassified |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&)::{lambda(int)#1}) | 1 | 91.5 | 0.0 | unclassified |
| void gemmk1_kernel<int, float, 256, 5, true, false, false, false, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float, 0>(cublasGemmk1Params<float, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float, biasType<cublasGemvTensorStridedBatched<float>::value_type, float>::type>) | 1 | 81.7 | 0.0 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::float16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::float16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) | 1 | 65.8 | 0.0 | unclassified |
| aten::native_layer_norm | 8 | 61.6 | 204.2 | torch_retained_candidate_rejected |
| void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c10::Half, float>(int, float, c10::Half const*, c10::Half const*, c10::Half const*, float*, float*, c10::Half*) | 8 | 61.6 | 0.0 | torch_retained_candidate_rejected |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> >(int, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1>) | 9 | 36.2 | 0.0 | unclassified |
| aten::add | 9 | 24.0 | 168.2 | torch_reference_gaussian |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#10}::operator()() const::{lambda(c10::Half)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#10}::operator()() const::{lambda(c10::Half)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#10}::operator()() const::{lambda(c10::Half)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#10}::operator()() const::{lambda(c10::Half)#1} const&)::{lambda(int)#1}) | 4 | 22.9 | 0.0 | unclassified |
| aten::gelu | 4 | 12.2 | 70.3 | torch_retained |
| void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#3}::operator()() const::{lambda(c10::Half)#1}, std::array<char*, 2ul> >(int, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#3}::operator()() const::{lambda(c10::Half)#1}, std::array<char*, 2ul>) | 4 | 12.2 | 0.0 | unclassified |
| aten::mul_ | 4 | 11.0 | 60.6 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<c10::Half, c10::Half, c10::Half, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<c10::Half, c10::Half, c10::Half, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>) | 4 | 11.0 | 0.0 | unclassified |
| aten::index_select | 1 | 6.8 | 45.3 | unclassified |
| void at::native::(anonymous namespace)::indexSelectLargeIndex<c10::Half, long, unsigned int, 2, 2, -2, true>(at::cuda::detail::TensorInfo<c10::Half, unsigned int>, at::cuda::detail::TensorInfo<c10::Half const, unsigned int>, at::cuda::detail::TensorInfo<long const, unsigned int>, int, int, unsigned int, unsigned int, long) | 1 | 6.8 | 0.0 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<long, long, bool, at::native::(anonymous namespace)::CompareEqFunctor<long> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<long, long, bool, at::native::(anonymous namespace)::CompareEqFunctor<long> >, std::array<char*, 2ul>) | 1 | 3.0 | 0.0 | unclassified |
| aten::abs | 2 | 1.9 | 24.4 | torch_reference_gaussian |
| void at::native::vectorized_elementwise_kernel<4, at::native::AbsFunctor<float>, std::array<char*, 2ul> >(int, at::native::AbsFunctor<float>, std::array<char*, 2ul>) | 1 | 1.9 | 0.0 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<float>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<float>, std::array<char*, 2ul>) | 1 | 1.9 | 0.0 | unclassified |
| cudaDeviceSynchronize | 1 | 0.0 | 13815.2 | unclassified |
| cudaLaunchKernel | 101 | 0.0 | 1369.7 | unclassified |
| aten::linear | 28 | 0.0 | 512.5 | unclassified |
| aten::empty | 66 | 0.0 | 474.2 | unclassified |
| cudaMemcpyAsync | 11 | 0.0 | 245.0 | unclassified |
| aten::transpose | 50 | 0.0 | 187.5 | unclassified |
| aten::view | 107 | 0.0 | 171.2 | unclassified |
| aten::empty_strided | 13 | 0.0 | 121.4 | unclassified |
| cuLaunchKernel | 8 | 0.0 | 118.7 | unclassified |
| aten::as_strided | 76 | 0.0 | 118.0 | unclassified |
| aten::t | 28 | 0.0 | 114.0 | unclassified |
| aten::masked_fill | 8 | 0.0 | 89.2 | unclassified |
| aten::clone | 16 | 0.0 | 86.0 | unclassified |
| aten::_to_copy | 10 | 0.0 | 72.1 | unclassified |
| aten::reshape | 29 | 0.0 | 66.5 | unclassified |
| aten::layer_norm | 8 | 0.0 | 60.5 | torch_retained_candidate_rejected |
| aten::expand | 15 | 0.0 | 50.4 | unclassified |
| aten::unsqueeze | 10 | 0.0 | 50.1 | unclassified |
| aten::empty_like | 13 | 0.0 | 45.3 | unclassified |
| aten::to | 19 | 0.0 | 25.2 | unclassified |
| aten::resize_ | 2 | 0.0 | 21.4 | unclassified |
| aten::dropout | 17 | 0.0 | 17.0 | unclassified |
| aten::embedding | 1 | 0.0 | 14.7 | unclassified |
| aten::softmax | 4 | 0.0 | 12.2 | torch_reference_aten |
| cudaOccupancyMaxActiveBlocksPerMultiprocessor | 1 | 0.0 | 11.6 | unclassified |
| aten::contiguous | 5 | 0.0 | 10.3 | unclassified |
| aten::type_as | 4 | 0.0 | 9.3 | unclassified |
| aten::relu | 1 | 0.0 | 7.5 | torch_reference_gaussian |
| cudaDeviceGetAttribute | 8 | 0.0 | 6.4 | unclassified |
| aten::permute | 1 | 0.0 | 3.9 | unclassified |
| aten::expand_as | 1 | 0.0 | 2.0 | unclassified |
