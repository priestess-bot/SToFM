# SToFM R2 ATen Profile

Stage: `final`; precision: `fp32`.

| Operator | Calls | Self CUDA us | Self CPU us | Classification |
| --- | ---: | ---: | ---: | --- |
| CUDAGraphTreeManager.record_function (dynamo_timed) | 2 | 389908.2 | 0.0 | unclassified |
| Torch-Compiled Region: 0/0 | 1 | 6125.9 | 403.7 | gaussian_compiler_region |
| volta_sgemm_128x64_tn | 2 | 2907.9 | 0.0 | unclassified |
| triton_poi_fused_relu_1 | 1 | 1367.6 | 0.0 | unclassified |
| aten::addmm | 25 | 1257.0 | 4313.7 | optimized_flagos_aten |
| addmm_kernel | 25 | 1257.0 | 0.0 | unclassified |
| volta_sgemm_32x128_tn | 2 | 972.5 | 0.0 | unclassified |
| triton_poi_fused_abs_add_div_exp_mul_pow_sub_0 | 1 | 718.1 | 0.0 | gaussian_compiler_kernel |
| BaddbmmFunction | 4 | 500.4 | 739.9 | unclassified |
| baddbmm_kernel | 4 | 500.4 | 0.0 | unclassified |
| aten::copy_ | 21 | 479.7 | 332.5 | unclassified |
| Memcpy DtoD (Device -> Device) | 6 | 408.9 | 0.0 | unclassified |
| aten::_softmax | 4 | 391.5 | 466.4 | optimized_flagos_aten |
| softmax_kernel_inner | 4 | 391.5 | 0.0 | unclassified |
| aten::masked_fill_ | 4 | 356.1 | 98.5 | unclassified |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float, bool)#1} const&)::{lambda(int)#1}) | 4 | 356.1 | 0.0 | unclassified |
| aten::bmm | 4 | 350.1 | 589.2 | optimized_flagos_aten |
| bmm_kernel | 4 | 350.1 | 0.0 | unclassified |
| aten::where | 3 | 294.4 | 96.4 | unclassified |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(bool, float, float)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(bool, float, float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(bool, float, float)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(bool, float, float)#1} const&)::{lambda(int)#1}) | 3 | 294.4 | 0.0 | unclassified |
| aten::isneginf | 3 | 173.5 | 67.5 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::isneginf_kernel_impl(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::isneginf_kernel_impl(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>) | 3 | 173.5 | 0.0 | unclassified |
| triton_poi_fused_clone_masked_fill_2 | 1 | 89.5 | 0.0 | unclassified |
| aten::native_layer_norm | 8 | 73.4 | 303.1 | torch_retained_candidate_rejected |
| void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<float, float>(int, float, float const*, float const*, float const*, float*, float*, float*) | 8 | 73.4 | 0.0 | torch_retained_candidate_rejected |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}) | 12 | 70.8 | 0.0 | unclassified |
| void gemmk1_kernel<int, float, 256, 5, true, false, false, false, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float, 0>(cublasGemmk1Params<float, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float, biasType<cublasGemvTensorStridedBatched<float>::value_type, float>::type>) | 1 | 70.3 | 0.0 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>) | 9 | 31.7 | 0.0 | unclassified |
| aten::add | 8 | 28.5 | 197.3 | torch_retained |
| aten::gelu | 4 | 13.3 | 104.9 | torch_retained |
| void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>) | 4 | 13.3 | 0.0 | unclassified |
| aten::fill_ | 7 | 12.4 | 104.6 | unclassified |
| aten::eq | 2 | 11.2 | 96.1 | unclassified |
| aten::mul_ | 4 | 10.6 | 84.5 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>) | 4 | 10.6 | 0.0 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, float, bool, at::native::(anonymous namespace)::CompareEqFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<float, float, bool, at::native::(anonymous namespace)::CompareEqFunctor<float> >, std::array<char*, 2ul>) | 1 | 8.3 | 0.0 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<long>, std::array<char*, 1ul> >(int, at::native::FillFunctor<long>, std::array<char*, 1ul>) | 4 | 7.1 | 0.0 | unclassified |
| aten::index_select | 1 | 6.9 | 50.7 | unclassified |
| void at::native::(anonymous namespace)::indexSelectLargeIndex<float, long, unsigned int, 2, 2, -2, true>(at::cuda::detail::TensorInfo<float, unsigned int>, at::cuda::detail::TensorInfo<float const, unsigned int>, at::cuda::detail::TensorInfo<long const, unsigned int>, int, int, unsigned int, unsigned int, long) | 1 | 6.9 | 0.0 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>) | 3 | 5.4 | 0.0 | unclassified |
| aten::add_ | 1 | 3.2 | 14.8 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<long, long, bool, at::native::(anonymous namespace)::CompareEqFunctor<long> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<long, long, bool, at::native::(anonymous namespace)::CompareEqFunctor<long> >, std::array<char*, 2ul>) | 1 | 2.9 | 0.0 | unclassified |
| CUDAGraphTreeManager.record_function (dynamo_timed) | 1 | 0.0 | 384444.8 | unclassified |
| cudaDeviceSynchronize | 6 | 0.0 | 6140.0 | unclassified |
| aten::empty | 81 | 0.0 | 3728.5 | unclassified |
| cudaLaunchKernel | 62 | 0.0 | 1058.3 | unclassified |
| cudaFree | 4 | 0.0 | 801.0 | unclassified |
| cudaMalloc | 4 | 0.0 | 733.1 | unclassified |
| cuLaunchKernel | 40 | 0.0 | 629.7 | unclassified |
| aten::linear | 25 | 0.0 | 608.6 | unclassified |
| aten::expand | 41 | 0.0 | 346.7 | unclassified |
| aten::baddbmm | 4 | 0.0 | 313.1 | optimized_flagos_aten |
| aten::view | 119 | 0.0 | 249.7 | unclassified |
| aten::mm | 3 | 0.0 | 215.3 | unclassified |
| cudaGraphInstantiateWithFlags | 1 | 0.0 | 202.5 | unclassified |
| aten::reshape | 46 | 0.0 | 201.7 | unclassified |
| aten::as_strided | 104 | 0.0 | 189.1 | unclassified |
| cudaMemcpyAsync | 6 | 0.0 | 183.0 | unclassified |
| aten::clone | 19 | 0.0 | 181.8 | unclassified |
| aten::transpose | 43 | 0.0 | 175.4 | unclassified |
| aten::t | 25 | 0.0 | 140.6 | unclassified |
| aten::empty_like | 21 | 0.0 | 137.7 | unclassified |
| aten::broadcast_to | 29 | 0.0 | 110.0 | unclassified |
| aten::empty_strided | 8 | 0.0 | 106.4 | unclassified |
| aten::masked_fill | 4 | 0.0 | 85.6 | unclassified |
| aten::layer_norm | 8 | 0.0 | 73.2 | torch_retained_candidate_rejected |
| cudaGraphLaunch | 1 | 0.0 | 64.6 | unclassified |
| TorchDynamo Cache Lookup | 1 | 0.0 | 60.4 | unclassified |
| triton_poi_fused_abs_add_div_exp_mul_pow_sub_0 | 1 | 0.0 | 60.1 | gaussian_compiler_kernel |
| aten::slice | 8 | 0.0 | 44.2 | unclassified |
| aten::resize_ | 4 | 0.0 | 32.6 | unclassified |
| aten::contiguous | 12 | 0.0 | 28.7 | unclassified |
| aten::zeros | 3 | 0.0 | 28.5 | unclassified |
| triton_poi_fused_clone_masked_fill_2 | 1 | 0.0 | 28.5 | unclassified |
| aten::_foreach_copy_ | 1 | 0.0 | 28.1 | unclassified |
| triton_poi_fused_relu_1 | 1 | 0.0 | 27.0 | unclassified |
| aten::unsqueeze | 8 | 0.0 | 26.6 | unclassified |
| aten::softmax | 4 | 0.0 | 24.1 | optimized_flagos_aten |
| aten::permute | 4 | 0.0 | 22.6 | unclassified |
| aten::dropout | 17 | 0.0 | 21.6 | unclassified |
