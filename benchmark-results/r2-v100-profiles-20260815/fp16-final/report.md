# SToFM R2 ATen Profile

Stage: `final`; precision: `fp16`.

| Operator | Calls | Self CUDA us | Self CPU us | Classification |
| --- | ---: | ---: | ---: | --- |
| CUDAGraphTreeManager.record_function (dynamo_timed) | 2 | 372253.1 | 0.0 | unclassified |
| Torch-Compiled Region: 0/0 | 1 | 6104.6 | 379.8 | gaussian_compiler_region |
| volta_sgemm_128x64_tn | 2 | 2900.9 | 0.0 | unclassified |
| triton_poi_fused_relu_5 | 1 | 1367.0 | 0.0 | unclassified |
| volta_sgemm_32x128_tn | 2 | 969.9 | 0.0 | unclassified |
| triton_poi_fused__to_copy_abs_add_div_exp_mul_pow_sub_3 | 1 | 699.7 | 0.0 | unclassified |
| BaddbmmFunction | 4 | 305.8 | 732.0 | unclassified |
| baddbmm_kernel | 4 | 305.8 | 0.0 | unclassified |
| aten::addmm | 25 | 292.9 | 4394.6 | optimized_flagos_aten |
| addmm_kernel | 25 | 292.9 | 0.0 | unclassified |
| aten::copy_ | 21 | 287.5 | 315.2 | unclassified |
| aten::where | 3 | 250.8 | 96.5 | unclassified |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool, c10::Half, c10::Half)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool, c10::Half, c10::Half)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool, c10::Half, c10::Half)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool, c10::Half, c10::Half)#1} const&)::{lambda(int)#1}) | 3 | 250.8 | 0.0 | unclassified |
| aten::masked_fill_ | 4 | 246.8 | 104.3 | unclassified |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(c10::Half, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(c10::Half, bool)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(c10::Half, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#11}::operator()() const::{lambda(c10::Half, bool)#1} const&)::{lambda(int)#1}) | 4 | 246.8 | 0.0 | unclassified |
| Memcpy DtoD (Device -> Device) | 6 | 219.2 | 0.0 | unclassified |
| aten::_softmax | 4 | 212.1 | 459.7 | optimized_flagos_aten |
| softmax_kernel_inner | 4 | 212.1 | 0.0 | unclassified |
| aten::bmm | 4 | 207.2 | 585.8 | optimized_flagos_aten |
| bmm_kernel | 4 | 207.2 | 0.0 | unclassified |
| aten::isneginf | 3 | 102.6 | 70.6 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::isneginf_kernel_impl(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(c10::Half)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::isneginf_kernel_impl(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(c10::Half)#1}, std::array<char*, 2ul>) | 3 | 102.6 | 0.0 | unclassified |
| triton_poi_fused__to_copy_clone_masked_fill_7 | 1 | 68.9 | 0.0 | unclassified |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#10}::operator()() const::{lambda(c10::Half)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#10}::operator()() const::{lambda(c10::Half)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#10}::operator()() const::{lambda(c10::Half)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#10}::operator()() const::{lambda(c10::Half)#1} const&)::{lambda(int)#1}) | 12 | 68.3 | 0.0 | unclassified |
| void gemmk1_kernel<int, float, 256, 5, true, false, false, false, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float, 0>(cublasGemmk1Params<float, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float, biasType<cublasGemvTensorStridedBatched<float>::value_type, float>::type>) | 1 | 65.6 | 0.0 | unclassified |
| aten::native_layer_norm | 8 | 60.4 | 302.2 | torch_retained_candidate_rejected |
| void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c10::Half, float>(int, float, c10::Half const*, c10::Half const*, c10::Half const*, float*, float*, c10::Half*) | 8 | 60.4 | 0.0 | torch_retained_candidate_rejected |
| void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<c10::Half>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::Half>, std::array<char*, 3ul>) | 9 | 25.2 | 0.0 | unclassified |
| aten::add | 8 | 22.3 | 218.0 | torch_retained |
| triton_poi_fused__to_copy_0 | 1 | 14.9 | 0.0 | unclassified |
| aten::fill_ | 7 | 12.4 | 109.0 | unclassified |
| aten::gelu | 4 | 12.3 | 100.5 | torch_retained |
| void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#3}::operator()() const::{lambda(c10::Half)#1}, std::array<char*, 2ul> >(int, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#3}::operator()() const::{lambda(c10::Half)#1}, std::array<char*, 2ul>) | 4 | 12.3 | 0.0 | unclassified |
| aten::mul_ | 4 | 10.4 | 85.9 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<c10::Half, c10::Half, c10::Half, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<c10::Half, c10::Half, c10::Half, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>) | 4 | 10.4 | 0.0 | unclassified |
| aten::eq | 2 | 9.8 | 99.6 | unclassified |
| triton_poi_fused_addmm_1 | 1 | 9.6 | 0.0 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<long>, std::array<char*, 1ul> >(int, at::native::FillFunctor<long>, std::array<char*, 1ul>) | 4 | 7.1 | 0.0 | unclassified |
| aten::index_select | 1 | 7.0 | 49.2 | unclassified |
| void at::native::(anonymous namespace)::indexSelectLargeIndex<c10::Half, long, unsigned int, 2, 2, -2, true>(at::cuda::detail::TensorInfo<c10::Half, unsigned int>, at::cuda::detail::TensorInfo<c10::Half const, unsigned int>, at::cuda::detail::TensorInfo<long const, unsigned int>, int, int, unsigned int, unsigned int, long) | 1 | 7.0 | 0.0 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<c10::Half, c10::Half, bool, at::native::(anonymous namespace)::CompareEqFunctor<c10::Half> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<c10::Half, c10::Half, bool, at::native::(anonymous namespace)::CompareEqFunctor<c10::Half> >, std::array<char*, 2ul>) | 1 | 6.8 | 0.0 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<c10::Half>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::Half>, std::array<char*, 1ul>) | 3 | 5.3 | 0.0 | unclassified |
| void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<long, long, bool, at::native::(anonymous namespace)::CompareEqFunctor<long> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<long, long, bool, at::native::(anonymous namespace)::CompareEqFunctor<long> >, std::array<char*, 2ul>) | 1 | 3.0 | 0.0 | unclassified |
| triton_poi_fused__to_copy_4 | 1 | 2.9 | 0.0 | unclassified |
| aten::add_ | 1 | 2.9 | 16.4 | unclassified |
| triton_poi_fused__to_copy_addmm_2 | 1 | 2.6 | 0.0 | unclassified |
| triton_poi_fused__to_copy_6 | 1 | 2.5 | 0.0 | unclassified |
| CUDAGraphTreeManager.record_function (dynamo_timed) | 1 | 0.0 | 366766.1 | unclassified |
| cudaDeviceSynchronize | 6 | 0.0 | 6099.4 | unclassified |
| aten::empty | 81 | 0.0 | 3700.7 | unclassified |
| cudaLaunchKernel | 62 | 0.0 | 1019.2 | unclassified |
| cudaMalloc | 6 | 0.0 | 1011.4 | unclassified |
| cudaFree | 6 | 0.0 | 1008.9 | unclassified |
| cuLaunchKernel | 45 | 0.0 | 659.9 | unclassified |
| aten::linear | 25 | 0.0 | 609.4 | unclassified |
| aten::baddbmm | 4 | 0.0 | 331.6 | optimized_flagos_aten |
| aten::expand | 41 | 0.0 | 265.6 | unclassified |
| aten::clone | 19 | 0.0 | 228.3 | unclassified |
| aten::view | 119 | 0.0 | 227.5 | unclassified |
| aten::reshape | 46 | 0.0 | 217.4 | unclassified |
| aten::as_strided | 104 | 0.0 | 210.7 | unclassified |
| aten::mm | 3 | 0.0 | 205.1 | unclassified |
| cudaGraphInstantiateWithFlags | 1 | 0.0 | 203.3 | unclassified |
| aten::transpose | 43 | 0.0 | 194.5 | unclassified |
| cudaMemcpyAsync | 6 | 0.0 | 174.8 | unclassified |
| aten::t | 25 | 0.0 | 159.3 | unclassified |
| aten::empty_like | 21 | 0.0 | 138.7 | unclassified |
| aten::empty_strided | 8 | 0.0 | 99.4 | unclassified |
| aten::broadcast_to | 29 | 0.0 | 95.7 | unclassified |
| cudaGraphLaunch | 1 | 0.0 | 86.4 | unclassified |
| aten::masked_fill | 4 | 0.0 | 85.1 | unclassified |
| aten::layer_norm | 8 | 0.0 | 73.8 | torch_retained_candidate_rejected |
| TorchDynamo Cache Lookup | 1 | 0.0 | 63.3 | unclassified |
| aten::resize_ | 4 | 0.0 | 58.4 | unclassified |
| triton_poi_fused__to_copy_0 | 1 | 0.0 | 46.2 | unclassified |
| aten::permute | 4 | 0.0 | 44.6 | unclassified |
| aten::slice | 8 | 0.0 | 37.3 | unclassified |
| aten::zeros | 3 | 0.0 | 35.2 | unclassified |
| aten::contiguous | 12 | 0.0 | 31.6 | unclassified |
| aten::_foreach_copy_ | 1 | 0.0 | 30.8 | unclassified |
