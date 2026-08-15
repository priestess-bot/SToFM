# Vision FlagOS Inference R2 ATen Profile

Stage: `swiglu_torch`; precision: `fp16`.

This trace is qualitative evidence only; timed latency is reported by the isolated suite.

| Operator | Calls | Self CUDA us | Self CPU us | Classification |
| --- | ---: | ---: | ---: | --- |
| aten::silu | 1 | 16.1 | 2679.3 | vision_swiglu_candidate |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::silu_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#5}::operator()() const::{lambda(c10::Half)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::silu_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#5}::operator()() const::{lambda(c10::Half)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::silu_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#5}::operator()() const::{lambda(c10::Half)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::silu_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#5}::operator()() const::{lambda(c10::Half)#1} const&)::{lambda(int)#1}) | 1 | 16.1 | 0.0 | unclassified |
| aten::mul | 1 | 12.4 | 35.1 | unclassified |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::Half, c10::Half, c10::Half, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::Half, c10::Half, c10::Half, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::Half, c10::Half, c10::Half, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::Half, c10::Half, c10::Half, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}) | 1 | 12.4 | 0.0 | unclassified |
| cudaLaunchKernel | 2 | 0.0 | 55.9 | unclassified |
| aten::slice | 2 | 0.0 | 38.1 | unclassified |
| aten::chunk | 1 | 0.0 | 30.5 | unclassified |
| aten::narrow | 2 | 0.0 | 21.4 | unclassified |
| aten::split | 1 | 0.0 | 13.6 | unclassified |
| cudaDeviceSynchronize | 1 | 0.0 | 10.2 | unclassified |
| aten::as_strided | 2 | 0.0 | 6.6 | unclassified |
| [memory] | 2 | 0.0 | 0.0 | unclassified |
