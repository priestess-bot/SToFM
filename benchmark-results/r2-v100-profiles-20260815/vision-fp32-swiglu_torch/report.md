# Vision FlagOS Inference R2 ATen Profile

Stage: `swiglu_torch`; precision: `fp32`.

This trace is qualitative evidence only; timed latency is reported by the isolated suite.

| Operator | Calls | Self CUDA us | Self CPU us | Classification |
| --- | ---: | ---: | ---: | --- |
| aten::mul | 1 | 19.5 | 26.1 | unclassified |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}) | 1 | 19.5 | 0.0 | unclassified |
| aten::silu | 1 | 18.5 | 2550.1 | vision_swiglu_candidate |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::silu_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::silu_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::silu_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::silu_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}) | 1 | 18.5 | 0.0 | unclassified |
| cudaLaunchKernel | 2 | 0.0 | 47.7 | unclassified |
| aten::slice | 2 | 0.0 | 34.5 | unclassified |
| aten::chunk | 1 | 0.0 | 28.0 | unclassified |
| aten::narrow | 2 | 0.0 | 19.7 | unclassified |
| aten::split | 1 | 0.0 | 13.1 | unclassified |
| cudaDeviceSynchronize | 1 | 0.0 | 8.4 | unclassified |
| aten::as_strided | 2 | 0.0 | 6.6 | unclassified |
| [memory] | 2 | 0.0 | 0.0 | unclassified |
