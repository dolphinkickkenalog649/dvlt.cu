#pragma once
// cutlass implicit-gemm conv2d (bf16 in/out, fp32 accum) for the depth head's 3x3 stride-1 convs.
// inputs are pre-padded channels-last. three n-tile widths (128/64/32) dispatched by Cout. Cin mult-8.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cutlass/cutlass.h>
#include <cutlass/bfloat16.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>

namespace dvltconv {

template<int TBN, int WN>
using Conv2dKernelBf16N = typename cutlass::conv::kernel::DefaultConv2dFprop<
    cutlass::bfloat16_t, cutlass::layout::TensorNHWC,
    cutlass::bfloat16_t, cutlass::layout::TensorNHWC,
    cutlass::bfloat16_t, cutlass::layout::TensorNHWC,
    float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, TBN, 32>,
    cutlass::gemm::GemmShape<64, WN, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized,
    cutlass::conv::StrideSupport::kStrided,
    8, 8
>::Kernel;
using Conv2dOpB   = cutlass::conv::device::ImplicitGemmConvolution<Conv2dKernelBf16N<128, 64>>;
using Conv2dOpB64 = cutlass::conv::device::ImplicitGemmConvolution<Conv2dKernelBf16N<64, 32>>;
using Conv2dOpB32 = cutlass::conv::device::ImplicitGemmConvolution<Conv2dKernelBf16N<32, 32>>;

template<class Op>
inline cutlass::Status conv3x3_cutlass_run_bf16(
    const cutlass::bfloat16_t* pad_bf, const cutlass::bfloat16_t* filt_bf, cutlass::bfloat16_t* out_nhwc,
    int B, int Cin, int Cout, int H, int W, void* workspace, cudaStream_t stream)
{
    int Hp = H + 2, Wp = W + 2;
    cutlass::conv::Conv2dProblemSize problem(
        {B, Hp, Wp, Cin}, {Cout, 3, 3, Cin}, {0, 0, 0, 0}, {1, 1}, {1, 1},
        {B, H, W, Cout}, cutlass::conv::Mode::kCrossCorrelation, 1);
    using RefA = cutlass::TensorRef<cutlass::bfloat16_t, cutlass::layout::TensorNHWC>;
    RefA a((cutlass::bfloat16_t*)pad_bf,  cutlass::layout::TensorNHWC::packed({B, Hp, Wp, Cin}));
    RefA b((cutlass::bfloat16_t*)filt_bf, cutlass::layout::TensorNHWC::packed({Cout, 3, 3, Cin}));
    RefA d(out_nhwc,                      cutlass::layout::TensorNHWC::packed({B, H, W, Cout}));
    typename Op::Arguments args(problem, a, b, d, d, {1.f, 0.f});
    Op op;
    cutlass::Status st = op.can_implement(args);
    if (st != cutlass::Status::kSuccess) return st;
    st = op.initialize(args, workspace, stream);
    if (st != cutlass::Status::kSuccess) return st;
    return op(stream);
}

// pad_bf [B,H+2,W+2,Cin] + filt_bf krsc [Cout,3,3,Cin] (both bf16, Cin a multiple of 8) ->
// out_nhwc [B,H,W,Cout] bf16. dispatches to the narrowest n tile that still covers Cout.
inline cutlass::Status conv3x3_cutlass_core_bf16(
    const __nv_bfloat16* pad_bf, const __nv_bfloat16* filt_bf, __nv_bfloat16* out_nhwc,
    int B, int Cin, int Cout, int H, int W, void* workspace, cudaStream_t stream = nullptr)
{
    auto* pa = reinterpret_cast<const cutlass::bfloat16_t*>(pad_bf);
    auto* pb = reinterpret_cast<const cutlass::bfloat16_t*>(filt_bf);
    auto* pd = reinterpret_cast<cutlass::bfloat16_t*>(out_nhwc);
    if (Cout <= 32)
        return conv3x3_cutlass_run_bf16<Conv2dOpB32>(pa, pb, pd, B, Cin, Cout, H, W, workspace, stream);
    if (Cout <= 64)
        return conv3x3_cutlass_run_bf16<Conv2dOpB64>(pa, pb, pd, B, Cin, Cout, H, W, workspace, stream);
    return conv3x3_cutlass_run_bf16<Conv2dOpB>(pa, pb, pd, B, Cin, Cout, H, W, workspace, stream);
}

} // namespace dvltconv
