#pragma once

// patch embedding via im2col + bf16 gemm.
// image [B, 3, H, W] f32 -> patch tokens [B, S, D] bf16.

#include "block_rope.cuh"

namespace dvlt {

// im2col + f32→bf16 in one pass.
// out[b*S + pr*Wp + pc, c*P*P + ph*P + pw] = img[b, c, pr*P+ph, pc*P+pw]
__global__ void im2col_bf16_kernel(
    const float*         __restrict__ img,  // [B, C, H, W] float32
    __nv_bfloat16*       __restrict__ out,  // [B*S, K] bf16
    int B, int C, int H, int W, int P, int Hp, int Wp
) {
    int K     = C * P * P;
    int total = B * Hp * Wp * K;
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int k  = idx % K;
    int s  = idx / K;
    int b  = s / (Hp * Wp);
    int ps = s % (Hp * Wp);
    int pr = ps / Wp;
    int pc = ps % Wp;

    int c  = k / (P * P);
    int pk = k % (P * P);
    int ph = pk / P;
    int pw = pk % P;

    out[idx] = __float2bfloat16(img[((b * C + c) * H + pr * P + ph) * W + pc * P + pw]);
}

struct PatchEmbedPlan {
    int B_, S_, K_, D_;
    __nv_bfloat16* patches_;  // [B*S, K] scratch

    LinearPlan gemm_;

    void init(CublasCtx& ctx, GpuArena& arena,
              int B, int C, int H, int W, int P, int D) {
        B_ = B; D_ = D;
        int Hp = H / P, Wp = W / P;
        S_  = Hp * Wp;
        K_  = C * P * P;
        int M = B_ * S_;
        patches_ = arena.alloc<__nv_bfloat16>((size_t)M * K_);
        gemm_.init(ctx, M, K_, D_);
    }

    // img [B, C, H, W] f32 device; weight [D, K] bf16; bias [D] bf16; out [B*S, D] bf16
    void fwd(
        CublasCtx&           ctx,
        const float*         img,
        const __nv_bfloat16* weight,
        const __nv_bfloat16* bias,
        __nv_bfloat16*       out,
        int C, int H, int W, int P,
        cudaStream_t         stream = nullptr
    ) {
        int Hp    = H / P, Wp = W / P;
        int M     = B_ * S_;
        int total = M * K_;
        im2col_bf16_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            img, patches_, B_, C, H, W, P, Hp, Wp);

        gemm_.exec(ctx, patches_, weight, out, nullptr, 1.f, 0.f, stream);

        int total_MD = M * D_;
        add_bias_bf16_kernel<<<(total_MD + 255) / 256, 256, 0, stream>>>(
            out, bias, total_MD, D_);
    }

    void destroy() { gemm_.destroy(); }
};

} // namespace dvlt
