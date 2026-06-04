#pragma once

// DINOv2 ViT-B/14 encoder with register tokens.
// patch_embed -> [CLS+pos, regs, patches+pos] -> 12 blocks -> layernorm -> extract patch tokens.

#include "patch_embed.cuh"
#include "dinov2_block.cuh"

namespace dvlt {

// assemble the input sequence [B,T,D], T=1+R+S: [CLS+pos, R regs (no pos), patches+pos].
__global__ void prepare_sequence_kernel(
    const __nv_bfloat16* __restrict__ patches,  // [B, S, D]
    const __nv_bfloat16* __restrict__ cls,      // [D]
    const __nv_bfloat16* __restrict__ pos,      // [(1+S)*D]
    const __nv_bfloat16* __restrict__ regs,     // [R*D]
    __nv_bfloat16*       __restrict__ out,      // [B, T, D]
    int B, int S, int R, int D
) {
    int T     = 1 + R + S;
    int total = B * T * D;
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int d  = idx % D;
    int bt = idx / D;
    int b  = bt / T;
    int t  = bt % T;

    __nv_bfloat16 val;
    if (t == 0) {
        val = __hadd(cls[d], pos[d]);
    } else if (t <= R) {
        val = regs[(t - 1) * D + d];
    } else {
        int s = t - (1 + R);
        val = __hadd(patches[b * S * D + s * D + d], pos[(1 + s) * D + d]);
    }
    out[idx] = val;
}

// extracts patch tokens: out[B, S, D] = x[B, T, D][:, 1+R:]
__global__ void extract_patch_tokens_kernel(
    const __nv_bfloat16* __restrict__ x,    // [B, T, D]
    __nv_bfloat16*       __restrict__ out,  // [B, S, D]
    int B, int T, int S, int D, int R
) {
    int total = B * S * D;
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int d  = idx % D;
    int bs = idx / D;
    int b  = bs / S;
    int s  = bs % S;
    int t  = 1 + R + s;

    out[idx] = x[(b * T + t) * D + d];
}

struct DinoV2EncoderWeights {
    const __nv_bfloat16*     patch_w;    // [D, K=3*P*P]
    const __nv_bfloat16*     patch_b;    // [D]
    const __nv_bfloat16*     cls;        // [D]
    const __nv_bfloat16*     pos;        // [(1+S)*D]
    const __nv_bfloat16*     regs;       // [R*D]
    const DinoV2BlockWeights* blocks;    // [num_blocks]
    int                       num_blocks;
    const __nv_bfloat16*     norm_w;    // [D]
    const __nv_bfloat16*     norm_b;    // [D]
};

struct DinoV2EncoderPlan {
    int B_, S_, T_, D_, H_, DH_, R_, P_;

    PatchEmbedPlan  patch_plan_;
    DinoV2BlockPlan block_plan_;

    __nv_bfloat16* patch_buf_;  // [B, S, D]
    __nv_bfloat16* seq_buf_;    // [B, T, D]
    __nv_bfloat16* out_buf_;    // [B, S, D]  final patch tokens

    void init(
        CublasCtx& ctx, GpuArena& arena,
        int B, int H_img, int W_img, int P,
        int D, int H_heads, int DH, int R
    ) {
        B_ = B; D_ = D; H_ = H_heads; DH_ = DH; R_ = R; P_ = P;
        S_ = (H_img / P) * (W_img / P);
        T_ = 1 + R_ + S_;

        patch_plan_.init(ctx, arena, B, 3, H_img, W_img, P, D);
        block_plan_.init(ctx, arena, B, T_, H_heads, D, DH);

        patch_buf_ = arena.alloc<__nv_bfloat16>((size_t)B_ * S_ * D_);
        seq_buf_   = arena.alloc<__nv_bfloat16>((size_t)B_ * T_ * D_);
        out_buf_   = arena.alloc<__nv_bfloat16>((size_t)B_ * S_ * D_);
    }

    // img [B, 3, H_img, W_img] float32 device ptr
    // out [B, S, D] bf16 device ptr (pass out_buf_ for the default)
    void fwd(
        CublasCtx&               ctx,
        const DinoV2EncoderWeights& w,
        const float*             img,
        int H_img, int W_img,
        __nv_bfloat16*           out,
        cudaStream_t             stream = nullptr
    ) {
        const int threads = 256;

        // 1. patch embed: img → patch_buf_ [B, S, D]
        patch_plan_.fwd(ctx, img, w.patch_w, w.patch_b,
                        patch_buf_, 3, H_img, W_img, P_, stream);

        // 2. assemble sequence [B, T, D]
        {
            int total = B_ * T_ * D_;
            prepare_sequence_kernel<<<(total + threads-1)/threads, threads, 0, stream>>>(
                patch_buf_, w.cls, w.pos, w.regs, seq_buf_, B_, S_, R_, D_);
        }

        // 3. transformer blocks (all share block_plan_, weights differ per block)
        for (int i = 0; i < w.num_blocks; i++)
            block_plan_.fwd(ctx, w.blocks[i], seq_buf_, stream);

        // 4. final layernorm (in-place over full sequence). dvlt overrides the dinov2 head
        // norm with a non-affine nn.LayerNorm (eps 1e-5); caller passes ones/zeros for w/b.
        layernorm_affine(seq_buf_, w.norm_w, w.norm_b, seq_buf_, B_, T_, D_, 1e-5f, stream);

        // 5. extract patch tokens
        {
            int total = B_ * S_ * D_;
            extract_patch_tokens_kernel<<<(total + threads-1)/threads, threads, 0, stream>>>(
                seq_buf_, out, B_, T_, S_, D_, R_);
        }
    }

    void destroy() {
        patch_plan_.destroy();
        block_plan_.destroy();
    }
};

} // namespace dvlt
