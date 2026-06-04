#pragma once

// DINOv2 ViT block: no rope, no qk_norm, with layerscale.
// x = x + ls1 * attn(norm1(x))
// x = x + ls2 * mlp(norm2(x))

#include "block_rope.cuh"

namespace dvlt {

struct DinoV2BlockWeights {
    const __nv_bfloat16* norm1_w;    // [D]
    const __nv_bfloat16* norm1_b;    // [D]
    const __nv_bfloat16* qkv_w;      // [3D, D]
    const __nv_bfloat16* qkv_b;      // [3D]
    const __nv_bfloat16* proj_w;     // [D, D]
    const __nv_bfloat16* proj_b;     // [D]
    const __nv_bfloat16* ls1_gamma;  // [D]
    const __nv_bfloat16* norm2_w;    // [D]
    const __nv_bfloat16* norm2_b;    // [D]
    const __nv_bfloat16* fc1_w;      // [4D, D]
    const __nv_bfloat16* fc1_b;      // [4D]
    const __nv_bfloat16* fc2_w;      // [D, 4D]
    const __nv_bfloat16* fc2_b;      // [D]
    const __nv_bfloat16* ls2_gamma;  // [D]
};

// shared plan for all blocks in the encoder (same B, T, D, H, DH).
struct DinoV2BlockPlan {
    int B_, T_, H_, D_, DH_, M_;
    float scale_;

    __nv_bfloat16* ln_buf_;      // [M, D]
    __nv_bfloat16* qkv_buf_;     // [M, 3D]
    __nv_bfloat16* q_buf_;       // [M, D]  = [B*H, T, DH]
    __nv_bfloat16* k_buf_;       // [M, D]
    __nv_bfloat16* v_buf_;       // [M, D]
    __nv_bfloat16* attn_buf_;    // [M, D]  flash output [B*H, T, DH]
    __nv_bfloat16* merged_buf_;  // [M, D]  after merge heads
    __nv_bfloat16* proj_buf_;    // [M, D]
    __nv_bfloat16* ln2_buf_;     // [M, D]
    __nv_bfloat16* fc1_buf_;     // [M, 4D]
    __nv_bfloat16* mlp_buf_;     // [M, D]

    LinearPlan qkv_plan_;   // M × D  → 3D
    LinearPlan proj_plan_;  // M × D  → D
    LinearPlan fc1_plan_;   // M × D  → 4D
    LinearPlan fc2_plan_;   // M × 4D → D

    void init(CublasCtx& ctx, GpuArena& arena, int B, int T, int H, int D, int DH) {
        B_ = B; T_ = T; H_ = H; D_ = D; DH_ = DH;
        M_     = B * T;
        scale_ = 1.f / sqrtf((float)DH);

        size_t MD  = (size_t)M_ * D_;
        size_t M4D = (size_t)M_ * 4 * D_;

        ln_buf_     = arena.alloc<__nv_bfloat16>(MD);
        qkv_buf_    = arena.alloc<__nv_bfloat16>(3 * MD);
        q_buf_      = arena.alloc<__nv_bfloat16>(MD);
        k_buf_      = arena.alloc<__nv_bfloat16>(MD);
        v_buf_      = arena.alloc<__nv_bfloat16>(MD);
        attn_buf_   = arena.alloc<__nv_bfloat16>(MD);
        merged_buf_ = arena.alloc<__nv_bfloat16>(MD);
        proj_buf_   = arena.alloc<__nv_bfloat16>(MD);
        ln2_buf_    = arena.alloc<__nv_bfloat16>(MD);
        fc1_buf_    = arena.alloc<__nv_bfloat16>(M4D);
        mlp_buf_    = arena.alloc<__nv_bfloat16>(MD);

        qkv_plan_.init(ctx, M_, D_,     3 * D_);
        proj_plan_.init(ctx, M_, D_,       D_);
        fc1_plan_.init(ctx, M_, D_,     4 * D_);
        fc2_plan_.init(ctx, M_, 4 * D_,   D_);
    }

    // x [B, T, D] bf16 - updated in-place
    void fwd(CublasCtx& ctx, const DinoV2BlockWeights& w,
             __nv_bfloat16* x, cudaStream_t stream = nullptr) {
        const int threads   = 256;
        const int total_MD  = M_ * D_;

        // attention branch

        layernorm_affine(x, w.norm1_w, w.norm1_b, ln_buf_, B_, T_, D_, 1e-6f, stream);

        qkv_plan_.exec(ctx, ln_buf_, w.qkv_w, qkv_buf_, nullptr, 1.f, 0.f, stream);

        {
            int tot = B_ * H_ * T_ * DH_;
            split_qkv_bias_kernel<<<(tot + threads-1)/threads, threads, 0, stream>>>(
                qkv_buf_, q_buf_, k_buf_, v_buf_, w.qkv_b, B_, T_, H_, DH_);
        }

        // no qk_norm, no rope
        launch_flash_attn(q_buf_, k_buf_, v_buf_, attn_buf_, B_ * H_, T_, scale_, stream);

        {
            int tot = B_ * H_ * T_ * DH_;
            merge_heads_kernel<<<(tot + threads-1)/threads, threads, 0, stream>>>(
                attn_buf_, merged_buf_, B_, H_, T_, DH_);
        }

        proj_plan_.exec(ctx, merged_buf_, w.proj_w, proj_buf_, nullptr, 1.f, 0.f, stream);

        // x += (proj_buf + proj_b) * ls1_gamma   (fused bias + layerscale + residual)
        layerscale_residual_bias_kernel<<<(total_MD + threads-1)/threads, threads, 0, stream>>>(
            x, proj_buf_, w.proj_b, w.ls1_gamma, total_MD, D_);

        // mlp branch

        layernorm_affine(x, w.norm2_w, w.norm2_b, ln2_buf_, B_, T_, D_, 1e-6f, stream);

        // fused fc1 gemm + bias + gelu in cutlass epilogue
        fc1_plan_.exec_gelu_bias(ctx, ln2_buf_, w.fc1_w, fc1_buf_, w.fc1_b, stream);

        fc2_plan_.exec(ctx, fc1_buf_, w.fc2_w, mlp_buf_, nullptr, 1.f, 0.f, stream);

        // x += (mlp_buf + fc2_b) * ls2_gamma   (fused bias + layerscale + residual)
        layerscale_residual_bias_kernel<<<(total_MD + threads-1)/threads, threads, 0, stream>>>(
            x, mlp_buf_, w.fc2_b, w.ls2_gamma, total_MD, D_);
    }

    void destroy() {
        qkv_plan_.destroy(); proj_plan_.destroy();
        fc1_plan_.destroy(); fc2_plan_.destroy();
    }
};

} // namespace dvlt
