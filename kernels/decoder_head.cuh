#pragma once

// dvlt decoder heads: shared transformer front-end (proj_in -> blocks -> norm),
// linear ray head (fp32 + pixel_shuffle), and the simple camera head.

#include <cmath>
#include "layernorm.cuh"
#include "block_rope.cuh"
#include "attention.cuh"
#include "linear.cuh"
#include "../include/gemm.h"

namespace dvlt {

// decoder transformer block (qk_norm + rope, plain residual, dim 384)
struct DecoderBlockWeights {
    const __nv_bfloat16 *norm1_w, *norm1_b;
    const __nv_bfloat16 *qkv_w, *qkv_b;
    const __nv_bfloat16 *q_norm_w, *q_norm_b, *k_norm_w, *k_norm_b;
    const __nv_bfloat16 *proj_w, *proj_b;
    const __nv_bfloat16 *norm2_w, *norm2_b;
    const __nv_bfloat16 *fc1_w, *fc1_b, *fc2_w, *fc2_b;
};

struct DecoderBlockPlan {
    int B_, N_, H_, D_, DH_, M_;
    float scale_, eps_;
    int max_len_;
    const float *cos_, *sin_;
    __nv_bfloat16 *ln_, *qkv_, *q_, *k_, *v_, *attn_, *merged_, *proj_, *ln2_, *fc1_, *mlp_;
    LinearPlan qkv_plan_, proj_plan_, fc1_plan_, fc2_plan_;

    void init(CublasCtx& ctx, GpuArena& arena, int B, int N, int H, int D, int DH,
              const float* cos_dev, const float* sin_dev, int max_len, float eps = 1e-5f) {
        B_ = B; N_ = N; H_ = H; D_ = D; DH_ = DH; M_ = B * N;
        scale_ = 1.f / sqrtf((float)DH); eps_ = eps;
        cos_ = cos_dev; sin_ = sin_dev; max_len_ = max_len;
        size_t MD = (size_t)M_ * D_, M4D = (size_t)M_ * 4 * D_;
        ln_ = arena.alloc<__nv_bfloat16>(MD);    qkv_ = arena.alloc<__nv_bfloat16>(3*MD);
        q_  = arena.alloc<__nv_bfloat16>(MD);    k_   = arena.alloc<__nv_bfloat16>(MD);
        v_  = arena.alloc<__nv_bfloat16>(MD);    attn_= arena.alloc<__nv_bfloat16>(MD);
        merged_ = arena.alloc<__nv_bfloat16>(MD); proj_ = arena.alloc<__nv_bfloat16>(MD);
        ln2_ = arena.alloc<__nv_bfloat16>(MD);   fc1_ = arena.alloc<__nv_bfloat16>(M4D);
        mlp_ = arena.alloc<__nv_bfloat16>(MD);
        qkv_plan_.init(ctx, M_, D_, 3*D_);  proj_plan_.init(ctx, M_, D_, D_);
        fc1_plan_.init(ctx, M_, D_, 4*D_);  fc2_plan_.init(ctx, M_, 4*D_, D_);
    }

    void fwd(CublasCtx& ctx, const DecoderBlockWeights& w, __nv_bfloat16* x,
             const int* pos, cudaStream_t stream = nullptr) {
        const int threads = 256, total_MD = M_*D_, tot = B_*H_*N_*DH_;
        layernorm_affine(x, w.norm1_w, w.norm1_b, ln_, B_, N_, D_, eps_, stream);
        qkv_plan_.exec(ctx, ln_, w.qkv_w, qkv_, nullptr, 1.f, 0.f, stream);
        launch_split_qknorm_rope<true, true>(qkv_, q_, k_, v_, w.qkv_b,
            w.q_norm_w, w.q_norm_b, w.k_norm_w, w.k_norm_b,
            pos, cos_, sin_, B_, N_, H_, DH_, max_len_, eps_, stream);
        launch_flash_attn(q_, k_, v_, attn_, B_*H_, N_, scale_, stream);
        merge_heads_kernel<<<(tot+threads-1)/threads, threads, 0, stream>>>(attn_, merged_, B_, H_, N_, DH_);
        proj_plan_.exec(ctx, merged_, w.proj_w, proj_, nullptr, 1.f, 0.f, stream);
        add_residual_bias_kernel<<<(total_MD+threads-1)/threads, threads, 0, stream>>>(
            x, proj_, w.proj_b, total_MD, D_);
        layernorm_affine(x, w.norm2_w, w.norm2_b, ln2_, B_, N_, D_, eps_, stream);
        fc1_plan_.exec_gelu_bias(ctx, ln2_, w.fc1_w, fc1_, w.fc1_b, stream);
        fc2_plan_.exec(ctx, fc1_, w.fc2_w, mlp_, nullptr, 1.f, 0.f, stream);
        add_residual_bias_kernel<<<(total_MD+threads-1)/threads, threads, 0, stream>>>(
            x, mlp_, w.fc2_b, total_MD, D_);
    }

    void destroy() { qkv_plan_.destroy(); proj_plan_.destroy(); fc1_plan_.destroy(); fc2_plan_.destroy(); }
};

// pixel shuffle for the linear head: lin [B, Sp, out*P*P] -> out [B, out, ph*P, pw*P], fp32.
__global__ void pixel_shuffle_kernel(
    const float* __restrict__ lin, float* __restrict__ out,
    int B, int ph, int pw, int outc, int P)
{
    int Hh = ph*P, Wh = pw*P, total = B*outc*Hh*Wh;
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int w = idx % Wh, h = (idx / Wh) % Hh, c = (idx / (Wh*Hh)) % outc, b = idx / (Wh*Hh*outc);
    int hp = h / P, i = h % P, wp = w / P, j = w % P;
    int sp = hp*pw + wp;
    int lin_c = c*P*P + i*P + j;
    int Sp = ph*pw, C2 = outc*P*P;
    out[idx] = lin[((size_t)b*Sp + sp)*C2 + lin_c];
}

// fp32 bias add: x[i] += bias[i % N]
__global__ void add_bias_f32_kernel(float* __restrict__ x, const float* __restrict__ bias, int total, int N) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < total) x[idx] += bias[idx % N];
}

__global__ void bf16_to_f32_kernel(const __nv_bfloat16* __restrict__ x, float* __restrict__ out, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) out[i] = __bfloat162float(x[i]);
}

// bf16 -> fp32 copy of patch tokens: x_norm [B, N, D][:, patch_start:] -> [B, Sp, D] fp32.
__global__ void extract_patches_f32_kernel(
    const __nv_bfloat16* __restrict__ x, float* __restrict__ out,
    int B, int N, int Sp, int patch_start, int D)
{
    int total = B*Sp*D, idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int d = idx % D, sp = (idx / D) % Sp, b = idx / (D*Sp);
    out[idx] = __bfloat162float(x[((size_t)b*N + patch_start + sp)*D + d]);
}

// patch tokens [B, N, D][:, patch_start:] -> chw feature map [B, D, ph, pw] fp32.
// out[b,c,hp,wp] = x[b, patch_start + hp*pw + wp, c].  (for the conv depth head)
__global__ void extract_patch_map_chw_f32_kernel(
    const __nv_bfloat16* __restrict__ x, float* __restrict__ out,
    int B, int N, int ph, int pw, int patch_start, int D)
{
    int Sp = ph*pw, total = B*D*Sp, idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int sp = idx % Sp, c = (idx / Sp) % D, b = idx / (Sp*D);
    out[idx] = __bfloat162float(x[((size_t)b*N + patch_start + sp)*D + c]);
}

struct DecoderHeadWeights {
    const __nv_bfloat16* proj_in_w;   // [384, 768]
    const __nv_bfloat16* proj_in_b;   // [384]
    const DecoderBlockWeights* blocks;
    int num_blocks;
    const __nv_bfloat16* norm_w;      // [384]
    const __nv_bfloat16* norm_b;      // [384]
    const float* head_w;              // [out*P*P, 384] fp32
    const float* head_b;              // [out*P*P] fp32
};

// shared transformer front-end + linear head. produces dense out [B, outc, H, W] (fp32)
// and the post-block features [B, N, 384] (bf16; camera token = features[:,0]).
struct DecoderHeadPlan {
    int B_, N_, Sp_, IN_, D_, H_, DH_, ph_, pw_, P_, outc_;
    LinearPlan proj_in_plan_;
    DecoderBlockPlan block_plan_;
    __nv_bfloat16 *proj_buf_, *feat_buf_, *norm_buf_;
    float *patch_f32_, *lin_f32_;

    void init(CublasCtx& ctx, GpuArena& arena, int B, int N, int IN, int D, int H, int DH,
              int ph, int pw, int P, int outc, const float* cos_dev, const float* sin_dev, int max_len) {
        B_=B; N_=N; IN_=IN; D_=D; H_=H; DH_=DH; ph_=ph; pw_=pw; P_=P; outc_=outc;
        Sp_ = ph*pw;
        size_t MD = (size_t)B*N*D;
        proj_buf_ = arena.alloc<__nv_bfloat16>(MD);
        feat_buf_ = arena.alloc<__nv_bfloat16>(MD);
        norm_buf_ = arena.alloc<__nv_bfloat16>(MD);
        patch_f32_ = arena.alloc<float>((size_t)B*Sp_*D);
        lin_f32_   = arena.alloc<float>((size_t)B*Sp_*outc*P*P);
        proj_in_plan_.init(ctx, B*N, IN, D);
        block_plan_.init(ctx, arena, B, N, H, D, DH, cos_dev, sin_dev, max_len, 1e-5f);
    }

    // shared front-end: proj_in -> blocks -> norm. writes norm_buf_; feat_out = post-block features
    // (pre-norm; camera token = feat_out[:,0]).
    void run_frontend(CublasCtx& ctx, const DecoderHeadWeights& w, const __nv_bfloat16* x,
                      const int* pos, __nv_bfloat16* feat_out, cudaStream_t stream = nullptr) {
        const int threads = 256;
        proj_in_plan_.exec(ctx, x, w.proj_in_w, proj_buf_, nullptr, 1.f, 0.f, stream);
        add_bias_bf16_kernel<<<((size_t)B_*N_*D_+threads-1)/threads, threads, 0, stream>>>(
            proj_buf_, w.proj_in_b, B_*N_*D_, D_);
        for (int i = 0; i < w.num_blocks; i++)
            block_plan_.fwd(ctx, w.blocks[i], proj_buf_, pos, stream);
        if (feat_out) cudaMemcpyAsync(feat_out, proj_buf_, (size_t)B_*N_*D_*2, cudaMemcpyDeviceToDevice, stream);
        layernorm_affine(proj_buf_, w.norm_w, w.norm_b, norm_buf_, B_, N_, D_, 1e-5f, stream);
    }

    // linear ray/depth head: front-end + fp32 linear + pixel_shuffle -> out_dense [B, outc, ph*P, pw*P].
    void fwd(CublasCtx& ctx, const DecoderHeadWeights& w, const __nv_bfloat16* x, const int* pos,
             float* out_dense, __nv_bfloat16* feat_out, int patch_start, cudaStream_t stream = nullptr) {
        const int threads = 256;
        run_frontend(ctx, w, x, pos, feat_out, stream);
        extract_patches_f32_kernel<<<((size_t)B_*Sp_*D_+threads-1)/threads, threads, 0, stream>>>(
            norm_buf_, patch_f32_, B_, N_, Sp_, patch_start, D_);
        int outdim = outc_*P_*P_;
        gemm_f32(ctx, patch_f32_, w.head_w, lin_f32_, B_*Sp_, D_, outdim, 1.f, 0.f, stream);
        add_bias_f32_kernel<<<((size_t)B_*Sp_*outdim+threads-1)/threads, threads, 0, stream>>>(
            lin_f32_, w.head_b, B_*Sp_*outdim, outdim);
        pixel_shuffle_kernel<<<((size_t)B_*outc_*ph_*P_*pw_*P_+threads-1)/threads, threads, 0, stream>>>(
            lin_f32_, out_dense, B_, ph_, pw_, outc_, P_);
    }

    // for the conv depth head: normed patch tokens -> chw feature map [B, D, ph, pw] fp32.
    void patch_map_f32(float* out_chw, int patch_start, cudaStream_t stream = nullptr) {
        const int threads = 256;
        extract_patch_map_chw_f32_kernel<<<((size_t)B_*D_*Sp_+threads-1)/threads, threads, 0, stream>>>(
            norm_buf_, out_chw, B_, N_, ph_, pw_, patch_start, D_);
    }

    void destroy() { proj_in_plan_.destroy(); block_plan_.destroy(); }
};

// simple camera head
// mlp = LN -> Linear -> ReLU -> Linear -> ReLU (bf16); fc_pose in fp32; relu on the trailing fov pair.
// pose_enc layout [Tx,Ty,Tz, qx,qy,qz,qw, fov_h, fov_w].

__global__ void relu_bf16_kernel(__nv_bfloat16* __restrict__ x, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) { float v = __bfloat162float(x[i]); x[i] = __float2bfloat16(v > 0.f ? v : 0.f); }
}

__global__ void relu_tail_f32_kernel(float* __restrict__ pose, int rows, int dim, int start) {
    int r = blockIdx.x*blockDim.x + threadIdx.x;
    if (r >= rows) return;
    for (int c = start; c < dim; c++) { float v = pose[r*dim + c]; pose[r*dim + c] = v > 0.f ? v : 0.f; }
}

struct SimpleCameraHeadWeights {
    const __nv_bfloat16 *ln_w, *ln_b;      // LayerNorm(in_dim)
    const __nv_bfloat16 *fc0_w, *fc0_b;    // Linear(in_dim, hidden)
    const __nv_bfloat16 *fc1_w, *fc1_b;    // Linear(hidden, hidden)
    const float *pose_w, *pose_b;          // Linear(hidden, pose_dim) fp32
};

struct SimpleCameraHeadPlan {
    int M_, IN_, HID_, POSE_;
    LinearPlan fc0_plan_, fc1_plan_;
    __nv_bfloat16 *ln_, *h0_, *h1_;
    float *hf_;

    void init(CublasCtx& ctx, GpuArena& arena, int M, int in_dim, int hidden, int pose_dim) {
        M_ = M; IN_ = in_dim; HID_ = hidden; POSE_ = pose_dim;
        ln_ = arena.alloc<__nv_bfloat16>((size_t)M*in_dim);
        h0_ = arena.alloc<__nv_bfloat16>((size_t)M*hidden);
        h1_ = arena.alloc<__nv_bfloat16>((size_t)M*hidden);
        hf_ = arena.alloc<float>((size_t)M*hidden);
        fc0_plan_.init(ctx, M, in_dim, hidden);
        fc1_plan_.init(ctx, M, hidden, hidden);
    }

    // cls [M, in_dim] bf16 -> pose [M, pose_dim] fp32.
    void fwd(CublasCtx& ctx, const SimpleCameraHeadWeights& w, const __nv_bfloat16* cls,
             float* pose, cudaStream_t stream = nullptr) {
        const int threads = 256;
        layernorm_affine(cls, w.ln_w, w.ln_b, ln_, M_, 1, IN_, 1e-5f, stream);
        fc0_plan_.exec(ctx, ln_, w.fc0_w, h0_, nullptr, 1.f, 0.f, stream);
        add_bias_bf16_kernel<<<((size_t)M_*HID_+threads-1)/threads, threads, 0, stream>>>(h0_, w.fc0_b, M_*HID_, HID_);
        relu_bf16_kernel<<<((size_t)M_*HID_+threads-1)/threads, threads, 0, stream>>>(h0_, M_*HID_);
        fc1_plan_.exec(ctx, h0_, w.fc1_w, h1_, nullptr, 1.f, 0.f, stream);
        add_bias_bf16_kernel<<<((size_t)M_*HID_+threads-1)/threads, threads, 0, stream>>>(h1_, w.fc1_b, M_*HID_, HID_);
        relu_bf16_kernel<<<((size_t)M_*HID_+threads-1)/threads, threads, 0, stream>>>(h1_, M_*HID_);
        // fp32 fc_pose
        bf16_to_f32_kernel<<<((size_t)M_*HID_+threads-1)/threads, threads, 0, stream>>>(h1_, hf_, M_*HID_);
        gemm_f32(ctx, hf_, w.pose_w, pose, M_, HID_, POSE_, 1.f, 0.f, stream);
        add_bias_f32_kernel<<<((size_t)M_*POSE_+threads-1)/threads, threads, 0, stream>>>(pose, w.pose_b, M_*POSE_, POSE_);
        relu_tail_f32_kernel<<<(M_+threads-1)/threads, threads, 0, stream>>>(pose, M_, POSE_, 7);
    }

    void destroy() { fc0_plan_.destroy(); fc1_plan_.destroy(); }
};

} // namespace dvlt
