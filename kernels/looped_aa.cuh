#pragma once

// dvlt looped aa core: depth-scaled attention block + the interval depth-scaling gate.
// per-channel gates s_attn/s_mlp/s_out (constant across tokens) scale the residual branches.

#include <cmath>
#include <vector>
#include <cassert>
#include "layernorm.cuh"
#include "block_rope.cuh"
#include "attention.cuh"
#include "linear.cuh"

namespace dvlt {

// interval depth-scaling gate weights (host, fp32): out = (w2 @ silu(w1 @ sinusoid(t) + b1) + b2) + 1.
struct IntervalGateWeights {
    std::vector<float> w1, b1;   // [hidden, 2*hidden], [hidden]
    std::vector<float> w2, b2;   // [num_gates*D, hidden], [num_gates*D]
    int hidden, out_dim;
};

// writes out[g.out_dim]; heap-free (stack scratch). model uses hidden=64; bump MAXH if larger.
inline void interval_gate(
    const IntervalGateWeights& g, float t_now, float t_next, float* out)
{
    constexpr int MAXH = 256;
    const int H = g.hidden, half = H / 2;
    assert(H <= MAXH && "interval_gate: hidden exceeds MAXH stack scratch");
    float freqs[MAXH / 2], emb[2 * MAXH], hid[MAXH];
    for (int h = 0; h < half; h++)
        freqs[h] = expf(-logf(10000.f) * (float)h / (float)half);

    auto sinu = [&](float v, int base) {
        for (int h = 0; h < half; h++) {
            float a = v * freqs[h];
            emb[base + h]        = cosf(a);
            emb[base + half + h] = sinf(a);
        }
    };
    sinu(t_now, 0);
    sinu(t_next, H);

    for (int i = 0; i < H; i++) {
        float acc = g.b1[i];
        for (int j = 0; j < 2 * H; j++) acc += g.w1[i * 2 * H + j] * emb[j];
        hid[i] = acc / (1.f + expf(-acc));   // silu
    }

    for (int o = 0; o < g.out_dim; o++) {
        float acc = g.b2[o];
        for (int j = 0; j < H; j++) acc += g.w2[o * H + j] * hid[j];
        out[o] = acc + 1.f;
    }
}

// gated element-wise kernels

// x[m,d] += s[d] * gamma[d] * (delta[m,d] + bias[d])   (gated layerscale + bias + residual)
__global__ void gated_ls_residual_bias_kernel(
    __nv_bfloat16* __restrict__       x,
    const __nv_bfloat16* __restrict__ delta,
    const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ gamma,
    const float* __restrict__         s,
    int total, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int c = idx % D;
    float dv = __bfloat162float(delta[idx]) + __bfloat162float(bias[c]);
    float g  = __bfloat162float(gamma[c]) * s[c];
    x[idx] = __float2bfloat16(__bfloat162float(x[idx]) + dv * g);
}

// x[m,d] = s[d] * x[m,d]   (s_out gate)
__global__ void scale_channels_kernel(
    __nv_bfloat16* __restrict__ x, const float* __restrict__ s, int total, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    x[idx] = __float2bfloat16(__bfloat162float(x[idx]) * s[idx % D]);
}

struct DepthScaledBlockWeights {
    const __nv_bfloat16 *norm1_w, *norm1_b;
    const __nv_bfloat16 *qkv_w, *qkv_b;
    const __nv_bfloat16 *q_norm_w, *q_norm_b, *k_norm_w, *k_norm_b;
    const __nv_bfloat16 *proj_w, *proj_b;
    const __nv_bfloat16 *ls1;
    const __nv_bfloat16 *norm2_w, *norm2_b;
    const __nv_bfloat16 *fc1_w, *fc1_b, *fc2_w, *fc2_b;
    const __nv_bfloat16 *ls2;
};

// shared plan for one depth-scaled attention block. supports rope (frame attn) or
// no-rope (global attn) via the HAS_ROPE arg to fwd.
struct DepthScaledBlockPlan {
    int B_, N_, H_, D_, DH_, M_;
    float scale_, eps_;
    int max_len_ = 0;
    const float *cos_ = nullptr, *sin_ = nullptr;

    __nv_bfloat16 *ln_, *qkv_, *q_, *k_, *v_, *attn_, *merged_, *proj_, *ln2_, *fc1_, *mlp_;
    LinearPlan qkv_plan_, proj_plan_, fc1_plan_, fc2_plan_;

    void init(CublasCtx& ctx, GpuArena& arena, int B, int N, int H, int D, int DH,
              const float* cos_dev, const float* sin_dev, int max_len, float eps = 1e-5f) {
        B_ = B; N_ = N; H_ = H; D_ = D; DH_ = DH; M_ = B * N;
        scale_ = 1.f / sqrtf((float)DH); eps_ = eps;
        cos_ = cos_dev; sin_ = sin_dev; max_len_ = max_len;
        size_t MD = (size_t)M_ * D_, M4D = (size_t)M_ * 4 * D_;
        ln_     = arena.alloc<__nv_bfloat16>(MD);
        qkv_    = arena.alloc<__nv_bfloat16>(3 * MD);
        q_      = arena.alloc<__nv_bfloat16>(MD);
        k_      = arena.alloc<__nv_bfloat16>(MD);
        v_      = arena.alloc<__nv_bfloat16>(MD);
        attn_   = arena.alloc<__nv_bfloat16>(MD);
        merged_ = arena.alloc<__nv_bfloat16>(MD);
        proj_   = arena.alloc<__nv_bfloat16>(MD);
        ln2_    = arena.alloc<__nv_bfloat16>(MD);
        fc1_    = arena.alloc<__nv_bfloat16>(M4D);
        mlp_    = arena.alloc<__nv_bfloat16>(MD);
        qkv_plan_.init(ctx, M_, D_, 3 * D_);
        proj_plan_.init(ctx, M_, D_, D_);
        fc1_plan_.init(ctx, M_, D_, 4 * D_);
        fc2_plan_.init(ctx, M_, 4 * D_, D_);
    }

    // x [B*N, D] bf16 in-place. pos [B*N, 2] int32 (frame rope) or nullptr (global).
    // s_attn/s_mlp/s_out: [D] fp32 device gates.
    void fwd(CublasCtx& ctx, const DepthScaledBlockWeights& w, __nv_bfloat16* x,
             const int* pos, const float* s_attn, const float* s_mlp, const float* s_out,
             bool has_rope, cudaStream_t stream = nullptr) {
        const int threads = 256, total_MD = M_ * D_, tot_qkv = B_ * H_ * N_ * DH_;

        layernorm_affine(x, w.norm1_w, w.norm1_b, ln_, B_, N_, D_, eps_, stream);
        qkv_plan_.exec(ctx, ln_, w.qkv_w, qkv_, nullptr, 1.f, 0.f, stream);
        if (has_rope)
            launch_split_qknorm_rope<true, true>(qkv_, q_, k_, v_, w.qkv_b,
                w.q_norm_w, w.q_norm_b, w.k_norm_w, w.k_norm_b,
                pos, cos_, sin_, B_, N_, H_, DH_, max_len_, eps_, stream);
        else
            launch_split_qknorm_rope<true, false>(qkv_, q_, k_, v_, w.qkv_b,
                w.q_norm_w, w.q_norm_b, w.k_norm_w, w.k_norm_b,
                nullptr, nullptr, nullptr, B_, N_, H_, DH_, max_len_, eps_, stream);
        launch_flash_attn(q_, k_, v_, attn_, B_ * H_, N_, scale_, stream);
        merge_heads_kernel<<<(tot_qkv + threads-1)/threads, threads, 0, stream>>>(
            attn_, merged_, B_, H_, N_, DH_);
        proj_plan_.exec(ctx, merged_, w.proj_w, proj_, nullptr, 1.f, 0.f, stream);
        gated_ls_residual_bias_kernel<<<(total_MD + threads-1)/threads, threads, 0, stream>>>(
            x, proj_, w.proj_b, w.ls1, s_attn, total_MD, D_);

        layernorm_affine(x, w.norm2_w, w.norm2_b, ln2_, B_, N_, D_, eps_, stream);
        fc1_plan_.exec_gelu_bias(ctx, ln2_, w.fc1_w, fc1_, w.fc1_b, stream);
        fc2_plan_.exec(ctx, fc1_, w.fc2_w, mlp_, nullptr, 1.f, 0.f, stream);
        gated_ls_residual_bias_kernel<<<(total_MD + threads-1)/threads, threads, 0, stream>>>(
            x, mlp_, w.fc2_b, w.ls2, s_mlp, total_MD, D_);

        scale_channels_kernel<<<(total_MD + threads-1)/threads, threads, 0, stream>>>(
            x, s_out, total_MD, D_);
    }

    void destroy() { qkv_plan_.destroy(); proj_plan_.destroy(); fc1_plan_.destroy(); fc2_plan_.destroy(); }
};

} // namespace dvlt
