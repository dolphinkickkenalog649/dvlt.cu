#pragma once

// transformer block with rope2d attention (qk_norm, layerscale).

#include "layernorm.cuh"
#include "rope2d.cuh"
#include "attention.cuh"
#include "linear.cuh"

namespace dvlt {

// helper kernels

// merge heads: [B*H, S, DH] = [B, H, S, DH] → [B, S, H, DH] = [M, D]
// inverse of split_qkv transpose; prepares attention output for proj GEMM.
__global__ void merge_heads_kernel(
    const __nv_bfloat16* __restrict__ in,   // [B, H, S, DH] row-major
    __nv_bfloat16* __restrict__ out,         // [B, S, H, DH] row-major
    int B, int H, int S, int DH
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * S * DH;
    if (idx >= total) return;

    // decode input [B, H, S, DH] index
    int dh = idx % DH;
    int s  = (idx / DH) % S;
    int h  = (idx / (DH * S)) % H;
    int b  = idx / (DH * S * H);

    // output [B, S, H, DH] = [M, D] index
    int out_idx = b * S * H * DH + s * H * DH + h * DH + dh;
    out[out_idx] = in[idx];
}

// plain bf16 bias add: x[i] += bias[i % N]
__global__ void add_bias_bf16_kernel(
    __nv_bfloat16* __restrict__       x,     // [M, N]
    const __nv_bfloat16* __restrict__ bias,  // [N]
    int total, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    x[idx] = __hadd(x[idx], bias[idx % N]);
}

// fused split + optional qkv bias: writes q,k,v each [B,H,S,DH] from in [M,3D],
// adding bias[3D] in the same pass (bias may be nullptr → plain split).
__global__ void split_qkv_bias_kernel(
    const __nv_bfloat16* __restrict__ in,    // [M=B*S, 3D]
    __nv_bfloat16* __restrict__ q,            // [B*H, S, DH]
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ bias,   // [3D] or nullptr
    int B, int S, int H, int DH
) {
    int D   = H * DH;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * S * DH;
    if (idx >= total) return;

    int dh = idx % DH;
    int s  = (idx / DH) % S;
    int h  = (idx / (DH * S)) % H;
    int b  = idx / (DH * S * H);

    int row = b * S + s;
    int col = h * DH + dh;   // 0..D-1 within one of the 3 qkv blocks

    if (bias) {
        q[idx] = __hadd(in[row * 3 * D + col],             bias[col]);
        k[idx] = __hadd(in[row * 3 * D + D + col],         bias[D + col]);
        v[idx] = __hadd(in[row * 3 * D + 2 * D + col],     bias[2 * D + col]);
    } else {
        q[idx] = in[row * 3 * D + col];
        k[idx] = in[row * 3 * D + D + col];
        v[idx] = in[row * 3 * D + 2 * D + col];
    }
}

// fused split + qkv-bias + optional qk-layernorm + optional rope2d, one pass over qkv_buf [M,3D].
// assumes DH==64: one warp per (b,h,s) token, lane t owns head dims t and t+32.

// warp layernorm over the 64-dim head held as (a=dim lane, b=dim lane+32).
__device__ __forceinline__ void qkr_norm64(
    float& a, float& b, int lane,
    const __nv_bfloat16* __restrict__ w, const __nv_bfloat16* __restrict__ bias, float eps)
{
    float sum = a + b;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
    float mean = sum * (1.f / 64.f);
    float da = a - mean, db = b - mean;
    float var = da * da + db * db;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) var += __shfl_xor_sync(0xffffffff, var, o);
    float inv = rsqrtf(var * (1.f / 64.f) + eps);
    a = da * inv * __bfloat162float(w[lane])      + __bfloat162float(bias[lane]);
    b = db * inv * __bfloat162float(w[lane + 32])  + __bfloat162float(bias[lane + 32]);
}

// rope2d on (a=y-half dim lane, b=x-half dim lane) using the lane^16 pair partner.
__device__ __forceinline__ void qkr_rope(
    float& a, float& b, int lane, float cy, float sy, float cx, float sx)
{
    float pa = __shfl_sync(0xffffffff, a, lane ^ 16);
    float pb = __shfl_sync(0xffffffff, b, lane ^ 16);
    if (lane < 16) {            // i0 of the pair
        a = a * cy - pa * sy;
        b = b * cx - pb * sx;
    } else {                    // i1 of the pair
        a = pa * sy + a * cy;
        b = pb * sx + b * cx;
    }
}

template<bool HAS_QKNORM, bool HAS_ROPE>
__global__ void split_qknorm_rope_kernel(
    const __nv_bfloat16* __restrict__ in,        // [M=B*S, 3D]
    __nv_bfloat16* __restrict__ q,                // [B*H, S, DH]
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ qkv_bias,   // [3D] or nullptr
    const __nv_bfloat16* __restrict__ q_norm_w,   // [DH] / nullptr
    const __nv_bfloat16* __restrict__ q_norm_b,
    const __nv_bfloat16* __restrict__ k_norm_w,
    const __nv_bfloat16* __restrict__ k_norm_b,
    const int*   __restrict__ pos,                // [B, S, 2] / nullptr
    const float* __restrict__ cos_t,              // [max_len, D2]
    const float* __restrict__ sin_t,
    int B, int S, int H, int DH, int D2, int max_len, float eps)
{
    const int lane = threadIdx.x & 31;
    const int ti   = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);  // global token
    const int BH   = B * H;
    if (ti >= BH * S) return;

    const int D  = H * DH;
    const int s  = ti % S;
    const int bh = ti / S;
    const int b  = bh / H;
    const int h  = bh % H;

    const int row  = b * S + s;        // row in [M, 3D]
    const int c0   = h * DH + lane;    // head col for dh=lane
    const int c1   = c0 + 32;          // head col for dh=lane+32

    float qa = __bfloat162float(in[row * 3 * D + c0]);
    float qb = __bfloat162float(in[row * 3 * D + c1]);
    float ka = __bfloat162float(in[row * 3 * D + D + c0]);
    float kb = __bfloat162float(in[row * 3 * D + D + c1]);
    float va = __bfloat162float(in[row * 3 * D + 2 * D + c0]);
    float vb = __bfloat162float(in[row * 3 * D + 2 * D + c1]);
    if (qkv_bias) {
        qa += __bfloat162float(qkv_bias[c0]);          qb += __bfloat162float(qkv_bias[c1]);
        ka += __bfloat162float(qkv_bias[D + c0]);      kb += __bfloat162float(qkv_bias[D + c1]);
        va += __bfloat162float(qkv_bias[2 * D + c0]);  vb += __bfloat162float(qkv_bias[2 * D + c1]);
    }

    if (HAS_QKNORM) {
        qkr_norm64(qa, qb, lane, q_norm_w, q_norm_b, eps);
        qkr_norm64(ka, kb, lane, k_norm_w, k_norm_b, eps);
    }

    if (HAS_ROPE) {
        int pos_y = pos[row * 2 + 0];
        int pos_x = pos[row * 2 + 1];
        if (pos_y >= max_len) pos_y = max_len - 1;
        if (pos_x >= max_len) pos_x = max_len - 1;
        const int pair = lane & 15;
        float cy = cos_t[(size_t)pos_y * D2 + pair], sy = sin_t[(size_t)pos_y * D2 + pair];
        float cx = cos_t[(size_t)pos_x * D2 + pair], sx = sin_t[(size_t)pos_x * D2 + pair];
        qkr_rope(qa, qb, lane, cy, sy, cx, sx);
        qkr_rope(ka, kb, lane, cy, sy, cx, sx);
    }

    const int ob = ti * DH;   // [BH, S, DH]
    q[ob + lane] = __float2bfloat16(qa);  q[ob + lane + 32] = __float2bfloat16(qb);
    k[ob + lane] = __float2bfloat16(ka);  k[ob + lane + 32] = __float2bfloat16(kb);
    v[ob + lane] = __float2bfloat16(va);  v[ob + lane + 32] = __float2bfloat16(vb);
}

// host launcher. DH must be 64. pos/cos/sin only read when HAS_ROPE, q/k norm
// weights only read when HAS_QKNORM.
template<bool HAS_QKNORM, bool HAS_ROPE>
inline void launch_split_qknorm_rope(
    const __nv_bfloat16* in, __nv_bfloat16* q, __nv_bfloat16* k, __nv_bfloat16* v,
    const __nv_bfloat16* qkv_bias,
    const __nv_bfloat16* q_norm_w, const __nv_bfloat16* q_norm_b,
    const __nv_bfloat16* k_norm_w, const __nv_bfloat16* k_norm_b,
    const int* pos, const float* cos_dev, const float* sin_dev,
    int B, int S, int H, int DH, int max_len, float eps, cudaStream_t stream = nullptr)
{
    const int D2 = DH / 4;
    const int total = B * H * S;
    const int wpb = 4;                 // warps per block
    const int threads = wpb * 32;
    const int blocks = (total + wpb - 1) / wpb;
    split_qknorm_rope_kernel<HAS_QKNORM, HAS_ROPE><<<blocks, threads, 0, stream>>>(
        in, q, k, v, qkv_bias, q_norm_w, q_norm_b, k_norm_w, k_norm_b,
        pos, cos_dev, sin_dev, B, S, H, DH, D2, max_len, eps);
}

// fused bias + layerscale + residual: x[i] += (delta[i] + bias[i%D]) * gamma[i%D].
// bias added in fp32 before the gamma multiply (one bf16 round, not two).
__global__ void layerscale_residual_bias_kernel(
    __nv_bfloat16* __restrict__       x,      // [M, D] in-out
    const __nv_bfloat16* __restrict__ delta,  // [M, D]
    const __nv_bfloat16* __restrict__ bias,   // [D]
    const __nv_bfloat16* __restrict__ gamma,  // [D]
    int total, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int c  = idx % D;
    float dv = __bfloat162float(delta[idx]) + __bfloat162float(bias[c]);
    float gv = __bfloat162float(gamma[c]);
    x[idx] = __float2bfloat16(__bfloat162float(x[idx]) + dv * gv);
}

// fused bias + plain residual (no layerscale): dst[i] += src[i] + bias[i%D].
__global__ void add_residual_bias_kernel(
    __nv_bfloat16* __restrict__       dst,   // [M, D] in-out
    const __nv_bfloat16* __restrict__ src,   // [M, D]
    const __nv_bfloat16* __restrict__ bias,  // [D]
    int total, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int c = idx % D;
    float r = __bfloat162float(dst[idx]) + __bfloat162float(src[idx]) + __bfloat162float(bias[c]);
    dst[idx] = __float2bfloat16(r);
}

} // namespace dvlt
