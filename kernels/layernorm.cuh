#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace dvlt {

static constexpr int LN_WARP_SIZE = 32;
static constexpr int LN_WARPS = 2;

template<int D>
__global__ __launch_bounds__(LN_WARPS * LN_WARP_SIZE, 1)
void layernorm_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ beta,
    __nv_bfloat16*       __restrict__ out,
    int B, int S, float eps
) {
    const int wid  = threadIdx.x / LN_WARP_SIZE;
    const int lane = threadIdx.x % LN_WARP_SIZE;
    const int batch = blockIdx.y;
    const int s     = blockIdx.x * LN_WARPS + wid;
    if (s >= S) return;

    const __nv_bfloat16* row = x + ((size_t)batch * S + s) * D;
    __nv_bfloat16*     orow = out + ((size_t)batch * S + s) * D;

    float mean = 0.f;
    for (int i = lane; i < D; i += LN_WARP_SIZE)
        mean += __bfloat162float(row[i]);
    mean += __shfl_down_sync(0xffffffff, mean, 16);
    mean += __shfl_down_sync(0xffffffff, mean, 8);
    mean += __shfl_down_sync(0xffffffff, mean, 4);
    mean += __shfl_down_sync(0xffffffff, mean, 2);
    mean += __shfl_down_sync(0xffffffff, mean, 1);
    mean = __shfl_sync(0xffffffff, mean, 0) * (1.f / D);

    float var = 0.f;
    for (int i = lane; i < D; i += LN_WARP_SIZE) {
        float v = __bfloat162float(row[i]) - mean;
        var += v * v;
    }
    var += __shfl_down_sync(0xffffffff, var, 16);
    var += __shfl_down_sync(0xffffffff, var, 8);
    var += __shfl_down_sync(0xffffffff, var, 4);
    var += __shfl_down_sync(0xffffffff, var, 2);
    var += __shfl_down_sync(0xffffffff, var, 1);
    var = __shfl_sync(0xffffffff, var, 0);
    float inv_std = rsqrtf(var * (1.f / D) + eps);

    for (int i = lane; i < D; i += LN_WARP_SIZE) {
        float v = (__bfloat162float(row[i]) - mean) * inv_std;
        float g = __bfloat162float(gamma[i]);
        float b = __bfloat162float(beta[i]);
        orow[i] = __float2bfloat16(v * g + b);
    }
}

template<int D>
void launch_layernorm(
    const __nv_bfloat16* x,
    const __nv_bfloat16* gamma,
    const __nv_bfloat16* beta,
    __nv_bfloat16*       out,
    int B, int S,
    float eps = 1e-6f,
    cudaStream_t stream = nullptr
) {
    dim3 grid((S + LN_WARPS - 1) / LN_WARPS, B);
    dim3 block(LN_WARPS * LN_WARP_SIZE);
    layernorm_kernel<D><<<grid, block, 0, stream>>>(x, gamma, beta, out, B, S, eps);
}

// runtime dim dispatch over the dims dvlt actually uses (encoder/aa 768, decoder 384).
inline void layernorm_affine(
    const __nv_bfloat16* x, const __nv_bfloat16* gamma, const __nv_bfloat16* beta,
    __nv_bfloat16* out, int B, int S, int D, float eps = 1e-6f, cudaStream_t stream = nullptr
) {
    switch (D) {
        case 768:  launch_layernorm<768>(x, gamma, beta, out, B, S, eps, stream);  break;
        case 384:  launch_layernorm<384>(x, gamma, beta, out, B, S, eps, stream);  break;
        case 1024: launch_layernorm<1024>(x, gamma, beta, out, B, S, eps, stream); break;
        default:
            fprintf(stderr, "layernorm_affine: unsupported D=%d\n", D);
            exit(1);
    }
}

} // namespace dvlt
