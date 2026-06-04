#pragma once

// runtime bicubic interpolation of the dinov2 pos embed [1+M*M, D] (M=37 native) to the
// cropped [1 + grid_h*grid_w, D] grid, in bf16 on the gpu.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace dvlt {

static float cubic_weight(float x) {
    float ax = fabsf(x);
    if (ax < 1.0f)
        return (1.5f * ax - 2.5f) * ax * ax + 1.0f;
    if (ax < 2.0f)
        return ((-0.5f * ax + 2.5f) * ax - 4.0f) * ax + 2.0f;
    return 0.0f;
}

// [C, Hin, Win] -> [C, Hout, Wout], align_corners=false
static std::vector<float> bicubic_resize(const float* src, int C, int Hin, int Win,
                                          int Hout, int Wout) {
    std::vector<float> dst(C * Hout * Wout);
    float scale_h = (float)Hin / Hout;
    float scale_w = (float)Win / Wout;

    for (int c = 0; c < C; c++) {
        for (int oh = 0; oh < Hout; oh++) {
            float ih = ((float)oh + 0.5f) * scale_h - 0.5f;
            int ih0 = (int)floorf(ih);
            for (int ow = 0; ow < Wout; ow++) {
                float iw = ((float)ow + 0.5f) * scale_w - 0.5f;
                int iw0 = (int)floorf(iw);

                float sum = 0.0f;
                for (int dy = -1; dy <= 2; dy++) {
                    int y = ih0 + dy;
                    int yc = y < 0 ? 0 : (y >= Hin ? Hin - 1 : y);
                    float wy = cubic_weight(ih - (float)y);
                    for (int dx = -1; dx <= 2; dx++) {
                        int x = iw0 + dx;
                        int xc = x < 0 ? 0 : (x >= Win ? Win - 1 : x);
                        float wx = cubic_weight(iw - (float)x);
                        sum += wy * wx * src[c * Hin * Win + yc * Win + xc];
                    }
                }
                dst[c * Hout * Wout + oh * Wout + ow] = sum;
            }
        }
    }
    return dst;
}

// interpolate raw pos embed and upload as bf16.
// raw_pos: [(1 + M*M) * D] f32 from the weight file.
// returns gpu bf16 pointer to [(1 + grid_h*grid_w) * D].
static __nv_bfloat16* interpolate_pos_embed(
    const std::vector<float>& raw_pos,
    int M, int D, int grid_h, int grid_w
) {
    const float* cls_pos   = raw_pos.data();
    const float* patch_pos = raw_pos.data() + D;

    int S_out = grid_h * grid_w;
    int total = (1 + S_out) * D;
    std::vector<__nv_bfloat16> result(total);

    // cls token position (no interpolation)
    for (int i = 0; i < D; i++)
        result[i] = __float2bfloat16(cls_pos[i]);

    if (grid_h == M && grid_w == M) {
        // no interpolation needed - direct copy
        for (int i = 0; i < M * M * D; i++)
            result[D + i] = __float2bfloat16(patch_pos[i]);
    } else {
        // transpose [M*M, D] -> [D, M, M]
        std::vector<float> chw((size_t)D * M * M);
        for (int c = 0; c < D; c++)
            for (int h = 0; h < M; h++)
                for (int w = 0; w < M; w++)
                    chw[c * M * M + h * M + w] = patch_pos[h * M * D + w * D + c];

        auto interp = bicubic_resize(chw.data(), D, M, M, grid_h, grid_w);

        // transpose back [D, grid_h, grid_w] -> [grid_h*grid_w, D]
        for (int c = 0; c < D; c++)
            for (int h = 0; h < grid_h; h++)
                for (int w = 0; w < grid_w; w++)
                    result[(1 + h * grid_w + w) * D + c] =
                        __float2bfloat16(interp[c * grid_h * grid_w + h * grid_w + w]);
    }

    __nv_bfloat16* dev;
    cudaMalloc(&dev, (size_t)total * 2);
    cudaMemcpy(dev, result.data(), (size_t)total * 2, cudaMemcpyHostToDevice);
    return dev;
}

} // namespace dvlt
