#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>

static const float IMG_MEAN[3] = { 0.485f, 0.456f, 0.406f };
static const float IMG_STD[3]  = { 0.229f, 0.224f, 0.225f };

static inline uint8_t denorm_u8(float v, int c) {
    float pix = v * IMG_STD[c] + IMG_MEAN[c];
    pix = pix < 0.f ? 0.f : (pix > 1.f ? 1.f : pix);
    return (uint8_t)(pix * 255.f + 0.5f);
}

// writes filtered PLY: only vertices where mask[i] != 0.
// pts [bn, h, w, 3] f32, imgs [bn, 3, h, w] f32 (imagenet-normalized), mask [bn*h*w] u8.
// buffers all vertices in memory before writing (one fwrite for the whole body).
static void write_pointcloud_ply_filtered(const char*    stem,
                                          const float*   world_pts_bhwc,
                                          const uint8_t* mask,
                                          const float*   imgs_bchw,
                                          int bn, int h, int w) {
    const int total_pixels = bn * h * w;
    int n_pts = 0;
    for (int i = 0; i < total_pixels; i++) n_pts += mask[i] ? 1 : 0;

    // 15 bytes per vertex: 3x float32 xyz + 3x uint8 rgb
    const size_t vertex_size = 15;
    std::vector<uint8_t> buf((size_t)n_pts * vertex_size);
    uint8_t* dst = buf.data();

    for (int b = 0; b < bn; b++) {
        const float* p    = world_pts_bhwc + (size_t)b * h * w * 3;
        const float* imgR = imgs_bchw + (size_t)b * 3 * h * w;
        const float* imgG = imgR + h * w;
        const float* imgB = imgG + h * w;

        for (int i = 0; i < h * w; i++) {
            if (!mask[(size_t)b * h * w + i]) continue;
            memcpy(dst, p + i * 3, 12);
            dst[12] = denorm_u8(imgR[i], 0);
            dst[13] = denorm_u8(imgG[i], 1);
            dst[14] = denorm_u8(imgB[i], 2);
            dst += vertex_size;
        }
    }

    char path[512];
    snprintf(path, sizeof(path), "%s.ply", stem);
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "ply_writer: cannot open '%s'\n", path); return; }

    fprintf(f,
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex %d\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n",
        n_pts);

    fwrite(buf.data(), 1, buf.size(), f);
    fclose(f);
}
