#pragma once

// 2d rope precomputed cos/sin tables (base=100, head_dim=64); applied in split_qknorm_rope_kernel.

#include <cmath>
#include <vector>

namespace dvlt {

// precompute cos/sin tables on the host; caller uploads to device.
struct RopeTable {
    std::vector<float> cos_tab;  // [max_len, D]
    std::vector<float> sin_tab;  // [max_len, D]
    int max_len, D;

    void build(float freq_base, int half_head_dim, int max_seq_len) {
        D       = half_head_dim;
        max_len = max_seq_len;
        cos_tab.resize((size_t)max_len * D);
        sin_tab.resize((size_t)max_len * D);
        for (int i = 0; i < D; i++) {
            float inv_f = 1.f / powf(freq_base, (float)(2 * i) / (float)(2 * D));
            for (int t = 0; t < max_len; t++) {
                float angle = t * inv_f;
                cos_tab[(size_t)t * D + i] = cosf(angle);
                sin_tab[(size_t)t * D + i] = sinf(angle);
            }
        }
    }
};

} // namespace dvlt
