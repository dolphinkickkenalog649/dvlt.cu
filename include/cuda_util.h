#pragma once
// minimal cuda helpers used by the cli: error check + host<->device float/bf16 transfers.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "cuda error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

namespace dvt {

inline float* to_dev_f32(const float* h, size_t n) {
    float* d; CUDA_CHECK(cudaMalloc(&d, n*4));
    CUDA_CHECK(cudaMemcpy(d, h, n*4, cudaMemcpyHostToDevice));
    return d;
}

inline std::vector<float> from_dev_f32(const float* d, size_t n) {
    std::vector<float> v(n);
    CUDA_CHECK(cudaMemcpy(v.data(), d, n*4, cudaMemcpyDeviceToHost));
    return v;
}

} // namespace dvt
