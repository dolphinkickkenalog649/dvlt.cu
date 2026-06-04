#pragma once
// ransac rays_to_pose: world-space rays -> camera extrinsics/intrinsics (DA3 homography fit).
// per frame: downsample rays/conf to patch grid -> ransac homography (8-pt weighted fits) ->
// refit on inliers -> ql-decompose into rotation + intrinsics -> translation from ray origins.
// sample draws are data-independent, precomputed once; the rest is deterministic linear algebra.

#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace dvlt {

// typed device alloc (count of elements). avoids the void**-cast aliasing footgun.
template <class T> inline void dmalloc(T*& p, size_t count) {
    cudaError_t e = cudaMalloc(&p, count * sizeof(T));
    if (e != cudaSuccess) { fprintf(stderr, "ransac dmalloc failed: %s\n", cudaGetErrorString(e)); exit(1); }
}

// bilinear downsample (channel-last, align_corners=true)
__global__ void bilinear_ac_kernel(const float* __restrict__ in, float* __restrict__ out,
                                    int BS, int ih, int iw, int oh, int ow, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BS * oh * ow) return;
    int j = idx % ow; int t = idx / ow; int i = t % oh; int bs = t / oh;
    float sy = (oh > 1) ? (float)i * (ih - 1) / (oh - 1) : 0.f;
    float sx = (ow > 1) ? (float)j * (iw - 1) / (ow - 1) : 0.f;
    int y0 = (int)floorf(sy); int y1 = min(y0 + 1, ih - 1); float wy = sy - y0;
    int x0 = (int)floorf(sx); int x1 = min(x0 + 1, iw - 1); float wx = sx - x0;
    const float* p00 = in + (((size_t)bs * ih + y0) * iw + x0) * C;
    const float* p01 = in + (((size_t)bs * ih + y0) * iw + x1) * C;
    const float* p10 = in + (((size_t)bs * ih + y1) * iw + x0) * C;
    const float* p11 = in + (((size_t)bs * ih + y1) * iw + x1) * C;
    float* o = out + ((size_t)idx) * C;
    for (int c = 0; c < C; c++) {
        float top = p00[c] * (1 - wx) + p01[c] * wx;
        float bot = p10[c] * (1 - wx) + p11[c] * wx;
        o[c] = top * (1 - wy) + bot * wy;
    }
}

// dst 2d correspondences + masked weights from patch rays (normalize dir, /z, z-mask).
__global__ void ransac_prep_kernel(const float* __restrict__ rays_patch,   // (BS,N,6)
                                   const float* __restrict__ conf_patch,    // (BS,N)
                                   float* __restrict__ dst2d, float* __restrict__ w,
                                   int BS, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BS * N) return;
    const float* r = rays_patch + (size_t)idx * 6;
    float d0 = r[0], d1 = r[1], d2 = r[2];
    float norm = sqrtf(d0 * d0 + d1 * d1 + d2 * d2) + 1e-20f;
    float n0 = d0 / norm, n1 = d1 / norm, n2 = d2 / norm;
    bool mask = fabsf(n2) > 1e-4f;            // origin z (=1) always passes
    if (mask) { dst2d[idx * 2] = d0 / d2; dst2d[idx * 2 + 1] = d1 / d2; w[idx] = conf_patch[idx]; }
    else      { dst2d[idx * 2] = n0;      dst2d[idx * 2 + 1] = n1;      w[idx] = 0.f; }
}

// 9x9 symmetric jacobi eigensolver (eigenvectors in v columns)
__device__ inline void jacobi9(float a[81], float v[81]) {
    for (int i = 0; i < 81; i++) v[i] = (i / 9 == i % 9) ? 1.f : 0.f;
    for (int sweep = 0; sweep < 24; sweep++) {
        float off = 0;
        for (int p = 0; p < 9; p++) for (int q = p + 1; q < 9; q++) off += a[p * 9 + q] * a[p * 9 + q];
        if (off < 1e-22f) break;
        for (int p = 0; p < 9; p++) {
            for (int q = p + 1; q < 9; q++) {
                float apq = a[p * 9 + q];
                if (fabsf(apq) < 1e-22f) continue;
                float phi = 0.5f * (a[q * 9 + q] - a[p * 9 + p]) / apq;
                float t = (phi >= 0 ? 1.f : -1.f) / (fabsf(phi) + sqrtf(phi * phi + 1.f));
                float c = 1.f / sqrtf(t * t + 1.f), s = t * c;
                for (int k = 0; k < 9; k++) {
                    float akp = a[k * 9 + p], akq = a[k * 9 + q];
                    a[k * 9 + p] = c * akp - s * akq; a[k * 9 + q] = s * akp + c * akq;
                }
                for (int k = 0; k < 9; k++) {
                    float apk = a[p * 9 + k], aqk = a[q * 9 + k];
                    a[p * 9 + k] = c * apk - s * aqk; a[q * 9 + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < 9; k++) {
                    float vkp = v[k * 9 + p], vkq = v[k * 9 + q];
                    v[k * 9 + p] = c * vkp - s * vkq; v[k * 9 + q] = s * vkp + c * vkq;
                }
            }
        }
    }
}

// smallest-eigenvalue eigenvector of AtA, reshaped row-major to 3x3 and normalized by H22.
__device__ inline void homography_from_AtA(float AtA[81], float H[9]) {
    float v[81];
    jacobi9(AtA, v);
    int mn = 0; float best = AtA[0];
    for (int k = 1; k < 9; k++) if (AtA[k * 9 + k] < best) { best = AtA[k * 9 + k]; mn = k; }
    float h22 = v[8 * 9 + mn]; if (fabsf(h22) < 1e-30f) h22 = 1e-30f;
    for (int k = 0; k < 9; k++) H[k] = v[k * 9 + mn] / h22;
}

// accumulate one point pair's two rows into AtA (sqrt-weighted dlt rows).
__device__ inline void accum_dlt(float AtA[81], float x, float y, float u, float v, float ww) {
    float a1[9] = {-x * ww, -y * ww, -ww, 0, 0, 0, x * u * ww, y * u * ww, u * ww};
    float a2[9] = {0, 0, 0, -x * ww, -y * ww, -ww, x * v * ww, y * v * ww, v * ww};
    for (int r = 0; r < 9; r++) for (int cc = 0; cc < 9; cc++)
        AtA[r * 9 + cc] += a1[r] * a1[cc] + a2[r] * a2[cc];
}

// one thread per (frame, ransac iter): 8-pt fit + inlier score over all N points.
__global__ void ransac_fit_score_kernel(const float* __restrict__ src2d,  // (N,2)
                                         const float* __restrict__ dst2d,  // (BS,N,2)
                                         const float* __restrict__ w,      // (BS,N)
                                         const int* __restrict__ cand,     // (BS,n_sample)
                                         const int* __restrict__ rand_idx, // (n_iter,8)
                                         float* __restrict__ H_batch,      // (BS,n_iter,9)
                                         float* __restrict__ score,        // (BS,n_iter)
                                         int BS, int N, int n_sample, int n_iter, float thr) {
    // one thread per (frame, iter): 8-pt fit + inlier score over all N points. the work is
    // jacobi-bound (one serial 9x9 eigensolve each), so a thread-per-task beats a block-per-task
    // (which would idle the block during the serial solve).
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= BS * n_iter) return;
    int it = gid % n_iter; int b = gid / n_iter;
    const float* dstb = dst2d + (size_t)b * N * 2;
    const float* wb = w + (size_t)b * N;

    float AtA[81]; for (int k = 0; k < 81; k++) AtA[k] = 0.f;
    for (int s = 0; s < 8; s++) {
        int p = cand[b * n_sample + rand_idx[it * 8 + s]];
        accum_dlt(AtA, src2d[p * 2], src2d[p * 2 + 1], dstb[p * 2], dstb[p * 2 + 1], sqrtf(wb[p]));
    }
    float H[9]; homography_from_AtA(AtA, H);
    for (int k = 0; k < 9; k++) H_batch[(size_t)gid * 9 + k] = H[k];

    float sc = 0.f;
    for (int n = 0; n < N; n++) {
        float x = src2d[n * 2], y = src2d[n * 2 + 1];
        float px = H[0] * x + H[1] * y + H[2];
        float py = H[3] * x + H[4] * y + H[5];
        float pz = H[6] * x + H[7] * y + H[8];
        pz = pz < 1e-8f ? 1e-8f : pz;
        float ex = px / pz - dstb[n * 2], ey = py / pz - dstb[n * 2 + 1];
        if (sqrtf(ex * ex + ey * ey) < thr) sc += wb[n];
    }
    score[gid] = sc;
}

// 3x3 ql via gram-schmidt qr of A@P with diag-sign canonicalization. fills R (row-major)
// and L (row-major lower-tri). matches dvlt._ql_decomposition.
__device__ inline void ql3(const float A[9], float R[9], float L[9]) {
    // A_tilde = A @ P  (reverse columns)
    float at[9];
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) at[i * 3 + j] = A[i * 3 + (2 - j)];
    // modified gram-schmidt on columns of at
    float q[9], r[9]; for (int k = 0; k < 9; k++) r[k] = 0.f;
    for (int j = 0; j < 3; j++) {
        float vv[3] = {at[0 * 3 + j], at[1 * 3 + j], at[2 * 3 + j]};
        for (int i = 0; i < j; i++) {
            float dot = q[0 * 3 + i] * vv[0] + q[1 * 3 + i] * vv[1] + q[2 * 3 + i] * vv[2];
            r[i * 3 + j] = dot;
            vv[0] -= dot * q[0 * 3 + i]; vv[1] -= dot * q[1 * 3 + i]; vv[2] -= dot * q[2 * 3 + i];
        }
        float nrm = sqrtf(vv[0] * vv[0] + vv[1] * vv[1] + vv[2] * vv[2]) + 1e-30f;
        r[j * 3 + j] = nrm;
        q[0 * 3 + j] = vv[0] / nrm; q[1 * 3 + j] = vv[1] / nrm; q[2 * 3 + j] = vv[2] / nrm;
    }
    // Q = Q_tilde @ P (reverse cols), L = P @ R_tilde @ P
    float Q[9], Lt[9];
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) Q[i * 3 + j] = q[i * 3 + (2 - j)];
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) Lt[i * 3 + j] = r[(2 - i) * 3 + (2 - j)];
    // sign fix by diag(L): Q[:,i]*=sign, L[i,:]*=sign
    for (int i = 0; i < 3; i++) {
        float sgn = (Lt[i * 3 + i] >= 0.f) ? 1.f : -1.f;
        for (int k = 0; k < 3; k++) { Q[k * 3 + i] *= sgn; Lt[i * 3 + k] *= sgn; }
    }
    for (int k = 0; k < 9; k++) { R[k] = Q[k]; L[k] = Lt[k]; }
}

__device__ inline float det3(const float A[9]) {
    return A[0] * (A[4] * A[8] - A[5] * A[7])
         - A[1] * (A[3] * A[8] - A[5] * A[6])
         + A[2] * (A[3] * A[7] - A[4] * A[6]);
}

// one thread per frame: pick best iter, refit on its inliers, ql -> R/focal/pp, T = conf-weighted
// mean of ray origins, then assemble extrinsics/intrinsics.
__global__ void ransac_refit_kernel(const float* __restrict__ src2d, const float* __restrict__ dst2d,
                                     const float* __restrict__ w, const float* __restrict__ H_batch,
                                     const float* __restrict__ score, const float* __restrict__ rays_patch,
                                     const float* __restrict__ conf_patch,
                                     float* __restrict__ Rout, float* __restrict__ Tout,
                                     float* __restrict__ focal, float* __restrict__ pp,
                                     float* __restrict__ extr, float* __restrict__ intr,
                                     int BS, int N, int n_iter, float thr, int H_img, int W_img) {
    // one block per frame: parallel inlier accumulation + conf-weighted translation, thread0 solves.
    int b = blockIdx.x;
    if (b >= BS) return;
    int tid = threadIdx.x, bd = blockDim.x;
    const float* dstb = dst2d + (size_t)b * N * 2;
    const float* wb = w + (size_t)b * N;
    const float* rp = rays_patch + (size_t)b * N * 6;
    const float* cp = conf_patch + (size_t)b * N;

    __shared__ float sAtA[81];
    __shared__ float sT[4];
    __shared__ int scnt;
    __shared__ int sbest;
    if (tid == 0) {
        int best = 0; float bs = score[b * n_iter];
        for (int it = 1; it < n_iter; it++) if (score[b * n_iter + it] > bs) { bs = score[b * n_iter + it]; best = it; }
        sbest = best; scnt = 0;
    }
    for (int k = tid; k < 81; k += bd) sAtA[k] = 0.f;
    if (tid < 4) sT[tid] = 0.f;
    __syncthreads();
    const float* Hb = H_batch + ((size_t)b * n_iter + sbest) * 9;

    float lAtA[81]; for (int k = 0; k < 81; k++) lAtA[k] = 0.f;
    float lTx = 0, lTy = 0, lTz = 0, lc = 0; int lcnt = 0;
    for (int n = tid; n < N; n += bd) {
        float cw = cp[n];
        lTx += rp[n * 6 + 3] * cw; lTy += rp[n * 6 + 4] * cw; lTz += rp[n * 6 + 5] * cw; lc += cw;
        float x = src2d[n * 2], y = src2d[n * 2 + 1];
        float px = Hb[0] * x + Hb[1] * y + Hb[2];
        float py = Hb[3] * x + Hb[4] * y + Hb[5];
        float pz = Hb[6] * x + Hb[7] * y + Hb[8]; pz = pz < 1e-8f ? 1e-8f : pz;
        float ex = px / pz - dstb[n * 2], ey = py / pz - dstb[n * 2 + 1];
        if (sqrtf(ex * ex + ey * ey) < thr) { accum_dlt(lAtA, x, y, dstb[n * 2], dstb[n * 2 + 1], sqrtf(wb[n])); lcnt++; }
    }
    for (int k = 0; k < 81; k++) atomicAdd(&sAtA[k], lAtA[k]);
    atomicAdd(&sT[0], lTx); atomicAdd(&sT[1], lTy); atomicAdd(&sT[2], lTz); atomicAdd(&sT[3], lc);
    atomicAdd(&scnt, lcnt);
    __syncthreads();
    if (tid != 0) return;

    int cnt = scnt;
    float AtA[81]; for (int k = 0; k < 81; k++) AtA[k] = sAtA[k];
    float Tx = sT[0], Ty = sT[1], Tz = sT[2], csum = sT[3];

    float Hf[9];
    if (cnt < 4) { for (int k = 0; k < 9; k++) Hf[k] = (k % 4 == 0) ? 1.f : 0.f; }
    else { homography_from_AtA(AtA, Hf); if (det3(Hf) < 0.f) for (int k = 0; k < 9; k++) Hf[k] = -Hf[k]; }

    float R[9], L[9]; ql3(Hf, R, L);
    float l22 = L[8]; if (fabsf(l22) < 1e-30f) l22 = 1e-30f;
    for (int k = 0; k < 9; k++) L[k] /= l22;
    float fx_n = L[0], fy_n = L[4], cx_n = L[6], cy_n = L[7];   // L[0,0],L[1,1],L[2,0],L[2,1]

    float fxi = 1.f / fminf(fmaxf(fx_n, 0.1f), 10.f);
    float fyi = 1.f / fminf(fmaxf(fy_n, 0.1f), 10.f);
    float fx = fxi * W_img / 2.f, fy = fyi * H_img / 2.f;
    float cx = (cx_n + 1.f) * W_img / 2.f, cy = (cy_n + 1.f) * H_img / 2.f;

    csum += 1e-8f; Tx /= csum; Ty /= csum; Tz /= csum;

    for (int k = 0; k < 9; k++) Rout[b * 9 + k] = R[k];
    Tout[b * 3 + 0] = Tx; Tout[b * 3 + 1] = Ty; Tout[b * 3 + 2] = Tz;
    focal[b * 2 + 0] = fx; focal[b * 2 + 1] = fy;
    pp[b * 2 + 0] = cx; pp[b * 2 + 1] = cy;

    float* E = extr + (size_t)b * 16;
    for (int k = 0; k < 16; k++) E[k] = 0.f;
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) E[i * 4 + j] = R[i * 3 + j];
    E[0 * 4 + 3] = Tx; E[1 * 4 + 3] = Ty; E[2 * 4 + 3] = Tz; E[3 * 4 + 3] = 1.f;
    float* I = intr + (size_t)b * 9;
    for (int k = 0; k < 9; k++) I[k] = 0.f;
    I[0] = fx; I[4] = fy; I[2] = cx; I[5] = cy; I[8] = 1.f;
}

// host orchestrator. rays/conf are full-res device buffers (BS,H,W,6) and (BS,H,W).
struct RansacPose {
    int BS, H, W, ph, pw, N, n_sample, n_iter;
    float thr = 0.2f;
    float *rays_patch = nullptr, *conf_patch = nullptr, *src2d = nullptr, *dst2d = nullptr, *w = nullptr;
    float *H_batch = nullptr, *score = nullptr;
    float *R = nullptr, *T = nullptr, *focal = nullptr, *pp = nullptr, *extr = nullptr, *intr = nullptr;
    int *cand = nullptr, *rand_idx = nullptr;

    void init(int bs, int h, int wd, int patch, int niter = 100) {
        BS = bs; H = h; W = wd; ph = h / patch; pw = wd / patch; N = ph * pw; n_iter = niter;
        n_sample = std::max(8, (int)(N * 0.3f));
        dmalloc(rays_patch, (size_t)BS * N * 6); dmalloc(conf_patch, (size_t)BS * N);
        dmalloc(src2d, (size_t)N * 2);           dmalloc(dst2d, (size_t)BS * N * 2);
        dmalloc(w, (size_t)BS * N);
        dmalloc(H_batch, (size_t)BS * n_iter * 9); dmalloc(score, (size_t)BS * n_iter);
        dmalloc(R, (size_t)BS * 9); dmalloc(T, (size_t)BS * 3);
        dmalloc(focal, (size_t)BS * 2); dmalloc(pp, (size_t)BS * 2);
        dmalloc(extr, (size_t)BS * 16); dmalloc(intr, (size_t)BS * 9);
        dmalloc(cand, (size_t)BS * n_sample); dmalloc(rand_idx, (size_t)n_iter * 8);

        // identity src 2d grid (frame-independent): (x_norm-1, y_norm-1), x_norm = (px+0.5)*2/W
        std::vector<float> s2(N * 2);
        for (int i = 0; i < ph; i++) {
            float yp = (ph > 1) ? (float)i * (H - 1) / (ph - 1) : (H - 1) / 2.f;
            float yn = (yp + 0.5f) * 2.f / H - 1.f;
            for (int j = 0; j < pw; j++) {
                float xp = (pw > 1) ? (float)j * (W - 1) / (pw - 1) : (W - 1) / 2.f;
                float xn = (xp + 0.5f) * 2.f / W - 1.f;
                s2[(i * pw + j) * 2] = xn; s2[(i * pw + j) * 2 + 1] = yn;
            }
        }
        cudaMemcpy(src2d, s2.data(), s2.size() * 4, cudaMemcpyHostToDevice);
    }

    // rand_idx_host: (n_iter*8) ints, precomputed manual_seed(42) draws.
    void run(const float* rays_dev, const float* conf_dev, const int* rand_idx_host, cudaStream_t st = 0) {
        cudaMemcpyAsync(rand_idx, rand_idx_host, (size_t)n_iter * 8 * 4, cudaMemcpyHostToDevice, st);
        int B1 = (BS * ph * pw + 255) / 256;
        bilinear_ac_kernel<<<B1, 256, 0, st>>>(rays_dev, rays_patch, BS, H, W, ph, pw, 6);
        bilinear_ac_kernel<<<B1, 256, 0, st>>>(conf_dev, conf_patch, BS, H, W, ph, pw, 1);
        int B2 = (BS * N + 255) / 256;
        ransac_prep_kernel<<<B2, 256, 0, st>>>(rays_patch, conf_patch, dst2d, w, BS, N);

        // top-n_sample candidate indices per frame (host argsort on masked weights)
        std::vector<float> wh((size_t)BS * N);
        cudaMemcpyAsync(wh.data(), w, (size_t)BS * N * 4, cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);
        std::vector<int> ch((size_t)BS * n_sample);
        std::vector<int> ord(N);
        for (int b = 0; b < BS; b++) {
            std::iota(ord.begin(), ord.end(), 0);
            const float* wb = wh.data() + (size_t)b * N;
            std::stable_sort(ord.begin(), ord.end(), [&](int a, int c) { return wb[a] > wb[c]; });
            for (int k = 0; k < n_sample; k++) ch[b * n_sample + k] = ord[k];
        }
        cudaMemcpyAsync(cand, ch.data(), ch.size() * 4, cudaMemcpyHostToDevice, st);

        ransac_fit_score_kernel<<<(BS * n_iter + 127) / 128, 128, 0, st>>>(src2d, dst2d, w, cand, rand_idx,
                                                    H_batch, score, BS, N, n_sample, n_iter, thr);
        ransac_refit_kernel<<<BS, 128, 0, st>>>(src2d, dst2d, w, H_batch, score, rays_patch,
                                                conf_patch, R, T, focal, pp, extr, intr,
                                                BS, N, n_iter, thr, H, W);
    }
};

} // namespace dvlt
