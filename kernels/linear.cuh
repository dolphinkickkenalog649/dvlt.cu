#pragma once
// linear layer: y = x @ W^T + b
// cublas backend - cuBLASLt for bf16 gemms with auto-tuned algorithms.
// W stored [N, K] (row-major), x [M, K], y [M, N], all row-major bf16.
// fused variant: y = gelu(x @ W^T + bias) via cuBLASLt epilogue.

#include "../include/gemm.h"
#include <cstdio>

namespace dvlt {

// bias-add after gemm: out[m,n] += bias[n]
// out [M,N] bf16, bias [N] fp32.
__global__ void linear_bias_add(
        __nv_bfloat16* __restrict__ out,
        const float*   __restrict__ bias,
        int total, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int n = idx % N;
        out[idx] = __float2bfloat16(__bfloat162float(out[idx]) + bias[n]);
    }
}

// cached linear plan: y = x @ w^T + b, w [out,in]. cuBLASLt, init once per (M,K,N).
struct LinearPlan {
    int M_ = 0, K_ = 0, N_ = 0;

    // cached cuBLASLt descriptors for plain gemm
    cublasLtMatmulDesc_t    desc_     = nullptr;
    cublasLtMatrixLayout_t  lW_       = nullptr;
    cublasLtMatrixLayout_t  lA_       = nullptr;
    cublasLtMatrixLayout_t  lC_       = nullptr;
    cublasLtMatmulAlgo_t    algo_;

    // cached cuBLASLt descriptors for gelu+bias gemm
    cublasLtMatmulDesc_t    gelu_desc_ = nullptr;
    cublasLtMatmulAlgo_t    gelu_algo_;
    bool                    gelu_ok_   = false;

    void init(CublasCtx& ctx, int M, int K, int N) {
        M_ = M; K_ = K; N_ = N;

        // row-major C[M,N] = A[M,K] @ W[N,K]^T, expressed col-major for cuBLASLt (transa=T, transb=N).
        cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;

        // plain gemm
        cublasLtMatmulDescCreate(&desc_, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasLtMatmulDescSetAttribute(desc_, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
        cublasLtMatmulDescSetAttribute(desc_, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

        cublasLtMatrixLayoutCreate(&lW_, CUDA_R_16BF, K, N, K);
        cublasLtMatrixLayoutCreate(&lA_, CUDA_R_16BF, K, M, K);
        cublasLtMatrixLayoutCreate(&lC_, CUDA_R_16BF, N, M, N);

        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        size_t ws = CublasCtx::WS_BYTES;
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &ws, sizeof(ws));
        cublasLtMatmulHeuristicResult_t result;
        int nr;
        cublasLtMatmulAlgoGetHeuristic(ctx.lt, desc_, lW_, lA_, lC_, lC_,
                                        pref, 1, &result, &nr);
        algo_ = result.algo;

        // gelu+bias gemm
        cublasLtMatmulDescCreate(&gelu_desc_, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasLtMatmulDescSetAttribute(gelu_desc_, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
        cublasLtMatmulDescSetAttribute(gelu_desc_, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_GELU_BIAS;
        cublasLtMatmulDescSetAttribute(gelu_desc_, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));
        cudaDataType_t biasType = CUDA_R_16BF;
        cublasLtMatmulDescSetAttribute(gelu_desc_, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                        &biasType, sizeof(biasType));

        cublasLtMatmulAlgoGetHeuristic(ctx.lt, gelu_desc_, lW_, lA_, lC_, lC_,
                                        pref, 1, &result, &nr);
        gelu_ok_ = (nr > 0);
        if (gelu_ok_) gelu_algo_ = result.algo;

        cublasLtMatmulPreferenceDestroy(pref);
    }

    // input [M,K], weight [N,K], output [M,N], bias [N] fp32 or nullptr.
    void exec(CublasCtx& ctx,
              const __nv_bfloat16* input,
              const __nv_bfloat16* weight,
              __nv_bfloat16*       output,
              const float*         bias   = nullptr,
              float                alpha  = 1.f,
              float                beta   = 0.f,
              cudaStream_t         stream = nullptr) {
        cublasLtMatmul(ctx.lt, desc_, &alpha, weight, lW_, input, lA_,
                       &beta, output, lC_, output, lC_,
                       &algo_, ctx.workspace, CublasCtx::WS_BYTES, stream);
        if (bias) {
            int total = M_ * N_;
            linear_bias_add<<<(total + 255) / 256, 256, 0, stream>>>(output, bias, total, N_);
        }
    }

    // fused: output[M,N] = gelu(input[M,K] @ weight[N,K]^T + bias[N])
    // bias is bf16 (same as fc1_b in weight structs).
    void exec_gelu_bias(CublasCtx& ctx,
                        const __nv_bfloat16* input,
                        const __nv_bfloat16* weight,
                        __nv_bfloat16*       output,
                        const __nv_bfloat16* bias,
                        cudaStream_t         stream = nullptr) {
        cublasLtMatmulDescSetAttribute(gelu_desc_, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                        &bias, sizeof(bias));
        float alpha = 1.f, beta = 0.f;
        cublasLtMatmul(ctx.lt, gelu_desc_, &alpha, weight, lW_, input, lA_,
                       &beta, output, lC_, output, lC_,
                       &gelu_algo_, ctx.workspace, CublasCtx::WS_BYTES, stream);
    }

    void destroy() {
        if (lW_)        { cublasLtMatrixLayoutDestroy(lW_);      lW_ = nullptr; }
        if (lA_)        { cublasLtMatrixLayoutDestroy(lA_);      lA_ = nullptr; }
        if (lC_)        { cublasLtMatrixLayoutDestroy(lC_);      lC_ = nullptr; }
        if (desc_)      { cublasLtMatmulDescDestroy(desc_);      desc_ = nullptr; }
        if (gelu_desc_) { cublasLtMatmulDescDestroy(gelu_desc_); gelu_desc_ = nullptr; }
    }
};

} // namespace dvlt
