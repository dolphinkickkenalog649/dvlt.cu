#pragma once

// fused multi-head attention (cutlass FMHA, sm80+).
// q,k,v,o: [BH, S, DH] bf16, non-causal, fp32 accumulate.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cutlass/bfloat16.h>
#include "kernel_forward.h"

namespace dvlt {

inline void launch_flash_attn(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    __nv_bfloat16*       o,
    int BH, int S,
    float scale,
    cudaStream_t stream = nullptr
) {
    constexpr int DH = 64;
    // <scalar_t, arch, is_aligned, kQueriesPerBlock, kKeysPerBlock, kMaxK,
    //  supports_dropout, supports_bias>
    using Attention = AttentionKernel<
        cutlass::bfloat16_t,
        cutlass::arch::Sm80,
        true,
        64, 64, DH,
        false,
        false>;
    static_assert(!Attention::kNeedsOutputAccumulatorBuffer,
                  "DH=64 config must keep output in registers (no accum buffer)");

    using elem_t = cutlass::bfloat16_t;
    typename Attention::Params p;
    p.query_ptr  = reinterpret_cast<elem_t*>(const_cast<__nv_bfloat16*>(q));
    p.key_ptr    = reinterpret_cast<elem_t*>(const_cast<__nv_bfloat16*>(k));
    p.value_ptr  = reinterpret_cast<elem_t*>(const_cast<__nv_bfloat16*>(v));
    p.output_ptr = reinterpret_cast<elem_t*>(o);
    p.logsumexp_ptr    = nullptr;   // fwd-only, not needed
    p.output_accum_ptr = nullptr;   // single-value-iteration → unused

    p.scale            = scale;
    p.num_heads        = 1;         // we fold heads into the batch dim
    p.num_batches      = BH;
    p.head_dim         = DH;
    p.head_dim_value   = DH;
    p.num_queries      = S;
    p.num_keys         = S;
    p.custom_mask_type = Attention::NoCustomMask;   // non-causal

    // [BH, S, DH] with num_heads=1: token stride = DH, batch stride = S*DH.
    p.q_strideH = DH;  p.k_strideH = DH;  p.v_strideH = DH;
    p.q_strideM = DH;  p.k_strideM = DH;  p.v_strideM = DH;
    p.q_strideB = (int64_t)S * DH;
    p.k_strideB = (int64_t)S * DH;
    p.v_strideB = (int64_t)S * DH;
    p.o_strideM = DH;  // head_dim_value * num_heads

    constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
    int smem_bytes = sizeof(typename Attention::SharedStorage);
    // opt-in to >48kb dynamic smem once per process (not during graph capture:
    // the decoder warmup launches fire before any capture begins).
    static bool smem_opted_in = false;
    if (!smem_opted_in && smem_bytes > 0xc000) {
        cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        smem_opted_in = true;
    }
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes, stream>>>(p);
}

} // namespace dvlt
