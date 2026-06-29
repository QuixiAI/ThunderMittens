#include <metal_stdlib>
#include "tk.metal"

namespace mittens {

// Multi-simdgroup, threadgroup-staged GEMM.  D = A @ B,  A (N,K), B (K,M).
//
// A threadgroup has N_WARPS simdgroups and computes a (BN x BM) output block,
// BM = N_WARPS * BM_PER_WARP. The A block (BN x BK) is cooperatively staged into
// threadgroup memory once per K-step and reused by every warp (fewer global A
// reads); each warp loads its own B columns and accumulates its output sub-block.
//
// BN=32, BK=16, BM_PER_WARP=16 -> with N_WARPS=2, BM=32; shapes need N%32, M%32, K%16.
template<typename T, int N_WARPS>
kernel void gemm_staged(
    device   T*   D [[buffer(0)]],
    device   T*   A [[buffer(1)]],
    device   T*   B [[buffer(2)]],
    const constant int &N [[buffer(3)]],
    const constant int &K [[buffer(4)]],
    const constant int &M [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  warp [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]) {
    using G = group<N_WARPS>;
    constexpr const int BN = 32;
    constexpr const int BK = 16;
    constexpr const int BM_PER_WARP = 16;

    using gl_mat = gl<T, 1, 1, -1, -1>;
    gl_mat gl_a(A, nullptr, nullptr, N, K);
    gl_mat gl_b(B, nullptr, nullptr, K, M);
    gl_mat gl_d(D, nullptr, nullptr, N, M);

    threadgroup st<T, BN, BK> sA;         // staged A block, shared by all warps
    rt<T, BN, BK> a_reg;
    rt<T, BK, BM_PER_WARP> b_reg;
    rt<float, BN, BM_PER_WARP> d_reg;
    zero(d_reg);

    const int by = tgid.y;                              // output row block (BN rows)
    const int bx = tgid.x;                              // output col block (BM cols)
    const int col_block = bx * N_WARPS + (int)warp;     // this warp's BM_PER_WARP column block

    for (int k = 0; k < K / BK; k++) {
        G::load(sA, gl_a, {0, 0, by, k}, tid);          // cooperative global -> shared
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        load(a_reg, sA, lane);                           // shared -> register (full A block)
        load(b_reg, gl_b, {0, 0, k, col_block}, lane);   // this warp's B columns
        mma_AB(d_reg, a_reg, b_reg, d_reg);
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);  // before sA is overwritten
    }
    store(gl_d, d_reg, {0, 0, by, col_block}, lane);
}

#define instantiate_gemm_staged(type_name, T)                                 \
   template [[host_name("gemm_staged_" #type_name)]] [[kernel]]               \
   void gemm_staged<T, 2>(                                                    \
     device T* D [[buffer(0)]], device T* A [[buffer(1)]], device T* B [[buffer(2)]], \
     const constant int &N [[buffer(3)]], const constant int &K [[buffer(4)]], \
     const constant int &M [[buffer(5)]],                                     \
     uint3 tgid [[threadgroup_position_in_grid]],                            \
     uint tid [[thread_index_in_threadgroup]],                               \
     uint warp [[simdgroup_index_in_threadgroup]],                           \
     uint lane [[thread_index_in_simdgroup]]);

instantiate_gemm_staged(float32, float);
instantiate_gemm_staged(bfloat16, bf16);

}
