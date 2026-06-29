#include <metal_stdlib>
#include "tk.metal"

namespace mittens {

// Complex GEMM:  D = A @ B  for complex A,B,D.  Each array carries a leading size-2
// axis (index 0 = real plane, 1 = imag plane): A is (2,N,K), B (2,K,M), D (2,N,M).
// Exercises the complex-multiply MMA (complex_mma_AB) — the core building block for
// fftconv. Fixed <4,2,4> tiling (32x16x32 block); shapes need N%32, M%32, K%16.
template<typename T, unsigned N_BLOCK, unsigned K_BLOCK, unsigned M_BLOCK>
kernel void cmplx_matmul(
    device   T*   D [[buffer(0)]],          // (2,N,M)
    device   T*   A [[buffer(1)]],          // (2,N,K)
    device   T*   B [[buffer(2)]],          // (2,K,M)
    const constant int &N [[buffer(3)]],
    const constant int &K [[buffer(4)]],
    const constant int &M [[buffer(5)]],
    uint3 threadgroup_id [[threadgroup_position_in_grid]],
    uint  simd_lane_id   [[thread_index_in_simdgroup]]) {
    using gl_t = gl<T, 1, 1, -1, -1>;
    // real plane at base, imag plane offset by the matrix element count.
    gl_t gAr(A,            nullptr, nullptr, N, K);
    gl_t gAi(A + N * K,    nullptr, nullptr, N, K);
    gl_t gBr(B,            nullptr, nullptr, K, M);
    gl_t gBi(B + K * M,    nullptr, nullptr, K, M);
    gl_t gDr(D,            nullptr, nullptr, N, M);
    gl_t gDi(D + N * M,    nullptr, nullptr, N, M);

    constexpr const int N_BE = N_BLOCK * TILE_DIM;
    constexpr const int K_BE = K_BLOCK * TILE_DIM;
    constexpr const int M_BE = M_BLOCK * TILE_DIM;
    crt<T, N_BE, K_BE, ducks::rt_layout::row> a;
    crt<T, K_BE, M_BE, ducks::rt_layout::row> b;
    crt<float, N_BE, M_BE, ducks::rt_layout::row> d;
    zero(d.real);
    zero(d.imag);

    const int Y = threadgroup_id.y;
    const int X = threadgroup_id.x;
    for (int k = 0; k < K / K_BE; k++) {
        load(a.real, gAr, {0, 0, Y, k}, simd_lane_id);
        load(a.imag, gAi, {0, 0, Y, k}, simd_lane_id);
        load(b.real, gBr, {0, 0, k, X}, simd_lane_id);
        load(b.imag, gBi, {0, 0, k, X}, simd_lane_id);
        complex_mma_AB(d, a, b, d);
    }
    store(gDr, d.real, {0, 0, Y, X}, simd_lane_id);
    store(gDi, d.imag, {0, 0, Y, X}, simd_lane_id);
}

#define instantiate_cmplx_matmul(type_name, T)                                \
   template [[host_name("cmplx_matmul_" #type_name)]] [[kernel]]              \
   void cmplx_matmul<T, 4, 2, 4>(                                            \
     device T* D [[buffer(0)]], device T* A [[buffer(1)]], device T* B [[buffer(2)]], \
     const constant int &N [[buffer(3)]], const constant int &K [[buffer(4)]], \
     const constant int &M [[buffer(5)]],                                     \
     uint3 threadgroup_id [[threadgroup_position_in_grid]],                  \
     uint simd_lane_id [[thread_index_in_simdgroup]]);

instantiate_cmplx_matmul(float32, float);
instantiate_cmplx_matmul(bfloat16, bf16);

}
