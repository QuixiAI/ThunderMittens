#include <metal_stdlib>
#include "tk.metal"

namespace mittens {

// Integer-path quantized GEMV (batch-1 decode), int8xint8 -> int32 accumulate, then scale.
// Apple has no int8 simdgroup_matrix, so this is the GEMV (per-lane idot4 + simd_sum) shape.
// Prefill stays on the dequant-to-half MMA (half tensor cores beat a hand-rolled int dot).

// ---- W8A8 / SmoothQuant: int8 weights (per-channel scale) x int8 activations (per-token scale).
//   out[n] = w_scale[n] * a_scale * sum_k Wq[n,k] * Xq[k]   (scales factor out of the int32 sum). ----
kernel void qgemv_w8a8(
    device   half*  D       [[buffer(0)]],   // (N, 1) output
    device   char*  Wq      [[buffer(1)]],   // (N, K) int8 weights, row-major
    device   char*  Xq      [[buffer(2)]],   // (K, 1) int8 activations
    device   half*  w_scale [[buffer(3)]],   // (N,) per-channel weight scale
    device   half*  a_scale [[buffer(4)]],   // (1,) per-tensor activation scale (one decode token)
    const constant int &N [[buffer(5)]],
    const constant int &K [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
    const int row = tgid.x;
    device const uint* wrow = (device const uint*)(Wq + (uint)row * K);  // 4 int8 / uint
    device const uint* xv   = (device const uint*)Xq;
    int acc = 0;
    for (int u = (int)lane; u < K / 4; u += 32) acc += idot4(wrow[u], xv[u]);  // dp4a
    acc = metal::simd_sum(acc);                                                // int32 reduce
    if (lane == 0) D[row] = half(float(acc) * float(w_scale[row]) * float(a_scale[0]));
}

// ---- BitNet W2A8: ternary 2-bit weights (per-group absmean scale) x int8 activations.
//   out[n] = a_scale * sum_g gscale[g] * sum_{k in g} (code-1) * Xq[k]   (per-group int32 sums). ----
kernel void qgemv_w2a8(
    device   half*  D       [[buffer(0)]],   // (N, 1) output
    device   uchar* Wq      [[buffer(1)]],   // (N, K/32) bitnet blocks
    device   char*  Xq      [[buffer(2)]],   // (K, 1) int8 activations
    device   half*  a_scale [[buffer(3)]],   // (1,) per-tensor activation scale
    const constant int &N [[buffer(4)]],
    const constant int &K [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
    const int row = tgid.x;
    const int bpr = K / bitnet::block_k;
    device const uchar* row_base = Wq + (uint)(row * bpr) * bitnet::block_bytes;
    float facc = 0.0f;
    for (int g = 0; g < bpr; g++) {
        device const uchar* base = row_base + (uint)g * bitnet::block_bytes;
        int iacc = 0;
        for (int k = (int)lane; k < bitnet::block_k; k += 32)
            iacc += bitnet::code(base, k) * (int)Xq[g * bitnet::block_k + k];
        iacc = metal::simd_sum(iacc);                          // int32 sum within the group
        facc += float(iacc) * float(bitnet::gscale(base));     // apply the per-group scale
    }
    if (lane == 0) D[row] = half(facc * float(a_scale[0]));
}

}
