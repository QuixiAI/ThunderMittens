#include "tk.metal"
#include <metal_stdlib>

namespace mittens {

// ---------------------------------------------------------------------------
// RMSNorm (forward), bf16 I/O, fp32 compute.
//
//   y = x * rsqrt(mean(x^2) + eps) * weight
//
// Like LayerNorm but with no mean-subtraction and no bias. One simdgroup (32
// lanes) processes one row of length D; the whole row fits in registers, so
// there is no threadgroup memory and no barrier (cross-lane reduction is via
// simd shuffles inside `sum`).
// ---------------------------------------------------------------------------
template <int D>
kernel void rms_norm(device   bf16  *x      [[buffer(0)]],
                     device   bf16  *weight [[buffer(1)]],
                     device   bf16  *o      [[buffer(2)]],
                     constant uint  &M      [[buffer(3)]],   // total rows = prod(shape[:-1])
                     constant float &eps    [[buffer(4)]],
                     uint3 blockIdx [[threadgroup_position_in_grid]],
                     uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D % TILE_DIM == 0, "D must be divisible by 8");
    const int row = blockIdx.x;

    using row_gl = gl<bf16, 1, 1, -1, D>;   // (M, D)
    using vec_gl = gl<bf16, 1, 1,  1, D>;   // (1, D) — weight
    row_gl gl_x(x, nullptr, nullptr, M, nullptr);
    row_gl gl_o(o, nullptr, nullptr, M, nullptr);
    vec_gl gl_w(weight, nullptr, nullptr, nullptr, nullptr);

    using vecD = rv_fl<D>;                   // naive layout, fp32 compute
    vecD xv, wv, sq;
    load(xv, gl_x, {0, 0, row, 0}, laneId);  // bf16 -> fp32 on the fly
    load(wv, gl_w, {0, 0, 0,   0}, laneId);

    // mean of squares
    float ms = 0.f;
    mul(sq, xv, xv);
    sum(ms, sq, laneId);
    ms /= (float)D;
    float inv = metal::rsqrt(ms + eps);      // scalar rsqrt

    // normalize, scale
    mul(xv, xv, inv);                        // * 1/rms (vec - scalar)
    mul(xv, xv, wv);                         // * weight (channelwise vec - vec)
    store(gl_o, xv, {0, 0, row, 0}, laneId); // fp32 -> bf16 on the fly
}

#define instantiate_rms_norm(DVAL)                                            \
  template [[host_name("rms_norm_" #DVAL)]] [[kernel]] void                   \
  rms_norm<DVAL>(device   bf16  *x      [[buffer(0)]],                        \
                 device   bf16  *weight [[buffer(1)]],                        \
                 device   bf16  *o      [[buffer(2)]],                        \
                 constant uint  &M      [[buffer(3)]],                        \
                 constant float &eps    [[buffer(4)]],                        \
                 uint3 blockIdx [[threadgroup_position_in_grid]],             \
                 uint  laneId   [[thread_index_in_simdgroup]]);

instantiate_rms_norm(256);
instantiate_rms_norm(512);
instantiate_rms_norm(768);
instantiate_rms_norm(1024);

}
