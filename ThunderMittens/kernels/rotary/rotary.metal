#include "tk.metal"
#include <metal_stdlib>

namespace mittens {

// ---------------------------------------------------------------------------
// Rotary positional embedding (RoPE), split-half / GPT-NeoX convention,
// matching mx.fast.rope(..., traditional=False). bf16 I/O, fp32 compute.
//
// With halves x1 = x[..., :D/2], x2 = x[..., D/2:] and per-position cos/sin of
// shape (N, D/2):
//     o1 = x1*cos - x2*sin
//     o2 = x2*cos + x1*sin
//
// cos/sin are precomputed and passed in (the kernel needs no trig op). One
// simdgroup (32 lanes) processes one (b,h,n) row; x is flattened to (M, D) with
// M = B*H*N, and the sequence position is n = row % N.
// ---------------------------------------------------------------------------
template <int D>
kernel void rotary(device   bf16 *x    [[buffer(0)]],
                   device   bf16 *cosb [[buffer(1)]],
                   device   bf16 *sinb [[buffer(2)]],
                   device   bf16 *o    [[buffer(3)]],
                   constant uint &N    [[buffer(4)]],   // sequence length
                   uint3 blockIdx [[threadgroup_position_in_grid]],
                   uint  laneId   [[thread_index_in_simdgroup]]) {
    constexpr int D2 = D / 2;
    static_assert(D2 % TILE_DIM == 0, "D/2 must be divisible by 8");
    const int row = blockIdx.x;
    const int n = row % (int)N;   // sequence position for this row

    // x/out are (M, D); the two halves are columns [0,D/2) and [D/2,D) selected
    // by coord .c (get<VEC> offsets by c * VEC::length = c * D/2). rows is
    // unused for indexing when b=d=0, so a dummy of 1 is fine.
    using row_gl = gl<bf16, 1, 1, -1, D>;
    using cs_gl  = gl<bf16, 1, 1, -1, D2>;   // (N, D/2)
    row_gl gl_x(x, nullptr, nullptr, 1, nullptr);
    row_gl gl_o(o, nullptr, nullptr, 1, nullptr);
    cs_gl  gl_c(cosb, nullptr, nullptr, N, nullptr);
    cs_gl  gl_s(sinb, nullptr, nullptr, N, nullptr);

    using vecH = rv_fl<D2>;
    vecH x1, x2, cv, sv, o1, o2, tmp;
    load(x1, gl_x, {0, 0, row, 0}, laneId);   // first half
    load(x2, gl_x, {0, 0, row, 1}, laneId);   // second half
    load(cv, gl_c, {0, 0, n,   0}, laneId);
    load(sv, gl_s, {0, 0, n,   0}, laneId);

    // o1 = x1*cos - x2*sin
    mul(o1, x1, cv);
    mul(tmp, x2, sv);
    sub(o1, o1, tmp);
    // o2 = x2*cos + x1*sin
    mul(o2, x2, cv);
    mul(tmp, x1, sv);
    add(o2, o2, tmp);

    store(gl_o, o1, {0, 0, row, 0}, laneId);
    store(gl_o, o2, {0, 0, row, 1}, laneId);
}

#define instantiate_rotary(DVAL)                                              \
  template [[host_name("rotary_" #DVAL)]] [[kernel]] void                     \
  rotary<DVAL>(device   bf16 *x    [[buffer(0)]],                             \
               device   bf16 *cosb [[buffer(1)]],                            \
               device   bf16 *sinb [[buffer(2)]],                            \
               device   bf16 *o    [[buffer(3)]],                            \
               constant uint &N    [[buffer(4)]],                            \
               uint3 blockIdx [[threadgroup_position_in_grid]],              \
               uint  laneId   [[thread_index_in_simdgroup]]);

instantiate_rotary(64);
instantiate_rotary(128);

}
