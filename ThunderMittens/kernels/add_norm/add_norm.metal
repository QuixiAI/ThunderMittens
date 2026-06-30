#include "tk.metal"
#include <metal_stdlib>

namespace mittens {

// ---------------------------------------------------------------------------
// Fused residual-add + normalization, bf16 I/O, fp32 compute.
//
// Every decoder block boundary computes `norm(x + residual)` and then feeds the
// *summed* residual (x + residual) into the next block. Fusing the add into the
// norm avoids an extra materialized add + global read/write of the hidden state.
//
// Two outputs:
//   o       = norm(x + residual) * weight (+ bias for LayerNorm)
//   res_out = x + residual                (the value the next block reads)
//
// Same register-resident, one-simdgroup-per-row structure as rms_norm/layernorm:
// the whole row of length D lives in registers (D/32 fp32 per lane), so there is
// no threadgroup memory and no barrier — the only cross-lane exchange is the
// warp-level `sum` reduction via simd shuffles. D ∈ {256,512,768,1024}.
//
// Ref: vLLM fused_add_rms_norm (layernorm_quant_kernels.cu), ONNX Runtime
// SkipLayerNorm (skip_layer_norm_impl.cu).
// ---------------------------------------------------------------------------
template <int D>
kernel void rms_norm_add(device   bf16  *x        [[buffer(0)]],
                         device   bf16  *residual  [[buffer(1)]],
                         device   bf16  *weight    [[buffer(2)]],
                         device   bf16  *o         [[buffer(3)]],
                         device   bf16  *res_out   [[buffer(4)]],
                         constant uint  &M         [[buffer(5)]],   // total rows = prod(shape[:-1])
                         constant float &eps       [[buffer(6)]],
                         uint3 blockIdx [[threadgroup_position_in_grid]],
                         uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D % TILE_DIM == 0, "D must be divisible by 8");
    const int row = blockIdx.x;

    using row_gl = gl<bf16, 1, 1, -1, D>;   // (M, D)
    using vec_gl = gl<bf16, 1, 1,  1, D>;   // (1, D) — weight
    row_gl gl_x(x,        nullptr, nullptr, M, nullptr);
    row_gl gl_r(residual, nullptr, nullptr, M, nullptr);
    row_gl gl_o(o,        nullptr, nullptr, M, nullptr);
    row_gl gl_ro(res_out, nullptr, nullptr, M, nullptr);
    vec_gl gl_w(weight,   nullptr, nullptr, nullptr, nullptr);

    using vecD = rv_fl<D>;                   // naive layout, fp32 compute
    vecD xv, rv, wv, sq;
    load(xv, gl_x, {0, 0, row, 0}, laneId);  // bf16 -> fp32 on the fly
    load(rv, gl_r, {0, 0, row, 0}, laneId);
    load(wv, gl_w, {0, 0, 0,   0}, laneId);

    add(xv, xv, rv);                         // x + residual
    store(gl_ro, xv, {0, 0, row, 0}, laneId); // write back x + residual (fp32 -> bf16)

    // mean of squares over the summed residual
    float ms = 0.f;
    mul(sq, xv, xv);
    sum(ms, sq, laneId);
    ms /= (float)D;
    float inv = metal::rsqrt(ms + eps);      // scalar rsqrt

    mul(xv, xv, inv);                        // * 1/rms (vec - scalar)
    mul(xv, xv, wv);                         // * weight (channelwise vec - vec)
    store(gl_o, xv, {0, 0, row, 0}, laneId); // fp32 -> bf16 on the fly
}

template <int D>
kernel void layernorm_add(device   bf16  *x        [[buffer(0)]],
                          device   bf16  *residual  [[buffer(1)]],
                          device   bf16  *weight    [[buffer(2)]],
                          device   bf16  *bias      [[buffer(3)]],
                          device   bf16  *o         [[buffer(4)]],
                          device   bf16  *res_out   [[buffer(5)]],
                          constant uint  &M         [[buffer(6)]],
                          constant float &eps       [[buffer(7)]],
                          uint3 blockIdx [[threadgroup_position_in_grid]],
                          uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D % TILE_DIM == 0, "D must be divisible by 8");
    const int row = blockIdx.x;

    using row_gl = gl<bf16, 1, 1, -1, D>;
    using vec_gl = gl<bf16, 1, 1,  1, D>;
    row_gl gl_x(x,        nullptr, nullptr, M, nullptr);
    row_gl gl_r(residual, nullptr, nullptr, M, nullptr);
    row_gl gl_o(o,        nullptr, nullptr, M, nullptr);
    row_gl gl_ro(res_out, nullptr, nullptr, M, nullptr);
    vec_gl gl_w(weight,   nullptr, nullptr, nullptr, nullptr);
    vec_gl gl_b(bias,     nullptr, nullptr, nullptr, nullptr);

    using vecD = rv_fl<D>;
    vecD xv, rv, wv, bv, sq;
    load(xv, gl_x, {0, 0, row, 0}, laneId);
    load(rv, gl_r, {0, 0, row, 0}, laneId);
    load(wv, gl_w, {0, 0, 0,   0}, laneId);
    load(bv, gl_b, {0, 0, 0,   0}, laneId);

    add(xv, xv, rv);                         // x + residual
    store(gl_ro, xv, {0, 0, row, 0}, laneId); // write back x + residual

    // mean
    float mean = 0.f;
    sum(mean, xv, laneId);
    mean /= (float)D;
    sub(xv, xv, mean);                       // x - mean

    // variance
    float var = 0.f;
    mul(sq, xv, xv);
    sum(var, sq, laneId);
    var /= (float)D;
    float inv = metal::rsqrt(var + eps);

    mul(xv, xv, inv);                        // * 1/std
    mul(xv, xv, wv);                         // * weight
    add(xv, xv, bv);                         // + bias
    store(gl_o, xv, {0, 0, row, 0}, laneId);
}

#define instantiate_rms_norm_add(DVAL)                                          \
  template [[host_name("rms_norm_add_" #DVAL)]] [[kernel]] void                 \
  rms_norm_add<DVAL>(device   bf16  *x        [[buffer(0)]],                    \
                     device   bf16  *residual  [[buffer(1)]],                   \
                     device   bf16  *weight    [[buffer(2)]],                   \
                     device   bf16  *o         [[buffer(3)]],                   \
                     device   bf16  *res_out   [[buffer(4)]],                   \
                     constant uint  &M         [[buffer(5)]],                   \
                     constant float &eps       [[buffer(6)]],                   \
                     uint3 blockIdx [[threadgroup_position_in_grid]],           \
                     uint  laneId   [[thread_index_in_simdgroup]]);

#define instantiate_layernorm_add(DVAL)                                         \
  template [[host_name("layernorm_add_" #DVAL)]] [[kernel]] void                \
  layernorm_add<DVAL>(device   bf16  *x        [[buffer(0)]],                   \
                      device   bf16  *residual  [[buffer(1)]],                  \
                      device   bf16  *weight    [[buffer(2)]],                  \
                      device   bf16  *bias      [[buffer(3)]],                  \
                      device   bf16  *o         [[buffer(4)]],                  \
                      device   bf16  *res_out   [[buffer(5)]],                  \
                      constant uint  &M         [[buffer(6)]],                  \
                      constant float &eps       [[buffer(7)]],                  \
                      uint3 blockIdx [[threadgroup_position_in_grid]],          \
                      uint  laneId   [[thread_index_in_simdgroup]]);

instantiate_rms_norm_add(256);
instantiate_rms_norm_add(512);
instantiate_rms_norm_add(768);
instantiate_rms_norm_add(1024);

instantiate_layernorm_add(256);
instantiate_layernorm_add(512);
instantiate_layernorm_add(768);
instantiate_layernorm_add(1024);

}
