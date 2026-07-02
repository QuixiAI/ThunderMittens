#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

// ---------------------------------------------------------------------------
// Fused cross-entropy fwd/bwd over the vocab axis. One simdgroup (32 lanes) per row,
// looping the vocab dim with stride 32. Never stores the (T, V) probabilities:
//   fwd emits per-row loss + lse (log-sum-exp), bwd recomputes p = exp(x - lse) on
//   the fly. Supports ignore_index (masked rows -> 0 loss / 0 grad), label smoothing,
//   and a z-loss regularizer (z_loss * lse^2). lse is in the NATURAL-log domain.
// ---------------------------------------------------------------------------

constant float CE_NEG_INF = -3.4028234663852886e38f;

template <typename T>
kernel void cross_entropy_fwd(device const T   *logits          [[buffer(0)]],  // (Tn, V)
                              device const int *targets         [[buffer(1)]],  // (Tn,)
                              device float     *loss            [[buffer(2)]],  // (Tn,)
                              device float     *lse_out         [[buffer(3)]],  // (Tn,)
                              constant int   &V                 [[buffer(4)]],
                              constant int   &ignore_index      [[buffer(5)]],
                              constant float &label_smoothing   [[buffer(6)]],
                              constant float &z_loss            [[buffer(7)]],
                              uint row  [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * V;
    const int y = targets[row];
    if (y == ignore_index) {
        if (lane == 0) { loss[row] = 0.0f; lse_out[row] = 0.0f; }
        return;
    }
    // Per-lane online (max, sumexp) over strided elements, plus sum of logits (label smoothing).
    float m = CE_NEG_INF, l = 0.0f, sx = 0.0f;
    for (int i = (int)lane; i < V; i += 32) {
        const float x = float(logits[base + i]);
        sx += x;
        const float nm = max(m, x);
        l = l * exp(m - nm) + exp(x - nm);
        m = nm;
    }
    const float M = simd_max(m);
    l = simd_sum(l * exp(m - M));
    sx = simd_sum(sx);
    const float lse = M + log(l);
    const float x_y = float(logits[base + y]);
    const float eps = label_smoothing;
    float ls = (1.0f - eps) * (lse - x_y);
    if (eps > 0.0f) ls += eps * (lse - sx / (float)V);
    if (z_loss > 0.0f) ls += z_loss * lse * lse;
    if (lane == 0) { loss[row] = ls; lse_out[row] = lse; }
}

template <typename T>
kernel void cross_entropy_bwd(device const T     *logits         [[buffer(0)]],  // (Tn, V)
                              device const int   *targets        [[buffer(1)]],  // (Tn,)
                              device const float *lse_in         [[buffer(2)]],  // (Tn,)
                              device const float *grad_out       [[buffer(3)]],  // (Tn,)
                              device T           *grad_logits    [[buffer(4)]],  // (Tn, V)
                              constant int   &V                  [[buffer(5)]],
                              constant int   &ignore_index       [[buffer(6)]],
                              constant float &label_smoothing    [[buffer(7)]],
                              constant float &z_loss             [[buffer(8)]],
                              uint row  [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * V;
    const int y = targets[row];
    if (y == ignore_index) {
        for (int i = (int)lane; i < V; i += 32) grad_logits[base + i] = T(0);
        return;
    }
    const float lse = lse_in[row];
    const float go = grad_out[row];
    const float eps = label_smoothing;
    const float zc = 1.0f + 2.0f * z_loss * lse;   // d(z_loss*lse^2)/dx folds through p
    const float smooth = eps / (float)V;
    for (int i = (int)lane; i < V; i += 32) {
        const float p = exp(float(logits[base + i]) - lse);
        float g = zc * p - smooth - (1.0f - eps) * ((i == y) ? 1.0f : 0.0f);
        grad_logits[base + i] = T(g * go);
    }
}

#define instantiate_cross_entropy(type_name, T)                                    \
  template [[host_name("cross_entropy_fwd_" #type_name)]] [[kernel]] void           \
  cross_entropy_fwd<T>(device const T *logits [[buffer(0)]],                        \
    device const int *targets [[buffer(1)]], device float *loss [[buffer(2)]],      \
    device float *lse_out [[buffer(3)]], constant int &V [[buffer(4)]],             \
    constant int &ignore_index [[buffer(5)]], constant float &label_smoothing [[buffer(6)]], \
    constant float &z_loss [[buffer(7)]],                                          \
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]); \
  template [[host_name("cross_entropy_bwd_" #type_name)]] [[kernel]] void           \
  cross_entropy_bwd<T>(device const T *logits [[buffer(0)]],                        \
    device const int *targets [[buffer(1)]], device const float *lse_in [[buffer(2)]], \
    device const float *grad_out [[buffer(3)]], device T *grad_logits [[buffer(4)]], \
    constant int &V [[buffer(5)]], constant int &ignore_index [[buffer(6)]],        \
    constant float &label_smoothing [[buffer(7)]], constant float &z_loss [[buffer(8)]], \
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]);

instantiate_cross_entropy(float32, float)
instantiate_cross_entropy(float16, half)
instantiate_cross_entropy(bfloat16, bf16)
