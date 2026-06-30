#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

// ---------------------------------------------------------------------------
// Runtime per-token (per-row) activation quantization. These are the first
// GPU-side quantizers in ThunderMittens (everything in tk/quant.py is host numpy).
//
// One simdgroup (32 lanes) processes one row of length D (any D): cross-lane
// simd_max gives the per-row absmax, scale = absmax / QMAX, then each element is
// encoded (round-to-nearest) to fp8 e4m3 (QMAX=448) or symmetric int8 (QMAX=127).
//
//   codes[row, i] = encode(x[row, i] / scale[row]) ;  scale[row] = absmax(row) / QMAX
//
// Reconstruct as scale[row] * decode(codes[row, i]). Ref: vLLM
// dynamic_per_token_scaled_fp8_quant (quantization/w8a8/fp8/common.cu).
// ---------------------------------------------------------------------------

template <typename T>
kernel void quantize_per_token_fp8(device const T *x     [[buffer(0)]],
                                   device uchar   *codes [[buffer(1)]],
                                   device float   *scale [[buffer(2)]],
                                   constant int   &D     [[buffer(3)]],
                                   uint row  [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * D;
    float amax = 0.0f;
    for (int i = (int)lane; i < D; i += 32) {
        amax = max(amax, fabs(float(x[base + i])));
    }
    amax = simd_max(amax);
    const float s = amax / 448.0f;
    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
    for (int i = (int)lane; i < D; i += 32) {
        codes[base + i] = tk_e4m3_encode(float(x[base + i]) * inv);
    }
    if (lane == 0) {
        scale[row] = s;
    }
}

template <typename T>
kernel void quantize_per_token_int8(device const T *x     [[buffer(0)]],
                                    device char    *codes [[buffer(1)]],
                                    device float   *scale [[buffer(2)]],
                                    constant int   &D     [[buffer(3)]],
                                    uint row  [[threadgroup_position_in_grid]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * D;
    float amax = 0.0f;
    for (int i = (int)lane; i < D; i += 32) {
        amax = max(amax, fabs(float(x[base + i])));
    }
    amax = simd_max(amax);
    const float s = amax / 127.0f;
    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
    for (int i = (int)lane; i < D; i += 32) {
        codes[base + i] = tk_int8_encode(float(x[base + i]) * inv);
    }
    if (lane == 0) {
        scale[row] = s;
    }
}

#define instantiate_quant_rt(type_name, T)                                     \
  template [[host_name("quantize_per_token_fp8_" #type_name)]] [[kernel]] void \
  quantize_per_token_fp8<T>(device const T *x [[buffer(0)]],                   \
                            device uchar *codes [[buffer(1)]],                 \
                            device float *scale [[buffer(2)]],                 \
                            constant int &D [[buffer(3)]],                     \
                            uint row [[threadgroup_position_in_grid]],         \
                            uint lane [[thread_index_in_simdgroup]]);          \
  template [[host_name("quantize_per_token_int8_" #type_name)]] [[kernel]] void\
  quantize_per_token_int8<T>(device const T *x [[buffer(0)]],                  \
                             device char *codes [[buffer(1)]],                 \
                             device float *scale [[buffer(2)]],                \
                             constant int &D [[buffer(3)]],                    \
                             uint row [[threadgroup_position_in_grid]],        \
                             uint lane [[thread_index_in_simdgroup]]);

instantiate_quant_rt(float32, float)
instantiate_quant_rt(float16, half)
instantiate_quant_rt(bfloat16, bf16)
