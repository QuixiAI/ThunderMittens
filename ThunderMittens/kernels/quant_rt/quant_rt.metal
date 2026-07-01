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
    using T4 = vec<T, 4>;
    const long base = (long)row * D;
    const int nchunks = (D % 4 == 0) ? D / 4 : 0;   // vec4 path only for aligned D
    float amax = 0.0f;
    for (int c = (int)lane; c < nchunks; c += 32) {
        const float4 v = float4(((device const T4*)(x + base))[c]);
        amax = max(amax, max(max(fabs(v.x), fabs(v.y)), max(fabs(v.z), fabs(v.w))));
    }
    for (int i = nchunks * 4 + (int)lane; i < D; i += 32) {
        amax = max(amax, fabs(float(x[base + i])));
    }
    amax = simd_max(amax);
    const float s = amax / 448.0f;
    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
    for (int c = (int)lane; c < nchunks; c += 32) {
        const float4 v = float4(((device const T4*)(x + base))[c]) * inv;
        ((device uchar4*)(codes + base))[c] =
            uchar4(tk_e4m3_encode(v.x), tk_e4m3_encode(v.y), tk_e4m3_encode(v.z), tk_e4m3_encode(v.w));
    }
    for (int i = nchunks * 4 + (int)lane; i < D; i += 32) {
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
    using T4 = vec<T, 4>;
    const long base = (long)row * D;
    const int nchunks = (D % 4 == 0) ? D / 4 : 0;
    float amax = 0.0f;
    for (int c = (int)lane; c < nchunks; c += 32) {
        const float4 v = float4(((device const T4*)(x + base))[c]);
        amax = max(amax, max(max(fabs(v.x), fabs(v.y)), max(fabs(v.z), fabs(v.w))));
    }
    for (int i = nchunks * 4 + (int)lane; i < D; i += 32) {
        amax = max(amax, fabs(float(x[base + i])));
    }
    amax = simd_max(amax);
    const float s = amax / 127.0f;
    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
    for (int c = (int)lane; c < nchunks; c += 32) {
        const float4 v = float4(((device const T4*)(x + base))[c]) * inv;
        ((device char4*)(codes + base))[c] =
            char4(tk_int8_encode(v.x), tk_int8_encode(v.y), tk_int8_encode(v.z), tk_int8_encode(v.w));
    }
    for (int i = nchunks * 4 + (int)lane; i < D; i += 32) {
        codes[base + i] = tk_int8_encode(float(x[base + i]) * inv);
    }
    if (lane == 0) {
        scale[row] = s;
    }
}

// ---------------------------------------------------------------------------
// Per-tensor (global) dynamic quantization. Two passes: (1) reduce the global
// absmax into an atomic_uint via the P3 order-preserving float mapping; (2) read
// it back, form scale = absmax/QMAX, and encode every element. Complements the
// per-row quantizers; exercises mittens::atomic_max_float.
// ---------------------------------------------------------------------------
template <typename T>
kernel void quant_tensor_absmax(device const T *x         [[buffer(0)]],
                                device atomic_uint *scale_u [[buffer(1)]],
                                constant int  &n          [[buffer(2)]],
                                uint tid  [[thread_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
    // 16 elements per thread (vec4 x4) -> 16x fewer contended atomics than the old
    // one-element-per-thread version, and vectorized loads.
    using T4 = vec<T, 4>;
    const long base = (long)tid * 16;
    float amax = 0.0f;
    if (base + 16 <= (long)n) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < 4; ++j) {
            const float4 v = float4(((device const T4*)(x + base))[j]);
            amax = max(amax, max(max(fabs(v.x), fabs(v.y)), max(fabs(v.z), fabs(v.w))));
        }
    } else {
        for (long i = base; i < (long)n; ++i) amax = max(amax, fabs(float(x[i])));
    }
    amax = simd_max(amax);
    if (lane == 0 && amax > 0.0f) { atomic_max_float(scale_u, amax); }   // P3
}

template <typename T>
kernel void quant_tensor_encode_fp8(device const T   *x         [[buffer(0)]],
                                    device const uint *scale_u  [[buffer(1)]],
                                    device uchar     *codes     [[buffer(2)]],
                                    device float     *scale_out [[buffer(3)]],
                                    constant int     &n         [[buffer(4)]],
                                    uint tid [[thread_position_in_grid]]) {
    using T4 = vec<T, 4>;
    const float s = orderable_uint_to_float(scale_u[0]) / 448.0f;
    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
    if (tid == 0) { scale_out[0] = s; }
    const long base = (long)tid * 4;
    if (base + 4 <= (long)n) {
        const float4 v = float4(((device const T4*)(x + base))[0]) * inv;
        ((device uchar4*)(codes + base))[0] =
            uchar4(tk_e4m3_encode(v.x), tk_e4m3_encode(v.y), tk_e4m3_encode(v.z), tk_e4m3_encode(v.w));
    } else {
        for (long i = base; i < (long)n; ++i) codes[i] = tk_e4m3_encode(float(x[i]) * inv);
    }
}

template <typename T>
kernel void quant_tensor_encode_int8(device const T   *x         [[buffer(0)]],
                                     device const uint *scale_u  [[buffer(1)]],
                                     device char      *codes     [[buffer(2)]],
                                     device float     *scale_out [[buffer(3)]],
                                     constant int     &n         [[buffer(4)]],
                                     uint tid [[thread_position_in_grid]]) {
    using T4 = vec<T, 4>;
    const float s = orderable_uint_to_float(scale_u[0]) / 127.0f;
    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
    if (tid == 0) { scale_out[0] = s; }
    const long base = (long)tid * 4;
    if (base + 4 <= (long)n) {
        const float4 v = float4(((device const T4*)(x + base))[0]) * inv;
        ((device char4*)(codes + base))[0] =
            char4(tk_int8_encode(v.x), tk_int8_encode(v.y), tk_int8_encode(v.z), tk_int8_encode(v.w));
    } else {
        for (long i = base; i < (long)n; ++i) codes[i] = tk_int8_encode(float(x[i]) * inv);
    }
}

#define instantiate_quant_tensor(type_name, T)                                 \
  template [[host_name("quant_tensor_absmax_" #type_name)]] [[kernel]] void     \
  quant_tensor_absmax<T>(device const T *x [[buffer(0)]],                       \
                         device atomic_uint *scale_u [[buffer(1)]],             \
                         constant int &n [[buffer(2)]],                         \
                         uint tid [[thread_position_in_grid]],                  \
                         uint lane [[thread_index_in_simdgroup]]);              \
  template [[host_name("quant_tensor_encode_fp8_" #type_name)]] [[kernel]] void \
  quant_tensor_encode_fp8<T>(device const T *x [[buffer(0)]],                   \
                             device const uint *scale_u [[buffer(1)]],          \
                             device uchar *codes [[buffer(2)]],                 \
                             device float *scale_out [[buffer(3)]],             \
                             constant int &n [[buffer(4)]],                     \
                             uint tid [[thread_position_in_grid]]);             \
  template [[host_name("quant_tensor_encode_int8_" #type_name)]] [[kernel]] void\
  quant_tensor_encode_int8<T>(device const T *x [[buffer(0)]],                  \
                              device const uint *scale_u [[buffer(1)]],         \
                              device char *codes [[buffer(2)]],                 \
                              device float *scale_out [[buffer(3)]],            \
                              constant int &n [[buffer(4)]],                    \
                              uint tid [[thread_position_in_grid]]);

instantiate_quant_tensor(float32, float)
instantiate_quant_tensor(float16, half)
instantiate_quant_tensor(bfloat16, bf16)

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
