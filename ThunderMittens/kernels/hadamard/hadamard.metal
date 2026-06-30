#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

template <typename T, int D>
kernel void hadamard(device const T *x [[buffer(0)]],
                     device T *out [[buffer(1)]],
                     constant float &scale [[buffer(2)]],
                     uint row [[threadgroup_position_in_grid]],
                     uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float buf[D];

    const long base = (long)row * D;
    buf[tid] = float(x[base + tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int h = 1; h < D; h <<= 1) {
        if ((tid & h) == 0) {
            const uint j = tid;
            const float a = buf[j];
            const float b = buf[j + h];
            buf[j] = a + b;
            buf[j + h] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    out[base + tid] = T(buf[tid] * scale);
}

#define instantiate_hadamard(type_name, T, DVAL)                            \
  template [[host_name("hadamard_" #type_name "_" #DVAL)]] [[kernel]] void \
  hadamard<T, DVAL>(device const T *x [[buffer(0)]],                        \
                    device T *out [[buffer(1)]],                            \
                    constant float &scale [[buffer(2)]],                    \
                    uint row [[threadgroup_position_in_grid]],              \
                    uint tid [[thread_position_in_threadgroup]]);

#define instantiate_hadamard_type(type_name, T) \
  instantiate_hadamard(type_name, T, 64)        \
  instantiate_hadamard(type_name, T, 128)       \
  instantiate_hadamard(type_name, T, 256)       \
  instantiate_hadamard(type_name, T, 512)

instantiate_hadamard_type(float32, float)
instantiate_hadamard_type(float16, half)
instantiate_hadamard_type(bfloat16, bf16)
