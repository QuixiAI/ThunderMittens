#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

// ---------------------------------------------------------------------------
// Sampling kernels: keep the final decode step on-GPU. One simdgroup (32 lanes)
// per row of logits (vocab dimension V, any size, looped with stride 32).
//
// P1 — threadgroup argmax-with-index: cross-lane max carrying the index, ties
// broken toward the smaller index (matches numpy argmax / TRT-LLM TopK_2).
// ---------------------------------------------------------------------------

constant float SMP_NEG_INF = -3.4028234663852886e38f;

constant int SAMPLE_MAX_K = 64;

// Butterfly argmax-with-index all-reduce: every lane ends with (max val, its idx),
// ties toward the smaller index. (Lane 0 holds the result too, so the single-winner
// kernels still write correctly; the all-reduce form lets top-k mask the winner.)
static inline void simd_argmax(thread float &val, thread int &idx) {
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = simd_shuffle_xor(val, off);
        const int oi = simd_shuffle_xor(idx, off);
        if (ov > val || (ov == val && oi < idx)) {
            val = ov;
            idx = oi;
        }
    }
}

// P4 — counter-based hash RNG (reproducible, NOT cryptographic). u(seed,row,i) in [0,1).
// A Murmur3-style integer finalizer over a mixed counter; replicated bit-for-bit in numpy
// so stochastic sampling has an exact, deterministic oracle.
static inline float rng_uniform(uint seed, uint row, uint i) {
    uint x = seed * 0x9E3779B9u + row * 0x85EBCA77u + i * 0xC2B2AE3Du;
    x ^= x >> 16; x *= 0x7FEB352Du;
    x ^= x >> 15; x *= 0x846CA68Bu;
    x ^= x >> 16;
    return float(x >> 8) * (1.0f / 16777216.0f);   // 24-bit mantissa -> [0,1)
}

template <typename T>
kernel void argmax(device const T *logits  [[buffer(0)]],
                   device int     *out_idx [[buffer(1)]],
                   constant int   &V       [[buffer(2)]],
                   uint row  [[threadgroup_position_in_grid]],
                   uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * V;
    float best = SMP_NEG_INF;
    int bi = (int)lane < V ? (int)lane : 0;
    for (int i = (int)lane; i < V; i += 32) {
        const float v = float(logits[base + i]);
        if (v > best || (v == best && i < bi)) {
            best = v;
            bi = i;
        }
    }
    simd_argmax(best, bi);
    if (lane == 0) {
        out_idx[row] = bi;
    }
}

// Stochastic categorical sampling via the Gumbel-max trick:
//   token = argmax_i ( logits[i]/temperature + Gumbel_i ),  Gumbel_i = -log(-log(u_i))
// which samples exactly from softmax(logits/temperature). The draw is fully determined
// by (seed, row), so a numpy reference reproducing rng_uniform/Gumbel matches exactly.
template <typename T>
kernel void sample_categorical(device const T *logits  [[buffer(0)]],
                               device int     *out_idx [[buffer(1)]],
                               constant int   &V       [[buffer(2)]],
                               constant uint  &seed    [[buffer(3)]],
                               constant float &invtemp [[buffer(4)]],
                               uint row  [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * V;
    float best = SMP_NEG_INF;
    int bi = (int)lane < V ? (int)lane : 0;
    for (int i = (int)lane; i < V; i += 32) {
        const float u = rng_uniform(seed, (uint)row, (uint)i);
        const float g = -log(-log(max(u, 1e-20f)));     // Gumbel(0,1)
        const float p = float(logits[base + i]) * invtemp + g;
        if (p > best || (p == best && i < bi)) {
            best = p;
            bi = i;
        }
    }
    simd_argmax(best, bi);
    if (lane == 0) {
        out_idx[row] = bi;
    }
}

// Top-k sampling: restrict to the k highest-logit tokens, then Gumbel-max sample
// among them (== sampling from softmax over the top-k with temperature). The top-k
// is k iterations of argmax-with-masking; the draw is reproducible from (seed, row).
template <typename T>
kernel void top_k_sample(device const T *logits  [[buffer(0)]],
                         device int     *out_idx [[buffer(1)]],
                         constant int   &V       [[buffer(2)]],
                         constant int   &K       [[buffer(3)]],
                         constant uint  &seed    [[buffer(4)]],
                         constant float &invtemp [[buffer(5)]],
                         uint row  [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * V;
    int chosen_id[SAMPLE_MAX_K];
    float chosen_logit[SAMPLE_MAX_K];

    for (int k = 0; k < K; ++k) {
        float best = SMP_NEG_INF;
        int bi = (int)lane < V ? (int)lane : 0;
        for (int i = (int)lane; i < V; i += 32) {
            bool taken = false;
            for (int j = 0; j < k; ++j) {
                if (chosen_id[j] == i) { taken = true; }
            }
            if (taken) { continue; }
            const float v = float(logits[base + i]);
            if (v > best || (v == best && i < bi)) {
                best = v;
                bi = i;
            }
        }
        simd_argmax(best, bi);    // all lanes hold the k-th token
        chosen_id[k] = bi;
        chosen_logit[k] = best;
    }

    // Gumbel-max among the k selected tokens.
    float best = SMP_NEG_INF;
    int bi = chosen_id[0];
    for (int j = 0; j < K; ++j) {
        const float u = rng_uniform(seed, (uint)row, (uint)chosen_id[j]);
        const float g = -log(-log(max(u, 1e-20f)));
        const float p = chosen_logit[j] * invtemp + g;
        if (p > best || (p == best && chosen_id[j] < bi)) {
            best = p;
            bi = chosen_id[j];
        }
    }
    if (lane == 0) {
        out_idx[row] = bi;
    }
}

// Apply temperature + repetition/presence/frequency penalties to logits, given the
// generated token history. penalty_histogram builds per-row occurrence counts (P3
// atomics) over prev_tokens (T, L); apply_penalty then transforms each logit. Order
// matches vLLM: temperature, then (if seen) repetition, presence, frequency.
kernel void penalty_histogram(device const int *prev_tokens [[buffer(0)]],
                              device atomic_int *counts      [[buffer(1)]],
                              constant int &V  [[buffer(2)]],
                              constant int &L  [[buffer(3)]],
                              constant int &TL [[buffer(4)]],
                              uint tid [[thread_position_in_grid]]) {
    if ((int)tid >= TL) { return; }
    const int row = (int)tid / L;
    const int tok = prev_tokens[tid];
    if (tok >= 0 && tok < V) {
        atomic_fetch_add_explicit(&counts[(long)row * V + tok], 1, memory_order_relaxed);
    }
}

template <typename T>
kernel void apply_penalty(device const T     *logits   [[buffer(0)]],
                          device const int   *counts   [[buffer(1)]],
                          device T           *out      [[buffer(2)]],
                          constant int   &V        [[buffer(3)]],
                          constant float &invtemp  [[buffer(4)]],
                          constant float &rep      [[buffer(5)]],
                          constant float &presence [[buffer(6)]],
                          constant float &freq     [[buffer(7)]],
                          uint row  [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * V;
    for (int v = (int)lane; v < V; v += 32) {
        float ls = float(logits[base + v]) * invtemp;
        const int c = counts[base + v];
        if (c > 0) {
            ls = (ls < 0.0f) ? (ls * rep) : (ls / rep);
            ls -= presence;
            ls -= freq * float(c);
        }
        out[base + v] = T(ls);
    }
}

// Top-p (nucleus) sampling without a full sort: bisection on a (temperature-scaled)
// logit threshold L finds the smallest set {l >= L} whose softmax mass >= p (each
// step is one simd-reduction of the surviving mass), then Gumbel-max samples among
// those survivors. Reproducible from (seed, row). Temperature is applied before top-p.
template <typename T>
kernel void top_p_sample(device const T *logits  [[buffer(0)]],
                         device int     *out_idx [[buffer(1)]],
                         constant int   &V       [[buffer(2)]],
                         constant float &p       [[buffer(3)]],
                         constant uint  &seed    [[buffer(4)]],
                         constant float &invtemp [[buffer(5)]],
                         uint row  [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * V;

    // max (temperature-scaled) logit and softmax denominator Z (all lanes get the reductions).
    float mx = SMP_NEG_INF;
    for (int i = (int)lane; i < V; i += 32) {
        mx = max(mx, float(logits[base + i]) * invtemp);
    }
    mx = simd_max(mx);
    float Z = 0.0f;
    for (int i = (int)lane; i < V; i += 32) {
        Z += exp(float(logits[base + i]) * invtemp - mx);
    }
    Z = simd_sum(Z);

    // Bisect the threshold L: keep the largest L whose mass(L)=sum_{ls>=L} softmax >= p.
    // That yields the smallest nucleus with cumulative mass >= p.
    float lo = mx - 40.0f, hi = mx;
    for (int it = 0; it < 32; ++it) {
        const float mid = 0.5f * (lo + hi);
        float sm = 0.0f;
        for (int i = (int)lane; i < V; i += 32) {
            const float ls = float(logits[base + i]) * invtemp;
            if (ls >= mid) { sm += exp(ls - mx); }
        }
        sm = simd_sum(sm) / Z;
        if (sm >= p) { lo = mid; } else { hi = mid; }
    }
    const float L = lo;

    // Gumbel-max over the nucleus {ls >= L}.
    float best = SMP_NEG_INF;
    int bi = (int)lane < V ? (int)lane : 0;
    for (int i = (int)lane; i < V; i += 32) {
        const float ls = float(logits[base + i]) * invtemp;
        if (ls < L) { continue; }
        const float u = rng_uniform(seed, (uint)row, (uint)i);
        const float g = -log(-log(max(u, 1e-20f)));
        const float pert = ls + g;
        if (pert > best || (pert == best && i < bi)) {
            best = pert;
            bi = i;
        }
    }
    simd_argmax(best, bi);
    if (lane == 0) {
        out_idx[row] = bi;
    }
}

#define instantiate_sampling(type_name, T)                                     \
  template [[host_name("argmax_" #type_name)]] [[kernel]] void                 \
  argmax<T>(device const T *logits [[buffer(0)]],                             \
            device int *out_idx [[buffer(1)]],                                \
            constant int &V [[buffer(2)]],                                    \
            uint row [[threadgroup_position_in_grid]],                        \
            uint lane [[thread_index_in_simdgroup]]);                         \
  template [[host_name("top_k_sample_" #type_name)]] [[kernel]] void           \
  top_k_sample<T>(device const T *logits [[buffer(0)]],                       \
                  device int *out_idx [[buffer(1)]],                          \
                  constant int &V [[buffer(2)]],                              \
                  constant int &K [[buffer(3)]],                              \
                  constant uint &seed [[buffer(4)]],                          \
                  constant float &invtemp [[buffer(5)]],                      \
                  uint row [[threadgroup_position_in_grid]],                  \
                  uint lane [[thread_index_in_simdgroup]]);                   \
  template [[host_name("top_p_sample_" #type_name)]] [[kernel]] void           \
  top_p_sample<T>(device const T *logits [[buffer(0)]],                       \
                  device int *out_idx [[buffer(1)]],                          \
                  constant int &V [[buffer(2)]],                              \
                  constant float &p [[buffer(3)]],                            \
                  constant uint &seed [[buffer(4)]],                          \
                  constant float &invtemp [[buffer(5)]],                      \
                  uint row [[threadgroup_position_in_grid]],                  \
                  uint lane [[thread_index_in_simdgroup]]);                   \
  template [[host_name("apply_penalty_" #type_name)]] [[kernel]] void          \
  apply_penalty<T>(device const T *logits [[buffer(0)]],                      \
                   device const int *counts [[buffer(1)]],                    \
                   device T *out [[buffer(2)]],                               \
                   constant int &V [[buffer(3)]],                             \
                   constant float &invtemp [[buffer(4)]],                     \
                   constant float &rep [[buffer(5)]],                         \
                   constant float &presence [[buffer(6)]],                    \
                   constant float &freq [[buffer(7)]],                        \
                   uint row [[threadgroup_position_in_grid]],                 \
                   uint lane [[thread_index_in_simdgroup]]);                  \
  template [[host_name("sample_categorical_" #type_name)]] [[kernel]] void     \
  sample_categorical<T>(device const T *logits [[buffer(0)]],                 \
                        device int *out_idx [[buffer(1)]],                    \
                        constant int &V [[buffer(2)]],                        \
                        constant uint &seed [[buffer(3)]],                    \
                        constant float &invtemp [[buffer(4)]],                \
                        uint row [[threadgroup_position_in_grid]],            \
                        uint lane [[thread_index_in_simdgroup]]);

instantiate_sampling(float32, float)
instantiate_sampling(float16, half)
instantiate_sampling(bfloat16, bf16)
