#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

// ---------------------------------------------------------------------------
// DeepSeek Multi-head Latent Attention (MLA) — preprocessing kernels.
//
// P1: mla_q_norm_rope — the fused Q-path. Per (token, head): optional RMSNorm over
// the full head dim (no-weight for the V4/V3.2 Q-norm, weighted for a kv_a-style
// norm, or none), then GPT-J *interleaved* RoPE on the last `rope_dim` dims (the
// `nope` prefix passes through), bf16 store. Head layout: head_dim = nope_dim +
// rope_dim (e.g. 192 = 128+64 for V2/V3, or 512 = 448+64 for V4).
//
// One warp (32 lanes) per (token, head). Each lane owns head_dim/32 CONTIGUOUS
// elements — even (head_dim % 64 == 0), so every interleaved pair (g, g+1) with g
// even is resident in a single lane (no cross-lane shuffle), and the full-head
// sum-of-squares is a per-lane contiguous sum + one simd_sum. nope_dim even ⇒ a
// pair never straddles the nope/rope boundary.
//
// cos/sin are separate (max_pos, rope_dim/2) bf16 tables (the ThunderMittens RoPE
// convention), indexed by positions[token]. Golden: rmsnorm_no_weight +
// apply_rope_gptj_last_k in vLLM's test_fused_deepseek_v4_qnorm_rope_kv_insert.
// ---------------------------------------------------------------------------
template <int D>
kernel void mla_q_norm_rope(device const bf16 *q          [[buffer(0)]],
                            device const bf16 *cosb        [[buffer(1)]],
                            device const bf16 *sinb        [[buffer(2)]],
                            device const int  *positions   [[buffer(3)]],
                            device bf16       *out         [[buffer(4)]],
                            constant int &num_heads        [[buffer(5)]],
                            constant int &nope_dim         [[buffer(6)]],
                            constant int &rope_dim         [[buffer(7)]],
                            constant int &norm_mode        [[buffer(8)]],   // 0 none,1 rms,2 rms+w
                            constant float &eps            [[buffer(9)]],
                            device const bf16 *norm_weight [[buffer(10)]],  // (D,), read iff mode 2
                            uint3 blockIdx [[threadgroup_position_in_grid]],
                            uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D % 64 == 0, "mla_q_norm_rope needs head_dim divisible by 64");
    constexpr int PER_LANE = D / 32;              // contiguous, even
    const int row = blockIdx.x;                   // (token, head) flattened
    const int token = row / num_heads;
    const int pos = positions[token];
    const int rope_half = rope_dim / 2;
    const long base = (long)row * D + (long)laneId * PER_LANE;

    // Full-head RMS (no-weight) if requested.
    float rms = 1.0f;
    if (norm_mode != 0) {
        float ss = 0.0f;
        for (int k = 0; k < PER_LANE; ++k) { const float v = float(q[base + k]); ss += v * v; }
        ss = simd_sum(ss);
        rms = metal::rsqrt(ss / (float)D + eps);
    }

    const long wbase = (long)laneId * PER_LANE;   // norm_weight index for this lane's chunk
    const long csbase = (long)pos * rope_half;
    for (int k = 0; k < PER_LANE; k += 2) {
        const int g0 = (int)laneId * PER_LANE + k;   // even global index (start of a pair)
        float v0 = float(q[base + k]) * rms;
        float v1 = float(q[base + k + 1]) * rms;
        if (norm_mode == 2) {
            v0 *= float(norm_weight[wbase + k]);
            v1 *= float(norm_weight[wbase + k + 1]);
        }
        if (g0 >= nope_dim) {
            const int p = (g0 - nope_dim) / 2;       // rope pair index
            const float c = float(cosb[csbase + p]);
            const float s = float(sinb[csbase + p]);
            out[base + k]     = bf16(v0 * c - v1 * s);
            out[base + k + 1] = bf16(v0 * s + v1 * c);
        } else {
            out[base + k]     = bf16(v0);
            out[base + k + 1] = bf16(v1);
        }
    }
}

#define instantiate_mla_q_norm_rope(DVAL)                                      \
  template [[host_name("mla_q_norm_rope_" #DVAL)]] [[kernel]] void             \
  mla_q_norm_rope<DVAL>(device const bf16 *q [[buffer(0)]],                    \
                        device const bf16 *cosb [[buffer(1)]],                 \
                        device const bf16 *sinb [[buffer(2)]],                 \
                        device const int  *positions [[buffer(3)]],            \
                        device bf16       *out [[buffer(4)]],                  \
                        constant int &num_heads [[buffer(5)]],                 \
                        constant int &nope_dim [[buffer(6)]],                  \
                        constant int &rope_dim [[buffer(7)]],                  \
                        constant int &norm_mode [[buffer(8)]],                 \
                        constant float &eps [[buffer(9)]],                     \
                        device const bf16 *norm_weight [[buffer(10)]],         \
                        uint3 blockIdx [[threadgroup_position_in_grid]],       \
                        uint  laneId   [[thread_index_in_simdgroup]]);

instantiate_mla_q_norm_rope(128);
instantiate_mla_q_norm_rope(192);
instantiate_mla_q_norm_rope(256);
instantiate_mla_q_norm_rope(512);
