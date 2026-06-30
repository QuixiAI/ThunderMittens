#include "tk.metal"
#include <metal_stdlib>

namespace mittens {

// ---------------------------------------------------------------------------
// Fused RoPE + paged-KV insert, bf16 I/O, fp32 compute.
//
// A serving prefill/decode step rotates the new K (split-half / GPT-NeoX RoPE,
// matching mx.fast.rope(traditional=False)) and writes the rotated K together
// with V straight into the paged KV cache at slot_mapping[token] — fusing what
// was three separate kernels (rotary + two cache scatters).
//
//   k1 = k[:D/2], k2 = k[D/2:]                  (per (token, kv_head) row)
//   ko1 = k1*cos - k2*sin ;  ko2 = k2*cos + k1*sin
//   key_cache[slot, kv_head]   = [ko1, ko2]     (rotated)
//   value_cache[slot, kv_head] = v              (unrotated)
//
// cos/sin are precomputed (P, D/2) and indexed by positions[token]. The paged
// cache (num_blocks, block_size, num_kv_heads, D) is contiguous, so it flattens
// to (R, D) and the destination row is
//   dst = (block * block_size + block_offset) * num_kv_heads + kv_head.
//
// One simdgroup (32 lanes) handles one (token, kv_head) row, flattened to
// (M = num_tokens * num_kv_heads, D). slot_mapping[token] < 0 skips the token.
// Q-RoPE is the existing `rotary` kernel (Q is not written to the cache).
//
// Ref: vLLM fused_*_qknorm_rope_kv_insert kernels (warp = one (token, head-slot)).
// ---------------------------------------------------------------------------
template <int D>
kernel void rope_kv_insert(device   bf16 *k            [[buffer(0)]],
                           device   bf16 *v            [[buffer(1)]],
                           device   bf16 *cosb         [[buffer(2)]],
                           device   bf16 *sinb         [[buffer(3)]],
                           device   const int  *positions    [[buffer(4)]],
                           device   const long *slot_mapping [[buffer(5)]],
                           device   bf16 *key_cache    [[buffer(6)]],
                           device   bf16 *value_cache  [[buffer(7)]],
                           constant int  &num_kv_heads [[buffer(8)]],
                           constant int  &block_size   [[buffer(9)]],
                           uint3 blockIdx [[threadgroup_position_in_grid]],
                           uint  laneId   [[thread_index_in_simdgroup]]) {
    constexpr int D2 = D / 2;
    static_assert(D2 % TILE_DIM == 0, "D/2 must be divisible by 8");

    const int row = blockIdx.x;                  // (token, kv_head) flattened
    const int token = row / num_kv_heads;
    const int kv_head = row % num_kv_heads;

    const long slot = slot_mapping[token];
    if (slot < 0) {
        return;                                  // padding token — skip
    }
    const long block = slot / block_size;
    const long block_offset = slot % block_size;
    const long dst_row =
        (block * block_size + block_offset) * (long)num_kv_heads + (long)kv_head;
    const int pos = positions[token];

    using row_gl = gl<bf16, 1, 1, -1, D>;        // (M, D) source / (R, D) cache
    using cs_gl  = gl<bf16, 1, 1, -1, D2>;       // (P, D/2)
    row_gl gl_k(k,           nullptr, nullptr, 1, nullptr);
    row_gl gl_v(v,           nullptr, nullptr, 1, nullptr);
    row_gl gl_kc(key_cache,  nullptr, nullptr, 1, nullptr);
    row_gl gl_vc(value_cache,nullptr, nullptr, 1, nullptr);
    cs_gl  gl_c(cosb,        nullptr, nullptr, 1, nullptr);
    cs_gl  gl_s(sinb,        nullptr, nullptr, 1, nullptr);

    using vecH = rv_fl<D2>;
    vecH k1, k2, cv, sv, o1, o2, tmp;
    load(k1, gl_k, {0, 0, row, 0}, laneId);      // first half of K
    load(k2, gl_k, {0, 0, row, 1}, laneId);      // second half of K
    load(cv, gl_c, {0, 0, pos, 0}, laneId);
    load(sv, gl_s, {0, 0, pos, 0}, laneId);

    // ko1 = k1*cos - k2*sin
    mul(o1, k1, cv);
    mul(tmp, k2, sv);
    sub(o1, o1, tmp);
    // ko2 = k2*cos + k1*sin
    mul(o2, k2, cv);
    mul(tmp, k1, sv);
    add(o2, o2, tmp);

    // Rotated K -> paged cache row.
    store(gl_kc, o1, {0, 0, (int)dst_row, 0}, laneId);
    store(gl_kc, o2, {0, 0, (int)dst_row, 1}, laneId);

    // V (unrotated) -> paged cache row.
    using vecD = rv_fl<D>;
    vecD vv;
    load(vv, gl_v, {0, 0, row, 0}, laneId);
    store(gl_vc, vv, {0, 0, (int)dst_row, 0}, laneId);
}

#define instantiate_rope_kv_insert(DVAL)                                       \
  template [[host_name("rope_kv_insert_" #DVAL)]] [[kernel]] void              \
  rope_kv_insert<DVAL>(device   bf16 *k            [[buffer(0)]],              \
                       device   bf16 *v            [[buffer(1)]],              \
                       device   bf16 *cosb         [[buffer(2)]],             \
                       device   bf16 *sinb         [[buffer(3)]],             \
                       device   const int  *positions    [[buffer(4)]],       \
                       device   const long *slot_mapping [[buffer(5)]],       \
                       device   bf16 *key_cache    [[buffer(6)]],             \
                       device   bf16 *value_cache  [[buffer(7)]],             \
                       constant int  &num_kv_heads [[buffer(8)]],             \
                       constant int  &block_size   [[buffer(9)]],             \
                       uint3 blockIdx [[threadgroup_position_in_grid]],        \
                       uint  laneId   [[thread_index_in_simdgroup]]);

instantiate_rope_kv_insert(64);
instantiate_rope_kv_insert(128);

}
