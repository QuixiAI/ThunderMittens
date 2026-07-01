#include "tk.metal"
#include <metal_stdlib>
namespace mittens {

// Mamba-2 / SSD (selective state space) forward, materialized chunked form, bf16, D=64.
//   Y_t = sum_{j<=t} (C_t . B_j) * exp(cumlog_t - cumlog_j) * X_j
// where cumlog = cumsum(log a) is the running log-decay (precomputed on host, fp32).
// This is the SSD attention-equivalent: M = (C @ B^T) (.) L, L the decay-causal matrix,
// Y = M @ X. One simdgroup per (batch, head, query-chunk); loops over key-chunks <= query.
//
// The decay matrix L[i,j] = exp(cumlog_i[i] - cumlog_j[j]) is built from broadcasts:
// add_row(colvec cumlog_i) then sub_col(rowvec cumlog_j), then exp.
template <int D>
kernel void mamba2(device   bf16     *C  [[buffer(0)]],
                   device   bf16     *Bm [[buffer(1)]],
                   device   bf16     *X  [[buffer(2)]],
                   device   float    *cl [[buffer(3)]],   // cumlog (B,H,N), fp32
                   device   bf16     *Y  [[buffer(4)]],
                   constant unsigned &N  [[buffer(5)]],
                   constant unsigned &H  [[buffer(6)]],
                   uint3 blockIdx [[threadgroup_position_in_grid]],
                   uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D == 64, "mamba2 currently supports D=64");
    using gl_t  = gl<bfloat, 1, -1, -1, D>;           // C, B, X, Y : (B,H,N,D)
    using gl_cl = gl<float, 1, -1, 1, -1>;            // cumlog (B,H,N) viewed sequence-along-cols
    gl_t  gC(C, nullptr, H, N, nullptr);
    gl_t  gB(Bm, nullptr, H, N, nullptr);
    gl_t  gX(X, nullptr, H, N, nullptr);
    gl_t  gY(Y, nullptr, H, N, nullptr);
    // gl.get<VEC> offsets by idx.c * VEC::length, so index the sequence chunk via idx.c for
    // BOTH the col_vec and row_vec loads (a vec read pulls VEC::length contiguous values).
    gl_cl gcl(cl, nullptr, H, nullptr, N);

    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int qi = blockIdx.x;                         // this query chunk

    rt_bf<8, D> c_reg;
    load(c_reg, gC, {batch, head, qi, 0}, laneId);
    typename rt_fl<8, 8>::col_vec cumlog_i;             // per query row
    load(cumlog_i, gcl, {batch, head, 0, qi}, laneId);

    rt_fl<8, D> y_reg;
    zero(y_reg);

    for (int kj = 0; kj <= qi; kj++) {
        rt_bf<8, D, ducks::rt_layout::col> b_reg;       // col layout for C @ B^T
        rt_bf<8, D> x_reg;
        load(b_reg, gB, {batch, head, kj, 0}, laneId);
        load(x_reg, gX, {batch, head, kj, 0}, laneId);
        typename rt_fl<8, 8>::row_vec cumlog_j;          // per key col
        load(cumlog_j, gcl, {batch, head, 0, kj}, laneId);

        rt_fl<8, 8> att;
        zero(att);
        mma_ABt(att, c_reg, b_reg, att);                 // C @ B^T

        // decay_log[i,j] = cumlog_i[i] - cumlog_j[j]; decay = exp(decay_log)
        rt_fl<8, 8> decay;
        zero(decay);
        add_row(decay, decay, cumlog_i);
        sub_col(decay, decay, cumlog_j);
        exp(decay, decay);
        mul(att, att, decay);

        if (kj == qi) {
            float zero_fill = 0.0f;
            make_causal(att, att, laneId, zero_fill);    // future positions -> 0
        }

        rt_bf<8, 8> att_bf;
        copy(att_bf, att);
        mma_AB(y_reg, att_bf, x_reg, y_reg);
    }
    store(gY, y_reg, {batch, head, qi, 0}, laneId);
}

#define instantiate_mamba2(D)                                  \
  template [[host_name("mamba2_" #D)]] [[kernel]] void         \
  mamba2<D>(device bf16 *C [[buffer(0)]], device bf16 *Bm [[buffer(1)]], \
    device bf16 *X [[buffer(2)]], device float *cl [[buffer(3)]], \
    device bf16 *Y [[buffer(4)]], \
    constant unsigned &N [[buffer(5)]], constant unsigned &H [[buffer(6)]], \
    uint3 blockIdx [[threadgroup_position_in_grid]], \
    uint laneId [[thread_index_in_simdgroup]]);

instantiate_mamba2(64);

// ---------------------------------------------------------------------------
// Chunked linear-time SSD (3 kernels), shared by mamba2 AND lin_attn_decay
// (identical math; lin_attn_decay feeds cl = -slope*position). The kernel above
// is the QUADRATIC materialized form: every 8-row query tile rescans all
// earlier key tiles — O(N²·D) work. The chunked form is O(N·(L+D)·D):
//   K1 ssd_chunk_kv:   KV_c = sum_{j in chunk} exp(cl[r_c]-cl[j]) b_j^T x_j,
//                      referenced at r_c = last row of chunk c            (fp32)
//   K2 ssd_chunk_scan: S_c = sum_{c'<c} (prod decays) KV_c', re-referenced via
//                      the per-chunk factor L_c = exp(cl[r_c]-cl[r_{c-1}])
//   K3 ssd_chunk_out:  intra-chunk decay tiles (<= L keys, the loop above,
//                      chunk-bounded) + inter y += diag(exp(cl_i-cl[r_{c-1}]))
//                      * (C @ S_c)
// All exponents are <= 0 for a decreasing cumlog (a < 1), so everything stays
// bounded. Chunk L = 64 rows.
// ---------------------------------------------------------------------------
constant constexpr const int SSD_CHUNK_L = 64;

template <int D>
kernel void ssd_chunk_kv(device   bf16     *Bm [[buffer(0)]],
                         device   bf16     *X  [[buffer(1)]],
                         device   float    *cl [[buffer(2)]],
                         device   float    *S  [[buffer(3)]],   // (B,H,C,D,D)
                         constant unsigned &N  [[buffer(4)]],
                         constant unsigned &H  [[buffer(5)]],
                         uint3 blockIdx [[threadgroup_position_in_grid]],
                         uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D == 64, "ssd_chunk_kv currently supports D=64");
    constexpr int TPC = SSD_CHUNK_L / 8;
    using gl_t  = gl<bfloat, 1, -1, -1, D>;
    using gl_cl = gl<float, 1, -1, 1, -1>;
    gl_t  gB(Bm, nullptr, H, N, nullptr);
    gl_t  gX(X, nullptr, H, N, nullptr);
    gl_cl gcl(cl, nullptr, H, nullptr, N);
    const int c = blockIdx.x, head = blockIdx.y, batch = blockIdx.z;
    const int C = (int)N / SSD_CHUNK_L;
    const float cl_rc = cl[((long)batch * (int)H + head) * (long)N
                           + (long)(c + 1) * SSD_CHUNK_L - 1];   // reference r_c

    rt_fl<D, D> kv;
    zero(kv);
    for (int t = 0; t < TPC; ++t) {
        const int tile = c * TPC + t;
        rt_bf<8, D, ducks::rt_layout::col> b_reg;
        rt_bf<8, D> x_reg;
        load(b_reg, gB, {batch, head, tile, 0}, laneId);
        load(x_reg, gX, {batch, head, tile, 0}, laneId);
        // per-row weight w_j = exp(cl[r_c] - cl[j])  (<= 1)
        typename rt_fl<8, 8>::col_vec lj, w;
        load(lj, gcl, {batch, head, 0, tile}, laneId);
        sub(w, lj, cl_rc);
        mul(w, w, -1.0f);
        exp(w, w);
        rt_fl<8, D> x_fl;
        copy(x_fl, x_reg);
        mul_row(x_fl, x_fl, w);
        rt_bf<8, D> x_w;
        copy(x_w, x_fl);
        mma_AtB(kv, b_reg, x_w, kv);
    }
    gl<float, 1, -1, D, D> gs(S, nullptr, (int)H * C, nullptr, nullptr);
    store(gs, kv, {batch, head * C + c, 0, 0}, laneId);
}

// Exclusive decayed prefix over chunks: run = L_c * run + KV_c, S_ex[c] = run-before.
template <int D>
kernel void ssd_chunk_scan(device const float *Sin [[buffer(0)]],
                           device const float *cl  [[buffer(1)]],
                           device float       *Sex [[buffer(2)]],
                           constant unsigned  &C   [[buffer(3)]],
                           constant unsigned  &N   [[buffer(4)]],
                           uint3 blockIdx [[threadgroup_position_in_grid]],
                           uint  tid      [[thread_index_in_threadgroup]]) {
    const long bh = blockIdx.x;
    const long base = bh * (long)C * D * D;
    device const float* clbh = cl + bh * (long)N;
    for (int e = (int)tid; e < D * D; e += 256) {
        float run = 0.0f;
        long idx = base + e;
        for (int c = 0; c < (int)C; ++c, idx += D * D) {
            const float t = Sin[idx];
            Sex[idx] = run;
            // re-reference S from r_{c-1} to r_c before adding KV_c (run is 0 at c=0)
            const float lam = (c > 0)
                ? metal::exp(clbh[(long)(c + 1) * SSD_CHUNK_L - 1]
                             - clbh[(long)c * SSD_CHUNK_L - 1])
                : 0.0f;
            run = lam * run + t;
        }
    }
}

// Per-query-tile output: intra-chunk decay tiles (bounded by the chunk) plus the
// inter-chunk state term. Grid (N/8, H, B) like the quadratic kernel.
template <int D>
kernel void ssd_chunk_out(device   bf16       *Cq [[buffer(0)]],
                          device   bf16       *Bm [[buffer(1)]],
                          device   bf16       *X  [[buffer(2)]],
                          device   float      *cl [[buffer(3)]],
                          device   const float *Sex [[buffer(4)]],
                          device   bf16       *Y  [[buffer(5)]],
                          constant unsigned   &N  [[buffer(6)]],
                          constant unsigned   &H  [[buffer(7)]],
                          uint3 blockIdx [[threadgroup_position_in_grid]],
                          uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D == 64, "ssd_chunk_out currently supports D=64");
    constexpr int TPC = SSD_CHUNK_L / 8;
    using gl_t  = gl<bfloat, 1, -1, -1, D>;
    using gl_cl = gl<float, 1, -1, 1, -1>;
    gl_t  gC(Cq, nullptr, H, N, nullptr);
    gl_t  gB(Bm, nullptr, H, N, nullptr);
    gl_t  gX(X, nullptr, H, N, nullptr);
    gl_t  gY(Y, nullptr, H, N, nullptr);
    gl_cl gcl(cl, nullptr, H, nullptr, N);
    const int qi = blockIdx.x, head = blockIdx.y, batch = blockIdx.z;
    const int c = qi / TPC;                       // this tile's chunk
    const int C = (int)N / SSD_CHUNK_L;

    rt_bf<8, D> c_reg;
    load(c_reg, gC, {batch, head, qi, 0}, laneId);
    typename rt_fl<8, 8>::col_vec cumlog_i;
    load(cumlog_i, gcl, {batch, head, 0, qi}, laneId);

    rt_fl<8, D> y_reg;
    zero(y_reg);

    // inter-chunk: y = diag(exp(cl_i - cl[r_{c-1}])) * (C @ S_c)   (skip for chunk 0)
    if (c > 0) {
        rt_fl<D, D> s_fl;
        gl<float, 1, -1, D, D> gs(const_cast<device float*>(Sex), nullptr,
                                  (int)H * C, nullptr, nullptr);
        load(s_fl, gs, {batch, head * C + c, 0, 0}, laneId);
        rt_bf<D, D> s_bf;
        copy(s_bf, s_fl);
        mma_AB(y_reg, c_reg, s_bf, y_reg);
        const float cl_ref = cl[((long)batch * (int)H + head) * (long)N
                                + (long)c * SSD_CHUNK_L - 1];     // r_{c-1}
        typename rt_fl<8, 8>::col_vec w;
        sub(w, cumlog_i, cl_ref);
        exp(w, w);
        mul_row(y_reg, y_reg, w);
    }

    // intra-chunk: the quadratic loop, bounded to this chunk
    for (int kj = c * TPC; kj <= qi; kj++) {
        rt_bf<8, D, ducks::rt_layout::col> b_reg;
        rt_bf<8, D> x_reg;
        load(b_reg, gB, {batch, head, kj, 0}, laneId);
        load(x_reg, gX, {batch, head, kj, 0}, laneId);
        typename rt_fl<8, 8>::row_vec cumlog_j;
        load(cumlog_j, gcl, {batch, head, 0, kj}, laneId);

        rt_fl<8, 8> att;
        zero(att);
        mma_ABt(att, c_reg, b_reg, att);

        rt_fl<8, 8> decay;
        zero(decay);
        add_row(decay, decay, cumlog_i);
        sub_col(decay, decay, cumlog_j);
        exp(decay, decay);
        mul(att, att, decay);

        if (kj == qi) {
            float zero_fill = 0.0f;
            make_causal(att, att, laneId, zero_fill);
        }

        rt_bf<8, 8> att_bf;
        copy(att_bf, att);
        mma_AB(y_reg, att_bf, x_reg, y_reg);
    }
    store(gY, y_reg, {batch, head, qi, 0}, laneId);
}

#define instantiate_ssd_chunk(D)                                                     \
  template [[host_name("ssd_chunk_kv_" #D)]] [[kernel]] void                         \
  ssd_chunk_kv<D>(device bf16 *Bm [[buffer(0)]], device bf16 *X [[buffer(1)]],       \
    device float *cl [[buffer(2)]], device float *S [[buffer(3)]],                   \
    constant unsigned &N [[buffer(4)]], constant unsigned &H [[buffer(5)]],          \
    uint3 blockIdx [[threadgroup_position_in_grid]],                                 \
    uint laneId [[thread_index_in_simdgroup]]);                                      \
  template [[host_name("ssd_chunk_scan_" #D)]] [[kernel]] void                       \
  ssd_chunk_scan<D>(device const float *Sin [[buffer(0)]],                           \
    device const float *cl [[buffer(1)]], device float *Sex [[buffer(2)]],           \
    constant unsigned &C [[buffer(3)]], constant unsigned &N [[buffer(4)]],          \
    uint3 blockIdx [[threadgroup_position_in_grid]],                                 \
    uint tid [[thread_index_in_threadgroup]]);                                       \
  template [[host_name("ssd_chunk_out_" #D)]] [[kernel]] void                        \
  ssd_chunk_out<D>(device bf16 *Cq [[buffer(0)]], device bf16 *Bm [[buffer(1)]],     \
    device bf16 *X [[buffer(2)]], device float *cl [[buffer(3)]],                    \
    device const float *Sex [[buffer(4)]], device bf16 *Y [[buffer(5)]],             \
    constant unsigned &N [[buffer(6)]], constant unsigned &H [[buffer(7)]],          \
    uint3 blockIdx [[threadgroup_position_in_grid]],                                 \
    uint laneId [[thread_index_in_simdgroup]]);

instantiate_ssd_chunk(64);

}
