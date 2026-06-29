/**
 * @file
 * @brief In-register dequantization of quantized weight blocks into half tiles.
 *
 * "Marlin's method" on Apple: quantized weights are stored block-wise (a small fp16 scale +
 * packed low-precision codes), dequantized to `half` here, then fed to a standard
 * `simdgroup_matrix` MMA. The dequant math is pure IEEE-fp16 bit arithmetic, valid on Metal
 * `half` verbatim. Block layouts mirror llama.cpp's GGUF formats (ggml-common.h); the dequant
 * constants follow llama.cpp (ggml-metal.metal) and Marlin (dequant.h).
 *
 * A "format" is a small struct exposing `block_k` (weights per block), `block_bytes`, and a
 * `dequant(device const uchar* base, int col) -> half` for the weight at column `col` of the
 * block starting at byte `base`. `dequant_into_shared<FMT>` cooperatively dequantizes a tile.
 */
#pragma once
#include "../../../../common/common.metal"
#include "../../../../types/types.metal"

namespace mittens {

// ---- q8_0 : { half d; int8 qs[32]; }  — 34 bytes, 32 weights/block, value = d * q ----
struct q8_0 {
    constant static constexpr const int block_k     = 32;
    constant static constexpr const int block_bytes = 34;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half d = ((device const half*)base)[0];      // fp16 scale at offset 0
        const char q = ((device const char*)(base + 2))[col];  // signed int8 codes at offset 2
        return d * half(q);
    }
};

// ---- q4_0 : { half d; uint8 qs[16]; } — 18 bytes, 32 weights/block. Nibble packing (ggml):
//   weight col<16 -> qs[col]&0xF ; col>=16 -> qs[col-16]>>4 ; value = d * (nibble - 8). ----
struct q4_0 {
    constant static constexpr const int block_k     = 32;
    constant static constexpr const int block_bytes = 18;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half d = ((device const half*)base)[0];      // fp16 scale at offset 0
        device const uchar* qs = base + 2;                 // 16 packed-nibble bytes
        const int nib = (col < 16) ? (qs[col] & 0x0F) : (qs[col - 16] >> 4);
        return d * half(nib - 8);
    }
};

// ---- q4_K : { half d; half dmin; uint8 scales[12]; uint8 qs[128]; } — 144 bytes, 256/block.
//   8 sub-blocks of 32; each has a 6-bit scale `sc` and 6-bit min `m` (packed in `scales`,
//   extracted GGUF-style). value = (d*sc)*nibble - (dmin*m). ----
struct q4_K {
    constant static constexpr const int block_k     = 256;
    constant static constexpr const int block_bytes = 144;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half d    = ((device const half*)base)[0];
        const half dmin = ((device const half*)base)[1];
        device const uchar* scales = base + 4;
        device const uchar* qs     = base + 16;
        const int chunk = col / 64;        // 0..3
        const int pos   = col % 64;        // 0..63
        int sub, nib;
        if (pos < 32) { sub = chunk * 2;     nib = qs[chunk * 32 + pos]        & 0x0F; }
        else          { sub = chunk * 2 + 1; nib = qs[chunk * 32 + (pos - 32)] >> 4;   }
        // get_scale_min_k4(sub): unpack the 6-bit scale `sc` and min `m`
        uchar sc, m;
        if (sub < 4) { sc = scales[sub] & 63; m = scales[sub + 4] & 63; }
        else {
            sc = (scales[sub + 4] & 0x0F) | ((scales[sub - 4] >> 6) << 4);
            m  = (scales[sub + 4] >> 4)   | ((scales[sub]     >> 6) << 4);
        }
        return d * half(sc) * half(nib) - dmin * half(m);
    }
};

// ---- kU4B8 : GPTQ/Marlin grouped int4, group=128. { half scale; uint8 qs[64]; } — 66 bytes.
//   unsigned 4-bit with bias 8; value = scale * (nibble - 8). Nibble packing like q4_0 (col<64 ->
//   qs[col]&0xF ; col>=64 -> qs[col-64]>>4). Larger group than q4_0 (more compression). ----
struct kU4B8 {
    constant static constexpr const int block_k     = 128;
    constant static constexpr const int block_bytes = 66;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half scale = ((device const half*)base)[0];
        device const uchar* qs = base + 2;
        const int nib = (col < 64) ? (qs[col] & 0x0F) : (qs[col - 64] >> 4);
        return scale * half(nib - 8);
    }
};

// Cooperatively dequantize an (BN x BK) weight tile into a shared half tile. `kb` is the K-tile
// index in units of BK (the MMA K-step). The quant grouping (FMT::block_k) is DECOUPLED from BK:
// each tile column maps to its quant block via the global K index, so large blocks (e.g. q4_K's
// 256) work with a small BK. Requires FMT::block_k % BK == 0 and K % FMT::block_k == 0.
//   Packed layout: block(n, b) starts at byte (n*(K/FMT::block_k) + b) * FMT::block_bytes.
//   `group_threads` = total threads in the threadgroup; `threadIdx` = flat thread index.
template<typename FMT, int BN, int BK>
METAL_FUNC void dequant_into_shared(threadgroup st<half, BN, BK>& dst,
                                    device const uchar* Wq, int N, int K,
                                    int by, int kb, int group_threads, uint threadIdx) {
    const int blocks_per_row = K / FMT::block_k;
    for (int e = (int)threadIdx; e < BN * BK; e += group_threads) {
        const int row  = e / BK;
        const int tcol = e % BK;
        const int grow = by * BN + row;
        const int gk   = kb * BK + tcol;                 // global K column
        const int blk  = gk / FMT::block_k;              // which quant block
        const int cib  = gk % FMT::block_k;              // column within that block
        device const uchar* base = Wq + (uint)(grow * blocks_per_row + blk) * FMT::block_bytes;
        dst[int2(row, tcol)] = FMT::dequant(base, cib);
    }
}

} // namespace mittens
