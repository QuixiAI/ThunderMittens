// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/**
 *  Varlen / paged-prefill causal flash attention. Queries are packed HEAD-MAJOR and padded per
 *  sequence to a multiple of 8 rows: q_hm is (H, total_padded, D) bf16, D in {64,128}. K/V are
 *  read from the paged cache key_cache/value_cache (num_blocks, block_size, H_KV, D) bf16 via
 *  block_table (B, max_blocks) int32 and context_lens (B,) int32 (context_len >= q_len supports a
 *  cached prefix). The host-built worklist selects the batch and local query offset of each 8-row
 *  tile: tile_seq (n_tiles,), tile_local0 (n_tiles,), seq_qlen (B,), all int32.
 *  Returns o_hm (H, total_padded, D) bf16. block_size must be a multiple of 8.
 **/
array attn_varlen_prefill(
    const array& q_hm,
    const array& key_cache,
    const array& value_cache,
    const array& block_table,
    const array& context_lens,
    const array& tile_seq,
    const array& tile_local0,
    const array& seq_qlen,
    float scale,
    StreamOrDevice s = {});

class AttnVarlenPrefill : public Primitive {
 public:
  explicit AttnVarlenPrefill(Stream stream, float scale) : Primitive(stream), scale_(scale) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "AttnVarlenPrefill"; }
  void print(std::ostream& os) override { os << "AttnVarlenPrefill"; }
  bool is_equivalent(const Primitive& other) const override {
    return scale_ == static_cast<const AttnVarlenPrefill&>(other).scale_;
  }

 private:
  float scale_;
};

} // namespace mlx::core
