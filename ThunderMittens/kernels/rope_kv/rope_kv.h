// Copyright © 2023 Apple Inc.

#pragma once

#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/**
 *  Fused RoPE (split-half / GPT-NeoX) + paged-KV insert. Rotates K, then writes
 *  rotated K and (unrotated) V into the paged caches at slot_mapping[token].
 *  Returns the two updated caches (the un-inserted slots are copied through).
 *
 *  k, v      : (num_tokens, num_kv_heads, D)  bf16
 *  cos, sin  : (P, D/2)                        bf16 (precomputed)
 *  positions : (num_tokens,)                   int  (per-token RoPE position)
 *  slot_mapping : (num_tokens,)                int  (paged slot; < 0 skips)
 *  key_cache, value_cache : (num_blocks, block_size, num_kv_heads, D) bf16
 *  D must be 64 or 128.
 **/
std::vector<array> rope_kv_insert(
    const array& k,
    const array& v,
    const array& cos,
    const array& sin,
    const array& positions,
    const array& slot_mapping,
    const array& key_cache,
    const array& value_cache,
    StreamOrDevice s = {});

class RopeKvInsert : public Primitive {
 public:
  explicit RopeKvInsert(Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&,
      const std::vector<array>&,
      const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&,
      const std::vector<array>&,
      const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&,
      const std::vector<int>&) override;
  const char* name() const { return "RopeKvInsert"; }
  void print(std::ostream& os) override { os << "RopeKvInsert"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

} // namespace mlx::core
