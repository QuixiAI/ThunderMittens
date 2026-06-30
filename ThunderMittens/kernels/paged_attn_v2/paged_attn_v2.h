// Copyright © 2023 Apple Inc.

#pragma once

#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/**
 *  Long-context paged decode attention (vLLM-v2 partition/reduce). Splits each
 *  (head, batch) query across KV-sequence partitions, computes a local softmax
 *  per partition, then merges via the log-sum-exp rescaling trick. GQA/MQA aware
 *  (num_q_heads may be a multiple of the cache's num_kv_heads).
 *
 *  q : (batch, num_heads, D); caches : (num_blocks, block_size, num_kv_heads, D).
 *  D ∈ {64,128}. Returns out : (batch, num_heads, D). partition_size must be a
 *  positive multiple of block_size.
 **/
array paged_attention_v2(
    const array& q,
    const array& key_cache,
    const array& value_cache,
    const array& block_table,
    const array& context_lens,
    float scale = 0.0f,
    int partition_size = 512,
    StreamOrDevice s = {});

// --- internal primitives (not bound directly) ---

class PagedAttentionV2Partition : public Primitive {
 public:
  PagedAttentionV2Partition(Stream stream, float scale, int num_partitions, int partition_size)
      : Primitive(stream),
        scale_(scale),
        num_partitions_(num_partitions),
        partition_size_(partition_size) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "PagedAttentionV2Partition"; }
  void print(std::ostream& os) override { os << "PagedAttentionV2Partition"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const PagedAttentionV2Partition&>(other);
    return scale_ == o.scale_ && num_partitions_ == o.num_partitions_ &&
        partition_size_ == o.partition_size_;
  }

 private:
  float scale_;
  int num_partitions_;
  int partition_size_;
};

class PagedAttentionV2Reduce : public Primitive {
 public:
  explicit PagedAttentionV2Reduce(Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "PagedAttentionV2Reduce"; }
  void print(std::ostream& os) override { os << "PagedAttentionV2Reduce"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

} // namespace mlx::core
