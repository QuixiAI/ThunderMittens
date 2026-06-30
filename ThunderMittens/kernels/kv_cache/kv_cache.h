// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

std::vector<array> kv_cache_scatter(
    const array& key,
    const array& value,
    const array& slot_mapping,
    int num_blocks,
    int block_size,
    StreamOrDevice s = {});

std::vector<array> kv_cache_gather(
    const array& key_cache,
    const array& value_cache,
    const array& block_table,
    const array& cu_seq_lens,
    int num_tokens,
    StreamOrDevice s = {});

std::vector<array> kv_cache_copy_blocks(
    const array& key_cache,
    const array& value_cache,
    const array& block_mapping,
    StreamOrDevice s = {});

std::vector<array> kv_cache_scales(
    const array& key,
    const array& value,
    StreamOrDevice s = {});

array paged_attention(
    const array& q,
    const array& key_cache,
    const array& value_cache,
    const array& block_table,
    const array& context_lens,
    float scale = 0.0f,
    StreamOrDevice s = {});

class KvCacheScatter : public Primitive {
 public:
  KvCacheScatter(Stream stream, int block_size)
      : Primitive(stream), block_size_(block_size) {}

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
  const char* name() const { return "KvCacheScatter"; }

  void print(std::ostream& os) override { os << "KvCacheScatter"; }
  bool is_equivalent(const Primitive& other) const override {
    return block_size_ == static_cast<const KvCacheScatter&>(other).block_size_;
  }

 private:
  int block_size_;
};

class KvCacheGather : public Primitive {
 public:
  KvCacheGather(Stream stream, int num_tokens)
      : Primitive(stream), num_tokens_(num_tokens) {}

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
  const char* name() const { return "KvCacheGather"; }

  void print(std::ostream& os) override { os << "KvCacheGather"; }
  bool is_equivalent(const Primitive& other) const override {
    return num_tokens_ == static_cast<const KvCacheGather&>(other).num_tokens_;
  }

 private:
  int num_tokens_;
};

class KvCacheCopyBlocks : public Primitive {
 public:
  explicit KvCacheCopyBlocks(Stream stream) : Primitive(stream) {}

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
  const char* name() const { return "KvCacheCopyBlocks"; }

  void print(std::ostream& os) override { os << "KvCacheCopyBlocks"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

class KvCacheScales : public Primitive {
 public:
  explicit KvCacheScales(Stream stream) : Primitive(stream) {}

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
  const char* name() const { return "KvCacheScales"; }

  void print(std::ostream& os) override { os << "KvCacheScales"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

class PagedAttention : public Primitive {
 public:
  PagedAttention(Stream stream, float scale) : Primitive(stream), scale_(scale) {}

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
  const char* name() const { return "PagedAttention"; }

  void print(std::ostream& os) override { os << "PagedAttention"; }
  bool is_equivalent(const Primitive& other) const override {
    return scale_ == static_cast<const PagedAttention&>(other).scale_;
  }

 private:
  float scale_;
};

} // namespace mlx::core
