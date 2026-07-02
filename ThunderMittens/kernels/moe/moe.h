// Copyright © 2023 Apple Inc.

#pragma once

#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/**
 *  MoE routing: top-k expert selection with renormalized softmax weights.
 *  logits is (num_tokens, num_experts), float32/float16/bfloat16. Returns
 *  (topk_ids int32, topk_weights float32), both (num_tokens, k). The weights are
 *  softmax over the k selected logits (Mixtral renormalized top-k rule). k <= 16.
 **/
std::vector<array> moe_route_topk(const array& logits, int k, StreamOrDevice s = {});

/**
 *  MoE permute: group the T*K routing rows by expert id. topk_ids is (num_tokens, k)
 *  int32. Returns 5 int32 arrays [sorted_row_idx (T*K), offsets (E+1), inv_idx (T*K),
 *  counts (E, scratch), cursor (E, scratch)] — callers use the first three. A flat
 *  routing row r maps to token r/k, slot r%k; offsets[e] is expert e's start.
 **/
std::vector<array> moe_permute(const array& topk_ids, int num_experts, StreamOrDevice s = {});

/**
 *  MoE padded schedule (GPU replacement for the host glue): turns moe_permute's compact
 *  layout into 32-row-padded per-expert segments for the grouped GEMMs. Static worst-case
 *  sizing: total_pad_max = ceil32(T*K + 31*E), max_tiles = total_pad_max/32; -1 sentinels
 *  mark tiles/rows beyond the real (data-dependent) total. Returns int32 arrays
 *  [expert_of_tile (max_tiles), gather_idx (total_pad_max), inv_pad (T*K), off_pad (E+1)].
 *  inv_pad[r] is the padded row finalize must read for routing row r.
 **/
std::vector<array> moe_pad_schedule(
    const array& sorted_row_idx, const array& offsets, int k, StreamOrDevice s = {});

/**
 *  MoE gather: permuted_input[p, :] = x[gather_idx[p], :] (zeros where gather_idx[p] < 0).
 *  x (T, H) float32/bfloat16; returns (len(gather_idx), H).
 **/
array moe_gather(const array& x, const array& gather_idx, StreamOrDevice s = {});

/**
 *  MoE finalize: out[t] = sum_k topk_weights[t,k] * expert_out[inv_idx[t*k+k]].
 *  expert_out is (T*K, Hdim) in permuted order; topk_weights (T, k) f32. Returns (T, Hdim).
 **/
array moe_finalize(
    const array& expert_out, const array& inv_idx, const array& topk_weights, int k,
    StreamOrDevice s = {});

/**
 *  Fused grouped expert GEMM: out = permuted_input @ W[expert]. permuted_input (total_rows, H)
 *  with rows grouped by expert, each segment padded to a 32-multiple; W (E, H, H);
 *  expert_of_tile (total_rows/32,) int32 gives the expert of each 32-row tile. Returns
 *  (total_rows, H). float32/bfloat16; requires total_rows % 32 == 0 and H % 32 == 0.
 **/
array moe_grouped_gemm(
    const array& permuted_input, const array& W, const array& expert_of_tile, StreamOrDevice s = {});

/** Rectangular grouped GEMM: out (total_rows, N_out) = A (total_rows, K_dim) @ W[e] (K_dim, N_out).
 *  W is (E, K_dim, N_out); K_dim % 16 == 0, N_out % 32 == 0, total_rows % 32 == 0. **/
array moe_grouped_gemm_rect(
    const array& A, const array& W, const array& expert_of_tile, StreamOrDevice s = {});

/** Fused SiLU-GLU GEMM1: out (total_rows, inter) = silu(A @ W1_gate) * (A @ W1_up).
 *  A (total_rows, H); W1 (E, H, 2*inter) laid out [gate | up]. H % 16 == 0, inter % 32 == 0. **/
array moe_grouped_gemm_swiglu(
    const array& A, const array& W1, const array& expert_of_tile, StreamOrDevice s = {});

class MoeRouteTopk : public Primitive {
 public:
  MoeRouteTopk(Stream stream, int k) : Primitive(stream), k_(k) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MoeRouteTopk"; }
  void print(std::ostream& os) override { os << "MoeRouteTopk"; }
  bool is_equivalent(const Primitive& other) const override {
    return k_ == static_cast<const MoeRouteTopk&>(other).k_;
  }

 private:
  int k_;
};

class MoePadSchedule : public Primitive {
 public:
  MoePadSchedule(Stream stream, int num_experts, int k)
      : Primitive(stream), num_experts_(num_experts), k_(k) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MoePadSchedule"; }
  void print(std::ostream& os) override { os << "MoePadSchedule"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const MoePadSchedule&>(other);
    return num_experts_ == o.num_experts_ && k_ == o.k_;
  }

 private:
  int num_experts_;
  int k_;
};

class MoeGather : public Primitive {
 public:
  explicit MoeGather(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MoeGather"; }
  void print(std::ostream& os) override { os << "MoeGather"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

class MoePermute : public Primitive {
 public:
  MoePermute(Stream stream, int num_experts) : Primitive(stream), num_experts_(num_experts) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MoePermute"; }
  void print(std::ostream& os) override { os << "MoePermute"; }
  bool is_equivalent(const Primitive& other) const override {
    return num_experts_ == static_cast<const MoePermute&>(other).num_experts_;
  }

 private:
  int num_experts_;
};

class MoeGroupedGemm : public Primitive {
 public:
  explicit MoeGroupedGemm(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MoeGroupedGemm"; }
  void print(std::ostream& os) override { os << "MoeGroupedGemm"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

class MoeGroupedGemmRect : public Primitive {
 public:
  explicit MoeGroupedGemmRect(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MoeGroupedGemmRect"; }
  void print(std::ostream& os) override { os << "MoeGroupedGemmRect"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

class MoeGroupedGemmSwiglu : public Primitive {
 public:
  explicit MoeGroupedGemmSwiglu(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MoeGroupedGemmSwiglu"; }
  void print(std::ostream& os) override { os << "MoeGroupedGemmSwiglu"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

class MoeFinalize : public Primitive {
 public:
  MoeFinalize(Stream stream, int k) : Primitive(stream), k_(k) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MoeFinalize"; }
  void print(std::ostream& os) override { os << "MoeFinalize"; }
  bool is_equivalent(const Primitive& other) const override {
    return k_ == static_cast<const MoeFinalize&>(other).k_;
  }

 private:
  int k_;
};

} // namespace mlx::core
