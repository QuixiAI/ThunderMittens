// Copyright © 2023 Apple Inc.

#pragma once

#include <string>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/** Quantized-KV flash attention: softmax(QK^T)·V with K and V supplied as quantized blocks
 *  (format `format`, e.g. "q8_0"/"q4_0"/"fp8_e4m3"). q is bf16 (B,H,N,D); kq/vq are uint8
 *  (B,H,N, D/block_k, block_bytes); out is bf16 (B,H,N,D). D in {64,128}, N%8==0. */
array attn_q(const array& q, const array& kq, const array& vq,
             const std::string& format = "q8_0", StreamOrDevice s = {});

class AttnQ : public Primitive {
 public:
  explicit AttnQ(Stream stream, std::string format)
      : Primitive(stream), fmt_(std::move(format)) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  void print(std::ostream& os) override { os << "AttnQ[" << fmt_ << "]"; }
  bool is_equivalent(const Primitive& other) const override;
  void eval(const std::vector<array>&, std::vector<array>&);

 private:
  std::string fmt_;
};

} // namespace mlx::core
