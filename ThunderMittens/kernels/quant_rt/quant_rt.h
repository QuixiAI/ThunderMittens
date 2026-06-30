// Copyright © 2023 Apple Inc.

#pragma once

#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/**
 *  Runtime per-token (per-row) activation quantization on the GPU.
 *  Returns (codes, scale): scale[row] = absmax(row) / QMAX, codes = encode(x/scale).
 *  Reconstruct as scale[row] * decode(codes[row]).
 *
 *  x : (..., D), float32/float16/bfloat16. codes has x's shape; scale has x.shape[:-1].
 *   - fp8 : codes uint8 (e4m3, QMAX=448)
 *   - int8: codes int8  (symmetric, QMAX=127)
 **/
std::vector<array> quantize_per_token_fp8(const array& x, StreamOrDevice s = {});
std::vector<array> quantize_per_token_int8(const array& x, StreamOrDevice s = {});

class QuantizePerTokenFp8 : public Primitive {
 public:
  explicit QuantizePerTokenFp8(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "QuantizePerTokenFp8"; }
  void print(std::ostream& os) override { os << "QuantizePerTokenFp8"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

class QuantizePerTokenInt8 : public Primitive {
 public:
  explicit QuantizePerTokenInt8(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "QuantizePerTokenInt8"; }
  void print(std::ostream& os) override { os << "QuantizePerTokenInt8"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

} // namespace mlx::core
