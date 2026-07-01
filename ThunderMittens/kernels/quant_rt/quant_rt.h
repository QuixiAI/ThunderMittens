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

/**
 *  Per-tensor (global) dynamic quantization: one scale = global_absmax / QMAX (via a P3
 *  atomic-max reduction). Returns [codes, scale (scalar), scale_u (uint32 scratch)]; callers
 *  use the first two. fp8 e4m3 (QMAX=448) or symmetric int8 (QMAX=127).
 **/
std::vector<array> quantize_per_tensor_fp8(const array& x, StreamOrDevice s = {});
std::vector<array> quantize_per_tensor_int8(const array& x, StreamOrDevice s = {});

class QuantizePerTensor : public Primitive {
 public:
  QuantizePerTensor(Stream stream, bool is_int8) : Primitive(stream), is_int8_(is_int8) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "QuantizePerTensor"; }
  void print(std::ostream& os) override { os << "QuantizePerTensor"; }
  bool is_equivalent(const Primitive& other) const override {
    return is_int8_ == static_cast<const QuantizePerTensor&>(other).is_int8_;
  }

 private:
  bool is_int8_;
};

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
