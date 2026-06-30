// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

///////////////////////////////////////////////////////////////////////////////
// Operations
///////////////////////////////////////////////////////////////////////////////

/**
 *  Fused residual-add + RMSNorm over the last axis. Returns two arrays:
 *      out     = (x + residual) * rsqrt(mean((x+residual)^2) + eps) * weight
 *      res_out = x + residual   (the summed residual the next block consumes)
 *
 *  x and residual are (..., D); weight is (D,). bf16 in/out, fp32 compute.
 *  D must be one of {256, 512, 768, 1024}.
 **/
std::vector<array> rms_norm_add(
    const array& x,
    const array& residual,
    const array& weight,
    float eps = 1e-5f,
    StreamOrDevice s = {});

/**
 *  Fused residual-add + LayerNorm over the last axis. Returns two arrays:
 *      out     = ((x+residual) - mean) * rsqrt(var + eps) * weight + bias
 *      res_out = x + residual
 *
 *  x and residual are (..., D); weight and bias are (D,). bf16 in/out.
 **/
std::vector<array> layernorm_add(
    const array& x,
    const array& residual,
    const array& weight,
    const array& bias,
    float eps = 1e-5f,
    StreamOrDevice s = {});

///////////////////////////////////////////////////////////////////////////////
// Primitives
///////////////////////////////////////////////////////////////////////////////

class RMSNormAdd : public Primitive {
 public:
  explicit RMSNormAdd(Stream stream, float eps)
    : Primitive(stream), eps_(eps) {};

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  std::vector<array> jvp(
      const std::vector<array>& primals,
      const std::vector<array>& tangents,
      const std::vector<int>& argnums) override;
  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  const char* name() const { return "RMSNormAdd"; }
  void print(std::ostream& os) override { os << "RMSNormAdd"; }
  bool is_equivalent(const Primitive& other) const override;

 private:
  float eps_;
};

class LayerNormAdd : public Primitive {
 public:
  explicit LayerNormAdd(Stream stream, float eps)
    : Primitive(stream), eps_(eps) {};

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  std::vector<array> jvp(
      const std::vector<array>& primals,
      const std::vector<array>& tangents,
      const std::vector<int>& argnums) override;
  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  const char* name() const { return "LayerNormAdd"; }
  void print(std::ostream& os) override { os << "LayerNormAdd"; }
  bool is_equivalent(const Primitive& other) const override;

 private:
  float eps_;
};

} // namespace mlx::core
