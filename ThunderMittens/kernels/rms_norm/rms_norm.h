// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

///////////////////////////////////////////////////////////////////////////////
// Operation
///////////////////////////////////////////////////////////////////////////////

/**
 *  RMSNorm (forward), normalized over the last axis:
 *      y = x * rsqrt(mean(x^2) + eps) * weight
 *
 *  x is (..., D); weight is (D,). bf16 in/out, fp32 compute. No mean-subtraction,
 *  no bias (cf. LayerNorm).
 **/
array rms_norm(
    const array& x,      // Input array, normalized over the last axis
    const array& weight, // Per-channel scale, shape (D,)
    float eps = 1e-5f,   // Numerical stability epsilon
    StreamOrDevice s = {} // Stream on which to schedule the operation
);

///////////////////////////////////////////////////////////////////////////////
// Primitive
///////////////////////////////////////////////////////////////////////////////

class RMSNorm : public Primitive {
 public:
  explicit RMSNorm(Stream stream, float eps)
    : Primitive(stream), eps_(eps) {};

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  /** The Jacobian-vector product. */
  std::vector<array> jvp(
      const std::vector<array>& primals,
      const std::vector<array>& tangents,
      const std::vector<int>& argnums) override;

  /** The vector-Jacobian product. */
  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

  /** Vectorize the primitive across the given axes. */
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  /** Print the primitive. */
  void print(std::ostream& os) override {
    os << "RMSNorm";
  }

  /** Equivalence check **/
  bool is_equivalent(const Primitive& other) const override;

  /** Fall back implementation for evaluation on CPU */
  void eval(const std::vector<array>& inputs, std::vector<array>& outputs);

 private:
  float eps_;
};

} // namespace mlx::core
