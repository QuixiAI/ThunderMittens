// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "rms_norm/rms_norm.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <vecLib/cblas_new.h>
#endif

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

///////////////////////////////////////////////////////////////////////////////
// Operation Implementation
///////////////////////////////////////////////////////////////////////////////

/**
 *  RMSNorm over the last axis:
 *      y = x * rsqrt(mean(x^2) + eps) * weight
 *
 *  Inputs are assumed bf16 and row-contiguous. The last dim D must be one of
 *  the instantiated widths {256, 512, 768, 1024}.
 **/
array rms_norm(
    const array& x,
    const array& weight,
    float eps /* = 1e-5f */,
    StreamOrDevice s /* = {} */
) {
  assert(x.dtype() == bfloat16 && weight.dtype() == bfloat16);
  const int D = x.shape(-1);
  assert(weight.ndim() == 1 && weight.shape(0) == D);
  assert((D == 256 || D == 512 || D == 768 || D == 1024) &&
         "rms_norm: last dim must be 256, 512, 768, or 1024");

  return array(
      /* const std::vector<int>& shape = */ x.shape(),
      /* Dtype dtype = */ bfloat16,
      /* std::shared_ptr<Primitive> primitive = */
      std::make_shared<RMSNorm>(to_stream(s), eps),
      /* const std::vector<array>& inputs = */ {x, weight});
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Common Backend Implementation
///////////////////////////////////////////////////////////////////////////////

void RMSNorm::eval(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // No CPU fallback yet; use mx.fast.rms_norm for a reference.
  assert(false);
}

void RMSNorm::eval_cpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  eval(inputs, outputs);
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Metal Backend Implementation
///////////////////////////////////////////////////////////////////////////////

void RMSNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 2);
  auto& x = inputs[0];
  auto& weight = inputs[1];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);

  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  const int D = x.shape(-1);
  const uint32_t M = static_cast<uint32_t>(x.size() / D); // rows = prod(shape[:-1])

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_rms_norm(enc, x, weight, out, M, D, eps_);
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Transforms
///////////////////////////////////////////////////////////////////////////////

std::vector<array> RMSNorm::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  throw std::runtime_error("RMSNorm has no jvp implementation.");
}

std::vector<array> RMSNorm::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  throw std::runtime_error("RMSNorm has no vjp implementation.");
}

std::pair<std::vector<array>, std::vector<int>> RMSNorm::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  throw std::runtime_error("RMSNorm has no vmap implementation.");
}

bool RMSNorm::is_equivalent(const Primitive& other) const {
  const RMSNorm& r_other = static_cast<const RMSNorm&>(other);
  return eps_ == r_other.eps_;
}

} // namespace mlx::core
