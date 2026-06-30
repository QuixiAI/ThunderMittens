// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <sstream>
#include <stdexcept>
#include <string>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "glu/glu.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

static bool valid_glu_mode(const std::string& mode) {
  return mode == "reglu" || mode == "geglu" || mode == "swiglu" ||
         mode == "swiglu_oai" || mode == "geglu_erf" ||
         mode == "geglu_quick";
}

array glu(
    const array& x,
    const array& gate,
    const std::string& mode,
    float alpha,
    float limit,
    StreamOrDevice s) {
  if (!valid_glu_mode(mode)) {
    std::ostringstream msg;
    msg << "glu: unsupported mode '" << mode
        << "'; expected reglu, geglu, swiglu, swiglu_oai, geglu_erf, or geglu_quick";
    throw std::invalid_argument(msg.str());
  }
  if (x.shape() != gate.shape()) {
    throw std::invalid_argument("glu: x and gate must have the same shape");
  }
  auto promoted_dtype = promote_types(x.dtype(), gate.dtype());
  if (!(promoted_dtype == float32 || promoted_dtype == float16 ||
        promoted_dtype == bfloat16)) {
    throw std::invalid_argument("glu: dtype must be float32, float16, or bfloat16");
  }

  auto x_casted = contiguous(astype(x, promoted_dtype, s), false, s);
  auto gate_casted = contiguous(astype(gate, promoted_dtype, s), false, s);

  return array(
      x.shape(),
      promoted_dtype,
      std::make_shared<Glu>(to_stream(s), mode, alpha, limit),
      {x_casted, gate_casted});
}

void Glu::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("Glu has no CPU implementation.");
}

void Glu::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 2);
  auto& x = inputs[0];
  auto& gate = inputs[1];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  const uint32_t n = static_cast<uint32_t>(out.size());
  tk::launch_glu(enc, x, gate, out, n, mode_, type_to_name(out), alpha_, limit_);
}

std::vector<array> Glu::jvp(
    const std::vector<array>&,
    const std::vector<array>&,
    const std::vector<int>&) {
  throw std::runtime_error("Glu has no jvp implementation.");
}

std::vector<array> Glu::vjp(
    const std::vector<array>&,
    const std::vector<array>&,
    const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("Glu has no vjp implementation.");
}

std::pair<std::vector<array>, std::vector<int>> Glu::vmap(
    const std::vector<array>&,
    const std::vector<int>&) {
  throw std::runtime_error("Glu has no vmap implementation.");
}

bool Glu::is_equivalent(const Primitive& other) const {
  const Glu& r_other = static_cast<const Glu&>(other);
  return mode_ == r_other.mode_ && alpha_ == r_other.alpha_ &&
         limit_ == r_other.limit_;
}

} // namespace mlx::core
