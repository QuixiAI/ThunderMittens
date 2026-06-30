// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "quant_rt/quant_rt.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

static std::vector<int> qrt_scale_shape(const array& x) {
  std::vector<int> sh(x.shape().begin(), x.shape().end() - 1);
  if (sh.empty()) {
    sh.push_back(1);
  }
  return sh;
}

static void qrt_check(const array& x, const char* name) {
  if (x.ndim() < 1) {
    throw std::invalid_argument(std::string(name) + ": x must have at least 1 dimension");
  }
  if (!(x.dtype() == float32 || x.dtype() == float16 || x.dtype() == bfloat16)) {
    throw std::invalid_argument(std::string(name) + ": x must be float32, float16, or bfloat16");
  }
}

std::vector<array> quantize_per_token_fp8(const array& x, StreamOrDevice s /* = {} */) {
  qrt_check(x, "quantize_per_token_fp8");
  auto x_c = contiguous(x, false, s);
  return array::make_arrays(
      {x.shape(), qrt_scale_shape(x)},
      {uint8, float32},
      std::make_shared<QuantizePerTokenFp8>(to_stream(s)),
      {x_c});
}

std::vector<array> quantize_per_token_int8(const array& x, StreamOrDevice s /* = {} */) {
  qrt_check(x, "quantize_per_token_int8");
  auto x_c = contiguous(x, false, s);
  return array::make_arrays(
      {x.shape(), qrt_scale_shape(x)},
      {int8, float32},
      std::make_shared<QuantizePerTokenInt8>(to_stream(s)),
      {x_c});
}

void QuantizePerTokenFp8::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("QuantizePerTokenFp8 has no CPU implementation.");
}
void QuantizePerTokenInt8::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("QuantizePerTokenInt8 has no CPU implementation.");
}

void QuantizePerTokenFp8::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& codes = outputs[0];
  auto& scale = outputs[1];
  auto& s = stream();
  auto& d = metal::device(s.device);
  codes.set_data(allocator::malloc_or_wait(codes.nbytes()));
  scale.set_data(allocator::malloc_or_wait(scale.nbytes()));
  const int D = x.shape(-1);
  const int rows = static_cast<int>(x.size() / D);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_quantize_per_token_fp8(enc, x, codes, scale, rows, D, type_to_name(x));
}

void QuantizePerTokenInt8::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& codes = outputs[0];
  auto& scale = outputs[1];
  auto& s = stream();
  auto& d = metal::device(s.device);
  codes.set_data(allocator::malloc_or_wait(codes.nbytes()));
  scale.set_data(allocator::malloc_or_wait(scale.nbytes()));
  const int D = x.shape(-1);
  const int rows = static_cast<int>(x.size() / D);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_quantize_per_token_int8(enc, x, codes, scale, rows, D, type_to_name(x));
}

#define TK_QRT_NO_AUTODIFF(CLASS, LABEL)                                     \
  std::vector<array> CLASS::jvp(                                             \
      const std::vector<array>&, const std::vector<array>&,                  \
      const std::vector<int>&) {                                             \
    throw std::runtime_error(LABEL " has no jvp implementation.");           \
  }                                                                          \
  std::vector<array> CLASS::vjp(                                             \
      const std::vector<array>&, const std::vector<array>&,                  \
      const std::vector<int>&, const std::vector<array>&) {                  \
    throw std::runtime_error(LABEL " has no vjp implementation.");           \
  }                                                                          \
  std::pair<std::vector<array>, std::vector<int>> CLASS::vmap(               \
      const std::vector<array>&, const std::vector<int>&) {                  \
    throw std::runtime_error(LABEL " has no vmap implementation.");          \
  }

TK_QRT_NO_AUTODIFF(QuantizePerTokenFp8, "QuantizePerTokenFp8")
TK_QRT_NO_AUTODIFF(QuantizePerTokenInt8, "QuantizePerTokenInt8")

} // namespace mlx::core
