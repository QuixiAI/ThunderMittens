// Copyright © 2024 Apple Inc.

#include <cmath>
#include <stdexcept>
#include <string>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "mla/mla.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

static array mla_contig_bf16(const array& x, StreamOrDevice s) {
  return contiguous(astype(x, bfloat16, s), false, s);
}

array mla_q_norm_rope(
    const array& q,
    const array& cos,
    const array& sin,
    const array& positions,
    const array& norm_weight,
    int num_heads,
    int nope_dim,
    int rope_dim,
    int norm_mode,
    float eps,
    StreamOrDevice s) {
  if (q.ndim() < 2) {
    throw std::invalid_argument("mla_q_norm_rope: q must be at least 2-D (…, head_dim)");
  }
  const int head_dim = q.shape(-1);
  if (head_dim % 64 != 0 || head_dim != nope_dim + rope_dim) {
    throw std::invalid_argument("mla_q_norm_rope: head_dim must be nope_dim+rope_dim and %64==0");
  }
  if (nope_dim % 2 != 0 || rope_dim % 2 != 0) {
    throw std::invalid_argument("mla_q_norm_rope: nope_dim and rope_dim must be even");
  }
  if (cos.shape(-1) != rope_dim / 2 || sin.shape(-1) != rope_dim / 2) {
    throw std::invalid_argument("mla_q_norm_rope: cos/sin must be (max_pos, rope_dim/2)");
  }
  if (norm_mode < 0 || norm_mode > 2) {
    throw std::invalid_argument("mla_q_norm_rope: norm_mode must be 0, 1, or 2");
  }

  auto q_c = mla_contig_bf16(q, s);
  auto cos_c = mla_contig_bf16(cos, s);
  auto sin_c = mla_contig_bf16(sin, s);
  auto pos_c = contiguous(astype(positions, int32, s), false, s);
  auto w_c = mla_contig_bf16(norm_weight, s);

  return array(
      q.shape(),
      bfloat16,
      std::make_shared<MlaQNormRope>(to_stream(s), num_heads, nope_dim, rope_dim, norm_mode, eps),
      {q_c, cos_c, sin_c, pos_c, w_c});
}

void MlaQNormRope::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("MlaQNormRope has no CPU implementation.");
}

void MlaQNormRope::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& q = inputs[0];
  auto& cos = inputs[1];
  auto& sin = inputs[2];
  auto& positions = inputs[3];
  auto& norm_weight = inputs[4];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  const int head_dim = q.shape(-1);
  const int M = static_cast<int>(q.size() / head_dim);   // num_tokens * num_heads
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_mla_q_norm_rope(
      enc, q, cos, sin, positions, norm_weight, out, M, num_heads_, nope_dim_, rope_dim_,
      norm_mode_, eps_, head_dim);
}

#define TK_MLA_NO_AUTODIFF(CLASS, LABEL)                                     \
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

TK_MLA_NO_AUTODIFF(MlaQNormRope, "MlaQNormRope")

} // namespace mlx::core
