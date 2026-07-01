// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <stdexcept>
#include <string>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "rope_kv/rope_kv.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

std::vector<array> rope_kv_insert(
    const array& k,
    const array& v,
    const array& cos,
    const array& sin,
    const array& positions,
    const array& slot_mapping,
    const array& key_cache,
    const array& value_cache,
    StreamOrDevice s /* = {} */) {
  if (k.ndim() != 3 || v.ndim() != 3 || k.shape() != v.shape()) {
    throw std::invalid_argument("rope_kv_insert: k/v must have shape (num_tokens, num_kv_heads, D)");
  }
  if (key_cache.ndim() != 4 || value_cache.ndim() != 4 || key_cache.shape() != value_cache.shape()) {
    throw std::invalid_argument(
        "rope_kv_insert: caches must have shape (num_blocks, block_size, num_kv_heads, D)");
  }
  const int D = k.shape(2);
  if (!(D == 64 || D == 128)) {
    throw std::invalid_argument("rope_kv_insert: D must be 64 or 128");
  }
  if (key_cache.shape(2) != k.shape(1) || key_cache.shape(3) != D) {
    throw std::invalid_argument("rope_kv_insert: cache heads/head_size must match k");
  }
  if (cos.ndim() != 2 || cos.shape(1) != D / 2 || cos.shape() != sin.shape()) {
    throw std::invalid_argument("rope_kv_insert: cos/sin must have shape (P, D/2)");
  }
  if (positions.ndim() != 1 || positions.shape(0) != k.shape(0)) {
    throw std::invalid_argument("rope_kv_insert: positions must have shape (num_tokens,)");
  }
  if (slot_mapping.ndim() != 1 || slot_mapping.shape(0) != k.shape(0)) {
    throw std::invalid_argument("rope_kv_insert: slot_mapping must have shape (num_tokens,)");
  }
  if (k.dtype() != bfloat16 || key_cache.dtype() != bfloat16) {
    throw std::invalid_argument("rope_kv_insert: k/v/caches must be bfloat16");
  }

  auto k_c = contiguous(astype(k, bfloat16, s), false, s);
  auto v_c = contiguous(astype(v, bfloat16, s), false, s);
  auto cos_c = contiguous(astype(cos, bfloat16, s), false, s);
  auto sin_c = contiguous(astype(sin, bfloat16, s), false, s);
  auto pos_c = contiguous(astype(positions, int32, s), false, s);
  auto slot_c = contiguous(astype(slot_mapping, int64, s), false, s);
  auto kc_c = contiguous(key_cache, false, s);
  auto vc_c = contiguous(value_cache, false, s);

  return array::make_arrays(
      {key_cache.shape(), value_cache.shape()},
      {bfloat16, bfloat16},
      std::make_shared<RopeKvInsert>(to_stream(s)),
      {k_c, v_c, cos_c, sin_c, pos_c, slot_c, kc_c, vc_c});
}

void RopeKvInsert::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("RopeKvInsert has no CPU implementation.");
}

void RopeKvInsert::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& k = inputs[0];
  auto& v = inputs[1];
  auto& cos = inputs[2];
  auto& sin = inputs[3];
  auto& positions = inputs[4];
  auto& slot_mapping = inputs[5];
  auto& key_cache_in = inputs[6];
  auto& value_cache_in = inputs[7];
  auto& key_out = outputs[0];
  auto& value_out = outputs[1];

  auto& s = stream();
  auto& d = metal::device(s.device);
  key_out.set_data(allocator::malloc_or_wait(key_out.nbytes()));
  value_out.set_data(allocator::malloc_or_wait(value_out.nbytes()));

  const int num_tokens = k.shape(0);
  const int num_kv_heads = k.shape(1);
  const int D = k.shape(2);
  const int block_size = key_cache_in.shape(1);
  const std::string tn = type_to_name(key_out);

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  // Copy the existing caches through, then overwrite the inserted slot rows.
  const uint64_t total = static_cast<uint64_t>(key_out.size());
  tk::launch_kv_cache_clone(enc, key_cache_in, value_cache_in, key_out, value_out, total, tn);
  tk::launch_rope_kv_insert(
      enc, k, v, cos, sin, positions, slot_mapping, key_out, value_out,
      num_tokens * num_kv_heads, num_kv_heads, block_size, D);
}

std::vector<array> RopeKvInsert::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("RopeKvInsert has no jvp implementation.");
}
std::vector<array> RopeKvInsert::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("RopeKvInsert has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> RopeKvInsert::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("RopeKvInsert has no vmap implementation.");
}

// --------------------------- rope_kv_insert_norm ---------------------------

std::vector<array> rope_kv_insert_norm(
    const array& k, const array& v, const array& cos, const array& sin, const array& positions,
    const array& slot_mapping, const array& key_cache, const array& value_cache,
    const array& norm_weight, float eps, bool gemma, StreamOrDevice s /* = {} */) {
  if (k.ndim() != 3 || v.ndim() != 3 || k.shape() != v.shape()) {
    throw std::invalid_argument("rope_kv_insert_norm: k/v must be (num_tokens, num_kv_heads, D)");
  }
  const int D = k.shape(2);
  if (!(D == 64 || D == 128)) {
    throw std::invalid_argument("rope_kv_insert_norm: D must be 64 or 128");
  }
  if (key_cache.ndim() != 4 || key_cache.shape() != value_cache.shape() ||
      key_cache.shape(2) != k.shape(1) || key_cache.shape(3) != D) {
    throw std::invalid_argument("rope_kv_insert_norm: cache must be (num_blocks, block_size, num_kv_heads, D)");
  }
  if (cos.ndim() != 2 || cos.shape(1) != D / 2 || cos.shape() != sin.shape()) {
    throw std::invalid_argument("rope_kv_insert_norm: cos/sin must be (P, D/2)");
  }
  if (norm_weight.ndim() != 1 || norm_weight.shape(0) != D) {
    throw std::invalid_argument("rope_kv_insert_norm: norm_weight must be (D,)");
  }
  if (k.dtype() != bfloat16 || key_cache.dtype() != bfloat16) {
    throw std::invalid_argument("rope_kv_insert_norm: k/v/caches must be bfloat16");
  }

  auto k_c = contiguous(astype(k, bfloat16, s), false, s);
  auto v_c = contiguous(astype(v, bfloat16, s), false, s);
  auto cos_c = contiguous(astype(cos, bfloat16, s), false, s);
  auto sin_c = contiguous(astype(sin, bfloat16, s), false, s);
  auto pos_c = contiguous(astype(positions, int32, s), false, s);
  auto slot_c = contiguous(astype(slot_mapping, int64, s), false, s);
  auto kc_c = contiguous(key_cache, false, s);
  auto vc_c = contiguous(value_cache, false, s);
  auto w_c = contiguous(astype(norm_weight, bfloat16, s), false, s);

  return array::make_arrays(
      {key_cache.shape(), value_cache.shape()}, {bfloat16, bfloat16},
      std::make_shared<RopeKvInsertNorm>(to_stream(s), eps, gemma),
      {k_c, v_c, cos_c, sin_c, pos_c, slot_c, kc_c, vc_c, w_c});
}

void RopeKvInsertNorm::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("RopeKvInsertNorm has no CPU implementation.");
}

void RopeKvInsertNorm::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& k = inputs[0];
  auto& v = inputs[1];
  auto& cos = inputs[2];
  auto& sin = inputs[3];
  auto& positions = inputs[4];
  auto& slot_mapping = inputs[5];
  auto& key_cache_in = inputs[6];
  auto& value_cache_in = inputs[7];
  auto& norm_weight = inputs[8];
  auto& key_out = outputs[0];
  auto& value_out = outputs[1];

  auto& s = stream();
  auto& d = metal::device(s.device);
  key_out.set_data(allocator::malloc_or_wait(key_out.nbytes()));
  value_out.set_data(allocator::malloc_or_wait(value_out.nbytes()));

  const int num_tokens = k.shape(0);
  const int num_kv_heads = k.shape(1);
  const int D = k.shape(2);
  const int block_size = key_cache_in.shape(1);
  const std::string tn = type_to_name(key_out);

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  const uint64_t total = static_cast<uint64_t>(key_out.size());
  tk::launch_kv_cache_clone(enc, key_cache_in, value_cache_in, key_out, value_out, total, tn);
  tk::launch_rope_kv_insert_norm(
      enc, k, v, cos, sin, positions, slot_mapping, key_out, value_out, norm_weight,
      num_tokens * num_kv_heads, num_kv_heads, block_size, D, eps_, gemma_ ? 1 : 0);
}

std::vector<array> RopeKvInsertNorm::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("RopeKvInsertNorm has no jvp implementation.");
}
std::vector<array> RopeKvInsertNorm::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("RopeKvInsertNorm has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> RopeKvInsertNorm::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("RopeKvInsertNorm has no vmap implementation.");
}

} // namespace mlx::core
