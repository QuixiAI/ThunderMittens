// Copyright © 2023-2024 Apple Inc.

#include <cassert>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "attn_varlen/attn_varlen.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

array attn_varlen_prefill(
    const array& q_hm,
    const array& key_cache,
    const array& value_cache,
    const array& block_table,
    const array& context_lens,
    const array& tile_seq,
    const array& tile_local0,
    const array& seq_qlen,
    float scale,
    StreamOrDevice s /* = {} */) {
  assert(q_hm.dtype() == bfloat16 && key_cache.dtype() == bfloat16 &&
         value_cache.dtype() == bfloat16);
  assert(q_hm.ndim() == 3);
  const int D = q_hm.shape(2);
  assert(D == 64 || D == 128);
  // The kernel assumes contiguous head-major q/o and contiguous index arrays; the Python wrapper
  // builds q_hm by transpose (a strided view), so force row-contiguity here.
  return array(
      q_hm.shape(), bfloat16,
      std::make_shared<AttnVarlenPrefill>(to_stream(s), scale),
      {contiguous(q_hm, false, s), contiguous(key_cache, false, s),
       contiguous(value_cache, false, s), contiguous(astype(block_table, int32, s), false, s),
       contiguous(astype(context_lens, int32, s), false, s),
       contiguous(astype(tile_seq, int32, s), false, s),
       contiguous(astype(tile_local0, int32, s), false, s),
       contiguous(astype(seq_qlen, int32, s), false, s)});
}

void AttnVarlenPrefill::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("AttnVarlenPrefill has no CPU implementation.");
}

void AttnVarlenPrefill::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 8);
  auto& q_hm = inputs[0];
  auto& key_cache = inputs[1];
  auto& value_cache = inputs[2];
  auto& block_table = inputs[3];
  auto& context_lens = inputs[4];
  auto& tile_seq = inputs[5];
  auto& tile_local0 = inputs[6];
  auto& seq_qlen = inputs[7];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  const int H = q_hm.shape(0);
  const int total_padded = q_hm.shape(1);
  const int D = q_hm.shape(2);
  const int H_KV = key_cache.shape(2);
  const int block_size = key_cache.shape(1);
  const int bt_stride = block_table.shape(1);
  const int n_tiles = tile_seq.shape(0);

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_attn_varlen_prefill(enc, q_hm, key_cache, value_cache, block_table, context_lens,
                                 tile_seq, tile_local0, seq_qlen, out, n_tiles, total_padded, H,
                                 H_KV, block_size, bt_stride, scale_, D);
}

std::vector<array> AttnVarlenPrefill::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("AttnVarlenPrefill has no jvp implementation.");
}
std::vector<array> AttnVarlenPrefill::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("AttnVarlenPrefill has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> AttnVarlenPrefill::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("AttnVarlenPrefill has no vmap implementation.");
}

} // namespace mlx::core
