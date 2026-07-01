// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "mamba2/mamba2.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <vecLib/cblas_new.h>
#endif

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

array ssd_chunked(const array& Cq, const array& Bm, const array& X, const array& cl,
                  StreamOrDevice s) {
  const int B = Cq.shape(0), H = Cq.shape(1), N = Cq.shape(2), D = Cq.shape(3);
  const int C = N / 64;   // chunk L = 64 (must match SSD_CHUNK_L in the metal)
  auto kv = array({B, H, C, D, D}, float32,
                  std::make_shared<SsdChunkKV>(to_stream(s)), {Bm, X, cl});
  auto sex = array({B, H, C, D, D}, float32,
                   std::make_shared<SsdChunkScan>(to_stream(s)), {kv, cl});
  return array(Cq.shape(), bfloat16,
               std::make_shared<SsdChunkOut>(to_stream(s)), {Cq, Bm, X, cl, sex});
}

array mamba2(const array& C, const array& B, const array& X, const array& cumlog,
             StreamOrDevice s) {
  assert(C.dtype() == bfloat16 && B.dtype() == bfloat16 && X.dtype() == bfloat16);
  assert(cumlog.dtype() == float32);
  assert(C.shape() == B.shape() && B.shape() == X.shape());
  const int N = C.shape(2), D = C.shape(3);
  assert(D == 64 && "mamba2 currently supports D=64");
  assert(N % 8 == 0 && "mamba2: N must be a multiple of 8");
  if (N % 64 == 0 && N >= 128) {
    return ssd_chunked(C, B, X, cumlog, s);   // linear-time chunked pipeline
  }
  return array(C.shape(), bfloat16,
               std::make_shared<Mamba2>(to_stream(s)), {C, B, X, cumlog});
}

void SsdChunkKV::eval_cpu(const std::vector<array>&, std::vector<array>&) { assert(false); }
void SsdChunkKV::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& bm = inputs[0]; auto& x = inputs[1]; auto& cl = inputs[2];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int B = bm.shape(0), H = bm.shape(1), N = bm.shape(2), D = bm.shape(3);
  const int C = out.shape(2);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_ssd_chunk_kv(enc, bm, x, cl, out, static_cast<unsigned>(N),
                          static_cast<unsigned>(H), B, C, D);
}
std::vector<array> SsdChunkKV::jvp(const std::vector<array>&, const std::vector<array>&,
                                   const std::vector<int>&) {
  throw std::runtime_error("SsdChunkKV has no jvp implementation.");
}
std::vector<array> SsdChunkKV::vjp(const std::vector<array>&, const std::vector<array>&,
                                   const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("SsdChunkKV has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> SsdChunkKV::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("SsdChunkKV has no vmap implementation.");
}
bool SsdChunkKV::is_equivalent(const Primitive&) const { return true; }

void SsdChunkScan::eval_cpu(const std::vector<array>&, std::vector<array>&) { assert(false); }
void SsdChunkScan::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& sin = inputs[0]; auto& cl = inputs[1];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int B = sin.shape(0), H = sin.shape(1), C = sin.shape(2), D = sin.shape(3);
  const int N = cl.shape(2);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_ssd_chunk_scan(enc, sin, cl, out, static_cast<unsigned>(C),
                            static_cast<unsigned>(N), B * H, D);
}
std::vector<array> SsdChunkScan::jvp(const std::vector<array>&, const std::vector<array>&,
                                     const std::vector<int>&) {
  throw std::runtime_error("SsdChunkScan has no jvp implementation.");
}
std::vector<array> SsdChunkScan::vjp(const std::vector<array>&, const std::vector<array>&,
                                     const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("SsdChunkScan has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> SsdChunkScan::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("SsdChunkScan has no vmap implementation.");
}
bool SsdChunkScan::is_equivalent(const Primitive&) const { return true; }

void SsdChunkOut::eval_cpu(const std::vector<array>&, std::vector<array>&) { assert(false); }
void SsdChunkOut::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& cq = inputs[0]; auto& bm = inputs[1]; auto& x = inputs[2];
  auto& cl = inputs[3]; auto& sex = inputs[4];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int B = cq.shape(0), H = cq.shape(1), N = cq.shape(2), D = cq.shape(3);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_ssd_chunk_out(enc, cq, bm, x, cl, sex, out, static_cast<unsigned>(N),
                           static_cast<unsigned>(H), B, D);
}
std::vector<array> SsdChunkOut::jvp(const std::vector<array>&, const std::vector<array>&,
                                    const std::vector<int>&) {
  throw std::runtime_error("SsdChunkOut has no jvp implementation.");
}
std::vector<array> SsdChunkOut::vjp(const std::vector<array>&, const std::vector<array>&,
                                    const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("SsdChunkOut has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> SsdChunkOut::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("SsdChunkOut has no vmap implementation.");
}
bool SsdChunkOut::is_equivalent(const Primitive&) const { return true; }

void Mamba2::eval(const std::vector<array>&, std::vector<array>&) { assert(false); }
void Mamba2::eval_cpu(const std::vector<array>& in, std::vector<array>& out) { eval(in, out); }

void Mamba2::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  assert(inputs.size() == 4);
  auto& C = inputs[0]; auto& B = inputs[1]; auto& X = inputs[2]; auto& cumlog = inputs[3];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int Bsz = C.shape(0), H = C.shape(1), N = C.shape(2), D = C.shape(3);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_mamba2(enc, C, B, X, cumlog, out, static_cast<unsigned>(N),
                    static_cast<unsigned>(H), Bsz, D);
}

std::vector<array> Mamba2::jvp(const std::vector<array>&, const std::vector<array>&,
                               const std::vector<int>&) {
  throw std::runtime_error("Mamba2 has no jvp implementation.");
}
std::vector<array> Mamba2::vjp(const std::vector<array>&, const std::vector<array>&,
                               const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("Mamba2 has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> Mamba2::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("Mamba2 has no vmap implementation.");
}
bool Mamba2::is_equivalent(const Primitive&) const { return true; }

} // namespace mlx::core
