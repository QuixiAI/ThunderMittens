# Reference Kernel Porting Status

Date: 2026-06-30

This ledger records the inventory pass over every repository under `.reference/`.
The decision unit is a logical kernel family, not every template instantiation or
benchmark variant. A kernel should be ported when it adds a reusable
ThunderMittens primitive, is not already covered by an existing Metal kernel, is
not tightly coupled to another framework runtime, and can be validated with a
small correctness test.

## Ported In This Pass

| Reference | Logical kernels | ThunderMittens port |
| --- | --- | --- |
| `.reference/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal` | `kernel_reglu`, `kernel_geglu`, `kernel_swiglu`, `kernel_swiglu_oai`, `kernel_geglu_erf`, `kernel_geglu_quick` | `ThunderMittens/kernels/glu/` exposes `tk.glu`, `tk.reglu`, `tk.geglu`, `tk.swiglu`, `tk.swiglu_oai`, `tk.geglu_erf`, `tk.geglu_quick` for MLX and Torch MPS. |
| `.reference/vllm/csrc/libtorch_stable/activation_kernels.cu` | `act_and_mul`, parameterized GLU, `swigluoai_and_mul` | Covered by the same `glu` port. The vLLM activation family is semantically the same useful missing primitive. |
| `.reference/vllm-metal/vllm_metal/metal/kernels_v2/reshape_and_cache.metal`, `gather_kv_cache.metal`, `copy_blocks.metal`, `kv_scale_update.metal` | KV-cache scatter/reshape, gather, block copy, scale update helpers | `ThunderMittens/kernels/kv_cache/` exposes `tk.kv_cache_scatter`, `tk.kv_cache_gather`, `tk.kv_cache_copy_blocks`, and `tk.kv_cache_scales` for MLX and Torch MPS. The cache layout is `(num_blocks, block_size, num_heads, head_size)`. |
| `.reference/vllm-metal/vllm_metal/metal/kernels_v2/pagedattention*.metal`, `.reference/vllm/csrc/attention/*paged_attention*`, `.reference/vllm/csrc/libtorch_stable/attention/paged_attention*.cu` | Decode-time paged attention over block tables | `ThunderMittens/kernels/kv_cache/` exposes `tk.paged_attention` for MLX and Torch MPS. It supports f32/f16/bf16 same-type Q/K/V caches and head sizes 64 and 128. |
| `.reference/mlx/mlx/backend/metal/kernels/hadamard.h`, `.reference/llama.cpp/ggml/src/ggml-cuda/fwht.cu`, vLLM TurboQuant FWHT helpers | Walsh-Hadamard/FWHT transforms over power-of-two rows | `ThunderMittens/kernels/hadamard/` exposes `tk.hadamard` for MLX and Torch MPS with final-axis sizes 64, 128, 256, and 512. |

## Existing Coverage

These reference families already have first-class ThunderMittens kernels or a
close direct equivalent:

| Reference repos | Reference families | Existing ThunderMittens coverage |
| --- | --- | --- |
| `BitNet` | `ladder_int8xint2_kernel`, generated BitNet LUT/GEMM configs | `qgemv_int`, `qgemm_int`, `qgemm/qgemv` `bitnet` format. |
| `HipKittens` | softmax, layernorm, rotary, GQA attention forward/backward, bf16/fp8 GEMM variants | `softmax`, `layernorm`, `rms_norm`, `rotary`, `attn_fwd`, `attn_causal`, `attn_bwd`, `matmul_custom`, `gemm_staged`, `qgemm`. |
| `llama.cpp` | unary/binary elementwise, softmax, norms, RoPE, flash attention, quantized MV/MM, copy/get rows, pools | Covered by existing specialized kernels where ThunderMittens wants them; generic tensor ops remain better left to MLX/PyTorch. |
| `llm.metal` | GPT-2 layernorm, softmax, attention forward/backward, residual add, GELU, matmul | Covered by existing norms, attention, GELU, matmul, and add-style kernels. |
| `mlx` | CUDA backend softmax, norm, RoPE, scan, copy, GEMV, quantized QMV, reductions | ThunderMittens already targets selected Metal kernels; generic framework internals should stay in MLX. |
| `vllm` | quant GEMM/GEMV, RMSNorm, RoPE, Mamba selective scan | Dense/quant GEMM, norms, RoPE, attention, and Mamba-style kernels are already represented where reusable outside vLLM runtime state. |

## Rejected For Now

| Repo | Kernel families found | Decision |
| --- | --- | --- |
| `Metal-Puzzles` | tutorial map/zip/guard/broadcast/reduce/conv/matmul examples | Do not port. They are learning exercises, not production kernels. |
| `alloy` | compiler-generated MSL, cache tests, `alloy_decode_chunk_*` server kernels | Do not port. Alloy is a compiler/runtime; generated kernels should not be copied into this hand-written kernel library. |
| `llama.cpp` | repeat, concat, cumsum, triangular fill, argmax/argsort, im2col/conv/upscale/pad/roll/arange/timestep embedding, optimizer steps, memset/count-equal | Do not port now. They are generic framework/model-utility ops or llama.cpp runtime details, not ThunderMittens primitives. |
| `llm.metal` | embedding encoder, cross entropy, classifier loss, AdamW, training-only backward helpers, QKV permute/unpermute | Do not port now. Useful for a full training stack, but ThunderMittens does not currently expose a training graph/loss API. |
| `metal-benchmarks` | throughput/cache/float64/command-concurrency microbenchmarks | Do not port. These are hardware probes, not reusable model kernels. |
| `mlx-examples` | no standalone GPU kernel entrypoints found in this pass | Nothing to port. |
| `vllm-metal` | partitioned paged-attention reductions, FP8 cache decode paths, TurboQuant encode/decode/cache update variants, Gated Delta Net kernels, MLA-specific cache helpers | Do not port now. The reusable same-type KV-cache and decode-attention core is covered; these variants need quantized cache formats or model/runtime APIs that ThunderMittens does not expose yet. |
| `vllm` | sampler/top-k, MoE align/permute/grouped top-k, DeepSeek fused cache kernels, CUDA/ROCm-specific Q4 WMMA/RDNA kernels, distributed/cache-runtime mutation entrypoints | Do not port now. These depend on vLLM serving scheduler state, CUDA/ROCm-specific intrinsics, or a MoE/runtime API not present here. |

## Future Candidates

These were not ported in this pass, but are worth reconsidering if the public API
needs them:

| Candidate | Source | Why wait |
| --- | --- | --- |
| Fused Q/K norm + RoPE + KV-cache insert | `vllm` fused QK norm/RoPE/cache kernels | Now has a compatible KV-cache layout, but still needs a deliberate fused decode API and shape contract. |
| MoE top-k, token alignment, expert permute/unpermute | `vllm` MoE kernels | Needs a MoE execution API and expert routing representation. |
| Fused classifier / cross entropy | `llm.metal` | Useful for training, but should be designed with a loss/autograd API rather than as an isolated kernel. |
| RWKV/Gated Delta Net/SSM extras | `llama.cpp` | Model-family-specific sequence kernels; port when those model APIs are in scope. |
| Dynamic quant/dequant, FP8 KV caches, and per-token quantization | `mlx`, `vllm`, `vllm-metal` | Useful once ThunderMittens exposes quantization as a standalone API, not only packed-weight matmul/attention. |

## Notes

- The GLU port uses a flat 1-D elementwise Metal kernel instead of a fixed row
  tile, so it supports real MLP intermediate sizes such as 4096 and 11008.
- The MLX extension needed a small compatibility update: each local primitive
  class now has a `name()` method. This does not affect launch behavior, but it
  lets the extension rebuild against newer MLX headers that require primitive
  names.
- The KV-cache port intentionally uses same-type floating caches for now. It
  preserves the useful cache scatter/gather/copy/decode behavior without locking
  ThunderMittens into a vLLM-specific FP8/TurboQuant cache representation.
- `kv_cache_scales` computes the FP8 scale convention used by vLLM
  (`absmax / 240`) as a standalone helper; it does not yet mutate external scale
  tensors in-place the way the vLLM runtime helper can.
