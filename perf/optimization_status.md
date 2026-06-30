# ThunderMittens Optimization Status

Running notebook for the Metal kernel optimization project.

This file is the tracking ledger for each kernel as it moves through reference
review, baselining, experiments, verification, and landing. The first qgemv
optimization pass is recorded below.

## Run Metadata Template

Fill this out for every benchmark batch.

```text
Date:
Git revision / working-tree label:
Machine:
Chip / GPU cores:
Memory:
macOS:
Xcode / Metal tools:
Python:
MLX:
PyTorch:
Power mode / thermal notes:
Command:
Raw results path:
Notes:
```

## Global Status

| Area | Status | Notes |
|---|---|---|
| Measurement harness | Implemented | `perf/bench_kernels.py` writes `run.json`, `results.jsonl`, and `summary.md`; supports `--backend mlx|torch|both`, `--preset smoke|quick|comprehensive`, `--kernel`, `--formats`, timing counts, and Markdown output. |
| Reference inventory | In progress | `qgemv`, `qgemm`, `qflux`, integer quant kernels, and `attn_q` reference passes recorded below. Search `.reference/` before each kernel and record exact files read. |
| Correctness baseline | Existing tests available | Run from `ThunderMittens/kernels`: `python -m pytest */correctness/ tk_torch/tests/ tests_parity/ -q`. |
| Raw results storage | Implemented | Harness writes `perf/results/YYYY-MM-DD/<run-id>/`; raw result directories are git-ignored by `perf/results/.gitignore`. |
| Publication | Not started | Commit/push only after verified wins and explicit user request. No AI attribution trailers. |

## Harness Smoke Validation

These are harness validation runs only, not performance baselines for decision-making.

| Date | Backend | Command | Result | Raw results |
|---|---|---|---|---|
| 2026-06-29 | MLX | `.venv/bin/python perf/bench_kernels.py --backend mlx --preset smoke --kernel all --warmup 0 --iters 1 --repeats 1 --output-dir perf/results/smoke-mlx --markdown` | 34 ok, 0 skipped, 0 failed | `perf/results/smoke-mlx/summary.md` |
| 2026-06-29 | PyTorch MPS | `.venv/bin/python perf/bench_kernels.py --backend torch --preset smoke --kernel all --warmup 0 --iters 1 --repeats 1 --output-dir perf/results/smoke-torch --markdown` | 34 ok, 0 skipped, 0 failed | `perf/results/smoke-torch/summary.md` |
| 2026-06-29 | MLX | `.venv/bin/python perf/bench_kernels.py --backend mlx --preset quick --kernel qgemv,qgemm --formats q8_0,q4_0 --warmup 0 --iters 1 --repeats 1 --output-dir perf/results/quick-qgemv-qgemm --no-markdown` | 8 ok, 4 skipped, 0 failed; skips are qgemm M<32 unsupported tile shapes | `perf/results/quick-qgemv-qgemm/results.jsonl` |
| 2026-06-30 | MLX | `.venv/bin/python perf/bench_kernels.py --backend mlx --preset comprehensive --kernel all --warmup 3 --iters 10 --repeats 3 --output-dir perf/results/final-all-mlx-20260630 --markdown` | 196 ok, 4 skipped, 0 failed | `perf/results/final-all-mlx-20260630/results.jsonl` |

Full baseline commands to run next:

```bash
.venv/bin/python perf/bench_kernels.py --backend mlx --preset comprehensive --kernel all
.venv/bin/python perf/bench_kernels.py --backend torch --preset smoke --kernel all
```

## Kernel Matrix

| Kernel | Status | Priority | Reference leads | Baseline state | Open first question |
|---|---|---:|---|---|---|
| `qgemv` | Optimized pass 1 | P0 | llama.cpp Metal `mul_mv`, vLLM-Metal quant, BitNet | Baseline and q8/q4 pass complete | Does packed-weight GB/s scale with bits per weight and beat dequant-then-matmul? |
| `qgemm` | Investigated pass 1 | P0 | ThunderKittens GEMM, llama.cpp Metal, Marlin-style notes | Baseline and rejected experiments complete | Direct fragment path remains best among tested variants; small-M crossover still needs a separate API design. |
| `qflux` | Investigated pass 1 | P1 | `qgemm`, ThunderKittens flux | Baseline and rejected experiment complete | Register epilogue fusion was correct but not a repeatable win; keep split epilogue for now. |
| `qgemv_int` | Optimized pass 1 | P1 | BitNet W2A8, integer-dot paths | Baseline and W2A8 pass complete | W2A8 avoids per-group simd reductions; W8A8 unroll did not hold up. |
| `qgemm_int` | Optimized pass 1 | P2 | BitNet, integer-dot paths | Baseline and W2A8 pass complete | Exact-int prefill remains slower than half-MMA, but W2A8 no longer pays per-group reductions. |
| `attn_q` | Investigated pass 1 | P1 | MLX SDPA, llm.metal attention/softmax | Baseline, rejected experiments, and harness fix complete | q4 multiwarp is slower than q4 single-warp on the same shape; keep it guarded out. |
| `attn_fwd` | Investigated pass 1 | P1 | MLX SDPA, llm.metal attention/softmax | Baseline and rejected experiments complete | Register-map fusion was not reproducible; keep original path. |
| `attn_causal` | Investigated pass 1 | P1 | MLX SDPA, llm.metal causal softmax | Baseline and rejected experiments complete | Fused map helped some shapes but regressed smaller D64 cases. |
| `attn_multiwarp` | Investigated pass 1 | P2 | MLX SDPA, local multiwarp staging | Baseline and rejected experiments complete | Shared K/V staging is shape-sensitive and should not be routed blindly. |
| `attn_bwd` | Optimized pass 1 | P1 | ThunderKittens attention backward, llm.metal backward | Baseline and K-layout pass complete | dQ was paying a redundant K global load; derive col layout from the row tile instead. |
| `matmul_custom` | Investigated pass 1 | P2 | ThunderKittens GEMM, Apple MPP guide | Baseline and rejected routing experiment complete | Staged routing was shape-sensitive; keep explicit kernels separate for now. |
| `gemm_staged` | Investigated pass 1 | P2 | ThunderKittens GEMM, Apple MPP guide | Baseline complete | Existing 2-simdgroup A-staged tile remains useful but not a universal matmul replacement. |
| `flux` | Investigated pass 1 | P2 | ThunderKittens flux, llm.metal epilogues | Baseline and rejected staged experiment complete | Staged Flux was mixed; keep single-simdgroup fused epilogues. |
| `layernorm` | Investigated pass 1 | P2 | ThunderKittens layernorm, llm.metal layernorm | Baseline and rejected multi-row experiment complete | One simdgroup per row beats packing four rows into a threadgroup. |
| `rms_norm` | Investigated pass 1 | P2 | MLX fast path, layernorm sibling | Baseline and rejected multi-row experiment complete | Current row-wise reduction layout remains best in tested shapes. |
| `softmax` | Investigated pass 1 | P2 | Attention softmax, llm.metal softmax | Baseline and rejected multi-row experiment complete | Four-row threadgroups regressed; standalone softmax stays launch/overhead dominated. |
| `gelu` | Investigated pass 1 | P3 | ThunderKittens/MLX activation refs | Baseline and rejected multi-row experiment complete | Standalone GELU remains launch-bound; prefer fusion at call sites. |
| `rotary` | Investigated pass 1 | P3 | ThunderKittens rotary, MLX RoPE | Baseline complete | Split-half RoPE is simple bandwidth work; no retained change. |
| `add_rt` | Investigated pass 1 | P3 | N/A simple bandwidth kernel | Baseline complete | Keep as smoke/calibration kernel. |
| `linear_attn` | Investigated pass 1 | P2 | ThunderKittens linear_attention | Baseline complete | No retained change; O(N) scan idea for decay/Mamba did not hold up on Metal. |
| `lin_attn_causal` | Investigated pass 1 | P2 | ThunderKittens based/linear attention | Baseline complete | Existing scan-style causal kernel remains best among tested linear-family changes. |
| `lin_attn_decay` | Investigated pass 1 | P2 | ThunderKittens linear_attention, retention refs | Baseline and rejected scan experiment complete | Recurrence scan reduced algorithmic work but lost parallelism and failed larger-shape correctness. |
| `hedgehog` | Investigated pass 1 | P2 | ThunderKittens hedgehog | Baseline complete | No retained change in pass 1. |
| `based` | Investigated pass 1 | P2 | ThunderKittens based | Baseline complete | No retained change in pass 1. |
| `mamba2` | Investigated pass 1 | P2 | ThunderKittens mamba2 | Baseline and rejected scan experiment complete | Recurrence scan was slower and failed N512 correctness. |
| `cmplx_matmul` | Optimized pass 1 | P3 | ThunderKittens fftconv building blocks | Baseline and large-K dispatch complete | Use in-place complex MMA accumulation only for K>=512. |
| `fftconv` | Investigated pass 1 | P3 | ThunderKittens fftconv | Baseline and rejected ping-pong experiment complete | Register-copy removal increased liveness/register pressure and regressed. |

## qgemv

Status: optimized pass 1 complete in working tree.
Priority: P0.

### API And Contract

- Public entry points: `tk.qgemv(Wq, x, format=...)`; `tk.qgemm(..., M==1)` routes here.
- Backends: MLX extension and PyTorch MPS wrapper share `qgemv.metal` and `tk_launch.h`.
- Dtypes: packed `uchar` weights, `half` activations/output.
- Supported shapes: `Wq` represents `(N, K)` quantized rows; `x` is `(K, 1)`.
- Correctness tests: `ThunderMittens/kernels/qgemv/correctness/test_qgemv.py`.

### References Inspected

| File | Notes / ideas |
|---|---|
| `.reference/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal` | `mul_vec_q_n_f32_impl`, `kernel_mul_mv_q8_0_f32_impl`, and `kernel_mul_mv_ext_*` distribute multiple packed blocks across one simdgroup instead of assigning every lane to the same block. |
| `.reference/llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h` | q4_0 uses more rows per threadgroup in llama.cpp; q8_0 uses a separate row factor and shared reduction path. Multi-row threadgroups remain a follow-up. |
| `ThunderMittens/include/ops/warp/register/tile/dequant.metal` | Current per-element `FMT::dequant` API is correct and broad, but q8_0/q4_0 hot paths can avoid repeated generic per-element unpack work. |

### Baseline And Verification

Baseline selected-format harness:

```bash
.venv/bin/python perf/bench_kernels.py --backend mlx --preset comprehensive --kernel qgemv --formats q8_0,q4_0,q4_K,kU4B8,fp8_e4m3,bitnet --warmup 3 --iters 20 --repeats 3 --output-dir perf/results/qgemv-baseline-20260630 --markdown
```

Final selected-format harness:

```bash
.venv/bin/python perf/bench_kernels.py --backend mlx --preset comprehensive --kernel qgemv --formats q8_0,q4_0,q4_K,kU4B8,fp8_e4m3,bitnet --warmup 3 --iters 20 --repeats 3 --output-dir perf/results/qgemv-final-hostdispatch-20260630 --markdown
```

Verification:

| Test | Result | Notes |
|---|---|---|
| `../../.venv/bin/python -m pytest qgemv/correctness/test_qgemv.py -q` | 117 passed | Run from `ThunderMittens/kernels` after final rebuild. |
| `.venv/bin/python perf/bench_kernels.py --backend torch --preset quick --kernel qgemv --formats q8_0,q4_0 --warmup 1 --iters 3 --repeats 2 --output-dir perf/results/qgemv-final-torch2-20260630 --markdown` | 3 ok, 0 skipped, 0 failed | Final-source PyTorch MPS verification. |

### Experiments

| ID | Hypothesis | Change | Result | Decision |
|---|---|---|---|---|
| E1 | A block-oriented loop for `block_k == 32` will avoid per-element `k/block_k` and `k%block_k`. | Added a generic `qgemv_bk32<FMT>` template and instantiated all 32-wide formats. | Regressed selected q8_0/q4_0/bitnet/fp8 small cases and q4_0 large-shape timing. Compiler likely already handled the division/modulo, while the new loop hurt scheduling. | Rejected and removed. |
| E2 | q8_0/q4_0 should process multiple packed blocks per simdgroup, following llama.cpp's lane/block distribution. | Added `qgemv_q8_0_fast` and `qgemv_q4_0_fast`; q8_0 processes 8 blocks per simdgroup iteration with 4 lanes per block; q4_0 processes 16 blocks with 2 lanes per block. | Correctness passed. Focused decode timing shows q8_0 wins clearly versus forced generic dispatch; q4_0 is positive but more variable. | Kept. |
| E3 | Small K should keep the old generic path if optimized packed-dot code is not beneficial there. | Added `qgemv_q8_0_small` and `qgemv_q4_0_small`; `launch_qgemv` dispatches those for `K <= 512`. | Correctness passed; avoids tying tiny harness behavior to the large-K optimized kernel shape. | Kept. |

Focused A/B timing notes, MLX, same seed and shapes, median ms:

| Shape `(N,K)` | Format | Generic dispatch | Final dispatch | Direction |
|---|---|---:|---:|---|
| `(4096,4096)` | q8_0 | 0.410 | 0.219 | 1.87x faster |
| `(11008,4096)` | q8_0 | 0.615 | 0.335 | 1.84x faster |
| `(11008,8192)` | q8_0 | 0.609 | 0.501 | 1.21x faster |
| `(4096,4096)` | q4_0 | 0.235 | 0.223 | 1.05x faster |
| `(11008,4096)` | q4_0 | 0.360 | 0.347 | 1.04x faster |
| `(11008,8192)` | q4_0 | 0.624 | 0.304-0.537 | variable, still faster in observed runs |

### Decision Log

- Keep q8_0/q4_0 packed-dot specialized kernels for large-K decode.
- Keep generic `_small` q8_0/q4_0 kernels for `K <= 512`.
- Do not extend this pattern blindly to all `block_k == 32` formats; the generic block loop experiment regressed.

### Follow-Ups

- Add a first-class focused qgemv decode preset/script so raw large-shape A/B results are saved instead of pasted from ad hoc runs.
- Test multi-row-per-threadgroup qgemv, especially for q4_0 and q8_0, inspired by llama.cpp `NR0` constants.
- Consider q4_1/q5_0/q5_1 packed-dot siblings only after q8/q4 results stabilize under the focused harness.
- Investigate q4_0 variance; one run showed large speedups, while conservative repeated timings showed smaller but still positive wins.

## qgemm

Status: investigated pass 1 complete; no qgemm code changes retained.
Priority: P0.

### API And Contract

- Public entry points: `tk.qgemm(Wq, X, format=...)`; `tk.qgemm_direct(...)`.
- Current default: both public MLX entry points dispatch `qgemm_frag_*`, the direct-to-fragment path. The dequant-to-shared `qgemm_*` kernels remain compiled but are not the default public route.
- Backends: MLX extension and PyTorch MPS wrapper share `qgemm.metal`; PyTorch dispatches the direct fragment launcher.
- Dtypes: packed `uchar` weights, `half` activations/output, fp32 accumulators.
- Supported tile constraints: `N % 32 == 0`, `M % 32 == 0`; `tk.qgemm` only routes `M == 1` through qgemv, so `1 < M < 32` remains unsupported by this kernel.
- Correctness tests: `ThunderMittens/kernels/qgemm/correctness/test_qgemm.py`.

### References Inspected

| File | Notes / ideas |
|---|---|
| `.reference/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal` | `dequantize_q4_0`, `dequantize_q8_0`, and `kernel_mul_mm_*` use per-format dequant helpers feeding simdgroup matrix paths. This inspired a q8/q4 fragment-fill specialization experiment. |
| `.reference/vllm/csrc/rocm/q_gemm_rdna3_wmma.cu` | The 2-wave WMMA notes discuss sharing/dequant staging to hide dequant and reduce redundant loads. This inspired the X-shared two-simdgroup experiment. |
| `ThunderMittens/kernels/qgemm/qgemm.metal` | Current direct fragment path already avoids shared W staging and barriers. The public default has already absorbed the most important staged-vs-fragment optimization. |

### Baseline And Verification

Baseline selected-format harness:

```bash
.venv/bin/python perf/bench_kernels.py --backend mlx --preset comprehensive --kernel qgemm,qgemm_direct --formats q8_0,q4_0,q4_K,kU4B8,fp8_e4m3,bitnet --warmup 3 --iters 20 --repeats 3 --output-dir perf/results/qgemm-baseline-20260630 --markdown
```

Final restored-source harness:

```bash
.venv/bin/python perf/bench_kernels.py --backend mlx --preset comprehensive --kernel qgemm,qgemm_direct --formats q8_0,q4_0,q4_K,kU4B8,fp8_e4m3,bitnet --warmup 3 --iters 20 --repeats 3 --output-dir perf/results/qgemm-restored-20260630 --markdown
```

Verification:

| Test | Result | Notes |
|---|---|---|
| `../../.venv/bin/python -m pytest qgemm/correctness/test_qgemm.py -q` | 116 passed | Run from `ThunderMittens/kernels` after final restored rebuild. |
| Final restored-source harness | 15 ok, 4 skipped, 0 failed | Skips are expected `M=2/4/8/16` cases because qgemm requires `M % 32 == 0`. |

Selected restored-source timing, MLX, median ms:

| Shape `(N,K,M)` | Format | `qgemm` | `qgemm_direct` | Notes |
|---|---|---:|---:|---|
| `(128,512,64)` | q8_0 | 0.2427 | 0.2376 | Both public entry points use the fragment path. |
| `(128,512,64)` | q4_0 | 0.2410 | 0.2390 | Stable relative to restored direct-fragment path. |
| `(128,512,64)` | q4_K | 0.6416 | 0.6408 | No experiment improved this path; timing was noisy across runs. |
| `(128,512,64)` | kU4B8 | 0.4636 | 0.4777 | No retained change. |
| `(128,512,64)` | fp8_e4m3 | 0.2716 | 0.2835 | No retained change. |
| `(128,512,64)` | bitnet | 0.2616 | 0.2510 | No retained change. |

### Experiments

| ID | Hypothesis | Change | Result | Decision |
|---|---|---|---|---|
| E1 | q8_0/q4_0 can beat the generic fragment dequantizer by hoisting the block base and scale once per fragment row. | Added temporary q8/q4 register-fill helpers and routed `qgemm_frag_q8_0` / `qgemm_frag_q4_0` to dedicated fast host functions; retained `_generic` names for A/B routing. | Correctness passed. Focused A/B was mixed and noisy: some duplicate rows improved, others tied or regressed, and the restored generic path stayed competitive. The added code was not justified by a reliable win. | Rejected and removed. |
| E2 | A two-simdgroup qgemm fragment kernel can share each X tile across two adjacent N blocks and reduce duplicated activation loads. | Added temporary `qgemm_frag_xshared_*` kernels for selected formats and dispatched them when `N % 64 == 0`. | Correctness passed, but performance regressed clearly: q4_0 `0.2483 -> 0.3878` ms, q4_K `0.3234 -> 0.5654` ms, q8_0 `0.2499 -> 0.2827` ms versus baseline medians. Shared-memory barriers and occupancy cost outweighed saved X loads. | Rejected and removed. |

### Decision Log

- Keep the existing direct-to-fragment `qgemm_frag_*` default.
- Do not add q8/q4 fragment-fill specializations without a more stable, lower-noise microbenchmark that can select host functions in one binary.
- Do not pursue X-sharing for the current 32x32 fragment qgemm geometry; L2/global load behavior plus barrier-free direct loads beat explicit shared staging.
- Treat `qgemm_direct` as a naming/API cleanup candidate later, not a performance variant, because it currently dispatches the same path as `qgemm`.

### Follow-Ups

- Add a benchmark-only way to select direct fragment, staged, and experimental qgemm host functions in one compiled metallib; rebuild-to-rebuild comparisons were too noisy for marginal experiments.
- Revisit `1 < M < 32` support as an API/kernel design problem. Current qgemv routing only handles `M == 1`.
- Investigate q4_K/kU4B8 variance and absolute throughput separately; the pass here ruled out two structural changes but did not deeply tune complex quant formats.

## qflux

Status: investigated pass 1 complete; no qflux code changes retained.
Priority: P1.

### API And Contract

- Public entry point: `tk.qflux_gelu(Wq, X, bias, format=...)`.
- Backends: MLX extension and PyTorch MPS wrapper share `qflux.metal` and `tk_launch.h`.
- Dtypes: packed `uchar` weights, `half` activations/bias/output, fp32 accumulators.
- Operation: `D = gelu(dequantize(Wq) @ X + bias)`.
- Supported tile constraints: `N % 32 == 0`, `M % 32 == 0`.
- Correctness tests: `ThunderMittens/kernels/qflux/correctness/test_qflux.py`.

### References Inspected

| File | Notes / ideas |
|---|---|
| `ThunderMittens/kernels/qflux/qflux.metal` | qflux already uses the `qgemm_frag` style direct-to-register quantized matmul and then runs `add_col` plus `gelu` on the accumulator tile. |
| `ThunderMittens/kernels/flux/flux.metal` | The non-quantized fused flux kernel uses the same conceptual epilogue, so qflux-specific wins need to come from quantized matmul or accumulator epilogue mechanics. |
| `.reference/llm.metal/dev/cuda/matmul_forward.cu` | cublasLt-style fused epilogues inspired testing a single register pass for bias plus GELU. |
| `.reference/llm.metal/dev/cuda/gelu_forward.cu` | Confirms standalone GELU is simple elementwise work and should usually be fused when launch or memory traffic dominates. |

### Baseline And Verification

Baseline selected-format harness:

```bash
.venv/bin/python perf/bench_kernels.py --backend mlx --preset comprehensive --kernel qflux_gelu --formats q8_0,q4_0,q4_K,kU4B8,fp8_e4m3,bitnet --warmup 3 --iters 20 --repeats 3 --output-dir perf/results/qflux-baseline-20260630 --markdown
```

Experiment harnesses:

```bash
.venv/bin/python perf/bench_kernels.py --backend mlx --preset comprehensive --kernel qflux_gelu --formats q8_0,q4_0,q4_K,kU4B8,fp8_e4m3,bitnet --warmup 3 --iters 20 --repeats 3 --output-dir perf/results/qflux-fused-epilogue-20260630 --markdown
.venv/bin/python perf/bench_kernels.py --backend mlx --preset comprehensive --kernel qflux_gelu --formats q8_0,q4_0,q4_K,kU4B8,fp8_e4m3,bitnet --warmup 5 --iters 30 --repeats 5 --output-dir perf/results/qflux-selective-epilogue-20260630 --markdown
.venv/bin/python perf/bench_kernels.py --backend mlx --preset comprehensive --kernel qflux_gelu --formats q4_K,kU4B8,fp8_e4m3,bitnet --warmup 8 --iters 50 --repeats 7 --output-dir perf/results/qflux-selective-heavy-repeat-20260630 --markdown
```

Verification:

| Test | Result | Notes |
|---|---|---|
| `../../.venv/bin/python -m pytest qflux/correctness/test_qflux.py -q` | 58 passed | Passed with the fused experiment and again after restoring source. |
| Baseline harness | 6 ok, 0 skipped, 0 failed | All selected quant formats covered. |
| Selective and focused experiment harnesses | 10 ok, 0 skipped, 0 failed | Performance-only experiment; correctness separately passed. |

Selected timing, MLX, median ms:

| Format | Baseline | Full fused first run | Selective fused repeat | Focused heavy repeat | Decision |
|---|---:|---:|---:|---:|---|
| q8_0 | 0.2684 | 0.2752 | 0.2647 | n/a | Do not fuse. |
| q4_0 | 0.2730 | 0.2703 | 0.2641 | n/a | Do not fuse; tiny changes are within noise. |
| q4_K | 0.6615 | 0.6496 | 0.6582 | 0.6560 | Do not fuse. |
| kU4B8 | 0.4970 | 0.4894 | 0.4962 | 0.4950 | Do not fuse. |
| fp8_e4m3 | 0.3115 | 0.2058 | 0.3085 | 0.3030 | Do not fuse; one fast run did not repeat. |
| bitnet | 0.2749 | 0.1827 | 0.2802 | 0.2740 | Do not fuse; one fast run did not repeat. |

### Experiments

| ID | Hypothesis | Change | Result | Decision |
|---|---|---|---|---|
| E1 | Combining bias add and tanh-GELU into one register-tile pass should reduce accumulator read/write work and improve the fused epilogue. | Added a temporary `add_col_gelu` helper over `rt<float, 32, 32>` and tested both all-format fusion and selective fusion for heavier formats. | Correctness passed. The first run showed large wins for fp8/bitnet, but selective and focused repeat runs did not reproduce them; q8/q4 were neutral or slightly worse. | Rejected and removed. |

### Decision Log

- Keep the existing split epilogue: `add_col(d_reg, ...)` followed by `gelu(d_reg, d_reg)`.
- Treat one-off qflux wins as noise unless reproduced in a same-binary A/B harness or with a repeated rebuild protocol.
- qflux’s main remaining opportunity is likely shared with qgemm: quantized fragment dequant throughput for complex formats, not the simple register epilogue.

### Follow-Ups

- Add same-binary A/B host names for qflux epilogues if epilogue work is revisited; rebuild-to-rebuild variance is too high for small deltas.
- Revisit after q4_K/kU4B8 dequant paths are tuned in qgemm, since qflux inherits that matmul core.
- Compare fused qflux versus explicit `qgemm + bias + gelu` at the Python graph level to confirm the public fused operator is still valuable despite no internal epilogue change.

## qgemv_int

Status: optimized pass 1 complete in working tree.
Priority: P1.

### API And Contract

- Public entry points: `tk.qgemv_w8a8(Wq, Xq, w_scale, a_scale)` and `tk.qgemv_w2a8(Wq, Xq, a_scale)`.
- Backends: MLX extension and PyTorch MPS wrapper share `qgemv_int.metal` and `tk_launch.h`.
- Dtypes: int8 activations, int8 W8A8 weights, BitNet uint8-packed W2A8 weights, half scales/output.
- Operation:
  - W8A8: exact int8xint8 int32 dot, then per-row weight scale and activation scale.
  - W2A8: ternary weights with per-group half scale times int8 activations, then activation scale.
- Supported shapes: decode GEMV `(N,K) @ (K,1)`; W8A8 requires `K % 4 == 0`; W2A8 requires `K` represented as 32-wide BitNet blocks.
- Correctness tests: `ThunderMittens/kernels/qgemv_int/correctness/test_qgemv_int.py`.

### References Inspected

| File | Notes / ideas |
|---|---|
| `.reference/BitNet/gpu/bitnet_kernels/bitnet_kernels.h` | CUDA W2A8 uses shape-specialized decode kernels, decodes int2 to int8, uses `__dp4a`, and processes multiple rows per block. The direct Metal analogue is constrained by the current BitNet block layout and lack of int8 matrix hardware. |
| `.reference/BitNet/src/ggml-bitnet-mad.cpp` | CPU W2A8 kernels emphasize vector dot operations and multi-row/multi-column variants. This supports revisiting multi-row threadgroups later. |
| `ThunderMittens/include/ops/warp/register/tile/dequant.metal` | Defines `idot4` and BitNet `code/gscale`; current W8A8 already uses the expected dp4a-like primitive. |

### Baseline And Verification

Baseline harness:

```bash
.venv/bin/python perf/bench_kernels.py --backend mlx --preset comprehensive --kernel qgemv_w8a8,qgemv_w2a8 --warmup 3 --iters 20 --repeats 3 --output-dir perf/results/qgemv-int-baseline-20260630 --markdown
```

Final harness:

```bash
.venv/bin/python perf/bench_kernels.py --backend mlx --preset comprehensive --kernel qgemv_w8a8,qgemv_w2a8 --warmup 5 --iters 30 --repeats 5 --output-dir perf/results/qgemv-int-final-laneacc-20260630 --markdown
```

Verification:

| Test | Result | Notes |
|---|---|---|
| `../../.venv/bin/python -m pytest qgemv_int/correctness/test_qgemv_int.py -q` | 5 passed | NumPy oracle emits warnings for some BitNet reference matmuls, but tests pass. |
| Final MLX harness | 2 ok, 0 skipped, 0 failed | W8A8 and W2A8 default shape covered. |
| `.venv/bin/python perf/bench_kernels.py --backend torch --preset quick --kernel qgemv_w8a8,qgemv_w2a8 --warmup 1 --iters 3 --repeats 2 --output-dir perf/results/qgemv-int-final-torch-20260630 --markdown` | 2 ok, 0 skipped, 0 failed | PyTorch MPS wrapper verified with same metallib source. |

Selected timing, MLX, median ms:

| Shape `(N,K)` | Kernel | Baseline | Final | Direction |
|---|---|---:|---:|---|
| `(128,256)` | W8A8 | 0.2401 | 0.1629 | No source change; run-to-run/harness variance. |
| `(128,256)` | W2A8 | 0.1597 | 0.1258 | 1.27x faster. |
| `(4096,4096)` | W2A8 | 0.5610 | 0.4431-0.4470 | About 1.26x faster. |
| `(11008,4096)` | W2A8 | 0.4984 | 0.5289-0.746 | Mixed/noisy; not a clear win at this shape. |
| `(11008,8192)` | W2A8 | 0.7575 | 0.6473-0.6710 | About 1.13-1.17x faster. |

### Experiments

| ID | Hypothesis | Change | Result | Decision |
|---|---|---|---|---|
| E1 | W2A8 should avoid one `simd_sum` per 32-weight group; per-lane scaled accumulation plus one final reduction is algebraically equivalent and cuts reduction overhead. | Replaced per-group int reduction with `lane_acc += float(code * xq) * gscale`, followed by one final `metal::simd_sum(lane_acc)`. | Correctness passed. Standard W2A8 median improved `0.1597 -> 0.1258` ms. Focused decode improved `4096x4096` and `11008x8192`, while `11008x4096` was noisy/mixed. | Kept. |
| E2 | W8A8 can gain from four-way unrolling the `idot4` loop to reduce loop overhead and expose ILP. | Temporarily accumulated four strided `idot4` chains per lane before the tail loop. | Correctness passed, but focused timing was not a clean win: `4096x4096` regressed versus the prior build and larger shapes tied. | Rejected and removed. |

### Decision Log

- Keep the W2A8 lane-scaled single-reduction path.
- Keep W8A8 unchanged; it already uses the local `idot4` primitive and unrolling did not justify extra code.
- Do not add shape-gated W2A8 dispatch until a same-binary variant harness exists; the only mixed shape was noisy and not enough to offset the simple general reduction win.

### Follow-Ups

- Add focused quantized-decode shapes to `perf/bench_kernels.py` or a companion script so qgemv/qgemv_int large-shape timings are saved as JSONL.
- Revisit multi-row/threadgroup W2A8 inspired by BitNet CUDA/CPU references; current path still launches one simdgroup per output row.
- Consider a same-binary W2A8 A/B host pair before shape-gating, especially for `N=11008,K=4096`.

## qgemm_int

Status: optimized pass 1 complete in working tree.
Priority: P2.

### API And Contract

- Public entry points: `tk.qgemm_w8a8(Wq, Xq, w_scale, a_scale)` and `tk.qgemm_w2a8(Wq, Xq, a_scale)`.
- Backends: MLX extension and PyTorch MPS wrapper share `qgemm_int.metal` and `tk_launch.h`.
- Dtypes: int8 activations, int8 W8A8 weights, BitNet uint8-packed W2A8 weights, half scales/output.
- Operation:
  - W8A8: exact int8xint8 int32 dot per `(n,m)`, then per-row weight scale and per-token activation scale.
  - W2A8: ternary weights with per-group half scale times int8 activations, then per-token activation scale.
- Supported shapes: prefill GEMM `(N,K) @ (K,M)` with token-major activation storage `(M,K)`.
- Correctness tests: `ThunderMittens/kernels/qgemm_int/correctness/test_qgemm_int.py`.

### References Inspected

| File | Notes / ideas |
|---|---|
| `.reference/BitNet/gpu/bitnet_kernels/bitnet_kernels.h` | CUDA W2A8 prefill/decode relies on shape-specialized int2 decode plus `__dp4a`; the current Metal exact-int prefill is a scalar-ALU path, not tensor-core MMA. |
| `.reference/BitNet/src/ggml-bitnet-mad.cpp` | CPU code contains multiple row/column variants, reinforcing that the current one-simdgroup-per-row plus M loop is only a first exactness path. |
| `ThunderMittens/kernels/qgemm_int/qgemm_int.metal` | qgemm_int repeats row-wise GEMV work for each M column; W2A8 had the same per-group `simd_sum` cost fixed in qgemv_int. |

### Baseline And Verification

Baseline harness:

```bash
.venv/bin/python perf/bench_kernels.py --backend mlx --preset comprehensive --kernel qgemm_w8a8,qgemm_w2a8 --warmup 3 --iters 20 --repeats 3 --output-dir perf/results/qgemm-int-baseline-20260630 --markdown
```

Final harness:

```bash
.venv/bin/python perf/bench_kernels.py --backend mlx --preset comprehensive --kernel qgemm_w8a8,qgemm_w2a8 --warmup 5 --iters 30 --repeats 5 --output-dir perf/results/qgemm-int-laneacc-20260630 --markdown
```

Verification:

| Test | Result | Notes |
|---|---|---|
| `../../.venv/bin/python -m pytest qgemm_int/correctness/test_qgemm_int.py -q` | 4 passed | NumPy oracle emits warnings for some BitNet reference matmuls, but tests pass. |
| Final MLX harness | 2 ok, 0 skipped, 0 failed | W8A8 and W2A8 default prefill shape covered. |
| `.venv/bin/python perf/bench_kernels.py --backend torch --preset quick --kernel qgemm_w8a8,qgemm_w2a8 --warmup 1 --iters 3 --repeats 2 --output-dir perf/results/qgemm-int-final-torch-20260630 --markdown` | 2 ok, 0 skipped, 0 failed | PyTorch MPS wrapper verified with same metallib source. |

Selected timing, MLX, median ms:

| Shape `(N,K,M)` | Kernel | Baseline | Final | Direction |
|---|---|---:|---:|---|
| `(128,512,64)` | W8A8 | 0.4307 | 0.4462 | No source change; run-to-run variance. |
| `(128,512,64)` | W2A8 | 0.8088 | 0.7643 | 1.06x faster. |

### Experiments

| ID | Hypothesis | Change | Result | Decision |
|---|---|---|---|---|
| E1 | The W2A8 qgemv single-reduction rewrite should also help prefill, because each `(n,m)` output currently performs one `simd_sum` per 32-weight group. | Replaced per-group int reduction inside each M iteration with `lane_acc += float(code * xq) * gscale`, followed by one final `metal::simd_sum(lane_acc)` per output. | Correctness passed. Default W2A8 median improved `0.8088 -> 0.7643` ms. | Kept. |

### Decision Log

- Keep the W2A8 lane-scaled single-reduction path in qgemm_int.
- Do not attempt W8A8 unrolling here; the qgemv_int unroll did not hold up, and qgemm_int is not the primary performance route for prefill.
- Treat qgemm_int as an exactness path unless future multi-row/M-tiled integer kernels materially improve throughput.

### Follow-Ups

- Compare qgemm_int against regular qgemm for larger exact-int use cases before spending more time; half-MMA is expected to remain faster for prefill.
- Revisit multi-row or M-tiled qgemm_int only with a same-binary variant harness.
- Consider sharing a small W2A8 lane-accumulation helper between qgemv_int and qgemm_int if another integer kernel adopts the same pattern.

## attn_q

Status: investigated pass 1 complete; no `attn_q.metal` kernel change retained.
Priority: P1.

### API And Contract

- Public entry point: `tk.attn_q(q, Kq, Vq, format=..., causal=False, multiwarp=False)`.
- Backends: MLX extension and PyTorch MPS wrapper share `attn_q.metal` and `tk_launch.h`.
- Dtypes: bf16 Q/output, packed uchar K/V, fp32 accumulators.
- Supported formats: single-warp q8_0/q4_0/fp8_e4m3; multiwarp q8_0/fp8_e4m3 only.
- Supported head dimensions: D in `{64,128}`.
- Correctness tests: `ThunderMittens/kernels/attn_q/correctness/test_attn_q.py`.

### References Inspected

| File | Notes / ideas |
|---|---|
| `.reference/mlx/mlx/backend/metal/scaled_dot_product_attention.cpp` | MLX chooses bq/bk geometry, function constants for causal/alignment/mask state, and dispatches a 4-warp attention path for full attention. This supports treating multiwarp as shape/format-dependent rather than universally better. |
| `.reference/mlx/mlx/backend/metal/kernels/sdpa_vector.h` | Vector SDPA uses online softmax with per-simdgroup partials, threadgroup aggregation, and explicit barriers only when combining across simdgroups. This inspired the barrier scope experiment. |
| `.reference/llm.metal/dev/cuda/attention_backward.cu` | Causal softmax fuses scale and triangular masking into online softmax and uses warp-level reductions. Confirms the current one-warp online softmax structure is the right baseline. |
| `.reference/llm.metal/dev/cuda/softmax_forward.cu` | Online softmax variants reinforce avoiding extra materialization of attention scores unless the kernel needs a backward/saved-probability path. |

### Baseline And Verification

Baseline harness:

```bash
.venv/bin/python perf/bench_kernels.py --backend mlx --preset comprehensive --kernel attn_q --formats q8_0,q4_0,fp8_e4m3 --warmup 3 --iters 20 --repeats 3 --output-dir perf/results/attn-q-baseline2-20260630 --markdown
```

Final harness:

```bash
.venv/bin/python perf/bench_kernels.py --backend mlx --preset comprehensive --kernel attn_q --formats q8_0,q4_0,fp8_e4m3 --warmup 5 --iters 30 --repeats 5 --output-dir perf/results/attn-q-final-20260630 --markdown
```

Verification:

| Test | Result | Notes |
|---|---|---|
| `../../.venv/bin/python -m pytest attn_q/correctness/test_attn_q.py -q` | 16 passed | q4 causal coverage was added; expected NumPy warnings remain. |
| Final MLX harness | 8 ok, 0 skipped, 0 failed | Harness now avoids unsupported q4 multiwarp entry points. |
| `.venv/bin/python perf/bench_kernels.py --backend torch --preset quick --kernel attn_q --formats q8_0,q4_0,fp8_e4m3 --warmup 1 --iters 3 --repeats 2 --output-dir perf/results/attn-q-final-torch-20260630 --markdown` | 8 ok, 0 skipped, 0 failed | PyTorch MPS wrapper verified with the same guarded case set. |

Selected timing notes, MLX, median ms:

| Shape / variant | Format | Median | Baseline | Notes |
|---|---|---:|---:|---|
| `(1,2,64,64)` noncausal | q8_0 | 0.2886 | 0.1833 | Final snapshot; no kernel change retained. |
| `(1,2,64,64)` causal | q4_0 | 0.1457 | 0.1587 | q4 causal now has explicit pytest coverage. |
| `(1,4,128,64)` multiwarp | q8_0 | 0.1557 | 0.1195 | Existing q8 multiwarp remains useful versus q8 single-warp on same shape. |
| `(1,4,128,64)` multiwarp | fp8_e4m3 | 0.1613 | 0.1135 | Existing fp8 multiwarp remains useful versus fp8 single-warp on same shape. |

### Experiments

| ID | Hypothesis | Change | Result | Decision |
|---|---|---|---|---|
| E1 | Single-warp `attn_q` only uses one simdgroup, so `simdgroup_barrier` should be cheaper than `threadgroup_barrier` around K staging. | Temporarily replaced the two single-warp threadgroup barriers with simdgroup barriers. | Correctness passed, but A/B was mixed: q8/q4 slowed while one fp8 run improved and later failed to reproduce. | Rejected and removed. |
| E2 | The barrier-scope change may be fp8-specific because fp8 dequant work differs from q8/q4. | Added a compile-time fp8-only barrier helper using `metal::is_same_v<FMT, fp8_e4m3>`. | Correctness passed, but final timing did not preserve the earlier fp8 gain and perturbed integer-format timings. | Rejected and removed. |
| E3 | q4_0 should be able to use the same multiwarp structure as q8_0/fp8_e4m3. | Temporarily instantiated `attn_q_mw_q4_0_{64,128}`, expanded tests, and benchmarked it. | Correctness passed, but same-shape timing rejected it: q4 single-warp `(1,4,128,64)` measured about `0.167` ms while q4 multiwarp measured about `0.319` ms. | Rejected and removed. |
| E4 | The perf harness should not emit unsupported q4 multiwarp cases, and q4 causal should be part of correctness coverage. | Kept the harness format guard for multiwarp cases and added q4_0 to causal pytest parametrization. | MLX and Torch harnesses both report 8 ok, 0 skipped, 0 failed; pytest is 16 passed. | Kept. |

### Decision Log

- Keep existing `attn_q.metal` algorithms unchanged for this pass.
- Keep q4 multiwarp guarded out in `perf/bench_kernels.py`; single-warp q4 is faster on the tested multiwarp shape.
- Keep q4 causal correctness coverage because the kernel is instantiated and benchmarked.
- Revisit attention tuning with longer-context shapes and a same-binary variant harness; these tiny shapes are noisy and often dominated by launch/runtime overhead.

### Follow-Ups

- Add same-shape single-vs-multiwarp benchmark specs for q8/fp8 attention so future changes compare the relevant alternatives directly.
- Test longer contexts where K/V quantization bandwidth reduction can dominate dequant and launch overhead.
- Consider function-constant or dispatch-level selection for multiwarp only after q8/fp8 long-context wins are quantified against single-warp.

## Dense Attention Forward

Status: investigated pass 1 complete; no `attn_fwd`, `attn_causal`, or `attn_multiwarp` source changes retained.
Priority: P1/P2.

### API And Contract

- Public entry points: `tk.attn_fwd(q,k,v)`, `tk.attn_causal(q,k,v)`, and `tk.attn_multiwarp(q,k,v)`.
- Backends: MLX extension and PyTorch MPS wrapper share the same Metal kernels.
- Dtypes: bf16 Q/K/V/output, fp32 accumulators.
- Supported head dimensions: D in `{64,128}`; sequence lengths are tile-aligned in current launchers.
- Correctness tests: `attn_fwd/correctness/test_attn.py`, `attn_causal/correctness/test_attn_causal.py`, `attn_multiwarp/correctness/test_attn_multiwarp.py`.

### References Inspected

| File | Notes / ideas |
|---|---|
| `.reference/mlx/mlx/backend/metal/scaled_dot_product_attention.cpp` | MLX uses function constants and shape-selected bq/bk geometry; this argues for measured dispatch selection rather than assuming a universal single/multiwarp route. |
| `.reference/mlx/mlx/backend/metal/kernels/sdpa_vector.h` | MLX vector SDPA combines simdgroup partials with threadgroup memory only at cross-simdgroup aggregation points; this informed the multiwarp/staging review. |
| `.reference/llm.metal/dev/cuda/attention_backward.cu` | Causal softmax fuses scale and triangular masking in an online algorithm, matching the current ThunderMittens structure. |
| `.reference/llm.metal/dev/cuda/softmax_forward.cu` | Online softmax variants support the current no-materialized-probabilities forward path. |

### Baseline And Verification

Baseline harness:

```bash
.venv/bin/python perf/bench_kernels.py --backend mlx --preset comprehensive --kernel attn_fwd,attn_causal,attn_multiwarp --warmup 5 --iters 30 --repeats 5 --output-dir perf/results/attn-dense-baseline-20260630 --markdown
```

Verification:

| Test | Result | Notes |
|---|---|---|
| `../../.venv/bin/python -m pytest attn_fwd/correctness/test_attn.py attn_causal/correctness/test_attn_causal.py attn_multiwarp/correctness/test_attn_multiwarp.py -q` | 12 passed | Run after final restored-source rebuild. |
| Baseline MLX harness | 10 ok, 0 skipped, 0 failed | Baseline is also the final retained-source performance snapshot for this pass. |
| `.venv/bin/python perf/bench_kernels.py --backend torch --preset quick --kernel attn_fwd,attn_causal,attn_multiwarp --warmup 1 --iters 3 --repeats 2 --output-dir perf/results/attn-dense-final-torch-20260630 --markdown` | 6 ok, 0 skipped, 0 failed | PyTorch MPS wrapper check. |

Baseline timing, MLX, median ms:

| Kernel | Shape | Median | MLX baseline | Notes |
|---|---|---:|---:|---|
| `attn_fwd` D64 | `(1,2,256,64)` | 0.2529 | 0.2487 | Rough parity with MLX at small shape. |
| `attn_fwd` D64 | `(2,4,512,64)` | 0.4811 | 0.4686 | Rough parity. |
| `attn_fwd` D128 | `(1,2,256,128)` | 0.2982 | 0.3177 | Slightly ahead in this run. |
| `attn_fwd` D64 | `(1,4,1024,64)` | 0.3439 | 0.3453 | Rough parity. |
| `attn_causal` D64 | `(2,4,512,64)` | 0.2814 | 0.8489 | Causal mask savings are visible versus dense MLX baseline. |
| `attn_multiwarp` D64 | `(1,2,256,64)` | 0.1971 | 0.1466 | Existing multiwarp path trails MLX at this small shape. |

### Experiments

| ID | Hypothesis | Change | Result | Decision |
|---|---|---|---|---|
| E1 | `attn_fwd` should route aligned non-causal shapes through `attn_multiwarp` when shared K/V staging wins. | Measured same-shape single-warp versus multiwarp for `(1,2,256,64)`, `(2,4,512,64)`, `(1,4,1024,64)`, and `(1,2,256,128)`. | Results were mixed and noisy; multiwarp only won one long D64 sample and regressed most smaller shapes in the focused run. | Rejected; no dispatch change. |
| E2 | Fuse `sub` plus `exp2` register maps in `attn_fwd` using the existing `subexp2` helper to reduce per-KV-block register passes. | Temporarily replaced two map pairs in `attn_fwd`. | Correctness passed. One rerun improved all four shapes, but a later rerun after final rebuild regressed all four (`0.2805/0.4913/0.3045/0.3580` ms versus baseline `0.2529/0.4811/0.2982/0.3439`). | Rejected and removed. |
| E3 | The same fused map should help causal and multiwarp online softmax loops. | Temporarily added local `subexp2` helpers to `attn_causal` and `attn_multiwarp`. | Correctness passed, but causal D64 small/medium regressed and multiwarp regressed in the full slice. | Rejected and removed. |

### Decision Log

- Keep dense attention kernels unchanged in this pass.
- Do not auto-route `attn_fwd` to `attn_multiwarp` without longer-context, same-binary dispatch data.
- Do not retain the fused `subexp2` maps; the win was not reproducible enough for a hot attention path.

### Follow-Ups

- Add same-shape `attn_fwd` versus `attn_multiwarp` benchmark specs to the harness instead of using ad hoc `CaseSpec` runs.
- Add longer-context attention shapes; current presets are noisy and close to launch/runtime overhead for several cases.
- If revisiting register-map fusion, expose both host functions in one metallib to avoid rebuild-to-rebuild variance.

## Attention Backward

Status: optimized pass 1 complete in working tree.
Priority: P1.

### API And Contract

- Public entry point: `tk.attn_bwd(q, k, v, do, l_vec)`.
- Backends: MLX extension and PyTorch MPS wrapper share `attn_bwd.metal`.
- Dtypes: bf16 Q/K/V/dO/dQ/dK/dV, fp32 logsumexp and intermediate accumulators.
- Supported head dimensions: D in `{64,128}` for the current launch wrappers.
- Correctness tests: `ThunderMittens/kernels/attn_bwd/correctness/test_attn_bwd.py`.

### References Inspected

| File | Notes / ideas |
|---|---|
| `.reference/llm.metal/dev/cuda/attention_backward.cu` | FlashAttention-style backward splits dQ and dK/dV work and reuses loaded K/V tiles aggressively. This motivated checking duplicate K loads in the Metal dQ path. |
| `ThunderMittens/kernels/attn_bwd/attn_bwd.metal` | `attn_bwd_dq` loaded the same K tile twice, once as col layout and once as row layout. The row layout is already needed for `mma_ABt`, and `swap_layout` can derive the col layout used by `mma_AB`. |
| `ThunderMittens/include/ops/warp/register/tile/maps.metal` | Existing layout conversion helpers made this a local change with no new shared-memory staging. |

### Baseline And Verification

Baseline harness:

```bash
.venv/bin/python perf/bench_kernels.py --backend mlx --preset comprehensive --kernel attn_bwd --warmup 5 --iters 30 --repeats 5 --output-dir perf/results/attn-bwd-baseline-20260630 --markdown
```

Final optimized harness:

```bash
.venv/bin/python perf/bench_kernels.py --backend mlx --preset comprehensive --kernel attn_bwd --warmup 5 --iters 30 --repeats 5 --output-dir perf/results/attn-bwd-k-swap-20260630 --markdown
```

Verification:

| Test | Result | Notes |
|---|---|---|
| `../../.venv/bin/python -m pytest attn_bwd/correctness/test_attn_bwd.py -q` | 5 passed | NumPy oracle emits existing overflow/divide warnings, but all tests pass. |
| Final MLX harness | 4 ok, 0 skipped, 0 failed | Error unchanged from baseline. |
| `.venv/bin/python perf/bench_kernels.py --backend torch --preset quick --kernel attn_bwd --warmup 1 --iters 3 --repeats 2 --output-dir perf/results/attn-bwd-final-torch-20260630 --markdown` | 4 ok, 0 skipped, 0 failed | PyTorch MPS wrapper check. |

Final timing, MLX, median ms:

| Variant | Shape | Baseline | Final | Direction | Error |
|---|---|---:|---:|---:|---:|
| noncausal | `(1,2,64,64)` | 0.2660 | 0.2412 | 1.10x faster | 0.00685 |
| causal | `(1,2,64,64)` | 0.2601 | 0.1571 | 1.66x faster | 0.00685 |
| noncausal | `(1,2,64,128)` | 0.3376 | 0.1637 | 2.06x faster | 0.00460 |
| causal | `(1,2,64,128)` | 0.3329 | 0.1661 | 2.00x faster | 0.00460 |

### Experiments

| ID | Hypothesis | Change | Result | Decision |
|---|---|---|---|---|
| E1 | Remove dKV loads that are provably unused so the compiler has less dead code to reason about. | Removed unused `k_row`, `v_row`, and `q_col` loads/declarations from `attn_bwd_dkv`. | Correctness passed. Timings by itself were mixed: small noncausal improved slightly, causal D128 regressed in the repeat. | Kept only as cleanup in the final combined faster version; do not count it as the perf win. |
| E2 | Avoid a redundant K global read in dQ by loading K once in row layout and deriving the col layout with `swap_layout`. | Replaced the separate `load(k_col, ...)` in `attn_bwd_dq` with `swap_layout(k_col, k_row, laneId)`. | Correctness passed. All benchmarked shapes improved, with D128 roughly 2x faster and unchanged error. | Kept. |

### Decision Log

- Keep the `attn_bwd_dq` K-layout swap; it removes redundant global memory traffic and is a large repeatable win.
- Keep the dKV dead-load cleanup as source cleanup in the retained faster combined version, but do not treat it as independently proven performance-positive.
- Do not change launch geometry in this pass; the dominant bottleneck found here was duplicate K traffic, not block sizing.

### Follow-Ups

- Add longer sequence and larger batch/head backward shapes to the comprehensive preset; the current backward slice is intentionally small.
- If tuning dKV next, isolate it with a benchmark mode that can time dQ and dKV sub-kernels separately.
- Check whether a similar row-to-col layout derivation can remove duplicate loads in other kernels before adding shared-memory staging.

## Dense GEMM And Flux

Status: investigated pass 1 complete; no `matmul_custom`, `gemm_staged`, or `flux` source changes retained.
Priority: P2.

### API And Contract

- Public entry points: `tk.matmul_custom(x,y)`, `tk.gemm_staged(x,y)`, `tk.flux_gelu(x,w,bias)`, and `tk.flux_gate(x,w,bias,gate,residual)`.
- Backends: MLX extension and PyTorch MPS wrapper share the same Metal kernels and launch helpers.
- Dtypes: float32 and bf16 supported by source; current comprehensive perf slice uses bf16.
- Supported kernel tile constraints: `N % 32 == 0`, `M % 32 == 0`, `K % 16 == 0`; the high-level `tk.matmul_custom` wrapper pads and slices arbitrary shapes.
- Correctness tests: `matmul_custom/correctness/test_matmul.py`, `gemm_staged/correctness/test_gemm_staged.py`, `flux/correctness/test_flux.py`.

### References Inspected

| File | Notes / ideas |
|---|---|
| `ThunderMittens/kernels/gemm_staged/gemm_staged.metal` | Existing two-simdgroup kernel stages A once per K step and reuses it across two output-column warps. This inspired testing it as a public `matmul_custom` route and as the base for staged Flux epilogues. |
| `ThunderMittens/kernels/flux/flux.metal` | Current Flux kernels use a single-simdgroup register-tile GEMM followed by in-register epilogues; there is no global round trip to fuse away. |
| `.reference/llm.metal/dev/cuda/matmul_forward.cu` | Fused epilogue patterns support keeping Flux epilogues in-register, but do not imply that larger multi-warp tiles are better on Metal. |
| `ThunderMittens/kernels/tk_launch.h` | Local comments already record a rejected 4-simdgroup BM=128 staged GEMM tile, so this pass tested the existing 2-simdgroup geometry only. |

### Baseline And Verification

Baseline harness:

```bash
.venv/bin/python perf/bench_kernels.py --backend mlx --preset comprehensive --kernel matmul_custom,gemm_staged,flux_gelu,flux_gate --warmup 5 --iters 30 --repeats 5 --output-dir perf/results/dense-gemm-baseline-20260630 --markdown
```

Final restored-source harness:

```bash
.venv/bin/python perf/bench_kernels.py --backend mlx --preset comprehensive --kernel matmul_custom,gemm_staged,flux_gelu,flux_gate --warmup 5 --iters 30 --repeats 5 --output-dir perf/results/dense-gemm-restored-20260630 --markdown
```

Verification:

| Test | Result | Notes |
|---|---|---|
| `../../.venv/bin/python -m pytest matmul_custom/correctness/test_matmul.py gemm_staged/correctness/test_gemm_staged.py flux/correctness/test_flux.py -q` | 32 passed | Run after final restored-source rebuild. |
| Final MLX harness | 16 ok, 0 skipped, 0 failed | Restored source after rejecting staged routing/Flux variants. |
| `.venv/bin/python perf/bench_kernels.py --backend torch --preset quick --kernel matmul_custom,gemm_staged,flux_gelu,flux_gate --warmup 1 --iters 3 --repeats 2 --output-dir perf/results/dense-gemm-final-torch-20260630 --markdown` | 8 ok, 0 skipped, 0 failed | PyTorch MPS wrapper check. |

Restored-source timing, MLX, selected median ms:

| Kernel | Shape | Median | Notes |
|---|---|---:|---|
| `matmul_custom` | `(128,64,128)` | 0.1533 | Wrapper still routes to the single-simdgroup matmul kernel. |
| `matmul_custom` | `(512,512,512)` | 0.1418 | Large shape did not benefit from staged routing in the repeat. |
| `gemm_staged` | `(128,64,128)` | 0.1071 | Existing 2-simdgroup staged kernel remains available explicitly. |
| `gemm_staged` | `(512,512,512)` | 0.1397 | Similar large-shape throughput to `matmul_custom`. |
| `flux_gelu` | `(512,512,512)` | 0.1441 | Fused epilogue remains single-kernel and in-register. |
| `flux_gate` | `(512,512,512)` | 0.1442 | Residual load makes it slightly more bandwidth-sensitive. |

### Experiments

| ID | Hypothesis | Change | Result | Decision |
|---|---|---|---|---|
| E1 | Since `gemm_staged` often beats `matmul_custom`, route `matmul_custom` through the staged pipeline with two simdgroups. | Temporarily changed `launch_matmul_custom` to use `gemm_staged_*` and dispatch 64 threads. | Correctness passed. Performance was shape-sensitive: mid-size cases improved slightly, but tiny and large shapes regressed in the comprehensive slice. | Rejected and removed. |
| E2 | Flux can reuse A across adjacent output-column tiles like `gemm_staged`, then apply the same epilogues in-register. | Added temporary `flux_gelu_*_staged` and `flux_gate_*_staged` kernels and routed launch helpers to them. | Correctness passed. `flux_gelu` showed only small/noisy gains; `flux_gate` regressed small and medium shapes and tied large. | Rejected and removed. |

### Decision Log

- Keep `matmul_custom`, `gemm_staged`, and Flux launch routing unchanged.
- Keep `gemm_staged` as an explicit kernel rather than silently replacing `matmul_custom`.
- Do not add staged Flux variants until a same-binary dispatcher can shape-gate them and the win is larger than benchmark noise.

### Follow-Ups

- Add a same-binary matmul variant harness if public routing decisions become important; rebuild-to-rebuild noise makes marginal routing changes hard to judge.
- Test larger rectangular LLM shapes beyond the current square-ish preset before revisiting `matmul_custom` routing.
- Revisit Flux only with shape-gated variants or when the epilogue grows enough that arithmetic, not A-load reuse, dominates.

## Row-Wise And Elementwise

Status: investigated pass 1 complete; no `layernorm`, `rms_norm`, `softmax`, `gelu`, `rotary`, or `add_rt` source changes retained.
Priority: P2/P3.

### API And Contract

- Public entry points: `tk.layernorm`, `tk.rms_norm`, `tk.softmax`, `tk.gelu`, `tk.rotary`, and `tk.add_rt`.
- Backends: MLX extension and PyTorch MPS wrapper share the same Metal kernels and launch helpers.
- Dtypes: row-wise normalization/softmax/GELU/RoPE use bf16 I/O with fp32 compute; `add_rt` supports f32/f16/bf16.
- Supported row widths: `layernorm`, `rms_norm`, `softmax`, `gelu` instantiate D in `{256,512,768,1024}`; `rotary` instantiates D in `{64,128}`.
- Correctness tests: `layernorm`, `rms_norm`, `softmax`, `gelu`, `rotary`, and `add_rt` correctness directories.

### References Inspected

| File | Notes / ideas |
|---|---|
| `.reference/llm.metal/dev/cuda/layernorm_forward.cu` | Tests multiple thread/block mappings for row-wise normalization. This motivated testing whether several independent rows per Metal threadgroup improve scheduling. |
| `.reference/llm.metal/dev/cuda/softmax_forward.cu` | Warp-level softmax reductions support the current one-simdgroup-per-row baseline. |
| `.reference/llm.metal/dev/cuda/gelu_forward.cu` | Standalone GELU is simple elementwise work; performance should usually come from fusion rather than a more complex standalone kernel. |
| `ThunderMittens/kernels/*norm`, `softmax`, `gelu`, `rotary`, `add_rt` | Current kernels keep each row/tile entirely in registers and use simdgroup reductions where needed. |

### Baseline And Verification

Baseline harness:

```bash
.venv/bin/python perf/bench_kernels.py --backend mlx --preset comprehensive --kernel layernorm,rms_norm,softmax,gelu,rotary,add_rt --warmup 5 --iters 30 --repeats 5 --output-dir perf/results/rowwise-baseline-20260630 --markdown
```

Final restored-source harness:

```bash
.venv/bin/python perf/bench_kernels.py --backend mlx --preset comprehensive --kernel layernorm,rms_norm,softmax,gelu,rotary,add_rt --warmup 5 --iters 30 --repeats 5 --output-dir perf/results/rowwise-restored-20260630 --markdown
```

Verification:

| Test | Result | Notes |
|---|---|---|
| `../../.venv/bin/python -m pytest layernorm/correctness/test_layernorm.py rms_norm/correctness/test_rms_norm.py softmax/correctness/test_softmax.py gelu/correctness/test_gelu.py rotary/correctness/test_rotary.py add_rt/correctness/test_add.py -q` | 31 passed | Run after final restored-source rebuild. |
| Final MLX harness | 30 ok, 0 skipped, 0 failed | Restored source after rejecting multi-row threadgroups. |
| `.venv/bin/python perf/bench_kernels.py --backend torch --preset quick --kernel layernorm,rms_norm,softmax,gelu,rotary,add_rt --warmup 1 --iters 3 --repeats 2 --output-dir perf/results/rowwise-final-torch-20260630 --markdown` | 14 ok, 0 skipped, 0 failed | PyTorch MPS wrapper check. |

Restored-source timing, MLX, selected median ms:

| Kernel | Shape | Median | Notes |
|---|---|---:|---|
| `layernorm` | `(4,64,512)` | 0.1221 | One simdgroup per row. |
| `rms_norm` | `(1,256,768)` | 0.1486 | One reduction plus weight multiply. |
| `softmax` | `(2,128,1024)` | 0.1691 | Max and sum reductions dominate row work. |
| `gelu` | `(2,128,1024)` | 0.1566 | Standalone elementwise launch remains expensive. |
| `rotary` | `(1,2,128,128)` | 0.1430 | Split-half memory pattern, no trig. |
| `add_rt` | `(64,128)` bf16 | 0.1204 | Smoke/calibration tile kernel. |

### Experiments

| ID | Hypothesis | Change | Result | Decision |
|---|---|---|---|---|
| E1 | Packing four independent rows into one threadgroup can reduce threadgroup scheduling overhead while preserving one simdgroup per row. | Temporarily added `simdgroup_index_in_threadgroup` to `layernorm`, `rms_norm`, `softmax`, and `gelu`; launch helpers dispatched `ceil(M/4)` groups of 128 threads. | Correctness passed. Performance regressed every affected benchmark case: GELU, layernorm, RMSNorm, and softmax slowed by roughly 18-72%. | Rejected and removed. |

### Decision Log

- Keep the one-simdgroup-per-row kernels for normalization, softmax, and GELU.
- Keep `rotary` and `add_rt` unchanged in this pass; both are simple bandwidth/launch-bound kernels in the current preset.
- Treat standalone GELU as a fusion candidate rather than a standalone micro-optimization target.

### Follow-Ups

- Add larger row-count and longer-D row-wise shapes before revisiting scheduling changes; current cases sit close to fixed launch/runtime overhead.
- If optimizing elementwise bandwidth seriously, add a direct contiguous vector kernel variant for `add_rt` and GELU in a same-binary A/B harness.
- Consider graph-level fusion opportunities for GELU, residual add, and normalization epilogues rather than complicating the standalone kernels.

## Linear / State-Space Attention Family

Status: investigated pass 1 complete; no retained source changes.
Priority: P2.

### Scope

Kernels covered: `linear_attn`, `lin_attn_causal`, `lin_attn_decay`, `hedgehog`, `based`, and `mamba2`.

### References Inspected

| File | Notes / ideas |
|---|---|
| `.reference/ThunderKittens/kernels/linear_attention/linear_attention.cu` | Reinforces scan/state formulations for causal linear attention. |
| `.reference/ThunderKittens/kernels/hedgehog/hedgehog.cu` | Feature-map kernels are sensitive to recompute versus materialization. |
| `.reference/ThunderKittens/kernels/based/linear_attn.cu` | Taylor-map shape is small enough that scalar overhead matters. |
| `.reference/ThunderKittens/kernels/mamba2/mamba2.cu` | Mamba/SSD reference motivated testing a recurrent state update rather than materialized chunk loops. |

### Baseline And Verification

| Command | Result | Notes |
|---|---|---|
| `../../.venv/bin/python -m pytest linear_attn/correctness/test_linear_attn.py hedgehog/correctness/test_hedgehog.py lin_attn_causal/correctness/test_lin_attn_causal.py lin_attn_decay/correctness/test_lin_attn_decay.py based/correctness/test_based.py mamba2/correctness/test_mamba2.py -q` | 18 passed | Expected NumPy overflow warnings in decay/based references. |
| `.venv/bin/python perf/bench_kernels.py --backend mlx --preset comprehensive --kernel linear_attn,hedgehog,lin_attn_causal,lin_attn_decay,based,mamba2 --warmup 5 --iters 30 --repeats 5 --output-dir perf/results/linear-family-baseline-20260630 --markdown` | 17 ok | Baseline only. |

Baseline MLX median ms:

| Kernel | Shapes / variants | Medians |
|---|---|---|
| `linear_attn` | N128, N256, N512 | 0.4652, 0.6084, 0.6370 |
| `lin_attn_causal` | N128, N256, N512 | 0.2131, 0.2659, 0.4803 |
| `lin_attn_decay` | N128, N256, N512 | 0.1556, 0.2148, 0.2846 |
| `hedgehog` | N128, N256, N512 | 0.5377, 0.3884, 0.5983 |
| `mamba2` | N128, N256, N512 | 0.1253, 0.1463, 0.2455 |
| `based` | N64, N256 | 0.1492, 0.1931 |

### Experiments

| ID | Hypothesis | Change | Result | Decision |
|---|---|---|---|---|
| E1 | `lin_attn_decay` and `mamba2` can replace O(chunk^2) materialized key-chunk loops with a per-head recurrent state scan, reducing arithmetic. | Temporarily rewrote both kernels to dispatch one simdgroup per `(batch, head)` and scan all chunks, maintaining `D x D` state and handling the current causal chunk separately. | Correctness passed focused pytest but benchmark failed larger shape checks: `lin_attn_decay` N256 max_abs 2.407, `mamba2` N512 max_abs 5.301. Passing cases regressed: `lin_attn_decay` N128 0.1556 -> 0.5138, N512 0.2846 -> 0.5818; `mamba2` N128 0.1253 -> 0.2783, N256 0.1463 -> 0.6212. | Rejected and removed. |

### Decision Log

- Keep the current linear-family kernels unchanged.
- The recurrent scan form is algorithmically attractive, but on current Metal it loses too much chunk-level parallelism and introduces numeric divergence on larger shapes.
- Future work needs same-binary variants and larger/head-rich shapes before attempting state-space restructuring again.

## Complex GEMM And FFTConv

Status: `cmplx_matmul` optimized pass 1 complete; `fftconv` investigated pass 1 complete with no retained source change.
Priority: P3.

### API And Contract

- `cmplx_matmul`: `A (2,N,K)`, `B (2,K,M)`, output `(2,N,M)`, dtype f32 or bf16, requires `N%32==0`, `M%32==0`, `K%16==0`.
- `fftconv`: float32 Monarch FFT convolution, `S in {16,32}`, one simdgroup per `(batch, head)`.

### References Inspected

| File | Notes / ideas |
|---|---|
| `.reference/ThunderKittens/kernels/fftconv/fftconv_non_pc.cu` | One-warp Monarch path repeatedly copies/accumulates complex tiles; useful comparison for current Metal implementation. |
| `.reference/ThunderKittens/kernels/fftconv/fftconv_pc.cu` | Larger persistent CUDA path keeps common FFT/twiddle/filter tiles in scratch and handles more batch/head work per CTA; not directly portable to the current one-simdgroup Metal kernel. |
| `.reference/ThunderKittens/include/ops/group/mma/warp.cuh` | Complex MMA accumulates four real MMAs; motivated removing accumulator self-copy for in-place accumulation. |

### Baseline And Verification

| Command | Result | Notes |
|---|---|---|
| `../../.venv/bin/python -m pytest cmplx_matmul/correctness/test_cmplx_matmul.py fftconv/correctness/test_fftconv.py -q` | 10 passed | Focused correctness before and after retained change. |
| `.venv/bin/python perf/bench_kernels.py --backend mlx --preset comprehensive --kernel cmplx_matmul,fftconv --warmup 5 --iters 30 --repeats 5 --output-dir perf/results/complex-fft-baseline-20260630 --markdown` | 6 ok | Baseline. |
| `.venv/bin/python perf/bench_kernels.py --backend mlx --preset comprehensive --kernel cmplx_matmul,fftconv --warmup 5 --iters 30 --repeats 5 --output-dir perf/results/complex-fft-final-solo-20260630 --markdown` | 6 ok | Final solo MLX timing. |
| `.venv/bin/python perf/bench_kernels.py --backend torch --preset quick --kernel cmplx_matmul,fftconv --warmup 1 --iters 3 --repeats 2 --output-dir perf/results/complex-fft-final-torch-20260630 --markdown` | 4 ok | PyTorch MPS wrapper check. |

Final solo MLX timing:

| Kernel | Shape | Baseline ms | Final ms | Decision |
|---|---:|---:|---:|---|
| `cmplx_matmul` | 32x16x32 | 0.2014 | 0.2028 | Small-K route uses original kernel; difference is noise. |
| `cmplx_matmul` | 128x64x128 | 0.1943 | 0.1884 | Small-K route uses original kernel; difference is noise/favorable. |
| `cmplx_matmul` | 256x128x256 | 0.2392 | 0.2337 | Small-K route uses original kernel; difference is noise/favorable. |
| `cmplx_matmul` | 512x512x512 | 0.4872 | 0.2986 | Keep in-place complex MMA for K>=512. |
| `fftconv` | S16 `(1,1,16)` | 0.1460 | 0.1352 | Source unchanged; timing noise/favorable. |
| `fftconv` | S32 `(2,2,32)` | 0.2267 | 0.2427 | Source unchanged; timing noise/unfavorable. |

### Experiments

| ID | Hypothesis | Change | Result | Decision |
|---|---|---|---|---|
| E1 | Large-K complex GEMM wastes register moves by passing `d` as both output and accumulator to `complex_mma_AB`, which copies accumulator tiles before every K-block. | Added `complex_mma_AB_inplace` and a large-K `cmplx_matmul_*` path; instantiated original path as `*_small`; launch uses original for `K < 512`, in-place for `K >= 512`. | Correctness passed. Large 512^3 bf16 median improved from 0.4872 ms to 0.2986 ms; smaller shapes remain routed to original. | Kept. |
| E2 | FFTConv can reduce register tile copy instructions by ping-ponging `x`/`y` between stages instead of copying `y` back to `x`. | Temporarily rewrote Monarch pipeline to alternate destination/source tiles. | Correctness passed, but performance regressed badly: S16 0.1460 -> 0.2117 ms, S32 0.2267 -> 0.4637 ms. Likely increased liveness/register pressure. | Rejected and removed. |

### Follow-Ups

- Add larger FFTConv shapes closer to the ThunderKittens 1024/4096 sequence references before revisiting persistent/shared-state designs.
- Add same-binary large-K `cmplx_matmul` variants if more shape gates are needed beyond `K >= 512`.
- Consider moving in-place complex MMA into the shared substrate only after another kernel benefits from the same helper.

## Per-Kernel Notebook Template

Copy this section for each active investigation.

````markdown
## <kernel>

Status:
Owner:
Branch:
Priority:

### API And Contract

- Public entry points:
- Backends:
- Dtypes:
- Supported shapes:
- Correctness tests:
- Known constraints:

### References Inspected

| File | Notes / ideas |
|---|---|
|  |  |

### Baseline

Command:

```bash

```

Correctness:

| Test | Result | Notes |
|---|---|---|
|  |  |  |

Performance:

| Shape / format | Baseline | Current TM | Metric | Delta | Notes |
|---|---:|---:|---:|---:|---|
|  |  |  |  |  |  |

### Bottleneck Hypothesis

- Bytes moved:
- FLOPs / ops:
- Arithmetic intensity:
- Achieved GB/s:
- Achieved GFLOP/s:
- Suspected bottleneck:
- Evidence:

### Experiments

| ID | Hypothesis | Change | Shapes | Result | Decision |
|---|---|---|---|---|---|
| E0 | Establish baseline | No code change |  |  |  |

### Decision Log

- Pending.

### Follow-Ups

- Pending.
````

## Decision Log

| Date | Kernel | Decision | Evidence | Follow-up |
|---|---|---|---|---|
| 2026-06-30 | qgemv | Keep q8_0/q4_0 packed-dot large-K kernels plus small-K generic dispatch. | Correctness passed; q8_0 focused A/B up to 1.87x faster, q4_0 positive but variable. | Add saved focused decode benchmark harness before the next qgemv pass. |
| 2026-06-30 | qgemm | Keep existing direct-to-fragment default; reject q8/q4 fragment-fill and X-shared experiments. | Correctness passed; fragment-fill A/B was mixed, X-shared regressed q4_0/q4_K/q8_0 clearly. | Add same-binary variant selection before another marginal qgemm pass. |
| 2026-06-30 | qflux | Keep split `add_col` then `gelu` epilogue; reject temporary fused register epilogue. | Correctness passed, but selective and focused repeats did not reproduce the initial fp8/bitnet wins. | Revisit only with same-binary A/B or after qgemm dequant tuning. |
| 2026-06-30 | qgemv_int | Keep W2A8 lane-scaled single-reduction path; reject W8A8 unroll. | Correctness and PyTorch checks passed; W2A8 default improved `0.1597 -> 0.1258` ms and large K improved, while W8A8 unroll was mixed. | Add saved focused decode shapes and same-binary W2A8 variants before shape gating. |
| 2026-06-30 | qgemm_int | Keep W2A8 lane-scaled single-reduction path. | Correctness and PyTorch checks passed; W2A8 default prefill improved `0.8088 -> 0.7643` ms. | Revisit only with larger exact-int use cases and same-binary variants. |
| 2026-06-30 | attn_q | Keep existing kernels; retain harness guard for unsupported q4 multiwarp and add q4 causal correctness coverage. | Barrier experiments were not reproducible; q4 multiwarp was correct but slower than q4 single-warp on `(1,4,128,64)` (`~0.319` vs `~0.167` ms). | Add longer-context attention benchmarks and same-shape single-vs-multiwarp specs. |
| 2026-06-30 | dense attention | Keep `attn_fwd`, `attn_causal`, and `attn_multiwarp` unchanged. | Correctness passed; multiwarp dispatch and fused-register-map experiments were mixed or regressed on repeat. | Add longer-context and same-binary attention variant benchmarks. |
| 2026-06-30 | attn_bwd | Keep K row-to-col layout derivation in dQ and dKV dead-load cleanup. | Correctness and PyTorch checks passed; MLX medians improved from `0.266/0.260/0.338/0.333` ms to `0.241/0.157/0.164/0.166` ms. | Add larger backward shapes and separate dQ/dKV timing before tuning launch geometry. |
| 2026-06-30 | dense GEMM/Flux | Keep `matmul_custom`, `gemm_staged`, `flux_gelu`, and `flux_gate` unchanged. | Correctness and PyTorch checks passed; staged `matmul_custom` routing and staged Flux variants were mixed, with Flux gate regressions and no large-shape win. | Add same-binary variant selection and larger rectangular GEMM shapes before another routing pass. |
| 2026-06-30 | row-wise/elementwise | Keep `layernorm`, `rms_norm`, `softmax`, `gelu`, `rotary`, and `add_rt` unchanged. | Correctness and PyTorch checks passed; four-rows-per-threadgroup regressed every tested norm/softmax/GELU case. | Add larger row-wise shapes and same-binary vector variants before revisiting. |
| 2026-06-30 | linear/state-space attention | Keep `linear_attn`, `lin_attn_causal`, `lin_attn_decay`, `hedgehog`, `based`, and `mamba2` unchanged. | Correctness passed after rollback; recurrence scan experiment for `lin_attn_decay`/`mamba2` regressed every passing measured shape and failed larger-shape correctness. | Revisit with same-binary variants and larger/head-rich shapes before restructuring scan/state kernels. |
| 2026-06-30 | complex/FFT | Keep large-K in-place `cmplx_matmul` dispatch; keep `fftconv` unchanged. | Correctness and PyTorch checks passed; `cmplx_matmul` 512^3 improved `0.4872 -> 0.2986` ms, while FFTConv ping-pong rewrite regressed S16/S32. | Add larger FFTConv sequence shapes and consider promoting in-place complex MMA only after another kernel needs it. |
