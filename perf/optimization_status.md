# ThunderMittens — performance status

Running notebook for the per-kernel optimization loop described in `perf/perf.md`.
Numbers are throughput-style median per-call ms from `perf/bench_kernels.py`
(adaptive batched timing, ≥2 ms per timed sample), Apple M4 Max 40-core
(~546 GB/s DRAM), MLX backend unless noted. Baseline run:
`perf/results/2026-07-01/172040-mlx-quick/` at `d902519`.

**Timing-methodology note (2026-07-01):** the harness was rewritten. Earlier
numbers in this file used one submit+sync per call, which adds a ~0.15–0.25 ms
latency floor and swamped small kernels; conclusions drawn only from per-call-sync
timing (notably "staged paged attention is slower") did NOT survive the fix —
see the serving section.

## Baseline classification (2026-07-01, quick preset)

Speedup = best-baseline ms / tk ms (>1 means tk wins).

### Already ahead of the framework — protect, don't churn
| kernel | shape | tk ms | vs | speedup |
|---|---|---:|---|---:|
| layernorm | 4096×1024 | 0.066 | mx.fast.layer_norm | 1.97 |
| rms_norm | 4096×1024 | 0.035 | mx.fast.rms_norm | 1.93 |
| softmax | 4096×1024 | 0.046 | mx.softmax | 1.57 |
| gelu | 16384×1024 | 0.161 | mx.nn.gelu_approx | 2.62 |
| add_norm (fused) | 4096×1024 | 0.093 | add + mx.fast.rms_norm | 1.97 |
| attn_causal | 1×8×2048×128 | 0.795 | sdpa+mask | 3.87 |
| attn_fwd D=128 | 1×8×2048×128 | 1.499 | sdpa | 1.11 |
| attn_bwd | 1×8×1024×64 | 0.741 | mx.vjp naive | 2.46 |
| lin_attn_causal | 2×8×4096×64 | 2.050 | masked naive | 6.36 |
| matmul_custom | 2048³ bf16 | 1.233 | mx.matmul | 1.01 |
| flux gate | 2048³ | 1.227 | matmul+epilogue | 1.08 |
| cmplx_matmul | 1024³ | 0.663 | 4×mx.matmul | 1.22 |
| moe grouped | E8 H2048 | 1.401 | per-expert loop | 1.45 |
| paged_attention_v2 | 8×32 ctx2048 | 0.382 | v1 | 4.58 |

### Gaps — the optimization queue (worst first, weighted by real-model impact)
| # | kernel | shape | tk ms | vs | speedup | first hypothesis |
|---|---|---|---:|---|---:|---|
| 1 | qgemm (staged route) | q4_0 M=512 | 11.88 | tk.qgemm_direct | 0.11 | routing bug: staged path collapses at large M; direct is 9× faster |
| 2 | qgemv generic fmts | q4_K 4096² | 0.199 | fp16 matmul | 0.24 | per-element div/mod + branchy dequant in the generic template (fast paths exist only for q8_0/q4_0); W-GB/s 44–127 vs 200–430 for fast paths |
| 3 | attn_q | q4_0 1×8×1024×128 | 1.225 | attn_fwd on dequant K/V | 0.32 | in-kernel dequant dominates; multiwarp already 1.2–1.65× better |
| 4 | linear_attn (non-causal) | 2×8×4096×64 | 2.197 | Q@(KᵀV) via mx.matmul | 0.06 | grid is B·H simdgroups — no sequence parallelism; hedgehog same (0.38) |
| 5 | rotary | 1×32×2048×128 | 0.187 | mx.fast.rope | 0.44 | mx computes trig in-kernel (no cos/sin table reads) + vectorized; ours reads tables scalar |
| 6 | v2_fp8 paged decode | 8×32 ctx2048 | 0.859 | v2 bf16 cache | 0.44 | dequant-on-read costs 2.3× despite half the bytes |
| 7 | glu (all modes) | 16384×4096 | 3.925 | silu(x)*gate composed | 0.60 | scalar loads; 103 GB/s vs 546 peak |
| 8 | add_rt | 4096×1024 | 0.155 | mx add | 0.34 | 8×8 register-tile machinery for a pure elementwise op |
| 9 | hadamard D≤128 | 16384×128 | 0.150 | matmul vs H | 0.35 | geometry: too little work per threadgroup at small D (D=512 wins 2.1×) |
| 10 | qgemv q8_0/q4_0 | 4096×4096 | 0.087 | mlx_q4/q8 gemv | 0.63 | N=4096 → 4096 single-simdgroup TGs; occupancy. At N=11008 q4_0 BEATS mlx 1.73× |
| 11 | qgemv_int w8a8 | 4096² | 0.147 | tk.qgemv q8_0 | 0.26 | same small-N geometry issue |
| 12 | attn_fwd D=64 | 1×8×1024×64 | 0.238 | sdpa | 0.77 | D=64 tile geometry |
| 13 | quantize_per_tensor_fp8 | 16384×1024 | 1.529 | (per_token: 0.274) | — | 33 GB/s; global atomic-max pass dominates |
| 14 | flux gelu @1024³ | 1024³ | 0.348 | matmul+gelu | 0.65 | small-shape only (2048³ is 1.09) — low priority |

### Serving decode re-measurement (supersedes the 2026-06 table)
With pipelined timing at 8×32×2048×128: v1 1.837 ms, **staged 0.981 ms (1.80×
FASTER than v1)**, v2(p256) 0.382 ms. The earlier "staged 1.7× slower" was an
artifact of per-call-sync timing. v2 remains the default and is still the right
choice. TODO: sweep partition_size per context; re-check staged under real
decode loops (one call per step, no pipelining) before changing any default.

## Per-kernel log

### qgemv — status: LANDED (2026-07-01, three stacked wins)
Three changes, all format-generic, validated by the full regression
(862 MLX + 601 parity/MPS green):

1. **E1 — branchless float-code decoders** (`dequant.metal`): `tk_e4m3/e5m2/
   e2m1/e3m2/e2m3_decode` now shift the code into the fp16 field positions and
   rescale by a power-of-two constant instead of branch + `metal::exp2`.
   Verified exact over every finite code in numpy before landing. Subnormal
   codes (e==0) take a select computed in normal-half arithmetic — REQUIRED
   because tk_torch's offline `xcrun metal -O2` build flushes subnormal
   arithmetic (fast-math FTZ) while MLX's metallib does not; the pure-bit-trick
   version silently broke ONLY torch-side nvfp4 (caught by parity tests).
2. **E2 — block-major qgemv walk** (`qgemv.metal` template): each lane owns an
   8-col contiguous span inside a block (no per-element div/mod, `half4` X
   loads, scale reads CSE across the span).
3. **E3 — `tk_dequant8<FMT>` span decoders** (`dequant.metal`): per-format
   specializations unpack the block/sub-block scales ONCE per 8-col span
   (q4_K/q5_K/q6_K/q2_K/q3_K/iq4_xs/iq4_nl/kU4B8/kU4/hqq/nvfp4/mxfp4/q4_1/
   q5_0/q5_1). Every 8-span is provably inside one sub-block/nibble-half for
   all these layouts.

Results (ms, N=11008 K=4096 M=1; baseline = fp16 `mx.matmul` on the same run):
| format | before | after | vs fp16 matmul | W-GB/s |
|---|---:|---:|---:|---:|
| q4_K | 0.391 | 0.090 | 2.4× faster | ~281 |
| q5_K | 0.568* | 0.115 | 2.1× | ~271 |
| q6_K | 0.411 | 0.116 | 1.9× | ~250 |
| iq4_xs | 0.267* | 0.083 | 2.6× | ~281 |
| kU4B8 | 0.279* | 0.071 | 3.1× | ~322 |
| hqq | 0.220 | 0.070 | 3.1× | ~364 |
| nvfp4 | 0.321 | 0.100 | 2.3× | ~254 |
| mxfp4 | 0.288 | 0.098 | 2.5× | ~245 |
| fp8_e4m3 | 0.262 | 0.103 | 2.0× | ~466 |
| bitnet | 0.188 | 0.113 | 2.1× | ~125 |
(*= measured mid-way, post-E2 pre-E3.) Every format now beats the fp16 GEMV
2–3×; before, most LOST to it. At N=4096² all formats sit at 0.029–0.055 ms.

- Hand-written q8_0/q4_0 fast paths: still ~15–25% ahead of the improved
  template (measured by routing q8_0/q4_0 through the template) — KEPT.
- Rejected/deferred: multi-row-per-simdgroup geometry (llama.cpp N_R0-style X
  register reuse) — remaining headroom looks ≤10–30% and formats are now at
  245–466 W-GB/s; revisit if decode becomes the bottleneck again.
- Side effect of E1 on serving: **paged_attention_v2_fp8 0.859 → 0.487 ms**
  (dequant-on-read penalty 2.3× → 1.25× vs the bf16 cache) — queue #6 done.
- attn_q did NOT move (its cost is the per-element dequant_into_shared/register
  structure, not decode ALU) — see attention section.
- Also fixed: tk_torch metallib staleness check ignored `include/` substrate
  changes (stale-metallib false-greens); now walks include/*.metal mtimes.

### qgemm — status: baselined (no action needed)
- The "M=512 q4_0 is 9× slower than qgemm_direct" queue item was a MEASUREMENT
  ARTIFACT: `tk.qgemm` and `tk.qgemm_direct` construct the identical primitive
  (both direct_=true) and measure at parity once the harness warms the GPU
  clocks properly. Two harness fixes landed: a 1 s pre-run clock ramp, and
  time-based (≥50 ms) per-thunk warmup — before these, whichever thunk was
  timed first in a case read 1.5–5× slow.
- Clean numbers: q4_0/q8_0 at parity with fp16 `mx.matmul` for M∈{32,128,512}
  (compute-bound; the win is memory footprint). fp8_e4m3 M≥128 ~13% behind
  q8_0 even after E1 — remaining gap is in the fragment-path decode; candidate:
  span-decode (tk_dequant8) inside dequant_into_register/shared, shared with
  the attn_q work.

### Elementwise/row family — status: LANDED (2026-07-01)
- layernorm/rms_norm/softmax/gelu/add_norm already beat MLX fast ops (1.5–2.6×)
  — untouched.
- **rotary**: one-simdgroup-per-row + scalar substrate loads → flat one thread
  per 4 rotation pairs (bf16_4 vectors, fp32 math; ABI +M at buffer 5).
  0.187→0.078 ms at (1,32,2048,128); 0.44× → ~0.95× of mx.fast.rope (the
  remaining few % is the cos/sin table reads mx avoids by computing trig
  in-kernel — judged not worth matching).
- **glu** (6 modes): scalar → vec4 + scalar tail. 3.93→1.11 ms at 16384×4096
  (362 GB/s); 0.60× → 2.3× vs composed silu·gate.
- **add_rt**: 8×8-register-tile demo layout → flat 8 elems/thread (16-byte
  transactions). 0.34× → ~1.05× of mx add (bf16), parity f32. The rt smoke-test
  role moved to the Xcode primitive tests.
- **hadamard**: D-thread threadgroup with log2(D) barriers → one simdgroup/row,
  in-register + simd_shuffle_xor butterflies, zero barriers, zero threadgroup
  memory. D=128: 0.150→0.037 ms (1.4× vs matmul-H); D=512: 0.298→0.061 ms
  (554 GB/s ≈ roofline, 10× vs matmul-H).
- **quantize_per_tensor_fp8** (quant_rt): absmax pass now 16 elems/thread
  (16× fewer contended atomics) + vec4 encode. 1.53→0.35 ms at 16384×1024
  (4.3×); per_token 0.27→0.21 (vec4 main loops). Encoder function untouched —
  codes stay bit-identical across backends.
- torch-MPS note: the same kernels win on MPS too (glu 1.38× vs torch silu·mul,
  hadamard D=512 10×), but SMALL kernels carry ~0.05–0.1 ms extra per-call
  overhead from the tk_torch dispatch glue (add_rt bf16 0.42× of torch's add
  there despite parity on MLX) — host-glue follow-up, not a kernel issue.

### Attention — status: measured & recorded (2026-07-01)
- With clean (clock-warmed) timing: fwd D=128 **1.24× ahead** of sdpa, causal
  **3.8–5.8× ahead**, bwd 2.1–2.5× ahead of the mx.vjp naive. fwd D=64 is 0.91×
  (the earlier 0.77× was first-timed-thunk bias); remaining hypothesis —
  D=64-specific sequence-block tuning — deferred as ≤10%.
- multiwarp stays ≈0.8–0.93× of single-warp fwd → keep non-default (standing
  conclusion re-confirmed).
- **attn_q**: V now staged through shared memory like K (span dequant), K/V
  staging uses the span-decoding dequant_into_shared. fp8 single-warp 1.61→1.11
  ms, multiwarp 0.98→0.82–0.94 at (1,8,1024,128). Structural finding: the
  remaining ~2.5× gap vs attend-on-dequantized-KV is the 8-row-KV-tile shared
  round-trip (tiny tiles, 2 barriers per 8 rows), NOT dequant ALU. Options
  deferred: 32-row KV tiles (rectangular causal masking complexity) or an
  op-level dequant-to-scratch route (~0.45 ms estimated, at 2× memory).
  Recommendation recorded: prefer multiwarp=True for attn_q today.

### Linear-attention family — status: LANDED (routing) (2026-07-01)
- Causal/scan kernels (lin_attn_causal, mamba2, based, lin_attn_decay) healthy;
  lin_attn_causal beats the masked-naive baseline 5–6×.
- **Non-causal linear_attn and hedgehog now ROUTE to framework composition by
  default** (`use_kernel=True` keeps the ported kernels; parity tests pin it).
  Rationale: the kernels run one simdgroup per (batch,head) — 16 simdgroups
  total at (2,8,·,·) — while the composition uses the whole GPU per GEMM.
  linear_attn 2.20→0.142 ms (15.5×), hedgehog 0.64→0.25 ms (2.5×). A
  sequence-parallel split-KV kernel was considered and rejected: it cannot beat
  two mx.matmul calls and needs cross-threadgroup reduction scratch.
- lin_attn_decay wrapper now builds the decay ramp on-device (was numpy per
  call); neutral in pipelined benchmarks, removes the host stall in real use.

### Complex — status: healthy (no action)
- cmplx_matmul ≥ 4-matmul composition (1.0–1.22×); fftconv ≈ mx.fft path.

### GEMM (matmul_custom / gemm_staged / flux) — status: recorded (no action)
- With clean timing, flux gelu/gate beat the composed baseline at ALL measured
  shapes (1.08–1.16×) — the earlier 0.65× at 1024³ was first-thunk bias.
- matmul_custom @1024³ is 0.58× of mx.matmul while gemm_staged @1024³ is at
  parity (0.177 vs 0.176 ms); both ≈ parity at ≥2048³. Finding recorded:
  2-simdgroup staged wins at the smaller size. Not routed — these are TK-parity
  /calibration kernels; mx.matmul is the practical dense GEMM.

### Serving — status: LANDED items (2026-07-01)
- v2_fp8 dequant-on-read: fixed by the branchless decoders — 0.859→0.487 ms
  (penalty vs bf16 cache 2.3× → 1.25×, with half the cache bytes).
- quantize_per_tensor_fp8: 4.3× (see elementwise section).
- Still open: partition_size sweep per context; staged-vs-v1 default re-check
  under non-pipelined single-call decode (the pipelined harness reverses the
  old conclusion; a real decode loop sits between the two regimes).

## Decision log
- 2026-07-01: harness rewritten (schema v1, all families, batched timing);
  perf.md updated to match reality (reference mirrors, device context, timing
  methodology, serving section).
- 2026-07-01: TWO timing-bias fixes (1 s clock pre-ramp; ≥50 ms time-based
  per-thunk warmup). Queue items #1 (qgemm M=512 "routing bug"), most of #12
  (attn D=64) and #14 (flux gelu small) were artifacts of the biased harness —
  always re-verify a gap on the fixed harness before optimizing.
- 2026-07-01: qgemv E1/E2/E3 landed (branchless decoders + block-major walk +
  span dequant) — every quant format now beats the fp16 GEMV; commit a38e8a8.
- 2026-07-01: rotary/glu/add_rt/hadamard flat-vectorized geometry; attn_q V
  staging; commit f16b9d5.
- 2026-07-01: linear_attn/hedgehog routed to framework composition
  (use_kernel escape hatch); quant_rt vectorized + atomic-thinning; lin_attn_decay
  device-side ramp.
- Rejected this pass: multi-row qgemv geometry (≤10–30% left, formats at
  245–466 W-GB/s), attn_q 32-row KV tiles (complexity vs niche kernel),
  matmul_custom small-shape routing (calibration kernel), fp16 LUT decoders
  (bit-tricks already exact + cheap).
