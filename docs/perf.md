Performance concerns & where to focus
Tier 1 — verify the core claim first
The entire premise is that quantization is the high-value Apple win because decode is weight-bandwidth-bound. This is currently asserted, not measured. Everything else is secondary to closing this loop.

Benchmark qgemv vs. the naive path. Compare qgemv(Wq, x) against dequantize(Wq) @ x via mx.matmul, and against a pure fp16 mx.matmul. Report tokens/sec or GB/s, not just "it runs." If qgemv doesn't beat dequant-then-matmul, the decode path has no reason to exist.
Compute the bandwidth ceiling. For each decode shape, weight bytes moved / measured time = effective GB/s. Compare to the device's peak unified-memory bandwidth. A memory-bound kernel should land near peak; if you're at 30% of peak, there's a bug or a stall, not a win.
Confirm the 4–8× weight shrink translates to wall-clock. The byte reduction is arithmetic; the speedup is empirical. Show q4_0 decode is actually ~Nx faster than fp16 decode, not just Nx smaller on disk.

Tier 2 — known structural risks in the kernels
These are places where "modeled on llama.cpp" didn't fully transfer.

Threadgroup tile layout / bank conflicts. dequant_into_shared writes natural row-major (dst[int2(row, tcol)]). llama.cpp deliberately lays dequantized halves out as contiguous 8×8 blocks so simdgroup_load is conflict-free. Verify whether the row-major write causes bank conflicts on simdgroup_load; this is invisible to correctness tests and a common throughput leak.
qgemv reduction efficiency. The decode kernel is the bandwidth-critical one and the least likely to be fast by default. Check: are loads coalesced across the 32 lanes? Is one-simdgroup-per-output-row enough occupancy, or are you leaving the GPU idle? llama.cpp's mul_mv has had heavy Apple-specific tuning for exactly these reasons.
Per-element dequant cost in the inner loop. Each weight goes through a dequant(base, col) call with branches (e.g. q4_K's col/64, pos<32, get_scale_min_k4). Branchy per-element dequant inside the hot loop can dominate. Check whether the compiler is hoisting/vectorizing or whether you're paying it per weight.

Tier 3 — manage expectations on the prefill (qgemm) path

Don't expect qgemm to beat mx.matmul for batched/prefill. At batch ≥ 8 you're often compute-bound, where dequant-to-half + MMA has little headroom over just dequantizing and calling the vendor matmul. Your own earlier gemm_staged finding showed Apple's lack of async copy means clever staging doesn't beat the simple path — the same logic applies here. Benchmark it, but frame the qgemm win as memory footprint, not throughput.
The double threadgroup_barrier per K-step. qgemm barriers before the MMA and again before sW is overwritten. Confirm that's necessary and not serializing more than needed.

Tier 4 — measurement hygiene

Use proper GPU timing, not wall-clock around an async dispatch. Make sure you're calling mx.eval / torch.mps.synchronize() correctly and warming up before timing (you already do this in time_perf.py — extend that pattern, don't hand-roll new timing).
Test realistic shapes. Benchmark the shapes from the BitNet reference (e.g. N=3840 K=2560, N=13824 K=2560) and real LLM matmul dims, not just 128×128. Small square shapes hide both bandwidth and occupancy effects.
Separate dequant cost from matmul cost where possible, so you know which half to optimize if a kernel is slow.

The one-line version
Right now correctness is proven and performance is inferred. The whole point of weight-only quant is throughput, so the single highest-value next step is: extend time_perf.py to benchmark qgemv and qgemm against the dequant-then-mx.matmul baseline on real shapes, and report effective GB/s vs. peak. That one table converts the whole effort from "impressive-looking" to "verified."You said: and what experiments you would perform to detect slownessand what experiments you would perform to detect slowness1:01 PMClaude responded: Here's the experiment list, written so it slots into the same file.Here's the experiment list, written so it slots into the same file.
Experiments to detect slowness
Experiment 0 — establish the baselines (do this first)
Everything else is relative to these. For each test shape, measure and record:

mx.matmul of fp16 W @ x (the "if I didn't quantize at all" number)
dequantize(Wq) to fp16, then mx.matmul (the "naive quant" path your kernel must beat)
your qgemv / qgemm

If your kernel doesn't beat path #2, it has no reason to exist. If it doesn't get reasonably close to path #1's bandwidth on decode, it's not exploiting the quant. These two comparisons frame every result below.
Experiment 1 — roofline / bandwidth efficiency (the decisive one)
For each decode (qgemv) shape, compute effective bandwidth = weight bytes moved / measured time, and divide by the device's peak unified-memory bandwidth (look it up for the specific chip).

Near peak (>70%): memory-bound and healthy — the win is real.
Far below (<40%): something stalls — dequant overhead, poor coalescing, or low occupancy. This single ratio tells you whether it's slow before you know why.

Do the same arithmetic-intensity check on qgemm to confirm whether it's compute- or memory-bound at each batch size.
Experiment 2 — format sweep at fixed shape
Run all 11 formats on one fixed large decode shape. Plot time vs. bits-per-weight.

Expected: time scales roughly with weight bytes (q8_0 slowest, the 4-bit/ternary fastest) — that's the memory-bound signature you want to see.
Smell: if q4_K or the codebook-ish formats are anomalously slow despite fewer bytes, their per-element dequant (branches, get_scale_min_k4) is dominating, not bandwidth. This experiment isolates dequant-compute cost from memory cost essentially for free.

Experiment 3 — batch-size sweep (find the qgemv→qgemm crossover)
Fix N, K; sweep M = 1, 2, 4, 8, 16, 32, 64. Time both qgemv and qgemm at each (and the baselines).

Confirms your M==1→qgemv routing is actually the right threshold (ggml uses 8 — is yours?).
Reveals where qgemm stops being a win over mx.matmul. The shape of this curve tells you whether prefill quant is buying throughput or only footprint.

Experiment 4 — bank-conflict probe on dequant_into_shared
You can't see conflicts in correctness tests, so provoke them:

A/B the tile layout: time the current row-major shared write vs. a variant that pads the tile (+1 column) or writes contiguous 8×8 blocks like llama.cpp. If padding/reordering measurably speeds it up, you had conflicts.
Cheap proxy: if available, read GPU counters in Instruments (Metal System Trace) for threadgroup-memory stalls. If not, the A/B timing is the falsifiable test.

Experiment 5 — isolate dequant cost from MMA cost
Build a throwaway kernel variant that loads the packed weights, dequantizes into shared, then stores the shared tile instead of doing the MMA (or zeros the activation so the MMA is trivial). Time it against the full kernel.

The delta is the MMA's share; the remainder is dequant + load.
If "dequant only" is already near the full kernel's time, the dequant path is your bottleneck and Tier-2 work (branchy decode, coalescing) is where to focus — not the MMA.

Experiment 6 — occupancy / launch-geometry sweep on qgemv
qgemv launches one simdgroup per output row. Test whether that saturates the GPU:

Vary threadgroup size / rows-per-threadgroup and re-time.
If a different geometry is faster, the default is leaving lanes idle (low occupancy), which on a memory-bound kernel shows up as bandwidth well below peak (ties back to Experiment 1).

Experiment 7 — real-shape validation
Re-run Experiments 1–3 on the actual LLM dims from the BitNet reference (N=3840 K=2560, N=13824 K=2560, N=2560 K=6912, etc.) and a couple of common Llama matmul shapes. Small square 128×128 tiles hide both occupancy and bandwidth effects; the verdict must come from shapes you'd actually run.
Experiment 8 — variance / timing hygiene
For each measurement: warm up, run many iterations, report median + spread (not a single sample), and confirm you're synchronizing (mx.eval / torch.mps.synchronize()) so you're timing the GPU, not the dispatch. Extend the existing time_perf.py harness rather than hand-rolling — it already does warmup and clears the cache. A result with no variance bar is not yet a measurement.
Priority order
If you only do three: Experiment 1 (are you memory-bound and near peak?), Experiment 0+3 (do you beat the naive path, and where?), and Experiment 5 (is dequant or MMA the bottleneck?). Those three answer "is it slow, where, and why" — the rest localize the fix.