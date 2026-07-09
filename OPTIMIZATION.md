# Optimization Log — int8-transformer-kernels

The full iteration journey for the INT8 attention / MLP / decode kernels: the
evidence (before/after `ncu` numbers, validation results, negative results)
behind the distilled, still-actionable conclusions in
[`CLAUDE.md`](CLAUDE.md) §5–§6.

- **Graded MLP shape:** b=8, s=512, d_model=1024, d_ff=4096 (sweep grid).
- **Graded attention shape:** b=8, head_dim=64, seq 512–4096.
- **Decode shapes:** serving-scale B∈{32,64,128}, seq up to 32768.
- Latency = median of ≥5 runs; run-to-run noise ~2%, so |Δ| < 2% is noise.
- All five accuracy gates (`validate_int8.py`) must pass after every change.

---

## Summary

| # | Date | Kernel | Change | Result |
|---|---|---|---|---|
| 1 | 2026-06-12 | attn (prefill) | compile-time `HEAD_DIM` + fragment-load hoisting | **WIN** 1.6–2.2× |
| 2 | 2026-06-12 | mlp | dynamic per-token quantization | **ACCURACY** 4.6× max-err, ~5% latency cost |
| 3 | 2026-06-13 | attn | fuse softmax-weight pack into P@V loop | neutral (+1.5%) |
| 4 | 2026-06-13 | attn | native `mma.sync m16n8k32` QK^T | neutral (−1.4%) |
| 5 | 2026-06-13 | attn | fold V scale into P@V weights | neutral (−0.9%) |
| 6 | 2026-06-13 | attn | software-pipeline P@V `ldmatrix` | neutral (±0%) |
| 7 | 2026-06-13 | mlp | conflict-free hand-rolled `mma.sync m16n8k32` | **WIN** −26…−40% |
| 8 | 2026-06-13 | mlp | deepen cp.async K-stage 32→64 | **WIN** −6…−17% |
| 9 | 2026-06-13 | mlp | per-channel/per-token quant + real Gate 5 | **ACCURACY** fixes +64% ppl → −0.07% |
| 10 | 2026-06-13 | mlp | denser-mma reschedule | **NEGATIVE** +1.5…+3% (reverted) |
| 11 | 2026-06-13 | mlp | prepacked transpose-once forward | **WIN** −9…−22% @ s=512 |
| 12 | 2026-06-13 | mlp | finer 4×4 warp tiling | **NEGATIVE** +15…+28% (OFF flag) |
| 13 | 2026-06-13 | mlp | fuse `quantize_rows` into GEMM2 load | **NEGATIVE** +45…+63% (OFF flag) |
| 14 | 2026-06-13 | decode | split-KV flash-decoding (INT8 KV cache) | **WIN** D=128 1.23× vs SDPA |
| 15 | 2026-06-13 | decode | `dp4a` lane-per-position | **WIN** D=64 1.24–1.56× vs SDPA |
| 16 | 2026-06-13 | attn (accuracy) | SageAttention-style K/V smoothing | **NEGATIVE** no output benefit (no code) |
| 17 | 2026-06-13 | full block | first end-to-end block wall-clock | INT8 wins only long-context decode |
| 18 | 2026-06-13 | decode (bench) | vs FlashInfer FP8 (quantized-KV SOTA) | D=64 par-to-win + more accurate; D=128 −1.5× |
| 19 | 2026-06-13 | accuracy | Gate 5 runs the real INT8 MLP kernel in GPT-2 | real ppl +0.134% (< 2% bar) |
| 20 | 2026-07-09 | decode | TARGET_UNITS occupancy sweep (first decode `ncu`) | **NEUTRAL** — reg-capped at 10 blocks/SM; occupancy lever closed |
| 21 | 2026-07-09 | decode | batched lane-per-dim partial (D=128) | **WIN** −26…−35%; vs SDPA 1.19→1.75–1.82×; vs FP8 0.59–0.75→0.89–1.07× |
| 22 | 2026-07-09 | decode | batched lane-per-dim extended to D=64 (replaces dp4a) | **WIN** −15…−22%; vs SDPA 1.04–1.47→1.10–1.89×; vs FP8 **1.14–1.48× (outright win)** |

---

## Test data — the 8 validation datasets

`generate_test_data.py` writes 8 datasets to `testdata/validate/` (+ a
`manifest.json` with each dataset's rationale and thresholds):

| Dataset | Problem it targets | Key thresholds |
|---|---|---|
| `normal` | Regression baseline: healthy post-LayerNorm distribution | cos>0.999, top10>0.8, outlier<0.01 |
| `outlier` | LLM.int8()-style outlier channels (10–100×) | cos>0.995, outlier<0.02 |
| `boundary` | Codes piled at ±126/127: clamp asymmetry, rounding | cos>0.999 |
| `stress` | Heavy tails + 3-decade magnitudes + peaky softmax | cos>0.995, top10>0.7, outlier<0.05 |
| `short_seq` | seq=4: scalar fallback path | cos>0.99, no NaN |
| `long_seq` | seq=4096: 64 KV tiles of online-softmax depth | cos>0.99, no NaN |
| `zeros` | amax=0 divide-by-zero guard; softmax over equal scores | no NaN, output ≈ 0 |
| `constant` | All-ones: softmax exactly uniform; single-code exact | cos>0.99, no NaN |

---

## Iterations

### Iteration 1 — attention: compile-time HEAD_DIM + fragment hoisting
- **Change** (`int8_attention.cu`): templated the WMMA kernel on
  `(TILE_KV, HEAD_DIM)`; reordered QK^T / P@V loops (k-outer, fragment hoisting).
  Two negatives kept as ablation flags: cp.async double-buffering (0.58–0.82×),
  V register prefetch (0.75–0.9×).
- **Result**: **1.6–2.2×** across the sweep grid (vs FA2 0.30→0.62×). Accuracy
  unchanged (max err ~1e-3).

### Iteration 2 — MLP: dynamic quantization
- **Change** (`int8_mlp.cu`): GEMM1 epilogue gathers per-token row absmax →
  per-token hidden requant → GEMM2 per-row scales → dynamic per-tensor output
  scale. Static path kept as `INT8_MLP_DYNAMIC=0`. `__launch_bounds__(256,2)`
  caps regs (epilogue stats pushed 128→166).
- **Result**: latency +1–6% (worst +14%); accuracy at d_model=2048 max err
  0.49→0.107 (4.6×), mean err 9.2× better. Accuracy win at small latency cost.

### Iterations 3–6 — attention micro-restructures (all latency-neutral)
The five accuracy gates passed on all of these; the kernels stayed numerically
bit-identical (pure reschedules). None moved latency beyond noise — they refined
the structure that later let the occupancy analysis conclude the well is dry.

- **Iter 3** — fuse the REGPV softmax-weight pack into the P@V MMA loop
  (group-outer/slice-inner), drop `p_frag[][4]` for per-group `pf[4]`. +1.5%.
- **Iter 4** — replace WMMA `m16n16k16` QK^T with native
  `mma.sync.aligned.m16n8k32.s8` (Q/K loaded as 4-byte head-dim-contiguous smem
  words). Target: denser tensor pipe + fewer smem fragment round-trips. −1.4%.
- **Iter 5** — fold V's per-token scale `sV` out of the pre-barrier V-dequant
  loop into the softmax-weight pack (shorter pre-barrier critical path). −0.9%.
- **Iter 6** — software-pipeline the P@V `ldmatrix` (hoist slice 0, prefetch
  slice s+1 while slice s's MMAs issue). Output bit-identical. ±0%.

### Iteration 7 — MLP: conflict-free hand-rolled m16n8k32 (largest MLP win)
- **Change** (`int8_mlp.cu`): replaced both GEMMs' `load_matrix_sync` +
  `mma_sync(m16n16k16)` inner loop with hand-rolled native
  `mma.sync.aligned.m16n8k32.s8`, operands loaded as k-contiguous 4-byte smem
  words. B (weights) pre-transposed to `[N][K]` per forward. Bank map
  `(12·group + tid_grp) mod 32` is a conflict-free bijection (NOT XOR swizzle).
- **Profiling**: bank conflicts 16.78M → ~0 (−99.98%); `l1tex__throughput`
  78%/67% → 39%/29%; smem-load wavefronts 36.2M → ~9M. `tensor_op_imma` ~28%
  (≈unchanged). **Latency −26% to −40%** across the full 12-pt sweep.
- **Gates**: all PASS (normal cos=1.00000; Gate 5 random-weight +0.011%).

### Iteration 8 — MLP: deepen cp.async K-stage 32→64
- **Change** (`int8_mlp.cu`): `INT8_MLP_STAGE_K` 32→64 (one-line default; smem
  strides 48→80 stay conflict-free + 16B-aligned). After iter 7 cleared the
  smem-pipe ceiling, `long_scoreboard` (global-load latency) was the top stall;
  a deeper k-stage halves the global-load sync points.
- **Profiling**: `long_scoreboard` 18.7%/27.4% → 5.5%/7.7%; `tensor_op_imma`
  26.1%/30.9% → 29.7%/38.2%; `wait` becomes the new top stall. **Latency −6% to
  −17%** across the sweep, numerics bit-identical.
- **Gates**: all PASS. The two MLP ceilings (bank conflicts, global-load latency)
  are now both exhausted; the bottleneck is the `wait` MMA-dependency stall at
  structurally-fixed 2-block occupancy.

### Iteration 9 — MLP accuracy: per-channel/per-token quant + real Gate 5
- **Change**: (a) `int8_mlp.cu` templated
  `gemm_int8_wmma_f16_kernel<apply_gelu, a_scale_per_row, b_scale_per_col>`; new
  `int8_mlp_forward_per_channel` does per-token activation + per-channel weight +
  **per-token output** quant. (b) `validate_int8.py` Gate 5 rewritten to **real
  GPT-2 perplexity on real WikiText-2** via the kernel-faithful sim. The
  per-tensor scalar `int8_mlp_forward` device signature is UNCHANGED (new path
  only — owner relaxed the no-interface-change rule for the accuracy track).
- **Key finding**: per-tensor MLP *output* quant gives **+64.4%** GPT-2
  perplexity (crushes output-channel outliers); per-token output is the fix →
  **−0.068%**. The old random-weight Gate 5 masked all of this (falsely +0.011%).
- **Result**: `outlier`/`boundary`/`stress` MLP now PASS outright (outlier cos
  0.971 → >0.999). Perf (sweep) unchanged. The INT8 MLP is now accurate on a
  real model, validated by a realistic gate.

### Iteration 10 — MLP denser-mma reschedule — NEGATIVE (reverted)
- **Change (reverted)**: hoisted both row tiles' A fragments, made the col-tile
  loop outer so each B fragment loads once, 4 back-to-back MMAs into 4 accs.
- **Result**: **+1.5% to +3.0% across ALL 12 shapes.** GEMM2 metrics
  byte-identical (`tensor_op_imma` 38.25→38.24%) — ptxas was **already CSE-ing**
  the "redundant" B loads; the reorder bought nothing. GEMM1 got worse (more
  `wait` from the larger live-register footprint).
- **Lesson**: warp-level mma issue order is already at ptxas's optimum; the
  `wait` stall is structural (needs more accumulator chains, but `acc`=64 regs is
  maxed). The MLP occupancy *and* denser-mma levers are both dry.

### Iteration 11 — MLP: prepacked transpose-once forward
- **Change** (`int8_mlp.cu` + `int8_ext.cu`): the all-in-one forward
  re-transposes W1/W2 to `[N][K]` on *every* call; weights are constant across
  forwards. New `transpose_int8_weights()` (one-shot, at load) +
  `int8_mlp_forward_prepacked` (skips the internal transpose). Old signatures and
  the `tests/cuda/` flow UNCHANGED. `sweep.py` now grades the prepacked path.
- **Profiling** (torch.profiler split): all-in-one = 79% GEMM + 10.4% transpose
  + 7.4% quantize_rows; prepacked = 88.3% GEMM + **0% transpose**. End-to-end
  **−22.1% (d512 s512), −9.3% (d1024 s512), −11.9% (d2048 s512)**; shrinks with
  seq_len (GEMM dwarfs the fixed transpose). GEMM time unchanged, output
  bit-identical.
- **Gates**: all PASS. A real end-to-end win found *outside* the exhausted GEMM,
  by amortizing the static-weight transpose.

### Iteration 12 — MLP finer 4×4 warp tiling — NEGATIVE (OFF flag `INT8_MLP_FINE_WARP`)
- **Change (flagged OFF)**: `WARP_COL_TILES` 4→2 (`acc[2][2][8]`=32 regs, was 64)
  → 16-warp/512-thread block. The smaller acc frees the registers iters 8/10
  proved you can't free at 256 threads, so occupancy can finally rise.
- **Result**: `warps_active` **23.8/21.5% → 47.6/41.9%** (doubled, as designed);
  `wait` **19.0/28.2% → 13.7/14.5%** (halved). **But `tensor_op_imma` NET FELL**
  (30.2/38.3% → 24.6/29.8%) and **latency regressed +15% to +28%**, because
  halving the acc halves A/B-fragment reuse.
- **Lesson**: closes the occupancy lever from the *opposite* direction of iters
  8/10 — you *can* raise occupancy, but the operand reuse you trade away to
  shrink the acc costs more than the extra warps buy. Occupancy and reuse are
  coupled through the acc-tile register cost. Do not re-attempt.

### Iteration 13 — MLP fuse quantize_rows into GEMM2 load — NEGATIVE (OFF flag `INT8_MLP_FUSE_QUANT`)
- **Change (flagged OFF)**: `gemm_int8_wmma_f16_a16fused_kernel` reads FP16 hidden
  directly as A and quantizes each per-token row to INT8 in smem during cp.async
  staging (using GEMM1's per-row absmax), eliminating one kernel launch + the
  int8-hidden HBM round-trip (61440 B dynamic smem).
- **Result**: **REGRESSED +45% to +63%** (scales with K-stage count). GEMM2 is
  wait-bound; the on-load convert adds a serial `convert + __syncthreads` per
  K-stage *between* the cp.async-wait and the mma (not overlapped), plus FP16
  staging doubles A's load bytes.
- **Lesson**: the separate memory-bound `quantize_rows` (~70% HBM BW as its own
  kernel) is *cheaper* than paying conversion latency inside the latency-bound
  GEMM. Rules out on-load fusion without decoupling the convert from the mma
  critical path. MLP forward orchestration is now also exhausted.

### Iteration 14 — decode: split-KV flash-decoding (new regime)
- **Change**: new `int8_decode_attention.cu` (+ `int8_decode_attention_forward`)
  for the **decode** regime (seq_q == 1), which the square WMMA kernel can't serve
  (15/16 fragment padding). Decode is a bandwidth-bound GEMV over the KV cache —
  exactly where an INT8 KV cache pays off (K,V at 1 byte vs FP16's 2).
  - **v1** (one warp per (b,h)): correct but lost to FP16 SDPA (0.04–0.90×) —
    grid-starved, memory latency not hidden.
  - **v2** (split-KV / flash-decoding, shipped): NSPLIT warps per (b,h), each
    scanning one KV chunk into a partial online-softmax state; a combine kernel
    merges them with log-sum-exp. Multiplies resident warps → saturates BW.
- **Result**: **D=128 (B=64): 1.23× faster than FP16 SDPA** at all seq lens,
  757–826 GB/s (~52% of HBM peak), at half the KV bytes. **D=64 still loses
  (0.46–0.94×, ~390 GB/s)** — instruction-bound (DPL=2; the 6-step shuffle
  reduction + 2× `__expf` dominate the tiny load). KV footprint is structurally
  half FP16. Decode cos 0.99994–0.99995; all 5 gates still PASS.

### Iteration 15 — decode: dp4a lane-per-position (D=64 now wins)
- **Change** (`int8_decode_attention.cu`): new
  `int8_decode_partial_dp4a_kernel<HEAD_DIM>` flips the intra-warp division —
  each lane scores a whole KV **position** per 32-position tile via a full
  HEAD_DIM `__dp4a` dot (no per-position shuffle reduction; warp reduces only
  twice per tile). Host dispatches the best partial per head_dim.
- **Result**: **D=64 WIN.** B≥32 went **0.61–0.94× → 1.24–1.56×** vs FP16 SDPA;
  HBM BW ~390 → up to 786 GB/s (~2×). Removing the per-position reduction moved
  D=64 from instruction-bound back to bandwidth-bound. **D=64 → dp4a, D=128 →
  split-KV** is the dispatched optimum; **dp4a @ D=128 regresses** (register
  pressure) — do-not-retry. Decode attention now beats FP16 SDPA at every
  serving-scale (B≥32) shape for both head dims.

### Iteration 16 — attention K/V smoothing — NEGATIVE (no code)
- **Investigated**: SageAttention-style K/V smoothing (subtract per-channel mean
  before per-token quant) as the attention K/V outlier accuracy target.
- **Finding**: **no demonstrable output-level benefit.** For K, the correction
  `Q·μ_K` is constant across keys → softmax (shift-invariant in the key dim)
  cancels it → it never reaches the output. Purpose-built channel-mean-bias K
  (Dettmers magnitudes) does NOT fail the gate (output cos 0.9988–1.0000 plain).
  Smoothing moved only intermediate metrics (K-MAE 5.8×, outlier_ratio) that
  don't reach the gate metric (output cosine) — the textbook "moved an
  intermediate metric, not the result" trap. The existing `outlier`/`stress`
  datasets use zero-mean multiplicative spikes, where mean-subtraction is a no-op.
- **Conclusion**: target CLOSED. Do not re-attempt without a real-model
  distribution that demonstrably FAILs the output-cosine gate first.

### Iteration 17 — full-layer integration: first end-to-end block wall-clock
- **Change**: new `bench_block.py` assembles a pre-LN transformer block and times
  it end to end, INT8 vs FP16. INT8 covers only the attention core + MLP;
  LayerNorm, residual adds, and QKV/output projections stay FP16 and byte-
  identical in both paths, isolating the INT8 swap.
- **Result**: prefill S=512/1024/2048 = **0.52/0.54/0.55×** (INT8 slower);
  decode Skv=2K…32K = **0.83/0.92/1.05/1.12/1.19×** — crosses 1.0 at ~8K and the
  win grows with context. cos ≥ 0.9994 (prefill) / ≥ 0.9986 (decode).
- **Reconciliation**: this is NOT a contradiction of the "3.8–4.8× MLP" headline
  (that is vs the naive INT8 `_int_mm` pipeline). vs FP16 cuBLAS GEMM the INT8
  GEMM is 0.71–0.88× and always was — the win is vs naive INT8 deployment + the
  KV-cache bandwidth, exactly the memory-bound thesis.
- **New actionable cost surfaced**: the eager-mode **quant/dequant glue at
  inter-op boundaries** (~0.68 ms of the prefill block). Fairness bracket:
  removing the glue (perfect producer-fusion) still leaves prefill at 0.82× — it
  loses even with perfect fusion, because the kernels run ~0.7–0.8× cuBLAS at
  prefill. The decode long-context win is the robust signal; it survives both
  handicaps.

### Iteration 18 — decode vs a real quantized-KV SOTA (FlashInfer FP8)
- **Change**: new `bench_decode_sota.py`. Closes the biggest credibility gap: the
  decode win was only ever measured vs **FP16** SDPA, never vs a *quantized*
  decode kernel (prefill was benched vs the INT8 SOTA SageAttention; decode was
  not — an asymmetry).
- **Peer**: **FlashInfer 0.6.12 batch decode, FP8 (e4m3) paged KV.** FlashInfer
  has no INT8 KV on `sm_80`, so FP8 is the apples-to-apples peer — both stream
  **1 byte/elem** of KV → identical KV bandwidth, isolating kernel quality at
  equal bytes. (FP8 e4m3 is the production-deployed quantized-KV format.)
- **Fairness check**: `use_tensor_cores=True` on the FlashInfer wrapper is
  REQUIRED — it >2× speeds FP8 decode (3.29→1.52 ms @ B64 S8K); leaving the
  default `False` would have crippled the SOTA and produced a false win.
- **Result**:
  - **D=64 (our tuned dp4a path): par-to-WIN.** ours/fp8 mostly ≥1.0, up to
    **1.28×**, AND more accurate (ours cos **0.99995** vs fp8 **0.99922** — our
    per-token INT8 scales beat FP8's per-tensor scale).
  - **D=128 (our untuned split-KV): LOSS** ~1.3–1.7× (the known-weak path; now a
    concrete measured target).
  - **vs FlashInfer FP16** (well-tuned): ~1.06–1.21× — the half-bytes win, but
    smaller than the 1.2–1.6× vs the weaker PyTorch SDPA baseline.
- **Conclusion**: the decode story is now honest AND stronger. At D=64 our INT8
  decode is competitive-to-ahead of the production FP8 SOTA at equal KV bandwidth
  and more accurate — the win is real kernel quality, not "INT8 vs unquantized".
  On Ampere, INT8 is the *right* quantized format. Next kernel target: D=128 decode.

### Iteration 19 — harden Gate 5: run the SHIPPED kernel, not a simulation
- **Change**: Gate 5 (task-level real-GPT-2 perplexity) previously measured a
  PyTorch *simulation* of the kernel's quantization (`_kernel_faithful_mlp`).
  This iteration makes it run the **actual CUDA kernel** end-to-end inside a real
  GPT-2 forward, closing the sim-vs-real gap.
- **Two parts**:
  1. New entry point `int8_mlp_forward_per_channel_bias` (`int8_mlp.cu` +
     binding): the per-channel/per-token path plus an optional **per-channel
     pre-GELU bias** (GPT-2 `c_fc` bias) added inside the GEMM1 fused epilogue
     before GELU. `apply_bias` is an OFF-by-default template flag → every
     existing instantiation is byte-identical and the graded path is unchanged.
     `b2` (`c_proj` bias) stays FP16 at the caller after dequant.
  2. `validate_int8.py` Gate 5 now patches each block's MLP with
     `_kernel_real_mlp_fwd` (weights per-channel-quantized once, activation
     per-token per call, token rows zero-padded up to BLOCK_M=128 for the WMMA
     path), calling the real kernel. The standalone driver `gate5_real_kernel.py`
     shares the exact same forward (single source of truth) for ad-hoc
     model/context runs.
- **Why it matters**: the old gate could pass while the real kernel regressed —
  a model of the code is not the code. Gates 1–4 already check the kernel against
  its reference; Gate 5 now also exercises it at task level.
- **Result**: all 5 gates PASS. Real GPT-2 perplexity (WikiText-2, n_ctx=3072):
  FP16 **31.9462** → INT8 kernel **31.9890** = **+0.134%**, well under the 2%
  bar. The integrated gate reproduces the standalone driver's number exactly,
  confirming the wiring. (Naive per-tensor *output* quant would be +64%; the
  per-token output quant is what recovers it — do not revert.)
- **Note**: this is an accuracy/validation-integrity change, not a perf change;
  no inner loop or graded shape was touched.

### Iteration 20 — decode: TARGET_UNITS occupancy sweep — NEUTRAL (occupancy lever closed)
- **Context**: first-ever `ncu` profile of the decode kernels (they were never
  profiled in iters 14–18; B=64 H=16 S=8192, `sudo apt install nsight-compute`,
  binary at `/usr/lib/nsight-compute/ncu`). Both shipped partials turned out
  **memory-latency-bound at ~half of HBM peak**: D=128 lane-per-dim 49% DRAM /
  73–75% `long_scoreboard` / 57% `warps_active`; D=64 dp4a 48% DRAM / **89%**
  `long_scoreboard` / 58% `warps_active`; 48 regs/thread both (not reg-starved
  per thread, but 48 regs × 128 thr caps residency at 10 blocks/SM = 40/64
  warps). The combine kernel is negligible.
- **Change (experiment)**: made the host `TARGET_UNITS` heuristic (4096)
  runtime-overridable via env `INT8_DECODE_TARGET_UNITS` (read once; default
  unchanged) and swept 4096/8192/16384/32768, median-of-5 per shape.
- **Result**: NEUTRAL. All serving shapes within ±2% noise except D=128 long
  context (−3.1…−3.6% at S=32K) — because the extra blocks only queue behind
  the 10-block/SM register cap; resident warps do not increase. Matches the
  MLP's iter-12 lesson from the opposite direction: decode occupancy is closed
  **by the register ceiling**, not by grid size. The env knob is kept as the
  tuning hook (shipped default still 4096).
- **Conclusion**: latency must be hidden **per warp** (more independent loads
  in flight), not by more warps → iteration 21.

### Iteration 21 — decode: batched lane-per-dim partial for D=128 (the D=128 gap closes)
- **Change** (`int8_decode_attention.cu`): new
  `int8_decode_partial_batch_kernel<HEAD_DIM=128, NB>` replaces the original
  lane-per-dim partial for D=128 (old kernel kept under `INT8_DECODE_BATCH=0`).
  The original loop was a serial per-position chain — 1-word K load → 6-shuffle
  score reduction → per-position softmax rescale → V fma — so ≤1 KV row was in
  flight per warp. The batched kernel processes **NB positions per group**:
  1. issue all NB K-word + sK/sV scalar loads back-to-back (NB rows in flight),
     then NB `dp4a` partial dots each reduced with **one sm_80
     `redux.sync.add`** (`__reduce_add_sync`) instead of the 6-shuffle tree;
  2. **one** online-softmax update per group (group max is warp-uniform after
     redux → register max-tree, one corr `__expf`, one acc rescale, NB weight
     `__expf`s) — mathematically identical to per-position online softmax;
  3. issue all NB V `char4` loads, then NB fused fma-accumulates.
  Lane-per-dim layout is unchanged → every K/V access stays a fully-coalesced
  contiguous 128 B line. Shipped `NB=4` (48 regs — same 10-block/SM residency
  as before; NB=8→54 regs, NB=16→80 regs tie NB=4 on latency but cost blocks;
  env `INT8_DECODE_NB` selects 4/8/16 for sweeps). D=64 dispatch untouched.
- **Profiling** (B64 H16 S8192): DRAM throughput **49% → 82% of peak** at
  identical occupancy (54.7% warps, 48 regs) — pure memory-level-parallelism
  win, exactly what iter 20 predicted the axis had to be.
- **Result** (median of 5): D=128 S=8K **2.561 → 1.795 ms (−30%)**, S=32K
  **10.204 → 6.585 ms (−35%)**, ~1.30 TB/s effective KV bandwidth (~84% HBM
  peak). vs FP16 SDPA: **1.19–1.20× → 1.75–1.82×** across S=1K–32K. vs
  **FlashInfer FP8** (equal 1 B/elem KV): **0.59–0.75× → 0.89–1.07×** — the
  D=128 gap that iter 18 flagged as the biggest weakness is now par (win ≤8K,
  −10% at 16K/32K). Note the FP8 peer also *improved* between measurements
  (0.6.12 → 0.6.14, ~18% faster at some shapes), so this is par against a
  *faster* SOTA than iter 18 faced. D=64 rows unchanged (1.18–1.21× vs SDPA,
  regression-checked).
- **Gates**: all 5 PASS (Gate 5 real-GPT-2 +0.134%); new 16-case decode edge
  harness (S ∈ {1,31,33,127,128,1000,4096,8192} × both D) all PASS,
  cos = 1.000000 vs fp32 reference.
- **Remaining D=128 headroom**: NB=16 and TARGET_UNITS=16384 each buy ~1–2%
  at S=32K (where FP8 still leads ~10%) — a long-context-only dispatch tweak
  is the obvious next probe. D=64 dp4a (89% long_scoreboard, 48% DRAM) should
  respond to the same batching treatment: its lane-per-position layout makes
  K reads 64 B-strided per lane, so the fix there is likely cp.async smem
  staging or a hybrid — a separate iteration.
- **Journey chart**: `figures/make_roofline.py` renders the whole arc on an
  A100 roofline (`results/figures/roofline.png`, embedded in the README) —
  decode's iter-14→15→21 climb toward the INT8-KV roof, the FP16-SDPA
  "already at its roof" wedge that motivates the byte thesis, and the
  compute-bound prefill/MLP points honestly below theirs.

### Iteration 22 — decode: batched lane-per-dim extended to D=64 (dp4a superseded)
- **Hypothesis**: the shipped D=64 dp4a path had the same disease iter 21 cured
  at D=128 — 48% DRAM, **89%** `long_scoreboard` — but a different anatomy: its
  lane-per-position layout makes each lane read a whole 64 B K row as 16
  64 B-strided 4 B words (16× the LSU wavefronts of a coalesced access), and
  only ~1 row is in flight per warp.
- **Change** (`int8_decode_attention.cu`): generalized
  `int8_decode_partial_batch_kernel` to DPL=2 (`char2` K/V words, two-IMAD
  lane dot via a `lane_dot` overload; same `redux.sync.add` reduction, same
  one-rescale-per-group softmax). Host now routes **both** head dims to the
  batched kernel with NB=4; the dp4a path stays for ablation
  (`INT8_DECODE_BATCH64=0`). NB=4 again ties/beats NB=8/16 at 48 regs
  (58/89 regs) — the same "batch just enough, keep residency" optimum.
- **Profiling** (B64 H16 S8192): DRAM **48% → 63%**, `long_scoreboard`
  **89% → 60%**, occupancy unchanged (54% warps, 48 regs).
- **Result** (median of 5, tune harness): −15…−22% across serving shapes
  (S=32K B64H16: 5.282 → 4.148 ms, 813 → 1036 GB/s). Published-bench sweep:
  - **vs FP16 SDPA**: B≥32 **1.45–1.89×** (was 1.04–1.47×); even the
    grid-starved B=8 rows flip from **0.80–0.98× (loss) to 1.10–1.31× (win)** —
    batching restores in-flight bytes exactly where few warps are resident.
  - **vs FlashInfer FP8 0.6.14** (equal 1 B/elem KV): **1.14–1.48× — an
    outright win at every D=64 shape** (was 0.80–1.16× par). D=128 rows
    unchanged (0.89–1.07×, regression-checked).
- **Gates**: all 5 PASS; 16-case decode edge harness PASS (cos = 1.000000).
- **Conclusion**: one kernel structure (batched lane-per-dim, NB=4) is now the
  dispatched optimum for both head dims; iter 15's dp4a is superseded. The
  decode story vs the production FP8 SOTA is now **win at D=64, par at D=128**,
  at equal KV bytes and higher accuracy.
- **Downstream refresh + fairness bounds** (same day):
  - `bench_block.py` re-run with the new decode kernels: the end-to-end block
    decode crossover moves **~8K → ~4K** context and S=32K improves
    **1.19× → 1.51×** vs the FP16 block (prefill unchanged, 0.54×).
  - **Paging-tax bound**: FlashInfer serves *paged* KV (PAGE=16 — the
    vLLM-style production config) while our kernel is contiguous. Re-measured
    at PAGE=256 the FP8 peer speeds up ~3–12%: the D=64 win holds
    (1.14–1.31×), but D=128 reads **0.86–0.88×** — i.e. part of the D=128
    "par" at PAGE=16 is the peer's paging overhead. Recorded in the README
    claim table; the honest D=128 statement is "par in the production paged
    config, ~-13% against an unpaged FP8 bound".
  - SDPA-baseline sanity: PyTorch 2.7 auto-dispatches decode SDPA to the
    FLASH_ATTENTION backend (within 6% of the best backend at the probed
    shape) — the FP16 baseline is not a crippled strawman. Remaining decode headroom: D=128
  long-context (−10% at S≥16K) and the ~60% `long_scoreboard` still left at
  D=64 (~65% of roof) — likely needs cp.async smem staging or deeper NB with
  a register diet; diminishing returns vs pivoting to producer fusion.
