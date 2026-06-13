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
