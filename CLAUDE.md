# CLAUDE.md — operating guide for int8-transformer-kernels

This is the **operating guide**: how an agent should think and act when
continuing the INT8 kernel optimization in this repo. It is the distilled, *still-actionable*
conclusion of ~18 iterations. The blow-by-blow evidence behind every claim —
before/after `ncu` numbers, validation traces, negative results — lives in
[`OPTIMIZATION.md`](OPTIMIZATION.md). This file tells you **what to do and what
is already exhausted**; that file tells you **why**.

Target hardware: A100-SXM4-40GB, `sm_80`, CUDA 12.8, system PyTorch 2.7.

---

## 0. The five principles (read these first)

1. **Win on bytes, not on TOPS.** These kernels are memory-bound at the workload
   level. Optimize toward **fewer HBM bytes / fewer kernel launches / fused
   epilogues**, never toward tensor-pipe % for its own sake. A change that raises
   tensor-pipe % but adds an HBM round-trip is a *loss*; one that lowers it but
   removes a round-trip is a *win*.
2. **Latency is the only gate that counts.** `ncu` metrics are diagnostic — use
   them to form a hypothesis and *explain* a result, never as the goal. The most
   repeated lesson here: moving a bottleneck metric has *almost never* translated
   to a speedup. Occupancy raised to 4 blocks/SM, smem traffic −24%, finer warp
   tiling that halved the `wait` stall — every one moved its metric and every one
   was latency-neutral-or-worse.
3. **One coherent change per iteration.** Not a size limit — a single change may
   rewrite a whole main loop and touch several files, as long as it is *one idea*.
   Do not water down an ambitious idea into a safe one-liner; the gates are the
   safety net, so a reverted bold attempt is a better iteration than a committed
   no-op.
4. **Validate before you celebrate.** Run the 5 accuracy gates (§4) after *every*
   kernel change, and treat sweep latency as the **median of ≥5 runs** (|Δ| < 2%
   is noise on this box). Never claim a speedup from a single sweep.
5. **Pivot axis after 2–3 dead iterations on the same kernel.** A dry well is a
   signal to switch axis (accuracy/quantization granularity, a different kernel,
   or end-to-end integration), not a cue to grind harder. The do-not-retry list
   in §5 exists so you can recognize "I'm about to re-grind a dead axis" *before*
   spending the iteration.

---

## 1. Architecture

| File | Purpose |
|---|---|
| `kernels/int8_attention.cu` | INT8 **prefill** attention (seq_q == seq_kv). INT8 WMMA QK^T (s32 accum) + FP16 WMMA PV, per-token scales, online softmax, templated on `(TILE_KV, HEAD_DIM)`. |
| `kernels/int8_decode_attention.cu` | INT8 **decode** attention (seq_q == 1). Split-KV flash-decoding (per-chunk partial online-softmax + log-sum-exp combine), warp-per-(b,h,split), barrier-free. Host dispatches the best partial per head_dim: **D=64 → `dp4a` lane-per-position**, **D=128 → split-KV lane-per-dim**. `INT8_DECODE_DP4A=0` forces the original D=64 path (ablation). |
| `kernels/int8_mlp.cu` | INT8 MLP. 128×128-tile INT8 WMMA ×2, GELU fused into GEMM1, dynamic per-token quant. Entry points: `int8_mlp_forward` (per-tensor scalar scales, legacy/unchanged), `int8_mlp_forward_per_channel` (per-token act + per-channel weight + per-token output — the real-model accuracy path), `int8_mlp_forward_prepacked` (transpose-once static-weight path; what `sweep.py` grades). |
| `kernels/int8_common.cuh` | Shared device helpers: GELU, f32→i8, cp.async. |
| `kernels/quant_utils.cu` | Per-tensor quantize/dequantize utilities. |
| `kernels/int8_ext.cu` | pybind11 bindings (`torch.utils.cpp_extension.load`, extension name `int8_ext`). |

Data flow:

```
INT8 prefill attention:
  Q,K,V fp16 → [per-token quantize] → int8 + scales[B*H*S]
  → INT8 WMMA QK^T (s32) → ×sQ·sK·1/√d → online softmax (fp32 regs)
  → P fp16 → FP16 WMMA P@V (V dequant per-row) → out fp16

INT8 MLP (dynamic-quant path):
  x int8 → GEMM1 (INT8 WMMA) → epilogue ×sx·sW1 → GELU → fp16 hidden
           + per-token row absmax → quantize_rows → int8
  → GEMM2 (INT8 WMMA, per-row A scales) → fp16 out → quantize → int8 + scale
```

---

## 2. Environment setup — run FIRST in a fresh box

The system PyTorch JIT-builds the extension with `cpp_extension.load`, which
needs **ninja** and the **pybind11 C++ headers on the system include path**. A
fresh box usually ships neither, and the errors are non-obvious. Install in this
order (passwordless sudo is available):

```bash
# 1. ninja (required by torch cpp_extension.load)
command -v ninja || pip3 install --user ninja

# 2. pybind11 C++ headers ON THE SYSTEM INCLUDE PATH (/usr/include/pybind11).
#    The system torch does NOT add the pip pybind11 package's include dir to the
#    nvcc command line, so `pip install pybind11` alone does NOT fix the build.
ls /usr/include/pybind11/pybind11.h 2>/dev/null || sudo apt-get install -y pybind11-dev

# 3. Sanity-check end to end (~2 min):
rm -rf ~/.cache/torch_extensions/py312_cu128/int8_ext
python3 -c "from torch.utils.cpp_extension import load; \
  ext=load(name='int8_ext', sources=['kernels/int8_attention.cu', \
  'kernels/int8_decode_attention.cu','kernels/int8_mlp.cu', \
  'kernels/quant_utils.cu','kernels/int8_ext.cu'], \
  extra_cuda_cflags=['-arch=sm_80','--std=c++17','-O3']); print('BUILD OK')"
```

The extension rebuilds automatically after `.cu` edits; if the cache misbehaves
`rm -rf ~/.cache/torch_extensions/py312_cu128/int8_ext`. `flashinfer-python` is
needed only for `bench_decode_sota.py`; `transformers` + `datasets` only for
Gate 5.

---

## 3. Profiling — profile the GRADED shape

**CRITICAL: profile the shape `sweep.py` grades, not `test_int8.py --quick`.**
`--quick` (batch=2, seq=512) is grid-starved (≈1.18 blocks/SM → `warps_active`
~7%); the graded config (batch=8, head_dim=64, seq 512–4096) maxes occupancy (3
blocks/SM, `warps_active` ~18%). They are different regimes — profiling the toy
made earlier iterations chase an occupancy/register lever that does not exist in
the graded config. Use `profile_kernel.py` (sweep-matched shapes) for any
perf-relevant diagnosis.

GPU performance counters require sudo (`ERR_NVGPUCTRPERM` otherwise). Always
filter with `--kernel-name` + `grep` — full `ncu` output floods context:

```bash
sudo env "PATH=$PATH" HOME="$HOME" \
    ncu --kernel-name regex:int8_wmma --launch-count 2 \
    --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,\
launch__registers_per_thread,launch__occupancy_limit_registers \
    python3 profile_kernel.py attn 2>&1 | grep -A2 -E "Metric|int8_"
```

For per-kernel timings without privileges, `collect_profile.py` /
`torch.profiler` need no sudo. ptxas spill/reg check:
`nvcc -arch=sm_80 -O3 --std=c++17 --ptxas-options=-v -c kernels/int8_mlp.cu -o /dev/null 2>&1 | grep -E "registers|spill"`.

---

## 4. Accuracy validation — 5 gates, no skipping

Run `python validate_int8.py` after **every** optimization. All five must pass;
any failure stops the optimization immediately (apply the repair suggestions it
prints — per-channel quant → QK^T back to fp16 → adjust scales → worst stage back
to fp16 — then re-run from Gate 1).

| Gate | Checks | Criteria |
|---|---|---|
| 1 Math metrics | cosine, top-10 overlap, outlier ratio, NaN/Inf | cos>0.999 (outlier: >0.995), top10>0.8, outlier<0.01 |
| 2 Numeric stability | per-stage absmax ratio, per-stage NaN | ratio ∈ [1/1.5, 1.5] |
| 3 Stage error trace | per-stage cosine must not drop >1% vs previous | stop + report on violation |
| 4 Edge cases | short_seq / long_seq / zeros / constant | no NaN; cos>0.99 (zeros ≈ 0) |
| 5 Task-level | **real GPT-2 perplexity on WikiText-2** (`testdata/real_corpus.txt`) | perplexity increase < 2% (SKIPs if `transformers`/corpus missing) |

Gate 5 is the realistic gate: it runs each MLP block with the kernel's exact
per-channel weight + per-token activation + **per-token output** quant. Naive
per-tensor output quant gives **+64%** real-GPT-2 perplexity (it crushes output
channel outliers); per-token output recovers it (−0.07%). **Do not revert Gate 5
to random weights.** The 8 validation datasets (`generate_test_data.py`) target
specific failure modes — see [`OPTIMIZATION.md`](OPTIMIZATION.md) §"Test data".

---

## 5. What is exhausted — DO NOT re-attempt

The perf well is, by now, mostly dry. Before proposing a perf change, check it
is not on this list (full evidence in `OPTIMIZATION.md`):

**MLP GEMMs — CLOSED (perf-exhausted from every direction):**
- smem bank conflicts — FIXED (hand-rolled conflict-free `mma.sync m16n8k32`).
- global-load latency — FIXED (`STAGE_K` 32→64). A 3-stage cp.async pipeline is a
  dead end (latency already hidden; 3rd buffer breaks the smem ceiling).
- the `wait` (MMA-result dependency) stall is **structural** at 2 blocks/SM: more
  independent accumulator chains need registers the `acc[2][4][8]`=64-reg tile
  doesn't have. Denser-mma reschedule (iter 10) regressed and left GEMM2
  byte-identical (ptxas already optimal).
- occupancy is dead **from both sides**: can't raise it at 256 threads
  (reg-capped); raising it via finer 4×4/512-thread tiling (iter 12, flag
  `INT8_MLP_FINE_WARP`) *does* double `warps_active` and halve `wait`, but halves
  operand reuse so `tensor_op_imma` net falls and latency regresses.
- fusing `quantize_rows` into GEMM2's load (iter 13, flag `INT8_MLP_FUSE_QUANT`)
  regressed +45–63% — GEMM2 is wait-bound and the serial on-load convert per
  K-stage more than doubles it.
- the per-forward W transpose was amortized via `int8_mlp_forward_prepacked`
  (transpose-once at load) — forward orchestration is now also exhausted.

**Attention prefill — occupancy well is DRY:**
- forcing 4 blocks/SM (`<64,64>`) gave no speedup; a −24% smem-traffic cut gave
  none either. `tensor_pipe` stays ~28% regardless. The ceiling is the **per-warp
  online-softmax dependency chain** (`ldmatrix → mma → exp/MUFU → pack → mma` +
  cross-tile rescale), which neither more warps nor less smem traffic relieves.
- known negatives, do not retry: cp.async double-buffering (`INT8_ATTN_DB`), V
  register prefetch (`INT8_ATTN_VPREFETCH`).

**Decode:**
- `dp4a` @ D=128 is a do-not-retry (register-pressure regression). D=64 → dp4a,
  D=128 → split-KV is the dispatched optimum.

**Accuracy:**
- attention K/V smoothing (SageAttention-style, iter 16) has **no** output-level
  benefit here: K channel-bias *cancels in softmax* (shift-invariance), so it
  never reaches the output. Do not re-attempt without a real-model distribution
  that demonstrably FAILs the output-cosine gate first.

---

## 6. Where the remaining headroom actually is

In priority order:

1. **Decode D=128** — the one place we lose to a *fair* quantized peer
   (FlashInfer FP8, ~1.5×). Concrete, measured target. (`dp4a@128` is *not* the
   route — it regresses.)
2. **End-to-end producer-fusion** — the eager-mode quant/dequant glue at inter-op
   boundaries (~0.68 ms of the prefill block) is the one fusion lever per-kernel
   work can't reach: fuse the quantize into the *producer* (LayerNorm→int8 emit;
   attention-out→int8+scale emit) so the consumer reads int8 with no extra HBM
   round-trip. Needs NEW fused entry points (allowed), not changes to the
   exhausted inner loops. Helps decode; will **not** rescue prefill (which loses
   even with perfect fusion).
3. **Attention online-softmax dependency chain** — the only *kernel-internal*
   perf lever left, and deep/high-risk. Attempt only after 1–2.
4. **Accuracy / quantization granularity** — historically where the *shippable*
   wins came from once perf dried up (iter 9's per-channel/per-token fixed a +64%
   perplexity regression). When perf is dry, this is usually the right axis.

---

## 7. Hard constraints — do not break

- **LayerNorm and residual paths stay FP16.** Never quantize them (in
  `validate_int8.py`'s block harness or any full-layer integration).
- **Public interface signatures stay backward-compatible.** New quantization
  granularities are added as *new* entry points / overloads, never by breaking
  old ones. The per-tensor `int8_mlp_forward` (scalar scales) is UNCHANGED; per-
  channel work lives in `int8_mlp_forward_per_channel`. `int8_attention_forward`
  is untouched. This also keeps the `tests/cuda/` .bin smoke-test flow working.
- **Negative results are KEPT, not reverted** — but only behind an OFF-by-default
  ablation flag with a README/`OPTIMIZATION.md` note, so they are not silently
  re-attempted. A latency-neutral micro-opt that moved *no* metric is a no-op:
  report it and revert. (Under `evolve.sh`, a kept negative is signalled by a
  `NEGATIVE_RESULT:` line.)

---

## 8. Iteration workflow

1. Baseline: `python validate_int8.py` and `python sweep.py --kernel int8_*` —
   record the numbers.
2. Make one coherent change.
3. Correctness: `python validate_int8.py` (all 5 gates).
4. Performance: `python sweep.py` (median of ≥5; |Δ| < 2% is noise).
5. Commit only when both pass, carrying before/after numbers. Append a new entry
   to [`OPTIMIZATION.md`](OPTIMIZATION.md) (newest at the bottom). Record negative
   results too (behind ablation flags).

`evolve.sh` automates this loop (profile → optimize → validate → commit).
