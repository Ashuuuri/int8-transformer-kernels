// int8_decode_attention.cu — INT8 flash-decoding attention (seq_q == 1), split-KV.
//
// WHY A SEPARATE KERNEL.  The square WMMA kernel in int8_attention.cu assumes
// seq_q == seq_kv and tiles Q into 16-row WMMA fragments — useless when seq_q
// == 1 (autoregressive decode), where 15/16 of every tensor-core tile would be
// padding. Decode is a GEMV (Q·Kᵀ) + weighted sum (P·V), and it is
// **bandwidth-bound on streaming the KV cache**, not compute-bound. That is
// exactly where an INT8 KV cache pays off: K and V are 1 byte/elem instead of 2,
// so the dominant cost (reading the whole cache every step) roughly halves.
//
// SPLIT-KV (flash-decoding).  A v1 with one warp per (batch, head) only reached
// ~34% of HBM BW and lost to FP16 SDPA: decode's parallelism is batch*heads,
// which is small, so few warps are resident and memory latency is not hidden.
// The fix (this kernel) is to also split the KV dimension: NSPLIT warps
// cooperate on each (b,h), each scanning one KV chunk and producing a PARTIAL
// online-softmax state (m, l, unnormalized Σ exp(score−m)·V). A second kernel
// merges the NSPLIT partials per (b,h) with the standard log-sum-exp rescale.
// This multiplies resident warps by NSPLIT → fills the SMs → saturates BW, and
// also cures grid starvation at small batch. NSPLIT is chosen by the host to
// target a high total warp count.
//
// DESIGN (both kernels).  One warp per work-unit; WARPS_PER_BLOCK warps/block.
// Each of the 32 lanes owns DPL = HEAD_DIM/32 head dims (HEAD_DIM=64→2, 128→4)
// and keeps the running accumulator for just those dims (DPL floats/thread). The
// QK score is an int8×int8 partial per lane + a warp-shuffle reduction (NO
// __syncthreads, NO shared memory). K and V are streamed once, fully coalesced
// (consecutive lanes read consecutive dims). Barrier-free → stays bandwidth-
// bound, which is what makes the INT8 KV-cache win show up.
//
// LIMITATIONS.  seq_q == 1; HEAD_DIM ∈ {64, 128} (must be %32). Per-token scales,
// same convention as the prefill kernel: sQ[bh], sK[bh*S+j], sV[bh*S+j].

#include <cuda_fp16.h>
#include <math.h>
#include <cuda_runtime.h>

#ifndef DECODE_WARPS_PER_BLOCK
#define DECODE_WARPS_PER_BLOCK 4
#endif

// Partial-kernel layout selector.  0 = original "lane owns dims" (a 6-step
// warp-shuffle score reduction per KV position — instruction-bound at HEAD_DIM=64,
// only ~25% HBM BW). 1 = dp4a "lane owns position": each lane computes a FULL
// HEAD_DIM dot via __dp4a (no per-position reduction); the warp reduces only
// twice per 32-position tile (softmax max/sum). PV stays in dims-layout (coalesced
// V, p·sV broadcast by shuffle). See OPTIMIZATION_LOG iter 15.
#ifndef INT8_DECODE_DP4A
#define INT8_DECODE_DP4A 1
#endif

// ── Pass 1: per-(b,h, split) partial online-softmax over one KV chunk ──────
template <int HEAD_DIM>
__global__ void int8_decode_partial_kernel(
    const int8_t* __restrict__ Q,   // [BH, HEAD_DIM]
    const int8_t* __restrict__ K,   // [BH, S, HEAD_DIM]
    const int8_t* __restrict__ V,   // [BH, S, HEAD_DIM]
    const float*  __restrict__ sQ,  // [BH]
    const float*  __restrict__ sK,  // [BH*S]
    const float*  __restrict__ sV,  // [BH*S]
    float* __restrict__ p_m,        // [BH*NSPLIT]
    float* __restrict__ p_l,        // [BH*NSPLIT]
    float* __restrict__ p_acc,      // [BH*NSPLIT, HEAD_DIM]  (unnormalized)
    int S, int BH, int nsplit, int chunk, float attn_scale)
{
    constexpr int DPL = HEAD_DIM / 32;
    const int unit = blockIdx.x * DECODE_WARPS_PER_BLOCK + threadIdx.y;
    if (unit >= BH * nsplit) return;
    const int bh   = unit / nsplit;
    const int sp   = unit % nsplit;
    const int lane = threadIdx.x;
    const int base = lane * DPL;

    const int j0 = sp * chunk;
    const int j1 = min(S, j0 + chunk);

    int8_t q[DPL];
    #pragma unroll
    for (int i = 0; i < DPL; ++i) q[i] = Q[bh * HEAD_DIM + base + i];
    const float sq = sQ[bh];

    const int8_t* Kbh  = K  + (size_t)bh * S * HEAD_DIM;
    const int8_t* Vbh  = V  + (size_t)bh * S * HEAD_DIM;
    const float*  sKbh = sK + (size_t)bh * S;
    const float*  sVbh = sV + (size_t)bh * S;

    float m = -INFINITY, l = 0.0f;
    float acc[DPL];
    #pragma unroll
    for (int i = 0; i < DPL; ++i) acc[i] = 0.0f;

    for (int j = j0; j < j1; ++j) {
        const int8_t* kj = Kbh + (size_t)j * HEAD_DIM + base;
        int dot = 0;
        #pragma unroll
        for (int i = 0; i < DPL; ++i) dot += (int)q[i] * (int)kj[i];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            dot += __shfl_down_sync(0xffffffffu, dot, o);
        dot = __shfl_sync(0xffffffffu, dot, 0);

        const float score = (float)dot * sq * sKbh[j] * attn_scale;
        const float m_new = fmaxf(m, score);
        const float corr  = __expf(m - m_new);
        const float p     = __expf(score - m_new);
        l = l * corr + p;
        const int8_t* vj = Vbh + (size_t)j * HEAD_DIM + base;
        const float   sv = sVbh[j];
        #pragma unroll
        for (int i = 0; i < DPL; ++i)
            acc[i] = acc[i] * corr + p * ((float)vj[i] * sv);
        m = m_new;
    }

    if (lane == 0) { p_m[unit] = m; p_l[unit] = l; }
    #pragma unroll
    for (int i = 0; i < DPL; ++i)
        p_acc[(size_t)unit * HEAD_DIM + base + i] = acc[i];
}

// ── Pass 1 (dp4a variant): lane owns a KV POSITION, not dims ───────────────
// Each iteration a warp handles a TILE of 32 KV positions: lane L scores
// position (t+L) with a full HEAD_DIM dot via __dp4a — NO per-position shuffle
// reduction. The warp reduces only twice per tile (softmax max + sum). For the
// P·V accumulation we flip back to dims-layout (lane owns DPL dims of acc, V read
// coalesced) and broadcast each position's p·sV with a single __shfl. So the
// score path pays ~2 warp-reductions / 32 positions instead of 6 shuffles / 1.
template <int HEAD_DIM>
__global__ void int8_decode_partial_dp4a_kernel(
    const int8_t* __restrict__ Q,   // [BH, HEAD_DIM]
    const int8_t* __restrict__ K,   // [BH, S, HEAD_DIM]
    const int8_t* __restrict__ V,   // [BH, S, HEAD_DIM]
    const float*  __restrict__ sQ,  // [BH]
    const float*  __restrict__ sK,  // [BH*S]
    const float*  __restrict__ sV,  // [BH*S]
    float* __restrict__ p_m,        // [BH*NSPLIT]
    float* __restrict__ p_l,        // [BH*NSPLIT]
    float* __restrict__ p_acc,      // [BH*NSPLIT, HEAD_DIM]  (unnormalized)
    int S, int BH, int nsplit, int chunk, float attn_scale)
{
    constexpr int DPL = HEAD_DIM / 32;   // dims per lane for the acc/PV phase
    constexpr int NW  = HEAD_DIM / 4;    // int32 words per row for dp4a (QK phase)
    const int unit = blockIdx.x * DECODE_WARPS_PER_BLOCK + threadIdx.y;
    if (unit >= BH * nsplit) return;
    const int bh   = unit / nsplit;
    const int sp   = unit % nsplit;
    const int lane = threadIdx.x;
    const int base = lane * DPL;

    const int j0 = sp * chunk;
    const int j1 = min(S, j0 + chunk);

    // Q stays resident as NW int32 words (same for every position; packed int8x4).
    int q[NW];
    const int32_t* Qw = reinterpret_cast<const int32_t*>(Q + (size_t)bh * HEAD_DIM);
    #pragma unroll
    for (int w = 0; w < NW; ++w) q[w] = Qw[w];
    const float sq = sQ[bh];

    const int8_t* Kbh  = K  + (size_t)bh * S * HEAD_DIM;
    const int8_t* Vbh  = V  + (size_t)bh * S * HEAD_DIM;
    const float*  sKbh = sK + (size_t)bh * S;
    const float*  sVbh = sV + (size_t)bh * S;

    float m = -INFINITY, l = 0.0f;
    float acc[DPL];
    #pragma unroll
    for (int i = 0; i < DPL; ++i) acc[i] = 0.0f;

    for (int t = j0; t < j1; t += 32) {
        const int tile_n = min(32, j1 - t);
        const bool valid = lane < tile_n;             // lane scores position t+lane

        // QK: full-row dp4a, one score per lane, no reduction.
        float score = -INFINITY;
        float svl   = 0.0f;
        if (valid) {
            const int32_t* krow =
                reinterpret_cast<const int32_t*>(Kbh + (size_t)(t + lane) * HEAD_DIM);
            int dot = 0;
            #pragma unroll
            for (int w = 0; w < NW; ++w) dot = __dp4a(q[w], krow[w], dot);
            score = (float)dot * sq * sKbh[t + lane] * attn_scale;
            svl   = sVbh[t + lane];
        }

        // Softmax over the tile (2 warp reductions for the whole 32 positions).
        float tile_max = score;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, o));
        const float m_new = fmaxf(m, tile_max);
        const float corr  = __expf(m - m_new);
        const float p     = valid ? __expf(score - m_new) : 0.0f;
        float tile_l = p;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            tile_l += __shfl_xor_sync(0xffffffffu, tile_l, o);
        l = l * corr + tile_l;

        // P·V: dims-layout. Rescale acc, then broadcast each position's p·sV and
        // accumulate its (coalesced) V row into this lane's DPL dims.
        const float psv = p * svl;     // lane L holds position (t+L)'s weight
        #pragma unroll
        for (int i = 0; i < DPL; ++i) acc[i] *= corr;
        for (int jj = 0; jj < tile_n; ++jj) {
            const float w = __shfl_sync(0xffffffffu, psv, jj);
            const int8_t* vrow = Vbh + (size_t)(t + jj) * HEAD_DIM + base;
            #pragma unroll
            for (int i = 0; i < DPL; ++i) acc[i] += w * (float)vrow[i];
        }
        m = m_new;
    }

    if (lane == 0) { p_m[unit] = m; p_l[unit] = l; }
    #pragma unroll
    for (int i = 0; i < DPL; ++i)
        p_acc[(size_t)unit * HEAD_DIM + base + i] = acc[i];
}

// ── Pass 2: merge the NSPLIT partials per (b,h) (log-sum-exp rescale) ──────
template <int HEAD_DIM>
__global__ void int8_decode_combine_kernel(
    half* __restrict__ out,         // [BH, HEAD_DIM]
    const float* __restrict__ p_m,  // [BH*NSPLIT]
    const float* __restrict__ p_l,  // [BH*NSPLIT]
    const float* __restrict__ p_acc,// [BH*NSPLIT, HEAD_DIM]
    int BH, int nsplit)
{
    constexpr int DPL = HEAD_DIM / 32;
    const int bh = blockIdx.x * DECODE_WARPS_PER_BLOCK + threadIdx.y;
    if (bh >= BH) return;
    const int lane = threadIdx.x;
    const int base = lane * DPL;

    float gm = -INFINITY;
    for (int s = 0; s < nsplit; ++s) gm = fmaxf(gm, p_m[bh * nsplit + s]);

    float L = 0.0f;
    float ACC[DPL];
    #pragma unroll
    for (int i = 0; i < DPL; ++i) ACC[i] = 0.0f;

    for (int s = 0; s < nsplit; ++s) {
        const int unit = bh * nsplit + s;
        const float ls = p_l[unit];
        if (ls <= 0.0f) continue;                 // empty chunk
        const float w = __expf(p_m[unit] - gm);
        L += ls * w;
        #pragma unroll
        for (int i = 0; i < DPL; ++i)
            ACC[i] += p_acc[(size_t)unit * HEAD_DIM + base + i] * w;
    }

    const float inv = 1.0f / L;
    #pragma unroll
    for (int i = 0; i < DPL; ++i)
        out[bh * HEAD_DIM + base + i] = __float2half(ACC[i] * inv);
}

// ── Scratch for the partials (cached, grows as needed — avoids a cudaMalloc
//    on every decode step; single-stream use, as in the benchmark harness) ──
static float* g_decode_scratch = nullptr;
static size_t g_decode_scratch_bytes = 0;

static float* ensure_scratch(size_t bytes) {
    if (bytes > g_decode_scratch_bytes) {
        if (g_decode_scratch) cudaFree(g_decode_scratch);
        cudaMalloc(&g_decode_scratch, bytes);
        g_decode_scratch_bytes = bytes;
    }
    return g_decode_scratch;
}

// Host launcher. BH = batch*heads, S = kv length, head_dim ∈ {64,128}.
void int8_decode_attention_forward(
    const int8_t* Q, const int8_t* K, const int8_t* V, half* out,
    const float* sQ, const float* sK, const float* sV,
    int BH, int S, int head_dim)
{
    const float attn_scale = 1.0f / sqrtf((float)head_dim);

    // Choose NSPLIT to target a high total warp count (fill the SMs / hide
    // latency) without over-splitting into tiny chunks.
    const int MIN_CHUNK = 128, TARGET_UNITS = 4096, MAX_SPLIT = 128;
    int nsplit = TARGET_UNITS / BH; if (nsplit < 1) nsplit = 1;
    const int by_len = (S / MIN_CHUNK) > 1 ? (S / MIN_CHUNK) : 1;
    if (nsplit > by_len)   nsplit = by_len;
    if (nsplit > MAX_SPLIT) nsplit = MAX_SPLIT;
    const int chunk = (S + nsplit - 1) / nsplit;

    const size_t units = (size_t)BH * nsplit;
    const size_t bytes = units * 2 * sizeof(float)            // m, l
                       + units * (size_t)head_dim * sizeof(float); // acc
    float* p_m   = ensure_scratch(bytes);
    float* p_l   = p_m + units;
    float* p_acc = p_l + units;

    dim3 block(32, DECODE_WARPS_PER_BLOCK);
    dim3 gridP((units + DECODE_WARPS_PER_BLOCK - 1) / DECODE_WARPS_PER_BLOCK);
    dim3 gridC((BH    + DECODE_WARPS_PER_BLOCK - 1) / DECODE_WARPS_PER_BLOCK);

    // Ship the empirically-best partial per head_dim (bench_attn_decode, iter 15):
    //   HEAD_DIM=64  → dp4a lane-per-position WINS (1.24–1.56× @ B≥32, ~2× the BW);
    //   HEAD_DIM=128 → dp4a REGRESSES to 0.86–0.93× (NW=32 Q-regs cut occupancy),
    //                  so the original lane-per-dim split-KV (1.23×) stays.
    // INT8_DECODE_DP4A=0 forces the original D=64 path too (A/B / ablation).
    if (head_dim == 64) {
#if INT8_DECODE_DP4A
        int8_decode_partial_dp4a_kernel<64><<<gridP, block>>>(
            Q, K, V, sQ, sK, sV, p_m, p_l, p_acc, S, BH, nsplit, chunk, attn_scale);
#else
        int8_decode_partial_kernel<64><<<gridP, block>>>(
            Q, K, V, sQ, sK, sV, p_m, p_l, p_acc, S, BH, nsplit, chunk, attn_scale);
#endif
        int8_decode_combine_kernel<64><<<gridC, block>>>(
            out, p_m, p_l, p_acc, BH, nsplit);
    } else if (head_dim == 128) {
        int8_decode_partial_kernel<128><<<gridP, block>>>(
            Q, K, V, sQ, sK, sV, p_m, p_l, p_acc, S, BH, nsplit, chunk, attn_scale);
        int8_decode_combine_kernel<128><<<gridC, block>>>(
            out, p_m, p_l, p_acc, BH, nsplit);
    }
    // other head_dims unsupported (must be a multiple of 32)
}
