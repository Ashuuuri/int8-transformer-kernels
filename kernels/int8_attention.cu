// int8_attention.cu — INT8 quantized scaled dot-product attention.
//
// Per-token symmetric quantization:
//   Each token row has its own scale factor → scale arrays of size [batch*heads*seq_len].
//   QK^T element [i][j] = int32_acc * scale_Q[i] * scale_K[j] * attn_scale
//   V dequant row r: V_fp16[r][c] = V_int8[r][c] * scale_V[r]
//   Output is FP16 (can't factor out a single scale from weighted V rows).
//
// Three paths:
//   Double-buffered WMMA path (INT8_ATTN_DB=1, default):
//     cp.async double-buffers K/V (INT8) so tile t+1 loads while tile t
//     computes; V dequantized smem→smem after QK^T. TILE_KV: 64 for
//     head_dim≤128, 32 up to head_dim=256.
//
//   Single-buffer WMMA path (INT8_ATTN_DB=0, or seq_len not divisible):
//     load → sync → compute → sync, no overlap.
//     Template TILE_KV: 128 for head_dim≤128 (2 blocks/SM), 64 for head_dim=256.
//
//   Scalar path (INT8_ATTN_WMMA=0, or non-aligned dims):
//     Same algorithm, loop-based dot products.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <math.h>
#include <cstdint>
#include "int8_common.cuh"

#ifndef INT8_ATTN_WMMA
#define INT8_ATTN_WMMA 1
#endif

// cp.async double-buffered WMMA path. Measured SLOWER than the single-buffer
// kernel on A100 (0.58-0.82x): fitting two K/V buffers in smem forces TILE_KV
// from 128/64 down to 64/32, and the doubled per-tile overhead (softmax
// rescale of frag_out, warp shuffles, +1 barrier) outweighs the overlap.
// Kept for ablation; off by default.
#ifndef INT8_ATTN_DB
#define INT8_ATTN_DB 0
#endif

// V register prefetch: issue V's global loads at tile start so the QK^T MMAs
// hide their latency, then dequantize registers → smem afterwards. Measured
// SLOWER on A100 (0.75-0.9x): the extra block-wide barrier before attn×V
// costs more than the hidden latency — warp parallelism already covers the
// V loads. Kept for ablation; off by default.
#ifndef INT8_ATTN_VPREFETCH
#define INT8_ATTN_VPREFETCH 0
#endif

// Register-resident P for the attn × V product (default): the m16n16k16 INT8
// WMMA accumulator layout matches the mma.m16n8k16 A-fragment layout, so the
// softmax weights can be packed half2 in registers and fed straight into PTX
// mma.sync; V is loaded via ldmatrix.trans. Removes the s_scores smem
// round-trip (write + barrier + fragment reload) — the structural MIO/smem
// bottleneck. Old WMMA P@V path kept for ablation (INT8_ATTN_REGPV=0).
#ifndef INT8_ATTN_REGPV
#define INT8_ATTN_REGPV 1
#endif

// ── Scalar path ─────────────────────────────────────────────────────────
#define SCALAR_TILE_KV  64
#define SCALAR_BDIM    128

__global__ void int8_fused_attention_kernel(
    const int8_t* __restrict__ Q,
    const int8_t* __restrict__ K,
    const int8_t* __restrict__ V,
    half* __restrict__         out,
    const float* __restrict__ d_scale_Q,   // [batch*heads*seq_len]
    const float* __restrict__ d_scale_K,   // [batch*heads*seq_len]
    const float* __restrict__ d_scale_V,   // [batch*heads*seq_len]
    int seq_len, int head_dim,
    float attn_scale
) {
    const int bh  = blockIdx.y;
    const int qi  = blockIdx.x;
    const int tid = threadIdx.x;

    // Per-token scale for this query row
    const float sQ = __ldg(&d_scale_Q[bh * seq_len + qi]);

    extern __shared__ float smem[];
    float* s_Q      = smem;
    float* s_K      = s_Q + head_dim;
    float* s_V      = s_K + SCALAR_TILE_KV * head_dim;
    float* s_scores = s_V + SCALAR_TILE_KV * head_dim;
    float* s_out    = s_scores + SCALAR_TILE_KV;
    float* s_reduce = s_out + head_dim;

    // Load Q row (keep as float, scale already captured)
    for (int d = tid; d < head_dim; d += SCALAR_BDIM) {
        s_Q[d] = (float)Q[((size_t)bh * seq_len + qi) * head_dim + d];
        s_out[d] = 0.f;
    }
    __syncthreads();

    float rmax = -1e38f, rsum = 0.f;

    for (int t = 0; t < (seq_len + SCALAR_TILE_KV - 1) / SCALAR_TILE_KV; ++t) {
        const int ts   = t * SCALAR_TILE_KV;
        const int tlen = min(SCALAR_TILE_KV, seq_len - ts);

        // Load K tile (raw int8 → float, no scale yet)
        // Load V tile with per-row dequantization
        for (int idx = tid; idx < tlen * head_dim; idx += SCALAR_BDIM) {
            const int r = idx / head_dim, d = idx % head_dim;
            const size_t base = ((size_t)bh * seq_len + ts + r) * head_dim + d;
            s_K[r * head_dim + d] = (float)K[base];
            float sV_row = __ldg(&d_scale_V[bh * seq_len + ts + r]);
            s_V[r * head_dim + d] = (float)V[base] * sV_row;
        }
        __syncthreads();

        // Compute QK^T scores with per-token scales
        for (int j = tid; j < tlen; j += SCALAR_BDIM) {
            float acc = 0.f;
            for (int d = 0; d < head_dim; ++d) acc += s_Q[d] * s_K[j * head_dim + d];
            float sK_j = __ldg(&d_scale_K[bh * seq_len + ts + j]);
            s_scores[j] = acc * sQ * sK_j * attn_scale;
        }
        __syncthreads();

        // Online softmax: find tile max
        float lmax = -1e38f;
        for (int j = tid; j < tlen; j += SCALAR_BDIM) lmax = fmaxf(lmax, s_scores[j]);
        s_reduce[tid] = lmax;
        __syncthreads();
        for (int s = SCALAR_BDIM >> 1; s >= 1; s >>= 1) {
            if (tid < s) s_reduce[tid] = fmaxf(s_reduce[tid], s_reduce[tid + s]);
            __syncthreads();
        }

        const float new_max = fmaxf(rmax, s_reduce[0]);
        const float corr    = __expf(rmax - new_max);
        for (int d = tid; d < head_dim; d += SCALAR_BDIM) s_out[d] *= corr;
        rsum *= corr;

        float tile_sum = 0.f;
        for (int j = 0; j < tlen; ++j) {
            const float w = __expf(s_scores[j] - new_max);
            tile_sum += w;
            for (int d = tid; d < head_dim; d += SCALAR_BDIM)
                s_out[d] += w * s_V[j * head_dim + d];
        }
        rmax = new_max;
        rsum += tile_sum;
        __syncthreads();
    }

    // Normalize and write FP16 output
    const float inv_rsum = 1.f / rsum;
    for (int d = tid; d < head_dim; d += SCALAR_BDIM) {
        out[((size_t)bh * seq_len + qi) * head_dim + d] =
            __float2half(s_out[d] * inv_rsum);
    }
}

// ── WMMA path ──────────────────────────────────────────────────────────
#if INT8_ATTN_WMMA

using namespace nvcuda::wmma;

#define ATTN_WMMA_M     16
#define ATTN_WMMA_N     16
#define ATTN_WMMA_K     16
#define ATTN_TILE_Q     64   // 4 warps × 16 rows each
#define ATTN_BDIM      128   // 4 warps: better latency hiding + faster KV loading
#define ATTN_SMEM_PAD    8
#define ATTN_I8_PAD     16  // break INT8 smem bank conflicts (stride must not be multiple of 128 bytes)

// Maximum head_dim supported = ATTN_WMMA_N * ATTN_MAX_SLICES = 16 * 16 = 256.
#define ATTN_MAX_SLICES 16

// D = A·B + D — native INT8 tensor-core op: A s8 16x32, B s8 32x8, D s32 16x8.
// Used for QK^T: replaces the WMMA m16n16k16 path's load_matrix_sync round-trips
// with plain 4-byte smem reads (each thread's int8 lanes are head-dim-contiguous)
// and one native instruction per call (WMMA m16n16k16.s8 decomposes into several).
// The s32 accumulator element order (c0,c1 = rows groupID, cols 2*tid_in_grp,+1;
// c2,c3 = rows groupID+8) is exactly two n-halves of the WMMA m16n16k16 layout,
// so two calls (low/high n8) fill x[0..3]/x[4..7] and downstream code is unchanged.
__device__ __forceinline__ void attn_mma_m16n8k32_s8(
    int32_t* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

#if INT8_ATTN_REGPV
// Load four 8x8 FP16 tiles with transpose: row-major V in smem comes back as
// the col-major B fragments mma.m16n8k16 expects (r0/r1 = k0-7/k8-15 of the
// low n-half, r2/r3 = the high n-half).
__device__ __forceinline__ void attn_ldmatrix_x4_trans(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const half* src) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(src));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(a));
}

// D = A·B + D — A fp16 16x16 in registers, B fp16 16x8, D f32 16x8.
__device__ __forceinline__ void attn_mma_m16n8k16(
    float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t attn_pack_half2(float lo, float hi) {
    const half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}
#endif  // INT8_ATTN_REGPV

// TILE_KV is a template parameter: 128 for head_dim≤128, 64 for head_dim=256.
// Larger tiles → fewer iterations → less overhead → better perf.
// HEAD_DIM is also a template parameter: with it compile-time constant, every
// fragment-array index unrolls and frag_out lives in registers — a runtime
// head_dim forces frag_out[] into local memory, which costs a load+store per
// MMA and per softmax-rescale step.
template <int TILE_KV, int HEAD_DIM>
__global__ __launch_bounds__(ATTN_BDIM, 2)
void int8_wmma_attention_kernel(
    const int8_t* __restrict__ Q,
    const int8_t* __restrict__ K,
    const int8_t* __restrict__ V,
    half* __restrict__         out,
    const float* __restrict__ d_scale_Q,   // [batch*heads*seq_len]
    const float* __restrict__ d_scale_K,   // [batch*heads*seq_len]
    const float* __restrict__ d_scale_V,   // [batch*heads*seq_len]
    int seq_len,
    float attn_scale
) {
    constexpr int head_dim = HEAD_DIM;
    const int bh      = blockIdx.y;
    const int qi_base = blockIdx.x * ATTN_TILE_Q;
    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane    = tid % 32;

    extern __shared__ char smem_raw[];
    // INT8 row stride: pad to break bank conflicts.
    const int ks_i8 = head_dim + ATTN_I8_PAD;
    // FP16 strides (in half elements).
    const int vs = head_dim + ATTN_SMEM_PAD;      // V row stride
#if !INT8_ATTN_REGPV
    const int ss = TILE_KV + ATTN_SMEM_PAD;  // scores row stride
#endif

    // ── Single-buffer smem layout ──
    // [Q INT8] [K INT8] [V FP16] [scores FP16 (WMMA P@V only)] [K_scales]
    int8_t* s_Q_i8    = (int8_t*) smem_raw;
    int8_t* s_K_i8    = s_Q_i8 + ATTN_TILE_Q * ks_i8;
    half*   s_V_fp16  = (half*)(s_K_i8 + TILE_KV * ks_i8);
#if INT8_ATTN_REGPV
    float*  s_scale_K = (float*)(s_V_fp16 + TILE_KV * vs);
#else
    half*   s_scores  = s_V_fp16 + TILE_KV * vs;
    float*  s_scale_K = (float*)(s_scores + ATTN_TILE_Q * ss);
#endif

    // Load Q tile as INT8 (vectorized 16-byte loads).
    {
        const int q_vecs = ATTN_TILE_Q * head_dim / 16;
        const int4 zero4 = make_int4(0, 0, 0, 0);
        for (int vi = tid; vi < q_vecs; vi += ATTN_BDIM) {
            const int byte_off = vi * 16;
            const int row = byte_off / head_dim;
            const int col = byte_off % head_dim;
            const int qi  = qi_base + row;
            int4 val = (qi < seq_len)
                ? *reinterpret_cast<const int4*>(&Q[((size_t)bh * seq_len + qi) * head_dim + col])
                : zero4;
            *reinterpret_cast<int4*>(&s_Q_i8[row * ks_i8 + col]) = val;
        }
    }

    // Fragment element layout (m16n16k16 accumulator, sm_80):
    const int frow0   = lane / 4;
    const int frow1   = lane / 4 + 8;
    const int fcol_lo = (lane % 4) * 2;

    // Pre-fold attn_scale into Q scales.
    const int qi0_global = qi_base + warp_id * ATTN_WMMA_M + frow0;
    const int qi1_global = qi_base + warp_id * ATTN_WMMA_M + frow1;
    const float r_sQ0 = (qi0_global < seq_len)
        ? __ldg(&d_scale_Q[bh * seq_len + qi0_global]) * attn_scale : 0.f;
    const float r_sQ1 = (qi1_global < seq_len)
        ? __ldg(&d_scale_Q[bh * seq_len + qi1_global]) * attn_scale : 0.f;

    float rmax0 = -1e38f, rmax1 = -1e38f;
    float rsum0 = 0.f,    rsum1 = 0.f;

    constexpr int n_slices    = HEAD_DIM / ATTN_WMMA_N;
    constexpr int n_kv_groups = TILE_KV / ATTN_WMMA_N;
    constexpr int kv_vecs     = TILE_KV * HEAD_DIM / 16;

#if INT8_ATTN_REGPV
    // Raw accumulators with the m16n8k16 C layout — element order matches the
    // WMMA m16n16k16 accumulator .x[] order, so rescale/output code is shared.
    float frag_out[n_slices][8];
    #pragma unroll
    for (int s = 0; s < n_slices; ++s) {
        #pragma unroll
        for (int e = 0; e < 8; ++e) frag_out[s][e] = 0.f;
    }
#define FRAG_OUT(s, e) frag_out[s][e]
#else
    fragment<accumulator, ATTN_WMMA_M, ATTN_WMMA_N, ATTN_WMMA_K, float> frag_out[n_slices];
    #pragma unroll
    for (int s = 0; s < n_slices; ++s) fill_fragment(frag_out[s], 0.f);
#define FRAG_OUT(s, e) frag_out[s].x[e]
#endif

    const int num_tiles = seq_len / TILE_KV;

    __syncthreads();  // ensure Q load is complete

    // ── Main loop: load → sync → compute → sync ──
    // Up to 8 16-byte V vectors per thread across supported configs
    // (TILE_KV * head_dim / 16 / ATTN_BDIM ≤ 8 for head_dim ≤ 256).
    #define ATTN_V_REGS 8

    for (int t = 0; t < num_tiles; ++t) {
        const int ts = t * TILE_KV;
        const size_t kv_base = ((size_t)bh * seq_len + ts) * head_dim;
        const size_t scale_base = (size_t)bh * seq_len + ts;

#if INT8_ATTN_VPREFETCH
        // Load K (INT8) into smem; issue V loads into registers so the QK^T
        // MMAs below hide their global latency.
        int4  v_regs[ATTN_V_REGS];
        float v_scales[ATTN_V_REGS];
        #pragma unroll
        for (int r = 0; r < ATTN_V_REGS; ++r) {
            const int vi = tid + r * ATTN_BDIM;
            if (vi >= kv_vecs) break;
            const int byte_off = vi * 16;
            const int row = byte_off / head_dim;
            const int col = byte_off % head_dim;
            const size_t goff = kv_base + row * head_dim + col;
            *reinterpret_cast<int4*>(&s_K_i8[row * ks_i8 + col]) =
                *reinterpret_cast<const int4*>(&K[goff]);
            v_regs[r]   = *reinterpret_cast<const int4*>(&V[goff]);
            v_scales[r] = __ldg(&d_scale_V[scale_base + row]);
        }
#else
        // Load K (INT8), V (INT8→FP16 dequant), K_scales
        for (int vi = tid; vi < kv_vecs; vi += ATTN_BDIM) {
            const int byte_off = vi * 16;
            const int row = byte_off / head_dim;
            const int col = byte_off % head_dim;
            const size_t goff = kv_base + row * head_dim + col;
            // K: vectorized INT8 load
            *reinterpret_cast<int4*>(&s_K_i8[row * ks_i8 + col]) =
                *reinterpret_cast<const int4*>(&K[goff]);
            // V: vectorized load + per-row dequant to FP16
            int4 v_vec = *reinterpret_cast<const int4*>(&V[goff]);
            const float sv = __ldg(&d_scale_V[scale_base + row]);
            const int8_t* vb = reinterpret_cast<const int8_t*>(&v_vec);
            half* vdst = &s_V_fp16[row * vs + col];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                reinterpret_cast<half2*>(vdst)[i] = __halves2half2(
                    __float2half((float)vb[2*i]     * sv),
                    __float2half((float)vb[2*i + 1] * sv));
            }
        }
#endif
        for (int idx = tid; idx < TILE_KV; idx += ATTN_BDIM) {
            s_scale_K[idx] = __ldg(&d_scale_K[scale_base + idx]);
        }
        __syncthreads();

        // ── QK^T via native INT8 mma.sync m16n8k32 → INT32 accumulators ──
        // k-step is 32 (vs 16 for WMMA). Each logical n16 group g is two n8
        // mma calls (kv cols g*16+0..7 and +8..15) whose s32 accumulators land
        // in qk[g][0..3] / qk[g][4..7] — exactly the WMMA m16n16k16 .x[] order,
        // so the cast/softmax/P@V code below is untouched. Operands are loaded
        // as plain 4-byte smem reads instead of load_matrix_sync (each thread's
        // 4 int8 lanes are head-dim-contiguous), removing the WMMA fragment-load
        // smem round-trips that dominate the shared-memory wavefront count.
        int32_t qk[TILE_KV / ATTN_WMMA_N][8];
        #pragma unroll
        for (int g = 0; g < n_kv_groups; ++g)
            #pragma unroll
            for (int e = 0; e < 8; ++e) qk[g][e] = 0;

        const int groupID = lane >> 2;       // 0..7  (mma row / B-column index)
        const int tid_grp = lane & 3;         // 0..3  (k-segment within a group)
        const int8_t* qrow0 = s_Q_i8 + (warp_id * ATTN_WMMA_M + groupID) * ks_i8;
        const int8_t* qrow8 = qrow0 + 8 * ks_i8;
        #pragma unroll 2
        for (int k = 0; k < head_dim; k += 32) {
            const int ka = k + tid_grp * 4;
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(qrow0 + ka);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(qrow8 + ka);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(qrow0 + ka + 16);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(qrow8 + ka + 16);
            #pragma unroll
            for (int g = 0; g < n_kv_groups; ++g) {
                const int8_t* krl = s_K_i8 + (g * ATTN_WMMA_N + groupID) * ks_i8;
                const int8_t* krh = krl + 8 * ks_i8;
                const uint32_t b0l = *reinterpret_cast<const uint32_t*>(krl + ka);
                const uint32_t b1l = *reinterpret_cast<const uint32_t*>(krl + ka + 16);
                const uint32_t b0h = *reinterpret_cast<const uint32_t*>(krh + ka);
                const uint32_t b1h = *reinterpret_cast<const uint32_t*>(krh + ka + 16);
                attn_mma_m16n8k32_s8(&qk[g][0], a0, a1, a2, a3, b0l, b1l);
                attn_mma_m16n8k32_s8(&qk[g][4], a0, a1, a2, a3, b0h, b1h);
            }
        }

#if INT8_ATTN_VPREFETCH
        // Dequantize the prefetched V registers → s_V_fp16. The loads were
        // issued before QK^T, so they have already landed by now.
        #pragma unroll
        for (int r = 0; r < ATTN_V_REGS; ++r) {
            const int vi = tid + r * ATTN_BDIM;
            if (vi >= kv_vecs) break;
            const int byte_off = vi * 16;
            const int row = byte_off / head_dim;
            const int col = byte_off % head_dim;
            const float sv = v_scales[r];
            const int8_t* vb = reinterpret_cast<const int8_t*>(&v_regs[r]);
            half* vdst = &s_V_fp16[row * vs + col];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                reinterpret_cast<half2*>(vdst)[i] = __halves2half2(
                    __float2half((float)vb[2*i]     * sv),
                    __float2half((float)vb[2*i + 1] * sv));
            }
        }
#endif

        // Cast INT32 accumulators to float with per-element scaling.
        float sf[TILE_KV / ATTN_WMMA_N][8];
        float lmax0 = -1e38f, lmax1 = -1e38f;
        for (int g = 0; g < n_kv_groups; ++g) {
            const int kc_base = g * ATTN_WMMA_N;
            float sK_c0 = s_scale_K[kc_base + fcol_lo];
            float sK_c1 = s_scale_K[kc_base + fcol_lo + 1];
            float sK_c8 = s_scale_K[kc_base + fcol_lo + 8];
            float sK_c9 = s_scale_K[kc_base + fcol_lo + 9];

            sf[g][0] = (float)qk[g][0] * r_sQ0 * sK_c0;
            sf[g][1] = (float)qk[g][1] * r_sQ0 * sK_c1;
            sf[g][2] = (float)qk[g][2] * r_sQ1 * sK_c0;
            sf[g][3] = (float)qk[g][3] * r_sQ1 * sK_c1;
            sf[g][4] = (float)qk[g][4] * r_sQ0 * sK_c8;
            sf[g][5] = (float)qk[g][5] * r_sQ0 * sK_c9;
            sf[g][6] = (float)qk[g][6] * r_sQ1 * sK_c8;
            sf[g][7] = (float)qk[g][7] * r_sQ1 * sK_c9;

            lmax0 = fmaxf(lmax0, fmaxf(fmaxf(sf[g][0],sf[g][1]), fmaxf(sf[g][4],sf[g][5])));
            lmax1 = fmaxf(lmax1, fmaxf(fmaxf(sf[g][2],sf[g][3]), fmaxf(sf[g][6],sf[g][7])));
        }
        // Reduce max across threads sharing the same row.
        lmax0 = fmaxf(lmax0, __shfl_xor_sync(0xffffffff, lmax0, 1));
        lmax0 = fmaxf(lmax0, __shfl_xor_sync(0xffffffff, lmax0, 2));
        lmax1 = fmaxf(lmax1, __shfl_xor_sync(0xffffffff, lmax1, 1));
        lmax1 = fmaxf(lmax1, __shfl_xor_sync(0xffffffff, lmax1, 2));

        // Online softmax correction
        const float new_max0 = fmaxf(rmax0, lmax0), corr0 = __expf(rmax0 - new_max0);
        const float new_max1 = fmaxf(rmax1, lmax1), corr1 = __expf(rmax1 - new_max1);
        rmax0 = new_max0; rmax1 = new_max1;
        rsum0 *= corr0;   rsum1 *= corr1;
        #pragma unroll
        for (int s = 0; s < n_slices; ++s) {
            FRAG_OUT(s,0)*=corr0; FRAG_OUT(s,1)*=corr0;
            FRAG_OUT(s,2)*=corr1; FRAG_OUT(s,3)*=corr1;
            FRAG_OUT(s,4)*=corr0; FRAG_OUT(s,5)*=corr0;
            FRAG_OUT(s,6)*=corr1; FRAG_OUT(s,7)*=corr1;
        }

        float lsum0 = 0.f, lsum1 = 0.f;
#if INT8_ATTN_REGPV
        // Softmax weights → mma.m16n8k16 A fragments, consumed immediately by
        // the P@V MMAs of the SAME KV group. Fusing the weight-pack with the
        // P@V MMA keeps only one group's packed weights live (pf[4]) instead
        // of the full p_frag[n_kv_groups][4] array, and — the point of this
        // iteration — interleaves the softmax exp/pack (MUFU pipe) with the
        // ldmatrix+mma (LSU/tensor pipe) across groups instead of running them
        // as two serial phases, so the transcendental latency hides behind the
        // MMAs. (The global register peak is set by the QK^T cast loop, not
        // here, so the ptxas register count is unchanged.) The WMMA m16n16k16
        // accumulator element positions (frow0/frow1 × fcol_lo,+1,+8,+9) are
        // exactly the A-fragment positions of two n-halves, so no shuffle is
        // needed: a0={w0,w1}
        // a1={u0,u1} (cols fcol_lo,+1), a2={w8,w9} a3={u8,u9} (cols +8,+9).
        // V comes via ldmatrix.trans (four 8x8 tiles = both B fragments of a
        // 16-wide head-dim slice). s_V_fp16 has been visible since the load
        // barrier at tile start (the prefetch path writes it after QK^T, so
        // it needs an extra block barrier here).
#if INT8_ATTN_VPREFETCH
        __syncthreads();
#endif
        const int ld_row = lane & 15;          // k offset inside the tile
        const int ld_col = (lane >> 4) * 8;    // n offset (low/high half)
        #pragma unroll
        for (int g = 0; g < n_kv_groups; ++g) {
            const float w0=__expf(sf[g][0]-rmax0), w1=__expf(sf[g][1]-rmax0);
            const float w8=__expf(sf[g][4]-rmax0), w9=__expf(sf[g][5]-rmax0);
            lsum0 += w0+w1+w8+w9;
            const float u0=__expf(sf[g][2]-rmax1), u1=__expf(sf[g][3]-rmax1);
            const float u8=__expf(sf[g][6]-rmax1), u9=__expf(sf[g][7]-rmax1);
            lsum1 += u0+u1+u8+u9;
            uint32_t pf[4];
            pf[0] = attn_pack_half2(w0, w1);
            pf[1] = attn_pack_half2(u0, u1);
            pf[2] = attn_pack_half2(w8, w9);
            pf[3] = attn_pack_half2(u8, u9);
            #pragma unroll
            for (int s = 0; s < n_slices; ++s) {
                uint32_t b0, b1, b2, b3;
                attn_ldmatrix_x4_trans(b0, b1, b2, b3,
                    &s_V_fp16[(g * ATTN_WMMA_N + ld_row) * vs
                              + s * ATTN_WMMA_N + ld_col]);
                attn_mma_m16n8k16(&frag_out[s][0], pf, b0, b1);
                attn_mma_m16n8k16(&frag_out[s][4], pf, b2, b3);
            }
        }
#else
        // Softmax weights → s_scores (FP16, for attn × V WMMA).
        for (int g = 0; g < n_kv_groups; ++g) {
            const int qr0 = warp_id * ATTN_WMMA_M + frow0;
            const int qr1 = warp_id * ATTN_WMMA_M + frow1;
            const int kc  = g * ATTN_WMMA_N + fcol_lo;
            const float w0=__expf(sf[g][0]-rmax0), w1=__expf(sf[g][1]-rmax0);
            const float w8=__expf(sf[g][4]-rmax0), w9=__expf(sf[g][5]-rmax0);
            lsum0 += w0+w1+w8+w9;
            s_scores[qr0*ss+kc]=__float2half(w0); s_scores[qr0*ss+kc+1]=__float2half(w1);
            s_scores[qr0*ss+kc+8]=__float2half(w8); s_scores[qr0*ss+kc+9]=__float2half(w9);
            const float u0=__expf(sf[g][2]-rmax1), u1=__expf(sf[g][3]-rmax1);
            const float u8=__expf(sf[g][6]-rmax1), u9=__expf(sf[g][7]-rmax1);
            lsum1 += u0+u1+u8+u9;
            s_scores[qr1*ss+kc]=__float2half(u0); s_scores[qr1*ss+kc+1]=__float2half(u1);
            s_scores[qr1*ss+kc+8]=__float2half(u8); s_scores[qr1*ss+kc+9]=__float2half(u9);
        }
#endif
        lsum0 += __shfl_xor_sync(0xffffffff, lsum0, 1);
        lsum0 += __shfl_xor_sync(0xffffffff, lsum0, 2);
        lsum1 += __shfl_xor_sync(0xffffffff, lsum1, 1);
        lsum1 += __shfl_xor_sync(0xffffffff, lsum1, 2);
        rsum0 += lsum0;  rsum1 += lsum1;

        // attn × V (REGPV fuses P@V into the weights loop above).
#if !INT8_ATTN_REGPV
#if INT8_ATTN_VPREFETCH
        __syncthreads();  // s_V_fp16 was written block-wide after QK^T
#else
        __syncwarp();     // s_scores: warp-internal write → fragment read
#endif
        // k outer / s inner: each scores fragment is loaded once per k-step
        // and reused across all head_dim slices (was reloaded n_slices times).
        #pragma unroll
        for (int k = 0; k < TILE_KV; k += ATTN_WMMA_K) {
            fragment<matrix_a, ATTN_WMMA_M, ATTN_WMMA_N, ATTN_WMMA_K, half, row_major> fw;
            load_matrix_sync(fw, s_scores + warp_id * ATTN_WMMA_M * ss + k, ss);
            #pragma unroll
            for (int s = 0; s < n_slices; ++s) {
                fragment<matrix_b, ATTN_WMMA_M, ATTN_WMMA_N, ATTN_WMMA_K, half, row_major> fv;
                load_matrix_sync(fv, s_V_fp16 + k * vs + s * ATTN_WMMA_N, vs);
                mma_sync(frag_out[s], fw, fv, frag_out[s]);
            }
        }
#endif

        __syncthreads();
    }

    // Normalize and write FP16 output.
    const float inv0 = 1.f / rsum0, inv1 = 1.f / rsum1;
    #pragma unroll
    for (int s = 0; s < n_slices; ++s) {
        const int d    = s * ATTN_WMMA_N;
        const int qi0  = qi_base + warp_id * ATTN_WMMA_M + frow0;
        const int qi1  = qi_base + warp_id * ATTN_WMMA_M + frow1;
        const size_t b0 = ((size_t)bh * seq_len + qi0) * head_dim;
        const size_t b1 = ((size_t)bh * seq_len + qi1) * head_dim;
        if (qi0 < seq_len) {
            out[b0+d+fcol_lo]   = __float2half(FRAG_OUT(s,0)*inv0);
            out[b0+d+fcol_lo+1] = __float2half(FRAG_OUT(s,1)*inv0);
            out[b0+d+fcol_lo+8] = __float2half(FRAG_OUT(s,4)*inv0);
            out[b0+d+fcol_lo+9] = __float2half(FRAG_OUT(s,5)*inv0);
        }
        if (qi1 < seq_len) {
            out[b1+d+fcol_lo]   = __float2half(FRAG_OUT(s,2)*inv1);
            out[b1+d+fcol_lo+1] = __float2half(FRAG_OUT(s,3)*inv1);
            out[b1+d+fcol_lo+8] = __float2half(FRAG_OUT(s,6)*inv1);
            out[b1+d+fcol_lo+9] = __float2half(FRAG_OUT(s,7)*inv1);
        }
    }
#undef FRAG_OUT
}

// ── WMMA path with cp.async double buffering ──────────────────────────
// While tile t is being processed, tile t+1's K and V load asynchronously
// (global → smem, bypassing registers). K stays INT8 in smem for the INT8
// QK^T WMMA. V is staged as INT8, then dequantized to an FP16 buffer after
// the QK^T MMAs of the same tile (a cheap smem→smem pass), so the slow
// synchronous global load + dequant of the single-buffer kernel disappears
// from the critical path.
//
// Smem (dynamic):
//   s_Q_i8     [TILE_Q  * ks_i8]        INT8
//   s_K_i8     [2][TILE_KV * ks_i8]     INT8, double-buffered (cp.async)
//   s_V_i8     [2][TILE_KV * ks_i8]     INT8, double-buffered (cp.async)
//   s_V_fp16   [TILE_KV * vs]           FP16, rebuilt per tile
//   s_scores   [TILE_Q  * ss]           FP16
//   s_scale_K  [TILE_KV]                float, reloaded per tile
//
// TILE_KV=64 fits 2 blocks/SM up to head_dim=128 (73 KB); TILE_KV=32
// covers head_dim up to 256 (74 KB). Larger head_dims fall back.
template <int TILE_KV>
__global__ __launch_bounds__(ATTN_BDIM, 2)
void int8_wmma_attention_db_kernel(
    const int8_t* __restrict__ Q,
    const int8_t* __restrict__ K,
    const int8_t* __restrict__ V,
    half* __restrict__         out,
    const float* __restrict__ d_scale_Q,
    const float* __restrict__ d_scale_K,
    const float* __restrict__ d_scale_V,
    int seq_len, int head_dim,
    float attn_scale
) {
    const int bh      = blockIdx.y;
    const int qi_base = blockIdx.x * ATTN_TILE_Q;
    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane    = tid % 32;

    extern __shared__ char smem_raw[];
    const int ks_i8 = head_dim + ATTN_I8_PAD;
    const int vs    = head_dim + ATTN_SMEM_PAD;
    const int ss    = TILE_KV + ATTN_SMEM_PAD;

    int8_t* s_Q_i8   = (int8_t*) smem_raw;
    int8_t* s_K_i8_0 = s_Q_i8 + ATTN_TILE_Q * ks_i8;
    int8_t* s_K_i8_1 = s_K_i8_0 + TILE_KV * ks_i8;
    int8_t* s_V_i8_0 = s_K_i8_1 + TILE_KV * ks_i8;
    int8_t* s_V_i8_1 = s_V_i8_0 + TILE_KV * ks_i8;
    half*   s_V_fp16 = (half*)(s_V_i8_1 + TILE_KV * ks_i8);
    half*   s_scores = s_V_fp16 + TILE_KV * vs;
    float*  s_scale_K = (float*)(s_scores + ATTN_TILE_Q * ss);

    // Load Q tile as INT8 (vectorized 16-byte loads).
    {
        const int q_vecs = ATTN_TILE_Q * head_dim / 16;
        const int4 zero4 = make_int4(0, 0, 0, 0);
        for (int vi = tid; vi < q_vecs; vi += ATTN_BDIM) {
            const int byte_off = vi * 16;
            const int row = byte_off / head_dim;
            const int col = byte_off % head_dim;
            const int qi  = qi_base + row;
            int4 val = (qi < seq_len)
                ? *reinterpret_cast<const int4*>(&Q[((size_t)bh * seq_len + qi) * head_dim + col])
                : zero4;
            *reinterpret_cast<int4*>(&s_Q_i8[row * ks_i8 + col]) = val;
        }
    }

    const int frow0   = lane / 4;
    const int frow1   = lane / 4 + 8;
    const int fcol_lo = (lane % 4) * 2;

    const int qi0_global = qi_base + warp_id * ATTN_WMMA_M + frow0;
    const int qi1_global = qi_base + warp_id * ATTN_WMMA_M + frow1;
    const float r_sQ0 = (qi0_global < seq_len)
        ? __ldg(&d_scale_Q[bh * seq_len + qi0_global]) * attn_scale : 0.f;
    const float r_sQ1 = (qi1_global < seq_len)
        ? __ldg(&d_scale_Q[bh * seq_len + qi1_global]) * attn_scale : 0.f;

    float rmax0 = -1e38f, rmax1 = -1e38f;
    float rsum0 = 0.f,    rsum1 = 0.f;

    const int n_slices    = head_dim / ATTN_WMMA_N;
    const int n_kv_groups = TILE_KV / ATTN_WMMA_N;
    const int kv_vecs     = TILE_KV * head_dim / 16;

    fragment<accumulator, ATTN_WMMA_M, ATTN_WMMA_N, ATTN_WMMA_K, float> frag_out[ATTN_MAX_SLICES];
    for (int s = 0; s < n_slices; ++s) fill_fragment(frag_out[s], 0.f);

    const int num_tiles = seq_len / TILE_KV;

    // Kick off tile 0's K+V loads before entering the loop.
    {
        const size_t base = ((size_t)bh * seq_len) * head_dim;
        for (int vi = tid; vi < kv_vecs; vi += ATTN_BDIM) {
            const int byte_off = vi * 16;
            const int row = byte_off / head_dim;
            const int col = byte_off % head_dim;
            i8_cp_async_16B(s_K_i8_0 + row * ks_i8 + col, K + base + row * head_dim + col);
            i8_cp_async_16B(s_V_i8_0 + row * ks_i8 + col, V + base + row * head_dim + col);
        }
        i8_cp_async_commit();
    }

    __syncthreads();  // Q visible (cp.async groups handled separately)

    for (int t = 0; t < num_tiles; ++t) {
        const int ts = t * TILE_KV;
        const size_t scale_base = (size_t)bh * seq_len + ts;

        int8_t* s_K_cur = (t % 2 == 0) ? s_K_i8_0 : s_K_i8_1;
        int8_t* s_V_cur = (t % 2 == 0) ? s_V_i8_0 : s_V_i8_1;
        int8_t* s_K_nxt = (t % 2 == 0) ? s_K_i8_1 : s_K_i8_0;
        int8_t* s_V_nxt = (t % 2 == 0) ? s_V_i8_1 : s_V_i8_0;

        // Issue tile t+1's loads, then block until tile t is resident.
        if (t + 1 < num_tiles) {
            const size_t base = ((size_t)bh * seq_len + ts + TILE_KV) * head_dim;
            for (int vi = tid; vi < kv_vecs; vi += ATTN_BDIM) {
                const int byte_off = vi * 16;
                const int row = byte_off / head_dim;
                const int col = byte_off % head_dim;
                i8_cp_async_16B(s_K_nxt + row * ks_i8 + col, K + base + row * head_dim + col);
                i8_cp_async_16B(s_V_nxt + row * ks_i8 + col, V + base + row * head_dim + col);
            }
            i8_cp_async_commit();
            i8_cp_async_wait_one();   // tile t done; tile t+1 stays in flight
        } else {
            i8_cp_async_wait_all();
        }
        // K scales for tile t (tiny, L2-resident).
        for (int idx = tid; idx < TILE_KV; idx += ATTN_BDIM)
            s_scale_K[idx] = __ldg(&d_scale_K[scale_base + idx]);
        __syncthreads();  // [1/3] tile t's K/V (INT8) + scales visible

        // ── QK^T via INT8 WMMA → INT32 accumulator ──
        fragment<accumulator, ATTN_WMMA_M, ATTN_WMMA_N, ATTN_WMMA_K, int32_t>
            frag_qk[TILE_KV / ATTN_WMMA_N];
        #pragma unroll
        for (int g = 0; g < n_kv_groups; ++g) {
            fill_fragment(frag_qk[g], (int32_t)0);
            #pragma unroll 4
            for (int k = 0; k < head_dim; k += ATTN_WMMA_K) {
                fragment<matrix_a, ATTN_WMMA_M, ATTN_WMMA_N, ATTN_WMMA_K, int8_t, row_major> fq;
                fragment<matrix_b, ATTN_WMMA_M, ATTN_WMMA_N, ATTN_WMMA_K, int8_t, col_major> fk;
                load_matrix_sync(fq, s_Q_i8 + warp_id * ATTN_WMMA_M * ks_i8 + k, ks_i8);
                load_matrix_sync(fk, s_K_cur + g * ATTN_WMMA_N * ks_i8 + k,      ks_i8);
                mma_sync(frag_qk[g], fq, fk, frag_qk[g]);
            }
        }

        // ── Dequantize V: s_V_cur (INT8) → s_V_fp16 ──
        // K tile is consumed by the MMAs above; V_fp16 from the previous
        // tile was released at the loop-end barrier.
        for (int vi = tid; vi < kv_vecs; vi += ATTN_BDIM) {
            const int byte_off = vi * 16;
            const int row = byte_off / head_dim;
            const int col = byte_off % head_dim;
            const float sv = __ldg(&d_scale_V[scale_base + row]);
            int4 v_vec = *reinterpret_cast<const int4*>(&s_V_cur[row * ks_i8 + col]);
            const int8_t* vb = reinterpret_cast<const int8_t*>(&v_vec);
            half* vdst = &s_V_fp16[row * vs + col];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                reinterpret_cast<half2*>(vdst)[i] = __halves2half2(
                    __float2half((float)vb[2*i]     * sv),
                    __float2half((float)vb[2*i + 1] * sv));
            }
        }

        // Cast INT32 accumulators to float with per-element scaling.
        float sf[TILE_KV / ATTN_WMMA_N][8];
        float lmax0 = -1e38f, lmax1 = -1e38f;
        for (int g = 0; g < n_kv_groups; ++g) {
            const int kc_base = g * ATTN_WMMA_N;
            float sK_c0 = s_scale_K[kc_base + fcol_lo];
            float sK_c1 = s_scale_K[kc_base + fcol_lo + 1];
            float sK_c8 = s_scale_K[kc_base + fcol_lo + 8];
            float sK_c9 = s_scale_K[kc_base + fcol_lo + 9];

            sf[g][0] = (float)frag_qk[g].x[0] * r_sQ0 * sK_c0;
            sf[g][1] = (float)frag_qk[g].x[1] * r_sQ0 * sK_c1;
            sf[g][2] = (float)frag_qk[g].x[2] * r_sQ1 * sK_c0;
            sf[g][3] = (float)frag_qk[g].x[3] * r_sQ1 * sK_c1;
            sf[g][4] = (float)frag_qk[g].x[4] * r_sQ0 * sK_c8;
            sf[g][5] = (float)frag_qk[g].x[5] * r_sQ0 * sK_c9;
            sf[g][6] = (float)frag_qk[g].x[6] * r_sQ1 * sK_c8;
            sf[g][7] = (float)frag_qk[g].x[7] * r_sQ1 * sK_c9;

            lmax0 = fmaxf(lmax0, fmaxf(fmaxf(sf[g][0],sf[g][1]), fmaxf(sf[g][4],sf[g][5])));
            lmax1 = fmaxf(lmax1, fmaxf(fmaxf(sf[g][2],sf[g][3]), fmaxf(sf[g][6],sf[g][7])));
        }
        lmax0 = fmaxf(lmax0, __shfl_xor_sync(0xffffffff, lmax0, 1));
        lmax0 = fmaxf(lmax0, __shfl_xor_sync(0xffffffff, lmax0, 2));
        lmax1 = fmaxf(lmax1, __shfl_xor_sync(0xffffffff, lmax1, 1));
        lmax1 = fmaxf(lmax1, __shfl_xor_sync(0xffffffff, lmax1, 2));

        const float new_max0 = fmaxf(rmax0, lmax0), corr0 = __expf(rmax0 - new_max0);
        const float new_max1 = fmaxf(rmax1, lmax1), corr1 = __expf(rmax1 - new_max1);
        rmax0 = new_max0; rmax1 = new_max1;
        rsum0 *= corr0;   rsum1 *= corr1;
        #pragma unroll
        for (int s = 0; s < n_slices; ++s) {
            frag_out[s].x[0]*=corr0; frag_out[s].x[1]*=corr0;
            frag_out[s].x[2]*=corr1; frag_out[s].x[3]*=corr1;
            frag_out[s].x[4]*=corr0; frag_out[s].x[5]*=corr0;
            frag_out[s].x[6]*=corr1; frag_out[s].x[7]*=corr1;
        }

        // Softmax weights → s_scores (FP16, for attn × V WMMA).
        float lsum0 = 0.f, lsum1 = 0.f;
        for (int g = 0; g < n_kv_groups; ++g) {
            const int qr0 = warp_id * ATTN_WMMA_M + frow0;
            const int qr1 = warp_id * ATTN_WMMA_M + frow1;
            const int kc  = g * ATTN_WMMA_N + fcol_lo;
            const float w0=__expf(sf[g][0]-rmax0), w1=__expf(sf[g][1]-rmax0);
            const float w8=__expf(sf[g][4]-rmax0), w9=__expf(sf[g][5]-rmax0);
            lsum0 += w0+w1+w8+w9;
            s_scores[qr0*ss+kc]=__float2half(w0); s_scores[qr0*ss+kc+1]=__float2half(w1);
            s_scores[qr0*ss+kc+8]=__float2half(w8); s_scores[qr0*ss+kc+9]=__float2half(w9);
            const float u0=__expf(sf[g][2]-rmax1), u1=__expf(sf[g][3]-rmax1);
            const float u8=__expf(sf[g][6]-rmax1), u9=__expf(sf[g][7]-rmax1);
            lsum1 += u0+u1+u8+u9;
            s_scores[qr1*ss+kc]=__float2half(u0); s_scores[qr1*ss+kc+1]=__float2half(u1);
            s_scores[qr1*ss+kc+8]=__float2half(u8); s_scores[qr1*ss+kc+9]=__float2half(u9);
        }
        lsum0 += __shfl_xor_sync(0xffffffff, lsum0, 1);
        lsum0 += __shfl_xor_sync(0xffffffff, lsum0, 2);
        lsum1 += __shfl_xor_sync(0xffffffff, lsum1, 1);
        lsum1 += __shfl_xor_sync(0xffffffff, lsum1, 2);
        rsum0 += lsum0;  rsum1 += lsum1;

        __syncthreads();  // [2/3] s_V_fp16 fully written (block-wide)

        // attn × V via FP16 WMMA.
        for (int s = 0; s < n_slices; ++s) {
            const int d_base = s * ATTN_WMMA_N;
            #pragma unroll
            for (int k = 0; k < TILE_KV; k += ATTN_WMMA_K) {
                fragment<matrix_a, ATTN_WMMA_M, ATTN_WMMA_N, ATTN_WMMA_K, half, row_major> fw;
                fragment<matrix_b, ATTN_WMMA_M, ATTN_WMMA_N, ATTN_WMMA_K, half, row_major> fv;
                load_matrix_sync(fw, s_scores + warp_id * ATTN_WMMA_M * ss + k, ss);
                load_matrix_sync(fv, s_V_fp16 + k * vs + d_base,               vs);
                mma_sync(frag_out[s], fw, fv, frag_out[s]);
            }
        }

        __syncthreads();  // [3/3] s_V_fp16 / s_scores reusable next tile
    }

    // Normalize and write FP16 output.
    const float inv0 = 1.f / rsum0, inv1 = 1.f / rsum1;
    #pragma unroll
    for (int s = 0; s < n_slices; ++s) {
        const int d    = s * ATTN_WMMA_N;
        const int qi0  = qi_base + warp_id * ATTN_WMMA_M + frow0;
        const int qi1  = qi_base + warp_id * ATTN_WMMA_M + frow1;
        const size_t b0 = ((size_t)bh * seq_len + qi0) * head_dim;
        const size_t b1 = ((size_t)bh * seq_len + qi1) * head_dim;
        if (qi0 < seq_len) {
            out[b0+d+fcol_lo]   = __float2half(frag_out[s].x[0]*inv0);
            out[b0+d+fcol_lo+1] = __float2half(frag_out[s].x[1]*inv0);
            out[b0+d+fcol_lo+8] = __float2half(frag_out[s].x[4]*inv0);
            out[b0+d+fcol_lo+9] = __float2half(frag_out[s].x[5]*inv0);
        }
        if (qi1 < seq_len) {
            out[b1+d+fcol_lo]   = __float2half(frag_out[s].x[2]*inv1);
            out[b1+d+fcol_lo+1] = __float2half(frag_out[s].x[3]*inv1);
            out[b1+d+fcol_lo+8] = __float2half(frag_out[s].x[6]*inv1);
            out[b1+d+fcol_lo+9] = __float2half(frag_out[s].x[7]*inv1);
        }
    }
}

#endif  // INT8_ATTN_WMMA

// ── Public interface ───────────────────────────────────────────────────
// scale_Q/K/V are device arrays of size [batch*heads*seq_len] — per-token scales.
// attn_scale = 1/sqrt(head_dim) is host-computable.
// Output is FP16 (half*).
void int8_attention_forward(
    const int8_t* Q, const int8_t* K, const int8_t* V, half* out,
    const float* scale_Q, const float* scale_K, const float* scale_V,
    int batch, int heads, int seq_len, int head_dim
) {
    const float attn_scale = 1.f / sqrtf((float)head_dim);

#if INT8_ATTN_WMMA
    if (head_dim % ATTN_WMMA_K == 0) {
        const int BH = batch * heads;
        dim3 grid((seq_len + ATTN_TILE_Q - 1) / ATTN_TILE_Q, BH);
        const int ks_i8_host = head_dim + ATTN_I8_PAD;
        const int vs_host    = head_dim + ATTN_SMEM_PAD;

#if INT8_ATTN_DB
        // Double-buffered path: Q + 2×K(INT8) + 2×V(INT8) + V(FP16) + scores + K scales.
        auto calc_smem_db = [&](int tile_kv) -> size_t {
            const int ss = tile_kv + ATTN_SMEM_PAD;
            return (size_t)ATTN_TILE_Q * ks_i8_host +
                   (size_t)2 * tile_kv * ks_i8_host * 2 +          // K + V INT8, ×2 buffers
                   (size_t)tile_kv * vs_host * sizeof(half) +
                   (size_t)ATTN_TILE_Q * ss * sizeof(half) +
                   (size_t)tile_kv * sizeof(float);
        };
        // TILE_KV=64 → 73 KB at head_dim=128 (2 blocks/SM);
        // TILE_KV=32 → 74 KB at head_dim=256.
        const size_t smem_db64 = calc_smem_db(64);
        if (seq_len % 64 == 0 && smem_db64 <= 82000) {
            cudaFuncSetAttribute(int8_wmma_attention_db_kernel<64>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_db64);
            int8_wmma_attention_db_kernel<64><<<grid, ATTN_BDIM, smem_db64>>>(
                Q, K, V, out, scale_Q, scale_K, scale_V,
                seq_len, head_dim, attn_scale);
            return;
        }
        const size_t smem_db32 = calc_smem_db(32);
        if (seq_len % 32 == 0 && smem_db32 <= 82000) {
            cudaFuncSetAttribute(int8_wmma_attention_db_kernel<32>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_db32);
            int8_wmma_attention_db_kernel<32><<<grid, ATTN_BDIM, smem_db32>>>(
                Q, K, V, out, scale_Q, scale_K, scale_V,
                seq_len, head_dim, attn_scale);
            return;
        }
#endif  // INT8_ATTN_DB

        // Helper: compute smem for a given tile_kv
        auto calc_smem = [&](int tile_kv) -> size_t {
            size_t sz = (size_t)ATTN_TILE_Q * ks_i8_host +         // Q (INT8)
                   (size_t)tile_kv * ks_i8_host +                  // K (INT8)
                   (size_t)tile_kv * vs_host * sizeof(half) +      // V (FP16)
                   (size_t)tile_kv * sizeof(float);                // K_scales
#if !INT8_ATTN_REGPV
            sz += (size_t)ATTN_TILE_Q *
                  (tile_kv + ATTN_SMEM_PAD) * sizeof(half);        // scores (FP16)
#endif
            return sz;
        };

        // The kernel is templated on (TILE_KV, HEAD_DIM); head dims outside
        // {64, 128, 256} fall through to the scalar path.
        //
        // TILE_KV=128 first (halves tile count → less loop overhead), when
        // smem ≤ 82KB (2 blocks/SM). head_dim=256 needs 135KB → too much
        // register pressure (frag_qk[8]+sf[8][8]+frag_out[16] ≈ 276 regs)
        // and 1 block/SM kills latency hiding → TILE_KV=64 for head_dim=256.
#define LAUNCH_I8_WMMA_ATTN(KV, HD, SMEM)                                     \
        do {                                                                  \
            cudaFuncSetAttribute(int8_wmma_attention_kernel<KV, HD>,          \
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                 (int)(SMEM));                                \
            int8_wmma_attention_kernel<KV, HD><<<grid, ATTN_BDIM, SMEM>>>(    \
                Q, K, V, out, scale_Q, scale_K, scale_V,                      \
                seq_len, attn_scale);                                         \
        } while (0)

        const size_t smem_128 = calc_smem(128);
        if (seq_len % 128 == 0 && smem_128 <= 82000) {
            if (head_dim == 64)  { LAUNCH_I8_WMMA_ATTN(128, 64,  smem_128); return; }
            if (head_dim == 128) { LAUNCH_I8_WMMA_ATTN(128, 128, smem_128); return; }
        }
        if (seq_len % 64 == 0) {
            const size_t smem_64 = calc_smem(64);
            if (head_dim == 64)  { LAUNCH_I8_WMMA_ATTN(64, 64,  smem_64); return; }
            if (head_dim == 128) { LAUNCH_I8_WMMA_ATTN(64, 128, smem_64); return; }
            if (head_dim == 256) { LAUNCH_I8_WMMA_ATTN(64, 256, smem_64); return; }
        }
#undef LAUNCH_I8_WMMA_ATTN
    }
#endif

    const size_t smem =
        ((size_t)head_dim + (size_t)SCALAR_TILE_KV * head_dim * 2 +
         SCALAR_TILE_KV + head_dim + SCALAR_BDIM) * sizeof(float);
    cudaFuncSetAttribute(int8_fused_attention_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 grid(seq_len, batch * heads);
    int8_fused_attention_kernel<<<grid, SCALAR_BDIM, smem>>>(
        Q, K, V, out, scale_Q, scale_K, scale_V,
        seq_len, head_dim, attn_scale);
}
