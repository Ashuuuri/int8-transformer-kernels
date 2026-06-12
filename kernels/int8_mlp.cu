// int8_mlp.cu — INT8 quantized two-layer MLP (Linear → GELU → Linear).
//
// Architecture:
//   128×128 block tile, 8 warps, 16×16×16 WMMA, cp.async double-buffer (STAGE_K=32).
//   Each warp computes a 32×64 output sub-tile (2 row tiles × 4 col tiles).
//   Arithmetic intensity: 2×128×128 / ((128+128)×1) = 128 ops/byte (was 64).
//
//   Forward pass:
//     GEMM1: int8 x × int8 W1  →  INT32 acc × (scale_x·scale_W1) → GELU → INT8 hidden
//     GEMM2: int8 hidden × int8 W2  →  INT32 acc × (scale_hidden·scale_W2) → INT8 out
//
// Ablation flags:
//   INT8_MLP_STAGE_K=N   change K-stage depth (default 32)

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include "int8_common.cuh"

using namespace nvcuda;

// ── Tile / block shape ─────────────────────────────────────────────────
#define WMMA_M  16
#define WMMA_N  16
#define WMMA_K  16
#ifndef INT8_MLP_STAGE_K
#define INT8_MLP_STAGE_K 32
#endif
#define STAGE_K          INT8_MLP_STAGE_K
#define BLOCK_M          128                                    // was 64
#define BLOCK_N          128                                    // was 64
#define WARP_ROW_TILES   2                                      // each warp: 2 row tiles (32 rows)
#define WARP_COL_TILES   4                                      // each warp: 4 col tiles (64 cols)
#define WARP_ROW_GROUPS  (BLOCK_M / WMMA_M / WARP_ROW_TILES)   // 4
#define WARP_COL_GROUPS  (BLOCK_N / WMMA_N / WARP_COL_TILES)   // 2
#define WARPS_PER_BLOCK  (WARP_ROW_GROUPS * WARP_COL_GROUPS)   // 8
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * 32)                // 256
#define SCALAR_TILE      16

// INT8 smem padding: 16 bytes per row
#define SMEM_SKEW        16
#define A_SMEM_STRIDE    (STAGE_K + SMEM_SKEW)                  // 48
#define B_SMEM_STRIDE    (BLOCK_N + SMEM_SKEW)                  // 144

// cp.async: 16 bytes per copy = 16 INT8 elements
#define A_COPIES_PER_ROW (STAGE_K / 16)                         // 2
#define B_COPIES_PER_ROW (BLOCK_N / 16)                         // 8

// ── Scalar fallback ────────────────────────────────────────────────────
// Accumulates INT8 products as float, writes FP16 output.
template <bool apply_gelu>
__global__ void gemm_int8_scalar_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    half* __restrict__         C,
    int M, int N, int K,
    const float* __restrict__ d_scale_A,
    const float* __restrict__ d_scale_B
) {
    const float scale = __ldg(d_scale_A) * __ldg(d_scale_B);
    __shared__ float sA[SCALAR_TILE][SCALAR_TILE];
    __shared__ float sB[SCALAR_TILE][SCALAR_TILE];

    const int row = blockIdx.y * SCALAR_TILE + threadIdx.y;
    const int col = blockIdx.x * SCALAR_TILE + threadIdx.x;
    float acc = 0.f;

    for (int t = 0; t < (K + SCALAR_TILE - 1) / SCALAR_TILE; ++t) {
        const int kA = t * SCALAR_TILE + threadIdx.x;
        const int kB = t * SCALAR_TILE + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < M && kA < K) ? (float)A[row*K + kA] : 0.f;
        sB[threadIdx.y][threadIdx.x] = (kB  < K && col < N) ? (float)B[kB *N + col] : 0.f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < SCALAR_TILE; ++k) acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) {
        float v = acc * scale;
        if (apply_gelu) v = int8_gelu(v);
        C[row * N + col] = __float2half_rn(v);
    }
}

// ── Async tile loader (INT8 double-buffer) ─────────────────────────────
// Loads one BLOCK_M×STAGE_K slab of A and one STAGE_K×BLOCK_N slab of B
// into shared memory via cp.async.  Only called when dimensions are multiples
// of the tile sizes (enforced by the use_wmma check in int8_mlp_forward).
__device__ __forceinline__ void load_int8_tile_async(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int8_t* sA, int8_t* sB,
    int block_m, int block_n, int k0, int K, int N
) {
    const int A_COPIES = BLOCK_M * A_COPIES_PER_ROW;    // 256
    const int B_COPIES = STAGE_K * B_COPIES_PER_ROW;    // 256
    const int TOTAL    = A_COPIES + B_COPIES;            // 512

    for (int c = threadIdx.x; c < TOTAL; c += THREADS_PER_BLOCK) {
        if (c < A_COPIES) {
            const int row = c / A_COPIES_PER_ROW;
            const int col = (c % A_COPIES_PER_ROW) * 16;
            i8_cp_async_16B(sA + row * A_SMEM_STRIDE + col,
                            A  + (block_m + row) * K + k0 + col);
        } else {
            const int bc  = c - A_COPIES;
            const int row = bc / B_COPIES_PER_ROW;
            const int col = (bc % B_COPIES_PER_ROW) * 16;
            i8_cp_async_16B(sB + row * B_SMEM_STRIDE + col,
                            B  + (k0 + row) * N + block_n + col);
        }
    }
}

// ── INT8 WMMA GEMM → FP16 output ───────────────────────────────────────
// C (FP16) = dequant( A (INT8) × B (INT8) )
//   real[i,j] = int32_acc[i,j] * scale_A * scale_B   [+ GELU]
template <bool apply_gelu>
__global__ void gemm_int8_wmma_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    half* __restrict__         C,
    int M, int N, int K,
    const float* __restrict__ d_scale_A,
    const float* __restrict__ d_scale_B
) {
    const float scale = __ldg(d_scale_A) * __ldg(d_scale_B);

    // sA/sB (K-loop) and c_smem (epilogue) are temporally disjoint — overlap them.
    // K-loop: 2×(128×48) + 2×(32×144) = 21504 B
    // Epilogue: 8×4×256×4              = 32768 B
    // Overlapped: max(21504, 32768)    = 32768 B  < 48KB static limit
    constexpr int SMEM_AB   = 2 * BLOCK_M * A_SMEM_STRIDE + 2 * STAGE_K * B_SMEM_STRIDE;
    constexpr int SMEM_EPIL = WARPS_PER_BLOCK * WARP_COL_TILES * WMMA_M * WMMA_N * (int)sizeof(int32_t);
    constexpr int SMEM_SIZE = SMEM_AB > SMEM_EPIL ? SMEM_AB : SMEM_EPIL;
    __shared__ __align__(16) char _smem[SMEM_SIZE];

    int8_t* sA[2] = {(int8_t*)_smem,
                      (int8_t*)_smem + BLOCK_M * A_SMEM_STRIDE};
    int8_t* sB[2] = {(int8_t*)_smem + 2 * BLOCK_M * A_SMEM_STRIDE,
                      (int8_t*)_smem + 2 * BLOCK_M * A_SMEM_STRIDE + STAGE_K * B_SMEM_STRIDE};

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warp_m  = warp_id / WARP_COL_GROUPS;   // 0..3
    const int warp_ng = warp_id % WARP_COL_GROUPS;   // 0..1

    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;

    // Fragments: 2 row tiles × 4 col tiles per warp = 8 MMAs per K-step
    wmma::fragment<wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> a_frag[WARP_ROW_TILES];
    wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> b_frag[WARP_COL_TILES];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> acc[WARP_ROW_TILES][WARP_COL_TILES];

    #pragma unroll
    for (int rt = 0; rt < WARP_ROW_TILES; rt++)
        #pragma unroll
        for (int ct = 0; ct < WARP_COL_TILES; ct++)
            wmma::fill_fragment(acc[rt][ct], (int32_t)0);

    // Double-buffered K-loop
    int stage = 0;
    load_int8_tile_async(A, B, sA[0], sB[0], block_m, block_n, 0, K, N);
    i8_cp_async_commit();

    for (int k0 = 0; k0 < K; k0 += STAGE_K) {
        i8_cp_async_wait_all();
        __syncthreads();

        const int next_k0 = k0 + STAGE_K;
        const int next_s  = stage ^ 1;
        if (next_k0 < K) {
            load_int8_tile_async(A, B, sA[next_s], sB[next_s],
                                 block_m, block_n, next_k0, K, N);
            i8_cp_async_commit();
        }

        #pragma unroll
        for (int kk = 0; kk < STAGE_K; kk += WMMA_K) {
            #pragma unroll
            for (int rt = 0; rt < WARP_ROW_TILES; rt++) {
                const int8_t* a_ptr = sA[stage]
                    + (warp_m * WARP_ROW_TILES + rt) * WMMA_M * A_SMEM_STRIDE + kk;
                wmma::load_matrix_sync(a_frag[rt], a_ptr, A_SMEM_STRIDE);
            }
            #pragma unroll
            for (int ct = 0; ct < WARP_COL_TILES; ct++) {
                const int8_t* b_ptr = sB[stage]
                    + kk * B_SMEM_STRIDE + (warp_ng * WARP_COL_TILES + ct) * WMMA_N;
                wmma::load_matrix_sync(b_frag[ct], b_ptr, B_SMEM_STRIDE);
            }
            #pragma unroll
            for (int rt = 0; rt < WARP_ROW_TILES; rt++)
                #pragma unroll
                for (int ct = 0; ct < WARP_COL_TILES; ct++)
                    wmma::mma_sync(acc[rt][ct], a_frag[rt], b_frag[ct], acc[rt][ct]);
        }

        __syncthreads();
        stage = next_s;
    }

    // Epilogue: INT32 → float → ×scale → [GELU] → FP16
    // sA/sB no longer needed — repurpose _smem as c_smem
    int32_t* c_smem = reinterpret_cast<int32_t*>(_smem);
    int32_t* c_base = c_smem + warp_id * WARP_COL_TILES * WMMA_M * WMMA_N;

    #pragma unroll
    for (int rt = 0; rt < WARP_ROW_TILES; rt++) {
        #pragma unroll
        for (int ct = 0; ct < WARP_COL_TILES; ct++)
            wmma::store_matrix_sync(c_base + ct * WMMA_M * WMMA_N,
                                    acc[rt][ct], WMMA_N, wmma::mem_row_major);
        __syncwarp();

        const int row_base = block_m + (warp_m * WARP_ROW_TILES + rt) * WMMA_M;
        const int col_base = block_n + warp_ng * WARP_COL_TILES * WMMA_N;

        #pragma unroll
        for (int ct = 0; ct < WARP_COL_TILES; ct++) {
            int32_t* cp = c_base + ct * WMMA_M * WMMA_N;
            const int tile_col = col_base + ct * WMMA_N;
            #pragma unroll
            for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
                const int r = i / WMMA_N, c = i % WMMA_N;
                float v = (float)cp[i] * scale;
                if (apply_gelu) v = int8_gelu(v);
                C[(row_base + r) * N + (tile_col + c)] = __float2half_rn(v);
            }
        }
    }
}

// ── Static scale computation (1 thread, runs once per forward call) ───
// scale_hidden = sx * sw1 * d_model  (conservative per-tensor static bound)
// scale_out    = scale_hidden * sw2 * d_ff
__global__ void compute_static_scales_kernel(
    const float* __restrict__ sx,
    const float* __restrict__ sw1,
    const float* __restrict__ sw2,
    float* d_scale_hidden, float* d_inv_scale_hidden,
    float* d_scale_out,    float* d_inv_scale_out,
    int d_model, int d_ff
) {
    const float sh = __ldg(sx) * __ldg(sw1) * (float)d_model;
    const float so = sh * __ldg(sw2) * (float)d_ff;
    *d_scale_hidden     = sh;
    *d_inv_scale_hidden = 1.f / sh;
    *d_scale_out        = so;
    *d_inv_scale_out    = 1.f / so;
}

// ── INT8 WMMA GEMM → INT8 output ──────────────────────────────────────
// Same as gemm_int8_wmma_kernel but epilogue writes INT8 using a static scale.
template <bool apply_gelu>
__global__ void gemm_int8_wmma_i8_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int8_t* __restrict__       C,
    int M, int N, int K,
    const float* __restrict__ d_scale_A,
    const float* __restrict__ d_scale_B,
    const float* __restrict__ d_inv_scale_out
) {
    const float scale     = __ldg(d_scale_A) * __ldg(d_scale_B);
    const float inv_scale = __ldg(d_inv_scale_out);

    // sA/sB (K-loop) and c_smem (epilogue) are temporally disjoint — overlap them.
    constexpr int SMEM_AB   = 2 * BLOCK_M * A_SMEM_STRIDE + 2 * STAGE_K * B_SMEM_STRIDE;
    constexpr int SMEM_EPIL = WARPS_PER_BLOCK * WARP_COL_TILES * WMMA_M * WMMA_N * (int)sizeof(int32_t);
    constexpr int SMEM_SIZE = SMEM_AB > SMEM_EPIL ? SMEM_AB : SMEM_EPIL;
    __shared__ __align__(16) char _smem[SMEM_SIZE];

    int8_t* sA[2] = {(int8_t*)_smem,
                      (int8_t*)_smem + BLOCK_M * A_SMEM_STRIDE};
    int8_t* sB[2] = {(int8_t*)_smem + 2 * BLOCK_M * A_SMEM_STRIDE,
                      (int8_t*)_smem + 2 * BLOCK_M * A_SMEM_STRIDE + STAGE_K * B_SMEM_STRIDE};

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warp_m  = warp_id / WARP_COL_GROUPS;
    const int warp_ng = warp_id % WARP_COL_GROUPS;

    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;

    wmma::fragment<wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> a_frag[WARP_ROW_TILES];
    wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> b_frag[WARP_COL_TILES];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> acc[WARP_ROW_TILES][WARP_COL_TILES];

    #pragma unroll
    for (int rt = 0; rt < WARP_ROW_TILES; rt++)
        #pragma unroll
        for (int ct = 0; ct < WARP_COL_TILES; ct++)
            wmma::fill_fragment(acc[rt][ct], (int32_t)0);

    int stage = 0;
    load_int8_tile_async(A, B, sA[0], sB[0], block_m, block_n, 0, K, N);
    i8_cp_async_commit();

    for (int k0 = 0; k0 < K; k0 += STAGE_K) {
        i8_cp_async_wait_all();
        __syncthreads();
        const int next_k0 = k0 + STAGE_K;
        const int next_s  = stage ^ 1;
        if (next_k0 < K) {
            load_int8_tile_async(A, B, sA[next_s], sB[next_s],
                                 block_m, block_n, next_k0, K, N);
            i8_cp_async_commit();
        }
        #pragma unroll
        for (int kk = 0; kk < STAGE_K; kk += WMMA_K) {
            #pragma unroll
            for (int rt = 0; rt < WARP_ROW_TILES; rt++) {
                const int8_t* a_ptr = sA[stage]
                    + (warp_m * WARP_ROW_TILES + rt) * WMMA_M * A_SMEM_STRIDE + kk;
                wmma::load_matrix_sync(a_frag[rt], a_ptr, A_SMEM_STRIDE);
            }
            #pragma unroll
            for (int ct = 0; ct < WARP_COL_TILES; ct++) {
                const int8_t* b_ptr = sB[stage]
                    + kk * B_SMEM_STRIDE + (warp_ng * WARP_COL_TILES + ct) * WMMA_N;
                wmma::load_matrix_sync(b_frag[ct], b_ptr, B_SMEM_STRIDE);
            }
            #pragma unroll
            for (int rt = 0; rt < WARP_ROW_TILES; rt++)
                #pragma unroll
                for (int ct = 0; ct < WARP_COL_TILES; ct++)
                    wmma::mma_sync(acc[rt][ct], a_frag[rt], b_frag[ct], acc[rt][ct]);
        }
        __syncthreads();
        stage = next_s;
    }

    // Epilogue: INT32 → float → ×scale → [GELU] → INT8
    // sA/sB no longer needed — repurpose _smem as c_smem
    int32_t* c_smem = reinterpret_cast<int32_t*>(_smem);
    int32_t* c_base = c_smem + warp_id * WARP_COL_TILES * WMMA_M * WMMA_N;

    #pragma unroll
    for (int rt = 0; rt < WARP_ROW_TILES; rt++) {
        #pragma unroll
        for (int ct = 0; ct < WARP_COL_TILES; ct++)
            wmma::store_matrix_sync(c_base + ct * WMMA_M * WMMA_N,
                                    acc[rt][ct], WMMA_N, wmma::mem_row_major);
        __syncwarp();

        const int row_base = block_m + (warp_m * WARP_ROW_TILES + rt) * WMMA_M;
        const int col_base = block_n + warp_ng * WARP_COL_TILES * WMMA_N;

        #pragma unroll
        for (int ct = 0; ct < WARP_COL_TILES; ct++) {
            int32_t* cp = c_base + ct * WMMA_M * WMMA_N;
            const int tile_col = col_base + ct * WMMA_N;
            #pragma unroll
            for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
                const int r = i / WMMA_N, c = i % WMMA_N;
                float v = (float)cp[i] * scale;
                if (apply_gelu) v = int8_gelu(v);
                C[(row_base + r) * N + (tile_col + c)] = f32_to_i8(v, inv_scale);
            }
        }
    }
}

// ── Scalar fallback → INT8 output ─────────────────────────────────────
template <bool apply_gelu>
__global__ void gemm_int8_scalar_i8_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int8_t* __restrict__       C,
    int M, int N, int K,
    const float* __restrict__ d_scale_A,
    const float* __restrict__ d_scale_B,
    const float* __restrict__ d_inv_scale_out
) {
    const float scale     = __ldg(d_scale_A) * __ldg(d_scale_B);
    const float inv_scale = __ldg(d_inv_scale_out);
    __shared__ float sA[SCALAR_TILE][SCALAR_TILE];
    __shared__ float sB[SCALAR_TILE][SCALAR_TILE];

    const int row = blockIdx.y * SCALAR_TILE + threadIdx.y;
    const int col = blockIdx.x * SCALAR_TILE + threadIdx.x;
    float acc = 0.f;

    for (int t = 0; t < (K + SCALAR_TILE - 1) / SCALAR_TILE; ++t) {
        const int kA = t * SCALAR_TILE + threadIdx.x;
        const int kB = t * SCALAR_TILE + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < M && kA < K) ? (float)A[row*K + kA] : 0.f;
        sB[threadIdx.y][threadIdx.x] = (kB  < K && col < N) ? (float)B[kB *N + col] : 0.f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < SCALAR_TILE; ++k) acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) {
        float v = acc * scale;
        if (apply_gelu) v = int8_gelu(v);
        C[row * N + col] = f32_to_i8(v, inv_scale);
    }
}

// ── File-scope device buffer cache ────────────────────────────────────
static int8_t* s_hidden_int8      = nullptr;
static float*  s_scale_hidden     = nullptr;
static float*  s_inv_scale_hidden = nullptr;
static float*  s_scale_out        = nullptr;
static float*  s_inv_scale_out    = nullptr;
static size_t  s_hidden_cap       = 0;

// ── Public interface ───────────────────────────────────────────────────
// Uses static per-tensor scales (scale_x * scale_W1 * d_model for hidden,
// extended by scale_W2 * d_ff for output) to avoid dynamic reduce passes.
// Both GEMM epilogues write INT8 directly — no intermediate quantize kernels.
void int8_mlp_forward(
    const int8_t* x, const int8_t* W1, const int8_t* W2, int8_t* out,
    const float* scale_x, const float* scale_W1, const float* scale_W2,
    int batch, int seq_len, int d_model, int d_ff
) {
    const int T = batch * seq_len;

    const size_t hidden_i8_sz = (size_t)T * d_ff * sizeof(int8_t);

    if (hidden_i8_sz > s_hidden_cap) {
        cudaFree(s_hidden_int8);
        cudaFree(s_scale_hidden);  cudaFree(s_inv_scale_hidden);
        cudaFree(s_scale_out);     cudaFree(s_inv_scale_out);
        cudaMalloc(&s_hidden_int8,      hidden_i8_sz);
        cudaMalloc(&s_scale_hidden,     sizeof(float));
        cudaMalloc(&s_inv_scale_hidden, sizeof(float));
        cudaMalloc(&s_scale_out,        sizeof(float));
        cudaMalloc(&s_inv_scale_out,    sizeof(float));
        s_hidden_cap = hidden_i8_sz;
    }

    // Compute static scales on device — no cudaMemcpy, no extra kernel passes.
    compute_static_scales_kernel<<<1, 1>>>(
        scale_x, scale_W1, scale_W2,
        s_scale_hidden, s_inv_scale_hidden,
        s_scale_out,    s_inv_scale_out,
        d_model, d_ff);

    const bool use_wmma =
        (T      % BLOCK_M == 0) && (d_model % BLOCK_N == 0) &&
        (d_ff   % BLOCK_N == 0) && (d_model % STAGE_K == 0) &&
        (d_ff   % STAGE_K == 0);

    // ── GEMM 1: x @ W1 → INT8 hidden (with GELU, static scale) ─────────
    if (use_wmma) {
        dim3 grid(d_ff / BLOCK_N, T / BLOCK_M);
        gemm_int8_wmma_i8_kernel<true><<<grid, THREADS_PER_BLOCK>>>(
            x, W1, s_hidden_int8, T, d_ff, d_model,
            scale_x, scale_W1, s_inv_scale_hidden);
    } else {
        dim3 grid((d_ff + SCALAR_TILE-1)/SCALAR_TILE, (T + SCALAR_TILE-1)/SCALAR_TILE);
        gemm_int8_scalar_i8_kernel<true><<<grid, dim3(SCALAR_TILE, SCALAR_TILE)>>>(
            x, W1, s_hidden_int8, T, d_ff, d_model,
            scale_x, scale_W1, s_inv_scale_hidden);
    }

    // ── GEMM 2: hidden @ W2 → INT8 out (static scale) ───────────────────
    if (use_wmma) {
        dim3 grid(d_model / BLOCK_N, T / BLOCK_M);
        gemm_int8_wmma_i8_kernel<false><<<grid, THREADS_PER_BLOCK>>>(
            s_hidden_int8, W2, out, T, d_model, d_ff,
            s_scale_hidden, scale_W2, s_inv_scale_out);
    } else {
        dim3 grid((d_model + SCALAR_TILE-1)/SCALAR_TILE, (T + SCALAR_TILE-1)/SCALAR_TILE);
        gemm_int8_scalar_i8_kernel<false><<<grid, dim3(SCALAR_TILE, SCALAR_TILE)>>>(
            s_hidden_int8, W2, out, T, d_model, d_ff,
            s_scale_hidden, scale_W2, s_inv_scale_out);
    }
}

// Returns the static output scale from the most recent int8_mlp_forward call.
void int8_mlp_get_output_scale(float* host_out) {
    if (s_scale_out)
        cudaMemcpy(host_out, s_scale_out, sizeof(float), cudaMemcpyDeviceToHost);
    else
        *host_out = 1.f;
}
