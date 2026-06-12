// quant_utils.cu — Shared quantization utilities (FP16 <-> INT8 conversion).

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// ── Kernels ────────────────────────────────────────────────────────────

// Reduce abs-max over FP16 array; result written into d_amax (positive float).
// Uses unsigned atomicMax trick: for non-negative IEEE floats, bit-for-bit
// comparison equals float comparison.
__global__ void reduce_amax_f16_kernel(const half* __restrict__ src,
                                       int n, float* d_amax) {
    extern __shared__ unsigned smem_u[];
    const int tid = threadIdx.x;
    float v = 0.f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x)
        v = fmaxf(v, fabsf(__half2float(src[i])));
    smem_u[tid] = __float_as_uint(v);
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem_u[tid] = max(smem_u[tid], smem_u[tid + s]);
        __syncthreads();
    }
    if (tid == 0) atomicMax((unsigned int*)d_amax, smem_u[0]);
}

// Convert abs-max → scale = abs_max / 127.
__global__ void finalize_scale_kernel(float* d_scale) {
    *d_scale = *d_scale / 127.f;
}

// Quantize FP16 → INT8.  d_scale is a device pointer to the pre-computed scale.
__global__ void quantize_f16_to_i8_kernel(const half* __restrict__ src,
                                           int8_t* __restrict__ dst,
                                           int n, const float* d_scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = __half2float(src[i]) / *d_scale;
    dst[i] = (int8_t)__float2int_rn(fmaxf(-128.f, fminf(127.f, v)));
}

// Dequantize INT8 → FP16.
__global__ void dequantize_i8_to_f16_kernel(const int8_t* __restrict__ src,
                                             half* __restrict__ dst,
                                             int n, float scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half((float)src[i] * scale);
}

// ── Public interface ───────────────────────────────────────────────────

// Quantize: FP16 input → INT8 output.
// d_scale: device float[1] that will hold the computed per-tensor scale.
// Call order: (1) init d_scale to 0, (2) reduce abs-max, (3) ÷127, (4) quantize.
void quantize_fp16_to_int8(const half* input, int8_t* output,
                            float* d_scale, int n) {
    cudaMemset(d_scale, 0, sizeof(float));

    const int block = 256;
    const int grid = min((n + block - 1) / block, 1024);
    reduce_amax_f16_kernel<<<grid, block, block * sizeof(unsigned)>>>(
        input, n, d_scale);

    finalize_scale_kernel<<<1, 1>>>(d_scale);

    quantize_f16_to_i8_kernel<<<(n + block - 1) / block, block>>>(
        input, output, n, d_scale);
}

// Dequantize: INT8 input → FP16 output, given a host-side scale value.
void dequantize_int8_to_fp16(const int8_t* input, half* output,
                              float scale, int n) {
    const int block = 256;
    dequantize_i8_to_f16_kernel<<<(n + block - 1) / block, block>>>(
        input, output, n, scale);
}
