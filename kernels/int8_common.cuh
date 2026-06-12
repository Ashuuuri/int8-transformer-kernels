#pragma once
// int8_common.cuh — Device-side helpers shared by INT8 attention and MLP kernels.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// ── GELU (tanh approximation) ──────────────────────────────────────────
__device__ __forceinline__ float int8_gelu(float x) {
    const float kC = 0.7978845608028654f;
    const float kA = 0.044715f;
    return 0.5f * x * (1.f + tanhf(kC * (x + kA * x * x * x)));
}

// ── Saturating float → INT8 quantization ──────────────────────────────
// inv_scale = 1.f / scale  (caller pre-computes to avoid repeated division)
__device__ __forceinline__ int8_t f32_to_i8(float v, float inv_scale) {
    return (int8_t)__float2int_rn(fmaxf(-128.f, fminf(127.f, v * inv_scale)));
}

// ── cp.async helpers (Ampere sm_80+) ──────────────────────────────────
__device__ __forceinline__ void i8_cp_async_16B(void* dst, const void* src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(d), "l"(src));
#else
    *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(src);
#endif
}

__device__ __forceinline__ void i8_cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void i8_cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

// Wait until at most one async group remains in flight (double-buffer use:
// blocks on tile t while tile t+1 keeps loading in hardware).
__device__ __forceinline__ void i8_cp_async_wait_one() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 1;\n" ::);
#endif
}

// ── quant_utils.cu public interface ───────────────────────────────────
// Quantize FP16 → INT8 with dynamically-computed per-tensor scale.
// d_scale[0] (device) receives the computed scale after the call.
void quantize_fp16_to_int8(const half* input, int8_t* output,
                            float* d_scale, int n);

// Dequantize INT8 → FP16 using a host-side scale value.
void dequantize_int8_to_fp16(const int8_t* input, half* output,
                              float scale, int n);
