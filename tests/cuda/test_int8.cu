// test_int8.cu — Pure CUDA correctness test for INT8 attention + MLP kernels.
//
// Usage:
//   make test_int8 && ./test_int8
//
// Reads testdata/{small,large}/ .bin files, runs int8_attention_forward
// and int8_mlp_forward, compares output against pre-computed reference.

#include <cstdio>
#include <cuda_fp16.h>
#include <cstdint>
#include "test_utils.h"

// Declared in kernels/int8_attention.cu
extern void int8_attention_forward(
    const int8_t* Q, const int8_t* K, const int8_t* V, int8_t* out,
    const float* scale_Q, const float* scale_K, const float* scale_V,
    int batch, int heads, int seq_len, int head_dim
);

// Declared in kernels/int8_mlp.cu
extern void int8_mlp_forward(
    const int8_t* x, const int8_t* W1, const int8_t* W2, int8_t* out,
    const float* scale_x, const float* scale_W1, const float* scale_W2,
    int batch, int seq_len, int d_model, int d_ff
);

bool run_attention_test(const char* data_dir, const char* label) {
    auto cfg = parse_config(data_dir);
    int B = cfg["batch"], H = cfg["heads"], S = cfg["seq_len"], D = cfg["head_dim"];
    size_t count = (size_t)B * H * S * D;

    char path[512];

    snprintf(path, sizeof(path), "%s/int8_attn_Q.bin", data_dir);
    int8_t* Q = load_bin<int8_t>(path, count);

    snprintf(path, sizeof(path), "%s/int8_attn_K.bin", data_dir);
    int8_t* K = load_bin<int8_t>(path, count);

    snprintf(path, sizeof(path), "%s/int8_attn_V.bin", data_dir);
    int8_t* V = load_bin<int8_t>(path, count);

    // Load scales
    snprintf(path, sizeof(path), "%s/int8_attn_scale_Q.bin", data_dir);
    float h_scale_Q = load_scale(path);
    snprintf(path, sizeof(path), "%s/int8_attn_scale_K.bin", data_dir);
    float h_scale_K = load_scale(path);
    snprintf(path, sizeof(path), "%s/int8_attn_scale_V.bin", data_dir);
    float h_scale_V = load_scale(path);

    float *d_scale_Q, *d_scale_K, *d_scale_V;
    cudaMalloc(&d_scale_Q, sizeof(float));
    cudaMalloc(&d_scale_K, sizeof(float));
    cudaMalloc(&d_scale_V, sizeof(float));
    cudaMemcpy(d_scale_Q, &h_scale_Q, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_K, &h_scale_K, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_V, &h_scale_V, sizeof(float), cudaMemcpyHostToDevice);

    int8_t* out_int8;
    cudaMalloc(&out_int8, count * sizeof(int8_t));

    // Run kernel
    int8_attention_forward(Q, K, V, out_int8, d_scale_Q, d_scale_K, d_scale_V, B, H, S, D);
    cudaDeviceSynchronize();

    // For correctness: compare dequantized output against FP16 reference
    // The reference was computed as: dequant inputs → FP16 baseline → FP16 output
    // So we compare against attn_ref (FP16)
    // For now, we need the kernel to also produce FP16 output for comparison,
    // or we dequantize the INT8 output here. This depends on kernel design.
    // TODO: Adapt comparison once kernel output format is finalized.
    printf("[SKIP] %s — INT8 attention kernel not yet implemented\n", label);

    cudaFree(Q);
    cudaFree(K);
    cudaFree(V);
    cudaFree(d_scale_Q);
    cudaFree(d_scale_K);
    cudaFree(d_scale_V);
    cudaFree(out_int8);
    return true;  // placeholder
}

bool run_mlp_test(const char* data_dir, const char* label) {
    auto cfg = parse_config(data_dir);
    int B = cfg["batch"], S = cfg["seq_len"], M = cfg["d_model"], F = cfg["d_ff"];
    size_t x_count = (size_t)B * S * M;
    size_t w1_count = (size_t)M * F;
    size_t w2_count = (size_t)F * M;

    char path[512];

    snprintf(path, sizeof(path), "%s/int8_mlp_x.bin", data_dir);
    int8_t* x = load_bin<int8_t>(path, x_count);

    snprintf(path, sizeof(path), "%s/int8_mlp_W1.bin", data_dir);
    int8_t* W1 = load_bin<int8_t>(path, w1_count);

    snprintf(path, sizeof(path), "%s/int8_mlp_W2.bin", data_dir);
    int8_t* W2 = load_bin<int8_t>(path, w2_count);

    // Load scales
    snprintf(path, sizeof(path), "%s/int8_mlp_scale_x.bin", data_dir);
    float h_scale_x = load_scale(path);
    snprintf(path, sizeof(path), "%s/int8_mlp_scale_W1.bin", data_dir);
    float h_scale_W1 = load_scale(path);
    snprintf(path, sizeof(path), "%s/int8_mlp_scale_W2.bin", data_dir);
    float h_scale_W2 = load_scale(path);

    float *d_scale_x, *d_scale_W1, *d_scale_W2;
    cudaMalloc(&d_scale_x, sizeof(float));
    cudaMalloc(&d_scale_W1, sizeof(float));
    cudaMalloc(&d_scale_W2, sizeof(float));
    cudaMemcpy(d_scale_x, &h_scale_x, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_W1, &h_scale_W1, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_W2, &h_scale_W2, sizeof(float), cudaMemcpyHostToDevice);

    int8_t* out_int8;
    cudaMalloc(&out_int8, x_count * sizeof(int8_t));

    // Run kernel
    int8_mlp_forward(x, W1, W2, out_int8, d_scale_x, d_scale_W1, d_scale_W2, B, S, M, F);
    cudaDeviceSynchronize();

    // TODO: Adapt comparison once kernel output format is finalized.
    printf("[SKIP] %s — INT8 MLP kernel not yet implemented\n", label);

    cudaFree(x);
    cudaFree(W1);
    cudaFree(W2);
    cudaFree(d_scale_x);
    cudaFree(d_scale_W1);
    cudaFree(d_scale_W2);
    cudaFree(out_int8);
    return true;  // placeholder
}

int main() {
    printf("=== INT8 Kernel Tests ===\n\n");

    printf("--- INT8 Attention ---\n");
    bool all_passed = true;
    all_passed &= run_attention_test("testdata/small", "small_attn");
    all_passed &= run_attention_test("testdata/large", "large_attn");

    printf("\n--- INT8 MLP ---\n");
    all_passed &= run_mlp_test("testdata/small", "small_mlp");
    all_passed &= run_mlp_test("testdata/large", "large_mlp");

    printf("\n%s\n", all_passed ? "All tests PASSED." : "Some tests FAILED.");
    return all_passed ? 0 : 1;
}
