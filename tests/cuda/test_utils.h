// test_utils.h — Shared utilities for pure CUDA tests.
// Provides: load_bin, check_result, parse_config

#pragma once

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <map>

// ── Load raw binary file into device memory ────────────────────────────

template <typename T>
T* load_bin(const char* path, size_t count) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open %s\n", path);
        exit(1);
    }

    size_t bytes = count * sizeof(T);
    T* host = (T*)malloc(bytes);
    size_t read = fread(host, sizeof(T), count, f);
    fclose(f);

    if (read != count) {
        fprintf(stderr, "ERROR: %s — expected %zu elements, got %zu\n", path, count, read);
        exit(1);
    }

    T* dev;
    cudaMalloc(&dev, bytes);
    cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice);
    free(host);
    return dev;
}

// ── Load a single float32 scalar from .bin ─────────────────────────────

float load_scale(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open %s\n", path);
        exit(1);
    }
    float val;
    fread(&val, sizeof(float), 1, f);
    fclose(f);
    return val;
}

// ── Compare device tensor against reference .bin ───────────────────────

bool check_result_fp16(const half* d_actual, const char* ref_path, size_t count, float atol, const char* label) {
    // Load reference to host
    FILE* f = fopen(ref_path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open %s\n", ref_path);
        return false;
    }
    half* h_ref = (half*)malloc(count * sizeof(half));
    fread(h_ref, sizeof(half), count, f);
    fclose(f);

    // Copy actual to host
    half* h_actual = (half*)malloc(count * sizeof(half));
    cudaMemcpy(h_actual, d_actual, count * sizeof(half), cudaMemcpyDeviceToHost);

    // Compare
    float max_err = 0.0f;
    float sum_err = 0.0f;
    size_t worst_idx = 0;
    for (size_t i = 0; i < count; i++) {
        float diff = fabsf(__half2float(h_actual[i]) - __half2float(h_ref[i]));
        sum_err += diff;
        if (diff > max_err) {
            max_err = diff;
            worst_idx = i;
        }
    }
    float mean_err = sum_err / count;
    bool passed = (max_err <= atol);

    printf("[%s] %s  |  max err = %.6f  |  mean err = %.6f  |  atol = %.4f\n",
           passed ? "PASSED" : "FAILED", label, max_err, mean_err, atol);

    if (!passed) {
        printf("  Worst error at index %zu\n", worst_idx);
        printf("  Reference: %.6f\n", __half2float(h_ref[worst_idx]));
        printf("  Actual:    %.6f\n", __half2float(h_actual[worst_idx]));
        printf("  First 8 ref:");
        for (int i = 0; i < 8 && i < (int)count; i++)
            printf(" %.4f", __half2float(h_ref[i]));
        printf("\n  First 8 act:");
        for (int i = 0; i < 8 && i < (int)count; i++)
            printf(" %.4f", __half2float(h_actual[i]));
        printf("\n");
    }

    free(h_ref);
    free(h_actual);
    return passed;
}

// ── Parse config.txt from testdata directory ───────────────────────────

std::map<std::string, int> parse_config(const char* dir) {
    std::map<std::string, int> cfg;
    std::string path = std::string(dir) + "/config.txt";
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "ERROR: cannot open %s\n", path.c_str());
        exit(1);
    }
    std::string line;
    while (std::getline(f, line)) {
        size_t eq = line.find('=');
        if (eq != std::string::npos) {
            std::string key = line.substr(0, eq);
            std::string val = line.substr(eq + 1);
            if (key != "seed") {
                cfg[key] = std::stoi(val);
            }
        }
    }
    return cfg;
}
