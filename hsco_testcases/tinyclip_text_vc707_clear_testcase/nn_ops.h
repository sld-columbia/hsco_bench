/*
 * TinyCLIP FPGA Baremetal Implementation
 * Neural Network Operations
 *
 * FPGA Baremetal Notes:
 * - Uses sqrtf_approx / expf_approx / tanhf_approx
 * - All memory allocation via aligned_malloc (not malloc)
 * - No snprintf available; string helpers in tinyclip_text.h
 */

#ifndef NN_OPS_H
#define NN_OPS_H

#include "tensor.h"
#include <float.h>

/* ============================================ */
/* Math Approximations for RISC-V Baremetal     */
/* ============================================ */

static inline float sqrtf_approx(float x) {
#ifdef __riscv
    if (x <= 0.0f) return 0.0f;
    float xhalf = 0.5f * x;
    int i = *(int*)&x;
    i = 0x5f3759df - (i >> 1);
    x = *(float*)&i;
    x = x * (1.5f - xhalf * x * x);
    x = x * (1.5f - xhalf * x * x);
    return 1.0f / x;
#else
    return sqrtf(x);
#endif
}

static inline float fabsf_approx(float x) {
    return x < 0.0f ? -x : x;
}

/* Exp approximation for RISC-V baremetal */
static inline float expf_approx(float x) {
#ifdef __riscv
    if (x < -87.0f) return 0.0f;
    if (x > 88.0f) return 3.4e38f;
    const float LOG2E = 1.44269504f;
    const float LN2   = 0.69314718f;
    float t = x * LOG2E;
    int k = (int)(t + (t >= 0 ? 0.5f : -0.5f));
    float r = x - k * LN2;
    /* 5th-order polynomial for exp(r), |r| <= 0.347 */
    float exp_r = 1.0f + r * (1.0f + r * (0.5f + r * (0.166666667f + r * (0.041666667f + r * 0.008333333f))));
    /* Multiply by 2^k via IEEE 754 bit manipulation */
    union { float f; int32_t i; } u;
    u.i = (127 + k) << 23;
    return exp_r * u.f;
#else
    return expf(x);
#endif
}

/* Tanh approximation using exp: tanh(x) = (e^2x - 1) / (e^2x + 1) */
static inline float tanhf_approx(float x) {
#ifdef __riscv
    if (x > 10.0f) return 1.0f;
    if (x < -10.0f) return -1.0f;
    float e2x = expf_approx(2.0f * x);
    return (e2x - 1.0f) / (e2x + 1.0f);
#else
    return tanhf(x);
#endif
}

/* ============================================ */
/* CLIP Activation Functions                    */
/* ============================================ */

/* QuickGELU: x * sigmoid(1.702 * x) — used by CLIP/TinyCLIP */
static inline float quick_gelu(float x) {
    return x * (1.0f / (1.0f + expf_approx(-1.702f * x)));
}

/* Standard GELU (for reference, not used by TinyCLIP) */
static inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf_approx(0.7978845608f * (x + 0.044715f * x * x * x)));
}

/* ============================================ */
/* Layer Normalization                          */
/* ============================================ */

/* LayerNorm: input[size, normalized_size] -> output[size, normalized_size] */
static void layer_norm(
    const float *input, int size, int normalized_size,
    const float *gamma, const float *beta,
    float *output
) {
    const float eps = 1e-5f;
    int num_batches = size / normalized_size;

    for (int b = 0; b < num_batches; b++) {
        int base = b * normalized_size;

        float mean = 0.0f;
        for (int i = 0; i < normalized_size; i++) {
            mean += input[base + i];
        }
        mean /= normalized_size;

        float variance = 0.0f;
        for (int i = 0; i < normalized_size; i++) {
            float diff = input[base + i] - mean;
            variance += diff * diff;
        }
        variance /= normalized_size;

        float inv_std = 1.0f / sqrtf_approx(variance + eps);

        for (int i = 0; i < normalized_size; i++) {
            output[base + i] = gamma[i] * (input[base + i] - mean) * inv_std + beta[i];
        }
    }
}

/* ============================================ */
/* Linear Layer                                 */
/* ============================================ */

/* Linear batched: input[batch_size, in_features] @ weight.T + bias -> output[batch_size, out_features] */
static void linear_batched(
    const float *input, int batch_size, int in_features,
    const float *weight, const float *bias, int out_features,
    float *output
) {
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < out_features; i++) {
            float sum = bias[i];
            for (int j = 0; j < in_features; j++) {
                sum += input[b * in_features + j] * weight[i * in_features + j];
            }
            output[b * out_features + i] = sum;
        }
    }
}

/* ============================================ */
/* Matrix Operations                            */
/* ============================================ */

/* Matrix multiply: A[M, K] @ B[K, N] -> C[M, N] */
static void matmul(const float *A, const float *B, float *C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/* Matrix multiply with transposed B: A[M, K] @ B.T[N, K] -> C[M, N] */
static void matmul_transB(const float *A, const float *B, float *C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = sum;
        }
    }
}

/* Transpose: input[M, N] -> output[N, M] */
static void transpose_2d(const float *input, float *output, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            output[j * M + i] = input[i * N + j];
        }
    }
}

/* ============================================ */
/* Softmax                                      */
/* ============================================ */

/* Softmax (2D, per-row): input[rows, cols] -> output[rows, cols] */
static void softmax_2d(const float *input, int rows, int cols, float *output) {
    for (int i = 0; i < rows; i++) {
        int base = i * cols;

        float max_val = input[base];
        for (int j = 1; j < cols; j++) {
            if (input[base + j] > max_val) max_val = input[base + j];
        }

        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            output[base + j] = expf_approx(input[base + j] - max_val);
            sum += output[base + j];
        }

        for (int j = 0; j < cols; j++) {
            output[base + j] /= sum;
        }
    }
}

/* ============================================ */
/* Element-wise Operations                      */
/* ============================================ */

/* Add two arrays: output[i] = a[i] + b[i] */
static void add_arrays(const float *a, const float *b, int size, float *output) {
    for (int i = 0; i < size; i++) {
        output[i] = a[i] + b[i];
    }
}

/* Scale array: output[i] = input[i] * scale */
static void scale_array(const float *input, float scale, int size, float *output) {
    for (int i = 0; i < size; i++) {
        output[i] = input[i] * scale;
    }
}

/* QuickGELU array: output[i] = quick_gelu(input[i]) */
static void quick_gelu_array(const float *input, int size, float *output) {
    for (int i = 0; i < size; i++) {
        output[i] = input[i] * (1.0f / (1.0f + expf_approx(-1.702f * input[i])));
    }
}

/* ============================================ */
/* Utility Functions                            */
/* ============================================ */

/* L2 normalize: input[size] -> output[size] */
static void l2_normalize(const float *input, int size, float *output) {
    float norm = 0.0f;
    for (int i = 0; i < size; i++) {
        norm += input[i] * input[i];
    }
    norm = sqrtf_approx(norm + 1e-12f);

    for (int i = 0; i < size; i++) {
        output[i] = input[i] / norm;
    }
}

/* Matrix-vector multiply for projection: input[K] @ weight[K, N] -> output[N] */
/* This is different from linear() - weight is NOT transposed */
static void matvec_proj(const float *input, int K, const float *weight, int N, float *output) {
    for (int j = 0; j < N; j++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += input[k] * weight[k * N + j];
        }
        output[j] = sum;
    }
}

#endif /* NN_OPS_H */
