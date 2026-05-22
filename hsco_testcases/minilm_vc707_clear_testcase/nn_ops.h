/*
 * paraphrase-MiniLM-L3-v2 FPGA Baremetal Implementation
 * Neural Network Operations
 *
 * FPGA Baremetal Notes:
 * - Uses sqrtf_approx / expf_approx / tanhf_approx 
 * - All memory allocation via aligned_malloc (not malloc)
 * - No snprintf available; string helpers in minilm.h
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
/* BERT Activation Functions                    */
/* ============================================ */

/* GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
static inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf_approx(0.7978845608f * (x + 0.044715f * x * x * x)));
}

static void gelu_array(const float *input, int size, float *output) {
    for (int i = 0; i < size; i++) {
        output[i] = gelu(input[i]);
    }
}

/* ============================================ */
/* Layer Normalization                          */
/* ============================================ */

static void layer_norm(
    const float *input, int num_batches, int normalized_size,
    const float *gamma, const float *beta,
    float *output
) {
    const float eps = 1e-12f;

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

        float std = sqrtf_approx(variance + eps);

        for (int i = 0; i < normalized_size; i++) {
            output[base + i] = gamma[i] * (input[base + i] - mean) / std + beta[i];
        }
    }
}

/* ============================================ */
/* Linear and Matrix Operations                 */
/* ============================================ */

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

static void transpose_2d(const float *input, float *output, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            output[j * M + i] = input[i * N + j];
        }
    }
}

/* ============================================ */
/* Softmax (2D, per-row)                        */
/* ============================================ */

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
/* Multi-Head Self-Attention                    */
/* ============================================ */

static void multi_head_attention(
    const float *input, int seq_len, int hidden_size, int num_heads, int head_dim,
    const float *q_weight, const float *q_bias,
    const float *k_weight, const float *k_bias,
    const float *v_weight, const float *v_bias,
    const float *o_weight, const float *o_bias,
    const float *attention_mask,
    float *Q, float *K, float *V, float *attn_output,
    float *q_head, float *k_head, float *v_head,
    float *k_t, float *scores, float *scores_out, float *head_output,
    float *output
) {
    float scale = 1.0f / sqrtf_approx((float)head_dim);

    linear_batched(input, seq_len, hidden_size, q_weight, q_bias, hidden_size, Q);
    linear_batched(input, seq_len, hidden_size, k_weight, k_bias, hidden_size, K);
    linear_batched(input, seq_len, hidden_size, v_weight, v_bias, hidden_size, V);

    for (int h = 0; h < num_heads; h++) {
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                q_head[i * head_dim + j] = Q[i * hidden_size + h * head_dim + j];
                k_head[i * head_dim + j] = K[i * hidden_size + h * head_dim + j];
                v_head[i * head_dim + j] = V[i * hidden_size + h * head_dim + j];
            }
        }

        transpose_2d(k_head, k_t, seq_len, head_dim);

        matmul(q_head, k_t, scores, seq_len, head_dim, seq_len);

        for (int i = 0; i < seq_len * seq_len; i++) {
            scores[i] *= scale;
        }

        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                if (attention_mask[j] == 0.0f) {
                    scores[i * seq_len + j] = -10000.0f;
                }
            }
        }

        softmax_2d(scores, seq_len, seq_len, scores_out);

        matmul(scores_out, v_head, head_output, seq_len, seq_len, head_dim);

        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                attn_output[i * hidden_size + h * head_dim + j] = head_output[i * head_dim + j];
            }
        }
    }

    linear_batched(attn_output, seq_len, hidden_size, o_weight, o_bias, hidden_size, output);
}

/* ============================================ */
/* Feed-Forward Network                         */
/* ============================================ */

static void feed_forward(
    const float *input, int seq_len, int hidden_size, int intermediate_size,
    const float *fc1_weight, const float *fc1_bias,
    const float *fc2_weight, const float *fc2_bias,
    float *inter_buf, float *gelu_out,
    float *output
) {
    linear_batched(input, seq_len, hidden_size, fc1_weight, fc1_bias, intermediate_size, inter_buf);
    gelu_array(inter_buf, seq_len * intermediate_size, gelu_out);
    linear_batched(gelu_out, seq_len, intermediate_size, fc2_weight, fc2_bias, hidden_size, output);
}

/* ============================================ */
/* Residual Connection                          */
/* ============================================ */

static void add_arrays(const float *a, const float *b, int size, float *output) {
    for (int i = 0; i < size; i++) {
        output[i] = a[i] + b[i];
    }
}

/* ============================================ */
/* Mean Pooling (sentence-transformers style)   */
/* ============================================ */

static void mean_pooling(
    const float *token_embeddings, int seq_len, int hidden_size,
    const float *attention_mask,
    float *output
) {
    for (int i = 0; i < hidden_size; i++) {
        output[i] = 0.0f;
    }

    int valid_tokens = 0;
    for (int i = 0; i < seq_len; i++) {
        if (attention_mask[i] == 1.0f) {
            for (int j = 0; j < hidden_size; j++) {
                output[j] += token_embeddings[i * hidden_size + j];
            }
            valid_tokens++;
        }
    }

    if (valid_tokens > 0) {
        for (int i = 0; i < hidden_size; i++) {
            output[i] /= valid_tokens;
        }
    }
}

#endif /* NN_OPS_H */
