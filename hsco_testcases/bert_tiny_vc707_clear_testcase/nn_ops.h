/*
 * BERT-tiny FPGA Baremetal Implementation
 * Neural Network Operations
 *
 * FPGA Baremetal Notes:
 * - Uses sqrtf_approx / expf_approx / tanhf_approx for RISC-V (no hardware math)
 * - All memory allocation via aligned_malloc (not malloc)
 * - No snprintf available; string helpers in bert_tiny.h
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
    const float *input, int size, int normalized_size,
    const float *gamma, const float *beta,
    float *output
) {
    const float eps = 1e-12f;
    int num_batches = size / normalized_size;

    for (int b = 0; b < num_batches; b++) {
        const float *in_ptr = input + b * normalized_size;
        float *out_ptr = output + b * normalized_size;

        float mean = 0.0f;
        for (int i = 0; i < normalized_size; i++) {
            mean += in_ptr[i];
        }
        mean /= normalized_size;

        float variance = 0.0f;
        for (int i = 0; i < normalized_size; i++) {
            float diff = in_ptr[i] - mean;
            variance += diff * diff;
        }
        variance /= normalized_size;

        float std = sqrtf_approx(variance + eps);

        for (int i = 0; i < normalized_size; i++) {
            out_ptr[i] = gamma[i] * (in_ptr[i] - mean) / std + beta[i];
        }
    }
}

/* ============================================ */
/* Embedding Operations                         */
/* ============================================ */

static void embedding_lookup(
    const int *input_ids, int seq_len,
    const float *embedding_table, int vocab_size, int embed_dim,
    float *output
) {
    for (int i = 0; i < seq_len; i++) {
        int token_id = input_ids[i];
        if (token_id >= 0 && token_id < vocab_size) {
            for (int j = 0; j < embed_dim; j++) {
                output[i * embed_dim + j] = embedding_table[token_id * embed_dim + j];
            }
        }
    }
}

static void add_position_embeddings(
    const float *token_embeds, int seq_len, int embed_dim,
    const float *position_embeds, int max_position,
    float *output
) {
    int limit = seq_len < max_position ? seq_len : max_position;
    for (int i = 0; i < limit; i++) {
        for (int j = 0; j < embed_dim; j++) {
            int idx = i * embed_dim + j;
            output[idx] = token_embeds[idx] + position_embeds[idx];
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
        float max_val = input[i * cols];
        for (int j = 1; j < cols; j++) {
            if (input[i * cols + j] > max_val) max_val = input[i * cols + j];
        }

        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            output[i * cols + j] = expf_approx(input[i * cols + j] - max_val);
            sum += output[i * cols + j];
        }

        float inv_sum = 1.0f / sum;
        for (int j = 0; j < cols; j++) {
            output[i * cols + j] *= inv_sum;
        }
    }
}

/* ============================================ */
/* Multi-Head Self-Attention                    */
/* ============================================ */

/* MHA workspace layout (all sizes in floats):
 *   Q[S*H] | K[S*H] | V[S*H] | attn_output[S*H]
 *   | q_head[S*D] | k_head[S*D] | v_head[S*D] | k_t[D*S] | scores[S*S] | head_out[S*D]
 * where D = H / num_heads
 */
#define MHA_WORKSPACE_FLOATS(S, H, NH) \
    (4*(S)*(H) + 5*(S)*((H)/(NH)) + (S)*(S))

static void multi_head_attention(
    const float *input, int seq_len, int hidden_size, int num_heads,
    const float *q_weight, const float *q_bias,
    const float *k_weight, const float *k_bias,
    const float *v_weight, const float *v_bias,
    const float *o_weight, const float *o_bias,
    const float *attention_mask,
    float *output,
    float *Q, float *K, float *V, float *attn_output,
    float *q_head, float *k_head, float *v_head,
    float *k_t, float *scores, float *head_output
) {
    int head_dim = hidden_size / num_heads;
    float scale = 1.0f / sqrtf_approx((float)head_dim);
    int SH = seq_len * hidden_size;
    int SD = seq_len * head_dim;
    int SS = seq_len * seq_len;

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

        for (int i = 0; i < SS; i++) {
            scores[i] *= scale;
        }

        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                if (attention_mask[j] == 0.0f) {
                    scores[i * seq_len + j] = -10000.0f;
                }
            }
        }

        softmax_2d(scores, seq_len, seq_len, scores);

        matmul(scores, v_head, head_output, seq_len, seq_len, head_dim);

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
    float *output,
    float *inter_buf
) {
    linear_batched(input, seq_len, hidden_size, fc1_weight, fc1_bias, intermediate_size, inter_buf);
    gelu_array(inter_buf, seq_len * intermediate_size, inter_buf);
    linear_batched(inter_buf, seq_len, intermediate_size, fc2_weight, fc2_bias, hidden_size, output);
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
/* Pooling                                      */
/* ============================================ */

static void cls_pooling(
    const float *token_embeddings, int seq_len, int hidden_size,
    float *output
) {
    for (int i = 0; i < hidden_size; i++) {
        output[i] = token_embeddings[i];
    }
}

/* ============================================ */
/* Debug Utilities                              */
/* ============================================ */

static void print_float_debug(float x) {
    int neg = 0;
    if (x < 0) { neg = 1; x = -x; }
    int scaled = (int)(x * 10000.0f + 0.5f);
    int ipart  = scaled / 10000;
    int fpart  = scaled % 10000;
    if (neg) printf("-");
    printf("%d.%04d", ipart, fpart);
}

static void debug_tensor(const char *label, const float *data, int size) {
    float mn = data[0], mx = data[0], sum = 0.0f;
    for (int i = 0; i < size; i++) {
        if (data[i] < mn) mn = data[i];
        if (data[i] > mx) mx = data[i];
        sum += data[i];
    }
    printf("  [DBG] %s  size=%d  min=", label, size);
    print_float_debug(mn);
    printf(" max=");
    print_float_debug(mx);
    printf(" mean=");
    print_float_debug(sum / size);
    printf("\n        first5:");
    int n = size < 5 ? size : 5;
    for (int i = 0; i < n; i++) {
        printf(" ");
        print_float_debug(data[i]);
    }
    printf("\n");
}

#endif /* NN_OPS_H */
