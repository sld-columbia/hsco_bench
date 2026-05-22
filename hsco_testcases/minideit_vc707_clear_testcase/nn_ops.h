/*
 * Mini-DeiT-Ti FPGA Baremetal Implementation
 * Neural Network Operations
 *
 * FPGA Baremetal Notes:
 * - Uses sqrtf_approx / expf_approx / logf_approx / roundf_approx / tanhf_approx
 * - GELU uses tanh-based approximation
 * - All memory allocation via aligned_malloc (not malloc)
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

static inline int abs_int(int x) {
    return x < 0 ? -x : x;
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

/* Log approximation for RISC-V baremetal (needed for iRPE piecewise_index) */
static inline float logf_approx(float x) {
#ifdef __riscv
    if (x <= 0.0f) return -88.0f;
    /* Extract exponent and mantissa via IEEE 754 */
    union { float f; int32_t i; } u;
    u.f = x;
    int e = ((u.i >> 23) & 0xFF) - 127;
    u.i = (u.i & 0x007FFFFF) | (127 << 23);
    float m = u.f;
    /* Polynomial approximation for log(m), m in [1, 2) */
    /* log(m) ≈ (m-1) - 0.5*(m-1)^2 + 0.333*(m-1)^3 */
    float t = m - 1.0f;
    float log_m = t * (1.0f + t * (-0.5f + t * (0.333333f + t * (-0.25f + t * 0.2f))));
    return (float)e * 0.69314718f + log_m;
#else
    return logf(x);
#endif
}

/* Round to nearest integer (baremetal safe) */
static inline float roundf_approx(float x) {
#ifdef __riscv
    return (float)(int)(x + (x >= 0 ? 0.5f : -0.5f));
#else
    return roundf(x);
#endif
}

/* tanh approximation using exp */
static inline float tanhf_approx(float x) {
    if (x > 10.0f) return 1.0f;
    if (x < -10.0f) return -1.0f;
    float e2x = expf_approx(2.0f * x);
    float denom = e2x + 1.0f;
    return (e2x - 1.0f) * (1.0f / denom);
}

/* ============================================ */
/* Basic Operations                             */
/* ============================================ */

/* GELU using tanh approximation */
/* GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
static inline float gelu(float x) {
#ifdef __riscv
    const float SQRT_2_OVER_PI = 0.7978845608f;
    float x3 = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanhf_approx(inner));
#else
    float sqrt2_inv = 0.7071067811865476f;
    return 0.5f * x * (1.0f + erff(x * sqrt2_inv));
#endif
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
    const float *input, int batch_size, int seq_len, int dim,
    const float *gamma, const float *beta,
    float *output
) {
    const float eps = 1e-6f;
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            int base = (b * seq_len + s) * dim;

            /* Compute mean */
            float mean = 0.0f;
            for (int i = 0; i < dim; i++) {
                mean += input[base + i];
            }
            mean /= dim;

            /* Compute variance */
            float var = 0.0f;
            for (int i = 0; i < dim; i++) {
                float diff = input[base + i] - mean;
                var += diff * diff;
            }
            var /= dim;

            /* Normalize */
            float inv_std = 1.0f / sqrtf_approx(var + eps);
            for (int i = 0; i < dim; i++) {
                output[base + i] = (input[base + i] - mean) * inv_std * gamma[i] + beta[i];
            }
        }
    }
}

/* ============================================ */
/* Linear Layer                                 */
/* ============================================ */

static void linear(
    const float *input, int batch_size, int in_features,
    const float *weight, const float *bias, int out_features,
    float *output
) {
    /* input: [batch_size, in_features] */
    /* weight: [out_features, in_features] */
    /* output: [batch_size, out_features] */
    for (int b = 0; b < batch_size; b++) {
        for (int o = 0; o < out_features; o++) {
            float sum = bias[o];
            for (int i = 0; i < in_features; i++) {
                sum += input[b * in_features + i] * weight[o * in_features + i];
            }
            output[b * out_features + o] = sum;
        }
    }
}

/* ============================================ */
/* Softmax                                      */
/* ============================================ */

static void softmax(const float *input, int size, float *output) {
    /* Find max for numerical stability */
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    /* Compute sum of exp (first pass) */
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += expf_approx(input[i] - max_val);
    }

    /* Normalize: recompute exp and divide (output is write-only) */
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; i++) {
        output[i] = expf_approx(input[i] - max_val) * inv_sum;
    }
}

/* ============================================ */
/* Element-wise Operations                      */
/* ============================================ */

static void add_arrays(const float *a, const float *b, int size, float *output) {
    for (int i = 0; i < size; i++) {
        output[i] = a[i] + b[i];
    }
}

static void copy_tensor(float *dst, const float *src, int size) {
    for (int i = 0; i < size; i++) {
        dst[i] = src[i];
    }
}

/* ============================================ */
/* Patch Embedding (Conv2D)                     */
/* ============================================ */

/* input[in_c, img_size, img_size], weight[out_c, in_c, patch_size, patch_size],
   bias[out_c] -> output[patches_per_row*patches_per_row, out_c] */
static void patch_embed_conv(
    const float *input,
    const float *weight,
    const float *bias,
    int in_c, int out_c, int img_size, int patch_size, int patches_per_row,
    float *output
) {

    for (int oc = 0; oc < out_c; oc++) {
        for (int ph = 0; ph < patches_per_row; ph++) {
            for (int pw = 0; pw < patches_per_row; pw++) {
                int patch_idx = ph * patches_per_row + pw;
                float sum = bias[oc];

                for (int ic = 0; ic < in_c; ic++) {
                    for (int kh = 0; kh < patch_size; kh++) {
                        for (int kw = 0; kw < patch_size; kw++) {
                            int ih = ph * patch_size + kh;
                            int iw = pw * patch_size + kw;

                            int in_idx = ic * img_size * img_size + ih * img_size + iw;
                            int w_idx = oc * in_c * patch_size * patch_size +
                                       ic * patch_size * patch_size +
                                       kh * patch_size + kw;

                            sum += input[in_idx] * weight[w_idx];
                        }
                    }
                }

                output[patch_idx * out_c + oc] = sum;
            }
        }
    }
}

/* ============================================ */
/* iRPE (Image Relative Position Encoding) Helpers */
/* ============================================ */

/* Piecewise index function for iRPE */
static int piecewise_index(int relative_pos, float alpha, float beta, float gamma) {
    int rp_abs = abs_int(relative_pos);
    if (rp_abs <= (int)alpha) {
        return relative_pos;
    }
    int sign = (relative_pos > 0) ? 1 : -1;
    float y = sign * (alpha + logf_approx((float)rp_abs / alpha) / logf_approx(gamma / alpha) * (beta - alpha));
    int result = (int)roundf_approx(y);
    if (result > (int)beta) result = (int)beta;
    if (result < -(int)beta) result = -(int)beta;
    return result;
}

/* Get bucket ID for 2D product method */
static int get_bucket_id_product(int row1, int col1, int row2, int col2,
                                  float alpha, float beta, float gamma) {
    int beta_int = (int)beta;
    int S = 2 * beta_int + 1;

    int row_diff = row1 - row2;
    int col_diff = col1 - col2;

    int r = piecewise_index(row_diff, alpha, beta, gamma) + beta_int;
    int c = piecewise_index(col_diff, alpha, beta, gamma) + beta_int;

    return r * S + c;
}

/* Precompute bucket IDs for 14x14 grid (196 patches) */
static void compute_bucket_ids(float *bucket_ids, int height, int width,
                               float alpha, float beta, float gamma) {
    int L = height * width;
    for (int i = 0; i < L; i++) {
        int row1 = i / width;
        int col1 = i % width;
        for (int j = 0; j < L; j++) {
            int row2 = j / width;
            int col2 = j % width;
            bucket_ids[i * L + j] = (float)get_bucket_id_product(row1, col1, row2, col2,
                                                                 alpha, beta, gamma);
        }
    }
}

/* ============================================ */
/* Mini-DeiT Attention with iRPE and transforms */
/* ============================================ */

/*
 * mini_attention workspace layout (flat float array, size ATTN_WS_TOTAL):
 *   [0]                               qkv         seq_len * 3 * dim
 *   [seq_len*3*dim]                   q           seq_len * dim
 *   [seq_len*3*dim + seq_len*dim]     k           seq_len * dim
 *   [seq_len*3*dim + 2*seq_len*dim]   v           seq_len * dim
 *   [seq_len*3*dim + 3*seq_len*dim]   attn        num_heads * seq_len * seq_len
 *   [above + num_heads*seq*seq]       attn_out_ws seq_len * dim
 *   [above + seq_len*dim]             attn_tmp    num_heads * seq_len * seq_len
 *                                     (reused: q_lookup during heads, attn_transformed for conv)
 *   [above + num_heads*seq*seq]       bucket_ids  seq_len * seq_len (as float)
 */
/* x[seq_len, dim], qkv_weight[3*dim, dim], qkv_bias[3*dim],
   proj_weight[dim, dim], proj_bias[dim],
   rpe_k_table[head_dim, num_buckets] or NULL,
   conv_l/w_weight[num_heads, num_heads] or NULL
   -> output[seq_len, dim], workspace: flat buffer >= ATTN_WS_TOTAL floats */
static void mini_attention(
    const float *x,
    int seq_len, int dim, int num_heads,
    const float *qkv_weight,
    const float *qkv_bias,
    const float *proj_weight,
    const float *proj_bias,
    const float *rpe_k_table,
    const float *conv_l_weight,
    const float *conv_w_weight,
    float *output,
    float *workspace
) {
    int head_dim = dim / num_heads;
    float scale = 1.0f / sqrtf_approx((float)head_dim);

    /* iRPE parameters (ratio=1.9 for Mini-DeiT) */
    float alpha = 1.9f;
    float beta = 3.8f;
    float gamma_rpe = 15.2f;
    int num_buckets = 49;

    /* Partition workspace */
    float *qkv         = workspace;
    float *q           = qkv + seq_len * 3 * dim;
    float *k           = q + seq_len * dim;
    float *v           = k + seq_len * dim;
    float *attn        = v + seq_len * dim;
    float *attn_out_ws = attn + num_heads * seq_len * seq_len;
    float *attn_tmp    = attn_out_ws + seq_len * dim;
    /* attn_tmp reused: q_lookup (during heads) then attn_transformed (conv) */
    float *bucket_ids  = attn_tmp + num_heads * seq_len * seq_len;

    /* Precompute bucket IDs for 14x14 grid */
    compute_bucket_ids(bucket_ids, 14, 14, alpha, beta, gamma_rpe);

    /* QKV projection: [seq_len, dim] -> [seq_len, 3*dim] */
    linear(x, seq_len, dim, qkv_weight, qkv_bias, 3 * dim, qkv);

    /* Split into Q, K, V */
    for (int s = 0; s < seq_len; s++) {
        for (int d = 0; d < dim; d++) {
            q[s * dim + d] = qkv[s * 3 * dim + d];
            k[s * dim + d] = qkv[s * 3 * dim + dim + d];
            v[s * dim + d] = qkv[s * 3 * dim + 2 * dim + d];
        }
    }

    /* Scale Q */
    for (int i = 0; i < seq_len * dim; i++) {
        q[i] *= scale;
    }

    /* Multi-head attention with iRPE */
    for (int h = 0; h < num_heads; h++) {
        /* Q @ K^T */
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                float sum = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    sum += q[i * dim + h * head_dim + d] * k[j * dim + h * head_dim + d];
                }
                attn[h * seq_len * seq_len + i * seq_len + j] = sum;
            }
        }

        /* Add iRPE on keys: attn_tmp reused as q_lookup here */
        {
            /* attn_tmp reused as q_lookup: size seq_len*num_buckets << num_heads*seq*seq */
            for (int i = 0; i < seq_len; i++) {
                for (int b = 0; b < num_buckets; b++) {
                    float sum = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        sum += q[i * dim + h * head_dim + d] * rpe_k_table[d * num_buckets + b];
                    }
                    attn_tmp[i * num_buckets + b] = sum;
                }
            }
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < seq_len; j++) {
                    int bid = (int)bucket_ids[i * seq_len + j];
                    attn[h * seq_len * seq_len + i * seq_len + j] += attn_tmp[i * num_buckets + bid];
                }
            }
        }
    }

    /* Apply conv_l transform (before softmax); attn_tmp reused as attn_transformed */
    {
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                for (int h_out = 0; h_out < num_heads; h_out++) {
                    float sum = 0.0f;
                    for (int h_in = 0; h_in < num_heads; h_in++) {
                        sum += attn[h_in * seq_len * seq_len + i * seq_len + j]
                             * conv_l_weight[h_out * num_heads + h_in];
                    }
                    attn_tmp[h_out * seq_len * seq_len + i * seq_len + j] = sum;
                }
            }
        }
        for (int _i = 0; _i < num_heads * seq_len * seq_len; _i++)
            attn[_i] = attn_tmp[_i];
    }

    /* Softmax per head per row (self-aliased: safe for sequential element access) */
    for (int h = 0; h < num_heads; h++) {
        for (int i = 0; i < seq_len; i++) {
            float *row = attn + h * seq_len * seq_len + i * seq_len;
            softmax(row, seq_len, row);
        }
    }

    /* Apply conv_w transform (after softmax); attn_tmp reused as attn_transformed */
    {
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                for (int h_out = 0; h_out < num_heads; h_out++) {
                    float sum = 0.0f;
                    for (int h_in = 0; h_in < num_heads; h_in++) {
                        sum += attn[h_in * seq_len * seq_len + i * seq_len + j]
                             * conv_w_weight[h_out * num_heads + h_in];
                    }
                    attn_tmp[h_out * seq_len * seq_len + i * seq_len + j] = sum;
                }
            }
        }
        for (int _i = 0; _i < num_heads * seq_len * seq_len; _i++)
            attn[_i] = attn_tmp[_i];
    }

    /* Attention @ V */
    for (int h = 0; h < num_heads; h++) {
        for (int i = 0; i < seq_len; i++) {
            for (int d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    sum += attn[h * seq_len * seq_len + i * seq_len + j]
                         * v[j * dim + h * head_dim + d];
                }
                attn_out_ws[i * dim + h * head_dim + d] = sum;
            }
        }
    }

    /* Output projection */
    linear(attn_out_ws, seq_len, dim, proj_weight, proj_bias, dim, output);
}

/* ============================================ */
/* MLP Block                                    */
/* ============================================ */

/* x[seq_len, dim], fc1_weight[hidden_dim, dim], fc1_bias[hidden_dim],
   fc2_weight[dim, hidden_dim], fc2_bias[dim]
   -> output[seq_len, dim], hidden_ws: pre-allocated seq_len*hidden_dim floats */
static void mlp_block(
    const float *x,
    int seq_len, int dim, int hidden_dim,
    const float *fc1_weight,
    const float *fc1_bias,
    const float *fc2_weight,
    const float *fc2_bias,
    float *hidden_ws,
    float *output
) {
    /* FC1 + GELU */
    linear(x, seq_len, dim, fc1_weight, fc1_bias, hidden_dim, hidden_ws);
    gelu_array(hidden_ws, seq_len * hidden_dim, hidden_ws);  /* self-aliased: safe */

    /* FC2 */
    linear(hidden_ws, seq_len, hidden_dim, fc2_weight, fc2_bias, dim, output);
}

/* ============================================ */
/* Utility Functions                            */
/* ============================================ */

static void topk(const float *data, int size, int k, int *indices, float *values) {
    for (int i = 0; i < k; i++) {
        int max_idx = 0;
        float max_val = -FLT_MAX;

        for (int j = 0; j < size; j++) {
            /* Skip already selected */
            int skip = 0;
            for (int l = 0; l < i; l++) {
                if (indices[l] == j) {
                    skip = 1;
                    break;
                }
            }
            if (skip) continue;

            if (data[j] > max_val) {
                max_val = data[j];
                max_idx = j;
            }
        }

        indices[i] = max_idx;
        values[i] = max_val;
    }
}

#endif /* NN_OPS_H */
