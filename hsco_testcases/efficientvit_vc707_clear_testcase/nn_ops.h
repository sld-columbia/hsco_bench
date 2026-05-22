/*
 * EfficientViT-M0 FPGA Baremetal Implementation
 * Neural Network Operations
 *
 * FPGA Baremetal Notes:
 * - Uses sqrtf_approx / expf_approx
 * - All memory allocation via aligned_malloc (not malloc)
 */

#ifndef NN_OPS_H
#define NN_OPS_H

#include "tensor.h"
#include <float.h>

/* ============================================ */
/* Basic Operations                             */
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

static inline float sigmoid_scalar(float x) {
    return 1.0f / (1.0f + expf_approx(-x));
}

/* ============================================ */
/* Conv2D with bias (fused conv+bn for EfficientViT) */
/* ============================================ */

/* Standard Conv2D: input[C_in, H, W] -> output[C_out, H_out, W_out] */
static void conv2d(
    const float *input, int in_c, int in_h, int in_w,
    const float *weight, const float *bias,
    int out_c, int kernel_h, int kernel_w,
    int stride, int padding,
    int out_h, int out_w,
    float *output
) {
    
    for (int oc = 0; oc < out_c; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float sum = bias ? bias[oc] : 0.0f;
                
                for (int ic = 0; ic < in_c; ic++) {
                    for (int kh = 0; kh < kernel_h; kh++) {
                        for (int kw = 0; kw < kernel_w; kw++) {
                            int ih = oh * stride - padding + kh;
                            int iw = ow * stride - padding + kw;
                            
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                int in_idx = ic * in_h * in_w + ih * in_w + iw;
                                int w_idx = oc * in_c * kernel_h * kernel_w + 
                                           ic * kernel_h * kernel_w + 
                                           kh * kernel_w + kw;
                                sum += input[in_idx] * weight[w_idx];
                            }
                        }
                    }
                }
                
                output[oc * out_h * out_w + oh * out_w + ow] = sum;
            }
        }
    }
}

/* Depthwise Conv2D: input[C, H, W] -> output[C, H_out, W_out] */
static void conv2d_depthwise(
    const float *input, int channels, int in_h, int in_w,
    const float *weight, const float *bias,
    int kernel_h, int kernel_w,
    int stride, int padding,
    int out_h, int out_w,
    float *output
) {
    
    for (int c = 0; c < channels; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float sum = bias ? bias[c] : 0.0f;
                
                for (int kh = 0; kh < kernel_h; kh++) {
                    for (int kw = 0; kw < kernel_w; kw++) {
                        int ih = oh * stride - padding + kh;
                        int iw = ow * stride - padding + kw;
                        
                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            int in_idx = c * in_h * in_w + ih * in_w + iw;
                            int w_idx = c * kernel_h * kernel_w + kh * kernel_w + kw;
                            sum += input[in_idx] * weight[w_idx];
                        }
                    }
                }
                
                output[c * out_h * out_w + oh * out_w + ow] = sum;
            }
        }
    }
}

/* ============================================ */
/* Activation Functions                         */
/* ============================================ */

static void relu(const float *input, int size, float *output) {
    for (int i = 0; i < size; i++) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}

static void sigmoid(const float *input, int size, float *output) {
    for (int i = 0; i < size; i++) {
        output[i] = 1.0f / (1.0f + expf_approx(-input[i]));
    }
}

/* ============================================ */
/* Element-wise Operations                      */
/* ============================================ */

static void add_elementwise(const float *a, const float *b, int size, float *output) {
    for (int i = 0; i < size; i++) {
        output[i] = a[i] + b[i];
    }
}

static void multiply_elementwise(const float *a, const float *b, int size, float *output) {
    for (int i = 0; i < size; i++) {
        output[i] = a[i] * b[i];
    }
}

/* ============================================ */
/* Global Average Pooling                       */
/* ============================================ */

/* input[C, H, W] -> output[C] */
static void global_avg_pool(const float *input, int c, int h, int w, float *output) {
    int spatial_size = h * w;
    for (int i = 0; i < c; i++) {
        float sum = 0.0f;
        for (int j = 0; j < spatial_size; j++) {
            sum += input[i * spatial_size + j];
        }
        output[i] = sum / spatial_size;
    }
}

/* ============================================ */
/* Squeeze-Excite Block                         */
/* ============================================ */

/* SE: input[C, H, W] -> output[C, H, W] */
static void squeeze_excite(
    const float *input, int c, int h, int w,
    const float *fc1_w, const float *fc1_b, int squeeze_c,
    const float *fc2_w, const float *fc2_b,
    float *squeezed,
    float *excited,
    float *scale,
    float *output
) {
    int spatial_size = h * w;
    
    /* Global average pooling -> [C] (inlined) */
    for (int i = 0; i < c; i++) {
        float sum = 0.0f;
        for (int j = 0; j < spatial_size; j++) {
            sum += input[i * spatial_size + j];
        }
        squeezed[i] = sum / spatial_size;
    }
    
    /* FC1: [C] -> [squeeze_c] with ReLU */
    for (int i = 0; i < squeeze_c; i++) {
        float sum = fc1_b[i];
        for (int j = 0; j < c; j++) {
            /* fc1_w is [squeeze_c, c, 1, 1], treat as [squeeze_c, c] */
            sum += squeezed[j] * fc1_w[i * c + j];
        }
        excited[i] = sum > 0.0f ? sum : 0.0f;
    }
    
    /* FC2: [squeeze_c] -> [C] with Sigmoid */
    for (int i = 0; i < c; i++) {
        float sum = fc2_b[i];
        for (int j = 0; j < squeeze_c; j++) {
            sum += excited[j] * fc2_w[i * squeeze_c + j];
        }
        scale[i] = sigmoid_scalar(sum);
    }
    
    /* Scale input -> output */
    for (int i = 0; i < c; i++) {
        for (int j = 0; j < spatial_size; j++) {
            output[i * spatial_size + j] = input[i * spatial_size + j] * scale[i];
        }
    }
}

/* ============================================ */
/* Linear Layer                                 */
/* ============================================ */

/* input[in_features] -> output[out_features] */
static void linear(
    const float *input, int in_features,
    const float *weight, const float *bias, int out_features,
    float *output
) {
    for (int i = 0; i < out_features; i++) {
        float sum = bias ? bias[i] : 0.0f;
        for (int j = 0; j < in_features; j++) {
            sum += input[j] * weight[i * in_features + j];
        }
        output[i] = sum;
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
    
    /* Compute exp and sum */
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = expf_approx(input[i] - max_val);
        sum += output[i];
    }
    
    /* Normalize */
    if (sum > 0.0f) {
        for (int i = 0; i < size; i++) {
            output[i] /= sum;
        }
    }
}

/* ============================================ */
/* Attention Operations                         */
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

/* Transpose: input[M, N] -> output[N, M] */
static void transpose_2d(const float *input, float *output, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            output[j * M + i] = input[i * N + j];
        }
    }
}

/* Softmax along last dimension for attention */
static void softmax_2d(float *data, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        softmax(data + i * cols, cols, data + i * cols);
    }
}

/* ============================================ */
/* Utility Functions                            */
/* ============================================ */

/* Find top-k indices */
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
