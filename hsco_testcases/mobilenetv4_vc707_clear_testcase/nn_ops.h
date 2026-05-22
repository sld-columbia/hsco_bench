/*
 * MobileNetV4 Small FPGA Baremetal Implementation
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

/* ============================================ */
/* Basic Activation Functions                   */
/* ============================================ */

static inline float relu(float x) {
    return x > 0 ? x : 0;
}

static inline float relu6(float x) {
    if (x < 0) return 0;
    if (x > 6) return 6;
    return x;
}

static inline float hard_swish(float x) {
    if (x <= -3) return 0;
    if (x >= 3) return x;
    return x * (x + 3) / 6;
}

/* ============================================ */
/* Convolution Operations                       */
/* ============================================ */

static void conv2d(
    const float *input, int in_c, int in_h, int in_w,
    const float *weight,
    int out_c, int kernel_h, int kernel_w,
    int stride, int padding,
    int out_h, int out_w,
    float *output
) {

    for (int oc = 0; oc < out_c; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float sum = 0.0f;

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

static void conv2d_depthwise(
    const float *input, int channels, int in_h, int in_w,
    const float *weight,
    int kernel_h, int kernel_w,
    int stride, int padding,
    int out_h, int out_w,
    float *output
) {

    for (int c = 0; c < channels; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float sum = 0.0f;

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
/* Batch Normalization                          */
/* ============================================ */

static void batch_norm(
    const float *input, int channels, int h, int w,
    const float *gamma, const float *beta,
    const float *mean, const float *var,
    float *output
) {
    const float eps = 1e-5f;  // Standard BatchNorm epsilon
    for (int c = 0; c < channels; c++) {
        float scale = gamma[c] / sqrtf_approx(var[c] + eps);
        float shift = beta[c] - mean[c] * scale;

        for (int i = 0; i < h * w; i++) {
            int idx = c * h * w + i;
            output[idx] = input[idx] * scale + shift;
        }
    }
}

/* ============================================ */
/* Pooling and Linear                           */
/* ============================================ */

static void global_avg_pool(const float *input, int channels, int spatial_size, float *output) {
    const float inv_spatial = 1.0f / spatial_size;
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        for (int i = 0; i < spatial_size; i++) {
            sum += input[c * spatial_size + i];
        }
        output[c] = sum * inv_spatial;
    }
}

static void linear(const float *input, const float *weight, const float *bias,
                  int in_features, int out_features, float *output) {
    for (int o = 0; o < out_features; o++) {
        float sum = bias[o];
        for (int i = 0; i < in_features; i++) {
            sum += input[i] * weight[o * in_features + i];
        }
        output[o] = sum;
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
/* Utility Functions                            */
/* ============================================ */

static void topk(const float *data, int size, int k, int *indices, float *values) {
    for (int i = 0; i < k; i++) {
        indices[i] = -1;
        values[i] = -FLT_MAX;
    }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < k; j++) {
            if (data[i] > values[j]) {
                for (int m = k - 1; m > j; m--) {
                    values[m] = values[m - 1];
                    indices[m] = indices[m - 1];
                }
                values[j] = data[i];
                indices[j] = i;
                break;
            }
        }
    }
}

static void add_arrays(const float *a, const float *b, int size, float *output) {
    for (int i = 0; i < size; i++) {
        output[i] = a[i] + b[i];
    }
}

static void relu_array(const float *input, int size, float *output) {
    for (int i = 0; i < size; i++) {
        output[i] = relu(input[i]);
    }
}

static void relu6_array(const float *input, int size, float *output) {
    for (int i = 0; i < size; i++) {
        output[i] = relu6(input[i]);
    }
}

static void hard_swish_array(const float *input, int size, float *output) {
    for (int i = 0; i < size; i++) {
        output[i] = hard_swish(input[i]);
    }
}

static void apply_activation(const float *input, int size, const char *act_type, float *output) {
    if (strcmp(act_type, "relu") == 0) {
        relu_array(input, size, output);
    } else if (strcmp(act_type, "relu6") == 0) {
        relu6_array(input, size, output);
    } else if (strcmp(act_type, "hard_swish") == 0) {
        hard_swish_array(input, size, output);
    }
}

#define ACT_NONE       0
#define ACT_RELU       1
#define ACT_RELU6      2
#define ACT_HARD_SWISH 3

static void apply_activation_id(const float *input, int size, int act_type, float *output) {
    switch (act_type) {
        case ACT_RELU:       relu_array(input, size, output); break;
        case ACT_RELU6:      relu6_array(input, size, output); break;
        case ACT_HARD_SWISH: hard_swish_array(input, size, output); break;
        default: break;
    }
}

/* Debug: print tensor summary for comparison */
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
