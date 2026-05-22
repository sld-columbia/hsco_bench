/*
 * Autoencoder FPGA Baremetal Implementation
 * Neural Network Operations for Dense Autoencoder
 */

#ifndef NN_OPS_H
#define NN_OPS_H

#include "tensor.h"
#include <float.h>

static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

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

static void linear(
    const float *input, int in_features,
    const float *weight, const float *bias, int out_features,
    float *output
) {
    for (int i = 0; i < out_features; i++) {
        float sum = bias[i];
        for (int j = 0; j < in_features; j++) {
            sum += input[j] * weight[i * in_features + j];
        }
        output[i] = sum;
    }
}

static void batchnorm1d(
    const float *input, int features,
    const float *gamma, const float *beta,
    const float *running_mean, const float *running_var,
    float *output
) {
    float eps = 1e-5f;
    for (int i = 0; i < features; i++) {
        float scale = gamma[i] / sqrtf_approx(running_var[i] + eps);
        float bias = beta[i] - running_mean[i] * scale;
        output[i] = input[i] * scale + bias;
    }
}

static void relu_array(const float *input, int size, float *output) {
    for (int i = 0; i < size; i++) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}

static void dense_bn_relu(
    const float *input, int in_features,
    const float *weight, const float *bias_linear, int out_features,
    const float *bn_gamma, const float *bn_beta,
    const float *bn_mean, const float *bn_var,
    int apply_relu,
    float *temp_linear, float *temp_bn, float *output
) {
    linear(input, in_features, weight, bias_linear, out_features, temp_linear);
    if (apply_relu) {
        batchnorm1d(temp_linear, out_features, bn_gamma, bn_beta, bn_mean, bn_var, temp_bn);
        relu_array(temp_bn, out_features, output);
    } else {
        batchnorm1d(temp_linear, out_features, bn_gamma, bn_beta, bn_mean, bn_var, output);
    }
}

static float compute_mse(const float *a, const float *b, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum / size;
}

static void copy_tensor(float *dst, const float *src, int size) {
    memcpy(dst, src, size * sizeof(float));
}

#endif // NN_OPS_H
