/*
 * Autoencoder FPGA Baremetal Implementation
 * Dense Autoencoder Forward Pass
 * 
 * Architecture: input(640) -> 128 -> 128 -> 128 -> 128 -> 8 -> 128 -> 128 -> 128 -> 128 -> output(640)
 * Each layer has BatchNorm1d and ReLU (except final output layer)
 */

#ifndef AUTOENCODER_H
#define AUTOENCODER_H

#include "tensor.h"
#include "nn_ops.h"
// Accelerator drivers are included from systest.c before this file

#define INPUT_DIM 640
#define HIDDEN_DIM 128
#define BOTTLENECK_DIM 8
#define BN_EPS 1e-5f

typedef struct {
    float *buffer_a;
    float *buffer_b;
    float *temp_buffer;
    float *temp_buffer2;
} WorkBuffers;

static WorkBuffers* alloc_work_buffers(void) {
    WorkBuffers *buffers = (WorkBuffers*)aligned_malloc(sizeof(WorkBuffers));
    if (!buffers) return NULL;
    
    buffers->buffer_a = alloc_tensor_1d(INPUT_DIM);
    buffers->buffer_b = alloc_tensor_1d(INPUT_DIM);
    buffers->temp_buffer = alloc_tensor_1d(INPUT_DIM);
    buffers->temp_buffer2 = alloc_tensor_1d(INPUT_DIM);
    
    if (!buffers->buffer_a || !buffers->buffer_b || !buffers->temp_buffer || !buffers->temp_buffer2) {
        if (buffers->buffer_a) aligned_free(buffers->buffer_a);
        if (buffers->buffer_b) aligned_free(buffers->buffer_b);
        if (buffers->temp_buffer) aligned_free(buffers->temp_buffer);
        if (buffers->temp_buffer2) aligned_free(buffers->temp_buffer2);
        aligned_free(buffers);
        return NULL;
    }
    
    return buffers;
}

static void free_work_buffers(WorkBuffers *buffers) {
    if (buffers) {
        if (buffers->buffer_a) aligned_free(buffers->buffer_a);
        if (buffers->buffer_b) aligned_free(buffers->buffer_b);
        if (buffers->temp_buffer) aligned_free(buffers->temp_buffer);
        if (buffers->temp_buffer2) aligned_free(buffers->temp_buffer2);
        aligned_free(buffers);
    }
}

static void autoencoder_forward(const float *input, Model *model, WorkBuffers *buffers, float *output) {
    /* ================================================================
     * Phase 1: Resolve all weight pointers (data loading)
     * ================================================================ */

    // Encoder Layer 0: input(640) -> 128
    const float *enc0_w    = find_tensor(model, "encoder.0.weight")->data;
    const float *enc0_b    = find_tensor(model, "encoder.0.bias")->data;
    const float *enc0_bn_w = find_tensor(model, "encoder.1.weight")->data;
    const float *enc0_bn_b = find_tensor(model, "encoder.1.bias")->data;
    const float *enc0_bn_m = find_tensor(model, "encoder.1.running_mean")->data;
    const float *enc0_bn_v = find_tensor(model, "encoder.1.running_var")->data;

    // Encoder Layer 1: 128 -> 128
    const float *enc1_w    = find_tensor(model, "encoder.3.weight")->data;
    const float *enc1_b    = find_tensor(model, "encoder.3.bias")->data;
    const float *enc1_bn_w = find_tensor(model, "encoder.4.weight")->data;
    const float *enc1_bn_b = find_tensor(model, "encoder.4.bias")->data;
    const float *enc1_bn_m = find_tensor(model, "encoder.4.running_mean")->data;
    const float *enc1_bn_v = find_tensor(model, "encoder.4.running_var")->data;

    // Encoder Layer 2: 128 -> 128
    const float *enc2_w    = find_tensor(model, "encoder.6.weight")->data;
    const float *enc2_b    = find_tensor(model, "encoder.6.bias")->data;
    const float *enc2_bn_w = find_tensor(model, "encoder.7.weight")->data;
    const float *enc2_bn_b = find_tensor(model, "encoder.7.bias")->data;
    const float *enc2_bn_m = find_tensor(model, "encoder.7.running_mean")->data;
    const float *enc2_bn_v = find_tensor(model, "encoder.7.running_var")->data;

    // Encoder Layer 3: 128 -> 128
    const float *enc3_w    = find_tensor(model, "encoder.9.weight")->data;
    const float *enc3_b    = find_tensor(model, "encoder.9.bias")->data;
    const float *enc3_bn_w = find_tensor(model, "encoder.10.weight")->data;
    const float *enc3_bn_b = find_tensor(model, "encoder.10.bias")->data;
    const float *enc3_bn_m = find_tensor(model, "encoder.10.running_mean")->data;
    const float *enc3_bn_v = find_tensor(model, "encoder.10.running_var")->data;

    // Encoder Layer 4 (bottleneck): 128 -> 8
    const float *enc4_w    = find_tensor(model, "encoder.12.weight")->data;
    const float *enc4_b    = find_tensor(model, "encoder.12.bias")->data;
    const float *enc4_bn_w = find_tensor(model, "encoder.13.weight")->data;
    const float *enc4_bn_b = find_tensor(model, "encoder.13.bias")->data;
    const float *enc4_bn_m = find_tensor(model, "encoder.13.running_mean")->data;
    const float *enc4_bn_v = find_tensor(model, "encoder.13.running_var")->data;

    // Decoder Layer 0: 8 -> 128
    const float *dec0_w    = find_tensor(model, "decoder.0.weight")->data;
    const float *dec0_b    = find_tensor(model, "decoder.0.bias")->data;
    const float *dec0_bn_w = find_tensor(model, "decoder.1.weight")->data;
    const float *dec0_bn_b = find_tensor(model, "decoder.1.bias")->data;
    const float *dec0_bn_m = find_tensor(model, "decoder.1.running_mean")->data;
    const float *dec0_bn_v = find_tensor(model, "decoder.1.running_var")->data;

    // Decoder Layer 1: 128 -> 128
    const float *dec1_w    = find_tensor(model, "decoder.3.weight")->data;
    const float *dec1_b    = find_tensor(model, "decoder.3.bias")->data;
    const float *dec1_bn_w = find_tensor(model, "decoder.4.weight")->data;
    const float *dec1_bn_b = find_tensor(model, "decoder.4.bias")->data;
    const float *dec1_bn_m = find_tensor(model, "decoder.4.running_mean")->data;
    const float *dec1_bn_v = find_tensor(model, "decoder.4.running_var")->data;

    // Decoder Layer 2: 128 -> 128
    const float *dec2_w    = find_tensor(model, "decoder.6.weight")->data;
    const float *dec2_b    = find_tensor(model, "decoder.6.bias")->data;
    const float *dec2_bn_w = find_tensor(model, "decoder.7.weight")->data;
    const float *dec2_bn_b = find_tensor(model, "decoder.7.bias")->data;
    const float *dec2_bn_m = find_tensor(model, "decoder.7.running_mean")->data;
    const float *dec2_bn_v = find_tensor(model, "decoder.7.running_var")->data;

    // Decoder Layer 3: 128 -> 128
    const float *dec3_w    = find_tensor(model, "decoder.9.weight")->data;
    const float *dec3_b    = find_tensor(model, "decoder.9.bias")->data;
    const float *dec3_bn_w = find_tensor(model, "decoder.10.weight")->data;
    const float *dec3_bn_b = find_tensor(model, "decoder.10.bias")->data;
    const float *dec3_bn_m = find_tensor(model, "decoder.10.running_mean")->data;
    const float *dec3_bn_v = find_tensor(model, "decoder.10.running_var")->data;

    // Decoder Layer 4 (output): 128 -> 640 (no BatchNorm, no ReLU)
    const float *dec4_w = find_tensor(model, "decoder.12.weight")->data;
    const float *dec4_b = find_tensor(model, "decoder.12.bias")->data;

    
    float *buf_a = buffers->buffer_a;
    float *buf_b = buffers->buffer_b;
    float *tmp1  = buffers->temp_buffer;
    float *tmp2  = buffers->temp_buffer2;

    /* ================================================================
     * Phase 2: Compute
     * ================================================================ */

    // Encoder Layer 0: input(640) -> 128
    dense_bn_relu(input, INPUT_DIM, enc0_w, enc0_b, HIDDEN_DIM,
                  enc0_bn_w, enc0_bn_b, enc0_bn_m, enc0_bn_v,
                  1, tmp1, tmp2, buf_a);

    // Encoder Layer 1: 128 -> 128
    dense_bn_relu(buf_a, HIDDEN_DIM, enc1_w, enc1_b, HIDDEN_DIM,
                  enc1_bn_w, enc1_bn_b, enc1_bn_m, enc1_bn_v,
                  1, tmp1, tmp2, buf_b);

    // Encoder Layer 2: 128 -> 128
    dense_bn_relu(buf_b, HIDDEN_DIM, enc2_w, enc2_b, HIDDEN_DIM,
                  enc2_bn_w, enc2_bn_b, enc2_bn_m, enc2_bn_v,
                  1, tmp1, tmp2, buf_a);

    // Encoder Layer 3: 128 -> 128
    dense_bn_relu(buf_a, HIDDEN_DIM, enc3_w, enc3_b, HIDDEN_DIM,
                  enc3_bn_w, enc3_bn_b, enc3_bn_m, enc3_bn_v,
                  1, tmp1, tmp2, buf_b);

    // Encoder Layer 4 (bottleneck): 128 -> 8
    dense_bn_relu(buf_b, HIDDEN_DIM, enc4_w, enc4_b, BOTTLENECK_DIM,
                  enc4_bn_w, enc4_bn_b, enc4_bn_m, enc4_bn_v,
                  1, tmp1, tmp2, buf_a);

    // Decoder Layer 0: 8 -> 128
    dense_bn_relu(buf_a, BOTTLENECK_DIM, dec0_w, dec0_b, HIDDEN_DIM,
                  dec0_bn_w, dec0_bn_b, dec0_bn_m, dec0_bn_v,
                  1, tmp1, tmp2, buf_b);

    // Decoder Layer 1: 128 -> 128
    dense_bn_relu(buf_b, HIDDEN_DIM, dec1_w, dec1_b, HIDDEN_DIM,
                  dec1_bn_w, dec1_bn_b, dec1_bn_m, dec1_bn_v,
                  1, tmp1, tmp2, buf_a);

    // Decoder Layer 2: 128 -> 128
    dense_bn_relu(buf_a, HIDDEN_DIM, dec2_w, dec2_b, HIDDEN_DIM,
                  dec2_bn_w, dec2_bn_b, dec2_bn_m, dec2_bn_v,
                  1, tmp1, tmp2, buf_b);

    // Decoder Layer 3: 128 -> 128
    dense_bn_relu(buf_b, HIDDEN_DIM, dec3_w, dec3_b, HIDDEN_DIM,
                  dec3_bn_w, dec3_bn_b, dec3_bn_m, dec3_bn_v,
                  1, tmp1, tmp2, buf_a);

    // Decoder Layer 4 (output): 128 -> 640 (no BatchNorm, no ReLU)
    linear(buf_a, HIDDEN_DIM, dec4_w, dec4_b, INPUT_DIM, output);
}

static float compute_anomaly_score(const float *input, const float *output, int size) {
    return compute_mse(input, output, size);
}

#endif // AUTOENCODER_H
