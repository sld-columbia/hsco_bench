/*
 * MobileNetV4 Small FPGA Baremetal Implementation
 * Complete forward pass matching PyTorch timm implementation
 * Model: mobilenetv4_conv_small.e2400_r224_in1k
 *
 * FPGA Baremetal Notes:
 * - All dynamic allocations use aligned_malloc / aligned_free (not malloc/free)
 * - Large structs must NOT be on stack (stack is very small in baremetal)
 * - BSS is NOT zeroed on Ariane; use explicit init for static variables
 * - No snprintf available; use str_concat / str_build helpers
 */

#ifndef MOBILENETV4_H
#define MOBILENETV4_H

#include "nn_ops.h"

/* ============================================ */
/* String building helpers for baremetal        */
/* ============================================ */

/* Simple string concatenation */
static void str_concat(char *dest, const char *s1, const char *s2) {
    while (*s1) *dest++ = *s1++;
    while (*s2) *dest++ = *s2++;
    *dest = '\0';
}

/* Integer to string (non-negative only) */
static void int_to_str(char *buf, int n) {
    if (n == 0) {
        buf[0] = '0';
        buf[1] = '\0';
        return;
    }
    char tmp[12];
    int len = 0;
    while (n > 0) {
        tmp[len++] = '0' + (n % 10);
        n /= 10;
    }
    for (int i = 0; i < len; i++) {
        buf[i] = tmp[len - 1 - i];
    }
    buf[len] = '\0';
}

/* Concatenate 3 strings: s1 + s2 + s3 */
static void str_build3(char *dest, const char *s1, const char *s2, const char *s3) {
    while (*s1) *dest++ = *s1++;
    while (*s2) *dest++ = *s2++;
    while (*s3) *dest++ = *s3++;
    *dest = '\0';
}

/* Concatenate 4 strings: s1 + s2 + s3 + s4 */
static void str_build4(char *dest, const char *s1, const char *s2, const char *s3, const char *s4) {
    while (*s1) *dest++ = *s1++;
    while (*s2) *dest++ = *s2++;
    while (*s3) *dest++ = *s3++;
    while (*s4) *dest++ = *s4++;
    *dest = '\0';
}

/* ============================================================================ */
/* Helper: Conv + BatchNorm + Activation (fused for inference)                  */
/* ============================================================================ */
static void conv_bn_act(
    const float *input, int in_c, int h, int w,
    const float *conv_weight,
    const float *bn_gamma, const float *bn_beta,
    const float *bn_mean, const float *bn_var,
    int out_c, int kernel, int stride, int groups,
    int act_type,
    float *output
) {
    int padding = kernel / 2;
    int out_h = (h + 2 * padding - kernel) / stride + 1;
    int out_w = (w + 2 * padding - kernel) / stride + 1;

    if (groups == 1) {
        conv2d(input, in_c, h, w, conv_weight,
               out_c, kernel, kernel, stride, padding, out_h, out_w, output);
    } else {
        conv2d_depthwise(input, in_c, h, w, conv_weight,
                        kernel, kernel, stride, padding, out_h, out_w, output);
    }

    batch_norm(output, out_c, out_h, out_w,
              bn_gamma, bn_beta, bn_mean, bn_var, output);

    if (act_type == ACT_RELU) {
        relu_array(output, out_c * out_h * out_w, output);
    } else if (act_type == ACT_RELU6) {
        relu6_array(output, out_c * out_h * out_w, output);
    } else if (act_type == ACT_HARD_SWISH) {
        hard_swish_array(output, out_c * out_h * out_w, output);
    }
}

/* ============================================================================ */
/* Main Forward Pass                                                            */
/* ============================================================================ */
void mobilenetv4_forward(const float *input, Model *model, float *output) {
    printf("  Running MobileNetV4 Small forward pass...\n");

#define T(name) find_tensor(model, name)->data

    /* Pre-allocate all buffers */
    int buf_size = 96 * 112 * 112;
    int temp_size = 576 * 56 * 56;

    float *buf1 = alloc_tensor(buf_size);
    float *buf2 = alloc_tensor(buf_size);
    float *temp1 = alloc_tensor(temp_size);
    float *temp2 = alloc_tensor(temp_size);
    float *pooled = alloc_tensor(960);
    float *head_out = alloc_tensor(1280);
    float *zero_bias_1280 = alloc_tensor(1280);

    if (!buf1 || !buf2 || !temp1 || !temp2 || !pooled || !head_out || !zero_bias_1280) {
        printf("Error: Failed to allocate buffers\n");
        return;
    }

    const float *current = input;
    float *next = buf1;
    int h = 224, w = 224, c = 3;

    /* ======================================================================== */
    /* STEM: conv_stem + bn1 + ReLU                                             */
    /* [3, 224, 224] -> [32, 112, 112]                                          */
    /* ======================================================================== */
    printf("  - Stem: conv_stem (3x3 s2) + bn1 + ReLU\n");
    conv_bn_act(current, 3, 224, 224,
        T("conv_stem.weight"),
        T("bn1.weight"), T("bn1.bias"), T("bn1.running_mean"), T("bn1.running_var"),
        32, 3, 2, 1, ACT_RELU, next);
    current = next;
    next = (next == buf1) ? buf2 : buf1;
    h = 112; w = 112; c = 32;

    /* ======================================================================== */
    /* STAGE 0 (blocks.0): Two ConvBnAct blocks                                 */
    /* ======================================================================== */
    printf("  - Stage 0: blocks.0.0 (3x3 dw) + blocks.0.1 (1x1)\n");
    conv_bn_act(current, 32, 112, 112,
        T("blocks.0.0.conv.weight"),
        T("blocks.0.0.bn1.weight"), T("blocks.0.0.bn1.bias"),
        T("blocks.0.0.bn1.running_mean"), T("blocks.0.0.bn1.running_var"),
        32, 3, 2, 1, ACT_RELU, next);
    current = next;
    next = (next == buf1) ? buf2 : buf1;

    conv_bn_act(current, 32, 56, 56,
        T("blocks.0.1.conv.weight"),
        T("blocks.0.1.bn1.weight"), T("blocks.0.1.bn1.bias"),
        T("blocks.0.1.bn1.running_mean"), T("blocks.0.1.bn1.running_var"),
        32, 1, 1, 1, ACT_RELU, next);
    current = next;
    next = (next == buf1) ? buf2 : buf1;

    /* ======================================================================== */
    /* STAGE 1 (blocks.1): Two ConvBnAct blocks                                 */
    /* ======================================================================== */
    printf("  - Stage 1: blocks.1.0 (3x3 s2) + blocks.1.1 (1x1)\n");
    conv_bn_act(current, 32, 56, 56,
        T("blocks.1.0.conv.weight"),
        T("blocks.1.0.bn1.weight"), T("blocks.1.0.bn1.bias"),
        T("blocks.1.0.bn1.running_mean"), T("blocks.1.0.bn1.running_var"),
        96, 3, 2, 1, ACT_RELU, next);
    current = next;
    next = (next == buf1) ? buf2 : buf1;

    conv_bn_act(current, 96, 28, 28,
        T("blocks.1.1.conv.weight"),
        T("blocks.1.1.bn1.weight"), T("blocks.1.1.bn1.bias"),
        T("blocks.1.1.bn1.running_mean"), T("blocks.1.1.bn1.running_var"),
        64, 1, 1, 1, ACT_RELU, next);
    current = next;
    next = (next == buf1) ? buf2 : buf1;

    /* ======================================================================== */
    /* STAGE 2 (blocks.2): Expanded UIR blocks                                  */
    /* ======================================================================== */
    printf("  - Stage 2: blocks.2.x (expanded)\n");

    /* blocks.2.0 */
    conv_bn_act(current, 64, 28, 28,
        T("blocks.2.0.dw_start.conv.weight"),
        T("blocks.2.0.dw_start.bn.weight"), T("blocks.2.0.dw_start.bn.bias"),
        T("blocks.2.0.dw_start.bn.running_mean"), T("blocks.2.0.dw_start.bn.running_var"),
        64, 5, 1, 64, ACT_NONE, temp1);
    conv_bn_act(temp1, 64, 28, 28,
        T("blocks.2.0.pw_exp.conv.weight"),
        T("blocks.2.0.pw_exp.bn.weight"), T("blocks.2.0.pw_exp.bn.bias"),
        T("blocks.2.0.pw_exp.bn.running_mean"), T("blocks.2.0.pw_exp.bn.running_var"),
        192, 1, 1, 1, ACT_RELU, temp2);
    conv_bn_act(temp2, 192, 28, 28,
        T("blocks.2.0.dw_mid.conv.weight"),
        T("blocks.2.0.dw_mid.bn.weight"), T("blocks.2.0.dw_mid.bn.bias"),
        T("blocks.2.0.dw_mid.bn.running_mean"), T("blocks.2.0.dw_mid.bn.running_var"),
        192, 5, 2, 192, ACT_RELU, temp1);
    conv_bn_act(temp1, 192, 14, 14,
        T("blocks.2.0.pw_proj.conv.weight"),
        T("blocks.2.0.pw_proj.bn.weight"), T("blocks.2.0.pw_proj.bn.bias"),
        T("blocks.2.0.pw_proj.bn.running_mean"), T("blocks.2.0.pw_proj.bn.running_var"),
        96, 1, 1, 1, ACT_NONE, next);
    current = next;
    next = (next == buf1) ? buf2 : buf1;

    /* blocks.2.1 */
    conv_bn_act(current, 96, 14, 14,
        T("blocks.2.1.pw_exp.conv.weight"),
        T("blocks.2.1.pw_exp.bn.weight"), T("blocks.2.1.pw_exp.bn.bias"),
        T("blocks.2.1.pw_exp.bn.running_mean"), T("blocks.2.1.pw_exp.bn.running_var"),
        192, 1, 1, 1, ACT_RELU, temp1);
    conv_bn_act(temp1, 192, 14, 14,
        T("blocks.2.1.dw_mid.conv.weight"),
        T("blocks.2.1.dw_mid.bn.weight"), T("blocks.2.1.dw_mid.bn.bias"),
        T("blocks.2.1.dw_mid.bn.running_mean"), T("blocks.2.1.dw_mid.bn.running_var"),
        192, 3, 1, 192, ACT_RELU, temp2);
    conv_bn_act(temp2, 192, 14, 14,
        T("blocks.2.1.pw_proj.conv.weight"),
        T("blocks.2.1.pw_proj.bn.weight"), T("blocks.2.1.pw_proj.bn.bias"),
        T("blocks.2.1.pw_proj.bn.running_mean"), T("blocks.2.1.pw_proj.bn.running_var"),
        96, 1, 1, 1, ACT_NONE, next);
    add_arrays(next, current, 96 * 14 * 14, next);
    current = next;
    next = (next == buf1) ? buf2 : buf1;

    /* blocks.2.2 */
    conv_bn_act(current, 96, 14, 14,
        T("blocks.2.2.pw_exp.conv.weight"),
        T("blocks.2.2.pw_exp.bn.weight"), T("blocks.2.2.pw_exp.bn.bias"),
        T("blocks.2.2.pw_exp.bn.running_mean"), T("blocks.2.2.pw_exp.bn.running_var"),
        192, 1, 1, 1, ACT_RELU, temp1);
    conv_bn_act(temp1, 192, 14, 14,
        T("blocks.2.2.dw_mid.conv.weight"),
        T("blocks.2.2.dw_mid.bn.weight"), T("blocks.2.2.dw_mid.bn.bias"),
        T("blocks.2.2.dw_mid.bn.running_mean"), T("blocks.2.2.dw_mid.bn.running_var"),
        192, 3, 1, 192, ACT_RELU, temp2);
    conv_bn_act(temp2, 192, 14, 14,
        T("blocks.2.2.pw_proj.conv.weight"),
        T("blocks.2.2.pw_proj.bn.weight"), T("blocks.2.2.pw_proj.bn.bias"),
        T("blocks.2.2.pw_proj.bn.running_mean"), T("blocks.2.2.pw_proj.bn.running_var"),
        96, 1, 1, 1, ACT_NONE, next);
    add_arrays(next, current, 96 * 14 * 14, next);
    current = next;
    next = (next == buf1) ? buf2 : buf1;

    /* blocks.2.3 */
    conv_bn_act(current, 96, 14, 14,
        T("blocks.2.3.pw_exp.conv.weight"),
        T("blocks.2.3.pw_exp.bn.weight"), T("blocks.2.3.pw_exp.bn.bias"),
        T("blocks.2.3.pw_exp.bn.running_mean"), T("blocks.2.3.pw_exp.bn.running_var"),
        192, 1, 1, 1, ACT_RELU, temp1);
    conv_bn_act(temp1, 192, 14, 14,
        T("blocks.2.3.dw_mid.conv.weight"),
        T("blocks.2.3.dw_mid.bn.weight"), T("blocks.2.3.dw_mid.bn.bias"),
        T("blocks.2.3.dw_mid.bn.running_mean"), T("blocks.2.3.dw_mid.bn.running_var"),
        192, 3, 1, 192, ACT_RELU, temp2);
    conv_bn_act(temp2, 192, 14, 14,
        T("blocks.2.3.pw_proj.conv.weight"),
        T("blocks.2.3.pw_proj.bn.weight"), T("blocks.2.3.pw_proj.bn.bias"),
        T("blocks.2.3.pw_proj.bn.running_mean"), T("blocks.2.3.pw_proj.bn.running_var"),
        96, 1, 1, 1, ACT_NONE, next);
    add_arrays(next, current, 96 * 14 * 14, next);
    current = next;
    next = (next == buf1) ? buf2 : buf1;

    /* blocks.2.4 */
    conv_bn_act(current, 96, 14, 14,
        T("blocks.2.4.pw_exp.conv.weight"),
        T("blocks.2.4.pw_exp.bn.weight"), T("blocks.2.4.pw_exp.bn.bias"),
        T("blocks.2.4.pw_exp.bn.running_mean"), T("blocks.2.4.pw_exp.bn.running_var"),
        192, 1, 1, 1, ACT_RELU, temp1);
    conv_bn_act(temp1, 192, 14, 14,
        T("blocks.2.4.dw_mid.conv.weight"),
        T("blocks.2.4.dw_mid.bn.weight"), T("blocks.2.4.dw_mid.bn.bias"),
        T("blocks.2.4.dw_mid.bn.running_mean"), T("blocks.2.4.dw_mid.bn.running_var"),
        192, 3, 1, 192, ACT_RELU, temp2);
    conv_bn_act(temp2, 192, 14, 14,
        T("blocks.2.4.pw_proj.conv.weight"),
        T("blocks.2.4.pw_proj.bn.weight"), T("blocks.2.4.pw_proj.bn.bias"),
        T("blocks.2.4.pw_proj.bn.running_mean"), T("blocks.2.4.pw_proj.bn.running_var"),
        96, 1, 1, 1, ACT_NONE, next);
    add_arrays(next, current, 96 * 14 * 14, next);
    current = next;
    next = (next == buf1) ? buf2 : buf1;

    /* blocks.2.5 */
    conv_bn_act(current, 96, 14, 14,
        T("blocks.2.5.dw_start.conv.weight"),
        T("blocks.2.5.dw_start.bn.weight"), T("blocks.2.5.dw_start.bn.bias"),
        T("blocks.2.5.dw_start.bn.running_mean"), T("blocks.2.5.dw_start.bn.running_var"),
        96, 3, 1, 96, ACT_NONE, temp1);
    conv_bn_act(temp1, 96, 14, 14,
        T("blocks.2.5.pw_exp.conv.weight"),
        T("blocks.2.5.pw_exp.bn.weight"), T("blocks.2.5.pw_exp.bn.bias"),
        T("blocks.2.5.pw_exp.bn.running_mean"), T("blocks.2.5.pw_exp.bn.running_var"),
        384, 1, 1, 1, ACT_RELU, temp2);
    conv_bn_act(temp2, 384, 14, 14,
        T("blocks.2.5.pw_proj.conv.weight"),
        T("blocks.2.5.pw_proj.bn.weight"), T("blocks.2.5.pw_proj.bn.bias"),
        T("blocks.2.5.pw_proj.bn.running_mean"), T("blocks.2.5.pw_proj.bn.running_var"),
        96, 1, 1, 1, ACT_NONE, next);
    add_arrays(next, current, 96 * 14 * 14, next);
    current = next;
    next = (next == buf1) ? buf2 : buf1;

    /* ======================================================================== */
    /* STAGE 3 (blocks.3): Expanded UIR blocks                                  */
    /* ======================================================================== */
    printf("  - Stage 3: blocks.3.x (expanded)\n");

    /* blocks.3.0 */
    conv_bn_act(current, 96, 14, 14,
        T("blocks.3.0.dw_start.conv.weight"),
        T("blocks.3.0.dw_start.bn.weight"), T("blocks.3.0.dw_start.bn.bias"),
        T("blocks.3.0.dw_start.bn.running_mean"), T("blocks.3.0.dw_start.bn.running_var"),
        96, 3, 1, 96, ACT_NONE, temp1);
    conv_bn_act(temp1, 96, 14, 14,
        T("blocks.3.0.pw_exp.conv.weight"),
        T("blocks.3.0.pw_exp.bn.weight"), T("blocks.3.0.pw_exp.bn.bias"),
        T("blocks.3.0.pw_exp.bn.running_mean"), T("blocks.3.0.pw_exp.bn.running_var"),
        576, 1, 1, 1, ACT_RELU, temp2);
    conv_bn_act(temp2, 576, 14, 14,
        T("blocks.3.0.dw_mid.conv.weight"),
        T("blocks.3.0.dw_mid.bn.weight"), T("blocks.3.0.dw_mid.bn.bias"),
        T("blocks.3.0.dw_mid.bn.running_mean"), T("blocks.3.0.dw_mid.bn.running_var"),
        576, 3, 2, 576, ACT_RELU, temp1);
    conv_bn_act(temp1, 576, 7, 7,
        T("blocks.3.0.pw_proj.conv.weight"),
        T("blocks.3.0.pw_proj.bn.weight"), T("blocks.3.0.pw_proj.bn.bias"),
        T("blocks.3.0.pw_proj.bn.running_mean"), T("blocks.3.0.pw_proj.bn.running_var"),
        128, 1, 1, 1, ACT_NONE, next);
    current = next;
    next = (next == buf1) ? buf2 : buf1;

    /* blocks.3.1 */
    conv_bn_act(current, 128, 7, 7,
        T("blocks.3.1.dw_start.conv.weight"),
        T("blocks.3.1.dw_start.bn.weight"), T("blocks.3.1.dw_start.bn.bias"),
        T("blocks.3.1.dw_start.bn.running_mean"), T("blocks.3.1.dw_start.bn.running_var"),
        128, 5, 1, 128, ACT_NONE, temp1);
    conv_bn_act(temp1, 128, 7, 7,
        T("blocks.3.1.pw_exp.conv.weight"),
        T("blocks.3.1.pw_exp.bn.weight"), T("blocks.3.1.pw_exp.bn.bias"),
        T("blocks.3.1.pw_exp.bn.running_mean"), T("blocks.3.1.pw_exp.bn.running_var"),
        512, 1, 1, 1, ACT_RELU, temp2);
    conv_bn_act(temp2, 512, 7, 7,
        T("blocks.3.1.dw_mid.conv.weight"),
        T("blocks.3.1.dw_mid.bn.weight"), T("blocks.3.1.dw_mid.bn.bias"),
        T("blocks.3.1.dw_mid.bn.running_mean"), T("blocks.3.1.dw_mid.bn.running_var"),
        512, 5, 1, 512, ACT_RELU, temp1);
    conv_bn_act(temp1, 512, 7, 7,
        T("blocks.3.1.pw_proj.conv.weight"),
        T("blocks.3.1.pw_proj.bn.weight"), T("blocks.3.1.pw_proj.bn.bias"),
        T("blocks.3.1.pw_proj.bn.running_mean"), T("blocks.3.1.pw_proj.bn.running_var"),
        128, 1, 1, 1, ACT_NONE, next);
    add_arrays(next, current, 128 * 7 * 7, next);
    current = next;
    next = (next == buf1) ? buf2 : buf1;

    /* blocks.3.2 */
    conv_bn_act(current, 128, 7, 7,
        T("blocks.3.2.pw_exp.conv.weight"),
        T("blocks.3.2.pw_exp.bn.weight"), T("blocks.3.2.pw_exp.bn.bias"),
        T("blocks.3.2.pw_exp.bn.running_mean"), T("blocks.3.2.pw_exp.bn.running_var"),
        512, 1, 1, 1, ACT_RELU, temp1);
    conv_bn_act(temp1, 512, 7, 7,
        T("blocks.3.2.dw_mid.conv.weight"),
        T("blocks.3.2.dw_mid.bn.weight"), T("blocks.3.2.dw_mid.bn.bias"),
        T("blocks.3.2.dw_mid.bn.running_mean"), T("blocks.3.2.dw_mid.bn.running_var"),
        512, 5, 1, 512, ACT_RELU, temp2);
    conv_bn_act(temp2, 512, 7, 7,
        T("blocks.3.2.pw_proj.conv.weight"),
        T("blocks.3.2.pw_proj.bn.weight"), T("blocks.3.2.pw_proj.bn.bias"),
        T("blocks.3.2.pw_proj.bn.running_mean"), T("blocks.3.2.pw_proj.bn.running_var"),
        128, 1, 1, 1, ACT_NONE, next);
    add_arrays(next, current, 128 * 7 * 7, next);
    current = next;
    next = (next == buf1) ? buf2 : buf1;

    /* blocks.3.3 */
    conv_bn_act(current, 128, 7, 7,
        T("blocks.3.3.pw_exp.conv.weight"),
        T("blocks.3.3.pw_exp.bn.weight"), T("blocks.3.3.pw_exp.bn.bias"),
        T("blocks.3.3.pw_exp.bn.running_mean"), T("blocks.3.3.pw_exp.bn.running_var"),
        384, 1, 1, 1, ACT_RELU, temp1);
    conv_bn_act(temp1, 384, 7, 7,
        T("blocks.3.3.dw_mid.conv.weight"),
        T("blocks.3.3.dw_mid.bn.weight"), T("blocks.3.3.dw_mid.bn.bias"),
        T("blocks.3.3.dw_mid.bn.running_mean"), T("blocks.3.3.dw_mid.bn.running_var"),
        384, 5, 1, 384, ACT_RELU, temp2);
    conv_bn_act(temp2, 384, 7, 7,
        T("blocks.3.3.pw_proj.conv.weight"),
        T("blocks.3.3.pw_proj.bn.weight"), T("blocks.3.3.pw_proj.bn.bias"),
        T("blocks.3.3.pw_proj.bn.running_mean"), T("blocks.3.3.pw_proj.bn.running_var"),
        128, 1, 1, 1, ACT_NONE, next);
    add_arrays(next, current, 128 * 7 * 7, next);
    current = next;
    next = (next == buf1) ? buf2 : buf1;

    /* blocks.3.4 */
    conv_bn_act(current, 128, 7, 7,
        T("blocks.3.4.pw_exp.conv.weight"),
        T("blocks.3.4.pw_exp.bn.weight"), T("blocks.3.4.pw_exp.bn.bias"),
        T("blocks.3.4.pw_exp.bn.running_mean"), T("blocks.3.4.pw_exp.bn.running_var"),
        512, 1, 1, 1, ACT_RELU, temp1);
    conv_bn_act(temp1, 512, 7, 7,
        T("blocks.3.4.dw_mid.conv.weight"),
        T("blocks.3.4.dw_mid.bn.weight"), T("blocks.3.4.dw_mid.bn.bias"),
        T("blocks.3.4.dw_mid.bn.running_mean"), T("blocks.3.4.dw_mid.bn.running_var"),
        512, 3, 1, 512, ACT_RELU, temp2);
    conv_bn_act(temp2, 512, 7, 7,
        T("blocks.3.4.pw_proj.conv.weight"),
        T("blocks.3.4.pw_proj.bn.weight"), T("blocks.3.4.pw_proj.bn.bias"),
        T("blocks.3.4.pw_proj.bn.running_mean"), T("blocks.3.4.pw_proj.bn.running_var"),
        128, 1, 1, 1, ACT_NONE, next);
    add_arrays(next, current, 128 * 7 * 7, next);
    current = next;
    next = (next == buf1) ? buf2 : buf1;

    /* blocks.3.5 */
    conv_bn_act(current, 128, 7, 7,
        T("blocks.3.5.pw_exp.conv.weight"),
        T("blocks.3.5.pw_exp.bn.weight"), T("blocks.3.5.pw_exp.bn.bias"),
        T("blocks.3.5.pw_exp.bn.running_mean"), T("blocks.3.5.pw_exp.bn.running_var"),
        512, 1, 1, 1, ACT_RELU, temp1);
    conv_bn_act(temp1, 512, 7, 7,
        T("blocks.3.5.dw_mid.conv.weight"),
        T("blocks.3.5.dw_mid.bn.weight"), T("blocks.3.5.dw_mid.bn.bias"),
        T("blocks.3.5.dw_mid.bn.running_mean"), T("blocks.3.5.dw_mid.bn.running_var"),
        512, 3, 1, 512, ACT_RELU, temp2);
    conv_bn_act(temp2, 512, 7, 7,
        T("blocks.3.5.pw_proj.conv.weight"),
        T("blocks.3.5.pw_proj.bn.weight"), T("blocks.3.5.pw_proj.bn.bias"),
        T("blocks.3.5.pw_proj.bn.running_mean"), T("blocks.3.5.pw_proj.bn.running_var"),
        128, 1, 1, 1, ACT_NONE, next);
    add_arrays(next, current, 128 * 7 * 7, next);
    current = next;
    next = (next == buf1) ? buf2 : buf1;

    /* ======================================================================== */
    /* STAGE 4 (blocks.4): Final 1x1 conv                                       */
    /* ======================================================================== */
    printf("  - Stage 4: blocks.4.0 (1x1 conv)\n");
    conv_bn_act(current, 128, 7, 7,
        T("blocks.4.0.conv.weight"),
        T("blocks.4.0.bn1.weight"), T("blocks.4.0.bn1.bias"),
        T("blocks.4.0.bn1.running_mean"), T("blocks.4.0.bn1.running_var"),
        960, 1, 1, 1, ACT_RELU, next);
    current = next;
    next = (next == buf1) ? buf2 : buf1;

    /* ======================================================================== */
    /* HEAD: global_pool + conv_head + norm_head + classifier                    */
    /* ======================================================================== */
    printf("  - Head: global_pool + conv_head + classifier\n");

    /* Global average pooling: [960, 7, 7] -> [960] */
    global_avg_pool(current, 960, 49, pooled);

    /* conv_head: 1x1 conv (960->1280) + norm_head + ReLU */
    linear(pooled, T("conv_head.weight"), zero_bias_1280, 960, 1280, head_out);
    batch_norm(head_out, 1280, 1, 1,
              T("norm_head.weight"), T("norm_head.bias"),
              T("norm_head.running_mean"), T("norm_head.running_var"), head_out);
    relu_array(head_out, 1280, head_out);

    /* Classifier: linear (1280->1000) */
    linear(head_out, T("classifier.weight"), T("classifier.bias"), 1280, 1000, output);

    debug_tensor("logits [1000]", output, 1000);

    /* Cleanup */
    aligned_free(buf1);
    aligned_free(buf2);
    aligned_free(temp1);
    aligned_free(temp2);
    aligned_free(pooled);
    aligned_free(head_out);
    aligned_free(zero_bias_1280);

    printf("  - Forward pass complete!\n");

#undef T
}

#endif /* MOBILENETV4_H */
