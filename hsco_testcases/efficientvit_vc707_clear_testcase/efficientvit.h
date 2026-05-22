/*
 * EfficientViT-M0 FPGA Baremetal Implementation
 * Model Forward Pass
 *
 * FPGA Baremetal Notes:
 * - All dynamic allocations use aligned_malloc / aligned_free (not malloc/free)
 * - Large structs must NOT be on stack (stack is very small in baremetal)
 * - BSS is NOT zeroed on Ariane; use explicit init for static variables
 * - No snprintf available; use str_concat / str_build helpers
 */

#ifndef EFFICIENTVIT_H
#define EFFICIENTVIT_H

#include "tensor.h"
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

/* Concatenate 4 strings: s1 + s2 + s3 + s4 */
static void str_build4(char *dest, const char *s1, const char *s2, const char *s3, const char *s4) {
    while (*s1) *dest++ = *s1++;
    while (*s2) *dest++ = *s2++;
    while (*s3) *dest++ = *s3++;
    while (*s4) *dest++ = *s4++;
    *dest = '\0';
}

/* ============================================ */
/* Debug: print tensor summary for comparison   */
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
    printf("\n        first8:");
    int n = size < 8 ? size : 8;
    for (int i = 0; i < n; i++) {
        printf(" ");
        print_float_debug(data[i]);
    }
    printf("\n");
}

/* ============================================ */
/* Include attention after string helpers       */
/* ============================================ */
#include "attention.h"

/* ============================================ */
/* Layer Helper Functions                       */
/* ============================================ */

/* Conv2D + ReLU */
static void conv_relu(
    const float *input, int in_c, int h, int w,
    const float *weight, const float *bias,
    int out_c, int kh, int kw,
    int stride, int padding,
    int out_h, int out_w,
    float *output
) {
    conv2d(input, in_c, h, w, weight, bias,
           out_c, kh, kw, stride, padding, out_h, out_w, output);
    relu(output, out_c * out_h * out_w, output);
}

/* Depthwise Conv2D (no activation) */
static void dwconv(
    const float *input, int c, int h, int w,
    const float *weight, const float *bias,
    int kh, int kw,
    int stride, int padding,
    int out_h, int out_w,
    float *output
) {
    conv2d_depthwise(input, c, h, w, weight, bias,
                     kh, kw, stride, padding, out_h, out_w, output);
}

/* Pointwise Conv2D (1x1) */
static void pwconv(
    const float *input, int in_c, int h, int w,
    const float *weight, const float *bias,
    int out_c,
    float *output
) {
    conv2d(input, in_c, h, w, weight, bias,
           out_c, 1, 1, 1, 0, h, w, output);
}

/* ============================================ */
/* FFN Block                                    */
/* ============================================ */

static void ffn_block(
    float *x, int c, int h, int w,
    const float *pw1_w, const float *pw1_b, int hidden_c,
    const float *pw2_w, const float *pw2_b,
    float *temp, float *temp2
) {
    /* pw1 + ReLU */
    pwconv(x, c, h, w, pw1_w, pw1_b, hidden_c, temp);
    relu(temp, hidden_c * h * w, temp);
    
    /* pw2 (no activation, will be added as residual) */
    pwconv(temp, hidden_c, h, w, pw2_w, pw2_b, c, temp2);
    
    /* Residual add */
    add_elementwise(x, temp2, c * h * w, x);
}

/* ============================================ */
/* Residual DWConv Block                        */
/* ============================================ */

static void residual_dwconv(
    float *x, int c, int h, int w,
    const float *weight, const float *bias,
    int kh, int kw,
    float *temp
) {
    int pad = kh / 2;
    /* DWConv */
    dwconv(x, c, h, w, weight, bias, kh, kw, 1, pad, h, w, temp);
    
    /* Residual add */
    add_elementwise(x, temp, c * h * w, x);
}

/* ============================================ */
/* EfficientViT Block                           */
/* ============================================ */

static void efficientvit_block(
    float *x, int c, int h, int w,
    Model *model, const char *prefix
) {
    char name[256];
    int spatial_size = c * h * w;
    
    /* Allocate temp buffers */
    float *temp = alloc_tensor(c * 2, h, w);  /* For FFN hidden */
    float *temp2 = alloc_tensor(c, h, w);
    
    /* DW0 (residual) */
    str_concat(name, prefix, ".dw0.weight");
    Tensor *dw0_w = find_tensor(model, name);
    str_concat(name, prefix, ".dw0.bias");
    Tensor *dw0_b = find_tensor(model, name);
    residual_dwconv(x, c, h, w, dw0_w->data, dw0_b->data, 3, 3, temp2);
    
    /* FFN0 (residual) */
    str_concat(name, prefix, ".ffn0.pw1.weight");
    Tensor *ffn0_pw1_w = find_tensor(model, name);
    str_concat(name, prefix, ".ffn0.pw1.bias");
    Tensor *ffn0_pw1_b = find_tensor(model, name);
    str_concat(name, prefix, ".ffn0.pw2.weight");
    Tensor *ffn0_pw2_w = find_tensor(model, name);
    str_concat(name, prefix, ".ffn0.pw2.bias");
    Tensor *ffn0_pw2_b = find_tensor(model, name);
    ffn_block(x, c, h, w, ffn0_pw1_w->data, ffn0_pw1_b->data, c * 2,
              ffn0_pw2_w->data, ffn0_pw2_b->data, temp, temp2);
    
    /* Mixer (attention) - residual */
    /* Get window resolution from exported tensor */
    str_concat(name, prefix, ".mixer.window_resolution");
    Tensor *win_res_t = find_tensor(model, name);
    int window_resolution = win_res_t ? (int)win_res_t->data[0] : 7;
    
    str_concat(name, prefix, ".mixer");
    local_window_attention_full(x, c, h, w, model, name, window_resolution, temp2);
    add_elementwise(x, temp2, spatial_size, x);
    
    /* DW1 (residual) */
    str_concat(name, prefix, ".dw1.weight");
    Tensor *dw1_w = find_tensor(model, name);
    str_concat(name, prefix, ".dw1.bias");
    Tensor *dw1_b = find_tensor(model, name);
    residual_dwconv(x, c, h, w, dw1_w->data, dw1_b->data, 3, 3, temp2);
    
    /* FFN1 (residual) */
    str_concat(name, prefix, ".ffn1.pw1.weight");
    Tensor *ffn1_pw1_w = find_tensor(model, name);
    str_concat(name, prefix, ".ffn1.pw1.bias");
    Tensor *ffn1_pw1_b = find_tensor(model, name);
    str_concat(name, prefix, ".ffn1.pw2.weight");
    Tensor *ffn1_pw2_w = find_tensor(model, name);
    str_concat(name, prefix, ".ffn1.pw2.bias");
    Tensor *ffn1_pw2_b = find_tensor(model, name);
    ffn_block(x, c, h, w, ffn1_pw1_w->data, ffn1_pw1_b->data, c * 2,
              ffn1_pw2_w->data, ffn1_pw2_b->data, temp, temp2);
    
    aligned_free(temp);
    aligned_free(temp2);
}

/* ============================================ */
/* Patch Merging (Downsampling)                 */
/* ============================================ */

static void patch_merging(
    const float *input, int in_c, int h, int w,
    Model *model, const char *prefix,
    int out_c, float *output
) {
    char name[256];
    
    int hidden_c = in_c * 4;
    int out_h = (h + 1) / 2;
    int out_w = (w + 1) / 2;
    
    /* Conv1: pointwise expansion */
    str_concat(name, prefix, ".conv1.weight");
    Tensor *conv1_w = find_tensor(model, name);
    str_concat(name, prefix, ".conv1.bias");
    Tensor *conv1_b = find_tensor(model, name);
    
    float *x = alloc_tensor(hidden_c, h, w);
    pwconv(input, in_c, h, w, conv1_w->data, conv1_b->data, hidden_c, x);
    relu(x, hidden_c * h * w, x);
    
    /* Conv2: depthwise stride 2 */
    str_concat(name, prefix, ".conv2.weight");
    Tensor *conv2_w = find_tensor(model, name);
    str_concat(name, prefix, ".conv2.bias");
    Tensor *conv2_b = find_tensor(model, name);
    
    float *x2 = alloc_tensor(hidden_c, out_h, out_w);
    conv2d_depthwise(x, hidden_c, h, w, conv2_w->data, conv2_b->data,
                     3, 3, 2, 1, out_h, out_w, x2);
    relu(x2, hidden_c * out_h * out_w, x2);
    aligned_free(x);
    
    /* SE block */
    str_concat(name, prefix, ".se.fc1.weight");
    Tensor *se_fc1_w = find_tensor(model, name);
    str_concat(name, prefix, ".se.fc1.bias");
    Tensor *se_fc1_b = find_tensor(model, name);
    str_concat(name, prefix, ".se.fc2.weight");
    Tensor *se_fc2_w = find_tensor(model, name);
    str_concat(name, prefix, ".se.fc2.bias");
    Tensor *se_fc2_b = find_tensor(model, name);
    
    int squeeze_c = hidden_c / 4;
    float *se_squeezed = (float*)aligned_malloc(hidden_c * sizeof(float));
    float *se_excited = (float*)aligned_malloc(squeeze_c * sizeof(float));
    float *se_scale = (float*)aligned_malloc(hidden_c * sizeof(float));
    float *se_out = alloc_tensor(hidden_c, out_h, out_w);
    squeeze_excite(x2, hidden_c, out_h, out_w,
                   se_fc1_w->data, se_fc1_b->data, squeeze_c,
                   se_fc2_w->data, se_fc2_b->data,
                   se_squeezed, se_excited, se_scale, se_out);
    aligned_free(x2);
    aligned_free(se_squeezed);
    aligned_free(se_excited);
    aligned_free(se_scale);
    
    /* Conv3: pointwise projection */
    str_concat(name, prefix, ".conv3.weight");
    Tensor *conv3_w = find_tensor(model, name);
    str_concat(name, prefix, ".conv3.bias");
    Tensor *conv3_b = find_tensor(model, name);
    
    pwconv(se_out, hidden_c, out_h, out_w, conv3_w->data, conv3_b->data, out_c, output);
    aligned_free(se_out);
}

/* ============================================ */
/* Full Model Forward Pass                      */
/* ============================================ */

static void efficientvit_forward(
    const float *input,  /* [3, 224, 224] */
    Model *model,
    float *logits        /* [1000] */
) {
    int h = 224, w = 224;
    char name[256];
    
    printf("  Running patch embedding...\n");
    
    /* Patch embedding: 4 conv layers with stride 2 */
    /* [3, 224, 224] -> [8, 112, 112] */
    Tensor *pe0_w = find_tensor(model, "patch_embed.0.weight");
    Tensor *pe0_b = find_tensor(model, "patch_embed.0.bias");
    float *x = alloc_tensor(8, 112, 112);
    conv2d(input, 3, 224, 224, pe0_w->data, pe0_b->data, 8, 3, 3, 2, 1, 112, 112, x);
    relu(x, 8 * 112 * 112, x);
    debug_tensor("patch_embed.0 [8,112,112]", x, 8*112*112);
    
    /* [8, 112, 112] -> [16, 56, 56] */
    Tensor *pe2_w = find_tensor(model, "patch_embed.2.weight");
    Tensor *pe2_b = find_tensor(model, "patch_embed.2.bias");
    float *x2 = alloc_tensor(16, 56, 56);
    conv2d(x, 8, 112, 112, pe2_w->data, pe2_b->data, 16, 3, 3, 2, 1, 56, 56, x2);
    relu(x2, 16 * 56 * 56, x2);
    debug_tensor("patch_embed.2 [16,56,56]", x2, 16*56*56);
    aligned_free(x);
    
    /* [16, 56, 56] -> [32, 28, 28] */
    Tensor *pe4_w = find_tensor(model, "patch_embed.4.weight");
    Tensor *pe4_b = find_tensor(model, "patch_embed.4.bias");
    float *x3 = alloc_tensor(32, 28, 28);
    conv2d(x2, 16, 56, 56, pe4_w->data, pe4_b->data, 32, 3, 3, 2, 1, 28, 28, x3);
    relu(x3, 32 * 28 * 28, x3);
    debug_tensor("patch_embed.4 [32,28,28]", x3, 32*28*28);
    aligned_free(x2);
    
    /* [32, 28, 28] -> [64, 14, 14] */
    Tensor *pe6_w = find_tensor(model, "patch_embed.6.weight");
    Tensor *pe6_b = find_tensor(model, "patch_embed.6.bias");
    float *x4 = alloc_tensor(64, 14, 14);
    conv2d(x3, 32, 28, 28, pe6_w->data, pe6_b->data, 64, 3, 3, 2, 1, 14, 14, x4);
    debug_tensor("patch_embed.6 [64,14,14]", x4, 64*14*14);
    aligned_free(x3);
    
    printf("  Running blocks1...\n");
    /* Blocks1: 1 EfficientViT block at 14x14 */
    h = 14; w = 14;
    efficientvit_block(x4, 64, h, w, model, "blocks1.0");
    debug_tensor("blocks1.0 [64,14,14]", x4, 64*14*14);
    
    printf("  Running blocks2...\n");
    /* Blocks2: downsample + blocks at 7x7 */
    /* First: residual conv + ffn */
    float *temp = alloc_tensor(64 * 2, h, w);
    float *temp2 = alloc_tensor(64, h, w);
    
    Tensor *b2_0_0_w = find_tensor(model, "blocks2.0.0.weight");
    Tensor *b2_0_0_b = find_tensor(model, "blocks2.0.0.bias");
    residual_dwconv(x4, 64, h, w, b2_0_0_w->data, b2_0_0_b->data, 3, 3, temp2);
    
    Tensor *b2_0_1_pw1_w = find_tensor(model, "blocks2.0.1.pw1.weight");
    Tensor *b2_0_1_pw1_b = find_tensor(model, "blocks2.0.1.pw1.bias");
    Tensor *b2_0_1_pw2_w = find_tensor(model, "blocks2.0.1.pw2.weight");
    Tensor *b2_0_1_pw2_b = find_tensor(model, "blocks2.0.1.pw2.bias");
    ffn_block(x4, 64, h, w, b2_0_1_pw1_w->data, b2_0_1_pw1_b->data, 128,
              b2_0_1_pw2_w->data, b2_0_1_pw2_b->data, temp, temp2);
    
    aligned_free(temp);
    aligned_free(temp2);
    
    /* Patch merging: 14x14 -> 7x7, 64 -> 128 */
    float *x5 = alloc_tensor(128, 7, 7);
    patch_merging(x4, 64, h, w, model, "blocks2.1", 128, x5);
    aligned_free(x4);
    h = 7; w = 7;
    debug_tensor("blocks2.1 merge [128,7,7]", x5, 128*7*7);
    
    /* More residual blocks */
    temp = alloc_tensor(128 * 2, h, w);
    temp2 = alloc_tensor(128, h, w);
    
    Tensor *b2_2_0_w = find_tensor(model, "blocks2.2.0.weight");
    Tensor *b2_2_0_b = find_tensor(model, "blocks2.2.0.bias");
    residual_dwconv(x5, 128, h, w, b2_2_0_w->data, b2_2_0_b->data, 3, 3, temp2);
    
    Tensor *b2_2_1_pw1_w = find_tensor(model, "blocks2.2.1.pw1.weight");
    Tensor *b2_2_1_pw1_b = find_tensor(model, "blocks2.2.1.pw1.bias");
    Tensor *b2_2_1_pw2_w = find_tensor(model, "blocks2.2.1.pw2.weight");
    Tensor *b2_2_1_pw2_b = find_tensor(model, "blocks2.2.1.pw2.bias");
    ffn_block(x5, 128, h, w, b2_2_1_pw1_w->data, b2_2_1_pw1_b->data, 256,
              b2_2_1_pw2_w->data, b2_2_1_pw2_b->data, temp, temp2);
    
    aligned_free(temp);
    aligned_free(temp2);
    
    /* EfficientViT blocks */
    efficientvit_block(x5, 128, h, w, model, "blocks2.3");
    debug_tensor("blocks2.3 [128,7,7]", x5, 128*7*7);
    efficientvit_block(x5, 128, h, w, model, "blocks2.4");
    debug_tensor("blocks2.4 [128,7,7]", x5, 128*7*7);
    
    printf("  Running blocks3...\n");
    /* Blocks3: similar structure */
    temp = alloc_tensor(128 * 2, h, w);
    temp2 = alloc_tensor(128, h, w);
    
    Tensor *b3_0_0_w = find_tensor(model, "blocks3.0.0.weight");
    Tensor *b3_0_0_b = find_tensor(model, "blocks3.0.0.bias");
    if (b3_0_0_w) {
        residual_dwconv(x5, 128, h, w, b3_0_0_w->data, b3_0_0_b->data, 3, 3, temp2);
    }
    
    Tensor *b3_0_1_pw1_w = find_tensor(model, "blocks3.0.1.pw1.weight");
    Tensor *b3_0_1_pw1_b = find_tensor(model, "blocks3.0.1.pw1.bias");
    Tensor *b3_0_1_pw2_w = find_tensor(model, "blocks3.0.1.pw2.weight");
    Tensor *b3_0_1_pw2_b = find_tensor(model, "blocks3.0.1.pw2.bias");
    if (b3_0_1_pw1_w) {
        ffn_block(x5, 128, h, w, b3_0_1_pw1_w->data, b3_0_1_pw1_b->data, 256,
                  b3_0_1_pw2_w->data, b3_0_1_pw2_b->data, temp, temp2);
    }
    
    aligned_free(temp);
    aligned_free(temp2);
    
    /* Patch merging: 7x7 -> 4x4, 128 -> 192 */
    Tensor *pm_test = find_tensor(model, "blocks3.1.conv1.weight");
    float *x6;
    if (pm_test) {
        x6 = alloc_tensor(192, 4, 4);
        patch_merging(x5, 128, h, w, model, "blocks3.1", 192, x6);
        aligned_free(x5);
        h = 4; w = 4;
        debug_tensor("blocks3.1 merge [192,4,4]", x6, 192*4*4);
    } else {
        x6 = x5;  /* No patch merging in blocks3 */
    }
    
    /* More blocks if they exist */
    Tensor *b3_2_0_w = find_tensor(model, "blocks3.2.0.weight");
    if (b3_2_0_w) {
        temp = alloc_tensor(192 * 2, h, w);
        temp2 = alloc_tensor(192, h, w);
        
        Tensor *b3_2_0_b = find_tensor(model, "blocks3.2.0.bias");
        residual_dwconv(x6, 192, h, w, b3_2_0_w->data, b3_2_0_b->data, 3, 3, temp2);
        
        Tensor *b3_2_1_pw1_w = find_tensor(model, "blocks3.2.1.pw1.weight");
        Tensor *b3_2_1_pw1_b = find_tensor(model, "blocks3.2.1.pw1.bias");
        Tensor *b3_2_1_pw2_w = find_tensor(model, "blocks3.2.1.pw2.weight");
        Tensor *b3_2_1_pw2_b = find_tensor(model, "blocks3.2.1.pw2.bias");
        if (b3_2_1_pw1_w) {
            ffn_block(x6, 192, h, w, b3_2_1_pw1_w->data, b3_2_1_pw1_b->data, 384,
                      b3_2_1_pw2_w->data, b3_2_1_pw2_b->data, temp, temp2);
        }
        
        aligned_free(temp);
        aligned_free(temp2);
    }
    
    /* EfficientViT blocks in blocks3 */
    Tensor *b3_3_test = find_tensor(model, "blocks3.3.dw0.weight");
    if (b3_3_test) {
        efficientvit_block(x6, 192, h, w, model, "blocks3.3");
        debug_tensor("blocks3.3 [192,4,4]", x6, 192*4*4);
    }
    
    Tensor *b3_4_test = find_tensor(model, "blocks3.4.dw0.weight");
    if (b3_4_test) {
        efficientvit_block(x6, 192, h, w, model, "blocks3.4");
    }
    
    Tensor *b3_5_test = find_tensor(model, "blocks3.5.dw0.weight");
    if (b3_5_test) {
        efficientvit_block(x6, 192, h, w, model, "blocks3.5");
    }
    
    printf("  Running classification head...\n");
    /* Global average pooling */
    int final_c = pm_test ? 192 : 128;
    float *pooled = alloc_tensor_1d(final_c);
    global_avg_pool(x6, final_c, h, w, pooled);
    debug_tensor("pooled", pooled, final_c);
    aligned_free(x6);
    
    /* Classification head */
    Tensor *head_w = find_tensor(model, "head.weight");
    Tensor *head_b = find_tensor(model, "head.bias");
    linear(pooled, final_c, head_w->data, head_b->data, 1000, logits);
    debug_tensor("logits [1000]", logits, 1000);
    
    aligned_free(pooled);
}

#endif /* EFFICIENTVIT_H */
