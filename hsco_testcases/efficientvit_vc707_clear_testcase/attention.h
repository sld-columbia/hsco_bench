/*
 * EfficientViT-M0 FPGA Baremetal Implementation
 * Full CascadedGroupAttention and LocalWindowAttention
 *
 * FPGA Baremetal Notes:
 * - All memory allocation via aligned_malloc / aligned_free
 * - Uses sqrtf_approx
 * - String building via str_concat / str_build helpers (no snprintf)
 */

#ifndef ATTENTION_H
#define ATTENTION_H

#include "tensor.h"
#include "nn_ops.h"

/* ============================================ */
/* Window Partition / Reverse Operations        */
/* ============================================ */

/* Partition input into windows */
/* input: [C, H, W] -> output: [num_windows, C, wh, ww] */
static void window_partition(
    const float *input, int C, int H, int W,
    int window_h, int window_w,
    float *output
) {
    int nH = H / window_h;
    int nW = W / window_w;
    
    /* Rearrange: BCHW -> (nH*nW, C, wh, ww) */
    for (int nh = 0; nh < nH; nh++) {
        for (int nw = 0; nw < nW; nw++) {
            int win_idx = nh * nW + nw;
            for (int c = 0; c < C; c++) {
                for (int h = 0; h < window_h; h++) {
                    for (int w = 0; w < window_w; w++) {
                        int src_h = nh * window_h + h;
                        int src_w = nw * window_w + w;
                        int src_idx = c * H * W + src_h * W + src_w;
                        int dst_idx = win_idx * C * window_h * window_w +
                                     c * window_h * window_w +
                                     h * window_w + w;
                        output[dst_idx] = input[src_idx];
                    }
                }
            }
        }
    }
}

/* Reverse window partition */
/* input: [num_windows, C, wh, ww] -> output: [C, H, W] */
static void window_reverse(
    const float *input, int num_windows, int C, int window_h, int window_w,
    int H, int W,
    float *output
) {
    int nH = H / window_h;
    int nW = W / window_w;
    (void)num_windows;
    
    for (int nh = 0; nh < nH; nh++) {
        for (int nw = 0; nw < nW; nw++) {
            int win_idx = nh * nW + nw;
            for (int c = 0; c < C; c++) {
                for (int h = 0; h < window_h; h++) {
                    for (int w = 0; w < window_w; w++) {
                        int dst_h = nh * window_h + h;
                        int dst_w = nw * window_w + w;
                        int src_idx = win_idx * C * window_h * window_w +
                                     c * window_h * window_w +
                                     h * window_w + w;
                        int dst_idx = c * H * W + dst_h * W + dst_w;
                        output[dst_idx] = input[src_idx];
                    }
                }
            }
        }
    }
}

/* ============================================ */
/* Cascaded Group Attention (Full Implementation) */
/* ============================================ */

static void cascaded_group_attention_full(
    const float *x,      /* [C, H, W] */
    int C, int H, int W,
    Model *model, const char *prefix,
    float *output        /* [C, H, W] */
) {
    char name[256];
    int num_heads = 4;
    int head_dim = C / num_heads;
    int N = H * W;
    int key_dim = 16;
    
    /* Get first QKV weight (also used for early-exit check) */
    str_concat(name, prefix, ".qkvs.0.weight");
    Tensor *qkv0_w = find_tensor(model, name);
    if (!qkv0_w) {
        printf("Warning: Cannot find %s\n", name);
        memset(output, 0, C * H * W * sizeof(float));
        return;
    }
    int d = head_dim;
    float scale = 1.0f / sqrtf_approx((float)key_dim);
    
    /* Get attention biases */
    str_concat(name, prefix, ".attention_biases");
    Tensor *attn_biases = find_tensor(model, name);
    
    str_concat(name, prefix, ".attention_bias_idxs");
    Tensor *attn_bias_idxs = find_tensor(model, name);
    
    /* Split input into num_heads chunks */
    /* Use a single flat array for feats_in pointers */
    float *feats_in_buf[4];
    float *feats_out_buf[4];
    int head;
    for (head = 0; head < num_heads; head++) {
        feats_in_buf[head] = (float*)aligned_malloc(head_dim * N * sizeof(float));
        for (int c = 0; c < head_dim; c++) {
            memcpy(feats_in_buf[head] + c * N, 
                   x + (head * head_dim + c) * N, 
                   N * sizeof(float));
        }
    }
    
    /* Output features for each head */
    for (head = 0; head < num_heads; head++) {
        feats_out_buf[head] = (float*)aligned_malloc(d * N * sizeof(float));
    }
    
    /* Current feat for cascade */
    float *feat = (float*)aligned_malloc(head_dim * N * sizeof(float));
    memcpy(feat, feats_in_buf[0], head_dim * N * sizeof(float));
    
    /* Process each head with cascade */
    for (head = 0; head < num_heads; head++) {
        /* Cascade: add previous output to input (for head > 0) */
        if (head > 0) {
            for (int i = 0; i < head_dim * N; i++) {
                feat[i] += feats_in_buf[head][i];
            }
        }
        
        /* QKV projection */
        char idx_buf[12];
        int_to_str(idx_buf, head);
        str_build4(name, prefix, ".qkvs.", idx_buf, ".weight");
        Tensor *qkv_w = find_tensor(model, name);
        str_build4(name, prefix, ".qkvs.", idx_buf, ".bias");
        Tensor *qkv_b = find_tensor(model, name);
        
        int qkv_out_c = key_dim * 2 + d;
        float *qkv = (float*)aligned_malloc(qkv_out_c * N * sizeof(float));
        
        /* Reshape feat to [head_dim, H, W] for conv */
        float *feat_3d = (float*)aligned_malloc(head_dim * H * W * sizeof(float));
        memcpy(feat_3d, feat, head_dim * H * W * sizeof(float));
        
        float *qkv_3d = (float*)aligned_malloc(qkv_out_c * H * W * sizeof(float));
        conv2d(feat_3d, head_dim, H, W, qkv_w->data, qkv_b->data,
               qkv_out_c, 1, 1, 1, 0, H, W, qkv_3d);
        
        /* Split Q, K, V */
        float *q = (float*)aligned_malloc(key_dim * N * sizeof(float));
        float *k = (float*)aligned_malloc(key_dim * N * sizeof(float));
        float *v = (float*)aligned_malloc(d * N * sizeof(float));
        
        for (int c = 0; c < key_dim; c++) {
            memcpy(q + c * N, qkv_3d + c * N, N * sizeof(float));
            memcpy(k + c * N, qkv_3d + (key_dim + c) * N, N * sizeof(float));
        }
        for (int c = 0; c < d; c++) {
            memcpy(v + c * N, qkv_3d + (key_dim * 2 + c) * N, N * sizeof(float));
        }
        
        /* DW conv on Q */
        str_build4(name, prefix, ".dws.", idx_buf, ".weight");
        Tensor *dw_w = find_tensor(model, name);
        str_build4(name, prefix, ".dws.", idx_buf, ".bias");
        Tensor *dw_b = find_tensor(model, name);
        
        int kernel_size = 5;
        int pad = kernel_size / 2;
        
        float *q_conv = (float*)aligned_malloc(key_dim * N * sizeof(float));
        conv2d_depthwise(q, key_dim, H, W, dw_w->data, dw_b->data,
                         kernel_size, kernel_size, 1, pad, H, W, q_conv);
        
        /* Attention: (Q^T @ K) * scale + bias */
        /* Q: [key_dim, N], K: [key_dim, N] */
        /* Q^T: [N, key_dim], Q^T @ K: [N, N] */
        float *attn = (float*)aligned_malloc(N * N * sizeof(float));
        
        /* Compute Q^T @ K */
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int kk = 0; kk < key_dim; kk++) {
                    sum += q_conv[kk * N + i] * k[kk * N + j];
                }
                attn[i * N + j] = sum * scale;
            }
        }
        
        /* Add attention bias */
        if (attn_biases && attn_bias_idxs) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    int bias_idx = (int)attn_bias_idxs->data[i * N + j];
                    attn[i * N + j] += attn_biases->data[head * N + bias_idx];
                }
            }
        }
        
        /* Softmax along last dimension */
        for (int i = 0; i < N; i++) {
            softmax(attn + i * N, N, attn + i * N);
        }
        
        /* Output: V @ attn^T */
        /* V: [d, N], attn: [N, N], attn^T: [N, N] */
        /* V @ attn^T: [d, N] */
        float *out_head = feats_out_buf[head];
        for (int c = 0; c < d; c++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int i = 0; i < N; i++) {
                    sum += v[c * N + i] * attn[j * N + i];  /* attn^T[i,j] = attn[j,i] */
                }
                out_head[c * N + j] = sum;
            }
        }
        
        /* Update feat for next iteration */
        memcpy(feat, out_head, d * N * sizeof(float));
        
        /* Cleanup for this head */
        aligned_free(qkv);
        aligned_free(feat_3d);
        aligned_free(qkv_3d);
        aligned_free(q);
        aligned_free(k);
        aligned_free(v);
        aligned_free(q_conv);
        aligned_free(attn);
    }
    
    /* Concatenate all heads and project */
    int out_c = num_heads * d;
    float *concat = (float*)aligned_malloc(out_c * N * sizeof(float));
    for (head = 0; head < num_heads; head++) {
        memcpy(concat + head * d * N, feats_out_buf[head], d * N * sizeof(float));
    }
    
    /* ReLU */
    relu(concat, out_c * N, concat);
    
    /* Projection conv */
    str_concat(name, prefix, ".proj.weight");
    Tensor *proj_w = find_tensor(model, name);
    str_concat(name, prefix, ".proj.bias");
    Tensor *proj_b = find_tensor(model, name);
    
    float *concat_3d = (float*)aligned_malloc(out_c * H * W * sizeof(float));
    memcpy(concat_3d, concat, out_c * H * W * sizeof(float));
    
    conv2d(concat_3d, out_c, H, W, proj_w->data, proj_b->data,
           C, 1, 1, 1, 0, H, W, output);
    
    /* Cleanup */
    for (head = 0; head < num_heads; head++) {
        aligned_free(feats_in_buf[head]);
        aligned_free(feats_out_buf[head]);
    }
    aligned_free(feat);
    aligned_free(concat);
    aligned_free(concat_3d);
}

/* ============================================ */
/* Local Window Attention (Full Implementation) */
/* ============================================ */

static void local_window_attention_full(
    float *x,           /* [C, H, W] - modified in place */
    int C, int H, int W,
    Model *model, const char *prefix,
    int window_resolution,
    float *output       /* [C, H, W] */
) {
    char name[256];
    
    /* If resolution <= window_resolution, apply attention directly */
    if (H <= window_resolution && W <= window_resolution) {
        str_concat(name, prefix, ".attn");
        cascaded_group_attention_full(x, C, H, W, model, name, output);
        return;
    }
    
    /* Otherwise, partition into windows */
    int pad_b = (window_resolution - H % window_resolution) % window_resolution;
    int pad_r = (window_resolution - W % window_resolution) % window_resolution;
    
    int pH = H + pad_b;
    int pW = W + pad_r;
    int nH = pH / window_resolution;
    int nW = pW / window_resolution;
    int num_windows = nH * nW;
    
    /* Pad input if necessary */
    float *padded = NULL;
    if (pad_b > 0 || pad_r > 0) {
        padded = alloc_tensor(C, pH, pW);
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                memcpy(padded + c * pH * pW + h * pW,
                       x + c * H * W + h * W,
                       W * sizeof(float));
            }
        }
    } else {
        padded = x;
    }
    
    /* Partition into windows */
    float *windows = (float*)aligned_malloc(num_windows * C * window_resolution * window_resolution * sizeof(float));
    window_partition(padded, C, pH, pW, window_resolution, window_resolution, windows);
    
    /* Apply attention to each window */
    float *windows_out = (float*)aligned_malloc(num_windows * C * window_resolution * window_resolution * sizeof(float));
    
    str_concat(name, prefix, ".attn");
    for (int win = 0; win < num_windows; win++) {
        float *win_in = windows + win * C * window_resolution * window_resolution;
        float *win_out = windows_out + win * C * window_resolution * window_resolution;
        cascaded_group_attention_full(win_in, C, window_resolution, window_resolution,
                                      model, name, win_out);
    }
    
    /* Reverse window partition */
    float *result = (float*)aligned_malloc(C * pH * pW * sizeof(float));
    window_reverse(windows_out, num_windows, C, window_resolution, window_resolution, pH, pW, result);
    
    /* Remove padding */
    if (pad_b > 0 || pad_r > 0) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                memcpy(output + c * H * W + h * W,
                       result + c * pH * pW + h * pW,
                       W * sizeof(float));
            }
        }
        aligned_free(padded);
    } else {
        memcpy(output, result, C * H * W * sizeof(float));
    }
    
    aligned_free(windows);
    aligned_free(windows_out);
    aligned_free(result);
}

#endif /* ATTENTION_H */
