/*
 * TinyCLIP Vision — Single Transformer Layer (FPGA Accelerator Test)
 *
 * Runs exactly ONE transformer block from the TinyCLIP-ViT-8M-16 vision encoder.
 * All 10 layers share identical dimensions, so testing 1 layer exercises every
 * leaf kernel at its actual size.
 *
 * Architecture per layer:
 *   seq_len=197, embed_dim=256, heads=4, head_dim=64, mlp_hidden=1024
 *
 * Leaf kernels exercised:
 *   layer_norm, linear_batched, matmul_transB, scale_array, softmax_2d,
 *   matmul, add_arrays, quick_gelu_array
 *
 * FPGA Baremetal Notes:
 * - All dynamic allocations use aligned_malloc / aligned_free
 * - Math functions use _approx variants on RISC-V
 */

#ifndef TINYCLIP_VISION_H
#define TINYCLIP_VISION_H

#include "nn_ops.h"

/* Single-layer configuration (same dims for all 10 layers) */
#define VISION_SEQ_LEN 197
#define VISION_EMBED_DIM 256
#define VISION_HEADS 4
#define VISION_HEAD_DIM (VISION_EMBED_DIM / VISION_HEADS)  /* 64 */
#define VISION_MLP_DIM 1024

#define LN_EPS 1e-5f

/* ============================================ */
/* Debug Utilities (printf OK — not leaf kernels)*/
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

/* ============================================ */
/* Multi-Head Self-Attention                    */
/* All temp buffers passed in (externalized)    */
/* ============================================ */

/*
 * Externalized temp buffer sizes:
 *   qkv:         [seq_len * embed_dim * 3]
 *   q/k/v:       [seq_len * embed_dim]
 *   attn_output:  [seq_len * embed_dim]
 *   q/k/v_head:  [seq_len * head_dim]
 *   scores:      [seq_len * seq_len]
 *   scores_sm:   [seq_len * seq_len]
 *   head_out:    [seq_len * head_dim]
 */
static void multihead_attention(
    const float *x, int seq_len, int embed_dim,
    const float *in_proj_weight, const float *in_proj_bias,
    const float *out_proj_weight, const float *out_proj_bias,
    int num_heads,
    float *output,
    float *qkv,
    float *q, float *k, float *v,
    float *attn_output,
    float *q_head, float *k_head, float *v_head,
    float *scores, float *scores_sm,
    float *head_out
) {
    int head_dim = embed_dim / num_heads;
    float scale = 1.0f / sqrtf_approx((float)head_dim);

    /* QKV projection */
    linear_batched(x, seq_len, embed_dim, in_proj_weight, in_proj_bias, embed_dim * 3, qkv);

    /* Split into Q, K, V (loop copies, no memcpy) */
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < embed_dim; j++) {
            q[i * embed_dim + j] = qkv[i * embed_dim * 3 + j];
            k[i * embed_dim + j] = qkv[i * embed_dim * 3 + embed_dim + j];
            v[i * embed_dim + j] = qkv[i * embed_dim * 3 + embed_dim * 2 + j];
        }
    }

    /* Process each head */
    for (int h = 0; h < num_heads; h++) {
        /* Extract this head's Q, K, V */
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                q_head[i * head_dim + j] = q[i * embed_dim + h * head_dim + j];
                k_head[i * head_dim + j] = k[i * embed_dim + h * head_dim + j];
                v_head[i * head_dim + j] = v[i * embed_dim + h * head_dim + j];
            }
        }

        /* scores = Q @ K.T */
        matmul_transB(q_head, k_head, scores, seq_len, head_dim, seq_len);

        /* Scale scores into scores_sm */
        for (int s = 0; s < seq_len * seq_len; s++) {
            scores_sm[s] = scores[s] * scale;
        }

        /* Softmax */
        softmax_2d(scores_sm, seq_len, seq_len, scores);

        /* Apply attention to values: softmax(scores) @ V */
        matmul(scores, v_head, head_out, seq_len, seq_len, head_dim);

        /* Store head output into concatenated attn_output */
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                attn_output[i * embed_dim + h * head_dim + j] = head_out[i * head_dim + j];
            }
        }
    }

    /* Output projection */
    linear_batched(attn_output, seq_len, embed_dim, out_proj_weight, out_proj_bias, embed_dim, output);
}

/* ============================================ */
/* MLP Block (with QuickGELU)                   */
/* All temp buffers passed in (externalized)    */
/* ============================================ */

/* Temp buffers: hidden[seq_len*hidden_dim], gelu_out[seq_len*hidden_dim] */
static void mlp_block(
    const float *x, int seq_len, int embed_dim,
    const float *fc_weight, const float *fc_bias, int hidden_dim,
    const float *proj_weight, const float *proj_bias,
    float *output,
    float *hidden,
    float *gelu_out
) {
    /* First linear */
    linear_batched(x, seq_len, embed_dim, fc_weight, fc_bias, hidden_dim, hidden);

    /* QuickGELU activation */
    quick_gelu_array(hidden, seq_len * hidden_dim, gelu_out);

    /* Second linear (projection) */
    linear_batched(gelu_out, seq_len, hidden_dim, proj_weight, proj_bias, embed_dim, output);
}

/* ============================================ */
/* Transformer Block (ResidualAttentionBlock)   */
/* const input, all temps externalized          */
/* ============================================ */

/*
 * Temp buffers: ln_out/attn_out/residual/mlp_out [seq_len*embed_dim]
 * MHA temps: qkv, q, k, v, attn_out, q/k/v_head, scores, scores_sm, head_out
 * MLP temps: hidden, gelu_out
 */
static void transformer_block(
    const float *input, int seq_len, int embed_dim,
    const float *ln1_weight, const float *ln1_bias,
    const float *attn_in_proj_weight, const float *attn_in_proj_bias,
    const float *attn_out_proj_weight, const float *attn_out_proj_bias,
    int num_heads,
    const float *ln2_weight, const float *ln2_bias,
    const float *mlp_fc_weight, const float *mlp_fc_bias, int mlp_hidden_dim,
    const float *mlp_proj_weight, const float *mlp_proj_bias,
    float *output,
    float *ln_out, float *attn_out, float *residual, float *mlp_out,
    float *mha_qkv, float *mha_q, float *mha_k, float *mha_v,
    float *mha_attn_out, float *mha_q_head, float *mha_k_head, float *mha_v_head,
    float *mha_scores, float *mha_scores_sm, float *mha_head_out,
    float *mlp_hidden, float *mlp_gelu_out
) {
    int total_size = seq_len * embed_dim;

    /* LayerNorm 1 */
    layer_norm(input, total_size, embed_dim, ln1_weight, ln1_bias, ln_out);

    /* Self-attention */
    multihead_attention(ln_out, seq_len, embed_dim,
                       attn_in_proj_weight, attn_in_proj_bias,
                       attn_out_proj_weight, attn_out_proj_bias,
                       num_heads, attn_out,
                       mha_qkv, mha_q, mha_k, mha_v,
                       mha_attn_out, mha_q_head, mha_k_head, mha_v_head,
                       mha_scores, mha_scores_sm, mha_head_out);

    /* Residual connection 1: input + attn_out -> residual */
    add_arrays(input, attn_out, total_size, residual);

    /* LayerNorm 2 */
    layer_norm(residual, total_size, embed_dim, ln2_weight, ln2_bias, ln_out);

    /* MLP */
    mlp_block(ln_out, seq_len, embed_dim,
              mlp_fc_weight, mlp_fc_bias, mlp_hidden_dim,
              mlp_proj_weight, mlp_proj_bias, mlp_out,
              mlp_hidden, mlp_gelu_out);

    /* Residual connection 2: residual + mlp_out -> output */
    add_arrays(residual, mlp_out, total_size, output);
}

/* ============================================ */
/* Run Single Transformer Layer                 */
/* Allocates all workspace, runs 1 block        */
/* ============================================ */

/* layer_input/layer_output: [VISION_SEQ_LEN * VISION_EMBED_DIM] */
static void run_single_layer(
    const float *layer_input,
    const float *ln1_w, const float *ln1_b,
    const float *attn_in_w, const float *attn_in_b,
    const float *attn_out_w, const float *attn_out_b,
    const float *ln2_w, const float *ln2_b,
    const float *mlp_fc_w, const float *mlp_fc_b,
    const float *mlp_proj_w, const float *mlp_proj_b,
    float *layer_output
) {
    int S = VISION_SEQ_LEN;
    int D = VISION_EMBED_DIM;
    int H = VISION_HEAD_DIM;
    int M = VISION_MLP_DIM;

    /* Allocate all workspace buffers */
    /* Transformer block temps */
    float *ln_out   = (float *)aligned_malloc(S * D * sizeof(float));
    float *attn_out = (float *)aligned_malloc(S * D * sizeof(float));
    float *residual = (float *)aligned_malloc(S * D * sizeof(float));
    float *mlp_out  = (float *)aligned_malloc(S * D * sizeof(float));

    /* MHA temps */
    float *mha_qkv      = (float *)aligned_malloc(S * D * 3 * sizeof(float));
    float *mha_q        = (float *)aligned_malloc(S * D * sizeof(float));
    float *mha_k        = (float *)aligned_malloc(S * D * sizeof(float));
    float *mha_v        = (float *)aligned_malloc(S * D * sizeof(float));
    float *mha_attn_out = (float *)aligned_malloc(S * D * sizeof(float));
    float *mha_q_head   = (float *)aligned_malloc(S * H * sizeof(float));
    float *mha_k_head   = (float *)aligned_malloc(S * H * sizeof(float));
    float *mha_v_head   = (float *)aligned_malloc(S * H * sizeof(float));
    float *mha_scores   = (float *)aligned_malloc(S * S * sizeof(float));
    float *mha_scores_sm= (float *)aligned_malloc(S * S * sizeof(float));
    float *mha_head_out = (float *)aligned_malloc(S * H * sizeof(float));

    /* MLP temps */
    float *mlp_hidden   = (float *)aligned_malloc(S * M * sizeof(float));
    float *mlp_gelu_out = (float *)aligned_malloc(S * M * sizeof(float));

    /* Run the transformer block */
    transformer_block(
        layer_input, S, D,
        ln1_w, ln1_b,
        attn_in_w, attn_in_b,
        attn_out_w, attn_out_b,
        VISION_HEADS,
        ln2_w, ln2_b,
        mlp_fc_w, mlp_fc_b, M,
        mlp_proj_w, mlp_proj_b,
        layer_output,
        ln_out, attn_out, residual, mlp_out,
        mha_qkv, mha_q, mha_k, mha_v,
        mha_attn_out, mha_q_head, mha_k_head, mha_v_head,
        mha_scores, mha_scores_sm, mha_head_out,
        mlp_hidden, mlp_gelu_out
    );

    /* Free all workspace */
    aligned_free(mlp_gelu_out);
    aligned_free(mlp_hidden);
    aligned_free(mha_head_out);
    aligned_free(mha_scores_sm);
    aligned_free(mha_scores);
    aligned_free(mha_v_head);
    aligned_free(mha_k_head);
    aligned_free(mha_q_head);
    aligned_free(mha_attn_out);
    aligned_free(mha_v);
    aligned_free(mha_k);
    aligned_free(mha_q);
    aligned_free(mha_qkv);
    aligned_free(mlp_out);
    aligned_free(residual);
    aligned_free(attn_out);
    aligned_free(ln_out);
}

#endif /* TINYCLIP_VISION_H */
