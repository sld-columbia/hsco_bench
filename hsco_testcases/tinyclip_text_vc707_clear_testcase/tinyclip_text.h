/*
 * TinyCLIP Text-Only FPGA Baremetal Implementation
 * Text encoder only — text features computed from pre-computed token embeddings.
 *
 * Architecture: TinyCLIP-ViT-8M-16-Text-3M (text part only)
 * - Text: 77 context length, embed=256, 3 layers, 4 heads -> 512 output
 * - ~3M parameters, ~10 MB (float32, without token embedding table)
 *
 * FPGA Baremetal Notes:
 * - No snprintf available; uses str_build helpers
 * - All dynamic allocations use aligned_malloc / aligned_free
 * - Math functions use _approx variants on RISC-V
 * - Text encoder uses causal attention mask (upper triangular -inf)
 */

#ifndef TINYCLIP_TEXT_H
#define TINYCLIP_TEXT_H

#include "nn_ops.h"

/* Text encoder configuration */
#define TEXT_CONTEXT_LEN 77
#define TEXT_VOCAB_SIZE 49408
#define TEXT_EMBED_DIM 256
#define TEXT_LAYERS 3
#define TEXT_HEADS 4
#define TEXT_MLP_DIM 1024
#define TEXT_OUTPUT_DIM 512

#define LN_EPS 1e-5f

/* Large negative value for causal attention mask */
/* Cannot use -INFINITY on baremetal; use large negative float instead */
#define NEG_INF_MASK (-1e9f)

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
static void str_build4(char *dest, const char *s1, const char *s2,
                        const char *s3, const char *s4) {
    while (*s1) *dest++ = *s1++;
    while (*s2) *dest++ = *s2++;
    while (*s3) *dest++ = *s3++;
    while (*s4) *dest++ = *s4++;
    *dest = '\0';
}

/* ============================================ */
/* Debug Utilities (moved from nn_ops.h)        */
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
/* Multi-Head Self-Attention (with causal mask) */
/* ============================================ */

/*
 * multihead_attention workspace (caller-provided flat buffers):
 *   qkv          [seq_len * embed_dim * 3]     combined QKV projection
 *   Q            [seq_len * embed_dim]          split Q
 *   K            [seq_len * embed_dim]          split K
 *   V            [seq_len * embed_dim]          split V
 *   attn_output  [seq_len * embed_dim]          concatenated head outputs
 *   per_head (reused each iteration):
 *     q_head       [seq_len * head_dim]
 *     k_head       [seq_len * head_dim]
 *     v_head       [seq_len * head_dim]
 *     scores       [seq_len * seq_len]
 *     scores_sm    [seq_len * seq_len]           softmax output
 *     head_out     [seq_len * head_dim]
 */
static void multihead_attention(
    const float *x,
    const float *in_proj_weight, const float *in_proj_bias,
    const float *out_proj_weight, const float *out_proj_bias,
    int num_heads,
    const float *attn_mask,
    float *qkv,
    float *Q, float *K, float *V,
    float *attn_output,
    float *q_head, float *k_head, float *v_head,
    float *scores, float *scores_sm, float *head_out,
    float *output
) {
    int head_dim = TEXT_EMBED_DIM / num_heads;
    float scale = 1.0f / sqrtf_approx((float)head_dim);

    /* Compute combined QKV projection */
    linear_batched(x, TEXT_CONTEXT_LEN, TEXT_EMBED_DIM, in_proj_weight, in_proj_bias, TEXT_EMBED_DIM * 3, qkv);

    /* Split QKV into separate Q, K, V */
    for (int i = 0; i < TEXT_CONTEXT_LEN; i++) {
        for (int d = 0; d < TEXT_EMBED_DIM; d++) {
            Q[i * TEXT_EMBED_DIM + d] = qkv[i * TEXT_EMBED_DIM * 3 + d];
            K[i * TEXT_EMBED_DIM + d] = qkv[i * TEXT_EMBED_DIM * 3 + TEXT_EMBED_DIM + d];
            V[i * TEXT_EMBED_DIM + d] = qkv[i * TEXT_EMBED_DIM * 3 + TEXT_EMBED_DIM * 2 + d];
        }
    }

    /* Process each head */
    for (int h = 0; h < num_heads; h++) {
        /* Extract this head's Q, K, V slices */
        for (int i = 0; i < TEXT_CONTEXT_LEN; i++) {
            for (int d = 0; d < head_dim; d++) {
                q_head[i * head_dim + d] = Q[i * TEXT_EMBED_DIM + h * head_dim + d];
                k_head[i * head_dim + d] = K[i * TEXT_EMBED_DIM + h * head_dim + d];
                v_head[i * head_dim + d] = V[i * TEXT_EMBED_DIM + h * head_dim + d];
            }
        }

        /* Compute attention scores: Q @ K.T */
        matmul_transB(q_head, k_head, scores, TEXT_CONTEXT_LEN, head_dim, TEXT_CONTEXT_LEN);

        /* Scale scores */
        scale_array(scores, scale, TEXT_CONTEXT_LEN * TEXT_CONTEXT_LEN, scores_sm);

        /* Apply causal attention mask */
        for (int i = 0; i < TEXT_CONTEXT_LEN * TEXT_CONTEXT_LEN; i++) {
            scores_sm[i] += attn_mask[i];
        }

        /* Softmax */
        softmax_2d(scores_sm, TEXT_CONTEXT_LEN, TEXT_CONTEXT_LEN, scores);

        /* Apply attention to values: softmax(scores) @ V */
        matmul(scores, v_head, head_out, TEXT_CONTEXT_LEN, TEXT_CONTEXT_LEN, head_dim);

        /* Store head output into concatenated attn_output */
        for (int i = 0; i < TEXT_CONTEXT_LEN; i++) {
            for (int d = 0; d < head_dim; d++) {
                attn_output[i * TEXT_EMBED_DIM + h * head_dim + d] = head_out[i * head_dim + d];
            }
        }
    }

    /* Output projection */
    linear_batched(attn_output, TEXT_CONTEXT_LEN, TEXT_EMBED_DIM, out_proj_weight, out_proj_bias, TEXT_EMBED_DIM, output);
}

/* ============================================ */
/* MLP Block (with QuickGELU)                   */
/* ============================================ */

/*
 * mlp_block workspace (caller-provided):
 *   hidden     [seq_len * hidden_dim]   fc1 output
 *   gelu_out   [seq_len * hidden_dim]   QuickGELU output
 */
static void mlp_block(
    const float *x,
    const float *fc_weight, const float *fc_bias, int hidden_dim,
    const float *proj_weight, const float *proj_bias,
    float *hidden, float *gelu_out,
    float *output
) {
    /* First linear */
    linear_batched(x, TEXT_CONTEXT_LEN, TEXT_EMBED_DIM, fc_weight, fc_bias, hidden_dim, hidden);

    /* QuickGELU activation */
    quick_gelu_array(hidden, TEXT_CONTEXT_LEN * hidden_dim, gelu_out);

    /* Second linear (projection) */
    linear_batched(gelu_out, TEXT_CONTEXT_LEN, hidden_dim, proj_weight, proj_bias, TEXT_EMBED_DIM, output);
}

/* ============================================ */
/* Transformer Block (ResidualAttentionBlock)   */
/* ============================================ */

/*
 * transformer_block workspace (caller-provided):
 *   ln_out       [seq_len * embed_dim]         LayerNorm output
 *   attn_out     [seq_len * embed_dim]         attention output
 *   residual     [seq_len * embed_dim]         residual connection
 *   mlp_out      [seq_len * embed_dim]         MLP output
 *   + multihead_attention temps (qkv, Q, K, V, attn_output, per-head)
 *   + mlp_block temps (hidden, gelu_out)
 */
static void transformer_block(
    const float *input,
    const float *ln1_weight, const float *ln1_bias,
    const float *attn_in_proj_weight, const float *attn_in_proj_bias,
    const float *attn_out_proj_weight, const float *attn_out_proj_bias,
    int num_heads,
    const float *ln2_weight, const float *ln2_bias,
    const float *mlp_fc_weight, const float *mlp_fc_bias, int mlp_hidden_dim,
    const float *mlp_proj_weight, const float *mlp_proj_bias,
    const float *attn_mask,
    float *ln_out, float *attn_out, float *residual, float *mlp_out,
    float *mha_qkv, float *mha_Q, float *mha_K, float *mha_V, float *mha_attn_output,
    float *mha_q_head, float *mha_k_head, float *mha_v_head,
    float *mha_scores, float *mha_scores_sm, float *mha_head_out,
    float *mlp_hidden, float *mlp_gelu_out,
    float *output
) {
    int total_size = TEXT_CONTEXT_LEN * TEXT_EMBED_DIM;
    int head_dim = TEXT_EMBED_DIM / num_heads;

    /* LayerNorm 1 */
    layer_norm(input, total_size, TEXT_EMBED_DIM, ln1_weight, ln1_bias, ln_out);

    /* Self-attention with causal mask */
    multihead_attention(ln_out,
                       attn_in_proj_weight, attn_in_proj_bias,
                       attn_out_proj_weight, attn_out_proj_bias,
                       num_heads, attn_mask,
                       mha_qkv, mha_Q, mha_K, mha_V, mha_attn_output,
                       mha_q_head, mha_k_head, mha_v_head,
                       mha_scores, mha_scores_sm, mha_head_out,
                       attn_out);

    /* Residual: output = input + attn_out */
    add_arrays(input, attn_out, total_size, residual);

    /* LayerNorm 2 */
    layer_norm(residual, total_size, TEXT_EMBED_DIM, ln2_weight, ln2_bias, ln_out);

    /* MLP */
    mlp_block(ln_out,
              mlp_fc_weight, mlp_fc_bias, mlp_hidden_dim,
              mlp_proj_weight, mlp_proj_bias,
              mlp_hidden, mlp_gelu_out,
              mlp_out);

    /* Residual: x_out = x_mid + mlp(ln2(x_mid)); x_mid is residual */
    add_arrays(residual, mlp_out, total_size, output);
}

/* ============================================ */
/* Text Encoder (from pre-computed embeddings)  */
/* ============================================ */

static void text_encoder_from_embeds(
    const float *embeds,  /* [TEXT_CONTEXT_LEN, TEXT_EMBED_DIM] */
    int eot_pos,          /* pre-computed EOT token position */
    Model *model,
    float *output         /* [TEXT_OUTPUT_DIM] */
) {
    char name[256];
    char idx_buf[12];

    int embed_total = TEXT_CONTEXT_LEN * TEXT_EMBED_DIM;
    int head_dim = TEXT_EMBED_DIM / TEXT_HEADS;

    Tensor *pos_embed  = find_tensor(model, "_text_encoder.positional_embedding");
    Tensor *ln_final_w = find_tensor(model, "_text_encoder.ln_final.weight");
    Tensor *ln_final_b = find_tensor(model, "_text_encoder.ln_final.bias");
    Tensor *text_proj  = find_tensor(model, "_text_encoder.text_projection");

    if (!pos_embed || !ln_final_w || !ln_final_b || !text_proj) {
        printf("Error: Missing text encoder weights (no-embed version)\n");
        return;
    }

    /* ---- Allocate all workspace buffers once ---- */

    /* Ping-pong buffers for layer input/output */
    float *buf_a = alloc_tensor(embed_total);
    float *buf_b = alloc_tensor(embed_total);

    /* Causal attention mask */
    float *attn_mask = alloc_tensor(TEXT_CONTEXT_LEN * TEXT_CONTEXT_LEN);

    /* Transformer block temps */
    float *ln_out   = alloc_tensor(embed_total);
    float *attn_out = alloc_tensor(embed_total);
    float *residual = alloc_tensor(embed_total);
    float *mlp_out  = alloc_tensor(embed_total);

    /* MHA temps */
    float *mha_qkv         = alloc_tensor(TEXT_CONTEXT_LEN * TEXT_EMBED_DIM * 3);
    float *mha_Q           = alloc_tensor(embed_total);
    float *mha_K           = alloc_tensor(embed_total);
    float *mha_V           = alloc_tensor(embed_total);
    float *mha_attn_output = alloc_tensor(embed_total);
    float *mha_q_head      = alloc_tensor(TEXT_CONTEXT_LEN * head_dim);
    float *mha_k_head      = alloc_tensor(TEXT_CONTEXT_LEN * head_dim);
    float *mha_v_head      = alloc_tensor(TEXT_CONTEXT_LEN * head_dim);
    float *mha_scores      = alloc_tensor(TEXT_CONTEXT_LEN * TEXT_CONTEXT_LEN);
    float *mha_scores_sm   = alloc_tensor(TEXT_CONTEXT_LEN * TEXT_CONTEXT_LEN);
    float *mha_head_out    = alloc_tensor(TEXT_CONTEXT_LEN * head_dim);

    /* MLP temps */
    float *mlp_hidden   = alloc_tensor(TEXT_CONTEXT_LEN * TEXT_MLP_DIM);
    float *mlp_gelu_out = alloc_tensor(TEXT_CONTEXT_LEN * TEXT_MLP_DIM);

    /* Final LayerNorm output */
    float *x_normed = alloc_tensor(embed_total);

    /* ---- Add positional embedding: buf_a = embeds + pos_embed ---- */
    printf("    Adding positional embedding...\n");
    add_arrays(embeds, pos_embed->data, embed_total, buf_a);

    /* Build causal attention mask */
    printf("    Building causal attention mask (%dx%d)...\n", TEXT_CONTEXT_LEN, TEXT_CONTEXT_LEN);
    for (int i = 0; i < TEXT_CONTEXT_LEN; i++) {
        for (int j = 0; j < TEXT_CONTEXT_LEN; j++) {
            if (j > i) attn_mask[i * TEXT_CONTEXT_LEN + j] = NEG_INF_MASK;
            else       attn_mask[i * TEXT_CONTEXT_LEN + j] = 0.0f;
        }
    }

    /* ---- Encoder layers — ping-pong between buf_a and buf_b ---- */
    float *layer_in  = buf_a;
    float *layer_out = buf_b;

    for (int layer = 0; layer < TEXT_LAYERS; layer++) {
        printf("    Text layer %d/%d...\n", layer + 1, TEXT_LAYERS);
        int_to_str(idx_buf, layer);

        str_build4(name, "_text_encoder.transformer.resblocks.", idx_buf, ".ln_1.weight", "");
        Tensor *ln1_w = find_tensor(model, name);
        str_build4(name, "_text_encoder.transformer.resblocks.", idx_buf, ".ln_1.bias", "");
        Tensor *ln1_b = find_tensor(model, name);

        str_build4(name, "_text_encoder.transformer.resblocks.", idx_buf, ".attn.in_proj_weight", "");
        Tensor *attn_in_w = find_tensor(model, name);
        str_build4(name, "_text_encoder.transformer.resblocks.", idx_buf, ".attn.in_proj_bias", "");
        Tensor *attn_in_b = find_tensor(model, name);
        str_build4(name, "_text_encoder.transformer.resblocks.", idx_buf, ".attn.out_proj.weight", "");
        Tensor *attn_out_w = find_tensor(model, name);
        str_build4(name, "_text_encoder.transformer.resblocks.", idx_buf, ".attn.out_proj.bias", "");
        Tensor *attn_out_b = find_tensor(model, name);

        str_build4(name, "_text_encoder.transformer.resblocks.", idx_buf, ".ln_2.weight", "");
        Tensor *ln2_w = find_tensor(model, name);
        str_build4(name, "_text_encoder.transformer.resblocks.", idx_buf, ".ln_2.bias", "");
        Tensor *ln2_b = find_tensor(model, name);

        str_build4(name, "_text_encoder.transformer.resblocks.", idx_buf, ".mlp.c_fc.weight", "");
        Tensor *mlp_fc_w = find_tensor(model, name);
        str_build4(name, "_text_encoder.transformer.resblocks.", idx_buf, ".mlp.c_fc.bias", "");
        Tensor *mlp_fc_b = find_tensor(model, name);
        str_build4(name, "_text_encoder.transformer.resblocks.", idx_buf, ".mlp.c_proj.weight", "");
        Tensor *mlp_proj_w = find_tensor(model, name);
        str_build4(name, "_text_encoder.transformer.resblocks.", idx_buf, ".mlp.c_proj.bias", "");
        Tensor *mlp_proj_b = find_tensor(model, name);

        if (!ln1_w || !attn_in_w || !attn_out_w || !ln2_w || !mlp_fc_w || !mlp_proj_w) {
            printf("Error: Missing weights for text block %d\n", layer);
            continue;
        }

        transformer_block(layer_in,
                         ln1_w->data, ln1_b->data,
                         attn_in_w->data, attn_in_b->data,
                         attn_out_w->data, attn_out_b->data,
                         TEXT_HEADS,
                         ln2_w->data, ln2_b->data,
                         mlp_fc_w->data, mlp_fc_b->data, TEXT_MLP_DIM,
                         mlp_proj_w->data, mlp_proj_b->data,
                         attn_mask,
                         ln_out, attn_out, residual, mlp_out,
                         mha_qkv, mha_Q, mha_K, mha_V, mha_attn_output,
                         mha_q_head, mha_k_head, mha_v_head,
                         mha_scores, mha_scores_sm, mha_head_out,
                         mlp_hidden, mlp_gelu_out,
                         layer_out);

        /* Swap ping-pong pointers */
        float *tmp = layer_in; layer_in = layer_out; layer_out = tmp;
    }
    /* layer_in now points to the final layer output */

    printf("    Final LayerNorm + projection...\n");

    /* Final LayerNorm */
    layer_norm(layer_in, embed_total, TEXT_EMBED_DIM,
               ln_final_w->data, ln_final_b->data, x_normed);

    /* Extract EOT token features and project */
    const float *eot_features = x_normed + eot_pos * TEXT_EMBED_DIM;
    matvec_proj(eot_features, TEXT_EMBED_DIM, text_proj->data, TEXT_OUTPUT_DIM, output);

    /* ---- Free all workspace buffers ---- */
    aligned_free(x_normed);
    aligned_free(mlp_gelu_out);
    aligned_free(mlp_hidden);
    aligned_free(mha_head_out);
    aligned_free(mha_scores_sm);
    aligned_free(mha_scores);
    aligned_free(mha_v_head);
    aligned_free(mha_k_head);
    aligned_free(mha_q_head);
    aligned_free(mha_attn_output);
    aligned_free(mha_V);
    aligned_free(mha_K);
    aligned_free(mha_Q);
    aligned_free(mha_qkv);
    aligned_free(mlp_out);
    aligned_free(residual);
    aligned_free(attn_out);
    aligned_free(ln_out);
    aligned_free(attn_mask);
    aligned_free(buf_b);
    aligned_free(buf_a);

    printf("    Text encoder done.\n");
}

/* ============================================ */
/* Public API                                   */
/* ============================================ */

/* Encode text queries from pre-computed embeddings */
static void tinyclip_encode_text_from_embeds(
    const float *embeds,          /* [num_queries, context_length, TEXT_EMBED_DIM] */
    const int32_t *eot_positions, /* [num_queries] */
    int num_queries,
    int context_length,
    Model *model,
    float *features               /* [num_queries, TEXT_OUTPUT_DIM] */
) {
    float *l2_temp = alloc_tensor(TEXT_OUTPUT_DIM);

    for (int i = 0; i < num_queries; i++) {
        printf("  Query %d/%d (EOT at pos %d)...\n", i + 1, num_queries, (int)eot_positions[i]);
        const float *emb_i = embeds + (size_t)i * context_length * TEXT_EMBED_DIM;
        text_encoder_from_embeds(emb_i, eot_positions[i], model,
                                features + i * TEXT_OUTPUT_DIM);
    }

    /* L2 normalize each query's feature vector */
    for (int i = 0; i < num_queries; i++) {
        l2_normalize(features + i * TEXT_OUTPUT_DIM, TEXT_OUTPUT_DIM, l2_temp);
        for (int j = 0; j < TEXT_OUTPUT_DIM; j++) {
            features[i * TEXT_OUTPUT_DIM + j] = l2_temp[j];
        }
    }

    aligned_free(l2_temp);
}

#endif /* TINYCLIP_TEXT_H */
