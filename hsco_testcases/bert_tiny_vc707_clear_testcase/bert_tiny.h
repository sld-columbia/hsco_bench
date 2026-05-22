/*
 * BERT-tiny FPGA Baremetal Implementation
 * BERT-tiny Model Forward Pass
 *
 * Model specs:
 * - Hidden size: 128
 * - Num layers: 2
 * - Num attention heads: 2
 * - Intermediate size: 512
 * - Vocab size: 30522
 *
 * FPGA Baremetal Notes:
 * - No snprintf available; uses str_build helpers
 * - All dynamic allocations use aligned_malloc / aligned_free
 * - Progress prints added to detect hangs
 */

#ifndef BERT_TINY_H
#define BERT_TINY_H

#include "nn_ops.h"

#define MAX_SEQ_LEN 128
#define HIDDEN_SIZE 128
#define NUM_LAYERS 2
#define NUM_HEADS 2
#define INTERMEDIATE_SIZE 512

/* Pre-allocation workspace sizes (in floats) */
#define FFN_WORKSPACE_FLOATS(S, INTER) ((S) * (INTER))
#define BERT_LAYER_WORKSPACE_FLOATS(S, H, NH, INTER) \
    (3*(S)*(H) + FFN_WORKSPACE_FLOATS(S, INTER) + MHA_WORKSPACE_FLOATS(S, H, NH) + (S))

/* ============================================ */
/* String building helpers for baremetal        */
/* (no snprintf available on Ariane baremetal)  */
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

/* ============================================ */
/* BERT Encoder Layer                           */
/* ============================================ */

static void bert_layer(
    const float *input, int seq_len,
    Model *model, int layer_idx,
    const int *attention_mask,
    float *output,
    float *workspace
) {
    char idx_buf[4];
    char name[256];
    int_to_str(idx_buf, layer_idx);

    printf("    Layer %d: loading weights...\n", layer_idx);

    /* Build tensor names using str_build3: "encoder.layer." + idx + ".suffix" */
    str_build3(name, "encoder.layer.", idx_buf, ".attention.self.query.weight");
    Tensor *q_w = find_tensor(model, name);
    str_build3(name, "encoder.layer.", idx_buf, ".attention.self.query.bias");
    Tensor *q_b = find_tensor(model, name);

    str_build3(name, "encoder.layer.", idx_buf, ".attention.self.key.weight");
    Tensor *k_w = find_tensor(model, name);
    str_build3(name, "encoder.layer.", idx_buf, ".attention.self.key.bias");
    Tensor *k_b = find_tensor(model, name);

    str_build3(name, "encoder.layer.", idx_buf, ".attention.self.value.weight");
    Tensor *v_w = find_tensor(model, name);
    str_build3(name, "encoder.layer.", idx_buf, ".attention.self.value.bias");
    Tensor *v_b = find_tensor(model, name);

    str_build3(name, "encoder.layer.", idx_buf, ".attention.output.dense.weight");
    Tensor *o_w = find_tensor(model, name);
    str_build3(name, "encoder.layer.", idx_buf, ".attention.output.dense.bias");
    Tensor *o_b = find_tensor(model, name);

    str_build3(name, "encoder.layer.", idx_buf, ".attention.output.LayerNorm.weight");
    Tensor *attn_ln_w = find_tensor(model, name);
    str_build3(name, "encoder.layer.", idx_buf, ".attention.output.LayerNorm.bias");
    Tensor *attn_ln_b = find_tensor(model, name);

    str_build3(name, "encoder.layer.", idx_buf, ".intermediate.dense.weight");
    Tensor *fc1_w = find_tensor(model, name);
    str_build3(name, "encoder.layer.", idx_buf, ".intermediate.dense.bias");
    Tensor *fc1_b = find_tensor(model, name);

    str_build3(name, "encoder.layer.", idx_buf, ".output.dense.weight");
    Tensor *fc2_w = find_tensor(model, name);
    str_build3(name, "encoder.layer.", idx_buf, ".output.dense.bias");
    Tensor *fc2_b = find_tensor(model, name);

    str_build3(name, "encoder.layer.", idx_buf, ".output.LayerNorm.weight");
    Tensor *ffn_ln_w = find_tensor(model, name);
    str_build3(name, "encoder.layer.", idx_buf, ".output.LayerNorm.bias");
    Tensor *ffn_ln_b = find_tensor(model, name);

    if (!q_w || !k_w || !v_w || !o_w || !attn_ln_w || !fc1_w || !fc2_w || !ffn_ln_w ||
        !q_b || !k_b || !v_b || !o_b || !attn_ln_b || !fc1_b || !fc2_b || !ffn_ln_b) {
        printf("Error: Missing tensors for layer %d\n", layer_idx);
        return;
    }

    int SH = seq_len * HIDDEN_SIZE;

    int HD = HIDDEN_SIZE / NUM_HEADS;

    /* Partition workspace */
    float *attn_out    = workspace;
    float *attn_norm   = attn_out + SH;
    float *ffn_out     = attn_norm + SH;
    float *ffn_inter   = ffn_out + SH;
    float *mha_Q       = ffn_inter + seq_len * INTERMEDIATE_SIZE;
    float *mha_K       = mha_Q + SH;
    float *mha_V       = mha_K + SH;
    float *mha_attn    = mha_V + SH;
    float *mha_qh      = mha_attn + SH;
    float *mha_kh      = mha_qh + seq_len * HD;
    float *mha_vh      = mha_kh + seq_len * HD;
    float *mha_kt      = mha_vh + seq_len * HD;
    float *mha_scores  = mha_kt + seq_len * HD;
    float *mha_hout    = mha_scores + seq_len * seq_len;
    float *attn_mask_f = mha_hout + seq_len * HD;

    /* Convert int attention_mask to float for pipeline compatibility */
    for (int i = 0; i < seq_len; i++) attn_mask_f[i] = (float)attention_mask[i];

    printf("    Layer %d: self-attention...\n", layer_idx);
    multi_head_attention(
        input, seq_len, HIDDEN_SIZE, NUM_HEADS,
        q_w->data, q_b->data,
        k_w->data, k_b->data,
        v_w->data, v_b->data,
        o_w->data, o_b->data,
        attn_mask_f,
        attn_out,
        mha_Q, mha_K, mha_V, mha_attn,
        mha_qh, mha_kh, mha_vh,
        mha_kt, mha_scores, mha_hout
    );

    printf("    Layer %d: attention add+norm...\n", layer_idx);
    add_arrays(attn_out, input, SH, attn_out);
    layer_norm(attn_out, SH, HIDDEN_SIZE,
               attn_ln_w->data, attn_ln_b->data, attn_norm);

    printf("    Layer %d: feed-forward...\n", layer_idx);
    feed_forward(
        attn_norm, seq_len, HIDDEN_SIZE, INTERMEDIATE_SIZE,
        fc1_w->data, fc1_b->data,
        fc2_w->data, fc2_b->data,
        ffn_out, ffn_inter
    );

    printf("    Layer %d: FFN add+norm...\n", layer_idx);
    add_arrays(ffn_out, attn_norm, SH, ffn_out);
    layer_norm(ffn_out, SH, HIDDEN_SIZE,
               ffn_ln_w->data, ffn_ln_b->data, output);

    debug_tensor("layer_output", output, SH);
    printf("    Layer %d: done\n", layer_idx);
}

/* ============================================ */
/* BERT-tiny Full Forward Pass                  */
/* ============================================ */

static void bert_tiny_forward(
    const int *input_ids,
    const int *attention_mask,
    int seq_len,
    Model *model,
    float *output
) {
    printf("  Loading embedding tensors...\n");
    Tensor *word_emb = find_tensor(model, "embeddings.word_embeddings.weight");
    Tensor *pos_emb = find_tensor(model, "embeddings.position_embeddings.weight");
    Tensor *token_type_emb = find_tensor(model, "embeddings.token_type_embeddings.weight");
    Tensor *emb_ln_w = find_tensor(model, "embeddings.LayerNorm.weight");
    Tensor *emb_ln_b = find_tensor(model, "embeddings.LayerNorm.bias");

    if (!word_emb || !pos_emb || !token_type_emb || !emb_ln_w || !emb_ln_b) {
        printf("Error: Missing embedding tensors\n");
        return;
    }

    int SH = seq_len * HIDDEN_SIZE;

    /* Pre-allocate all buffers */
    float *embeds      = alloc_tensor(SH);
    float *embeds_norm = alloc_tensor(SH);
    float *hidden_a    = alloc_tensor(SH);
    float *hidden_b    = alloc_tensor(SH);
    float *layer_ws    = alloc_tensor(
        BERT_LAYER_WORKSPACE_FLOATS(seq_len, HIDDEN_SIZE, NUM_HEADS, INTERMEDIATE_SIZE));

    printf("  Computing embeddings (seq_len=%d)...\n", seq_len);
    embedding_lookup(input_ids, seq_len, word_emb->data, word_emb->shape[0], HIDDEN_SIZE, embeds);
    add_position_embeddings(embeds, seq_len, HIDDEN_SIZE, pos_emb->data, pos_emb->shape[0], embeds);

    /* Add token type embeddings (all type 0) */
    for (int i = 0; i < SH; i += HIDDEN_SIZE) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            embeds[i + j] += token_type_emb->data[j];
        }
    }

    printf("  Embedding LayerNorm...\n");
    layer_norm(embeds, SH, HIDDEN_SIZE,
               emb_ln_w->data, emb_ln_b->data, embeds_norm);
    debug_tensor("embeddings", embeds_norm, SH);

    /* Ping-pong buffers for encoder layers */
    float *layer_in  = hidden_a;
    float *layer_out = hidden_b;
    for (int i = 0; i < SH; i++) layer_in[i] = embeds_norm[i];

    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        printf("  Encoder layer %d/%d...\n", layer + 1, NUM_LAYERS);
        bert_layer(layer_in, seq_len, model, layer, attention_mask, layer_out, layer_ws);
        float *tmp = layer_in; layer_in = layer_out; layer_out = tmp;
    }

    printf("  CLS pooling...\n");
    cls_pooling(layer_in, seq_len, HIDDEN_SIZE, output);
    debug_tensor("cls_output", output, HIDDEN_SIZE);

    aligned_free(embeds);
    aligned_free(embeds_norm);
    aligned_free(hidden_a);
    aligned_free(hidden_b);
    aligned_free(layer_ws);

    printf("  Forward pass complete!\n");
}

#endif /* BERT_TINY_H */
