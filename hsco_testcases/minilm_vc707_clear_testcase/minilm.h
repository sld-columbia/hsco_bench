/*
 * paraphrase-MiniLM-L3-v2 FPGA Baremetal Implementation
 * MiniLM Model Forward Pass (LITE MODE)
 *
 * Lite mode: starts from pre-computed embeddings (skip embedding layer)
 * Only needs encoder weights (~21 MB instead of ~66 MB)
 *
 * Model specs:
 * - Hidden size: 384
 * - Num layers: 3
 * - Num attention heads: 12
 * - Intermediate size: 1536
 * - Head dimension: 32
 * - Pooling: Mean pooling (sentence-transformers style)
 *
 * FPGA Baremetal Notes:
 * - No snprintf available; uses str_build helpers
 * - All dynamic allocations use aligned_malloc / aligned_free
 */

#ifndef MINILM_H
#define MINILM_H

#include "nn_ops.h"

/* ============================================ */
/* Debug Utilities (non-kernel, orchestration)  */
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

#define MAX_SEQ_LEN 128
#define HIDDEN_SIZE 384
#define NUM_LAYERS 3
#define NUM_HEADS 12
#define INTERMEDIATE_SIZE 1536

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

/* ============================================ */
/* BERT Encoder Layer                           */
/* ============================================ */

static void bert_layer(
    const float *input,
    Model *model, int layer_idx,
    const float *attention_mask,
    float *attn_out, float *attn_norm, float *attn_residual,
    float *ffn_out, float *ffn_residual,
    float *mha_Q, float *mha_K, float *mha_V, float *mha_attn_output,
    float *mha_q_head, float *mha_k_head, float *mha_v_head,
    float *mha_k_t, float *mha_scores, float *mha_scores_out, float *mha_head_output,
    float *ffn_inter_buf, float *ffn_gelu_out,
    float *output
) {
    char idx_buf[4];
    char name[256];
    int_to_str(idx_buf, layer_idx);

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

    if (!q_w || !k_w || !v_w || !o_w || !attn_ln_w || !fc1_w || !fc2_w || !ffn_ln_w) {
        printf("Error: Missing tensors for layer %d\n", layer_idx);
        return;
    }

    /* Self-attention */
    multi_head_attention(
        input, MAX_SEQ_LEN, HIDDEN_SIZE, NUM_HEADS, HIDDEN_SIZE / NUM_HEADS,
        q_w->data, q_b->data,
        k_w->data, k_b->data,
        v_w->data, v_b->data,
        o_w->data, o_b->data,
        attention_mask,
        mha_Q, mha_K, mha_V, mha_attn_output,
        mha_q_head, mha_k_head, mha_v_head,
        mha_k_t, mha_scores, mha_scores_out, mha_head_output,
        attn_out
    );

    /* Attention residual + LayerNorm */
    add_arrays(attn_out, input, MAX_SEQ_LEN * HIDDEN_SIZE, attn_residual);
    layer_norm(attn_residual, MAX_SEQ_LEN, HIDDEN_SIZE,
               attn_ln_w->data, attn_ln_b->data, attn_norm);

    /* Feed-forward */
    feed_forward(
        attn_norm, MAX_SEQ_LEN, HIDDEN_SIZE, INTERMEDIATE_SIZE,
        fc1_w->data, fc1_b->data,
        fc2_w->data, fc2_b->data,
        ffn_inter_buf, ffn_gelu_out,
        ffn_out
    );

    /* FFN residual + LayerNorm */
    add_arrays(ffn_out, attn_norm, MAX_SEQ_LEN * HIDDEN_SIZE, ffn_residual);
    layer_norm(ffn_residual, MAX_SEQ_LEN, HIDDEN_SIZE,
               ffn_ln_w->data, ffn_ln_b->data, output);
}

/* ============================================ */
/* MiniLM Lite Forward Pass                     */
/* Starts from pre-computed embeddings           */
/* (skips embedding layer to save ~45 MB)       */
/* ============================================ */

static void minilm_forward_from_embeddings(
    const float *embeddings,
    const float *attention_mask,
    Model *model,
    float *output
) {
    int hidden_total = MAX_SEQ_LEN * HIDDEN_SIZE;
    int head_dim = HIDDEN_SIZE / NUM_HEADS;

    printf("  Allocating workspace buffers...\n");

    /* Double-buffer for layer input/output (avoid memcpy) */
    float *buf_a = alloc_tensor(hidden_total);
    float *buf_b = alloc_tensor(hidden_total);

    /* bert_layer temp buffers — allocated once, reused across layers */
    float *attn_out       = alloc_tensor(hidden_total);
    float *attn_norm      = alloc_tensor(hidden_total);
    float *attn_residual  = alloc_tensor(hidden_total);
    float *ffn_out        = alloc_tensor(hidden_total);
    float *ffn_residual   = alloc_tensor(hidden_total);

    /* multi_head_attention temp buffers */
    float *mha_Q           = alloc_tensor(hidden_total);
    float *mha_K           = alloc_tensor(hidden_total);
    float *mha_V           = alloc_tensor(hidden_total);
    float *mha_attn_output = alloc_tensor(hidden_total);
    float *mha_q_head      = alloc_tensor(MAX_SEQ_LEN * head_dim);
    float *mha_k_head      = alloc_tensor(MAX_SEQ_LEN * head_dim);
    float *mha_v_head      = alloc_tensor(MAX_SEQ_LEN * head_dim);
    float *mha_k_t         = alloc_tensor(head_dim * MAX_SEQ_LEN);
    float *mha_scores      = alloc_tensor(MAX_SEQ_LEN * MAX_SEQ_LEN);
    float *mha_scores_out  = alloc_tensor(MAX_SEQ_LEN * MAX_SEQ_LEN);
    float *mha_head_output = alloc_tensor(MAX_SEQ_LEN * head_dim);

    /* feed_forward temp buffers */
    float *ffn_inter_buf    = alloc_tensor(MAX_SEQ_LEN * INTERMEDIATE_SIZE);
    float *ffn_gelu_out     = alloc_tensor(MAX_SEQ_LEN * INTERMEDIATE_SIZE);

    /* Copy embeddings into buf_a as initial hidden states */
    for (int i = 0; i < hidden_total; i++) {
        buf_a[i] = embeddings[i];
    }

    debug_tensor("input_embeddings", buf_a, hidden_total);

    /* Encoder layers — ping-pong between buf_a and buf_b */
    float *layer_in = buf_a;
    float *layer_out = buf_b;
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        printf("  Encoder layer %d/%d...\n", layer + 1, NUM_LAYERS);
        bert_layer(
            layer_in, model, layer, attention_mask,
            attn_out, attn_norm, attn_residual,
            ffn_out, ffn_residual,
            mha_Q, mha_K, mha_V, mha_attn_output,
            mha_q_head, mha_k_head, mha_v_head,
            mha_k_t, mha_scores, mha_scores_out, mha_head_output,
            ffn_inter_buf, ffn_gelu_out,
            layer_out
        );
        debug_tensor("layer_output", layer_out, hidden_total);
        /* Swap: output of this layer becomes input of next */
        float *tmp = layer_in;
        layer_in = layer_out;
        layer_out = tmp;
    }
    /* After loop, layer_in points to final encoder output */

    printf("  Mean pooling...\n");
    mean_pooling(layer_in, MAX_SEQ_LEN, HIDDEN_SIZE, attention_mask, output);

    debug_tensor("sentence_embedding", output, HIDDEN_SIZE);

    /* Free all workspace buffers */
    aligned_free(ffn_gelu_out);
    aligned_free(ffn_inter_buf);
    aligned_free(mha_head_output);
    aligned_free(mha_scores_out);
    aligned_free(mha_scores);
    aligned_free(mha_k_t);
    aligned_free(mha_v_head);
    aligned_free(mha_k_head);
    aligned_free(mha_q_head);
    aligned_free(mha_attn_output);
    aligned_free(mha_V);
    aligned_free(mha_K);
    aligned_free(mha_Q);
    aligned_free(ffn_residual);
    aligned_free(ffn_out);
    aligned_free(attn_residual);
    aligned_free(attn_norm);
    aligned_free(attn_out);
    aligned_free(buf_b);
    aligned_free(buf_a);

    printf("  Forward pass complete!\n");
}

#endif /* MINILM_H */
