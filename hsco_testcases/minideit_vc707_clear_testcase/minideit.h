/*
 * Mini-DeiT-Ti FPGA Baremetal Implementation
 * Model Forward Pass
 *
 * FPGA Baremetal Notes:
 * - All dynamic allocations use aligned_malloc / aligned_free (not malloc/free)
 * - Large structs must NOT be on stack (stack is very small in baremetal)
 * - BSS is NOT zeroed on Ariane; use explicit init for static variables
 * - No snprintf available; use str_concat / str_build helpers
 */

#ifndef MINIDEIT_H
#define MINIDEIT_H

#include "tensor.h"
#include "nn_ops.h"

/* Mini-DeiT-Ti architecture constants */
#define EMBED_DIM 192
#define NUM_HEADS 3
#define NUM_BLOCKS 6
#define MLP_RATIO 4
#define SEQ_LEN 196       /* 14 * 14 patches */
#define REPEATED_TIMES 2
#define MLP_HIDDEN_DIM (EMBED_DIM * MLP_RATIO)  /* 768 */
#define NUM_BUCKETS 49

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

/* Concatenate 5 strings: s1 + s2 + s3 + s4 + s5 */
static void str_build5(char *dest, const char *s1, const char *s2, const char *s3, const char *s4, const char *s5) {
    while (*s1) *dest++ = *s1++;
    while (*s2) *dest++ = *s2++;
    while (*s3) *dest++ = *s3++;
    while (*s4) *dest++ = *s4++;
    while (*s5) *dest++ = *s5++;
    *dest = '\0';
}

/* ============================================ */
/* Workspace Size Macros                        */
/* ============================================ */

/* mini_attention workspace regions (floats) */
#define ATTN_WS_QKV   (SEQ_LEN * 3 * EMBED_DIM)                   /* 112896 */
#define ATTN_WS_Q     (SEQ_LEN * EMBED_DIM)                        /* 37632  */
#define ATTN_WS_K     (SEQ_LEN * EMBED_DIM)                        /* 37632  */
#define ATTN_WS_V     (SEQ_LEN * EMBED_DIM)                        /* 37632  */
#define ATTN_WS_ATTN  (NUM_HEADS * SEQ_LEN * SEQ_LEN)              /* 115248 */
#define ATTN_WS_AOUT  (SEQ_LEN * EMBED_DIM)                        /* 37632  */
#define ATTN_WS_TMP   (NUM_HEADS * SEQ_LEN * SEQ_LEN)              /* 115248 reused */
#define ATTN_WS_BKT   (SEQ_LEN * SEQ_LEN)                          /* 38416 ints */
#define ATTN_WS_TOTAL (ATTN_WS_QKV + ATTN_WS_Q + ATTN_WS_K + ATTN_WS_V + \
                       ATTN_WS_ATTN + ATTN_WS_AOUT + ATTN_WS_TMP + ATTN_WS_BKT)
/* ATTN_WS_TOTAL = 532336 floats ~ 2.0 MB */

/* mlp_block workspace */
#define MLP_WS_HIDDEN (SEQ_LEN * MLP_HIDDEN_DIM)                   /* 150528 */

/* mini_attention and mlp_block are called sequentially: share the same pool */
#define ATTN_MLP_WS   ATTN_WS_TOTAL   /* ATTN_WS_TOTAL > MLP_WS_HIDDEN */

/* transformer_block workspace regions */
#define BLOCK_WS_NORMED (SEQ_LEN * EMBED_DIM)                      /* 37632  */
#define BLOCK_WS_AOUT   (SEQ_LEN * EMBED_DIM)                      /* 37632  */
#define BLOCK_WS_MOUT   (SEQ_LEN * EMBED_DIM)                      /* 37632  */
#define BLOCK_WS_TOTAL  (BLOCK_WS_NORMED + BLOCK_WS_AOUT + BLOCK_WS_MOUT + ATTN_MLP_WS)
/* BLOCK_WS_TOTAL = 645232 floats ~ 2.46 MB */

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
    printf("\n        first5:");
    int n = size < 5 ? size : 5;
    for (int i = 0; i < n; i++) {
        printf(" ");
        print_float_debug(data[i]);
    }
    printf("\n");
}

/* ============================================ */
/* Transformer Block (RepeatedMiniBlock style)  */
/* ============================================ */

/*
 * transformer_block workspace layout (BLOCK_WS_TOTAL floats):
 *   normed    [0 .. seq_len*dim)                     LayerNorm output
 *   attn_out  [seq_len*dim .. 2*seq_len*dim)          attention output
 *   mlp_out   [2*seq_len*dim .. 3*seq_len*dim)        MLP output
 *   sub_ws    [3*seq_len*dim .. BLOCK_WS_TOTAL)       shared for mini_attention / mlp_block
 */
/* x[SEQ_LEN, EMBED_DIM] -> output[SEQ_LEN, EMBED_DIM] (must not alias x),
   workspace: flat buffer >= BLOCK_WS_TOTAL floats */
static void transformer_block(
    const float *x,
    Model *model, const char *block_prefix, int instance_idx,
    float *output,
    float *workspace
) {
    char name[512];
    char idx_buf[12];

    /* Partition workspace */
    float *normed   = workspace;
    float *attn_out = normed   + SEQ_LEN * EMBED_DIM;
    float *mlp_out  = attn_out + SEQ_LEN * EMBED_DIM;
    float *sub_ws   = mlp_out  + SEQ_LEN * EMBED_DIM;  /* shared: mini_attention then mlp_block */

    printf("    [block] %s inst=%d: start\n", block_prefix, instance_idx);

    int_to_str(idx_buf, instance_idx);

    /* LayerNorm 1 */
    printf("    [block] LayerNorm1...\n");
    str_build5(name, block_prefix, ".block.norm1.instances.", idx_buf, ".weight", "");
    Tensor *norm1_w = find_tensor(model, name);
    str_build5(name, block_prefix, ".block.norm1.instances.", idx_buf, ".bias", "");
    Tensor *norm1_b = find_tensor(model, name);

    if (!norm1_w || !norm1_b) {
        printf("Error: Cannot find norm1 tensors for %s instance %d\n", block_prefix, instance_idx);
        return;
    }

    layer_norm(x, 1, SEQ_LEN, EMBED_DIM, norm1_w->data, norm1_b->data, normed);

    /* Multi-Head Attention */
    printf("    [block] Attention...\n");
    str_concat(name, block_prefix, ".block.attn.qkv.weight");
    Tensor *qkv_w = find_tensor(model, name);
    str_concat(name, block_prefix, ".block.attn.qkv.bias");
    Tensor *qkv_b = find_tensor(model, name);
    str_concat(name, block_prefix, ".block.attn.proj.weight");
    Tensor *proj_w = find_tensor(model, name);
    str_concat(name, block_prefix, ".block.attn.proj.bias");
    Tensor *proj_b = find_tensor(model, name);

    str_build5(name, block_prefix, ".block.attn.rpe_k.instances.", idx_buf, ".lookup_table_weight", "");
    Tensor *rpe_k = find_tensor(model, name);
    str_build5(name, block_prefix, ".block.attn.conv_l.instances.", idx_buf, ".weight", "");
    Tensor *conv_l = find_tensor(model, name);
    str_build5(name, block_prefix, ".block.attn.conv_w.instances.", idx_buf, ".weight", "");
    Tensor *conv_wt = find_tensor(model, name);

    mini_attention(normed, SEQ_LEN, EMBED_DIM, NUM_HEADS,
                   qkv_w->data, qkv_b->data,
                   proj_w->data, proj_b->data,
                   rpe_k  ? rpe_k->data  : NULL,
                   conv_l ? conv_l->data : NULL,
                   conv_wt ? conv_wt->data : NULL,
                   attn_out, sub_ws);

    /* Residual 1: output = x + attn_out */
    printf("    [block] attention done\n");
    add_arrays(x, attn_out, SEQ_LEN * EMBED_DIM, output);

    /* LayerNorm 2 (reads output written above) */
    printf("    [block] LayerNorm2...\n");
    str_build5(name, block_prefix, ".block.norm2.instances.", idx_buf, ".weight", "");
    Tensor *norm2_w = find_tensor(model, name);
    str_build5(name, block_prefix, ".block.norm2.instances.", idx_buf, ".bias", "");
    Tensor *norm2_b = find_tensor(model, name);

    if (!norm2_w || !norm2_b) {
        printf("Error: Cannot find norm2 tensors for %s instance %d\n", block_prefix, instance_idx);
        return;
    }

    layer_norm(output, 1, SEQ_LEN, EMBED_DIM, norm2_w->data, norm2_b->data, normed);

    /* MLP */
    printf("    [block] MLP...\n");
    str_concat(name, block_prefix, ".block.mlp.fc1.weight");
    Tensor *fc1_w = find_tensor(model, name);
    str_concat(name, block_prefix, ".block.mlp.fc1.bias");
    Tensor *fc1_b = find_tensor(model, name);
    str_concat(name, block_prefix, ".block.mlp.fc2.weight");
    Tensor *fc2_w = find_tensor(model, name);
    str_concat(name, block_prefix, ".block.mlp.fc2.bias");
    Tensor *fc2_b = find_tensor(model, name);

    mlp_block(normed, SEQ_LEN, EMBED_DIM, MLP_HIDDEN_DIM,
              fc1_w->data, fc1_b->data,
              fc2_w->data, fc2_b->data,
              sub_ws, mlp_out);

    /* Residual 2: output += mlp_out (self-aliased add: safe) */
    add_arrays(output, mlp_out, SEQ_LEN * EMBED_DIM, output);

    printf("    [block] %s inst=%d: COMPLETE\n", block_prefix, instance_idx);
}

/* ============================================ */
/* Full Model Forward Pass                      */
/* ============================================ */

/* input[3, 224, 224] -> logits[1000] */
static void minideit_forward(
    const float *input,
    Model *model,
    float *logits
) {
    char name[256];
    char idx_buf[12];

    printf("  Config: embed_dim=%d, num_heads=%d, num_blocks=%d\n", EMBED_DIM, NUM_HEADS, NUM_BLOCKS);

    /* Pre-allocate all workspace up front */
    float *buf1     = alloc_tensor(SEQ_LEN * EMBED_DIM);  /* ping-pong A */
    float *buf2     = alloc_tensor(SEQ_LEN * EMBED_DIM);  /* ping-pong B */
    float *x_normed = alloc_tensor(SEQ_LEN * EMBED_DIM);  /* final LayerNorm output */
    float *pooled   = alloc_tensor(EMBED_DIM);            /* mean-pooled features */
    float *block_ws = alloc_tensor(BLOCK_WS_TOTAL);       /* transformer block workspace */

    /* 1. Patch Embedding: [3, 224, 224] -> buf1 [196, 192] */
    printf("  Running patch embedding...\n");
    Tensor *pe_weight = find_tensor(model, "patch_embed.proj.weight");
    Tensor *pe_bias   = find_tensor(model, "patch_embed.proj.bias");
    patch_embed_conv(input, pe_weight->data, pe_bias->data,
                     3, EMBED_DIM, 224, 16, 14, buf1);
    debug_tensor("patch_embed [196,192]", buf1, SEQ_LEN * EMBED_DIM);

    /* 2. Add positional embedding (in buf1) */
    Tensor *pos_embed = find_tensor(model, "pos_embed");
    for (int i = 0; i < SEQ_LEN * EMBED_DIM; i++) {
        buf1[i] += pos_embed->data[i];
    }

    /* 3. Transformer blocks - ping-pong between buf1 and buf2 */
    printf("  Running transformer blocks...\n");
    int total_calls = NUM_BLOCKS * REPEATED_TIMES;
    int call_idx    = 0;
    float *cur  = buf1;
    float *next = buf2;
    for (int b = 0; b < NUM_BLOCKS; b++) {
        int_to_str(idx_buf, b);
        str_concat(name, "blocks.", idx_buf);

        for (int r = 0; r < REPEATED_TIMES; r++) {
            call_idx++;
            printf("  === Block %d repeat %d  (%d/%d) ===\n", b, r, call_idx, total_calls);
            transformer_block((const float *)cur,
                              model, name, r, next, block_ws);
            printf("  === Block %d repeat %d DONE (%d/%d) ===\n\n", b, r, call_idx, total_calls);
            /* Swap ping-pong pointers */
            float *tmp = cur; cur = next; next = tmp;
        }
    }
    /* cur now points to the final block output */

    printf("  Running classification head...\n");

    /* 4. Final LayerNorm */
    Tensor *norm_w = find_tensor(model, "norm.weight");
    Tensor *norm_b = find_tensor(model, "norm.bias");
    layer_norm(cur, 1, SEQ_LEN, EMBED_DIM, norm_w->data, norm_b->data, x_normed);

    /* 5. Mean pooling over sequence (pooled already zeroed by alloc_tensor_1d) */
    for (int s = 0; s < SEQ_LEN; s++) {
        for (int d = 0; d < EMBED_DIM; d++) {
            pooled[d] += x_normed[s * EMBED_DIM + d];
        }
    }
    for (int d = 0; d < EMBED_DIM; d++) {
        pooled[d] /= SEQ_LEN;
    }
    debug_tensor("pooled [192]", pooled, EMBED_DIM);

    /* 6. Classification head */
    Tensor *head_w = find_tensor(model, "head.weight");
    Tensor *head_b = find_tensor(model, "head.bias");
    linear(pooled, 1, EMBED_DIM, head_w->data, head_b->data, 1000, logits);
    debug_tensor("logits [1000]", logits, 1000);

    aligned_free(buf1);
    aligned_free(buf2);
    aligned_free(x_normed);
    aligned_free(pooled);
    aligned_free(block_ws);
}

#endif /* MINIDEIT_H */
