/* Copyright (c) 2011-2026 Columbia University, System Level Design Group */
/* SPDX-License-Identifier: Apache-2.0 */

#include <stdio.h>
#ifndef __riscv
#include <stdlib.h>
#endif

#include <esp_accelerator.h>
#include <esp_probe.h>
#include "monitors.h"

/* ============================================================================ */
/* Accelerator Configuration (must be before minilm.h include)                  */
/* ============================================================================ */
/* #define USE_ACCELERATOR 1 */

typedef int32_t token_t;

/* Single entry point for all accelerator drivers */
#include "accel_drivers.h"
/* ============================================================================ */

#include "tensor.h"
#include "nn_ops.h"
#include "minilm.h"

/* MiniLM lite input: pre-computed embeddings, seq_len=128, hidden=384 */
#define SEQ_LEN 128

/* Memory layout configuration */
/* Model binary size: ~21.9MB (21,887,620 bytes), round to 0x1500000 (22,020,096) */
#define MODEL_DATA_OFFSET 0x1500000

/* test_input.bin layout (raw, no header): */
/*   [128 * 384 float32] pre-computed embeddings = 196,608 bytes */
/*   [128 int32] attention_mask = 512 bytes */
/* Total: 197,120 bytes */
#define EMBEDDING_SIZE (SEQ_LEN * HIDDEN_SIZE * (int)sizeof(float))
#define MASK_SIZE (SEQ_LEN * (int)sizeof(int))
#define TEST_INPUT_SIZE (EMBEDDING_SIZE + MASK_SIZE)

/* test_output.bin layout (raw, no header): */
/*   [384 float32] expected sentence embedding */
/* Total: 1,536 bytes */
#define TEST_OUTPUT_SIZE (HIDDEN_SIZE * (int)sizeof(float))

static void print_float(float x)
{
    int neg = 0;
    if (x < 0) { neg = 1; x = -x; }
    int scaled = (int)(x * 10000.0f + 0.5f);
    int ipart  = scaled / 10000;
    int fpart  = scaled % 10000;
    if (neg) printf("-");
    printf("%d.%04d", ipart, fpart);
}

/* 64-bit RISC-V cycle counter (Ariane is RV64, baremetal runs in M-mode) */
static inline uint64_t rdcycle64(void) {
    uint64_t val;
    __asm__ __volatile__("csrr %0, mcycle" : "=r"(val));
    return val;
}

int main(int argc, char * argv[])
{
    printf("============================================================\n");
    printf("  paraphrase-MiniLM-L3-v2 Sentence Embedding (FPGA Lite)\n");
    printf("  Lite mode: pre-computed embeddings + encoder only\n");
    printf("============================================================\n\n");

    const int CPU_TILE_IDX = 1;
    esp_monitor_args_t mon_args;
    mon_args.read_mode = ESP_MON_READ_SINGLE;
    mon_args.tile_index = CPU_TILE_IDX;
    mon_args.mon_index = MON_DVFS_BASE_INDEX + 3;
    unsigned int cycle_start, cycle_end, cycle_diff;
    uint64_t cycle_start_64, cycle_end_64, cycle_diff_64;

    printf("Loading encoder model from address 0xB0000000...\n");

    /* Allocate Model struct using aligned_malloc */
    /* Model has ~50 tensors, struct ~18KB, too large for stack in baremetal */
    Model *model = (Model *)aligned_malloc(sizeof(Model));
    if (!model) {
        printf("Failed to allocate Model struct\n");
        return 1;
    }
    printf("  Model struct allocated at %p (size=%d bytes)\n", model, (int)sizeof(Model));

    if (load_model_from_memory(model) != 0) {
        printf("Failed to load model\n");
        aligned_free(model);
        return 1;
    }
    printf("  Loaded %d tensors (encoder + pooler)\n\n", model->num_tensors);

    /* Allocate output buffer on heap (not stack!) */
    float *embedding = (float*)aligned_malloc(HIDDEN_SIZE * sizeof(float));
    if (!embedding) {
        printf("Failed to allocate embedding buffer\n");
        free_model(model);
        aligned_free(model);
        return 1;
    }

#ifdef USE_ACCELERATOR
    accel_init_all();
#endif

    /* Pre-computed embeddings loaded after model binary */
    /* Format: raw [128*384 float32 embeddings][128 int32 attention_mask] */
    float *precomputed_embeddings = (float *)((unsigned long)MB_ADDR + MODEL_DATA_OFFSET);
    int *attention_mask_int = (int *)((unsigned long)MB_ADDR + MODEL_DATA_OFFSET + EMBEDDING_SIZE);

    float *attention_mask = (float *)aligned_malloc(SEQ_LEN * sizeof(float));
    for (int i = 0; i < SEQ_LEN; i++) attention_mask[i] = (float)attention_mask_int[i];

    /* Expected output loaded after test input */
    /* Format: raw [384 float32 embedding] */
    float *expected_output = (float *)((unsigned long)MB_ADDR + MODEL_DATA_OFFSET + TEST_INPUT_SIZE);

    /* Count actual (non-padding) tokens */
    int actual_tokens = 0;
    for (int i = 0; i < SEQ_LEN; i++) {
        if (attention_mask_int[i] == 1) actual_tokens++;
    }

    printf("Running inference (encoder + mean pooling)...\n");
    printf("  Sequence length: %d\n", SEQ_LEN);
    printf("  Hidden size: %d\n", HIDDEN_SIZE);
    printf("  Actual tokens: %d\n", actual_tokens);
    printf("  Embeddings at %p\n", precomputed_embeddings);
    printf("  Attention mask at %p\n", attention_mask);
    printf("  First 3 embedding values: ");
    print_float(precomputed_embeddings[0]);
    printf(" ");
    print_float(precomputed_embeddings[1]);
    printf(" ");
    print_float(precomputed_embeddings[2]);
    printf("\n");

    cycle_start = esp_monitor(mon_args, NULL);
    cycle_start_64 = rdcycle64();

    minilm_forward_from_embeddings(precomputed_embeddings, attention_mask, model, embedding);

    cycle_end_64 = rdcycle64();
    cycle_end = esp_monitor(mon_args, NULL);
    cycle_diff = sub_monitor_vals(cycle_start, cycle_end);
    cycle_diff_64 = cycle_end_64 - cycle_start_64;

    printf("\n  Inference time (esp_monitor 32-bit): %u cycles\n", cycle_diff);
    printf("  Inference time (rdcycle    64-bit): %lu cycles\n", (unsigned long)cycle_diff_64);

    /* Print embedding results */
    printf("\n============================================================\n");
    printf("  Embedding Results\n");
    printf("============================================================\n");
    printf("  Embedding dimension: %d\n", HIDDEN_SIZE);
    debug_tensor("embedding [384]", embedding, HIDDEN_SIZE);

    /* Compute L2 norm */
    float norm = 0.0f;
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        norm += embedding[i] * embedding[i];
    }
    norm = sqrtf_approx(norm);
    printf("  L2 norm: ");
    print_float(norm);
    printf("\n");
    printf("============================================================\n");

    /* Sanity check: print expected output to verify data was loaded */
    printf("\n  Expected output (from Python) at %p:\n", expected_output);
    debug_tensor("expected [384]", expected_output, HIDDEN_SIZE);

    /* Check expected_output is not all zeros (data not loaded) */
    int nonzero_count = 0;
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        if (expected_output[i] != 0.0f) nonzero_count++;
    }
    if (nonzero_count == 0) {
        printf("    WARNING: expected_output is ALL ZEROS - test_output.bin may not be loaded!\n");
    } else {
        printf("    expected_output has %d/%d non-zero values (OK)\n", nonzero_count, HIDDEN_SIZE);
    }

    /* Verify against expected output */
    printf("\n  Verification against Python:\n");
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    int max_diff_idx = 0;
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        float diff = embedding[i] - expected_output[i];
        if (diff < 0.0f) diff = -diff;
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
        avg_diff += diff;
    }
    avg_diff = avg_diff / HIDDEN_SIZE;

    /* Print diff with higher precision using scaled integers */
    int max_diff_scaled = (int)(max_diff * 1000000.0f + 0.5f);
    int avg_diff_scaled = (int)(avg_diff * 1000000.0f + 0.5f);

    printf("    Max diff: ");
    print_float(max_diff);
    printf(" (x1e6: %d) at index %d\n", max_diff_scaled, max_diff_idx);
    printf("      embedding[%d]=", max_diff_idx);
    print_float(embedding[max_diff_idx]);
    printf("  expected[%d]=", max_diff_idx);
    print_float(expected_output[max_diff_idx]);
    printf("\n");

    printf("    Avg diff: ");
    print_float(avg_diff);
    printf(" (x1e6: %d)\n", avg_diff_scaled);

    /* Cosine similarity */
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        dot += embedding[i] * expected_output[i];
        norm_a += embedding[i] * embedding[i];
        norm_b += expected_output[i] * expected_output[i];
    }
    float cosine_sim = dot / (sqrtf_approx(norm_a) * sqrtf_approx(norm_b));
    printf("    Cosine similarity: ");
    print_float(cosine_sim);
    printf("\n");

    if (max_diff < 0.001f) {
        printf("    PASS - Results match Python implementation!\n");
    } else if (max_diff < 0.05f) {
        printf("    OK - Results close (small numerical differences from approx math)\n");
    } else if (max_diff < 0.1f) {
        printf("    WARN - Results have moderate differences\n");
    } else {
        printf("    FAIL - Results differ significantly from Python (max_diff > 0.1)\n");
    }

    printf("\n============================================================\n");
    printf("  Test completed\n");
    printf("============================================================\n");

    aligned_free(attention_mask);
    aligned_free(embedding);
    free_model(model);
    aligned_free(model);

#ifdef USE_ACCELERATOR
    accel_cleanup_all();
#endif

    return 0;
}
