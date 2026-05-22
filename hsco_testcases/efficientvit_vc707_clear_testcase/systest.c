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
/* Accelerator Configuration (must be before efficientvit.h include)            */
/* ============================================================================ */
/* #define USE_ACCELERATOR 1 */

typedef int32_t token_t;

/* Single entry point for all accelerator drivers */
#include "accel_drivers.h"
/* ============================================================================ */

#include "tensor.h"
#include "nn_ops.h"
#include "efficientvit.h"

/* EfficientViT-M0 input: 3 x 224 x 224 */
#define INPUT_C 3
#define INPUT_H 224
#define INPUT_W 224
#define INPUT_SIZE (INPUT_C * INPUT_H * INPUT_W)
#define NUM_CLASSES 1000

/* Memory layout configuration */
/* Model binary size: ~9.4MB (9,383,201 bytes), round to 0xA00000 (10,485,760) */
#define MODEL_DATA_OFFSET 0xA00000

/* ImageNet class names (partial, for common classes) */
static const char* get_class_name(int idx) {
    switch (idx) {
        case 1: return "goldfish";
        case 59: return "vine_snake";
        case 78: return "tick";
        case 111: return "nematode";
        case 207: return "golden_retriever";
        case 281: return "tabby";
        case 282: return "tiger_cat";
        case 284: return "Siamese_cat";
        case 285: return "Egyptian_cat";
        case 287: return "lynx";
        case 292: return "tiger";
        case 388: return "giant_panda";
        case 404: return "airliner";
        case 424: return "barbershop";
        case 506: return "coil";
        case 620: return "laptop";
        case 817: return "sports_car";
        case 933: return "cheeseburger";
        case 948: return "Granny_Smith";
        case 954: return "banana";
        default: return "unknown";
    }
}

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
    printf("  EfficientViT-M0 Image Classification (FPGA Baremetal)\n");
    printf("============================================================\n\n");

    const int CPU_TILE_IDX = 1;
    esp_monitor_args_t mon_args;
    mon_args.read_mode = ESP_MON_READ_SINGLE;
    mon_args.tile_index = CPU_TILE_IDX;
    mon_args.mon_index = MON_DVFS_BASE_INDEX + 3;
    unsigned int cycle_start, cycle_end, cycle_diff;
    uint64_t cycle_start_64, cycle_end_64, cycle_diff_64;

    printf("Loading model from address 0xB0000000...\n");
    
    /* Allocate Model struct using aligned_malloc */
    /* Model is ~148KB (MAX_TENSORS=512), too large for stack in baremetal */
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
    printf("  Loaded %d tensors\n\n", model->num_tensors);

    /* Allocate output buffer on heap (not stack!) */
    float *logits = (float*)aligned_malloc(NUM_CLASSES * sizeof(float));
    if (!logits) {
        printf("Failed to allocate logits buffer\n");
        free_model(model);
        aligned_free(model);
        return 1;
    }

    float *probs = (float*)aligned_malloc(NUM_CLASSES * sizeof(float));
    if (!probs) {
        printf("Failed to allocate probs buffer\n");
        aligned_free(logits);
        free_model(model);
        aligned_free(model);
        return 1;
    }

#ifdef USE_ACCELERATOR
    accel_init_all();
#endif

    /* Test input is loaded after model binary */
    /* test_input.bin format: raw float32 array [3 * 224 * 224] = 602112 bytes */
    float *test_input = (float *)((unsigned long)MB_ADDR + MODEL_DATA_OFFSET);
    
    /* Expected output is loaded after test input */
    /* test_output.bin format: raw float32 array [1000] = 4000 bytes */
    float *expected_output = (float *)((unsigned long)MB_ADDR + MODEL_DATA_OFFSET + INPUT_SIZE * sizeof(float));

    printf("Running inference...\n");
    printf("  Input: %dx%dx%d at %p\n", INPUT_C, INPUT_H, INPUT_W, test_input);

    cycle_start = esp_monitor(mon_args, NULL);
    cycle_start_64 = rdcycle64();

    efficientvit_forward(test_input, model, logits);

    cycle_end_64 = rdcycle64();
    cycle_end = esp_monitor(mon_args, NULL);
    cycle_diff = sub_monitor_vals(cycle_start, cycle_end);
    cycle_diff_64 = cycle_end_64 - cycle_start_64;

    printf("\n  Inference time (esp_monitor 32-bit): %u cycles\n", cycle_diff);
    printf("  Inference time (rdcycle    64-bit): %lu cycles\n", (unsigned long)cycle_diff_64);

    /* Apply softmax to get probabilities */
    memcpy(probs, logits, NUM_CLASSES * sizeof(float));
    softmax(probs, NUM_CLASSES, probs);

    /* Get top-5 predictions */
    int top5_idx[5];
    float top5_val[5];
    topk(probs, NUM_CLASSES, 5, top5_idx, top5_val);

    printf("\n============================================================\n");
    printf("  Classification Results\n");
    printf("============================================================\n");
    printf("  Top-5 Predictions:\n");
    for (int i = 0; i < 5; i++) {
        printf("    %d. %s (class %d): ", i + 1, get_class_name(top5_idx[i]), top5_idx[i]);
        print_float(top5_val[i]);
        printf("\n");
    }
    printf("============================================================\n");

    /* Sanity check: print expected output to verify data was loaded */
    printf("\n  Expected output (from Python) at %p:\n", expected_output);
    debug_tensor("expected [1000]", expected_output, NUM_CLASSES);

    /* Check expected_output is not all zeros (data not loaded) */
    int nonzero_count = 0;
    for (int i = 0; i < NUM_CLASSES; i++) {
        if (expected_output[i] != 0.0f) nonzero_count++;
    }
    if (nonzero_count == 0) {
        printf("    WARNING: expected_output is ALL ZEROS - test_output.bin may not be loaded!\n");
    } else {
        printf("    expected_output has %d/%d non-zero values (OK)\n", nonzero_count, NUM_CLASSES);
    }

    /* Verify against expected output (logits) */
    printf("\n  Verification against Python:\n");
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    int max_diff_idx = 0;
    for (int i = 0; i < NUM_CLASSES; i++) {
        float diff = logits[i] - expected_output[i];
        if (diff < 0.0f) diff = -diff;
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
        avg_diff += diff;
    }
    avg_diff = avg_diff / NUM_CLASSES;

    /* Print diff with higher precision using scaled integers */
    /* print_float only shows 4 decimal places, masking tiny differences */
    /* So we also print as scaled integer: diff * 1000000 */
    int max_diff_scaled = (int)(max_diff * 1000000.0f + 0.5f);
    int avg_diff_scaled = (int)(avg_diff * 1000000.0f + 0.5f);
    
    printf("    Max logit diff: ");
    print_float(max_diff);
    printf(" (x1e6: %d) at index %d\n", max_diff_scaled, max_diff_idx);
    printf("      logits[%d]=", max_diff_idx);
    print_float(logits[max_diff_idx]);
    printf("  expected[%d]=", max_diff_idx);
    print_float(expected_output[max_diff_idx]);
    printf("\n");

    printf("    Avg logit diff: ");
    print_float(avg_diff);
    printf(" (x1e6: %d)\n", avg_diff_scaled);
    
    if (max_diff < 0.001f) {
        printf("    PASS - Results match Python implementation!\n");
    } else if (max_diff < 0.1f) {
        printf("    OK - Results close (small numerical differences)\n");
    } else {
        printf("    FAIL - Results differ significantly from Python (max_diff > 0.1)\n");
    }

    printf("\n============================================================\n");
    printf("  Test completed\n");
    printf("============================================================\n");

    aligned_free(probs);
    aligned_free(logits);
    free_model(model);
    aligned_free(model);

#ifdef USE_ACCELERATOR
    accel_cleanup_all();
#endif

    return 0;
}
