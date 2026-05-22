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
/* Accelerator Configuration (must be before tinyclip_vision.h include)         */
/* ============================================================================ */
/* #define USE_ACCELERATOR 1 */

typedef int32_t token_t;

/* Single entry point for all accelerator drivers */
#include "accel_drivers.h"
/* ============================================================================ */

#include "tensor.h"
#include "nn_ops.h"
#include "tinyclip_vision.h"

/* ============================================================================ */
/* Test Configuration — Single Transformer Layer                                */
/* ============================================================================ */

/*
 * Memory layout (after model binary):
 *   MODEL_DATA_OFFSET + 0:             layer0_input.bin   [197*256 float32 = 201,728 bytes]
 *   MODEL_DATA_OFFSET + LAYER_IO_SIZE: layer0_output.bin  [197*256 float32 = 201,728 bytes]
 */
#define MODEL_DATA_OFFSET 0x2200000
#define LAYER_TOTAL_SIZE  (VISION_SEQ_LEN * VISION_EMBED_DIM)  /* 50432 */
#define LAYER_IO_SIZE     (LAYER_TOTAL_SIZE * (int)sizeof(float))  /* 201728 bytes */

/* Layer 0 weight tensor names */
static const char *LN1_W_NAME   = "_image_encoder.visual.transformer.resblocks.0.ln_1.weight";
static const char *LN1_B_NAME   = "_image_encoder.visual.transformer.resblocks.0.ln_1.bias";
static const char *ATTN_IN_W    = "_image_encoder.visual.transformer.resblocks.0.attn.in_proj_weight";
static const char *ATTN_IN_B    = "_image_encoder.visual.transformer.resblocks.0.attn.in_proj_bias";
static const char *ATTN_OUT_W   = "_image_encoder.visual.transformer.resblocks.0.attn.out_proj.weight";
static const char *ATTN_OUT_B   = "_image_encoder.visual.transformer.resblocks.0.attn.out_proj.bias";
static const char *LN2_W_NAME   = "_image_encoder.visual.transformer.resblocks.0.ln_2.weight";
static const char *LN2_B_NAME   = "_image_encoder.visual.transformer.resblocks.0.ln_2.bias";
static const char *MLP_FC_W     = "_image_encoder.visual.transformer.resblocks.0.mlp.c_fc.weight";
static const char *MLP_FC_B     = "_image_encoder.visual.transformer.resblocks.0.mlp.c_fc.bias";
static const char *MLP_PROJ_W   = "_image_encoder.visual.transformer.resblocks.0.mlp.c_proj.weight";
static const char *MLP_PROJ_B   = "_image_encoder.visual.transformer.resblocks.0.mlp.c_proj.bias";

/* 64-bit RISC-V cycle counter (Ariane is RV64, baremetal runs in M-mode) */
static inline uint64_t rdcycle64(void) {
    uint64_t val;
    __asm__ __volatile__("csrr %0, mcycle" : "=r"(val));
    return val;
}

int main(int argc, char * argv[])
{
    printf("============================================================\n");
    printf("  TinyCLIP Vision -- Single Transformer Layer Test\n");
    printf("  FPGA Baremetal Implementation\n");
    printf("============================================================\n\n");

    const int CPU_TILE_IDX = 1;
    esp_monitor_args_t mon_args;
    mon_args.read_mode = ESP_MON_READ_SINGLE;
    mon_args.tile_index = CPU_TILE_IDX;
    mon_args.mon_index = MON_DVFS_BASE_INDEX + 3;
    unsigned int cycle_start, cycle_end, cycle_diff;
    uint64_t cycle_start_64, cycle_end_64, cycle_diff_64;

    /* ================================================================ */
    /* Step 1: Load model (for layer 0 weights)                         */
    /* ================================================================ */

    printf("Loading model from address 0xB0000000...\n");

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

    /* Look up layer 0 weights */
    Tensor *ln1_w     = find_tensor(model, LN1_W_NAME);
    Tensor *ln1_b     = find_tensor(model, LN1_B_NAME);
    Tensor *attn_in_w = find_tensor(model, ATTN_IN_W);
    Tensor *attn_in_b = find_tensor(model, ATTN_IN_B);
    Tensor *attn_out_w= find_tensor(model, ATTN_OUT_W);
    Tensor *attn_out_b= find_tensor(model, ATTN_OUT_B);
    Tensor *ln2_w     = find_tensor(model, LN2_W_NAME);
    Tensor *ln2_b     = find_tensor(model, LN2_B_NAME);
    Tensor *mlp_fc_w  = find_tensor(model, MLP_FC_W);
    Tensor *mlp_fc_b  = find_tensor(model, MLP_FC_B);
    Tensor *mlp_proj_w= find_tensor(model, MLP_PROJ_W);
    Tensor *mlp_proj_b= find_tensor(model, MLP_PROJ_B);

    if (!ln1_w || !ln1_b || !attn_in_w || !attn_in_b ||
        !attn_out_w || !attn_out_b || !ln2_w || !ln2_b ||
        !mlp_fc_w || !mlp_fc_b || !mlp_proj_w || !mlp_proj_b) {
        printf("Error: Missing layer 0 weights\n");
        free_model(model);
        aligned_free(model);
        return 1;
    }
    printf("  All layer 0 weights found.\n");

#ifdef USE_ACCELERATOR
    accel_init_all();
#endif

    /* ================================================================ */
    /* Step 2: Load pre-computed layer input and expected output         */
    /* ================================================================ */

    float *layer_input    = (float *)((unsigned long)MB_ADDR + MODEL_DATA_OFFSET);
    float *expected_output = (float *)((unsigned long)MB_ADDR + MODEL_DATA_OFFSET + LAYER_IO_SIZE);

    printf("\n  Layer input at %p (%d floats)\n", layer_input, LAYER_TOTAL_SIZE);
    printf("  Expected output at %p (%d floats)\n", expected_output, LAYER_TOTAL_SIZE);

    /* Sanity check */
    printf("  Input first 3 values: ");
    print_float_debug(layer_input[0]);
    printf(" ");
    print_float_debug(layer_input[1]);
    printf(" ");
    print_float_debug(layer_input[2]);
    printf("\n");
    printf("  Expected output first 3 values: ");
    print_float_debug(expected_output[0]);
    printf(" ");
    print_float_debug(expected_output[1]);
    printf(" ");
    print_float_debug(expected_output[2]);
    printf("\n");

    /* Check if expected output is loaded (not all zeros) */
    int nonzero = 0;
    for (int i = 0; i < LAYER_TOTAL_SIZE && i < 100; i++) {
        if (expected_output[i] != 0.0f) { nonzero = 1; break; }
    }
    if (!nonzero) {
        printf("  WARNING: expected_output appears to be all zeros!\n");
    }

    /* ================================================================ */
    /* Step 3: Run single transformer layer                             */
    /* ================================================================ */

    float *computed_output = (float *)aligned_malloc(LAYER_IO_SIZE);
    if (!computed_output) {
        printf("Failed to allocate output buffer\n");
        free_model(model);
        aligned_free(model);
        return 1;
    }

    printf("\nRunning single transformer layer...\n");
    printf("  seq_len=%d  embed_dim=%d  heads=%d  mlp_dim=%d\n",
           VISION_SEQ_LEN, VISION_EMBED_DIM, VISION_HEADS, VISION_MLP_DIM);

    cycle_start = esp_monitor(mon_args, NULL);
    cycle_start_64 = rdcycle64();

    run_single_layer(
        layer_input,
        ln1_w->data, ln1_b->data,
        attn_in_w->data, attn_in_b->data,
        attn_out_w->data, attn_out_b->data,
        ln2_w->data, ln2_b->data,
        mlp_fc_w->data, mlp_fc_b->data,
        mlp_proj_w->data, mlp_proj_b->data,
        computed_output
    );

    cycle_end_64 = rdcycle64();
    cycle_end = esp_monitor(mon_args, NULL);
    cycle_diff = sub_monitor_vals(cycle_start, cycle_end);
    cycle_diff_64 = cycle_end_64 - cycle_start_64;
    printf("  Single layer time (esp_monitor 32-bit): %u cycles\n", cycle_diff);
    printf("  Single layer time (rdcycle    64-bit): %lu cycles\n", (unsigned long)cycle_diff_64);

    debug_tensor("computed_output", computed_output, LAYER_TOTAL_SIZE);

    /* ================================================================ */
    /* Step 4: Verification against golden reference                    */
    /* ================================================================ */

    printf("\n============================================================\n");
    printf("  Verification against golden reference\n");
    printf("============================================================\n");

    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    int max_diff_idx = 0;
    int mismatch_count = 0;
    float tolerance = 0.001f;  /* per-element tolerance */

    for (int i = 0; i < LAYER_TOTAL_SIZE; i++) {
        float diff = computed_output[i] - expected_output[i];
        if (diff < 0.0f) diff = -diff;
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
        avg_diff += diff;
        if (diff > tolerance) mismatch_count++;
    }
    avg_diff /= LAYER_TOTAL_SIZE;

    printf("  Total elements: %d\n", LAYER_TOTAL_SIZE);
    printf("  Elements > ");
    print_float_debug(tolerance);
    printf(" tolerance: %d\n", mismatch_count);
    printf("  Max diff: ");
    print_float_debug(max_diff);
    printf(" at index %d (row=%d, col=%d)\n",
           max_diff_idx, max_diff_idx / VISION_EMBED_DIM, max_diff_idx % VISION_EMBED_DIM);
    printf("  Avg diff: ");
    print_float_debug(avg_diff);
    printf("\n");

    /* Show first few element comparisons */
    printf("\n  First 5 element comparisons:\n");
    for (int i = 0; i < 5 && i < LAYER_TOTAL_SIZE; i++) {
        printf("    [%d] computed=", i);
        print_float_debug(computed_output[i]);
        printf("  expected=");
        print_float_debug(expected_output[i]);
        float d = computed_output[i] - expected_output[i];
        if (d < 0) d = -d;
        printf("  diff=");
        print_float_debug(d);
        printf("\n");
    }

    /* Show element at max diff */
    printf("\n  Max diff element [%d]:\n", max_diff_idx);
    printf("    computed=");
    print_float_debug(computed_output[max_diff_idx]);
    printf("  expected=");
    print_float_debug(expected_output[max_diff_idx]);
    printf("\n");

    /* PASS/FAIL verdict */
    printf("\n============================================================\n");
    if (max_diff < 0.001f) {
        printf("  PASS -- Output matches golden reference closely!\n");
    } else if (max_diff < 0.01f) {
        printf("  OK -- Small differences (expected with approx math)\n");
    } else if (max_diff < 0.1f) {
        printf("  WARN -- Moderate differences\n");
    } else {
        printf("  FAIL -- Large differences from golden reference\n");
    }
    printf("============================================================\n");

    printf("\n============================================================\n");
    printf("  Test completed\n");
    printf("============================================================\n");

    /* Cleanup */
    aligned_free(computed_output);
    free_model(model);
    aligned_free(model);

#ifdef USE_ACCELERATOR
    accel_cleanup_all();
#endif

    return 0;
}
