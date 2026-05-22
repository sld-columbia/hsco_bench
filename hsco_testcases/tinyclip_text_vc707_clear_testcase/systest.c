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
/* Accelerator Configuration (must be before tinyclip_text.h include)           */
/* ============================================================================ */
/* #define USE_ACCELERATOR 1 */

typedef int32_t token_t;

/* Single entry point for all accelerator drivers */
#include "accel_drivers.h"

/* 64-bit RISC-V cycle counter (Ariane is RV64, baremetal runs in M-mode) */
static inline uint64_t rdcycle64(void) {
    uint64_t val;
    __asm__ __volatile__("csrr %0, mcycle" : "=r"(val));
    return val;
}
/* ============================================================================ */

#include "tensor.h"
#include "nn_ops.h"
#include "tinyclip_text.h"

/* ============================================================================ */
/* Test Configuration                                                           */
/* ============================================================================ */

/* Memory layout configuration */
/* Text-only model binary size: ~10 MB (10,084,861 bytes original) */
/* Aligned version is slightly larger due to padding (~10.1 MB) */
/* Round up to 0xA00000 = 10,485,760 bytes (~10 MB) */
#define MODEL_DATA_OFFSET 0xA00000

/* text_embeds.bin layout: */
/*   [num_queries (u32)] [context_length (u32)] [embed_dim (u32)] */
/*   [eot_positions (int32[nq])] */
/*   [embeds (float32[nq * cl * ed])] */
/*   For 1 query: 12 + 4 + 1*77*256*4 = 78,864 bytes */
#define TEXT_EMBEDS_SIZE 78864

/* text_features_golden.bin layout: */
/*   [num_queries (u32)] [feature_dim (u32)] [float32 data...] */
/*   For 1 query: 8 + 1*512*4 = 2,056 bytes */
#define TEXT_FEATURES_GOLDEN_SIZE 2056

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

int main(int argc, char * argv[])
{
    printf("============================================================\n");
    printf("  TinyCLIP Text-Only Encoder\n");
    printf("  FPGA Baremetal Implementation\n");
    printf("============================================================\n\n");

    const int CPU_TILE_IDX = 1;
    esp_monitor_args_t mon_args;
    mon_args.read_mode = ESP_MON_READ_SINGLE;
    mon_args.tile_index = CPU_TILE_IDX;
    mon_args.mon_index = MON_DVFS_BASE_INDEX + 3;
    unsigned int cycle_start, cycle_end, cycle_diff;
    uint64_t cycle_start_64, cycle_end_64, cycle_diff_64;

    printf("Loading text-only model from address 0xB0000000...\n");

    /* Allocate Model struct using aligned_malloc */
    /* Model has ~40 tensors, struct ~148 KB, way too large for stack in baremetal */
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

#ifdef USE_ACCELERATOR
    accel_init_all();
#endif

    /* ================================================================ */
    /* Load test data from memory (after model binary)                  */
    /* ================================================================ */

    /* Pre-computed text embeddings loaded after model binary */
    uint8_t *text_embeds_addr = (uint8_t *)((unsigned long)MB_ADDR + MODEL_DATA_OFFSET);

    float *embeds = NULL;
    int32_t *eot_positions = NULL;
    int num_queries, context_length, embed_dim;
    if (load_text_embeds_from_memory(text_embeds_addr, &embeds, &eot_positions,
                                     &num_queries, &context_length, &embed_dim) != 0) {
        printf("Failed to load text embeddings\n");
        free_model(model);
        aligned_free(model);
        return 1;
    }
    printf("  Text embeds at %p (%d queries, ctx=%d, dim=%d)\n",
           embeds, num_queries, context_length, embed_dim);

    if (embed_dim != TEXT_EMBED_DIM) {
        printf("Error: embed_dim (%d) != TEXT_EMBED_DIM (%d)\n",
               embed_dim, TEXT_EMBED_DIM);
        free_model(model);
        aligned_free(model);
        return 1;
    }

    for (int i = 0; i < num_queries; i++) {
        printf("    query %d: EOT at position %d\n", i, (int)eot_positions[i]);
    }

    /* Golden text features loaded after text embeddings */
    uint8_t *golden_addr = (uint8_t *)((unsigned long)MB_ADDR + MODEL_DATA_OFFSET + TEXT_EMBEDS_SIZE);

    float *golden_features = NULL;
    int golden_nq, golden_dim;
    if (load_golden_features_from_memory(golden_addr, &golden_features,
                                          &golden_nq, &golden_dim) != 0) {
        printf("Failed to load golden features\n");
        free_model(model);
        aligned_free(model);
        return 1;
    }
    printf("  Golden features at %p (%d queries, %d-dim)\n",
           golden_features, golden_nq, golden_dim);

    if (golden_dim != TEXT_OUTPUT_DIM) {
        printf("Error: golden feature dim (%d) != TEXT_OUTPUT_DIM (%d)\n",
               golden_dim, TEXT_OUTPUT_DIM);
        free_model(model);
        aligned_free(model);
        return 1;
    }

    /* Quick sanity check on text embeddings */
    printf("  First 3 embed values: ");
    print_float(embeds[0]);
    printf(" ");
    print_float(embeds[1]);
    printf(" ");
    print_float(embeds[2]);
    printf("\n");

    /* Quick sanity check on golden features */
    printf("  First 3 golden values: ");
    print_float(golden_features[0]);
    printf(" ");
    print_float(golden_features[1]);
    printf(" ");
    print_float(golden_features[2]);
    printf("\n\n");

    /* ================================================================ */
    /* Allocate output buffer on heap                                   */
    /* ================================================================ */

    float *text_features = (float *)aligned_malloc(num_queries * TEXT_OUTPUT_DIM * sizeof(float));
    if (!text_features) {
        printf("Failed to allocate text features buffer\n");
        free_model(model);
        aligned_free(model);
        return 1;
    }

    /* ================================================================ */
    /* Encode text features from pre-computed embeddings                */
    /* ================================================================ */

    printf("Encoding text features from pre-computed embeddings...\n");
    cycle_start = esp_monitor(mon_args, NULL);
    cycle_start_64 = rdcycle64();

    tinyclip_encode_text_from_embeds(embeds, eot_positions, num_queries,
                                     context_length, model, text_features);

    cycle_end_64 = rdcycle64();
    cycle_end = esp_monitor(mon_args, NULL);
    cycle_diff = sub_monitor_vals(cycle_start, cycle_end);
    cycle_diff_64 = cycle_end_64 - cycle_start_64;
    printf("  Text encoding time: %u cycles (esp_monitor 32-bit)\n", cycle_diff);
    printf("  Text encoding time: %llu cycles (rdcycle64 64-bit)\n", (unsigned long long)cycle_diff_64);
    debug_tensor("text_features", text_features, num_queries * TEXT_OUTPUT_DIM);

    /* ================================================================ */
    /* Verification against golden reference                           */
    /* ================================================================ */

    printf("\n============================================================\n");
    printf("  Verification against golden reference\n");
    printf("============================================================\n");

    if (golden_nq != num_queries) {
        printf("  WARNING: golden queries (%d) != computed queries (%d)\n",
               golden_nq, num_queries);
    }

    int total_elems = num_queries * TEXT_OUTPUT_DIM;

    /* Check golden is not all zeros (data not loaded) */
    int nonzero_count = 0;
    for (int i = 0; i < total_elems; i++) {
        if (golden_features[i] != 0.0f) nonzero_count++;
    }

    if (nonzero_count == 0) {
        printf("  WARNING: golden features is ALL ZEROS - data may not be loaded!\n");
    } else {
        printf("  Golden features has %d/%d non-zero values (OK)\n", nonzero_count, total_elems);

        /* Compute max/avg diff */
        float max_diff = 0.0f;
        float avg_diff = 0.0f;
        int max_diff_idx = 0;
        for (int i = 0; i < total_elems; i++) {
            float diff = text_features[i] - golden_features[i];
            if (diff < 0.0f) diff = -diff;
            if (diff > max_diff) {
                max_diff = diff;
                max_diff_idx = i;
            }
            avg_diff += diff;
        }
        avg_diff = avg_diff / total_elems;

        printf("\n  First 8 computed vs golden:\n");
        int show = total_elems < 8 ? total_elems : 8;
        for (int i = 0; i < show; i++) {
            float diff = text_features[i] - golden_features[i];
            if (diff < 0.0f) diff = -diff;
            printf("    [%d] computed=", i);
            print_float(text_features[i]);
            printf("  golden=");
            print_float(golden_features[i]);
            printf("  diff=");
            print_float(diff);
            printf("\n");
        }

        printf("\n  Max diff: ");
        print_float(max_diff);
        printf(" at index %d\n", max_diff_idx);
        printf("  Avg diff: ");
        print_float(avg_diff);
        printf("\n");

        if (max_diff < 0.001f) {
            printf("  PASS - Text features match golden closely!\n");
        } else if (max_diff < 0.01f) {
            printf("  OK - Small numerical differences from approx math\n");
        } else if (max_diff < 0.1f) {
            printf("  WARN - Moderate differences (expected with approx math)\n");
        } else {
            printf("  FAIL - Large differences from golden\n");
        }
    }

    printf("\n============================================================\n");
    printf("  Test completed\n");
    printf("============================================================\n");

    /* Cleanup */
    aligned_free(text_features);
    /* embeds, eot_positions, golden_features point into memory-mapped region, no free needed */
    free_model(model);
    aligned_free(model);

#ifdef USE_ACCELERATOR
    accel_cleanup_all();
#endif

    return 0;
}
