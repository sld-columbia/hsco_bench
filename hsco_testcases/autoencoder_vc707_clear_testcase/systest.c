/* Copyright (c) 2011-2026 Columbia University, System Level Design Group */
/* SPDX-License-Identifier: Apache-2.0 */

#include <stdio.h>
#ifndef __riscv
#include <stdlib.h>
#endif

#include <esp_accelerator.h>
#include <esp_probe.h>
#include "monitors.h"

// ============================================================================
// Accelerator Configuration (must be before autoencoder.h include)
// ============================================================================
// #define USE_ACCELERATOR 1

typedef int32_t token_t;

// Single entry point for all accelerator drivers
#include "accel_drivers.h"
// ============================================================================

#include "tensor.h"
#include "nn_ops.h"
#include "autoencoder.h"

#define NUM_TEST_SAMPLES 5
#define INPUT_DIM 640

// Memory layout configuration
#define MODEL_DATA_OFFSET 0x110000  // Offset after model binary for test data

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
    printf("  Autoencoder Anomaly Detection (FPGA Baremetal)\n");
    printf("============================================================\n\n");

    const int CPU_TILE_IDX = 1;
    esp_monitor_args_t mon_args;
    mon_args.read_mode = ESP_MON_READ_SINGLE;
    mon_args.tile_index = CPU_TILE_IDX;
    mon_args.mon_index = MON_DVFS_BASE_INDEX + 3;
    unsigned int cycle_start, cycle_end, cycle_diff;
    uint64_t cycle_start_64, cycle_end_64, cycle_diff_64;

    printf("Loading model from address 0xB0000000...\n");
    
    // Allocate Model struct using aligned_malloc (like FPGA example)
    // Model is ~37KB, too large for stack in baremetal environment
    Model *model = (Model *)aligned_malloc(sizeof(Model));
    if (!model) {
        printf("Failed to allocate Model struct\n");
        return 1;
    }
    printf("  Model struct allocated at %p\n", model);
    
    if (load_model_from_memory(model) != 0) {
        printf("Failed to load model\n");
        aligned_free(model);
        return 1;
    }
    printf("  Loaded %d tensors\n\n", model->num_tensors);

    printf("Allocating work buffers...\n");
    WorkBuffers *work_buffers = alloc_work_buffers();
    if (!work_buffers) {
        printf("Failed to allocate work buffers\n");
        free_model(model);
        aligned_free(model);
        return 1;
    }

    float *output = (float*)aligned_malloc(INPUT_DIM * sizeof(float));
    if (!output) {
        printf("Failed to allocate output buffer\n");
        free_work_buffers(work_buffers);
        free_model(model);
        aligned_free(model);
        return 1;
    }

#ifdef USE_ACCELERATOR
    accel_init_all();
#endif

    // Test data is loaded after model binary
    // Model binary size: ~1.1MB (1,081,665 bytes), rounded to 0x110000 for safe alignment
    // test_input.bin format: [num_samples(uint32)][feature_dim(uint32)][data(float32 array)]
    uint32_t *test_data_base = (uint32_t *)((unsigned long)MB_ADDR + MODEL_DATA_OFFSET);
    uint32_t num_samples = test_data_base[0];
    uint32_t feature_dim = test_data_base[1];
    float *test_inputs = (float *)(&test_data_base[2]);
    
    // Test labels are loaded after test_input.bin (8 bytes header + 12800 bytes data = 12808 bytes)
    int *test_labels = (int *)((unsigned long)test_data_base + 12808);

    printf("\nRunning anomaly detection on %d test samples...\n", NUM_TEST_SAMPLES);
    printf("============================================================\n");

    float scores[NUM_TEST_SAMPLES];
    int num_correct = 0;

    cycle_start = esp_monitor(mon_args, NULL);
    cycle_start_64 = rdcycle64();

    for (int i = 0; i < NUM_TEST_SAMPLES; i++) {
        printf("  Sample %d: ", i + 1);
        
        float *input = test_inputs + i * INPUT_DIM;
        
        printf("fwd..");
        autoencoder_forward(input, model, work_buffers, output);
        
        printf("score..");
        scores[i] = compute_anomaly_score(input, output, INPUT_DIM);
        
        int label = test_labels[i];
        const char *label_str = (label == 0) ? "NORMAL" : "ANOMALY";
        
        printf("MSE=");
        print_float(scores[i]);
        printf(" [%s]\n", label_str);
    }

    cycle_end_64 = rdcycle64();
    cycle_end = esp_monitor(mon_args, NULL);
    cycle_diff = sub_monitor_vals(cycle_start, cycle_end);
    cycle_diff_64 = cycle_end_64 - cycle_start_64;

    printf("============================================================\n");
    printf("\n  Total execution time (esp_monitor 32-bit): %u cycles\n", cycle_diff);
    printf("  Total execution time (rdcycle    64-bit): %lu cycles\n", (unsigned long)cycle_diff_64);
    printf("  Average time per sample (32-bit): %u cycles\n", cycle_diff / NUM_TEST_SAMPLES);
    printf("  Average time per sample (64-bit): %lu cycles\n", (unsigned long)(cycle_diff_64 / NUM_TEST_SAMPLES));

    printf("\n============================================================\n");
    printf("  Anomaly Detection Summary\n");
    printf("============================================================\n");

    float max_normal = -1e9f;
    float min_anomaly = 1e9f;
    for (int i = 0; i < NUM_TEST_SAMPLES; i++) {
        int label = test_labels[i];
        if (label == 0 && scores[i] > max_normal) {
            max_normal = scores[i];
        }
        if (label == 1 && scores[i] < min_anomaly) {
            min_anomaly = scores[i];
        }
    }
    float threshold = (max_normal + min_anomaly) / 2.0f;

    printf("  Threshold (midpoint): ");
    print_float(threshold);
    printf("\n\n  Predictions:\n");

    int tp = 0, tn = 0, fp = 0, fn = 0;
    for (int i = 0; i < NUM_TEST_SAMPLES; i++) {
        int predicted = scores[i] > threshold ? 1 : 0;
        int actual = test_labels[i];
        
        const char *status;
        if (predicted == 1 && actual == 1) { tp++; status = "TP"; }
        else if (predicted == 0 && actual == 0) { tn++; status = "TN"; }
        else if (predicted == 1 && actual == 0) { fp++; status = "FP"; }
        else { fn++; status = "FN"; }
        
        printf("    Sample %d: score=", i + 1);
        print_float(scores[i]);
        printf(", pred=%s, actual=%s (%s)\n",
               predicted ? "ANOMALY" : "NORMAL",
               actual ? "ANOMALY" : "NORMAL",
               status);
        
        if (predicted == actual) {
            num_correct++;
        }
    }

    printf("\n  Results: TP=%d, TN=%d, FP=%d, FN=%d\n", tp, tn, fp, fn);
    if (tp + fp > 0) {
        int precision = (100 * tp) / (tp + fp);
        printf("  Precision: %d%%\n", precision);
    }
    if (tp + fn > 0) {
        int recall = (100 * tp) / (tp + fn);
        printf("  Recall: %d%%\n", recall);
    }
    int accuracy = (100 * (tp + tn)) / NUM_TEST_SAMPLES;
    printf("  Accuracy: %d%%\n", accuracy);

    printf("\n============================================================\n");
    printf("  Test completed: %d/%d correct\n", num_correct, NUM_TEST_SAMPLES);
    printf("============================================================\n");

    aligned_free(output);
    free_work_buffers(work_buffers);
    free_model(model);
    aligned_free(model);

#ifdef USE_ACCELERATOR
    accel_cleanup_all();
#endif

    return 0;
}
