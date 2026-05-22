/*
 * DUMMY Accelerator Driver (REFERENCE ONLY)
 *
 * This is a structural reference driver that pairs with the dummy
 * reference HLS kernel. It demonstrates every part of an ESP accelerator
 * driver (probe, DMA setup, fixed-point conversion, register write,
 * polling, output conversion) but the backing kernel performs only an
 * identity copy and is NOT useful for any real computation. Copy and
 * adapt this driver when writing a driver for your own accelerator.
 *
 * Provides:
 *   dummy_accel_init()    - Probe and initialize accelerator
 *   dummy_accel_exec()    - Execute accelerated operation (identity)
 *   dummy_accel_cleanup() - Release accelerator resources
 *
 * Requires: accel_common.h included before this file
 *           token_t typedef defined before this file
 */

#ifndef DUMMY_DRIVER_H
#define DUMMY_DRIVER_H

#include "dummy_cfg.h"

#ifdef USE_ACCELERATOR

// ============================================================================
// DUMMY accelerator state
//
// NOTE (Ariane baremetal): BSS is NOT guaranteed to be zeroed. Never rely on
// the static initializers below; always assign explicitly in dummy_accel_init()
// before any value is read. An explicit flag gates later use.
// ============================================================================
static struct esp_device *dummy_espdevs = NULL;
static struct esp_device *dummy_dev = NULL;
static unsigned dummy_coherence = ACC_COH_NONE;
static int dummy_init_done = 0;

// ============================================================================
// Initialize DUMMY accelerator
// Returns 0 on success, -1 on failure
// ============================================================================
static int dummy_accel_init(void) {
    int n;

    // Explicitly zero state (BSS may contain garbage on Ariane baremetal)
    dummy_espdevs = NULL;
    dummy_dev = NULL;
    dummy_coherence = ACC_COH_NONE;
    dummy_init_done = 0;

    printf("Scanning for DUMMY accelerator...\n");

    n = probe(&dummy_espdevs, VENDOR_SLD, DUMMY_DEV_ID, DUMMY_DEV_NAME);
    if (n == 0) {
        printf("DUMMY accelerator not found\n");
        return -1;
    }

    printf("Found %s\n", DUMMY_DEV_NAME);
    dummy_dev = &dummy_espdevs[0];

    // Check scatter-gather DMA
    if (ioread32(dummy_dev, PT_NCHUNK_MAX_REG) == 0) {
        printf("  -> scatter-gather DMA is disabled. Abort.\n");
        return -1;
    }
    if (ioread32(dummy_dev, PT_NCHUNK_MAX_REG) < NCHUNK(ACCEL_MEM_SIZE)) {
        printf("  -> Not enough TLB entries available. Abort.\n");
        return -1;
    }

    // Allocate shared memory (no-op if already allocated by another driver)
    if (accel_common_alloc() != 0) {
        return -1;
    }

    dummy_coherence = ACC_COH_NONE;
    dummy_init_done = 1;

    printf("DUMMY Accelerator initialized successfully\n");
    return 0;
}

// ============================================================================
// Execute DUMMY accelerated operation
// ============================================================================
static void dummy_accel_exec(
    const float *A,
    const float *B,
    float *O,
    int rows,
    int cols,
    int loaded_cols
) {
    if (!dummy_init_done) {
        printf("DUMMY exec called without successful init\n");
        return;
    }

    int A_size = rows * loaded_cols;
    int B_size = loaded_cols * cols;
    int O_size = rows * cols;

    // Memory layout: [A][gap?][B][gap?][O]
    // The HLS kernel rounds each region's DMA offset up to DMA_WORD_PER_BEAT,
    // so the driver must allocate A/B/O with the same alignment or the kernel
    // will DMA from/to the wrong address when any size is not a multiple of
    // DMA_WORD_PER_BEAT.
    #define ROUND_UP_BEAT(n) \
        (((n) + DUMMY_DMA_WORD_PER_BEAT - 1) / DUMMY_DMA_WORD_PER_BEAT \
         * DUMMY_DMA_WORD_PER_BEAT)
    int A_size_adj = ROUND_UP_BEAT(A_size);
    int B_size_adj = ROUND_UP_BEAT(B_size);
    int O_size_adj = ROUND_UP_BEAT(O_size);

    int A_offset = 0;
    int B_offset = A_size_adj;
    int O_offset = A_size_adj + B_size_adj;

    // Load A - convert to fixed-point (only write the real A_size elements;
    // any padding tokens in [A_size, A_size_adj) are "don't care").
    for (int i = 0; i < A_size; i++) {
        accel_shared_mem[A_offset + i] = float2fixed(A[i], DUMMY_FX_WL, DUMMY_FX_IL);
    }

    // Load B - convert to fixed-point
    for (int i = 0; i < B_size; i++) {
        accel_shared_mem[B_offset + i] = float2fixed(B[i], DUMMY_FX_WL, DUMMY_FX_IL);
    }

    // Note: No explicit fence needed — accel_shared_mem is in uncached
    // region, writes go directly to DDR. iowrite32 + esp_flush provide ordering.

    // Calculate actual memory size needed (use aligned sizes so we cover the
    // full range the accelerator may touch, including alignment padding).
    unsigned mem_size = (A_size_adj + B_size_adj + O_size_adj) * sizeof(token_t);
    #undef ROUND_UP_BEAT

    // Configure accelerator
    iowrite32(dummy_dev, COHERENCE_REG, dummy_coherence);
#ifndef __sparc
    iowrite32(dummy_dev, PT_ADDRESS_REG, (unsigned long long)accel_shared_ptable);
#else
    iowrite32(dummy_dev, PT_ADDRESS_REG, (unsigned)accel_shared_ptable);
#endif
    iowrite32(dummy_dev, PT_NCHUNK_REG, NCHUNK(mem_size));
    iowrite32(dummy_dev, PT_SHIFT_REG, CHUNK_SHIFT);
    iowrite32(dummy_dev, SRC_OFFSET_REG, 0x0);
    iowrite32(dummy_dev, DST_OFFSET_REG, 0x0);

    // Write DUMMY-specific configuration
    iowrite32(dummy_dev, DUMMY_REG_ROWS, rows);
    iowrite32(dummy_dev, DUMMY_REG_COLS, cols);
    iowrite32(dummy_dev, DUMMY_REG_LOADED_COLS, loaded_cols);

    // Flush caches
    esp_flush(dummy_coherence);

    // Note: No explicit fence needed before CMD_START — iowrite32 uses
    // volatile MMIO stores serialized on AXI bus, and esp_flush provides ordering.

    // Start accelerator
    iowrite32(dummy_dev, CMD_REG, CMD_MASK_START);

    // Wait for completion
    unsigned done = 0;
    while (!done) {
        done = ioread32(dummy_dev, STATUS_REG);
        done &= STATUS_MASK_DONE;
    }
    iowrite32(dummy_dev, CMD_REG, 0x0);

    // Read O - convert from fixed-point
    for (int i = 0; i < O_size; i++) {
        O[i] = fixed2float(accel_shared_mem[O_offset + i], DUMMY_FX_WL, DUMMY_FX_IL);
    }
}

// ============================================================================
// Cleanup DUMMY accelerator resources
// ============================================================================
static void dummy_accel_cleanup(void) {
    // Shared memory is freed by accel_common_free()
    dummy_dev = NULL;
    dummy_espdevs = NULL;
    dummy_init_done = 0;
}

#endif // USE_ACCELERATOR

#endif // DUMMY_DRIVER_H
