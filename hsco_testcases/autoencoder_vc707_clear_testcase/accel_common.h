/*
 * Accelerator Common Utilities
 * Shared macros and functions for all ESP hardware accelerators
 *
 * This file provides:
 *   - DMA scatter/gather macros
 *   - Fixed-point conversion utilities
 *   - Memory buffer size configuration
 *
 * Include this ONCE before any accelerator driver headers.
 * Requires: token_t typedef, esp_accelerator.h, esp_probe.h
 */

#ifndef ACCEL_COMMON_H
#define ACCEL_COMMON_H

#ifdef USE_ACCELERATOR

#include <esp_accelerator.h>
#include <esp_probe.h>

// ============================================================================
// DMA scatter/gather configuration (Do not modify)
// ============================================================================
#define DMA_RATIO 1
#define CHUNK_SHIFT 20
#define CHUNK_SIZE BIT(CHUNK_SHIFT)
#define NCHUNK(_sz) ((_sz % CHUNK_SIZE == 0) ? \
            (_sz / CHUNK_SIZE) : \
            (_sz / CHUNK_SIZE) + 1)

// ============================================================================
// Accelerator memory buffer size
// Shared across all accelerators (sized for largest operation)
// For autoencoder: largest layer is 640x128, need input + weight + output
// Max size: 640*128*4 + 640*4 + 128*4 = 330240 bytes, round to 512KB
// ============================================================================
#define ACCEL_MEM_SIZE (512 * 1024)

// ============================================================================
// Shared accelerator memory (allocated once, used by all accelerators)
// ============================================================================
static token_t *accel_shared_mem;
static unsigned **accel_shared_ptable;
static int accel_mem_ready;

// Must be called ONCE at program start before any driver init.
// BSS is not zeroed on Ariane baremetal, so we cannot rely on
// static initializers — explicit assignment is required.
static void accel_common_init(void) {
    accel_shared_mem = NULL;
    accel_shared_ptable = NULL;
    accel_mem_ready = 0;
}

static int accel_common_alloc(void) {
    if (accel_mem_ready) return 0;  // already allocated

    accel_shared_mem = aligned_malloc(sizeof(token_t) * ACCEL_MEM_SIZE);
    if (!accel_shared_mem) {
        printf("Failed to allocate shared accelerator memory\n");
        return -1;
    }
    printf("  Shared accelerator memory: %p\n", accel_shared_mem);

    accel_shared_ptable = aligned_malloc(NCHUNK(ACCEL_MEM_SIZE) * sizeof(unsigned *));
    if (!accel_shared_ptable) {
        printf("Failed to allocate shared page table\n");
        return -1;
    }
    for (int i = 0; i < NCHUNK(ACCEL_MEM_SIZE); i++) {
        accel_shared_ptable[i] = (unsigned *)&accel_shared_mem[i * (CHUNK_SIZE / sizeof(token_t))];
    }
    printf("  Shared page table: %p, nchunk=%lu\n", accel_shared_ptable, NCHUNK(ACCEL_MEM_SIZE));

    accel_mem_ready = 1;
    return 0;
}

static void accel_common_free(void) {
    if (accel_shared_ptable) {
        aligned_free(accel_shared_ptable);
        accel_shared_ptable = NULL;
    }
    if (accel_shared_mem) {
        aligned_free(accel_shared_mem);
        accel_shared_mem = NULL;
    }
}

// ============================================================================
// Fixed-point conversion utilities
// ============================================================================
static inline token_t float2fixed(float x, int wl, int il) {
    int fl = wl - il;
    float scale = (float)(1 << fl);
    int val = (int)(x * scale + (x >= 0 ? 0.5f : -0.5f));
    return (token_t)val;
}

static inline float fixed2float(token_t x, int wl, int il) {
    int fl = wl - il;
    float scale = (float)(1 << fl);
    return (float)((int)x) / scale;
}

#endif // USE_ACCELERATOR

#endif // ACCEL_COMMON_H
