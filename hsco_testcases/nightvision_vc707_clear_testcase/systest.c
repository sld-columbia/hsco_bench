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
/* Accelerator Configuration (must be before nightvision.h include)             */
/* ============================================================================ */
/* #define USE_ACCELERATOR 1 */

typedef int32_t token_t;

/* Single entry point for all accelerator drivers */
#include "accel_drivers.h"
/* ============================================================================ */

#include "nightvision.h"

/*
 * Nightvision FPGA Baremetal
 *
 * Memory-mapped data layout at DATA_ADDR (0xB0000000):
 *
 * test_input.bin (raw, no header):
 *   [nRows: uint32] [nCols: uint32] [imgA: float32 x nPxls] [imgB: float32 x nPxls]
 *
 * gold_output.bin (raw, no header) placed after input:
 *   [nPxls: uint32] [ens1: float32 x nPxls] [ens2: float32 x nPxls] [ens3: float32 x nPxls]
 */

/* Image dimensions (30x40 from export_input.py) */
#define NROWS 30
#define NCOLS 40
#define NPXLS (NROWS * NCOLS)

/* Memory layout configuration */
#ifdef __riscv
#define DATA_ADDR 0xB0000000
#else
#define DATA_ADDR 0x50000000
#endif

/* test_input.bin: 2*4 (header) + 2*nPxls*4 (imgA+imgB) = 8 + 9600 = 9608 bytes */
#define INPUT_SIZE (2 * sizeof(uint32_t) + 2 * NPXLS * sizeof(float))

/* gold_output.bin: 4 (nPxls header) + 3*nPxls*4 (ens1+ens2+ens3) = 4 + 14400 = 14404 bytes */
#define GOLD_SIZE (sizeof(uint32_t) + 3 * NPXLS * sizeof(float))

/* Gold output is placed right after input data */
#define GOLD_OFFSET INPUT_SIZE

/* Helper: read uint32 from potentially unaligned address */
static inline uint32_t read_u32_unaligned(const uint8_t *ptr) {
    return (uint32_t)ptr[0] |
           ((uint32_t)ptr[1] << 8) |
           ((uint32_t)ptr[2] << 16) |
           ((uint32_t)ptr[3] << 24);
}

static int verify(const char *name, const float *got,
                   const float *exp, int n)
{
    int exact = 0, close = 0, bad = 0;
    int i;
    for (i = 0; i < n; i++) {
        int gi = (int)got[i];
        int ei = (int)exp[i];
        int diff = gi - ei;
        if (diff < 0) diff = -diff;
        if (diff == 0) {
            exact++;
        } else if (diff <= 1) {
            close++;
        } else {
            if (bad < 5)
                printf("  %s[%d]: got %d, expected %d (diff=%d)\n", name, i, gi, ei, gi - ei);
            bad++;
        }
    }
    if (bad) {
        printf("  %s: FAIL (%d exact, %d off-by-1, %d bad)\n", name, exact, close, bad);
    } else if (close) {
        printf("  %s: PASS (%d exact, %d off-by-1)\n", name, exact, close);
    } else {
        printf("  %s: PASS (%d pixels exact match)\n", name, n);
    }
    return bad;
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
    printf("  Nightvision Image Fusion (FPGA Baremetal)\n");
    printf("  Image: %dx%d (%d pixels)\n", NROWS, NCOLS, NPXLS);
    printf("============================================================\n\n");

    const int CPU_TILE_IDX = 1;
    esp_monitor_args_t mon_args;
    mon_args.read_mode = ESP_MON_READ_SINGLE;
    mon_args.tile_index = CPU_TILE_IDX;
    mon_args.mon_index = MON_DVFS_BASE_INDEX + 3;
    unsigned int cycle_start, cycle_end, cycle_diff;
    uint64_t cycle_start_64, cycle_end_64, cycle_diff_64;

#ifdef USE_ACCELERATOR
    accel_init_all();
#endif

    /* ---- Load input from memory-mapped address ---- */
    printf("Loading input data from address 0x%x...\n", (unsigned int)DATA_ADDR);

    uint8_t *mem = (uint8_t *)DATA_ADDR;
    uint32_t nR_raw = read_u32_unaligned(mem + 0);
    uint32_t nC_raw = read_u32_unaligned(mem + 4);
    int nR = (int)nR_raw;
    int nC = (int)nC_raw;
    int nP = nR * nC;

    printf("  nRows=%d, nCols=%d, nPixels=%d\n", nR, nC, nP);

    if (nR != NROWS || nC != NCOLS) {
        printf("ERROR: Expected %dx%d, got %dx%d\n", NROWS, NCOLS, nR, nC);
        return 1;
    }

    /* imgA and imgB are float32 arrays directly in memory-mapped space */
    float *srcA_mm = (float *)(mem + 8);
    float *srcB_mm = (float *)(mem + 8 + nP * sizeof(float));

    /* Copy to heap-allocated float buffers */
    float *srcA = (float *)aligned_malloc(nP * sizeof(float));
    float *srcB = (float *)aligned_malloc(nP * sizeof(float));
    if (!srcA || !srcB) {
        printf("Failed to allocate source image buffers\n");
        return 1;
    }
    memcpy(srcA, srcA_mm, nP * sizeof(float));
    memcpy(srcB, srcB_mm, nP * sizeof(float));

    printf("  srcA[0..2]: %d %d %d\n", (int)srcA[0], (int)srcA[1], (int)srcA[2]);
    printf("  srcB[0..2]: %d %d %d\n", (int)srcB[0], (int)srcB[1], (int)srcB[2]);

    /* Allocate work buffers */
    float *buf1 = (float *)aligned_malloc(nP * sizeof(float));
    float *buf2 = (float *)aligned_malloc(nP * sizeof(float));
    float *sc1  = (float *)aligned_malloc(nP * sizeof(float));
    float *sc2  = (float *)aligned_malloc(nP * sizeof(float));
    float *dwtF = (float *)aligned_malloc(nP * sizeof(float));
    float *lapF = (float *)aligned_malloc(nP * sizeof(float));
    float *e1   = (float *)aligned_malloc(nP * sizeof(float));
    float *e2   = (float *)aligned_malloc(nP * sizeof(float));
    float *e3   = (float *)aligned_malloc(nP * sizeof(float));
    int *hA  = (int *)aligned_malloc((1 << 16) * sizeof(int));
    int *hB  = (int *)aligned_malloc((1 << 16) * sizeof(int));
    int *CDF = (int *)aligned_malloc((1 << 16) * sizeof(int));
    int *LUT = (int *)aligned_malloc((1 << 16) * sizeof(int));

    if (!buf1 || !buf2 || !sc1 || !sc2 || !dwtF || !lapF ||
        !e1 || !e2 || !e3 || !hA || !hB || !CDF || !LUT) {
        printf("Failed to allocate work buffers\n");
        return 1;
    }

    /* Temp buffers for kernel calls */
    float *pxlList = (float *)aligned_malloc(MEDIAN_NELEM * sizeof(float));
    float *dwtTmp  = (float *)aligned_malloc(nP * sizeof(float));
    float *rsTmp   = (float *)aligned_malloc(nP * sizeof(float));
    if (!pxlList || !dwtTmp || !rsTmp) {
        printf("Failed to allocate temp buffers\n");
        return 1;
    }

    /* Zero-init all work buffers (aligned_malloc doesn't zero) */
    memset(buf1, 0, nP * sizeof(float));
    memset(buf2, 0, nP * sizeof(float));
    memset(sc1,  0, nP * sizeof(float));
    memset(sc2,  0, nP * sizeof(float));
    memset(dwtF, 0, nP * sizeof(float));
    memset(lapF, 0, nP * sizeof(float));
    memset(e1,   0, nP * sizeof(float));
    memset(e2,   0, nP * sizeof(float));
    memset(e3,   0, nP * sizeof(float));
    memset(hA,   0, (1 << 16) * sizeof(int));
    memset(hB,   0, (1 << 16) * sizeof(int));

    /* ---- Full pipeline (mirrors original benchmark.c) ---- */
    printf("\nRunning pipeline...\n");

    cycle_start = esp_monitor(mon_args, NULL);
    cycle_start_64 = rdcycle64();

    imageMat2Gray(srcA, buf1, 16, NROWS, NCOLS);

    memcpy(buf1, srcA, nP * sizeof(float));
    memcpy(buf2, srcB, nP * sizeof(float));
    printf("  nf (noise filter)...\n");
    nf(buf1, sc1, NROWS, NCOLS, pxlList);
    nf(buf2, sc2, NROWS, NCOLS, pxlList);
    memcpy(buf1, sc1, nP * sizeof(float));
    memcpy(buf2, sc2, nP * sizeof(float));

    printf("  histogram equalization...\n");
    hist(srcA, hA, NROWS, NCOLS, 16);
    hist(srcB, hB, NROWS, NCOLS, 16);
    histEq(srcA, srcA, hA, NROWS, NCOLS, 16, 10, CDF, LUT);
    histEq(srcB, srcB, hB, NROWS, NCOLS, 16, 10, CDF, LUT);

    printf("  DWT...\n");
    memcpy(buf1, srcA, nP * sizeof(float));
    memcpy(buf2, srcB, nP * sizeof(float));
    dwt53(buf1, buf1, NROWS, NCOLS, dwtTmp);
    dwt53(buf2, buf2, NROWS, NCOLS, dwtTmp);
    dwt53_inverse(buf1, buf1, NROWS, NCOLS, dwtTmp);
    dwt53_inverse(buf2, buf2, NROWS, NCOLS, dwtTmp);

    printf("  DWT fusion...\n");
    dwtFusion(buf1, buf2, dwtF, NROWS, NCOLS, sc1, sc2, dwtTmp);
    rescaleImage(dwtF, rsTmp, NROWS, NCOLS, 16);
    memcpy(dwtF, rsTmp, nP * sizeof(float));

    printf("  Laplacian fusion...\n");
    lapFusionDD(buf1, buf2, lapF, NROWS, NCOLS);

    /* Reload original inputs for ensemble processing */
    memcpy(srcA, srcA_mm, nP * sizeof(float));
    memcpy(srcB, srcB_mm, nP * sizeof(float));

    printf("  ensemble_1...\n");
    ensemble_1(srcA, e1, NROWS, NCOLS, 16, 10);
    printf("  ensemble_2...\n");
    ensemble_2(srcA, srcB, e2, NROWS, NCOLS, 16, 10);
    printf("  ensemble_3...\n");
    ensemble_3(srcA, srcB, e3, NROWS, NCOLS, 16, 10);
    rescaleImage(e2, rsTmp, NROWS, NCOLS, 16);
    memcpy(e2, rsTmp, nP * sizeof(float));
    rescaleImage(e3, rsTmp, NROWS, NCOLS, 16);
    memcpy(e3, rsTmp, nP * sizeof(float));

    cycle_end_64 = rdcycle64();
    cycle_end = esp_monitor(mon_args, NULL);
    cycle_diff = sub_monitor_vals(cycle_start, cycle_end);
    cycle_diff_64 = cycle_end_64 - cycle_start_64;

    printf("Pipeline done.\n");
    printf("  Execution time (esp_monitor 32-bit): %u cycles\n", cycle_diff);
    printf("  Execution time (rdcycle    64-bit): %lu cycles\n\n", (unsigned long)cycle_diff_64);

    /* ---- Verify against gold output ---- */
    printf("============================================================\n");
    printf("  Verification\n");
    printf("============================================================\n");

    uint8_t *gold_mem = mem + GOLD_OFFSET;
    uint32_t gold_nPxls = read_u32_unaligned(gold_mem);
    printf("  Gold nPixels: %d\n", (int)gold_nPxls);

    if ((int)gold_nPxls != nP) {
        printf("  WARNING: gold nPxls=%d != nP=%d, skipping verification\n",
               (int)gold_nPxls, nP);
    } else {
        float *g1 = (float *)(gold_mem + 4);
        float *g2 = (float *)(gold_mem + 4 + nP * sizeof(float));
        float *g3 = (float *)(gold_mem + 4 + 2 * nP * sizeof(float));

        int fail = 0;
        fail += verify("ensemble_1", e1, g1, nP);
        fail += verify("ensemble_2", e2, g2, nP);
        fail += verify("ensemble_3", e3, g3, nP);

        printf("\n============================================================\n");
        if (fail) {
            printf("  TEST FAILED (%d total mismatches)\n", fail);
        } else {
            printf("  ALL TESTS PASSED\n");
        }
        printf("============================================================\n");
    }
    
    printf("\n============================================================\n");
    printf("  Test completed\n");
    printf("============================================================\n");

    /* Cleanup */
    aligned_free(srcA); aligned_free(srcB);
    aligned_free(buf1); aligned_free(buf2); aligned_free(sc1); aligned_free(sc2);
    aligned_free(dwtF); aligned_free(lapF);
    aligned_free(e1); aligned_free(e2); aligned_free(e3);
    aligned_free(hA); aligned_free(hB);
    aligned_free(pxlList); aligned_free(dwtTmp); aligned_free(rsTmp);

#ifdef USE_ACCELERATOR
    accel_cleanup_all();
#endif

    return 0;
}
