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
/* Accelerator Configuration (must be before wami.h include)                    */
/* ============================================================================ */
/* #define USE_ACCELERATOR 1 */

typedef int32_t token_t;

/* Single entry point for all accelerator drivers */
#include "accel_drivers.h"
/* ============================================================================ */

#include "wami.h"

#define LK_ITERATIONS 20

/*
 * WAMI FPGA Baremetal
 *
 * Memory-mapped data layout at DATA_ADDR (0xB0000000):
 *
 * test_input.bin (raw, no conversion needed):
 *   [srcRows: u32] [srcCols: u32] [padding: u32] [nRows: u32] [nCols: u32]
 *   [nModels: u32] [nTestImages: u32]
 *   [start_image: float x nRows*nCols]
 *   [mu: float x nRows*nCols*nModels]
 *   [sigma: float x nRows*nCols*nModels]
 *   [weight: float x nRows*nCols*nModels]
 *   For each test image:
 *     [bayer_image: float x srcRows*srcCols]
 *
 * gold_output.bin (raw) placed after input:
 *   [nTestImages: u32] [nRows: u32] [nCols: u32]
 *   For each test image:
 *     [foreground_mask: u8 x nRows*nCols]
 *     [num_foreground: int32]
 */

/* Memory layout configuration */
#ifdef __riscv
#define DATA_ADDR 0xB0000000
#else
#define DATA_ADDR 0x50000000
#endif

/* Helper: read uint32 from potentially unaligned address */
static inline uint32_t read_u32_unaligned(const uint8_t *ptr) {
    return (uint32_t)ptr[0] |
           ((uint32_t)ptr[1] << 8) |
           ((uint32_t)ptr[2] << 16) |
           ((uint32_t)ptr[3] << 24);
}

/* Helper: read int32 from potentially unaligned address */
static inline int32_t read_i32_unaligned(const uint8_t *ptr) {
    return (int32_t)read_u32_unaligned(ptr);
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
    printf("  WAMI Signal Processing (FPGA Baremetal)\n");
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
    printf("Loading input data from address 0x%08X...\n", (unsigned)DATA_ADDR);

    uint8_t *mem = (uint8_t *)DATA_ADDR;
    uint32_t offset = 0;

    uint32_t srcRows = read_u32_unaligned(mem + offset); offset += 4;
    uint32_t srcCols = read_u32_unaligned(mem + offset); offset += 4;
    uint32_t padding = read_u32_unaligned(mem + offset); offset += 4;
    uint32_t nRows   = read_u32_unaligned(mem + offset); offset += 4;
    uint32_t nCols   = read_u32_unaligned(mem + offset); offset += 4;
    uint32_t nModels = read_u32_unaligned(mem + offset); offset += 4;
    uint32_t nTestImages = read_u32_unaligned(mem + offset); offset += 4;

    printf("  srcRows=%d, srcCols=%d, padding=%d\n", (int)srcRows, (int)srcCols, (int)padding);
    printf("  nRows=%d, nCols=%d, nModels=%d, nTestImages=%d\n",
           (int)nRows, (int)nCols, (int)nModels, (int)nTestImages);

    if ((int)padding != 2) {
        printf("ERROR: Expected padding=%d, got %d\n", 2, (int)padding);
        return 1;
    }

    int M = nRows;
    int N = nCols;
    int nMN = M * N;

    if (M != WAMI_PROC_ROWS || N != WAMI_PROC_COLS) {
        printf("ERROR: Expected processed image %dx%d, got %dx%d\n",
               WAMI_PROC_ROWS, WAMI_PROC_COLS, M, N);
        return 1;
    }
    printf("  Processed image: %dx%d = %d pixels\n", M, N, nMN);
    int model_sz = nMN * nModels;
    int img_sz = srcRows * srcCols;

    /* Pointers directly into memory-mapped region (read-only for start_image, images) */
    float *start_image_mm = (float *)(mem + offset);
    offset += nMN * sizeof(float);

    /* mu/sigma/weight need to be writable (GMM modifies them in-place) */
    float *mu_mm    = (float *)(mem + offset); offset += model_sz * sizeof(float);
    float *sigma_mm = (float *)(mem + offset); offset += model_sz * sizeof(float);
    float *weight_mm = (float *)(mem + offset); offset += model_sz * sizeof(float);

    /* Bayer images are in memory-mapped space (float) */
    float *images_mm = (float *)(mem + offset);
    uint32_t images_offset = offset;
    offset += nTestImages * img_sz * sizeof(float);
    uint32_t input_size = offset;  /* total size of test_input.bin */

    printf("  start_image at offset 28, mu/sigma/weight follow\n");
    printf("  images at offset %u\n", images_offset);
    printf("  input_size = %u bytes, gold at 0x%08X\n",
           input_size, (unsigned)(DATA_ADDR + input_size));

    /* GMM model: separate input/output buffers (double-buffered, swap each frame) */
    float *mu_a     = (float *)aligned_malloc(model_sz * sizeof(float));
    float *sigma_a  = (float *)aligned_malloc(model_sz * sizeof(float));
    float *weight_a = (float *)aligned_malloc(model_sz * sizeof(float));
    float *mu_b     = (float *)aligned_malloc(model_sz * sizeof(float));
    float *sigma_b  = (float *)aligned_malloc(model_sz * sizeof(float));
    float *weight_b = (float *)aligned_malloc(model_sz * sizeof(float));
    if (!mu_a || !sigma_a || !weight_a || !mu_b || !sigma_b || !weight_b) {
        printf("Failed to allocate GMM model buffers\n");
        return 1;
    }
    memcpy(mu_a, mu_mm, model_sz * sizeof(float));
    memcpy(sigma_a, sigma_mm, model_sz * sizeof(float));
    memcpy(weight_a, weight_mm, model_sz * sizeof(float));
    /* mu_in/sigma_in/weight_in point to current input, out_* to current output */
    float *mu_in = mu_a;
    float *sigma_in = sigma_a;
    float *weight_in = weight_a;
    float *mu_out = mu_b;
    float *sigma_out = sigma_b;
    float *weight_out = weight_b;

    /* Allocate working buffers on heap (stack too small on Ariane) */
    rgb_pixel *rgbtmpimg = (rgb_pixel *)aligned_malloc(nMN * sizeof(rgb_pixel));
    float *img      = (float *)aligned_malloc(nMN * sizeof(float));
    float *tmplt    = (float *)aligned_malloc(nMN * sizeof(float));
    float *IWxp     = NULL;
    float *lk_error_img  = (float *)aligned_malloc(nMN * sizeof(float));
    float *lk_img_dx     = (float *)aligned_malloc(nMN * sizeof(float));
    float *lk_img_dy     = (float *)aligned_malloc(nMN * sizeof(float));
    float *lk_nabla_Ix   = (float *)aligned_malloc(nMN * sizeof(float));
    float *lk_nabla_Iy   = (float *)aligned_malloc(nMN * sizeof(float));
    float *lk_I_steepest = (float *)aligned_malloc(6 * nMN * sizeof(float));
    float *foreground    = (float *)aligned_malloc(nMN * sizeof(float));
    float *all_foregrounds = (float *)aligned_malloc(nMN * nTestImages * sizeof(float));
    int32_t *all_num_fgs = (int32_t *)aligned_malloc(nTestImages * sizeof(int32_t));

    if (!rgbtmpimg || !img || !tmplt ||
        !lk_error_img || !lk_img_dx || !lk_img_dy || !lk_nabla_Ix || !lk_nabla_Iy || !lk_I_steepest ||
        !foreground ||
        !all_foregrounds || !all_num_fgs ||
        !mu_a || !sigma_a || !weight_a || !mu_b || !sigma_b || !weight_b) {
        printf("Failed to allocate work buffers\n");
        return 1;
    }

    /* Zero-init all work buffers */
    memset(rgbtmpimg, 0, nMN * sizeof(rgb_pixel));
    memset(img, 0, nMN * sizeof(float));
    memset(tmplt, 0, nMN * sizeof(float));
    memset(lk_error_img, 0, nMN * sizeof(float));
    memset(lk_img_dx, 0, nMN * sizeof(float));
    memset(lk_img_dy, 0, nMN * sizeof(float));
    memset(lk_nabla_Ix, 0, nMN * sizeof(float));
    memset(lk_nabla_Iy, 0, nMN * sizeof(float));
    memset(lk_I_steepest, 0, 6 * nMN * sizeof(float));
    memset(foreground, 0, nMN * sizeof(float));
    memset(all_foregrounds, 0, nMN * nTestImages * sizeof(float));
    memset(all_num_fgs, 0, nTestImages * sizeof(int32_t));

    /* Copy start_image to tmplt (IWxp starts as start_image) */
    /* In the original code: IWxp = start_image initially */
    float *start_img_copy = (float *)aligned_malloc(nMN * sizeof(float));
    if (!start_img_copy) {
        printf("Failed to allocate start_image copy\n");
        return 1;
    }
    memcpy(start_img_copy, start_image_mm, nMN * sizeof(float));
    IWxp = start_img_copy;

    /* ---- Run WAMI Pipeline ---- */
    printf("\nRunning WAMI pipeline (%d frames, %dx%d)...\n",
           (int)nTestImages, (int)M, (int)N);
    /* Rough estimate: LK hessian dominates at ~500M cycles/iter at 50MHz */
    /* Per frame: 20 iters * ~50 * MN cycles / 50e6 = 20*MN/1e6 seconds */
    {
        unsigned int est_per_frame_sec = (unsigned int)((unsigned int)LK_ITERATIONS * nMN / 50000u);
        unsigned int est_total_sec = est_per_frame_sec * nTestImages;
        printf("  WARNING: Estimated ~%u sec/frame at 50MHz (total ~%u sec)\n",
               est_per_frame_sec, est_total_sec);
        printf("  Use wami_small_c_simple for smaller test data if too slow.\n\n");
    }

    cycle_start = esp_monitor(mon_args, NULL);
    cycle_start_64 = rdcycle64();

    float *swap;
    float warp_p[6];

    for (int a = 0; a < nTestImages; a++) {
        float *bayer_img = images_mm + img_sz * a;
        swap = tmplt; tmplt = IWxp; IWxp = swap;
        if (a == 0) {
            IWxp = (float *)aligned_malloc(nMN * sizeof(float));
            if (!IWxp) {
                printf("Failed to allocate IWxp\n");
                return 1;
            }
        }
        memset(IWxp, 0, sizeof(float) * nMN);

        wami_debayer(bayer_img, rgbtmpimg, srcRows, srcCols);
        printf("  Frame %d: debayer done  rgb[0]=(", (int)a);
        print_float_debug(rgbtmpimg[0].r); printf(",");
        print_float_debug(rgbtmpimg[0].g); printf(",");
        print_float_debug(rgbtmpimg[0].b); printf(")\n");

        rgb_to_grayscale(rgbtmpimg, img, WAMI_PROC_ROWS, WAMI_PROC_COLS);
        printf("  Frame %d: grayscale done  img[0]=", (int)a);
        print_float_debug(img[0]); printf(" img[100]=");
        print_float_debug(img[100]); printf("\n");

        for (int i = 0; i < 6; i++) warp_p[i] = 0;
        printf("  Frame %d: lucas-kanade (%d iters)...\n", (int)a, LK_ITERATIONS);
        kernel2_lucas_kanade(img, tmplt, warp_p, LK_ITERATIONS, WAMI_PROC_ROWS, WAMI_PROC_COLS, IWxp,
                             lk_error_img, lk_img_dx, lk_img_dy, lk_nabla_Ix, lk_nabla_Iy, lk_I_steepest);
        {
          printf("  Frame %d: LK done  warp=[", (int)a);
          for (int wi = 0; wi < 6; wi++) {
            if (wi) printf(",");
            print_float_debug(warp_p[wi]);
          }
          printf("]\n");
          printf("           IWxp[0]="); print_float_debug(IWxp[0]);
          printf(" IWxp[100]="); print_float_debug(IWxp[100]); printf("\n");
        }

        printf("  Frame %d: GMM...\n", (int)a);
        wami_gmm(WAMI_PROC_ROWS, WAMI_PROC_COLS, WAMI_GMM_NUM_MODELS, IWxp,
                 mu_in, sigma_in, weight_in,
                 mu_out, sigma_out, weight_out, foreground);

        /* Swap: this frame's output becomes next frame's input */
        { float *t;
          t = mu_in;     mu_in = mu_out;     mu_out = t;
          t = sigma_in;  sigma_in = sigma_out;  sigma_out = t;
          t = weight_in; weight_in = weight_out; weight_out = t;
        }

        /* Compute num_foreground from float array */
        int f = 0;
        for (int i = 0; i < nMN; i++)
            f += (foreground[i] > 0.5f) ? 1 : 0;

        printf("  Frame %d: GMM done  num_foreground=%d\n", (int)a, f);
        memcpy(all_foregrounds + a * nMN, foreground, nMN * sizeof(float));
        all_num_fgs[a] = f;
    }

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

    uint8_t *gold_mem = mem + input_size;
    uint32_t gold_nTest = read_u32_unaligned(gold_mem);
    uint32_t gold_nRows = read_u32_unaligned(gold_mem + 4);
    uint32_t gold_nCols = read_u32_unaligned(gold_mem + 8);

    printf("  Gold: nTestImages=%d, nRows=%d, nCols=%d\n",
           (int)gold_nTest, (int)gold_nRows, (int)gold_nCols);

    if (gold_nTest != nTestImages || gold_nRows != M || gold_nCols != N) {
        printf("  WARNING: gold dimensions mismatch, skipping verification\n");
    } else {
        int total_errors = 0;
        uint32_t gold_off = 12;  /* skip header: 3 * uint32 */

        for (int a = 0; a < nTestImages; a++) {
            uint8_t *gold_fg = gold_mem + gold_off;
            gold_off += nMN;
            int32_t gold_nfg = read_i32_unaligned(gold_mem + gold_off);
            gold_off += 4;

            int errors = 0;
            for (int i = 0; i < nMN; i++) {
                int got = (all_foregrounds[a * nMN + i] > 0.5f) ? 1 : 0;
                errors += (got != (int)gold_fg[i]);
            }
            errors += (all_num_fgs[a] != gold_nfg);

            if (errors) {
                printf("  Frame %d: FAIL (%d errors, got nfg=%d, gold nfg=%d)\n",
                       (int)a, errors, (int)all_num_fgs[a], (int)gold_nfg);
            } else {
                printf("  Frame %d: PASS (nfg=%d)\n", (int)a, (int)all_num_fgs[a]);
            }
            total_errors += errors;
        }

        printf("\n============================================================\n");
        if (total_errors) {
            printf("  TEST FAILED (%d total errors)\n", total_errors);
        } else {
            printf("  ALL TESTS PASSED\n");
        }
        printf("============================================================\n");
    }

    printf("\n============================================================\n");
    printf("  Test completed\n");
    printf("============================================================\n");

    /* Cleanup */
    /* Note: start_img_copy was swapped into tmplt at frame 0, freed as tmplt below */
    aligned_free(mu_a); aligned_free(sigma_a); aligned_free(weight_a);
    aligned_free(mu_b); aligned_free(sigma_b); aligned_free(weight_b);
    aligned_free(foreground); aligned_free(rgbtmpimg);
    aligned_free(img); aligned_free(tmplt); aligned_free(IWxp);
    aligned_free(lk_error_img); aligned_free(lk_img_dx); aligned_free(lk_img_dy);
    aligned_free(lk_nabla_Ix); aligned_free(lk_nabla_Iy); aligned_free(lk_I_steepest);
    aligned_free(all_foregrounds); aligned_free(all_num_fgs);

#ifdef USE_ACCELERATOR
    accel_cleanup_all();
#endif

    return 0;
}
