#ifndef WAMI_SIMPLE_H
#define WAMI_SIMPLE_H

#include <stdio.h>
#ifndef __riscv
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <assert.h>
#endif
#include <string.h>
#include <stdint.h>

/* aligned_malloc/aligned_free are provided by ESP baremetal linker.
 * Must declare here so wami.c sees the correct return type (void *).
 * Without this, implicit declaration assumes int return → pointer truncation on RV64. */
#ifdef __riscv
extern void *aligned_malloc(int size);
extern void  aligned_free(void *ptr);
#endif

typedef struct _rgb_pixel
{
    float r, g, b;
} rgb_pixel;

#define WAMI_GMM_NUM_MODELS 5
/* Processed image dimensions (srcRows/srcCols - 2*2 debayer padding).
 * Must match the test data; validated at runtime in systest.c. */
#define WAMI_PROC_ROWS 32
#define WAMI_PROC_COLS 32

/* --- Math Approximations for RISC-V Baremetal --------------------------- */

static inline float fabsf_approx(float x) {
    return x < 0.0f ? -x : x;
}

static inline float sqrtf_approx(float x) {
#ifdef __riscv
    if (x <= 0.0f) return 0.0f;
    float xhalf = 0.5f * x;
    int i = *(int*)&x;
    i = 0x5f3759df - (i >> 1);
    x = *(float*)&i;
    x = x * (1.5f - xhalf * x * x);
    x = x * (1.5f - xhalf * x * x);
    return 1.0f / x;
#else
    return sqrtf(x);
#endif
}

static inline float expf_approx(float x) {
#ifdef __riscv
    if (x < -87.0f) return 0.0f;
    if (x > 88.0f) return 3.4e38f;
    const float LOG2E = 1.44269504f;
    const float LN2   = 0.69314718f;
    float t = x * LOG2E;
    int k = (int)(t + (t >= 0 ? 0.5f : -0.5f));
    float r = x - k * LN2;
    float exp_r = 1.0f + r * (1.0f + r * (0.5f + r * (0.166666667f +
                  r * (0.041666667f + r * 0.008333333f))));
    union { float f; int32_t i; } u;
    u.i = (127 + k) << 23;
    return exp_r * u.f;
#else
    return expf(x);
#endif
}

static inline float floorf_approx(float x) {
#ifdef __riscv
    int i = (int)x;
    return (float)(x < (float)i ? i - 1 : i);
#else
    return floorf(x);
#endif
}

/* --- Debug float printing (RISC-V printf has no %f) ---------------------- */

static inline void print_float_debug(float x) {
    int neg = 0;
    if (x < 0) { neg = 1; x = -x; }
    if (neg) printf("-");
    if (x >= 2000000000.0f) {
        int m = (int)(x / 1000000.0f);
        printf("%de6", m);
    } else {
        int ipart = (int)x;
        int fpart = (int)((x - (float)ipart) * 10000.0f + 0.5f);
        if (fpart >= 10000) { ipart++; fpart -= 10000; }
        printf("%d.%04d", ipart, fpart);
    }
}

/* Public API — called from systest.c */
void wami_debayer(const float *bayer, rgb_pixel *debayered, int nrows, int ncols);
void rgb_to_grayscale(const rgb_pixel *color, float *out, int nrows, int ncols);
void kernel2_lucas_kanade(const float *img, const float *tmplt,
                          float *affine_warp, int num_iterations,
                          int nRows, int nCols,
                          float *warped_tmplt,
                          float *error_img, float *img_dx, float *img_dy,
                          float *nabla_Ix, float *nabla_Iy, float *I_steepest);
void wami_gmm(int nRows, int nCols, int nModels,
              const float *frame, const float *mu_in,
              const float *sigma_in, const float *weight_in,
              float *mu_dst, float *sigma_dst, float *weight_dst,
              float *foreground);

#endif
