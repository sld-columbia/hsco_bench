#ifndef _NIGHTVISION_H
#define _NIGHTVISION_H

/* --- Platform / Utility ------------------------------------------------- */

#include <stdio.h>
#ifndef __riscv
#include <stdlib.h>
#endif
#include <string.h>
#include <stdint.h>

/* aligned_malloc/aligned_free are provided by ESP baremetal linker.
 * Must declare here so nightvision.c sees the correct return type (void *).
 * Without this, implicit declaration assumes int return → pointer truncation on RV64. */
#ifdef __riscv
extern void *aligned_malloc(int size);
extern void  aligned_free(void *ptr);
#endif


#ifndef INT_MAX
#define INT_MAX 0x7FFFFFFF
#endif
#ifndef INT_MIN
#define INT_MIN (-INT_MAX - 1)
#endif

#define ABS(X)     ((X) < 0 ? (-(X)) : (X))
#define MAX(A,B)   ((A) > (B) ? (A) : (B))
#define MIN(A,B)   ((A) < (B) ? (A) : (B))

#define CLIP_INRANGE(LOW,VAL,HIGH) \
    ((VAL) < (LOW) ? (LOW) : ((VAL) > (HIGH) ? (HIGH) : (VAL)))

#define _USE_DOUBLEDENSITY_LAP_FUSION

#define MEDIAN_NELEM 9

/* --- Math Approximations for RISC-V Baremetal --------------------------- */

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

/* --- Function Declarations ---------------------------------------------- */

/* slowmedian */
void slowMedian3x3(const float *src, float *dst, int nRows, int nCols,
                   float *pxlList);

/* nf (noise filter) */
void nf(const float *streamA, float *out, int nRows, int nCols,
        float *pxlList);

/* histogram */
void hist(const float *streamA, int *h, int nRows, int nCols, int nBpp);
void histEq(const float *streamA, float *out, const int *h,
            int nRows, int nCols, int nInpBpp, int nOutBpp,
            int *CDF, int *LUT);

/* image operations */
void imageStats(const float *image, float *imgMean, float *imgStd,
                float *imgMax, float *imgMin, int nRows, int nCols);
void rescaleImage(const float *image, float *out,
                  int nRows, int nCols, int nBpp);
void imageAdd(const float *imageA, const float *imageB,
              float *imageC, int nRows, int nCols);
void imageSub(const float *imageA, const float *imageB,
              float *imageC, int nRows, int nCols);
void imageAvg(const float *imageA, const float *imageB,
              float *imageC, int nRows, int nCols);
void imageMax(const float *imageA, const float *imageB,
              float *imageC, int nRows, int nCols);
void imageAbsMax(const float *imageA, const float *imageB,
                 float *imageC, int nRows, int nCols);
void imageMat2Gray(const float *srcImage, float *dstImage,
                   int nBpp, int nRows, int nCols);
void subsamp(const float *in, float *out, int nRows, int nCols,
             int rowSkip, int colSkip);
void interp2(const float *src, float *dst, int nSrcRows, int nSrcCols);
void conv2d(const float *in, float *out, int nRows, int nCols,
            const float *filter, int nFilterRows, int nFilterCols);
void interpolator(const float *in, float *out, int nRows, int nCols,
                  const float *filter, int nFilterRows, int nFilterCols,
                  float *tmpBuf);

/* DWT */
void dwt53(const float *in, float *out, int nRows, int nCols,
           float *tmp);
void dwt53_row_transpose(float *work, float *out, int nRows, int nCols);
void dwt53_inverse(const float *in, float *out, int nRows, int nCols,
                   float *tmp);
void dwt53_row_transpose_inverse(const float *in, float *out,
                                 int nRows, int nCols);

/* DWT fusion */
void dwtFusion(const float *streamA, const float *streamB, float *out,
               int nRows, int nCols, float *imgA, float *imgB, float *dwtTmp);

/* Laplacian fusion (Double-Density) */
void lapFusionDD(const float *streamA, const float *streamB,
                 float *out, int nRows, int nCols);

/* Ensembles */
int ensemble_1(const float *streamA, float *out,
               int nRows, int nCols, int nInpBpp, int nOutBpp);
int ensemble_2(const float *streamA, const float *streamB,
               float *out, int nRows, int nCols, int nInpBpp, int nOutBpp);
int ensemble_3(const float *streamA, const float *streamB,
               float *out, int nRows, int nCols, int nInpBpp, int nOutBpp);

#endif /* _NIGHTVISION_H */
