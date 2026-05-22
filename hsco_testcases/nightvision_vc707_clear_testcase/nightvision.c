#include "nightvision.h"

/* ========================================================================
 * Helper: aligned calloc replacement for FPGA baremetal
 * ======================================================================== */

static void *nv_calloc(int count, int elem_size)
{
	int total = count * elem_size;
	void *ptr = aligned_malloc(total);
	if (ptr) memset(ptr, 0, total);
	return ptr;
}

/* ========================================================================
 * Slow Median (3x3)
 * ======================================================================== */

/* Insertion sort for small arrays */
static void insertion_sort(float *arr, int n)
{
	for (int i = 1; i < n; i++) {
		float key = arr[i];
		int j = i - 1;
		while (j >= 0 && arr[j] > key) {
			arr[j + 1] = arr[j];
			j--;
		}
		arr[j + 1] = key;
	}
}

void slowMedian3x3(const float *src, float *dst, int nRows, int nCols,
                   float *pxlList)
{
	for (int r = 1; r < nRows-1; r++) {
		for (int c = 1; c < nCols-1; c++) {
			int k = 0;
			for (int i = -1; i <= 1; i++)
				for (int j = -1; j <= 1; j++)
					pxlList[k++] = src[(r + i) * nCols + c + j];
			insertion_sort(pxlList, MEDIAN_NELEM);
			dst[r * nCols + c] = pxlList[4];
		}
	}
}

/* ========================================================================
 * Noise Filter
 * ======================================================================== */

void nf(const float *streamA, float *out, int nRows, int nCols,
        float *pxlList)
{
	slowMedian3x3(streamA, out, nRows, nCols, pxlList);
}

/* ========================================================================
 * Histogram / Histogram Equalization
 * ======================================================================== */

void hist(const float *streamA, int *h, int nRows, int nCols, int nBpp)
{
	int nBins = 1 << nBpp;
	int nPxls = nRows * nCols;

	for (int i = 0; i < nBins; i++) h[i] = 0;

	for (int i = 0; i < nPxls; i++) {
		if (streamA[i] >= nBins) {
			h[nBins-1]++;
		} else {
			h[(int)streamA[i]]++;
		}
	}
}

void histEq(const float *streamA, float *out, const int *h,
            int nRows, int nCols, int nInpBpp, int nOutBpp,
            int *CDF, int *LUT)
{
	int nOutBins = (1 << nOutBpp);
	int nInpBins = (1 << nInpBpp);
	int CDFmin = INT_MAX;
	int sum = 0;
	int nPxls = nRows * nCols;

	for (int i = 0; i < nInpBins; i++) { sum += h[i]; CDF[i] = sum; }
	for (int i = 0; i < nInpBins; i++) {
		if (h[i] < CDFmin) CDFmin = h[i];
	}
	for (int i = 0; i < nInpBins; i++) {
		LUT[i] = ((CDF[i] - CDFmin) * (nOutBins - 1)) / (nPxls - CDFmin);
	}
	for (int i = 0; i < nPxls; i++) { out[i] = LUT[(int)streamA[i]]; }
}

/* ========================================================================
 * Image Operations
 * ======================================================================== */

void imageStats(const float *image, float *imgMean, float *imgStd,
                float *imgMax, float *imgMin, int nRows, int nCols)
{
	int nPxls = nRows * nCols;
	float diffSq = 0.0f;
	float mean;
	float sum = 0.0f;
	float mn = 1e30f;
	float mx = -1e30f;

	if (nPxls <= 1) { *imgMean = 0; *imgStd = 0; return; }

	for (int i = 0; i < nPxls; i++) sum += image[i];
	mean = sum / nPxls;

	for (int i = 0; i < nPxls; i++) {
		diffSq += ((image[i] - mean) * (image[i] - mean));
		if (image[i] < mn) mn = image[i];
		if (image[i] > mx) mx = image[i];
	}

	*imgMean = mean;
	*imgStd  = sqrtf_approx(diffSq / (nPxls - 1));
	*imgMin  = mn;
	*imgMax  = mx;
}

void rescaleImage(const float *image, float *out,
                  int nRows, int nCols, int nBpp)
{
	int nGrayLevels = (1 << nBpp) - 1;
	int nPxls = nRows * nCols;
	float fMin, fMax, fMean, fStd;
	imageStats(image, &fMean, &fStd, &fMax, &fMin, nRows, nCols);

	for (int i = 0; i < nPxls; i++)
		out[i] = ((image[i] - fMin) / (fMax - fMin)) * nGrayLevels;
}

void imageAdd(const float *imageA, const float *imageB, float *imageC, int nRows, int nCols)
{
	for (int i = 0; i < nRows * nCols; i++) imageC[i] = imageA[i] + imageB[i];
}

void imageSub(const float *imageA, const float *imageB, float *imageC, int nRows, int nCols)
{
	for (int i = 0; i < nRows * nCols; i++) imageC[i] = imageA[i] - imageB[i];
}

void imageAvg(const float *imageA, const float *imageB, float *imageC, int nRows, int nCols)
{
	for (int i = 0; i < nRows * nCols; i++) imageC[i] = (imageA[i] + imageB[i]) * 0.5f;
}

void imageMax(const float *imageA, const float *imageB, float *imageC, int nRows, int nCols)
{
	for (int i = 0; i < nRows * nCols; i++) {
		imageC[i] = (imageA[i] > imageB[i]) ? imageA[i] : imageB[i];
	}
}

void imageAbsMax(const float *imageA, const float *imageB, float *imageC, int nRows, int nCols)
{
	for (int i = 0; i < nRows * nCols; i++) {
		float absA = (imageA[i] < 0.0f) ? -imageA[i] : imageA[i];
		float absB = (imageB[i] < 0.0f) ? -imageB[i] : imageB[i];
		imageC[i] = (absA > absB) ? imageA[i] : imageB[i];
	}
}

void imageMat2Gray(const float *srcImage, float *dstImage,
                   int nBpp, int nRows, int nCols)
{
	int nGrayLevels = (1 << nBpp) - 1;
	int nPxls = nRows * nCols;
	float mean, stdv, mn, mx;

	imageStats(srcImage, &mean, &stdv, &mx, &mn, nRows, nCols);
	float low  = mean - 2 * stdv - 0.5f;
	if (low < 0) low = 0;
	float high = mean + 2 * stdv + 0.5f;
	float delta = (high - low);
	if (delta == 0) delta = 1;

	for (int i = 0; i < nPxls; i++) {
		float Pxl = srcImage[i];
		if (Pxl < low) Pxl = low;
		else if (Pxl > high) Pxl = high;
		dstImage[i] = ((Pxl - low) / delta) * nGrayLevels;
	}
}

void subsamp(const float *in, float *out, int nRows, int nCols,
             int rowSkip, int colSkip)
{
	int outIdx = 0;
	for (int row = 0; row < nRows; row += rowSkip) {
		for (int col = 0; col < nCols; col += colSkip) {
			out[outIdx++] = in[row * nCols + col];
		}
	}
}

void interp2(const float *src, float *dst, int nSrcRows, int nSrcCols)
{
	int dstCols = nSrcCols * 2;

	for (int row = 0; row < nSrcRows; row++) {
		for (int col = 0; col < nSrcCols; col++) {
			int idx = (row * 2) * dstCols + col * 2;
			dst[idx] = src[row * nSrcCols + col];
			dst[idx + 1] = (src[row * nSrcCols + col] + src[row * nSrcCols + col + 1]) / 2;
		}
		for (int col = 0; col < nSrcCols; col++) {
			int idx = (row * 2 + 1) * dstCols + col * 2;
			dst[idx] = (src[row * nSrcCols + col] + src[(row + 1) * nSrcCols + col]) / 2;
			dst[idx + 1] = (src[row * nSrcCols + col] + src[row * nSrcCols + col + 1] +
			                src[(row + 1) * nSrcCols + col] + src[(row + 1) * nSrcCols + col + 1]) / 4;
		}
	}
}

void conv2d(const float *in, float *out, int nRows, int nCols,
            const float *filter, int nFilterRows, int nFilterCols)
{
	int rowOffset = nFilterRows / 2;
	int colOffset = nFilterCols / 2;

	for (int row = 0; row < nRows; row++) {
		for (int col = 0; col < nCols; col++) {
			float sum = 0;
			for (int m = 0; m < nFilterRows; m++) {
				for (int n = 0; n < nFilterCols; n++) {
					int ir = row - rowOffset + m;
					int ic = col - colOffset + n;
					if (ir >= 0 && ir < nRows && ic >= 0 && ic < nCols)
						sum += in[ir * nCols + ic] * filter[m * nFilterCols + n];
				}
			}
			out[row * nCols + col] = sum;
		}
	}
}

void interpolator(const float *in, float *out, int nRows, int nCols,
                  const float *filter, int nFilterRows, int nFilterCols,
                  float *tmpBuf)
{
	int nRows2 = nRows * 2;
	int nCols2 = nCols * 2;

	for (int i = 0; i < nRows2 * nCols2; i++) tmpBuf[i] = 0.0f;

	for (int row = 0, row2 = 0; row < nRows; row++, row2 += 2)
		for (int col = 0, col2 = 0; col < nCols; col++, col2 += 2)
			tmpBuf[row2 * nCols2 + col2] = in[row * nCols + col];

	conv2d(tmpBuf, out, nRows2, nCols2, filter, nFilterRows, nFilterCols);
}

/* ========================================================================
 * DWT (5/3 Lifting)
 * ======================================================================== */

void dwt53_row_transpose(float *work, float *out, int nrows, int ncols)
{
	for (int i = 0; i < nrows; i++) {
		int cur;
		for (int j = 1; j < ncols - 1; j += 2) {
			cur = i * ncols + j;
			work[cur] -= 0.5f * (work[cur - 1] + work[cur + 1]);
		}
		cur = i * ncols + ncols - 1;
		work[cur] -= work[cur - 1];

		for (int j = 2; j < ncols; j += 2) {
			cur = i * ncols + j;
			work[cur] += 0.25f * (work[cur - 1] + work[cur + 1]);
		}
		cur = i * ncols;
		work[cur] += 0.5f * work[cur + 1];

		for (int j = 0; j < ncols / 2; j++) {
			out[j * nrows + i] = work[i * ncols + 2 * j];
			out[(j + ncols / 2) * nrows + i] = work[i * ncols + 2 * j + 1];
		}
	}
}

void dwt53(const float *in, float *out, int nrows, int ncols, float *tmp)
{
	int nP = nrows * ncols;
	for (int ii = 0; ii < nP; ii++) out[ii] = in[ii];
	dwt53_row_transpose(out, tmp, nrows, ncols);
	dwt53_row_transpose(tmp, out, ncols, nrows);
}

void dwt53_row_transpose_inverse(const float *in, float *out, int nrows, int ncols)
{
	for (int i = 0; i < nrows; i++) {
		int cur;
		for (int j = 0; j < ncols / 2; j++) {
			out[i * ncols + 2 * j] = in[j * nrows + i];
			out[i * ncols + 2 * j + 1] = in[(j + ncols / 2) * nrows + i];
		}
		for (int j = 2; j < ncols; j += 2) {
			cur = i * ncols + j;
			out[cur] -= 0.25f * (out[cur - 1] + out[cur + 1]);
		}
		cur = i * ncols;
		out[cur] -= 0.5f * out[cur + 1];
		for (int j = 1; j < ncols - 1; j += 2) {
			cur = i * ncols + j;
			out[cur] += 0.5f * (out[cur - 1] + out[cur + 1]);
		}
		cur = i * ncols + ncols - 1;
		out[cur] += out[cur - 1];
	}
}

void dwt53_inverse(const float *in, float *out, int nrows, int ncols, float *tmp)
{
	dwt53_row_transpose_inverse(in, tmp, ncols, nrows);
	dwt53_row_transpose_inverse(tmp, out, nrows, ncols);
}

/* ========================================================================
 * DWT Fusion
 * ======================================================================== */

void dwtFusion(const float *streamA, const float *streamB, float *out,
               int nRows, int nCols, float *imgA, float *imgB, float *dwtTmp)
{
	dwt53(streamA, imgA, nRows, nCols, dwtTmp);
	dwt53(streamB, imgB, nRows, nCols, dwtTmp);
	imageAbsMax(imgA, imgB, out, nRows, nCols);
	dwt53_inverse(out, out, nRows, nCols, dwtTmp);
}

/* ========================================================================
 * Laplacian Pyramid Fusion (Double-Density)
 * ======================================================================== */

void lapFusionDD(const float *streamA, const float *streamB, float *out, int nRows, int nCols)
{
	int nSubSampRows, nSubSampCols, nSubSampRows2, nSubSampCols2;
	int nFRF = 5;
	int nFCF = 5;
	int nFRFD = 9;
	int nFCFD = 9;

	float F[] = {
		1,4,6,4,1, 4,16,24,16,4, 6,24,36,24,6, 4,16,24,16,4, 1,4,6,4,1
	};
	for (int i = 0; i < nFRF * nFCF; i++) F[i] /= 256.0f;

	float FD[] = {
		1,3,4,5,6,5,4,3,1,   3,9,12,15,18,15,12,9,3,
		4,12,16,20,24,20,16,12,4,   5,15,20,25,30,25,20,15,5,
		6,18,24,30,36,30,24,18,6,   5,15,20,25,30,25,20,15,5,
		4,12,16,20,24,20,16,12,4,   3,9,12,15,18,15,12,9,3,
		1,3,4,5,6,5,4,3,1
	};
	for (int i = 0; i < nFRFD * nFCFD; i++) FD[i] /= 1024.0f;

	float *LFA1  = (float *)nv_calloc(nRows*nCols, sizeof(float));
	float *LFA2D = (float *)nv_calloc(nRows*nCols, sizeof(float));
	float *LFA3D = (float *)nv_calloc(nRows*nCols, sizeof(float));
	float *LFA4D = (float *)nv_calloc(nRows*nCols, sizeof(float));
	float *HFA1  = (float *)nv_calloc(nRows*nCols, sizeof(float));
	float *HFA2D = (float *)nv_calloc(nRows*nCols, sizeof(float));
	float *HFA3D = (float *)nv_calloc(nRows*nCols, sizeof(float));
	float *HFA4D = (float *)nv_calloc(nRows*nCols, sizeof(float));
	float *LFB1  = (float *)nv_calloc(nRows*nCols, sizeof(float));
	float *LFB2D = (float *)nv_calloc(nRows*nCols, sizeof(float));
	float *LFB3D = (float *)nv_calloc(nRows*nCols, sizeof(float));
	float *LFB4D = (float *)nv_calloc(nRows*nCols, sizeof(float));
	float *HFB1  = (float *)nv_calloc(nRows*nCols, sizeof(float));
	float *HFB2D = (float *)nv_calloc(nRows*nCols, sizeof(float));
	float *HFB3D = (float *)nv_calloc(nRows*nCols, sizeof(float));
	float *HFB4D = (float *)nv_calloc(nRows*nCols, sizeof(float));
	float *tmp1      = (float *)nv_calloc(nRows*nCols, sizeof(float));
	float *tmp2      = (float *)nv_calloc(nRows*nCols, sizeof(float));
	float *tmp3      = (float *)nv_calloc(nRows*nCols, sizeof(float));
	float *interpTmp = (float *)nv_calloc(nRows*nCols, sizeof(float));

	if (!(LFA1 && LFA2D && LFA3D && LFA4D && HFA1 && HFA2D && HFA3D && HFA4D &&
	      LFB1 && LFB2D && LFB3D && LFB4D && HFB1 && HFB2D && HFB3D && HFB4D &&
	      tmp1 && tmp2 && tmp3 && interpTmp)) {
		if (LFA1) aligned_free(LFA1); if (LFA2D) aligned_free(LFA2D); if (LFA3D) aligned_free(LFA3D); if (LFA4D) aligned_free(LFA4D);
		if (HFA1) aligned_free(HFA1); if (HFA2D) aligned_free(HFA2D); if (HFA3D) aligned_free(HFA3D); if (HFA4D) aligned_free(HFA4D);
		if (LFB1) aligned_free(LFB1); if (LFB2D) aligned_free(LFB2D); if (LFB3D) aligned_free(LFB3D); if (LFB4D) aligned_free(LFB4D);
		if (HFB1) aligned_free(HFB1); if (HFB2D) aligned_free(HFB2D); if (HFB3D) aligned_free(HFB3D); if (HFB4D) aligned_free(HFB4D);
		if (tmp1) aligned_free(tmp1); if (tmp2) aligned_free(tmp2); if (tmp3) aligned_free(tmp3);
		if (interpTmp) aligned_free(interpTmp);
		return;
	}

	nSubSampRows = (nRows + 1) / 2;
	nSubSampCols = (nCols + 1) / 2;
	nSubSampRows2 = (nSubSampRows + 1) / 2;
	nSubSampCols2 = (nSubSampCols + 1) / 2;

	/* Stream A decomposition */
	conv2d(streamA, LFA1, nRows, nCols, F, nFRF, nFCF);
	imageSub(streamA, LFA1, HFA1, nRows, nCols);
	conv2d(LFA1, LFA2D, nRows, nCols, FD, nFRFD, nFCFD);
	imageSub(LFA1, LFA2D, HFA2D, nRows, nCols);
	subsamp(LFA2D, tmp1, nRows, nCols, 2, 2);
	conv2d(tmp1, LFA3D, nSubSampRows, nSubSampCols, FD, nFRFD, nFCFD);
	imageSub(tmp1, LFA3D, HFA3D, nSubSampRows, nSubSampCols);
	subsamp(LFA3D, tmp2, nSubSampRows, nSubSampCols, 2, 2);
	conv2d(tmp2, LFA4D, nSubSampRows2, nSubSampCols2, FD, nFRFD, nFCFD);
	imageSub(tmp2, LFA4D, HFA4D, nSubSampRows2, nSubSampCols2);

	/* Stream B decomposition */
	conv2d(streamB, LFB1, nRows, nCols, F, nFRF, nFCF);
	imageSub(streamB, LFB1, HFB1, nRows, nCols);
	conv2d(LFB1, LFB2D, nRows, nCols, FD, nFRFD, nFCFD);
	imageSub(LFB1, LFB2D, HFB2D, nRows, nCols);
	subsamp(LFB2D, tmp1, nRows, nCols, 2, 2);
	conv2d(tmp1, LFB3D, nSubSampRows, nSubSampCols, FD, nFRFD, nFCFD);
	imageSub(tmp1, LFB3D, HFB3D, nSubSampRows, nSubSampCols);
	subsamp(LFB3D, tmp2, nSubSampRows, nSubSampCols, 2, 2);
	conv2d(tmp2, LFB4D, nSubSampRows2, nSubSampCols2, FD, nFRFD, nFCFD);
	imageSub(tmp2, LFB4D, HFB4D, nSubSampRows2, nSubSampCols2);

	/* Fuse */
	imageAvg(LFA4D, LFB4D, tmp1, nSubSampRows2, nSubSampCols2);
	imageAbsMax(HFA4D, HFB4D, tmp2, nSubSampRows2, nSubSampCols2);
	imageAdd(tmp1, tmp2, tmp3, nSubSampRows2, nSubSampCols2);

	interpolator(tmp3, tmp1, nSubSampRows2, nSubSampCols2, FD, nFRFD, nFCFD, interpTmp);
	imageAbsMax(HFA3D, HFB3D, tmp2, nSubSampRows, nSubSampCols);
	conv2d(tmp2, tmp3, nSubSampRows, nSubSampCols, FD, nFRFD, nFCFD);
	imageAdd(tmp1, tmp3, tmp2, nSubSampRows, nSubSampCols);

	interpolator(tmp2, tmp1, nSubSampRows, nSubSampCols, FD, nFRFD, nFCFD, interpTmp);
	imageAbsMax(HFA2D, HFB2D, tmp2, nRows, nCols);
	conv2d(tmp2, tmp3, nRows, nCols, FD, nFRFD, nFCFD);
	imageAdd(tmp1, tmp3, tmp2, nRows, nCols);

	imageAbsMax(HFA1, HFB1, tmp1, nRows, nCols);
	imageAdd(tmp1, tmp2, out, nRows, nCols);

	aligned_free(LFA1); aligned_free(LFA2D); aligned_free(LFA3D); aligned_free(LFA4D);
	aligned_free(HFA1); aligned_free(HFA2D); aligned_free(HFA3D); aligned_free(HFA4D);
	aligned_free(LFB1); aligned_free(LFB2D); aligned_free(LFB3D); aligned_free(LFB4D);
	aligned_free(HFB1); aligned_free(HFB2D); aligned_free(HFB3D); aligned_free(HFB4D);
	aligned_free(tmp1); aligned_free(tmp2); aligned_free(tmp3); aligned_free(interpTmp);
}

/* ========================================================================
 * Ensembles
 * ======================================================================== */

int ensemble_1(const float *streamA, float *out, int nRows, int nCols, int nInpBpp, int nOutBpp)
{
	int nHistBins = 1 << nInpBpp;
	int *h   = (int *)nv_calloc(nHistBins, sizeof(int));
	int *CDF = (int *)nv_calloc(nHistBins, sizeof(int));
	int *LUT = (int *)nv_calloc(nHistBins, sizeof(int));
	float *wrkBuf1 = (float *)nv_calloc(nRows * nCols, sizeof(float));
	float *wrkBuf2 = (float *)nv_calloc(nRows * nCols, sizeof(float));
	float *pxlList = (float *)nv_calloc(MEDIAN_NELEM, sizeof(float));
	float *dwtTmp  = (float *)nv_calloc(nRows * nCols, sizeof(float));
	if (!(h && CDF && LUT && wrkBuf1 && wrkBuf2 && pxlList && dwtTmp)) {
		if (h) aligned_free(h); if (CDF) aligned_free(CDF); if (LUT) aligned_free(LUT);
		if (wrkBuf1) aligned_free(wrkBuf1); if (wrkBuf2) aligned_free(wrkBuf2);
		if (pxlList) aligned_free(pxlList); if (dwtTmp) aligned_free(dwtTmp); return -1;
	}

	for (int j = 0; j < nRows * nCols; j++) wrkBuf1[j] = streamA[j];
	for (int j = 0; j < nRows * nCols; j++) wrkBuf2[j] = 0.0f;

	nf(wrkBuf1, wrkBuf2, nRows, nCols, pxlList);
	hist(wrkBuf2, h, nRows, nCols, nInpBpp);
	histEq(wrkBuf2, out, h, nRows, nCols, nInpBpp, nOutBpp, CDF, LUT);
	dwt53(out, out, nRows, nCols, dwtTmp);

	for (int j = 0; j < nRows * nCols; j++) out[j] = wrkBuf2[j];
	aligned_free(wrkBuf2); aligned_free(wrkBuf1); aligned_free(h);
	aligned_free(CDF); aligned_free(LUT);
	aligned_free(pxlList); aligned_free(dwtTmp);
	return 0;
}

int ensemble_2(const float *streamA, const float *streamB, float *out,
               int nRows, int nCols, int nInpBpp, int nOutBpp)
{
	int nPxls = nRows * nCols;
	int nHistBins = 1 << nInpBpp;
	int *hA  = (int *)nv_calloc(nHistBins, sizeof(int));
	int *hB  = (int *)nv_calloc(nHistBins, sizeof(int));
	int *CDF = (int *)nv_calloc(nHistBins, sizeof(int));
	int *LUT = (int *)nv_calloc(nHistBins, sizeof(int));
	float *wA = (float *)nv_calloc(nPxls, sizeof(float));
	float *wB = (float *)nv_calloc(nPxls, sizeof(float));
	float *pxlList = (float *)nv_calloc(MEDIAN_NELEM, sizeof(float));
	float *dwtA   = (float *)nv_calloc(nPxls, sizeof(float));
	float *dwtB   = (float *)nv_calloc(nPxls, sizeof(float));
	float *dwtTmp = (float *)nv_calloc(nPxls, sizeof(float));
	if (!(hA && hB && CDF && LUT && wA && wB && pxlList && dwtA && dwtB && dwtTmp)) {
		if (hA) aligned_free(hA); if (hB) aligned_free(hB);
		if (CDF) aligned_free(CDF); if (LUT) aligned_free(LUT);
		if (wA) aligned_free(wA); if (wB) aligned_free(wB);
		if (pxlList) aligned_free(pxlList);
		if (dwtA) aligned_free(dwtA); if (dwtB) aligned_free(dwtB);
		if (dwtTmp) aligned_free(dwtTmp); return -1;
	}

	nf(streamA, wA, nRows, nCols, pxlList);
	hist(wA, hA, nRows, nCols, nInpBpp);
	histEq(wA, wA, hA, nRows, nCols, nInpBpp, nOutBpp, CDF, LUT);
	nf(streamB, wB, nRows, nCols, pxlList);
	hist(wB, hB, nRows, nCols, nInpBpp);
	histEq(wB, wB, hB, nRows, nCols, nInpBpp, nOutBpp, CDF, LUT);
	dwtFusion(wA, wB, out, nRows, nCols, dwtA, dwtB, dwtTmp);

	aligned_free(hA); aligned_free(hB); aligned_free(CDF); aligned_free(LUT);
	aligned_free(wA); aligned_free(wB);
	aligned_free(pxlList); aligned_free(dwtA); aligned_free(dwtB); aligned_free(dwtTmp);
	return 0;
}

int ensemble_3(const float *streamA, const float *streamB, float *out,
               int nRows, int nCols, int nInpBpp, int nOutBpp)
{
	int nHistBins = 1 << nInpBpp;
	int *hA  = (int *)nv_calloc(nHistBins, sizeof(int));
	int *hB  = (int *)nv_calloc(nHistBins, sizeof(int));
	int *CDF = (int *)nv_calloc(nHistBins, sizeof(int));
	int *LUT = (int *)nv_calloc(nHistBins, sizeof(int));
	float *wA = (float *)nv_calloc(nRows * nCols, sizeof(float));
	float *wB = (float *)nv_calloc(nRows * nCols, sizeof(float));
	float *pxlList = (float *)nv_calloc(MEDIAN_NELEM, sizeof(float));
	if (!(hA && hB && CDF && LUT && wA && wB && pxlList)) {
		if (hA) aligned_free(hA); if (hB) aligned_free(hB);
		if (CDF) aligned_free(CDF); if (LUT) aligned_free(LUT);
		if (wA) aligned_free(wA); if (wB) aligned_free(wB);
		if (pxlList) aligned_free(pxlList); return -1;
	}

	nf(streamA, wA, nRows, nCols, pxlList);
	hist(wA, hA, nRows, nCols, nInpBpp);
	histEq(wA, wA, hA, nRows, nCols, nInpBpp, nOutBpp, CDF, LUT);
	nf(streamB, wB, nRows, nCols, pxlList);
	hist(wB, hB, nRows, nCols, nInpBpp);
	histEq(wB, wB, hB, nRows, nCols, nInpBpp, nOutBpp, CDF, LUT);

	lapFusionDD(wA, wB, out, nRows, nCols);

	aligned_free(hA); aligned_free(hB); aligned_free(CDF); aligned_free(LUT);
	aligned_free(wA); aligned_free(wB); aligned_free(pxlList);
	return 0;
}
