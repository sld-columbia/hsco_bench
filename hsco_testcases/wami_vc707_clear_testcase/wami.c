/*
 * WAMI processing kernels.
 * Derived from the PERFECT Benchmark Suite (hpc.pnnl.gov/projects/PERFECT/)
 * Copyright (c) 2014, Battelle Memorial Institute / Georgia Tech Research Corporation.
 * Interpolation code Copyright (c) 2009, Dirk-Jan Kroon.
 * See original sources for full license terms (BSD-style).
 *
 * FPGA Baremetal Adaptation:
 *   - malloc/calloc/free → aligned_malloc/aligned_free/wami_calloc
 *   - math.h functions → approx versions (fabsf_approx, expf_approx, sqrtf_approx, floorf_approx)
 *   - assert() → removed (unavailable on baremetal)
 *   - fprintf(stderr,...) → printf(...)
 */

#include "wami.h"

/* ========================================================================
 * Forward declarations for file-local helpers
 * ======================================================================== */

static void  warp_image(const float *Iin, int nCols, int nRows, const float *W_xp, float *Iout);
static void  steepest_descent(const float *gradX_warped, const float *gradY_warped,
                               int nCols, int nRows, float *I_steepest);
static void  hessian(const float *I_steepest, int nCols, int nRows, int np, float *H);
static void  gradientXY(const float *Iin, int nCols, int nRows, float *out_Xgrad, float *out_Ygrad);
static void  sd_update(const float *VI_dW_dp, const float *error_img, int N_p, int nRows, int nCols, float *sd_delta_p);
static void  reshape(const float *in, int nRows, int nCols, int newRows, int newCols, float *out);
static void  subtract(const float *inA, const float *inB, int nRows, int nCols, float *out);
static void  add(const float *inA, const float *inB, int nRows, int nCols, float *out);
static void  mult(const float *inA, const float *inB, int nRows, int nCols, int nCommon, float *out);
static void  swap_row(float *data, int nRows, int nCols, int r1, int r2);
static void  scale_row(float *data, int nRows, int nCols, int r, float scale);
static void  scale_and_add_row(float *data, int nRows, int nCols, int r1, int r2, float scale);
static void  invert_gauss_jordan(const float *data, float *workspace, int nRows, int nCols, float *inverse);


/* ========================================================================
 * Debayer
 * ======================================================================== */

static float compute_and_clamp_pixel_f(float pos, float neg)
{
    float result = (pos - neg) * 0.125f;
    return result > 0.0f ? result : 0.0f;
}

static float compute_and_clamp_pixel_fractional_neg_f(float pos, float neg)
{
    float result = (pos * 2.0f - neg) * 0.0625f;
    return result > 0.0f ? result : 0.0f;
}

static float interp_G_at_RRR_or_G_at_BBB(
    const float *bayer, int row, int col, int nrows, int ncols)
{
    const float pos =
         2.0f * bayer[ncols*(row-1) + col] +
         2.0f * bayer[ncols*row + (col-1)] +
         4.0f * bayer[ncols*row + col] +
         2.0f * bayer[ncols*row + (col+1)] +
         2.0f * bayer[ncols*(row+1) + col];
    const float neg =
             bayer[ncols*row + (col+2)] +
             bayer[ncols*(row-2) + col] +
             bayer[ncols*row + (col-2)] +
             bayer[ncols*(row+2) + col];
    return compute_and_clamp_pixel_f(pos, neg);
}

static float interp_R_at_GRB_or_B_at_GBR(
    const float *bayer, int row, int col, int nrows, int ncols)
{
    const float pos =
        (bayer[ncols*(row-2) + col] + bayer[ncols*(row+2) + col]) * 0.5f +
        4.0f * bayer[ncols*row + (col-1)] +
        5.0f * bayer[ncols*row + col] +
        4.0f * bayer[ncols*row + (col+1)];
    const float neg =
            bayer[ncols*(row-1) + (col-1)] +
            bayer[ncols*(row-1) + (col+1)] +
            bayer[ncols*row + (col-2)] +
            bayer[ncols*row + (col+2)] +
            bayer[ncols*(row+1) + (col-1)] +
            bayer[ncols*(row+1) + (col+1)];
    return compute_and_clamp_pixel_f(pos, neg);
}

static float interp_R_at_GBR_or_B_at_GRB(
    const float *bayer, int row, int col, int nrows, int ncols)
{
    const float pos =
        4.0f * bayer[ncols*(row-1) + col] +
        (bayer[ncols*row + (col-2)] + bayer[ncols*row + (col+2)]) * 0.5f +
        5.0f * bayer[ncols*row + col] +
        4.0f * bayer[ncols*(row+1) + col];
    const float neg =
            bayer[ncols*(row-2) + col] +
            bayer[ncols*(row-1) + (col-1)] +
            bayer[ncols*(row-1) + (col+1)] +
            bayer[ncols*(row+1) + (col-1)] +
            bayer[ncols*(row+1) + (col+1)] +
            bayer[ncols*(row+2) + col];
    return compute_and_clamp_pixel_f(pos, neg);
}

static float interp_R_at_BBB_or_B_at_RRR(
    const float *bayer, int row, int col, int nrows, int ncols)
{
    const float pos =
        2.0f * bayer[ncols*(row-1) + (col-1)] +
        2.0f * bayer[ncols*(row-1) + (col+1)] +
        6.0f * bayer[ncols*row + col] +
        2.0f * bayer[ncols*(row+1) + (col-1)] +
        2.0f * bayer[ncols*(row+1) + (col+1)];
    const float neg =
        3.0f * bayer[ncols*(row-2) + col] +
        3.0f * bayer[ncols*row + (col-2)] +
        3.0f * bayer[ncols*row + (col+2)] +
        3.0f * bayer[ncols*(row+2) + col];
    return compute_and_clamp_pixel_fractional_neg_f(pos, neg);
}

void wami_debayer(
    const float *bayer,
    rgb_pixel *debayered,
    int nrows,
    int ncols)
{
    /*
     * Demosaic the following Bayer pattern:
     * R G ...
     * G B ...
     * ... ...
     */

    for (int row = 2; row < nrows - 2; row += 2)
    {
        for (int col = 2; col < ncols - 2; col += 2)
        {
            debayered[(ncols-(2*2))*(row-2) + (col-2)].r = bayer[ncols*(row) + (col)];
        }
    }

    for (int row = 2; row < nrows - 2; row += 2)
    {
        for (int col = 2+1; col < ncols - 2; col += 2)
        {
            debayered[(ncols-(2*2))*(row-2) + (col-2)].g = bayer[ncols*(row) + (col)];
        }
    }

    for (int row = 2+1; row < nrows - 2; row += 2)
    {
        for (int col = 2; col < ncols - 2; col += 2)
        {
            debayered[(ncols-(2*2))*(row-2) + (col-2)].g = bayer[ncols*(row) + (col)];
        }
    }

    for (int row = 2+1; row < nrows - 2; row += 2)
    {
        for (int col = 2+1; col < ncols - 2; col += 2)
        {
            debayered[(ncols-(2*2))*(row-2) + (col-2)].b = bayer[ncols*(row) + (col)];
        }
    }

    for (int row = 2; row < nrows - 2; row += 2)
    {
        for (int col = 2; col < ncols - 2; col += 2)
        {
            debayered[(ncols-(2*2))*(row-2) + (col-2)].g = interp_G_at_RRR_or_G_at_BBB(
                bayer, row, col, nrows, ncols);
        }
    }

    for (int row = 2+1; row < nrows - 2; row += 2)
    {
        for (int col = 2+1; col < ncols - 2; col += 2)
        {
            debayered[(ncols-(2*2))*(row-2) + (col-2)].g = interp_G_at_RRR_or_G_at_BBB(
                bayer, row, col, nrows, ncols);
        }
    }

    for (int row = 2; row < nrows - 2; row += 2)
    {
        for (int col = 2+1; col < ncols - 2; col += 2)
        {
            debayered[(ncols-(2*2))*(row-2) + (col-2)].r = interp_R_at_GRB_or_B_at_GBR(
                bayer, row, col, nrows, ncols);
        }
    }

    for (int row = 2+1; row < nrows - 2; row += 2)
    {
        for (int col = 2; col < ncols - 2; col += 2)
        {
            debayered[(ncols-(2*2))*(row-2) + (col-2)].b = interp_R_at_GRB_or_B_at_GBR(
                bayer, row, col, nrows, ncols);
        }
    }

    for (int row = 2+1; row < nrows - 2; row += 2)
    {
        for (int col = 2; col < ncols - 2; col += 2)
        {
            debayered[(ncols-(2*2))*(row-2) + (col-2)].r = interp_R_at_GBR_or_B_at_GRB(
                bayer, row, col, nrows, ncols);
        }
    }

    for (int row = 2; row < nrows - 2; row += 2)
    {
        for (int col = 2+1; col < ncols - 2; col += 2)
        {
            debayered[(ncols-(2*2))*(row-2) + (col-2)].b = interp_R_at_GBR_or_B_at_GRB(
                bayer, row, col, nrows, ncols);
        }
    }

    for (int row = 2+1; row < nrows - 2; row += 2)
    {
        for (int col = 2+1; col < ncols - 2; col += 2)
        {
            debayered[(ncols-(2*2))*(row-2) + (col-2)].r = interp_R_at_BBB_or_B_at_RRR(
                bayer, row, col, nrows, ncols);
        }
    }

    for (int row = 2; row < nrows - 2; row += 2)
    {
        for (int col = 2; col < ncols - 2; col += 2)
        {
            debayered[(ncols-(2*2))*(row-2) + (col-2)].b = interp_R_at_BBB_or_B_at_RRR(
                bayer, row, col, nrows, ncols);
        }
    }
}


/* ========================================================================
 * Gaussian Mixture Model (GMM)
 * ======================================================================== */

/* GMM constants — inlined so HLS kernel body is self-contained */
#define GMM_STDEV_THRESH         2.5f
#define GMM_INIT_STDEV          80.0f
#define GMM_ALPHA                0.1f
#define GMM_INIT_WEIGHT          0.01f
#define GMM_BACKGROUND_THRESH    0.9f
#define ONE_OVER_SQRT_TWO_PI     0.39894228040143270f

void
wami_gmm(
  int     nRows,
  int     nCols,
  int     nModels,
  const float * frame,
  const float * mu_in,
  const float * sigma_in,
  const float * weight_in,
  float * mu_dst,
  float * sigma_dst,
  float * weight_dst,
  float * foreground)
{
    int model_sz = nRows * nCols * nModels;
    int num_pixels = nRows * nCols;

    for (int i = 0; i < model_sz; i++) {
        mu_dst[i] = mu_in[i];
        sigma_dst[i] = sigma_in[i];
        weight_dst[i] = weight_in[i];
    }
    for (int i = 0; i < num_pixels; i++)
        foreground[i] = 0.0f;

    for (int row = 0; row < nRows; ++row)
    {
        for (int col = 0; col < nCols; ++col)
        {
            const float pixel = frame[row * nCols + col];
            const int gmm_base = row * nCols * nModels + col * nModels;
            int match = -1;
            float sum = 0.0f;
            float norm = 0.0f;
            int sorted_position = 0;

            for (int k = 0; k < nModels; ++k)
            {
                const int idx = gmm_base + k;
                const float diff = pixel - mu_dst[idx];
                const float abs_diff = fabsf_approx(diff);
                const float z = abs_diff / sigma_dst[idx];
                if (z < GMM_STDEV_THRESH)
                {
                    match = k;
                    break;
                }
            }

            for (int k = 0; k < nModels; ++k)
            {
                if (k == match)
                {
                    weight_dst[gmm_base + match] += GMM_ALPHA *
                        (1.0f - weight_dst[gmm_base + match]);
                }
                else
                {
                    weight_dst[gmm_base + k] *= (1.0f - GMM_ALPHA);
                }
            }

            if (match < 0)
            {
                mu_dst[gmm_base + nModels - 1] = pixel;
                sigma_dst[gmm_base + nModels - 1] = GMM_INIT_STDEV;
                weight_dst[gmm_base + nModels - 1] = GMM_INIT_WEIGHT;
            }

            for (int k = 0; k < nModels; ++k)
            {
                sum += weight_dst[gmm_base + k];
            }

            if (sum == 0.0f) {
                sum = 1.0f;
            }

            norm = 1.0f / sum;
            for (int k = 0; k < nModels; ++k)
            {
                weight_dst[gmm_base + k] *= norm;
            }

            if (match >= 0)
            {
                const float mu_k = mu_dst[gmm_base + match];
                const float sigma_k = sigma_dst[gmm_base + match];
                const float sigma_k_inv = 1.0f / sigma_k;
                const float rho = GMM_ALPHA * (ONE_OVER_SQRT_TWO_PI * sigma_k_inv) *
                    expf_approx( -1.0f * (pixel-mu_k)*(pixel-mu_k) / (2.0f * sigma_k * sigma_k) );
                mu_dst[gmm_base + match] = (1.0f - rho) * mu_k + rho * pixel;
                sigma_dst[gmm_base + match] = sqrtf_approx(
                    (1.0f - rho) * sigma_k * sigma_k +
                    rho * (pixel-mu_dst[gmm_base + match]) *
		     (pixel-mu_dst[gmm_base + match]));
                if (sigma_dst[gmm_base + match] <= 0) {
                    sigma_dst[gmm_base + match] = 1e-6f;
                }
            }

            sorted_position = 0;
            if (match != 0)
            {
                const int sort_from = (match >= 0) ? match : nModels - 1;
                const float new_significance = weight_dst[gmm_base + sort_from] / sigma_dst[gmm_base + sort_from];
                float other_significance;
                float new_mu;
                float new_sigma;
                float new_weight;
                int k;
                for (k = sort_from - 1; k >= 0; --k)
                {
                    other_significance = weight_dst[gmm_base + k] / sigma_dst[gmm_base + k];
                    if (new_significance <= other_significance)
                    {
                        break;
                    }
                }
                if (k == 0)
                {
                    if (other_significance >= new_significance)
                    {
                        sorted_position = 1;
                    }
                    else
                    {
                        sorted_position = 0;
                    }
                }
                else
                {
                    sorted_position = k + 1;
                }

                new_mu = mu_dst[gmm_base + sort_from];
                new_sigma = sigma_dst[gmm_base + sort_from];
                new_weight = weight_dst[gmm_base + sort_from];
                for (k = sort_from; k > sorted_position; --k)
                {
                    mu_dst[gmm_base + k] = mu_dst[gmm_base + k - 1];
                    sigma_dst[gmm_base + k] = sigma_dst[gmm_base + k - 1];
                    weight_dst[gmm_base + k] = weight_dst[gmm_base + k - 1];
                }
                mu_dst[gmm_base + sorted_position] = new_mu;
                sigma_dst[gmm_base + sorted_position] = new_sigma;
                weight_dst[gmm_base + sorted_position] = new_weight;
            }

            {
                float cumsum = weight_dst[gmm_base + 0];
                int B = 0;
                while (B < nModels - 1 && cumsum <= GMM_BACKGROUND_THRESH)
                {
                    cumsum += weight_dst[gmm_base + ++B];
                }
                foreground[row * nCols + col] = (float)(sorted_position > B);
            }
        }
    }
}


/* ========================================================================
 * RGB to Grayscale (ITU-R BT.709)
 * ======================================================================== */

void
rgb_to_grayscale(
    const rgb_pixel * color,
    float * out,
    int nrows,
    int ncols)
{
  for (int y = 0; y < nrows; y++) {
    for (int x = 0; x < ncols; x++) {
      out[y * ncols + x] = color[y * ncols + x].r * 0.2126f +
                           color[y * ncols + x].g * 0.7152f +
                           color[y * ncols + x].b * 0.0722f;
    }
  }
}


/* ========================================================================
 * Image Gradient (central difference)
 * ======================================================================== */

static void
gradientXY(const float *Iin, int nCols, int nRows, float *out_Xgrad, float *out_Ygrad)
{
  for (int y = 1; y < nRows - 1; y++) {
    for (int x = 1; x < nCols - 1; x++) {
      out_Xgrad[y * nCols + x] = (Iin[y * nCols + (x+1)] - Iin[y * nCols + (x-1)]) / 2.0;
      out_Ygrad[y * nCols + x] = (-Iin[(y-1) * nCols + x] + Iin[(y+1) * nCols + x]) / 2.0;
    }
  }

  for (int x = 1; x < nCols - 1; x++) {
    out_Xgrad[x]                   = (Iin[(x+1)] - Iin[(x-1)]) / 2.0;
    out_Xgrad[(nRows-1)*nCols + x] = (Iin[(nRows-1)*nCols + (x+1)] - Iin[(nRows-1)*nCols + (x-1)]) / 2.0;
    out_Ygrad[x]                   = -Iin[x] + Iin[nCols + x];
    out_Ygrad[(nRows-1)*nCols + x] = Iin[(nRows-1) * nCols + x] - Iin[(nRows-2) * nCols + x];
  }
  for (int y = 1; y < nRows - 1; y++) {
    out_Xgrad[y * nCols]            = -Iin[y * nCols] + Iin[y * nCols + 1];
    out_Xgrad[y * nCols + (nCols-1)] = Iin[y * nCols + (nCols-1)] - Iin[y * nCols + (nCols-2)];
    out_Ygrad[y * nCols]            = (Iin[(y+1) * nCols] - Iin[(y-1) * nCols]) / 2.0;
    out_Ygrad[y * nCols + (nCols-1)] = (Iin[(y+1) * nCols + (nCols-1)] - Iin[(y-1) * nCols + (nCols-1)]) / 2.0;
  }

  out_Ygrad[0]                           = -Iin[0] + Iin[nCols];
  out_Ygrad[(nRows-1)*nCols]             = Iin[(nRows-1) * nCols] - Iin[(nRows-2) * nCols];
  out_Ygrad[(nCols-1)]                   = -Iin[(nCols-1)] + Iin[nCols + (nCols-1)];
  out_Ygrad[(nRows-1)*nCols + (nCols-1)] = Iin[(nRows-1) * nCols + (nCols-1)] - Iin[(nRows-2) * nCols + (nCols-1)];
  out_Xgrad[0]                           = -Iin[0] + Iin[1];
  out_Xgrad[(nCols-1)]                   = Iin[(nCols-1)] - Iin[(nCols-2)];
  out_Xgrad[(nRows-1) * nCols]           = -Iin[(nRows-1) * nCols] + Iin[(nRows-1) * nCols + 1];
  out_Xgrad[(nRows-1) * nCols + (nCols-1)] = Iin[(nRows-1) * nCols + (nCols-1)] - Iin[(nRows-1) * nCols + (nCols-2)];
}


/* ========================================================================
 * Matrix Operations
 * ======================================================================== */

static void
reshape(const float * in, int nRows, int nCols, int newRows, int newCols, float * out)
{
  for (int cur_index = 0; cur_index < newRows * newCols; cur_index++) {
    out[(cur_index % newRows) * newCols + (cur_index / newRows)] =  in[(cur_index % nRows) * nCols + (cur_index / nRows)];
  }
}

static void
subtract(const float * inA, const float * inB, int nRows, int nCols, float * out)
{
  for (int y = 0; y < nRows; y++) {
    for (int x = 0; x < nCols; x++) {
      out[y * nCols + x] =  inA[y * nCols + x] - inB[y * nCols + x];
    }
  }
}

static void
add(const float * inA, const float * inB, int nRows, int nCols, float * out)
{
  for (int y = 0; y < nRows; y++) {
    for (int x = 0; x < nCols; x++) {
      out[y * nCols + x] =  inA[y * nCols + x] + inB[y * nCols + x];
    }
  }
}

static void
mult(const float * inA, const float * inB, int nRows, int nCols, int nCommon, float * out)
{
  for (int y = 0; y < nRows; y++) {
    for (int x = 0; x < nCols; x++) {
      out[y * nCols + x] = 0;
    }
  }

  for (int y = 0; y < nRows; y++) {
    for (int x = 0; x < nCols; x++) {
      for (int i = 0; i < nCommon; i++) {
        out[y * nCols + x] += inA[y * nCommon + i] * inB[i * nCols + x];
      }
    }
  }
}

static void
swap_row(float * data, int nRows, int nCols, int r1, int r2)
{
  for (int x = 0; x < nCols; x++) {
    float tmp = data[r1 * nCols + x];
    data[r1 * nCols + x] = data[r2 * nCols + x];
    data[r2 * nCols + x] = tmp;
  }
}

static void
scale_row(float * data, int nRows, int nCols, int r, float scale)
{
  for (int x = 0; x < nCols; x++) {
    data[r * nCols + x] *= scale;
  }
}

static void
scale_and_add_row(float * data, int nRows, int nCols, int r1, int r2, float scale)
{
  for (int x = 0; x < nCols; x++) {
    data[r1 * nCols + x] += scale * data[r2 * nCols + x];
  }
}

static void
invert_gauss_jordan(const float * data, float * workspace, int nRows, int nCols, float * inverse)
{
  for (int y = 0; y < nRows; y++) {
    for (int x = 0; x < nCols; x++) {
      inverse  [y * nCols + x] = (x == y) ? 1.0f : 0.0f;
      workspace[y * nCols + x] = data[y * nCols + x];
    }
  }

  for (int c = 0; c < nCols; c++) {
    float scale = 1.0f;

    if(workspace[c * nCols + c] < 1e-36 && workspace[c * nCols + c] > -1e-36) {
      for (int r = c + 1; r < nCols; r++) {
	if(!(workspace[r * nCols + c] < 1e-36 && workspace[r * nCols + c] > -1e-36)) {
	  swap_row(workspace, nRows, nCols, r, c);
	  swap_row(inverse, nRows, nCols, r, c);
	  break;
	}
      }
     }

    scale = 1.0f / workspace[c * nCols + c];
    scale_row(workspace, nRows, nCols, c, scale);
    scale_row(inverse, nRows, nCols, c, scale);

    scale = 1.0f / workspace[c * nCols + c];
    scale_row(workspace, nRows, nCols, c, scale);
    scale_row(inverse, nRows, nCols, c, scale);

    for (int r = 0; r < nRows; r++) {
      if(r != c) {
        scale = - workspace[r * nCols + c];
        scale_and_add_row(workspace, nRows, nCols, r, c, scale);
        scale_and_add_row(inverse, nRows, nCols, r, c, scale);
      }
    }
  }
}


/* ========================================================================
 * Lucas-Kanade Registration
 * ======================================================================== */

void
kernel2_lucas_kanade (
  const float *img,
  const float *tmplt,
  float      *affine_warp,
  int num_iterations,
  int nRows, int nCols,
  float *warped_tmplt,
  float *error_img,
  float *img_dx,
  float *img_dy,
  float *nabla_Ix,
  float *nabla_Iy,
  float *I_steepest)
{
  int MN = nRows * nCols;

  /* Small fixed-size buffers on stack (120 floats = 480 bytes) */
  float sd_delta_p[6] = {0};
  float delta_p[6]    = {0};
  float H[36]         = {0};
  float H_wsp[36]     = {0};
  float H_inv[36]     = {0};

  float *IWxp = warped_tmplt;
  int N_p = 6;

  /* Zero working buffers explicitly (no derived workspace pointers). */
  memset(error_img, 0, MN * sizeof(float));
  memset(img_dx, 0, MN * sizeof(float));
  memset(img_dy, 0, MN * sizeof(float));
  memset(nabla_Ix, 0, MN * sizeof(float));
  memset(nabla_Iy, 0, MN * sizeof(float));
  memset(I_steepest, 0, (6 * MN) * sizeof(float));

  gradientXY(img, nCols, nRows, img_dx, img_dy);

  for (int iteration = 0; iteration < num_iterations; iteration++) {
    warp_image(img, nRows, nCols, affine_warp, IWxp);

    subtract(tmplt, IWxp, nRows, nCols, error_img);

    warp_image(img_dx, nRows, nCols, affine_warp, nabla_Ix);
    warp_image(img_dy, nRows, nCols, affine_warp, nabla_Iy);

    steepest_descent(nabla_Ix, nabla_Iy, nRows, nCols, I_steepest);

    hessian(I_steepest, nRows, nCols, N_p, H);

    invert_gauss_jordan(H, H_wsp, 6, 6, H_inv);

    sd_update(I_steepest, error_img, N_p, nRows, nCols, sd_delta_p);

    mult(H_inv, sd_delta_p, 6, 1, 6, delta_p);

    reshape(delta_p, 6, 1, 2, 3, sd_delta_p);

    add(affine_warp, sd_delta_p, 2, 3, affine_warp);
  }

  warp_image(img, nRows, nCols, affine_warp, IWxp);
}

static void
sd_update(const float * VI_dW_dp, const float * error_img, int N_p, int nRows, int nCols, float * sd_delta_p)
{
  for (int i = 0; i < N_p; i++) {
    sd_delta_p[i] = 0.0f;
  }

  for (int i = 0; i < N_p; i++) {
    for (int y = 0; y < nRows; y++) {
      for (int x = 0; x < nCols; x++) {
        sd_delta_p[i] += error_img[y * nCols + x] * VI_dW_dp[y * N_p * nCols + (x + i * nCols)];
      }
    }
  }
}

static void
warp_image (const float *Iin, int nCols, int nRows, const float *W_xp, float *Iout)
{
  const float w0 = W_xp[0];
  const float w1 = W_xp[1];
  const float w2 = W_xp[2];
  const float w3 = W_xp[3];
  const float w4 = W_xp[4];
  const float w5 = W_xp[5];
  float Tlocalx;
  float Tlocaly;
  float compa0;
  float compa1;
  const float compb0 = w2;
  const float compb1 = w5;

  for (int y = 0; y < nRows; y++) {
    compa0 = w1 * ((float) y) + compb0;
    compa1 = (1.0f + w4) * ((float) y) + compb1;

    for (int x = 0; x < nCols; x++) {
      Tlocalx = (1.0f + w0) * ((float) x) + compa0;
      Tlocaly = w3 * ((float) x) + compa1;

      float fTlx = floorf_approx(Tlocalx);
      float fTly = floorf_approx(Tlocaly);
      int xBas0 = (int) fTlx;
      int yBas0 = (int) fTly;
      int xBas1 = xBas0 + 1;
      int yBas1 = yBas0 + 1;
      const int index = y * nCols + x;

      if (xBas0 < -1 || yBas0 < -1 || xBas1 > nCols || yBas1 > nRows) {
        Iout[index] = 0;
      } else {
        float xCom = Tlocalx - fTlx;
        float yCom = Tlocaly - fTly;
        float xComi = 1 - xCom;
        float yComi = 1 - yCom;
        float c0 = (yBas0 < 0         || xBas0 < 0        ) ? 0 : Iin[yBas0 * nCols + xBas0];
        float c1 = (yBas1 > (nRows-1) || xBas0 < 0        ) ? 0 : Iin[yBas1 * nCols + xBas0];
        float c2 = (yBas0 < 0         || xBas1 > (nCols-1)) ? 0 : Iin[yBas0 * nCols + xBas1];
        float c3 = (yBas1 > (nRows-1) || xBas1 > (nCols-1)) ? 0 : Iin[yBas1 * nCols + xBas1];
        Iout[index] = c0 * (xComi * yComi) + c1 * (xComi * yCom)
                    + c2 * (xCom * yComi)  + c3 * (xCom * yCom);
      }
    }
  }
}

static void
steepest_descent (const float *gradX_warped, const float *gradY_warped, int nCols, int nRows, float *I_steepest)
{
  float Jacobian_x[6];
  float Jacobian_y[6];

  for (int y = 0; y < nRows; y++) {
    for (int x = 0; x < nCols; x++) {
      const int index = y * nCols + x;

      Jacobian_x[0] = (float) x;
      Jacobian_x[1] = 0.0;
      Jacobian_x[2] = (float) y;
      Jacobian_x[3] = 0.0;
      Jacobian_x[4] = 1.0;
      Jacobian_x[5] = 0.0;

      Jacobian_y[0] = 0.0;
      Jacobian_y[1] = (float) x;
      Jacobian_y[2] = 0.0;
      Jacobian_y[3] = (float) y;
      Jacobian_y[4] = 0.0;
      Jacobian_y[5] = 1.0;

      for (int k = 0; k < 6; k++) {
        const int j_index = (6 * y * nCols) + (nCols * k) + x;
        I_steepest[j_index] = (Jacobian_x[k] * gradX_warped[index]) + (Jacobian_y[k] * gradY_warped[index]);
      }
    }
  }
}

static void
hessian (const float *I_steepest, int nCols, int nRows, int np, float *H)
{
  for (int i = 0; i < np; i++) {
    for (int j = 0; j < np; j++) {
      H[i * np + j] = 0;
    }
  }

  for (int i = 0; i < np; i++) {
    for (int j = 0; j < np; j++) {
      float total = 0.0f;
      for (int y = 0; y < nRows; y++) {
        for (int x = 0; x < nCols; x++) {
          const int index1 = (np * y * nCols) + (nCols * i) + x;
          const int index2 = (np * y * nCols) + (nCols * j) + x;
          total += I_steepest[index1] * I_steepest[index2];
        }
      }
      H[np * j + i] = total;
    }
  }
}
