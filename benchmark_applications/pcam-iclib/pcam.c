/* pcam.c */
#ifdef LOCAL_RUN
#include <stdio.h>
#include <stdlib.h>
#endif

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "pcam_weights.h"

/* TinyPcam: three stride-2 3x3 convolutions (channel widths come from the
 * exported header; currently 3->6->12->16, no bias, BatchNorm folded into
 * each conv's weight/bias at export time -- an MCU never runs live BatchNorm
 * statistics), ReLU after each, global average pool over the remaining
 * spatial map, then a linear classifier down to one logit.
 *
 * A model TRAINED to fit the device rather than shrunk afterwards: 16x16
 * input, channel widths 6-12-16 (2623 parameters). The activation buffers,
 * not the weights, forced 16x16 -- the msp430fr5994 linker script leaves a
 * 4 KB RAM region and the three buffers here total 2560 bytes.
 *
 * Measured cost of fitting the device, with both models trained at 16x16 for
 * 20 epochs so width is isolated from training resolution: resolution costs
 * about 0.054 AUROC (0.8459 at 64x64 -> 0.7922 at 16x16 at the same width)
 * and width a further 0.022 (0.7922 -> 0.7702). Resolution dominates.
 *
 * This is the ORIGINAL, UNAPPROXIMATED kernel: float throughout, no knobs and
 * no approximation operators. It is what CHAI receives as input.
 */

/* All dimensions come from the exported header, so re-exporting a different
 * architecture or resolution does not require editing this file. A stride-2,
 * padding-1 3x3 convolution maps N -> floor((N - 1) / 2) + 1. */
#define IN_SIZE   PCAM_INPUT_RESOLUTION
#define IN_CH     3

#define HALVE(n)  (((n) - 1) / 2 + 1)

#define C1_OUT_CH PCAM_CHANNELS_C1
#define C1_SIZE   HALVE(IN_SIZE)

#define C2_OUT_CH PCAM_CHANNELS_C2
#define C2_SIZE   HALVE(C1_SIZE)

#define C3_OUT_CH PCAM_CHANNELS_C3
#define C3_SIZE   HALVE(C2_SIZE)

/* Output: the model's classification result for the single baked-in image. */
float pcam_logit;
int pcam_predicted_label;

static float conv1_out[C1_OUT_CH][C1_SIZE][C1_SIZE];
static float conv2_out[C2_OUT_CH][C2_SIZE][C2_SIZE];
static float conv3_out[C3_OUT_CH][C3_SIZE][C3_SIZE];

/* The first layer reads its input through this pointer rather than naming the
 * baked-in array, so one compiled kernel can be scored on many inputs. The
 * accuracy metric counts how many DECISIONS an approximation changes, and a
 * single input cannot distinguish a uniform bias (fatal, since the decision
 * threshold is a fixed 0) from harmless noise. On the device it points at the
 * baked-in image exactly as before. */
const float *pcam_input = &input_image[0][0][0];

/* One stride-2, padding-1, 3x3 convolution + folded-BN bias + ReLU.
 * Zero-padded at the borders, matching PyTorch's Conv2d(padding=1) exactly.
 */
static void conv_block(
    const float *input, int in_ch, int in_size,
    const float *weight, const float *bias,
    float *output, int out_ch, int out_size)
{
  int oc, oy, ox, ic, ky, kx;
  for (oc = 0; oc < out_ch; oc++) {
    for (oy = 0; oy < out_size; oy++) {
      for (ox = 0; ox < out_size; ox++) {
        float sum = bias[oc];
        for (ic = 0; ic < in_ch; ic++) {
          for (ky = 0; ky < 3; ky++) {
            int iy = oy * 2 - 1 + ky;
            if (iy < 0 || iy >= in_size) continue;
            for (kx = 0; kx < 3; kx++) {
              int ix = ox * 2 - 1 + kx;
              if (ix < 0 || ix >= in_size) continue;
              float in_val = input[(ic * in_size + iy) * in_size + ix];
              float w = weight[((oc * in_ch + ic) * 3 + ky) * 3 + kx];
              sum += in_val * w;
            }
          }
        }
        /* ReLU */
        output[(oc * out_size + oy) * out_size + ox] = sum > 0.0f ? sum : 0.0f;
      }
    }
  }
}

void pcam_classify( )
{
  conv_block(pcam_input, IN_CH, IN_SIZE,
             &conv1_weight[0][0][0][0], conv1_bias,
             &conv1_out[0][0][0], C1_OUT_CH, C1_SIZE);

  conv_block(&conv1_out[0][0][0], C1_OUT_CH, C1_SIZE,
             &conv2_weight[0][0][0][0], conv2_bias,
             &conv2_out[0][0][0], C2_OUT_CH, C2_SIZE);

  conv_block(&conv2_out[0][0][0], C2_OUT_CH, C2_SIZE,
             &conv3_weight[0][0][0][0], conv3_bias,
             &conv3_out[0][0][0], C3_OUT_CH, C3_SIZE);

  /* Global average pool over the C3_SIZE x C3_SIZE spatial map, per channel,
   * then the linear classifier. */
  int c, y, x;
  float logit = classifier_bias[0];
  for (c = 0; c < C3_OUT_CH; c++) {
    float sum = 0.0f;
    for (y = 0; y < C3_SIZE; y++) {
      for (x = 0; x < C3_SIZE; x++) {
        sum += conv3_out[c][y][x];
      }
    }
    float mean = sum / (float)(C3_SIZE * C3_SIZE);
    logit += mean * classifier_weight[c];
  }

  pcam_logit = logit;
  /* sigmoid(logit) >= 0.5  <=>  logit >= 0 */
  pcam_predicted_label = (logit >= 0.0f) ? 1 : 0;
}

int main(int argc, const char** argv)
{
#ifdef LOCAL_RUN
  /* stdout carries one bare logit per evaluated input, matching lqi-iclib's
   * ground_truth.txt/output.txt convention; the error metric reads them all.
   * Debug context goes to stderr so it never pollutes that file.
   *
   * With a calibration set in the header every one of its inputs is scored,
   * which is what lets the metric count changed decisions instead of
   * inferring damage from one example. Without it this falls back to the
   * single baked-in input, so an older header still works -- but a one-line
   * reference cannot support an accuracy claim. */
#ifdef PCAM_EVAL_COUNT
  {
    int example;
    for (example = 0; example < PCAM_EVAL_COUNT; example++) {
      pcam_input = &input_image_set[example][0][0][0];
      pcam_classify( );
      printf("%.9g\n", pcam_logit);
    }
    fprintf(stderr, "evaluated %d inputs\n", PCAM_EVAL_COUNT);
  }
#else
  pcam_classify( );
  fprintf(stderr,
          "predicted_label=%d reference logit=%.6f true_label=%d abs_diff=%.9f\n",
          pcam_predicted_label, PCAM_REFERENCE_LOGIT, PCAM_TRUE_LABEL,
          fabsf(pcam_logit - PCAM_REFERENCE_LOGIT));
  printf("%.9g\n", pcam_logit);
#endif
#endif

#ifndef LOCAL_RUN
  /* On the device only the baked-in input is classified: the calibration set
   * exists to score accuracy on the host and would not fit in FRAM. */
  pcam_classify( );
  indicate_end();
#endif
  return 0;
}
