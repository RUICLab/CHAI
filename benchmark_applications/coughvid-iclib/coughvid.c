/* coughvid.c */
#ifdef LOCAL_RUN
#include <stdio.h>
#include <stdlib.h>
#endif

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "coughvid_weights.h"

/* TinyCoughDetect: three stride-2 3x3 convolutions (channel widths come from
 * the exported header; currently 1->6->12->16, no bias, BatchNorm folded into
 * each conv's weight/bias at export time -- an MCU never runs live BatchNorm
 * statistics), ReLU after each, global average pool over the remaining
 * spatial map, then a linear classifier down to one logit.
 *
 * Task: cough vs not-cough on COUGHVID, distilled from that dataset's own
 * published XGBoost cough detector. The label source is the published
 * classifier's output rather than human ground truth, so the defensible claim
 * is agreement with the published detector. Test AUROC 0.955 against the
 * 0.964 the published classifier reports for itself.
 *
 * Input is an already-computed, already-normalised log-mel spectrogram
 * (16 mels x 16 frames from a 1 s peak-energy crop). The STFT/mel front end
 * is deliberately not ported to C -- see the export script's docstring.
 *
 * This model was TRAINED to fit the device rather than shrunk afterwards.
 * The earlier COVID-task model used stride-1 convolutions with max-pooling,
 * so its first activation alone was 380160 bytes against the msp430fr5994
 * linker script's real 4096-byte RAM region. These convolutions are stride 2,
 * so the map shrinks immediately: the three activation buffers total 2560
 * bytes, and the weights plus baked-in input are 10948 bytes against ~13.7 KB
 * of usable FRAM.
 *
 * This is the ORIGINAL, UNAPPROXIMATED kernel: no knobs and no approximation
 * operators. It is what CHAI receives as input.
 */

/* All dimensions come from the exported header, so re-exporting a different
 * architecture or spectrogram size does not require editing this file. A
 * stride-2, padding-1 3x3 convolution maps N -> floor((N - 1) / 2) + 1. */
#define IN_MELS   COUGHVID_N_MELS
#define IN_FRAMES COUGHVID_FRAMES
#define IN_CH     1

#define HALVE(n)  (((n) - 1) / 2 + 1)

#define C1_OUT_CH COUGHVID_CHANNELS_C1
#define C1_H      HALVE(IN_MELS)
#define C1_W      HALVE(IN_FRAMES)

#define C2_OUT_CH COUGHVID_CHANNELS_C2
#define C2_H      HALVE(C1_H)
#define C2_W      HALVE(C1_W)

#define C3_OUT_CH COUGHVID_CHANNELS_C3
#define C3_H      HALVE(C2_H)
#define C3_W      HALVE(C2_W)

/* Output: the model's classification result for the single baked-in example. */
float coughvid_logit;
int coughvid_predicted_label;

static float conv1_out[C1_OUT_CH][C1_H][C1_W];
static float conv2_out[C2_OUT_CH][C2_H][C2_W];
static float conv3_out[C3_OUT_CH][C3_H][C3_W];

/* One stride-2, padding-1, 3x3 convolution + folded-BN bias + ReLU.
 * Zero-padded at the borders, matching PyTorch's Conv2d(padding=1) exactly.
 * Height and width are tracked separately because a spectrogram is not
 * square in general, even though it happens to be here.
 */
static void conv_block(
    const float *input, int in_ch, int in_h, int in_w,
    const float *weight, const float *bias,
    float *output, int out_ch, int out_h, int out_w)
{
  int oc, oy, ox, ic, ky, kx;
  for (oc = 0; oc < out_ch; oc++) {
    for (oy = 0; oy < out_h; oy++) {
      for (ox = 0; ox < out_w; ox++) {
        float sum = bias[oc];
        for (ic = 0; ic < in_ch; ic++) {
          for (ky = 0; ky < 3; ky++) {
            int iy = oy * 2 - 1 + ky;
            if (iy < 0 || iy >= in_h) continue;
            for (kx = 0; kx < 3; kx++) {
              int ix = ox * 2 - 1 + kx;
              if (ix < 0 || ix >= in_w) continue;
              float in_val = input[(ic * in_h + iy) * in_w + ix];
              float w = weight[((oc * in_ch + ic) * 3 + ky) * 3 + kx];
              sum += in_val * w;
            }
          }
        }
        /* ReLU */
        output[(oc * out_h + oy) * out_w + ox] = sum > 0.0f ? sum : 0.0f;
      }
    }
  }
}

/* The first layer reads its input through this pointer rather than naming the
 * baked-in array directly, so the same compiled kernel can be scored on many
 * inputs. That matters because the accuracy metric now counts how many
 * DECISIONS an approximation changes, and a single input cannot distinguish a
 * uniform bias (fatal, since the decision threshold is a fixed 0) from
 * harmless noise -- see utils/error_analyzer.py::_decision_disagreement.
 * On the device it points at the baked-in spectrogram exactly as before; a
 * real deployment would point it at a sensor buffer. */
const float *coughvid_input = &input_spectrogram[0][0][0];

void coughvid_classify( )
{
  conv_block(coughvid_input, IN_CH, IN_MELS, IN_FRAMES,
             &conv1_weight[0][0][0][0], conv1_bias,
             &conv1_out[0][0][0], C1_OUT_CH, C1_H, C1_W);

  conv_block(&conv1_out[0][0][0], C1_OUT_CH, C1_H, C1_W,
             &conv2_weight[0][0][0][0], conv2_bias,
             &conv2_out[0][0][0], C2_OUT_CH, C2_H, C2_W);

  conv_block(&conv2_out[0][0][0], C2_OUT_CH, C2_H, C2_W,
             &conv3_weight[0][0][0][0], conv3_bias,
             &conv3_out[0][0][0], C3_OUT_CH, C3_H, C3_W);

  /* Global average pool over the remaining map, per channel, then the
   * linear classifier. */
  int c, y, x;
  float logit = classifier_bias[0];
  for (c = 0; c < C3_OUT_CH; c++) {
    float sum = 0.0f;
    for (y = 0; y < C3_H; y++) {
      for (x = 0; x < C3_W; x++) {
        sum += conv3_out[c][y][x];
      }
    }
    float mean = sum / (float)(C3_H * C3_W);
    logit += mean * classifier_weight[c];
  }

  coughvid_logit = logit;
  /* sigmoid(logit) >= 0.5  <=>  logit >= 0 */
  coughvid_predicted_label = (logit >= 0.0f) ? 1 : 0;
}

int main(int argc, const char** argv)
{
#ifdef LOCAL_RUN
  /* stdout carries one bare logit per evaluated input, matching lqi-iclib's
   * ground_truth.txt/output.txt convention; the error metric reads them all.
   * Debug context goes to stderr so it never pollutes that file.
   *
   * When the exported header carries an evaluation set, every input in it is
   * scored. That is what lets the metric count changed decisions rather than
   * guess from one example. Without the set this falls back to the single
   * baked-in input, so a header exported before this change still works. */
#ifdef COUGHVID_EVAL_COUNT
  {
    int example;
    for (example = 0; example < COUGHVID_EVAL_COUNT; example++) {
      coughvid_input = &input_spectrogram_set[example][0][0][0];
      coughvid_classify( );
      printf("%.9g\n", coughvid_logit);
    }
    fprintf(stderr, "evaluated %d inputs\n", COUGHVID_EVAL_COUNT);
  }
#else
  coughvid_classify( );
  fprintf(stderr,
          "predicted_label=%d reference logit=%.6f true_label=%d abs_diff=%.9f\n",
          coughvid_predicted_label, COUGHVID_REFERENCE_LOGIT, COUGHVID_TRUE_LABEL,
          fabsf(coughvid_logit - COUGHVID_REFERENCE_LOGIT));
  printf("%.9g\n", coughvid_logit);
#endif
#endif

#ifndef LOCAL_RUN
  /* On the device only the single baked-in input is classified: the
   * evaluation set exists to score accuracy on the host, and baking hundreds
   * of spectrograms into FRAM would not fit. */
  coughvid_classify( );
#endif

#ifndef LOCAL_RUN
  indicate_end();
#endif
  return 0;
}
