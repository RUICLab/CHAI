"""Added-operator checks must test behaviour, not vocabulary.

T21-T30 were validated by keyword presence while T1-T20 were validated
structurally. On ar-iclib that rejected three consecutive, semantically correct
zero-order-hold implementations of T23 because they named the carried value
``last_sample`` instead of containing the token ``hold``, while missing a
division by zero that shipped in every arm.
"""
from utils.operator_conformance import (
    nonconforming_added_operator_ids,
    validate_knob_safety,
    _has,
)

ZOH_NO_KEYWORD = """
static void acquire_window(accelWindow window) {
  /*Knob Variables Declaration Start*/
  int decimation_factor = 1;
  /*Knob Variables Declaration End*/
  accelReading sample;
  accelReading last_sample;
  unsigned samplesInWindow = 0;
  while (samplesInWindow < ACCEL_WINDOW_SIZE) {
    if (samplesInWindow % decimation_factor == 0) {
      ACCEL_singleSample(&sample);
      last_sample = sample;
    }
    window[samplesInWindow++] = last_sample;
  }
}
"""

PLAIN_PERFORATION = """
static void acquire_window(accelWindow window) {
  /*Knob Variables Declaration Start*/
  int iteration_pct = 100;
  /*Knob Variables Declaration End*/
  accelReading sample;
  unsigned limit = (ACCEL_WINDOW_SIZE * iteration_pct) / 100;
  for (unsigned i = 0; i < limit; i++) {
    ACCEL_singleSample(&sample);
    window[i] = sample;
  }
}
"""

SPATIAL_FILL_NO_MAGIC_NAMES = """
void image_kernel(unsigned char input[64][64], unsigned char output[64][64]) {
  /*Knob Variables Declaration Start*/
  int q = 1;
  /*Knob Variables Declaration End*/
  for (int y = 1; y < 63; y += q) {
    for (int x = 1; x < 63; x += q) {
      unsigned char v = expensive_pixel(input, y, x);
      for (int fy = 0; fy < q && y + fy < 63; ++fy) {
        for (int fx = 0; fx < q && x + fx < 63; ++fx) {
          output[y + fy][x + fx] = v;
        }
      }
    }
  }
}
"""

SPATIAL_PERFORATION_WITHOUT_FILL = """
void image_kernel(unsigned char input[64][64], unsigned char output[64][64]) {
  /*Knob Variables Declaration Start*/
  int q = 1;
  /*Knob Variables Declaration End*/
  for (int y = 1; y < 63; y += q)
    for (int x = 1; x < 63; x += q)
      output[y][x] = expensive_pixel(input, y, x);
}
"""


def test_t22_accepts_spatial_anchor_and_fill_structure_without_magic_names():
    assert 22 not in nonconforming_added_operator_ids(
        SPATIAL_FILL_NO_MAGIC_NAMES, [22]
    )


def test_t22_rejects_plain_2d_perforation_that_leaves_output_holes():
    assert 22 in nonconforming_added_operator_ids(
        SPATIAL_PERFORATION_WITHOUT_FILL, [22]
    )


def test_t23_accepts_zero_order_hold_that_avoids_the_keyword():
    assert 23 not in nonconforming_added_operator_ids(ZOH_NO_KEYWORD, [23])


def test_t23_still_rejects_decimation_without_reconstruction():
    assert 23 in nonconforming_added_operator_ids(PLAIN_PERFORATION, [23])


def test_keyword_checks_are_not_satisfied_by_a_comment():
    assert not _has("int x = 1; /* zero_order hold reconstruct */", r"\bhold\b")
    assert _has("int hold_value = 1;", r"\bhold\w*\b")


def test_divisor_knob_whose_range_includes_zero_is_rejected():
    error = validate_knob_safety(ZOH_NO_KEYWORD, [{"decimation_factor": [0, 8]}])
    assert "divisor" in error and "decimation_factor" in error


def test_divisor_knob_initialised_to_zero_is_rejected():
    code = ZOH_NO_KEYWORD.replace("int decimation_factor = 1;",
                                  "int decimation_factor = 0;")
    assert "divides by zero" in validate_knob_safety(
        code, [{"decimation_factor": [1, 8]}]
    )


def test_initialiser_outside_declared_range_is_rejected():
    error = validate_knob_safety(PLAIN_PERFORATION, [{"iteration_pct": [20, 80]}])
    assert "outside its declared range" in error


def test_safe_knob_declaration_passes():
    assert validate_knob_safety(PLAIN_PERFORATION, [{"iteration_pct": [50, 100]}]) == ""


def test_divisor_knob_with_safe_range_passes():
    assert validate_knob_safety(ZOH_NO_KEYWORD, [{"decimation_factor": [1, 8]}]) == ""


GENUINE_NIBBLE_LUT = """
int btbl_bitcnt(uint32_t x) {
  /*Knob Variables Declaration Start*/
  int stride = 1;
  /*Knob Variables Declaration End*/
  const uint8_t nib[16] = {0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4};
  int cnt = 0;
  for (int i = 0; i < 8; i += stride) {
    cnt += nib[(x >> (i * 4)) & 0x0F];
  }
  return cnt;
}
"""

BYTE_LUT_MISLABELLED_AS_NIBBLE = """
int btbl_bitcnt(uint32_t x) {
  /*Knob Variables Declaration Start*/
  int nibble_stride = 1;
  /*Knob Variables Declaration End*/
  int cnt = 0;
  for (int i = 0; i < 8; i += nibble_stride) {
    cnt += bits[((char *)&x)[i] & 0xFF];
  }
  return cnt;
}
"""


def test_t27_accepts_a_nibble_lut_whose_knob_is_not_named_nibble_stride():
    """The stride must be knob-controlled; its identifier is the model's choice."""
    assert 27 not in nonconforming_added_operator_ids(GENUINE_NIBBLE_LUT, [27])


KNOB_IS_ONLY_A_CORRECTION_TERM = """
int btbl_bitcnt(uint32_t x) {
  /*Knob Variables Declaration Start*/
  int correction = 0;
  /*Knob Variables Declaration End*/
  const uint8_t lut[16] = {0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4};
  int cnt = 0;
  for (int i = 0; i < 8; ++i)
    cnt += lut[(x >> (i * 4)) & 0xF];
  cnt += correction;
  return cnt;
}
"""

KNOB_SELECTS_THE_NIBBLE = """
int btbl_bitcnt(uint32_t x) {
  /*Knob Variables Declaration Start*/
  int nsel = 1;
  /*Knob Variables Declaration End*/
  const uint8_t nib[16] = {0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4};
  int cnt = 0;
  for (int i = 0; i < 8; ++i) {
    cnt += nib[(x >> (i * 4 * nsel)) & 0x0F];
  }
  return cnt;
}
"""


def test_t27_rejects_a_knob_that_does_not_control_nibble_traversal():
    """The loop walks all eight nibbles unconditionally and the knob is only
    added afterwards. Accepting this made the generalised checker useless."""
    assert 27 in nonconforming_added_operator_ids(
        KNOB_IS_ONLY_A_CORRECTION_TERM, [27]
    )


def test_t27_rejects_a_knob_that_only_selects_which_nibble_is_read():
    """T27 is defined as processing FEWER nibbles. A knob inside the shift
    expression changes which nibble is read while still running all eight
    iterations, so it adds error without removing work -- and for nsel >= 2 it
    shifts by 32 or more bits, which is undefined for a 32-bit value."""
    assert 27 in nonconforming_added_operator_ids(KNOB_SELECTS_THE_NIBBLE, [27])


def test_t27_rejects_a_knob_used_only_in_the_loop_initialiser():
    code = KNOB_SELECTS_THE_NIBBLE.replace(
        "int nsel = 1;", "int start = 0;"
    ).replace("(x >> (i * 4 * nsel))", "(x >> (i * 4))").replace(
        "for (int i = 0; i < 8; ++i)", "for (int i = start; i < 8; ++i)"
    )
    assert 27 in nonconforming_added_operator_ids(code, [27])


def test_t27_accepts_a_knob_in_the_loop_condition():
    code = KNOB_SELECTS_THE_NIBBLE.replace(
        "int nsel = 1;", "int nmax = 8;"
    ).replace("(x >> (i * 4 * nsel))", "(x >> (i * 4))").replace(
        "for (int i = 0; i < 8; ++i)", "for (int i = 0; i < nmax; ++i)"
    )
    assert 27 not in nonconforming_added_operator_ids(code, [27])


def test_t27_rejects_a_byte_lut_even_when_the_knob_is_named_nibble_stride():
    """bc-iclib produced exactly this: the existing 256-entry byte table with a
    stride, carrying the expected knob name. It is not nibble lookup."""
    assert 27 in nonconforming_added_operator_ids(
        BYTE_LUT_MISLABELLED_AS_NIBBLE, [27]
    )
